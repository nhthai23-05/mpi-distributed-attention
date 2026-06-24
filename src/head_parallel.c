#include "head_parallel.h"
#include "attention.h"
#include "profiler.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * Pack rank r's head columns from src [seq_len x d_model] into dst at byte
 * offset buf_off.  Columns [col_off, col_off+rc) for every row are copied
 * contiguously, producing a [seq_len x rc] row-major block ready for scatter.
 */
static void pack_columns(float *dst, int buf_off,
                         const Tensor *src, int col_off, int rc) {
    int seq_len = src->rows;
    for (int i = 0; i < seq_len; i++)
        memcpy(dst + buf_off + i * rc,
               &AT(*src, i, col_off),
               rc * sizeof(float));
}

/*
 * Inverse: unpack a [seq_len x rc] block from src at buf_off back into
 * columns [col_off, col_off+rc) of dst [seq_len x d_model].
 */
static void unpack_columns(Tensor *dst, int col_off,
                           const float *src, int buf_off, int rc) {
    int seq_len = dst->rows;
    for (int i = 0; i < seq_len; i++)
        memcpy(&AT(*dst, i, col_off),
               src + buf_off + i * rc,
               rc * sizeof(float));
}

void head_parallel_mha(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
                       Tensor *out_full, int num_heads, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int seq_len = (rank == 0) ? Q_full->rows : 0;
    int d_model = (rank == 0) ? Q_full->cols : 0;

    MPI_Bcast(&seq_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d_model, 1, MPI_INT, 0, comm);

    int d_k = d_model / num_heads;
    int heads_per_rank = num_heads / size;

    /* Last rank absorbs the remainder heads (if num_heads % size != 0). */
    int my_heads = heads_per_rank + (rank == size - 1 ? num_heads % size : 0);
    int my_cols  = my_heads * d_k;

    /*
     * Build per-rank sendcounts and displacements for the FULL slice
     * (seq_len rows × rh*d_k cols).  Only root fills these; other ranks
     * pass NULL to MPI_Scatterv/Gatherv where the spec says the args are
     * insignificant.
     */
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs     = malloc(size * sizeof(int));
        int off = 0;
        for (int r = 0; r < size; r++) {
            int rh = heads_per_rank + (r == size - 1 ? num_heads % size : 0);
            sendcounts[r] = seq_len * rh * d_k;
            displs[r]     = off;
            off          += sendcounts[r];
        }
    }

    /* Allocate local head slices. */
    Tensor Ql = tensor_alloc(seq_len, my_cols);
    Tensor Kl = tensor_alloc(seq_len, my_cols);
    Tensor Vl = tensor_alloc(seq_len, my_cols);

    /*
     * SCATTER: pack Q/K/V column blocks for each rank into a contiguous
     * buffer on root (single MPI_Scatterv per matrix instead of seq_len calls).
     */
    profiler_wait_barrier(comm);   /* separate sync wait from transfer */
    profiler_start(TIMER_COMM);
    {
        float *Q_packed = NULL, *K_packed = NULL, *V_packed = NULL;
        if (rank == 0) {
            int total = displs[size - 1] + sendcounts[size - 1];
            Q_packed = malloc((size_t)total * sizeof(float));
            K_packed = malloc((size_t)total * sizeof(float));
            V_packed = malloc((size_t)total * sizeof(float));

            int col_off = 0, buf_off = 0;
            for (int r = 0; r < size; r++) {
                int rh = heads_per_rank + (r == size - 1 ? num_heads % size : 0);
                int rc = rh * d_k;
                pack_columns(Q_packed, buf_off, Q_full, col_off, rc);
                pack_columns(K_packed, buf_off, K_full, col_off, rc);
                pack_columns(V_packed, buf_off, V_full, col_off, rc);
                col_off += rc;
                buf_off += seq_len * rc;
            }
        }

        MPI_Scatterv(Q_packed, sendcounts, displs, MPI_FLOAT,
                     Ql.data, seq_len * my_cols, MPI_FLOAT, 0, comm);
        MPI_Scatterv(K_packed, sendcounts, displs, MPI_FLOAT,
                     Kl.data, seq_len * my_cols, MPI_FLOAT, 0, comm);
        MPI_Scatterv(V_packed, sendcounts, displs, MPI_FLOAT,
                     Vl.data, seq_len * my_cols, MPI_FLOAT, 0, comm);

        /* Q, K, V scattered: three bulk transfers of this rank's slice. */
        for (int t = 0; t < 3; t++)
            profiler_count_msg((long)seq_len * my_cols * sizeof(float));

        if (rank == 0) { free(Q_packed); free(K_packed); free(V_packed); }
    }
    profiler_stop(TIMER_COMM);

    /* Each rank computes attention on each of its heads. */
    profiler_start(TIMER_COMPUTE);
    Tensor out_local = tensor_alloc(seq_len, my_cols);

    for (int h = 0; h < my_heads; h++) {
        Tensor Qh = tensor_alloc(seq_len, d_k);
        Tensor Kh = tensor_alloc(seq_len, d_k);
        Tensor Vh = tensor_alloc(seq_len, d_k);
        Tensor Oh = tensor_alloc(seq_len, d_k);

        for (int i = 0; i < seq_len; i++) {
            memcpy(&AT(Qh, i, 0), &AT(Ql, i, h * d_k), d_k * sizeof(float));
            memcpy(&AT(Kh, i, 0), &AT(Kl, i, h * d_k), d_k * sizeof(float));
            memcpy(&AT(Vh, i, 0), &AT(Vl, i, h * d_k), d_k * sizeof(float));
        }

        attention_seq(&Qh, &Kh, &Vh, &Oh);

        for (int i = 0; i < seq_len; i++)
            memcpy(&AT(out_local, i, h * d_k), &AT(Oh, i, 0), d_k * sizeof(float));

        tensor_free(&Qh); tensor_free(&Kh);
        tensor_free(&Vh); tensor_free(&Oh);
    }
    profiler_stop(TIMER_COMPUTE);

    /*
     * GATHER: collect out_local from every rank into a packed buffer on root,
     * then unpack each rank's block back into the right columns of out_full.
     */
    profiler_wait_barrier(comm);   /* separate sync wait from transfer */
    profiler_start(TIMER_COMM);
    {
        float *out_packed = (rank == 0)
            ? malloc((size_t)seq_len * d_model * sizeof(float))
            : NULL;

        MPI_Gatherv(out_local.data, seq_len * my_cols, MPI_FLOAT,
                    out_packed, sendcounts, displs, MPI_FLOAT, 0, comm);
        profiler_count_msg((long)seq_len * my_cols * sizeof(float));

        if (rank == 0) {
            int col_off = 0, buf_off = 0;
            for (int r = 0; r < size; r++) {
                int rh = heads_per_rank + (r == size - 1 ? num_heads % size : 0);
                int rc = rh * d_k;
                unpack_columns(out_full, col_off, out_packed, buf_off, rc);
                col_off += rc;
                buf_off += seq_len * rc;
            }
            free(out_packed);
        }
    }
    profiler_stop(TIMER_COMM);

    tensor_free(&Ql); tensor_free(&Kl);
    tensor_free(&Vl); tensor_free(&out_local);
    if (rank == 0) { free(sendcounts); free(displs); }
}
