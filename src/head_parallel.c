#include "head_parallel.h"
#include "attention.h"
#include "profiler.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void head_parallel_mha(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
                       Tensor *out_full, int num_heads, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int seq_len = (rank == 0) ? Q_full->rows : 0;
    int d_model = (rank == 0) ? Q_full->cols : 0;

    /* Broadcast dimensions */
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d_model, 1, MPI_INT, 0, comm);

    int d_k = d_model / num_heads;

    /* Each rank owns heads_per_rank consecutive heads */
    int heads_per_rank = num_heads / size;
    int my_heads       = heads_per_rank;
    /* last rank takes the remainder */
    if (rank == size - 1) my_heads += num_heads % size;

    int my_slice_cols = my_heads * d_k;  /* columns this rank owns */

    /* Scatter: root sends each rank its head columns for Q, K, V */
    profiler_start(TIMER_COMM);

    /* Build scatter counts/displacements (rows * cols_per_rank) */
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs     = malloc(size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int rh = (r == size - 1) ? heads_per_rank + num_heads % size : heads_per_rank;
            /* We scatter one row at a time via MPI_Scatterv in a loop */
            sendcounts[r] = rh * d_k;
            displs[r]     = offset;
            offset       += rh * d_k;
        }
    }

    /* Allocate local head slices */
    Tensor Ql = tensor_alloc(seq_len, my_slice_cols);
    Tensor Kl = tensor_alloc(seq_len, my_slice_cols);
    Tensor Vl = tensor_alloc(seq_len, my_slice_cols);

    /* Scatter row by row for Q, K, V */
    for (int i = 0; i < seq_len; i++) {
        float *qrow = (rank == 0) ? &AT(*Q_full, i, 0) : NULL;
        float *krow = (rank == 0) ? &AT(*K_full, i, 0) : NULL;
        float *vrow = (rank == 0) ? &AT(*V_full, i, 0) : NULL;

        MPI_Scatterv(qrow, sendcounts, displs, MPI_FLOAT,
                     &AT(Ql, i, 0), my_slice_cols, MPI_FLOAT, 0, comm);
        MPI_Scatterv(krow, sendcounts, displs, MPI_FLOAT,
                     &AT(Kl, i, 0), my_slice_cols, MPI_FLOAT, 0, comm);
        MPI_Scatterv(vrow, sendcounts, displs, MPI_FLOAT,
                     &AT(Vl, i, 0), my_slice_cols, MPI_FLOAT, 0, comm);
    }
    profiler_stop(TIMER_COMM);

    /* Each rank computes attention on each of its heads */
    profiler_start(TIMER_COMPUTE);
    Tensor out_local = tensor_alloc(seq_len, my_slice_cols);

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

    /* Gather results back to rank 0 */
    profiler_start(TIMER_COMM);
    for (int i = 0; i < seq_len; i++) {
        float *orow = (rank == 0) ? &AT(*out_full, i, 0) : NULL;
        MPI_Gatherv(&AT(out_local, i, 0), my_slice_cols, MPI_FLOAT,
                    orow, sendcounts, displs, MPI_FLOAT, 0, comm);
    }
    profiler_stop(TIMER_COMM);

    tensor_free(&Ql); tensor_free(&Kl);
    tensor_free(&Vl); tensor_free(&out_local);
    if (rank == 0) { free(sendcounts); free(displs); }
}
