#include "tensor_parallel.h"
#include "tensor.h"
#include "attention.h"
#include "profiler.h"
#include "progress.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ================================================================== */
/*  DEFAULT PATH — 1D row-block decomposition + local softmax          */
/*                                                                     */
/*  Each process owns a contiguous block of QUERY rows. Full K and V   */
/*  are broadcast to every process, so each process computes the       */
/*  attention for its rows entirely locally (softmax is per-row, so it */
/*  is embarrassingly parallel over query rows — no Allreduce needed). */
/*  Works for ANY process count (including the size-2 sub-groups used  */
/*  by hybrid mode).                                                   */
/* ================================================================== */
static void tp_allgather(const Tensor *Q_full, const Tensor *K_full,
                         const Tensor *V_full, Tensor *out_full,
                         MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int seq_len = (rank == 0) ? Q_full->rows : 0;
    int d_k     = (rank == 0) ? Q_full->cols : 0;
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d_k,     1, MPI_INT, 0, comm);

    /* Row partition: first `rem` ranks get one extra row */
    int base = seq_len / size;
    int rem  = seq_len % size;
    int my_rows = base + (rank < rem ? 1 : 0);

    int *cnt = malloc(size * sizeof(int));
    int *dsp = malloc(size * sizeof(int));
    int off = 0;
    for (int r = 0; r < size; r++) {
        int rr = base + (r < rem ? 1 : 0);
        cnt[r] = rr * d_k;
        dsp[r] = off;
        off   += rr * d_k;
    }

    float *Qb = malloc((size_t)my_rows * d_k * sizeof(float));
    float *Ob = malloc((size_t)my_rows * d_k * sizeof(float));
    float *Kf = malloc((size_t)seq_len * d_k * sizeof(float));
    float *Vf = malloc((size_t)seq_len * d_k * sizeof(float));

    /* Scatter query rows (non-overlapping -> well defined) */
    profiler_wait_barrier(comm);   /* separate sync wait from transfer */
    profiler_start(TIMER_COMM);
    MPI_Scatterv(rank == 0 ? Q_full->data : NULL, cnt, dsp, MPI_FLOAT,
                 Qb, my_rows * d_k, MPI_FLOAT, 0, comm);
    profiler_count_msg((long)my_rows * d_k * sizeof(float));

    /* Broadcast full K and V (each process needs all keys/values) */
    if (rank == 0) {
        memcpy(Kf, K_full->data, (size_t)seq_len * d_k * sizeof(float));
        memcpy(Vf, V_full->data, (size_t)seq_len * d_k * sizeof(float));
    }
    MPI_Bcast(Kf, seq_len * d_k, MPI_FLOAT, 0, comm);
    MPI_Bcast(Vf, seq_len * d_k, MPI_FLOAT, 0, comm);
    profiler_count_msg((long)seq_len * d_k * sizeof(float));   /* K */
    profiler_count_msg((long)seq_len * d_k * sizeof(float));   /* V */
    profiler_stop(TIMER_COMM);

    /* Local attention for each owned query row. Each row is fully local (the
     * softmax is per-row and this rank holds full K,V), so we reuse the shared
     * streaming kernel — identical math to the sequential baseline. */
    profiler_start(TIMER_COMPUTE);
    progress_begin("tensor-attn rows", my_rows);
    #pragma omp parallel
    {
        float *srow = malloc((size_t)seq_len * sizeof(float));
        #pragma omp for schedule(static)
        for (int i = 0; i < my_rows; i++)
            attention_row(&Qb[(size_t)i * d_k], Kf, Vf, seq_len, d_k,
                          &Ob[(size_t)i * d_k], srow);
        free(srow);
    }
    progress_end();
    profiler_stop(TIMER_COMPUTE);

    /* Gather output rows back to root */
    profiler_wait_barrier(comm);   /* separate sync wait from transfer */
    profiler_start(TIMER_COMM);
    MPI_Gatherv(Ob, my_rows * d_k, MPI_FLOAT,
                rank == 0 ? out_full->data : NULL, cnt, dsp, MPI_FLOAT, 0, comm);
    profiler_count_msg((long)my_rows * d_k * sizeof(float));
    profiler_stop(TIMER_COMM);

    free(Qb); free(Ob); free(Kf); free(Vf);
    free(cnt); free(dsp);
}

/* ================================================================== */
/*  CANNON PATH — 2D block decomposition + distributed softmax         */
/*                                                                     */
/*  Requires P = sqrt_p^2, seq_len % sqrt_p == 0, d_k % sqrt_p == 0.    */
/*  Uses MPI_Sendrecv_replace (ring/torus shifts) for the two matmuls   */
/*  and MPI_Allreduce over a ROW sub-communicator for the distributed   */
/*  softmax (a single row of S is split across the block-row).          */
/* ================================================================== */

/* Cannon's algorithm: C = A * B on a sqrt_p x sqrt_p torus.
 * A: [br x bk]  B: [bk x bc]  C: [br x bc]  (uniform blocks on every rank).
 * Canonical alignment: shift A left by row, B up by col, then unit shifts. */
static void cannon_matmul(float *A, float *B, float *C,
                          int br, int bk, int bc, int sqrt_p, MPI_Comm cart) {
    int crank;
    MPI_Comm_rank(cart, &crank);
    int coords[2];
    MPI_Cart_coords(cart, crank, 2, coords);

    memset(C, 0, (size_t)br * bc * sizeof(float));

    int src, dst;
    /* Initial alignment: A left by coords[0], B up by coords[1] */
    MPI_Cart_shift(cart, 1, -coords[0], &src, &dst);
    MPI_Sendrecv_replace(A, br * bk, MPI_FLOAT, dst, 1, src, 1, cart, MPI_STATUS_IGNORE);
    profiler_count_msg((long)br * bk * sizeof(float));
    MPI_Cart_shift(cart, 0, -coords[1], &src, &dst);
    MPI_Sendrecv_replace(B, bk * bc, MPI_FLOAT, dst, 2, src, 2, cart, MPI_STATUS_IGNORE);
    profiler_count_msg((long)bk * bc * sizeof(float));

    for (int step = 0; step < sqrt_p; step++) {
        for (int i = 0; i < br; i++)
            for (int k = 0; k < bk; k++) {
                float a = A[i * bk + k];
                for (int j = 0; j < bc; j++)
                    C[i * bc + j] += a * B[k * bc + j];
            }
        /* shift A left by 1, B up by 1 */
        MPI_Cart_shift(cart, 1, -1, &src, &dst);
        MPI_Sendrecv_replace(A, br * bk, MPI_FLOAT, dst, 3, src, 3, cart, MPI_STATUS_IGNORE);
        profiler_count_msg((long)br * bk * sizeof(float));
        MPI_Cart_shift(cart, 0, -1, &src, &dst);
        MPI_Sendrecv_replace(B, bk * bc, MPI_FLOAT, dst, 4, src, 4, cart, MPI_STATUS_IGNORE);
        profiler_count_msg((long)bk * bc * sizeof(float));
    }
}

/* Distributed softmax over S blocks. Each process holds an [bn x bn] block
 * = global rows [row*bn, ...) x global cols [col*bn, ...). A full row is split
 * across the block-row, so reduce max/sum over row_comm. */
static void distributed_softmax_blocks(float *S, int bn, MPI_Comm row_comm) {
    for (int a = 0; a < bn; a++) {
        float lmax = -INFINITY;
        for (int b = 0; b < bn; b++)
            if (S[a * bn + b] > lmax) lmax = S[a * bn + b];

        float gmax;
        MPI_Allreduce(&lmax, &gmax, 1, MPI_FLOAT, MPI_MAX, row_comm);
        profiler_count_msg((long)sizeof(float));

        float lsum = 0.0f;
        for (int b = 0; b < bn; b++) {
            S[a * bn + b] = expf(S[a * bn + b] - gmax);
            lsum += S[a * bn + b];
        }

        float gsum;
        MPI_Allreduce(&lsum, &gsum, 1, MPI_FLOAT, MPI_SUM, row_comm);
        profiler_count_msg((long)sizeof(float));

        for (int b = 0; b < bn; b++) S[a * bn + b] /= gsum;
    }
}

/* Scatter a [R x C] row-major matrix (valid on cart rank 0) into [R/sqrt_p x
 * C/sqrt_p] blocks; block (i,j) goes to the rank at cart coords (i,j). */
static void scatter_blocks_2d(const float *full, float *block_buf,
                              int R, int C, int sqrt_p, MPI_Comm cart) {
    int br = R / sqrt_p, bc = C / sqrt_p;
    int myrank;
    MPI_Comm_rank(cart, &myrank);

    if (myrank == 0) {
        for (int pi = 0; pi < sqrt_p; pi++)
            for (int pj = 0; pj < sqrt_p; pj++) {
                float *tmp = malloc((size_t)br * bc * sizeof(float));
                for (int a = 0; a < br; a++)
                    for (int b = 0; b < bc; b++)
                        tmp[a * bc + b] = full[(pi * br + a) * C + (pj * bc + b)];
                int dest, rc[2] = {pi, pj};
                MPI_Cart_rank(cart, rc, &dest);
                if (dest == 0) {
                    memcpy(block_buf, tmp, (size_t)br * bc * sizeof(float));
                } else {
                    MPI_Send(tmp, br * bc, MPI_FLOAT, dest, 77, cart);
                    profiler_count_msg((long)br * bc * sizeof(float));
                }
                free(tmp);
            }
    } else {
        MPI_Recv(block_buf, br * bc, MPI_FLOAT, 0, 77, cart, MPI_STATUS_IGNORE);
        profiler_count_msg((long)br * bc * sizeof(float));
    }
}

/* Inverse of scatter_blocks_2d: gather blocks back into a [R x C] matrix on
 * cart rank 0. */
static void gather_blocks_2d(float *full, const float *block_buf,
                             int R, int C, int sqrt_p, MPI_Comm cart) {
    int br = R / sqrt_p, bc = C / sqrt_p;
    int myrank;
    MPI_Comm_rank(cart, &myrank);

    if (myrank == 0) {
        for (int pi = 0; pi < sqrt_p; pi++)
            for (int pj = 0; pj < sqrt_p; pj++) {
                float *tmp = malloc((size_t)br * bc * sizeof(float));
                int src, rc[2] = {pi, pj};
                MPI_Cart_rank(cart, rc, &src);
                if (src == 0) {
                    memcpy(tmp, block_buf, (size_t)br * bc * sizeof(float));
                } else {
                    MPI_Recv(tmp, br * bc, MPI_FLOAT, src, 78, cart, MPI_STATUS_IGNORE);
                    profiler_count_msg((long)br * bc * sizeof(float));
                }
                for (int a = 0; a < br; a++)
                    for (int b = 0; b < bc; b++)
                        full[(pi * br + a) * C + (pj * bc + b)] = tmp[a * bc + b];
                free(tmp);
            }
    } else {
        MPI_Send(block_buf, br * bc, MPI_FLOAT, 0, 78, cart);
        profiler_count_msg((long)br * bc * sizeof(float));
    }
}

static void tp_cannon(const Tensor *Q_full, const Tensor *K_full,
                      const Tensor *V_full, Tensor *out_full, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int sqrt_p = (int)round(sqrt((double)size));
    if (sqrt_p * sqrt_p != size) {
        if (rank == 0)
            fprintf(stderr, "cannon: process count %d is not a perfect square\n", size);
        MPI_Abort(comm, 1);
    }

    int seq_len = (rank == 0) ? Q_full->rows : 0;
    int d_k     = (rank == 0) ? Q_full->cols : 0;
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d_k,     1, MPI_INT, 0, comm);

    if (seq_len % sqrt_p != 0 || d_k % sqrt_p != 0) {
        if (rank == 0)
            fprintf(stderr, "cannon: seq_len(%d) and d_k(%d) must be divisible by sqrt_p(%d)\n",
                    seq_len, d_k, sqrt_p);
        MPI_Abort(comm, 1);
    }

    int bn = seq_len / sqrt_p;
    int bk = d_k / sqrt_p;

    int dims[2] = {sqrt_p, sqrt_p}, periods[2] = {1, 1};
    MPI_Comm cart;
    MPI_Cart_create(comm, 2, dims, periods, 0, &cart);
    int crank;
    MPI_Comm_rank(cart, &crank);
    int coords[2];
    MPI_Cart_coords(cart, crank, 2, coords);

    /* Row sub-communicator (processes sharing a block-row) for softmax */
    MPI_Comm row_comm;
    MPI_Comm_split(cart, coords[0], coords[1], &row_comm);

    float *Aq  = malloc((size_t)bn * bk * sizeof(float));  /* Q block   */
    float *Bkt = malloc((size_t)bk * bn * sizeof(float));  /* K^T block */
    float *S   = malloc((size_t)bn * bn * sizeof(float));  /* scores    */
    float *Vb  = malloc((size_t)bn * bk * sizeof(float));  /* V block   */
    float *Ob  = malloc((size_t)bn * bk * sizeof(float));  /* output    */

    /* Distribute Q, K^T, V into 2D blocks */
    profiler_start(TIMER_COMM);
    float *Kt_full = NULL;
    if (rank == 0) {
        Tensor kt = transpose(K_full);            /* [d_k x seq_len] */
        Kt_full = malloc((size_t)d_k * seq_len * sizeof(float));
        memcpy(Kt_full, kt.data, (size_t)d_k * seq_len * sizeof(float));
        tensor_free(&kt);
    }
    scatter_blocks_2d(rank == 0 ? Q_full->data : NULL, Aq,  seq_len, d_k,     sqrt_p, cart);
    scatter_blocks_2d(Kt_full,                          Bkt, d_k,     seq_len, sqrt_p, cart);
    scatter_blocks_2d(rank == 0 ? V_full->data : NULL, Vb,  seq_len, d_k,     sqrt_p, cart);
    if (rank == 0) free(Kt_full);
    profiler_stop(TIMER_COMM);

    /* Pass 1: S = Q * K^T (Cannon), then scale */
    profiler_start(TIMER_COMPUTE);
    cannon_matmul(Aq, Bkt, S, bn, bk, bn, sqrt_p, cart);
    float scale = 1.0f / sqrtf((float)d_k);
    for (int t = 0; t < bn * bn; t++) S[t] *= scale;
    profiler_stop(TIMER_COMPUTE);

    /* Distributed softmax over the block-row */
    profiler_start(TIMER_COMM);
    distributed_softmax_blocks(S, bn, row_comm);
    profiler_stop(TIMER_COMM);

    /* Pass 2: O = softmax(S) * V (Cannon) */
    profiler_start(TIMER_COMPUTE);
    cannon_matmul(S, Vb, Ob, bn, bn, bk, sqrt_p, cart);
    profiler_stop(TIMER_COMPUTE);

    /* Gather output blocks back to root */
    profiler_start(TIMER_COMM);
    gather_blocks_2d(rank == 0 ? out_full->data : NULL, Ob, seq_len, d_k, sqrt_p, cart);
    profiler_stop(TIMER_COMM);

    free(Aq); free(Bkt); free(S); free(Vb); free(Ob);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart);
}

/* ================================================================== */
/*  RING PATH — sharded K,V streamed around a ring + online softmax    */
/*                                                                     */
/*  Each rank owns a contiguous block of N/P QUERY rows AND a block of */
/*  N/P key/value rows (K,V are NOT replicated). The K,V blocks are    */
/*  passed around a ring; at each of the P steps a rank folds the block */
/*  it currently holds into a running (Flash-style) online softmax over */
/*  its local queries, so it never materializes the full score row and  */
/*  never holds more than O(N/P) of K,V. The shift is double-buffered    */
/*  with MPI_Isend/Irecv so the transfer overlaps the compute.          */
/*                                                                     */
/*  Result is exact up to floating-point reassociation (keys are summed */
/*  in a different order than the sequential baseline, ~1e-6, like      */
/*  Cannon). Works for ANY process count.                               */
/* ================================================================== */
static void tp_ring(const Tensor *Q_full, const Tensor *K_full,
                    const Tensor *V_full, Tensor *out_full, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int seq_len = (rank == 0) ? Q_full->rows : 0;
    int d_k     = (rank == 0) ? Q_full->cols : 0;
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d_k,     1, MPI_INT, 0, comm);

    int base = seq_len / size;
    int rem  = seq_len % size;
    int my_rows  = base + (rank < rem ? 1 : 0);
    int max_rows = base + (rem > 0 ? 1 : 0);   /* largest block (ring buffer size) */

    /* Same row partition for Q, K and V (rank r owns rows [.., ..)). */
    int *cnt = malloc(size * sizeof(int));
    int *dsp = malloc(size * sizeof(int));
    int *rows_of = malloc(size * sizeof(int));   /* rows in each rank's block */
    int off = 0;
    for (int r = 0; r < size; r++) {
        int rr = base + (r < rem ? 1 : 0);
        rows_of[r] = rr;
        cnt[r] = rr * d_k;
        dsp[r] = off;
        off   += rr * d_k;
    }

    float *Qb = malloc((size_t)max_rows * d_k * sizeof(float));   /* local queries  */
    float *Kc = calloc((size_t)max_rows * d_k, sizeof(float));    /* K block (cur)  */
    float *Vc = calloc((size_t)max_rows * d_k, sizeof(float));    /* V block (cur)  */
    float *Kn = malloc((size_t)max_rows * d_k * sizeof(float));   /* K block (next) */
    float *Vn = malloc((size_t)max_rows * d_k * sizeof(float));   /* V block (next) */
    float *Ob = malloc((size_t)max_rows * d_k * sizeof(float));   /* output rows    */

    /* online-softmax state per local query row */
    float *m   = malloc((size_t)max_rows * sizeof(float));        /* running max */
    float *l   = malloc((size_t)max_rows * sizeof(float));        /* running sum */
    float *acc = calloc((size_t)max_rows * d_k, sizeof(float));   /* running sum p*V */
    for (int i = 0; i < my_rows; i++) { m[i] = -INFINITY; l[i] = 0.0f; }

    /* Scatter each rank its own block of Q, K and V (no broadcast of full K,V). */
    profiler_wait_barrier(comm);
    profiler_start(TIMER_COMM);
    MPI_Scatterv(rank == 0 ? Q_full->data : NULL, cnt, dsp, MPI_FLOAT,
                 Qb, my_rows * d_k, MPI_FLOAT, 0, comm);
    MPI_Scatterv(rank == 0 ? K_full->data : NULL, cnt, dsp, MPI_FLOAT,
                 Kc, my_rows * d_k, MPI_FLOAT, 0, comm);
    MPI_Scatterv(rank == 0 ? V_full->data : NULL, cnt, dsp, MPI_FLOAT,
                 Vc, my_rows * d_k, MPI_FLOAT, 0, comm);
    profiler_count_msg((long)my_rows * d_k * sizeof(float) * 3);
    profiler_stop(TIMER_COMM);

    float scale = 1.0f / sqrtf((float)d_k);
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    progress_begin("ring-attn steps", size);
    for (int step = 0; step < size; step++) {
        /* The block we currently hold originated on rank `src`. */
        int src = (rank - step + size) % size;
        int blk_rows = rows_of[src];

        /* Prefetch the next block while we compute on the current one (overlap). */
        MPI_Request reqs[4];
        int nreq = 0;
        if (step < size - 1) {
            profiler_start(TIMER_COMM);
            MPI_Irecv(Kn, max_rows * d_k, MPI_FLOAT, prev, 50, comm, &reqs[nreq++]);
            MPI_Irecv(Vn, max_rows * d_k, MPI_FLOAT, prev, 51, comm, &reqs[nreq++]);
            MPI_Isend(Kc, max_rows * d_k, MPI_FLOAT, next, 50, comm, &reqs[nreq++]);
            MPI_Isend(Vc, max_rows * d_k, MPI_FLOAT, next, 51, comm, &reqs[nreq++]);
            profiler_count_msg((long)max_rows * d_k * sizeof(float) * 2);  /* K,V */
            profiler_stop(TIMER_COMM);
        }

        /* Fold this block into the online softmax for every local query row.
         * Each i is independent (private m/l/acc row), so it threads cleanly. */
        profiler_start(TIMER_COMPUTE);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < my_rows; i++) {
            const float *qi = &Qb[(size_t)i * d_k];
            float *ai = &acc[(size_t)i * d_k];
            float mi = m[i], li = l[i];
            for (int j = 0; j < blk_rows; j++) {
                const float *kj = &Kc[(size_t)j * d_k];
                float dot = 0.0f;
                for (int k = 0; k < d_k; k++) dot += qi[k] * kj[k];
                float s = dot * scale;
                if (s > mi) {                 /* new running max -> rescale history */
                    float corr = expf(mi - s);   /* mi=-inf at start => corr=0 */
                    for (int k = 0; k < d_k; k++) ai[k] *= corr;
                    li *= corr;
                    mi = s;
                }
                float p = expf(s - mi);
                li += p;
                const float *vj = &Vc[(size_t)j * d_k];
                for (int k = 0; k < d_k; k++) ai[k] += p * vj[k];
            }
            m[i] = mi; l[i] = li;
        }
        profiler_stop(TIMER_COMPUTE);

        /* Finish the shift; the buffer we received becomes current. */
        if (step < size - 1) {
            profiler_start(TIMER_COMM);
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
            profiler_stop(TIMER_COMM);
            float *t;
            t = Kc; Kc = Kn; Kn = t;
            t = Vc; Vc = Vn; Vn = t;
        }
        progress_update(step + 1);
    }
    progress_end();

    /* Normalize: o_i = acc_i / l_i. */
    profiler_start(TIMER_COMPUTE);
    for (int i = 0; i < my_rows; i++) {
        float inv = (l[i] != 0.0f) ? 1.0f / l[i] : 0.0f;
        for (int k = 0; k < d_k; k++) Ob[(size_t)i * d_k + k] = acc[(size_t)i * d_k + k] * inv;
    }
    profiler_stop(TIMER_COMPUTE);

    profiler_wait_barrier(comm);
    profiler_start(TIMER_COMM);
    MPI_Gatherv(Ob, my_rows * d_k, MPI_FLOAT,
                rank == 0 ? out_full->data : NULL, cnt, dsp, MPI_FLOAT, 0, comm);
    profiler_count_msg((long)my_rows * d_k * sizeof(float));
    profiler_stop(TIMER_COMM);

    free(Qb); free(Kc); free(Vc); free(Kn); free(Vn); free(Ob);
    free(m); free(l); free(acc);
    free(cnt); free(dsp); free(rows_of);
}

/* ================================================================== */
/*  Public dispatch                                                    */
/* ================================================================== */
void tensor_parallel_attention(const Tensor *Q_full, const Tensor *K_full,
                                const Tensor *V_full, Tensor *out_full,
                                TPVariant variant, MPI_Comm comm) {
    switch (variant) {
        case TP_CANNON: tp_cannon(Q_full, K_full, V_full, out_full, comm);    break;
        case TP_RING:   tp_ring(Q_full, K_full, V_full, out_full, comm);      break;
        case TP_1D:
        default:        tp_allgather(Q_full, K_full, V_full, out_full, comm); break;
    }
}
