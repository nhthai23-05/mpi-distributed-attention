#include "tensor_parallel.h"
#include "tensor.h"
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
    profiler_start(TIMER_COMM);
    MPI_Scatterv(rank == 0 ? Q_full->data : NULL, cnt, dsp, MPI_FLOAT,
                 Qb, my_rows * d_k, MPI_FLOAT, 0, comm);

    /* Broadcast full K and V (each process needs all keys/values) */
    if (rank == 0) {
        memcpy(Kf, K_full->data, (size_t)seq_len * d_k * sizeof(float));
        memcpy(Vf, V_full->data, (size_t)seq_len * d_k * sizeof(float));
    }
    MPI_Bcast(Kf, seq_len * d_k, MPI_FLOAT, 0, comm);
    MPI_Bcast(Vf, seq_len * d_k, MPI_FLOAT, 0, comm);
    profiler_stop(TIMER_COMM);

    /* Local attention for each owned query row */
    profiler_start(TIMER_COMPUTE);
    float scale = 1.0f / sqrtf((float)d_k);
    float *srow = malloc((size_t)seq_len * sizeof(float));

    progress_begin("tensor-attn rows", my_rows);
    for (int i = 0; i < my_rows; i++) {
        /* scores = Q_i . K_j^T * scale, track row max for stability */
        float mx = -INFINITY;
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d_k; k++)
                dot += Qb[i * d_k + k] * Kf[j * d_k + k];
            srow[j] = dot * scale;
            if (srow[j] > mx) mx = srow[j];
        }
        /* softmax (local — this row lives entirely on this process) */
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            srow[j] = expf(srow[j] - mx);
            sum += srow[j];
        }
        for (int j = 0; j < seq_len; j++) srow[j] /= sum;

        /* O_i = softmax_row . V
         * Accumulate row-by-row over j so V is read sequentially (row-major)
         * instead of strided down a column. Each Ob[i][k] still sums j in the
         * same 0..seq_len-1 order, so the result is bitwise identical. */
        float *orow = &Ob[i * d_k];
        for (int k = 0; k < d_k; k++) orow[k] = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float w = srow[j];
            const float *vrow = &Vf[j * d_k];
            for (int k = 0; k < d_k; k++)
                orow[k] += w * vrow[k];
        }
        progress_update(i + 1);
    }
    progress_end();
    free(srow);
    profiler_stop(TIMER_COMPUTE);

    /* Gather output rows back to root */
    profiler_start(TIMER_COMM);
    MPI_Gatherv(Ob, my_rows * d_k, MPI_FLOAT,
                rank == 0 ? out_full->data : NULL, cnt, dsp, MPI_FLOAT, 0, comm);
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
    MPI_Cart_shift(cart, 0, -coords[1], &src, &dst);
    MPI_Sendrecv_replace(B, bk * bc, MPI_FLOAT, dst, 2, src, 2, cart, MPI_STATUS_IGNORE);

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
        MPI_Cart_shift(cart, 0, -1, &src, &dst);
        MPI_Sendrecv_replace(B, bk * bc, MPI_FLOAT, dst, 4, src, 4, cart, MPI_STATUS_IGNORE);
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

        float lsum = 0.0f;
        for (int b = 0; b < bn; b++) {
            S[a * bn + b] = expf(S[a * bn + b] - gmax);
            lsum += S[a * bn + b];
        }

        float gsum;
        MPI_Allreduce(&lsum, &gsum, 1, MPI_FLOAT, MPI_SUM, row_comm);

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
                if (dest == 0)
                    memcpy(block_buf, tmp, (size_t)br * bc * sizeof(float));
                else
                    MPI_Send(tmp, br * bc, MPI_FLOAT, dest, 77, cart);
                free(tmp);
            }
    } else {
        MPI_Recv(block_buf, br * bc, MPI_FLOAT, 0, 77, cart, MPI_STATUS_IGNORE);
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
                if (src == 0)
                    memcpy(tmp, block_buf, (size_t)br * bc * sizeof(float));
                else
                    MPI_Recv(tmp, br * bc, MPI_FLOAT, src, 78, cart, MPI_STATUS_IGNORE);
                for (int a = 0; a < br; a++)
                    for (int b = 0; b < bc; b++)
                        full[(pi * br + a) * C + (pj * bc + b)] = tmp[a * bc + b];
                free(tmp);
            }
    } else {
        MPI_Send(block_buf, br * bc, MPI_FLOAT, 0, 78, cart);
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
/*  Public dispatch                                                    */
/* ================================================================== */
void tensor_parallel_attention(const Tensor *Q_full, const Tensor *K_full,
                                const Tensor *V_full, Tensor *out_full,
                                int use_cannon, MPI_Comm comm) {
    if (use_cannon)
        tp_cannon(Q_full, K_full, V_full, out_full, comm);
    else
        tp_allgather(Q_full, K_full, V_full, out_full, comm);
}
