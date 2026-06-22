#ifndef TENSOR_PARALLEL_H
#define TENSOR_PARALLEL_H

#include "tensor.h"
#include <mpi.h>

/*
 * M2 — Tensor Parallelism for a SINGLE attention head distributed across all
 * ranks in `comm`.  Computes out = softmax(Q * K^T / sqrt(d_k)) * V.
 *
 * use_cannon == 0 (default):
 *   1D row-block decomposition + local softmax. Works for ANY process count.
 * use_cannon == 1:
 *   2D Cannon's algorithm (MPI_Sendrecv_replace) + distributed softmax
 *   (MPI_Allreduce over a row sub-communicator). Requires:
 *     - size of `comm` is a perfect square (P = sqrt_p^2)
 *     - seq_len and d_k both divisible by sqrt_p
 *
 * Q_full, K_full, V_full: [seq_len x d_k]  (meaningful on rank 0 only)
 * out_full: [seq_len x d_k]                 (result on rank 0 only)
 */
void tensor_parallel_attention(const Tensor *Q_full, const Tensor *K_full,
                                const Tensor *V_full, Tensor *out_full,
                                int use_cannon, MPI_Comm comm);

#endif
