#ifndef TENSOR_PARALLEL_H
#define TENSOR_PARALLEL_H

#include "tensor.h"
#include <mpi.h>

/*
 * Tensor-parallel variant selector.
 *   TP_1D     row-block decomposition; broadcasts full K,V to every rank.
 *             Simple, works for ANY process count, bitwise-identical to seq.
 *   TP_CANNON 2D Cannon's algorithm + distributed softmax. Needs a perfect-square
 *             process count and seq_len, d_k both divisible by sqrt(P).
 *   TP_RING   Ring (Flash-style) attention: K,V are SHARDED (N/P rows per rank,
 *             not replicated) and streamed around a ring with non-blocking
 *             double-buffered shifts, folded into a running online softmax. Each
 *             rank holds only O(N/P) of K,V at once, and the shift overlaps the
 *             compute, so exposed communication time is largely hidden. Works for
 *             any process count.
 */
typedef enum { TP_1D = 0, TP_CANNON = 1, TP_RING = 2 } TPVariant;

/*
 * M2 — Tensor Parallelism for a SINGLE attention head distributed across all
 * ranks in `comm`.  Computes out = softmax(Q * K^T / sqrt(d_k)) * V.
 *
 * Q_full, K_full, V_full: [seq_len x d_k]  (meaningful on rank 0 only)
 * out_full: [seq_len x d_k]                 (result on rank 0 only)
 */
void tensor_parallel_attention(const Tensor *Q_full, const Tensor *K_full,
                                const Tensor *V_full, Tensor *out_full,
                                TPVariant variant, MPI_Comm comm);

#endif
