#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

/*
 * Streaming single-query-row attention (the shared compute kernel).
 *
 *   o = softmax( q . K^T / sqrt(d_k) ) . V
 *
 * q:   [d_k]               one query vector
 * K,V: [n x d_k]           all keys / values, row-major
 * o:   [d_k]               output row (overwritten)
 * s:   [n]                 caller-provided score scratch (avoids per-row malloc)
 *
 * Processes one score row at a time, so it never materializes the full n x n
 * score matrix: O(n) extra memory instead of O(n^2). The accumulation order is
 * identical to the old materialized matmul path, so results are bitwise the
 * same as the reference baseline.
 */
void attention_row(const float *q, const float *K, const float *V,
                   int n, int d_k, float *o, float *s);

/*
 * Sequential single-head attention.
 * Q: [seq_len x d_k]  K: [seq_len x d_k]  V: [seq_len x d_v]
 * out: [seq_len x d_v]  (must be pre-allocated)
 *
 * out = softmax(Q * K^T / sqrt(d_k)) * V
 */
void attention_seq(const Tensor *Q, const Tensor *K, const Tensor *V, Tensor *out);

/*
 * Multi-head sequential attention (reference baseline).
 * Q_full, K_full, V_full: [seq_len x d_model]
 * out_full: [seq_len x d_model]  (pre-allocated)
 * num_heads: number of heads
 */
void mha_seq(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
             Tensor *out_full, int num_heads);

#endif
