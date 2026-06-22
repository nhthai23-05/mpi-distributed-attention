#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

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
