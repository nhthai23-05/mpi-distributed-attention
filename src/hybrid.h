#ifndef HYBRID_H
#define HYBRID_H

#include "tensor.h"
#include <mpi.h>

/*
 * M3 — Hybrid Parallelism.
 *
 * Splits MPI_COMM_WORLD into sub-communicators (one per head group).
 * Each sub-communicator runs tensor-parallel attention on its assigned heads.
 * Results are gathered back to rank 0.
 *
 * Q_full, K_full, V_full: [seq_len x d_model]  (meaningful on rank 0)
 * out_full: [seq_len x d_model]                 (result on rank 0)
 * num_heads: total heads (must be divisible by num_groups)
 * num_groups: number of sub-communicator groups (each gets num_heads/num_groups heads)
 */
void hybrid_mha(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
                Tensor *out_full, int num_heads, int num_groups, MPI_Comm comm);

#endif
