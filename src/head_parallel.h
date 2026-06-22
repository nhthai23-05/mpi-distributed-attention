#ifndef HEAD_PARALLEL_H
#define HEAD_PARALLEL_H

#include "tensor.h"
#include <mpi.h>

/*
 * M1 — Head Parallelism (task parallelism).
 *
 * Q_full, K_full, V_full: [seq_len x d_model]  (only meaningful on rank 0)
 * out_full: [seq_len x d_model]                 (result written on rank 0)
 * num_heads: total attention heads (should equal MPI world size for simplicity)
 * comm: communicator to use (typically MPI_COMM_WORLD)
 */
void head_parallel_mha(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
                       Tensor *out_full, int num_heads, MPI_Comm comm);

#endif
