#include "hybrid.h"
#include "tensor_parallel.h"
#include "profiler.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * M3 — Hybrid (task x data) attention.
 *
 * Processes split into `num_groups` groups via MPI_Comm_split (data level inside
 * a group, task level across groups). Distribution and collection between the
 * master and the group roots use a COLLECTIVE over a dedicated group-roots
 * sub-communicator (MPI_Scatter / MPI_Gather) instead of a serial rank-0
 * Send/Recv loop, so the fan-out/fan-in is O(log G) rather than O(G) and there
 * is no master serialization bottleneck. Groups are equal size (main.c enforces
 * size % G == 0 and heads % G == 0), so the per-group blocks are uniform and a
 * plain Scatter/Gather suffices.
 */
void hybrid_mha(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
                Tensor *out_full, int num_heads, int num_groups, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int seq_len = (rank == 0) ? Q_full->rows : 0;
    int d_model = (rank == 0) ? Q_full->cols : 0;
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d_model, 1, MPI_INT, 0, comm);

    int d_k             = d_model / num_heads;
    int heads_per_group = num_heads / num_groups;
    int procs_per_group = size / num_groups;
    int my_group        = rank / procs_per_group;

    /* sub_comm: the ranks within one group (used for the inner 1D tensor kernel). */
    profiler_start(TIMER_COMM);
    MPI_Comm sub_comm;
    MPI_Comm_split(comm, my_group, rank, &sub_comm);
    int sub_rank, sub_size;
    MPI_Comm_rank(sub_comm, &sub_rank);
    MPI_Comm_size(sub_comm, &sub_size);

    /* roots_comm: exactly one rank per group (each group's root, sub_rank == 0),
     * ordered by group so roots_comm rank g == group g's root, and global rank 0
     * (group 0's root) is roots_comm rank 0. Other ranks get MPI_COMM_NULL. */
    MPI_Comm roots_comm;
    MPI_Comm_split(comm, sub_rank == 0 ? 0 : MPI_UNDEFINED, my_group, &roots_comm);
    profiler_stop(TIMER_COMM);

    int group_cols = heads_per_group * d_k;

    /* Each group root's Q/K/V/O slice for its heads. */
    Tensor Qg = tensor_alloc(seq_len, group_cols);
    Tensor Kg = tensor_alloc(seq_len, group_cols);
    Tensor Vg = tensor_alloc(seq_len, group_cols);
    Tensor Og = tensor_alloc(seq_len, group_cols);
    tensor_zero(&Og);

    /* DISTRIBUTE: master packs each group's column slice contiguously and
     * scatters one equal-size block to every group root in a single collective. */
    profiler_wait_barrier(comm);   /* separate sync wait from transfer */
    profiler_start(TIMER_COMM);
    if (roots_comm != MPI_COMM_NULL) {
        int roots_rank;
        MPI_Comm_rank(roots_comm, &roots_rank);
        size_t blk = (size_t)seq_len * group_cols;
        float *packQ = NULL, *packK = NULL, *packV = NULL;

        if (roots_rank == 0) {
            packQ = malloc(blk * num_groups * sizeof(float));
            packK = malloc(blk * num_groups * sizeof(float));
            packV = malloc(blk * num_groups * sizeof(float));
            for (int g = 0; g < num_groups; g++) {
                int cs = g * group_cols;     /* group g owns columns [cs, cs+group_cols) */
                for (int i = 0; i < seq_len; i++) {
                    memcpy(packQ + g * blk + (size_t)i * group_cols, &AT(*Q_full, i, cs), group_cols * sizeof(float));
                    memcpy(packK + g * blk + (size_t)i * group_cols, &AT(*K_full, i, cs), group_cols * sizeof(float));
                    memcpy(packV + g * blk + (size_t)i * group_cols, &AT(*V_full, i, cs), group_cols * sizeof(float));
                }
            }
        }

        MPI_Scatter(packQ, seq_len * group_cols, MPI_FLOAT, Qg.data, seq_len * group_cols, MPI_FLOAT, 0, roots_comm);
        MPI_Scatter(packK, seq_len * group_cols, MPI_FLOAT, Kg.data, seq_len * group_cols, MPI_FLOAT, 0, roots_comm);
        MPI_Scatter(packV, seq_len * group_cols, MPI_FLOAT, Vg.data, seq_len * group_cols, MPI_FLOAT, 0, roots_comm);
        profiler_count_msg((long)seq_len * group_cols * sizeof(float) * 3);

        if (roots_rank == 0) { free(packQ); free(packK); free(packV); }
    }
    profiler_stop(TIMER_COMM);

    /* COMPUTE: each group runs the 1D tensor-parallel kernel on its heads, one at
     * a time, within its sub-communicator. */
    for (int h = 0; h < heads_per_group; h++) {
        Tensor Qh = tensor_alloc(seq_len, d_k);
        Tensor Kh = tensor_alloc(seq_len, d_k);
        Tensor Vh = tensor_alloc(seq_len, d_k);
        Tensor Oh = tensor_alloc(seq_len, d_k);
        tensor_zero(&Oh);

        if (sub_rank == 0) {
            for (int i = 0; i < seq_len; i++) {
                memcpy(&AT(Qh, i, 0), &AT(Qg, i, h * d_k), d_k * sizeof(float));
                memcpy(&AT(Kh, i, 0), &AT(Kg, i, h * d_k), d_k * sizeof(float));
                memcpy(&AT(Vh, i, 0), &AT(Vg, i, h * d_k), d_k * sizeof(float));
            }
        }

        /* Sub-groups are usually not perfect squares (e.g. size 2), so use the 1D
         * path which works for any process count. */
        tensor_parallel_attention(&Qh, &Kh, &Vh, &Oh, TP_1D, sub_comm);

        if (sub_rank == 0) {
            for (int i = 0; i < seq_len; i++)
                memcpy(&AT(Og, i, h * d_k), &AT(Oh, i, 0), d_k * sizeof(float));
        }

        tensor_free(&Qh); tensor_free(&Kh);
        tensor_free(&Vh); tensor_free(&Oh);
    }

    /* COLLECT: gather each group root's result back to the master in a single
     * collective, then unpack into the right columns of out_full. */
    profiler_wait_barrier(comm);   /* separate sync wait from transfer */
    profiler_start(TIMER_COMM);
    if (roots_comm != MPI_COMM_NULL) {
        int roots_rank;
        MPI_Comm_rank(roots_comm, &roots_rank);
        size_t blk = (size_t)seq_len * group_cols;
        float *packO = (roots_rank == 0) ? malloc(blk * num_groups * sizeof(float)) : NULL;

        MPI_Gather(Og.data, seq_len * group_cols, MPI_FLOAT, packO, seq_len * group_cols, MPI_FLOAT, 0, roots_comm);
        profiler_count_msg((long)seq_len * group_cols * sizeof(float));

        if (roots_rank == 0) {
            for (int g = 0; g < num_groups; g++) {
                int cs = g * group_cols;
                for (int i = 0; i < seq_len; i++)
                    memcpy(&AT(*out_full, i, cs), packO + g * blk + (size_t)i * group_cols, group_cols * sizeof(float));
            }
            free(packO);
        }
    }
    profiler_stop(TIMER_COMM);

    tensor_free(&Qg); tensor_free(&Kg);
    tensor_free(&Vg); tensor_free(&Og);
    if (roots_comm != MPI_COMM_NULL) MPI_Comm_free(&roots_comm);
    MPI_Comm_free(&sub_comm);
}
