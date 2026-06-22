#include "hybrid.h"
#include "tensor_parallel.h"
#include "profiler.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void hybrid_mha(const Tensor *Q_full, const Tensor *K_full, const Tensor *V_full,
                Tensor *out_full, int num_heads, int num_groups, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int seq_len = (rank == 0) ? Q_full->rows : 0;
    int d_model = (rank == 0) ? Q_full->cols : 0;
    MPI_Bcast(&seq_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&d_model, 1, MPI_INT, 0, comm);

    int d_k = d_model / num_heads;
    int heads_per_group = num_heads / num_groups;
    int procs_per_group = size / num_groups;

    /* Assign each rank to a group (color) */
    int my_group = rank / procs_per_group;
    int my_color = my_group;

    profiler_start(TIMER_COMM);
    MPI_Comm sub_comm;
    MPI_Comm_split(comm, my_color, rank, &sub_comm);
    profiler_stop(TIMER_COMM);

    int sub_rank, sub_size;
    MPI_Comm_rank(sub_comm, &sub_rank);
    MPI_Comm_size(sub_comm, &sub_size);

    /* The root of each sub-comm (sub_rank == 0) needs the head slices for its group */
    int head_start = my_group * heads_per_group;
    int group_cols = heads_per_group * d_k;

    /* Tensor for this group's Q/K/V slice (only meaningful at group root == sub_rank 0) */
    Tensor Qg = tensor_alloc(seq_len, group_cols);
    Tensor Kg = tensor_alloc(seq_len, group_cols);
    Tensor Vg = tensor_alloc(seq_len, group_cols);
    Tensor Og = tensor_alloc(seq_len, group_cols);
    tensor_zero(&Og);

    /* The global root (rank 0) extracts each group's slice and sends to the group root */
    profiler_start(TIMER_COMM);
    if (rank == 0) {
        /* Own group: copy directly */
        for (int i = 0; i < seq_len; i++) {
            memcpy(&AT(Qg, i, 0), &AT(*Q_full, i, head_start * d_k), group_cols * sizeof(float));
            memcpy(&AT(Kg, i, 0), &AT(*K_full, i, head_start * d_k), group_cols * sizeof(float));
            memcpy(&AT(Vg, i, 0), &AT(*V_full, i, head_start * d_k), group_cols * sizeof(float));
        }

        /* Send slices to each other group's root (the first rank of that group) */
        for (int g = 1; g < num_groups; g++) {
            int dest_rank  = g * procs_per_group;   /* global rank of group g's root */
            int gs         = g * heads_per_group;
            int gc         = heads_per_group * d_k;

            float *tmp_q = malloc((size_t)seq_len * gc * sizeof(float));
            float *tmp_k = malloc((size_t)seq_len * gc * sizeof(float));
            float *tmp_v = malloc((size_t)seq_len * gc * sizeof(float));

            for (int i = 0; i < seq_len; i++) {
                memcpy(tmp_q + i * gc, &AT(*Q_full, i, gs * d_k), gc * sizeof(float));
                memcpy(tmp_k + i * gc, &AT(*K_full, i, gs * d_k), gc * sizeof(float));
                memcpy(tmp_v + i * gc, &AT(*V_full, i, gs * d_k), gc * sizeof(float));
            }

            MPI_Send(tmp_q, seq_len * gc, MPI_FLOAT, dest_rank, 10, comm);
            MPI_Send(tmp_k, seq_len * gc, MPI_FLOAT, dest_rank, 11, comm);
            MPI_Send(tmp_v, seq_len * gc, MPI_FLOAT, dest_rank, 12, comm);
            free(tmp_q); free(tmp_k); free(tmp_v);
        }
    } else if (sub_rank == 0 && my_group > 0) {
        /* Group root (not global root): receive from global root */
        MPI_Recv(Qg.data, seq_len * group_cols, MPI_FLOAT, 0, 10, comm, MPI_STATUS_IGNORE);
        MPI_Recv(Kg.data, seq_len * group_cols, MPI_FLOAT, 0, 11, comm, MPI_STATUS_IGNORE);
        MPI_Recv(Vg.data, seq_len * group_cols, MPI_FLOAT, 0, 12, comm, MPI_STATUS_IGNORE);
    }
    profiler_stop(TIMER_COMM);

    /* Each group runs tensor-parallel attention on its heads sequentially */
    for (int h = 0; h < heads_per_group; h++) {
        Tensor Qh = tensor_alloc(seq_len, d_k);
        Tensor Kh = tensor_alloc(seq_len, d_k);
        Tensor Vh = tensor_alloc(seq_len, d_k);
        Tensor Oh = tensor_alloc(seq_len, d_k);
        tensor_zero(&Oh);

        /* Extract head h from the group slice (only sub_rank 0 has valid data) */
        if (sub_rank == 0) {
            for (int i = 0; i < seq_len; i++) {
                memcpy(&AT(Qh, i, 0), &AT(Qg, i, h * d_k), d_k * sizeof(float));
                memcpy(&AT(Kh, i, 0), &AT(Kg, i, h * d_k), d_k * sizeof(float));
                memcpy(&AT(Vh, i, 0), &AT(Vg, i, h * d_k), d_k * sizeof(float));
            }
        }

        /* Run tensor-parallel attention for this head within the sub-communicator.
         * Sub-groups are typically not perfect squares (e.g. size 2), so we use
         * the 1D path (use_cannon = 0) which works for any process count. */
        tensor_parallel_attention(&Qh, &Kh, &Vh, &Oh, 0, sub_comm);

        /* sub_rank 0 writes result into Og */
        if (sub_rank == 0) {
            for (int i = 0; i < seq_len; i++)
                memcpy(&AT(Og, i, h * d_k), &AT(Oh, i, 0), d_k * sizeof(float));
        }

        tensor_free(&Qh); tensor_free(&Kh);
        tensor_free(&Vh); tensor_free(&Oh);
    }

    /* Collect results from each group's root back to global root (rank 0) */
    profiler_start(TIMER_COMM);
    if (rank == 0) {
        /* Copy own group result directly */
        for (int i = 0; i < seq_len; i++)
            memcpy(&AT(*out_full, i, head_start * d_k), &AT(Og, i, 0), group_cols * sizeof(float));

        /* Receive from each other group's root */
        for (int g = 1; g < num_groups; g++) {
            int src_rank = g * procs_per_group;
            int gs       = g * heads_per_group;
            int gc       = heads_per_group * d_k;
            float *tmp   = malloc((size_t)seq_len * gc * sizeof(float));

            MPI_Recv(tmp, seq_len * gc, MPI_FLOAT, src_rank, 20, comm, MPI_STATUS_IGNORE);

            for (int i = 0; i < seq_len; i++)
                memcpy(&AT(*out_full, i, gs * d_k), tmp + i * gc, gc * sizeof(float));
            free(tmp);
        }
    } else if (sub_rank == 0 && my_group > 0) {
        MPI_Send(Og.data, seq_len * group_cols, MPI_FLOAT, 0, 20, comm);
    }
    profiler_stop(TIMER_COMM);

    tensor_free(&Qg); tensor_free(&Kg);
    tensor_free(&Vg); tensor_free(&Og);
    MPI_Comm_free(&sub_comm);
}
