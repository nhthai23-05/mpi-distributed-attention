#include "profiler.h"
#include <mpi.h>
#include <stdio.h>
#include <string.h>

static double t_start[TIMER_COUNT];
static double t_accum[TIMER_COUNT];

void profiler_init(void) {
    memset(t_accum, 0, sizeof(t_accum));
}

void profiler_start(TimerKind k) {
    t_start[k] = MPI_Wtime();
}

void profiler_stop(TimerKind k) {
    t_accum[k] += MPI_Wtime() - t_start[k];
}

double profiler_get(TimerKind k) {
    return t_accum[k];
}

void profiler_print_csv(int rank, int seq_len) {
    printf("%d,%d,%.6f,%.6f,%.6f\n",
           rank, seq_len,
           t_accum[TIMER_IO],
           t_accum[TIMER_COMPUTE],
           t_accum[TIMER_COMM]);
}

void profiler_print_summary(int rank) {
    if (rank != 0) return;
    printf("[Rank 0]  IO=%.4fs  Compute=%.4fs  Comm=%.4fs\n",
           t_accum[TIMER_IO],
           t_accum[TIMER_COMPUTE],
           t_accum[TIMER_COMM]);
}
