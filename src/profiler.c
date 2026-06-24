#include "profiler.h"
#include <mpi.h>
#include <stdio.h>
#include <string.h>

static double t_start[TIMER_COUNT];
static double t_accum[TIMER_COUNT];
static long   g_msgs    = 0;     /* MPI data transfers this rank took part in   */
static long   g_bytes   = 0;     /* bytes this rank sent/received               */
static int    g_wait_on = 0;     /* wait-profiling toggle (see header)          */
static double g_latency = 0.0;   /* worst-case one-way link latency, seconds    */

void profiler_init(void) {
    memset(t_accum, 0, sizeof(t_accum));
    g_msgs    = 0;
    g_bytes   = 0;
    g_latency = 0.0;
    /* g_wait_on is intentionally NOT reset here so the order of
     * profiler_init / profiler_enable_wait in main() does not matter. */
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

void profiler_count_msg(long bytes) {
    g_msgs  += 1;
    g_bytes += bytes;
}

long profiler_msgs(void)  { return g_msgs; }
long profiler_bytes(void) { return g_bytes; }

void profiler_enable_wait(int on) {
    g_wait_on = on;
}

void profiler_wait_barrier(MPI_Comm comm) {
    if (!g_wait_on) return;
    profiler_start(TIMER_WAIT);
    MPI_Barrier(comm);
    profiler_stop(TIMER_WAIT);
}

double profiler_measure_latency(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size < 2) { g_latency = 0.0; return 0.0; }

    const int reps = 64;
    double buf = 0.0;
    double worst = 0.0;   /* worst-case one-way latency over all rank-0 links */

    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            /* one warm-up round trip (exclude connection setup) */
            MPI_Send(&buf, 1, MPI_DOUBLE, r, 99, comm);
            MPI_Recv(&buf, 1, MPI_DOUBLE, r, 99, comm, MPI_STATUS_IGNORE);

            double t0 = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                MPI_Send(&buf, 1, MPI_DOUBLE, r, 99, comm);
                MPI_Recv(&buf, 1, MPI_DOUBLE, r, 99, comm, MPI_STATUS_IGNORE);
            }
            double oneway = (MPI_Wtime() - t0) / reps / 2.0;
            if (oneway > worst) worst = oneway;
        }
    } else {
        MPI_Recv(&buf, 1, MPI_DOUBLE, 0, 99, comm, MPI_STATUS_IGNORE);
        MPI_Send(&buf, 1, MPI_DOUBLE, 0, 99, comm);
        for (int i = 0; i < reps; i++) {
            MPI_Recv(&buf, 1, MPI_DOUBLE, 0, 99, comm, MPI_STATUS_IGNORE);
            MPI_Send(&buf, 1, MPI_DOUBLE, 0, 99, comm);
        }
    }

    MPI_Bcast(&worst, 1, MPI_DOUBLE, 0, comm);
    g_latency = worst;
    return worst;
}

double profiler_latency(void) { return g_latency; }

void profiler_print_csv(int rank, int seq_len) {
    printf("%d,%d,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%.3f\n",
           rank, seq_len,
           t_accum[TIMER_IO],
           t_accum[TIMER_COMPUTE],
           t_accum[TIMER_COMM],
           t_accum[TIMER_WAIT],
           g_msgs,
           g_bytes,
           g_latency * 1e6);   /* latency in microseconds */
}

void profiler_print_summary(int rank) {
    if (rank != 0) return;
    printf("[Rank 0]  IO=%.4fs  Compute=%.4fs  Comm=%.4fs  Wait=%.4fs  "
           "msgs=%ld  bytes=%ld  link_latency=%.1fus\n",
           t_accum[TIMER_IO],
           t_accum[TIMER_COMPUTE],
           t_accum[TIMER_COMM],
           t_accum[TIMER_WAIT],
           g_msgs,
           g_bytes,
           g_latency * 1e6);
}
