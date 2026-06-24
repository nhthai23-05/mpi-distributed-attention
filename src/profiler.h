#ifndef PROFILER_H
#define PROFILER_H

#include <mpi.h>

/* Per-rank timing categories.
 *   IO       - input generation
 *   COMPUTE  - local matrix work
 *   COMM     - actual MPI data transfer
 *   WAIT     - idle time blocked at a synchronization point (only measured when
 *              wait-profiling is enabled; separates "waiting for the slowest
 *              rank" from real transfer, so a slow run can be explained). */
typedef enum { TIMER_IO = 0, TIMER_COMPUTE = 1, TIMER_COMM = 2, TIMER_WAIT = 3,
               TIMER_COUNT = 4 } TimerKind;

void   profiler_init(void);
void   profiler_start(TimerKind k);
void   profiler_stop(TimerKind k);
double profiler_get(TimerKind k);

/* Communication-volume counters. Call once per MPI data transfer this rank
 * participates in, passing the number of bytes this rank sends/receives. */
void   profiler_count_msg(long bytes);
long   profiler_msgs(void);
long   profiler_bytes(void);

/* Idle/wait timing. When enabled, profiler_wait_barrier() brackets an
 * MPI_Barrier into TIMER_WAIT so the following collective measures pure
 * transfer. Disabled by default so headline timing runs are unperturbed. */
void   profiler_enable_wait(int on);
void   profiler_wait_barrier(MPI_Comm comm);

/* One-off network-latency probe: rank-0 ping-pongs a few bytes with every other
 * rank; returns (and stores, broadcast to all ranks) the worst-case one-way
 * link latency in seconds. Returns 0 for a single process. Uses raw MPI so it
 * does not pollute the message/byte counters. */
double profiler_measure_latency(MPI_Comm comm);
double profiler_latency(void);   /* last measured one-way latency, seconds */

/* CSV line per rank:
 *   rank,seq_len,t_io,t_compute,t_comm,t_wait,msgs,bytes,latency_us */
void   profiler_print_csv(int rank, int seq_len);

/* Human-readable summary (rank 0 only). */
void   profiler_print_summary(int rank);

#endif
