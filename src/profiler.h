#ifndef PROFILER_H
#define PROFILER_H

/* Three timer categories tracked per rank */
typedef enum { TIMER_IO = 0, TIMER_COMPUTE = 1, TIMER_COMM = 2, TIMER_COUNT = 3 } TimerKind;

void   profiler_init(void);
void   profiler_start(TimerKind k);
void   profiler_stop(TimerKind k);
double profiler_get(TimerKind k);

/* Print one CSV line per rank: rank,seq_len,t_io,t_compute,t_comm */
void   profiler_print_csv(int rank, int seq_len);

/* Print human-readable summary (rank 0 only) */
void   profiler_print_summary(int rank);

#endif
