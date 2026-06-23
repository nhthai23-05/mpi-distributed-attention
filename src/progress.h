#ifndef PROGRESS_H
#define PROGRESS_H

/*
 * Lightweight, rank-0-only progress meter that prints to STDERR.
 *
 * It is a no-op unless explicitly enabled (--progress), and it only ever
 * prints from rank 0, so it never corrupts the CSV that the benchmark
 * scripts collect on STDOUT.
 *
 * Typical use inside a long loop:
 *     progress_begin("tensor-attn rows", my_rows);
 *     for (i = 0; i < my_rows; i++) { ...; progress_update(i + 1); }
 *     progress_end();
 *
 * progress_update() is cheap and internally rate-limited (~2 prints/sec),
 * so it is safe to call on every iteration of a hot loop.
 */
void progress_enable(int on);                       /* call once after MPI_Init */
void progress_begin(const char *label, long total); /* start a phase           */
void progress_update(long done);                    /* report units completed  */
void progress_end(void);                            /* finish the phase         */

#endif
