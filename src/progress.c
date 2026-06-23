#include "progress.h"
#include <mpi.h>
#include <stdio.h>

static int         g_on    = 0;    /* enabled by --progress              */
static int         g_rank  = -1;   /* cached world rank (only 0 prints)  */
static long        g_total = 1;    /* total units in the current phase   */
static double      g_t0    = 0.0;  /* phase start time (MPI_Wtime)       */
static double      g_last  = -1.0; /* elapsed at last print (rate limit) */
static const char *g_label = "work";

void progress_enable(int on) {
    g_on = on;
    if (g_rank < 0) MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
}

void progress_begin(const char *label, long total) {
    if (!g_on || g_rank != 0) return;
    g_label = label ? label : "work";
    g_total = total > 0 ? total : 1;
    g_t0    = MPI_Wtime();
    g_last  = -1.0;
    fprintf(stderr, "[progress] %s: starting (%ld units)\r", g_label, g_total);
    fflush(stderr);
}

void progress_update(long done) {
    if (!g_on || g_rank != 0) return;

    double elapsed = MPI_Wtime() - g_t0;
    /* Rate-limit to ~2 prints/sec, but always allow the final tick. */
    if (done < g_total && g_last >= 0.0 && (elapsed - g_last) < 0.5) return;
    g_last = elapsed;

    double frac = (double)done / (double)g_total;
    if (frac < 1e-9) frac = 1e-9;
    double eta = elapsed * (1.0 - frac) / frac;   /* simple linear extrapolation */

    fprintf(stderr, "[progress] %s: %3.0f%%  elapsed=%6.1fs  eta=%6.1fs   \r",
            g_label, frac * 100.0, elapsed, eta);
    fflush(stderr);
}

void progress_end(void) {
    if (!g_on || g_rank != 0) return;
    double elapsed = MPI_Wtime() - g_t0;
    fprintf(stderr, "[progress] %s: 100%%  elapsed=%6.1fs              \n",
            g_label, elapsed);
    fflush(stderr);
}
