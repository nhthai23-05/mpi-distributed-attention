#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "data_gen.h"
#include "profiler.h"
#include "attention.h"
#include "head_parallel.h"
#include "tensor_parallel.h"
#include "hybrid.h"
#include "progress.h"

/* ------------------------------------------------------------------ */
/* Configuration defaults                                               */
/* ------------------------------------------------------------------ */
#define DEFAULT_SEQ_LEN   512
#define DEFAULT_NUM_HEADS 4
#define DEFAULT_D_MODEL   512    /* must be divisible by num_heads */

/* ------------------------------------------------------------------ */
/* Correctness check                                                    */
/* ------------------------------------------------------------------ */
static float max_abs_error(const Tensor *a, const Tensor *b) {
    float err = 0.0f;
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a->data[i] - b->data[i]);
        if (d > err) err = d;
    }
    return err;
}

/* ------------------------------------------------------------------ */
/* CLI                                                                  */
/* ------------------------------------------------------------------ */
typedef enum { MODE_SEQ, MODE_HEAD, MODE_TENSOR, MODE_HYBRID } Mode;

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --mode   <seq|head|tensor|hybrid>  (default: seq)\n"
        "  --seq-len <N>                       (default: %d)\n"
        "  --heads  <H>                        (default: %d)\n"
        "  --d-model <D>                       (default: %d)\n"
        "  --groups <G>                        groups for hybrid mode (default: 2)\n"
        "  --cannon                            tensor mode: use Cannon's algorithm\n"
        "                                      (needs perfect-square procs)\n"
        "  --csv                               output CSV timing line per rank\n"
        "  --no-check                          skip correctness check\n"
        "  --progress                          print rank-0 progress + ETA to stderr\n"
        "  --profile-wait                      add barriers to separate idle WAIT\n"
        "                                      time from real transfer time\n",
        prog, DEFAULT_SEQ_LEN, DEFAULT_NUM_HEADS, DEFAULT_D_MODEL);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Parse arguments */
    Mode mode       = MODE_SEQ;
    int  seq_len    = DEFAULT_SEQ_LEN;
    int  num_heads  = DEFAULT_NUM_HEADS;
    int  d_model    = DEFAULT_D_MODEL;
    int  num_groups = 2;
    int  use_cannon = 0;
    int  csv_output = 0;
    int  do_check   = 1;
    int  show_progress = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i+1 < argc) {
            i++;
            if      (strcmp(argv[i], "seq")    == 0) mode = MODE_SEQ;
            else if (strcmp(argv[i], "head")   == 0) mode = MODE_HEAD;
            else if (strcmp(argv[i], "tensor") == 0) mode = MODE_TENSOR;
            else if (strcmp(argv[i], "hybrid") == 0) mode = MODE_HYBRID;
            else { if (rank == 0) usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
        } else if (strcmp(argv[i], "--seq-len") == 0 && i+1 < argc) {
            seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i+1 < argc) {
            num_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--d-model") == 0 && i+1 < argc) {
            d_model = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--groups") == 0 && i+1 < argc) {
            num_groups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cannon") == 0) {
            use_cannon = 1;
        } else if (strcmp(argv[i], "--csv") == 0) {
            csv_output = 1;
        } else if (strcmp(argv[i], "--no-check") == 0) {
            do_check = 0;
        } else if (strcmp(argv[i], "--progress") == 0) {
            show_progress = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            if (rank == 0) usage(argv[0]);
            MPI_Finalize(); return 0;
        }
    }

    /* Validate configuration up front. Every rank parsed the same args, so all
     * reach the same verdict and exit collectively — no silent wrong results
     * or div-by-zero deep inside the parallel modes. */
    {
        const char *why = NULL;
        if      (seq_len   <= 0)                why = "--seq-len must be > 0";
        else if (num_heads <= 0)                why = "--heads must be > 0";
        else if (d_model   <= 0)                why = "--d-model must be > 0";
        else if (d_model % num_heads != 0)      why = "--d-model must be divisible by --heads";
        else if (mode == MODE_HYBRID) {
            if      (num_groups <= 0)             why = "--groups must be > 0";
            else if (size % num_groups != 0)     why = "process count must be divisible by --groups";
            else if (num_heads % num_groups != 0) why = "--heads must be divisible by --groups";
        }
        if (why) {
            if (rank == 0)
                fprintf(stderr,
                    "config error: %s (seq_len=%d heads=%d d_model=%d groups=%d procs=%d)\n",
                    why, seq_len, num_heads, d_model, num_groups, size);
            MPI_Finalize();
            return 1;
        }
    }

    /* Print CSV header once from rank 0 */
    if (csv_output && rank == 0)
        printf("rank,seq_len,t_io,t_compute,t_comm\n");

    profiler_init();
    progress_enable(show_progress);

    /* Allocate Q, K, V on all ranks (data_gen makes them identical everywhere) */
    Tensor Q = tensor_alloc(seq_len, d_model);
    Tensor K = tensor_alloc(seq_len, d_model);
    Tensor V = tensor_alloc(seq_len, d_model);

    profiler_start(TIMER_IO);
    data_gen_fill(&Q, 1);
    data_gen_fill(&K, 2);
    data_gen_fill(&V, 3);
    profiler_stop(TIMER_IO);

    /* Sequential baseline (rank 0 only, for correctness check) */
    Tensor out_seq = tensor_alloc(seq_len, d_model);
    Tensor out_par = tensor_alloc(seq_len, d_model);
    tensor_zero(&out_par);

    if (rank == 0 && (do_check || mode == MODE_SEQ)) {
        double t0 = MPI_Wtime();
        mha_seq(&Q, &K, &V, &out_seq, num_heads);
        double t1 = MPI_Wtime();
        if (!csv_output)
            printf("[SEQ] time=%.4fs  seq_len=%d  d_model=%d  heads=%d\n",
                   t1 - t0, seq_len, d_model, num_heads);
    }

    if (mode == MODE_SEQ) {
        if (rank == 0 && !csv_output)
            printf("Sequential mode done.\n");
        goto done;
    }

    /* --- Parallel modes --- */
    double wall_start = MPI_Wtime();

    if (mode == MODE_HEAD) {
        head_parallel_mha(&Q, &K, &V, &out_par, num_heads, MPI_COMM_WORLD);
    } else if (mode == MODE_TENSOR) {
        /* Tensor parallel on a single head (use first head's slice as a demo) */
        int d_k = d_model / num_heads;
        Tensor Q1 = tensor_alloc(seq_len, d_k);
        Tensor K1 = tensor_alloc(seq_len, d_k);
        Tensor V1 = tensor_alloc(seq_len, d_k);
        Tensor O1 = tensor_alloc(seq_len, d_k);
        tensor_zero(&O1);

        if (rank == 0) {
            for (int i = 0; i < seq_len; i++) {
                memcpy(&AT(Q1, i, 0), &AT(Q, i, 0), d_k * sizeof(float));
                memcpy(&AT(K1, i, 0), &AT(K, i, 0), d_k * sizeof(float));
                memcpy(&AT(V1, i, 0), &AT(V, i, 0), d_k * sizeof(float));
            }
        }

        tensor_parallel_attention(&Q1, &K1, &V1, &O1, use_cannon, MPI_COMM_WORLD);

        /* For correctness check, embed the single-head result into out_par */
        if (rank == 0) {
            tensor_zero(&out_par);
            for (int i = 0; i < seq_len; i++)
                memcpy(&AT(out_par, i, 0), &AT(O1, i, 0), d_k * sizeof(float));

            /* Re-compute seq baseline for just head 0 to compare */
            tensor_zero(&out_seq);
            Tensor Oh_seq = tensor_alloc(seq_len, d_k);
            attention_seq(&Q1, &K1, &V1, &Oh_seq);
            for (int i = 0; i < seq_len; i++)
                memcpy(&AT(out_seq, i, 0), &AT(Oh_seq, i, 0), d_k * sizeof(float));
            tensor_free(&Oh_seq);
        }
        tensor_free(&Q1); tensor_free(&K1);
        tensor_free(&V1); tensor_free(&O1);
    } else if (mode == MODE_HYBRID) {
        hybrid_mha(&Q, &K, &V, &out_par, num_heads, num_groups, MPI_COMM_WORLD);
    }

    double wall_end = MPI_Wtime();

    /* --- Output --- */
    if (csv_output) {
        MPI_Barrier(MPI_COMM_WORLD);
        profiler_print_csv(rank, seq_len);
    } else if (rank == 0) {
        printf("[%s] wall=%.4fs  seq_len=%d  d_model=%d  heads=%d  procs=%d\n",
               mode == MODE_HEAD ? "HEAD" : mode == MODE_TENSOR ? "TENSOR" : "HYBRID",
               wall_end - wall_start, seq_len, d_model, num_heads, size);
        profiler_print_summary(rank);   /* IO / compute / comm split for rank 0 */
    }

    /* --- Correctness check --- */
    if (do_check && rank == 0 && mode != MODE_SEQ) {
        float err = max_abs_error(&out_seq, &out_par);
        if (err < 1e-3f)
            printf("CORRECTNESS PASS  max_err=%.2e\n", err);
        else
            printf("CORRECTNESS FAIL  max_err=%.2e  <-- INVESTIGATE\n", err);
    }

done:
    tensor_free(&Q); tensor_free(&K); tensor_free(&V);
    tensor_free(&out_seq); tensor_free(&out_par);
    MPI_Finalize();
    return 0;
}
