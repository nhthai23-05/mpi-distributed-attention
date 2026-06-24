# Distributed Multi-Head Attention (Hybrid MPI Parallelism)

A from-scratch implementation of the Transformer **Multi-Head Attention**
mechanism in pure **C + OpenMPI**, parallelized across a cluster using
**Hybrid Parallelism** — combining *Head Parallelism* (task decomposition) and
*Tensor Parallelism* (data decomposition).

The core computation:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

No deep-learning frameworks are used — all linear algebra, the numerically
stable softmax, and the distributed communication are implemented directly with
MPI primitives.

---

## Parallelism modes

The program runs in three modes, selectable at the command line. Each is
verified against a single-process sequential baseline.

| Mode | Strategy | Decomposition | Key MPI calls |
|---|---|---|---|
| `head` (M1) | One (or more) attention head per process | **Task** | `MPI_Scatterv`, `MPI_Gatherv` |
| `tensor` (M2) | One head split across all processes | **Data** | `MPI_Bcast` / `MPI_Allreduce`; `MPI_Sendrecv_replace` (Cannon) |
| `hybrid` (M3) | Processes split into groups; each group runs M2 on its heads | **Task + Data** | `MPI_Comm_split` + the above |

**Tensor mode has three paths:**
- **Default (1D):** row-block decomposition + local softmax. Broadcasts the full
  K,V to every rank. Works for any process count; this is what `hybrid` uses
  internally. Bitwise-identical to the sequential baseline.
- **Cannon (`--cannon`):** 2D block matrix multiply using `MPI_Sendrecv_replace`
  ring shifts + distributed softmax via `MPI_Allreduce` over a row
  sub-communicator. Requires a perfect-square process count and `seq_len`,
  `d_k` divisible by √P.
- **Ring (`--ring`):** Flash-style attention — K,V are **sharded** (N/P rows per
  rank, not replicated) and streamed around a ring with non-blocking
  `MPI_Isend`/`MPI_Irecv` double buffering, folded into a running **online
  softmax**. Each rank holds only O(N/P) of K,V and the shift overlaps the
  compute. Works for any process count; matches the baseline to ~1e-7.

---

## Project structure

```
.
├── Makefile                 # builds hybrid_attention + mpi-prime
├── hostfile                 # 4-node cluster definition (master/slave1..3)
├── mpi-prime.c              # cluster smoke-test (3rd-party, Burkardt)
├── CHECKLIST.md             # manual setup / run / report checklist
├── src/
│   ├── tensor.{c,h}         # matrix ops, numerically stable softmax
│   ├── attention.{c,h}      # sequential baseline (ground truth)
│   ├── data_gen.{c,h}       # deterministic in-memory input generation
│   ├── profiler.{c,h}       # MPI_Wtime timers: I/O vs compute vs comm
│   ├── head_parallel.{c,h}  # M1 — Head Parallelism
│   ├── tensor_parallel.{c,h}# M2 — Tensor Parallelism (1D + Cannon)
│   ├── hybrid.{c,h}         # M3 — Hybrid (MPI_Comm_split)
│   └── main.c               # CLI, dispatch, correctness check
├── scripts/
│   ├── test_cluster.sh      # Part 4: prove the cluster works (mpi-prime)
│   ├── bench_n_sizing.sh    # Chart B: runtime vs input size N
│   ├── bench_granularity.sh # Chart C: per-process compute/comm balance
│   └── bench_speedup.sh     # Chart D: speedup vs process count
└── results/                 # benchmark CSV output
```

---

## Requirements

- A C compiler and OpenMPI (`mpicc`, `mpirun`) — **same version on every node**.
  ```bash
  sudo apt install -y build-essential openmpi-bin libopenmpi-dev
  ```
- For multi-node runs: passwordless SSH + NFS across the cluster
  (see [CHECKLIST.md](CHECKLIST.md)).

---

## Build

```bash
make            # builds ./hybrid_attention and ./mpi-prime
make clean      # removes binaries and object files
```

Built with **OpenMP** (`-fopenmp`): each MPI rank threads the per-query-row loop.
Control the per-rank thread count with `OMP_NUM_THREADS` (e.g. `OMP_NUM_THREADS=4`)
for a hybrid MPI+OpenMP run. The MPI-scaling benchmark scripts pin it to `1` by
default so their numbers are pure MPI scaling.

---

## Run

```bash
# General form
mpirun --hostfile hostfile -np <N> ./hybrid_attention --mode <head|tensor|hybrid> [options]
```

Common options:

| Flag | Meaning | Default |
|---|---|---|
| `--mode <m>` | `seq`, `head`, `tensor`, or `hybrid` | `seq` |
| `--seq-len <N>` | sequence length (input size) | 512 |
| `--heads <H>` | number of attention heads | 4 |
| `--d-model <D>` | model dimension (`d_k = D / H`) | 512 |
| `--groups <G>` | sub-comm groups for hybrid mode | 2 |
| `--cannon` | tensor mode: use Cannon's algorithm | off |
| `--ring` | tensor mode: ring (Flash-style) attention — sharded K,V + overlapped streaming softmax | off |
| `--csv` | emit per-rank CSV: `rank,seq_len,t_io,t_compute,t_comm,t_wait,msgs,bytes,latency_us` | off |
| `--no-check` | skip the correctness check | off |
| `--progress` | print rank-0 progress + ETA to stderr | off |
| `--profile-wait` | bracket barriers so idle WAIT time is split from real transfer | off |

Examples:

```bash
# Local quick test (4 ranks on this machine)
mpirun --oversubscribe -np 4 ./hybrid_attention --mode hybrid --seq-len 512

# Tensor mode with Cannon's algorithm (perfect-square procs)
mpirun --oversubscribe -np 4 ./hybrid_attention --mode tensor --cannon --seq-len 1024

# On the cluster
mpirun --hostfile hostfile -np 16 ./hybrid_attention --mode hybrid --seq-len 2048
```

---

## Correctness

Every parallel run compares its output against the sequential baseline and
prints:

```
CORRECTNESS PASS  max_err=0.00e+00
```

The 1D paths match the baseline exactly; the Cannon path shows a tiny
floating-point difference (`~1e-7`) due to a different summation order, which is
expected and well within tolerance.

---

## Benchmarks

Set `TOTAL_PROCS` to your total physical core count. Output lands in `results/`.

```bash
# Chart B — find N where runtime ≈ 2–3 min
TOTAL_PROCS=<cores> MODE=hybrid bash scripts/bench_n_sizing.sh

# Chart C — per-process compute vs communication (load balance, ≤25% target)
TOTAL_PROCS=<cores> N=<chosen_N> MODE=hybrid bash scripts/bench_granularity.sh

# Chart D — speedup as processes scale 1 → 2×cores
# (tensor mode: same kernel at every P, including the P=1 anchor — hybrid can't run at P=1)
TOTAL_PROCS=<2×cores> N=<chosen_N> MODE=tensor bash scripts/bench_speedup.sh

# Chart E — ring vs 1D tensor path (algorithm comparison: memory + overlap)
TOTAL_PROCS=<cores> N=<chosen_N> bash scripts/bench_ring_vs_1d.sh

# Chart F — OpenMP thread scaling (1 MPI rank, vary threads)
MAX_THREADS=<cores> N=<chosen_N> bash scripts/bench_threads.sh
```

Each run separates **computation**, **communication** and idle **wait** time via
`MPI_Wtime()` and counts messages, bytes, and link latency, so the charts can show
all of them. Turn the CSVs into figures with `python3 scripts/make_charts.py`.

---

## Cluster setup

Multi-node execution requires SSH keys, NFS, OpenMPI, and a cluster smoke-test.
The full step-by-step (with exact commands per machine) is in
**[CHECKLIST.md](CHECKLIST.md)**. Always run `bash scripts/test_cluster.sh` and
wait for `cluster OK` before running the project across nodes.
