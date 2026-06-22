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

**Tensor mode has two paths:**
- **Default (1D):** row-block decomposition + local softmax. Works for any
  process count; this is what `hybrid` uses internally.
- **Cannon (`--cannon`):** 2D block matrix multiply using `MPI_Sendrecv_replace`
  ring shifts + distributed softmax via `MPI_Allreduce` over a row
  sub-communicator. Requires a perfect-square process count and `seq_len`,
  `d_k` divisible by √P.

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
| `--csv` | emit per-rank timing as CSV | off |
| `--no-check` | skip the correctness check | off |

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
TOTAL_PROCS=<2×cores> N=<chosen_N> MODE=hybrid bash scripts/bench_speedup.sh
```

Each run separates **computation time** from **communication time** via
`MPI_Wtime()`, so the charts can show both.

---

## Cluster setup

Multi-node execution requires SSH keys, NFS, OpenMPI, and a cluster smoke-test.
The full step-by-step (with exact commands per machine) is in
**[CHECKLIST.md](CHECKLIST.md)**. Always run `bash scripts/test_cluster.sh` and
wait for `cluster OK` before running the project across nodes.
