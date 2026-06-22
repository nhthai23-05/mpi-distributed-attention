# Distributed Multi-Head Attention — Hybrid MPI Parallelism

## Problem

A solo engineer must design, implement, benchmark, and document a
distributed Multi-Head Attention system (pure C/C++ + OpenMPI) from a blank
repository in **≤ 2 days** (deadline: 2026-06-24). The deliverable is a graded
academic artifact: a 10–20 page report and working code. Without a clear scope
contract the team will over-engineer early stages and run out of time before
producing the benchmark data the grader explicitly requires.

## Evidence

- Graded rubric provided by instructor (see Grading Criteria section below).
- README already defines the 3-milestone architecture, so technical direction
  is settled; what's missing is scope discipline under a compressed timeline.

## Infrastructure

- **4-node cluster** named `master`, `slave1`, `slave2`, `slave3` (matches the
  `id_rsa_master` / `id_rsa_slaveN` SSH key convention).
- Passwordless SSH from `master` to all slaves; `/etc/hosts` maps the names.
- **NFS** shares the project dir (server = `master`, clients = slaves): compile
  once on `master`, binary visible everywhere. No data files are shared (input
  is generated in-memory).
- Same OpenMPI version required on every node.
- Cluster health is gated by `scripts/test_cluster.sh` (runs `mpi-prime`) before
  any project run.

## Users

- **Primary**: solo engineer — architects, implements, measures, and writes the
  report.
- **Evaluator**: Professor / TA who runs `mpirun`, checks correctness, and reads
  the report against the rubric.
- **Not for**: production inference, external researchers, anyone outside this
  course submission.

## Hypothesis

We believe **a working Hybrid Parallelism MPI implementation of Multi-Head
Attention with self-consistency verification and the three required benchmark
charts** will **satisfy all graded criteria** for **the course evaluation
team**. We'll know we're right when **the report is 10–20 pages, all three
charts are present, and the parallel output matches the sequential baseline on
at least one representative input size**.

## Grading Criteria (source of truth)

The report must cover four sections. Every implementation decision should be
traceable to at least one of these:

| Section | What the grader checks |
|---|---|
| **Parallelism type** | Task-level vs. data-level classification |
| **Decomposition technique** | Which of: data / exploratory / recursive / speculative / hybrid |
| **Parallelization method** | Mapping strategy (1D / 2D block); communication strategy (blocking or non-blocking, topology: tree / ring / torus / master-slave); load balancing; pseudo-code |
| **Results** | (A) correctness, (B) N-sizing chart, (C) granularity chart, (D) speedup chart |

### Result charts in detail

**(B) N-sizing chart** — processes fixed at total physical CPU cores across all
4 machines. X-axis: input size N. Y-axis: wall-clock time. Two lines: with
communication time and without. Target: find N such that runtime ≈ 2–3 minutes.

**(C) Granularity chart** — processes fixed as above, input size = N. One bar
per process; each bar split into compute time (one colour) vs. communication
time (another). Goal: verify load balance. If any two processes differ by > 25%
in total time → adjust granularity (finer or coarser) and re-run.

**(D) Speedup chart** — input size = 2N. Vary process count: 1, 2, 4, 8, …, 2X
(X = total physical cores). Two wall-clock lines (with/without comm) plus a
speedup curve.

## Success Metrics

| Metric | Target | How measured |
|---|---|---|
| Correctness | Max element-wise error < ε vs. sequential baseline | `assert` or diff in main after `MPI_Gather` |
| Report length | 10–20 pages | Page count before submission |
| N-sizing chart present | Yes | Chart in report |
| Granularity chart present | Yes | Chart in report |
| Speedup chart present | Yes | Chart in report |
| Load balance | No two processes differ > 25% | Granularity chart inspection |
| Speedup at max process count | > 1× (any improvement) | Speedup chart |
| Deadline met | 2026-06-24 | Submission timestamp |

> **Note on ε**: floating-point tolerance not yet specified. Use the same
> precision (float32 or float64) throughout; decide before first compile.

## Scope

### MVP (minimum to pass all four graded sections)

1. Single binary that accepts `--mode [head|tensor|hybrid]` and `--np <count>`.
2. M1 — Head Parallelism: `MPI_Scatter` heads to processes, local matmul +
   softmax, `MPI_Gather` results.
3. M2 — Tensor Parallelism: 4 processes co-compute one large matmul using block
   shifting (`MPI_Sendrecv_replace`) + distributed softmax via `MPI_Allreduce`.
4. M3 — Hybrid: `MPI_Comm_split` into two sub-communicators; each runs M2
   internally; results merged by M1 gather.
5. Sequential baseline in same binary (`--np 1` or rank 0 solo path) for
   self-consistency check.
6. `MPI_Wtime()` instrumentation: separate timers for I/O, compute, and
   communication at every stage.
7. Benchmark runner script: automates N-sizing, granularity, and speedup runs
   and dumps raw `.csv` for charting.
8. 10–20 page report covering all four graded sections.

### Decomposition framing for the report

**Hybrid (task + data)**:
- M1 = **task parallelism** — each head is an independent task assigned to a
  process group.
- M2 = **data parallelism** — rows/blocks of Q, K, V distributed across
  processes within a sub-communicator.
- M3 = both combined.

### Communication strategy (to document in report)

- **Topology**: flat ring within each sub-communicator for block shifting;
  master–worker for final gather across sub-communicators.
- **Blocking vs. non-blocking**: `MPI_Sendrecv_replace` (blocking) for Cannon
  block shifts; `MPI_Allreduce` (collective) for distributed softmax.
- **Mapping**: 2D n/√p × n/√p block decomposition for Tensor Parallelism within
  each sub-communicator; 1D slice per head for Head Parallelism.

### Out of scope

- GPU / CUDA acceleration — not required by rubric; risks 2-day timeline.
- PyTorch numerical baseline — self-consistency test is sufficient per decision.
- Physical LAN migration (Stage 4 in README) — benchmarks will run on VLAN;
  physical LAN is a post-submission stretch goal.
- Dynamic load balancing at runtime — static assignment is sufficient if
  granularity chart passes the 25% rule.
- File I/O from `.bin` files for benchmark — generate synthetic tensors in-memory
  to avoid I/O bottleneck skewing charts. Binary I/O can be added after charts
  are done.

## Delivery Milestones

| # | Milestone | Outcome | Status | Plan |
|---|---|---|---|---|
| 1 | Head Parallelism (M1) | `mpirun -np 4` runs; sequential == distributed output | complete | `.claude/plans/distributed-attention.plan.md` |
| 2 | Tensor Parallelism (M2) | Block-shift matmul + distributed softmax pass correctness check | complete | `.claude/plans/distributed-attention.plan.md` |
| 3 | Hybrid Integration (M3) | `MPI_Comm_split` runs; M2 isolated per sub-comm; correctness passes | complete | `.claude/plans/distributed-attention.plan.md` |
| 4 | Benchmark data collected | All three charts have raw data; N found for 2–3 min runtime | pending | `.claude/plans/distributed-attention.plan.md` |
| 5 | Report complete | 10–20 pages, all four sections, all three charts embedded | pending | `.claude/plans/distributed-attention.plan.md` |

> **Timeline reality**: 2 days for all 5 milestones from a blank repo.
> Recommended split: Day 1 = M1 + M2 + M3 code. Day 2 = benchmarks + report.

## Open Questions

- [ ] **Model dimensions** (d_model, num_heads, seq_len, d_k) — must be fixed
      before first compile; num_heads must be divisible by the number of process
      groups used in M1.
- [ ] **CPU core count per machine** — run `nproc` on `master`/`slave1`/`slave2`/`slave3` before any
      benchmark; total cores = X in speedup chart.
- [ ] **Floating-point precision** — float32 or float64? Affects both performance
      and correctness tolerance ε.
- [ ] **Self-consistency tolerance ε** — what max error between sequential and
      parallel outputs is acceptable? (Suggested: < 1e-4 for float32.)
- [ ] **Synthetic data generation** — who writes the tensor generator? Needed
      before M1 test can run.
- [ ] **Report language** — Vietnamese or English?

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 2-day timeline with blank repo | High | Critical | Daily scope cut: ship M1+M2+M3 code by end of Day 1; no feature work on Day 2 |
| Silent mathematical error at M3 (wrong `comm` in `MPI_Allreduce`) | High | High | Run correctness check after every milestone commit before moving to next |
| Distributed Softmax overflow (large N) | Medium | High | Use log-sum-exp trick; implement before M2 benchmark runs |
| Unknown core count delays benchmark planning | Medium | Medium | Run `nproc` in first 30 minutes of Day 1 |
| Load balance fails 25% rule on first run | Medium | Medium | Adjust block size; re-run; build benchmark script on Day 1 for fast iteration |
| VLAN latency makes benchmarks unrepresentative | Low | Low | Document in report as known limitation; physical LAN deferred |

---
*Status: DRAFT — requirements only. Implementation planning pending via `/plan`.*
