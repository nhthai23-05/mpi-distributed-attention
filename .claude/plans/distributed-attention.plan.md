# Plan: Distributed Multi-Head Attention — Hybrid MPI Parallelism

**Source PRD**: `.claude/prds/distributed-attention.prd.md`
**Selected Milestone**: All 5 milestones (2-day sprint, blank repo)
**Complexity**: Large

---

## Summary

Build a single C binary from scratch that implements Multi-Head Attention in
three parallelism modes (head, tensor, hybrid) using OpenMPI. Day 1 produces
working, correct code across all three modes. Day 2 produces benchmark data
for the three required charts and a complete 10–20 page report.

---

## Patterns to Mirror

No existing code — patterns are established by this plan.

| Category | Source | Pattern |
|---|---|---|
| Naming | new | `snake_case` for all C identifiers; files match module name (`tensor.c`, `attention.c`) |
| Error handling | new | All MPI calls wrapped in `MPI_CHECK(call)` macro that prints rank + line and calls `MPI_Abort` |
| Profiling | new | `timer_start/stop(TIMER_IO\|COMPUTE\|COMM)` API backed by `MPI_Wtime()` arrays indexed by rank |
| Tests | new | Sequential vs. parallel diff in `verify.sh`; uses `mpirun -np 1` as the ground-truth baseline |

---

## Project File Map

```
mpi-distributed-attention/
├── Makefile
├── hostfile                    ← one hostname per line, e.g. node0 slots=4
├── src/
│   ├── main.c                  ← CLI parsing + mode dispatch
│   ├── tensor.h / tensor.c     ← Tensor struct, alloc, matmul, transpose, softmax
│   ├── data_gen.h / data_gen.c ← Deterministic synthetic Q/K/V generator
│   ├── profiler.h / profiler.c ← MPI_Wtime timers; print_timings_csv()
│   ├── attention.h / attention.c ← Sequential baseline (called when np=1 or --seq)
│   ├── head_parallel.h / head_parallel.c ← M1: Head Parallelism
│   ├── tensor_parallel.h / tensor_parallel.c ← M2: Cannon + distributed softmax
│   └── hybrid.h / hybrid.c     ← M3: MPI_Comm_split + sub-comm dispatch
└── scripts/
    ├── bench_n_sizing.sh        ← Sweep N, record wall time
    ├── bench_granularity.sh     ← Per-process compute/comm breakdown at N
    └── bench_speedup.sh         ← Vary process count at 2N
```

---

## Files to Change

| File | Action | Why |
|---|---|---|
| `Makefile` | CREATE | Build all `.c` into one binary `hybrid_attention` |
| `src/tensor.h/c` | CREATE | Shared data structure and math primitives |
| `src/data_gen.h/c` | CREATE | Deterministic synthetic input so all ranks start with same data |
| `src/profiler.h/c` | CREATE | MPI_Wtime wrappers; CSV output for chart scripts |
| `src/attention.h/c` | CREATE | Sequential single-process baseline for correctness gate |
| `src/head_parallel.h/c` | CREATE | M1 implementation |
| `src/tensor_parallel.h/c` | CREATE | M2 implementation |
| `src/hybrid.h/c` | CREATE | M3 implementation |
| `src/main.c` | CREATE | Entry point, mode dispatch, correctness check |
| `scripts/bench_*.sh` | CREATE | Automate the three benchmark experiments |
| `hostfile` | CREATE | MPI process placement |

---

## Key Design Decisions

### Fixed parameters (fill in before first compile)

```c
#define SEQ_LEN   N         // swept in benchmarks; start with 512 for smoke test
#define NUM_HEADS 4         // must equal number of M1 groups
#define D_MODEL   512
#define D_K       (D_MODEL / NUM_HEADS)   // 128
#define D_V       D_K
#define DTYPE     float     // float32; ε = 1e-4 for correctness check
```

### M1 — Head Parallelism (task parallelism)

- `rank r` owns heads `[r * heads_per_rank, (r+1) * heads_per_rank)`.
- Root scatters Q, K, V slices per head → each rank runs full sequential attention
  on its heads → root gathers all head outputs → concatenate.
- MPI calls: `MPI_Scatter` (Q/K/V slices), `MPI_Gather` (output slices).
- Communication topology: **master–worker** (flat, rank 0 is master).

```
Rank 0  → head 0, head 1        (if NUM_HEADS=4, P=4: 1 head per rank)
Rank 1  → head 1
Rank 2  → head 2
Rank 3  → head 3
```

### M2 — Tensor Parallelism (data parallelism, Cannon's Algorithm)

- 4 processes arranged in a 2×2 Cartesian grid.
- Compute `S = Q × K^T / sqrt(d_k)` where Q and K^T are distributed in 2D blocks.
- **Cannon initial alignment**: shift A (Q blocks) left by row index, shift B (K^T
  blocks) up by column index using `MPI_Sendrecv_replace`.
- **Cannon main loop** (√P = 2 steps): local matmul accumulate, shift A left 1,
  shift B up 1.
- **Distributed softmax** on S: `MPI_Allreduce(MAX)` for numerical stability,
  then `MPI_Allreduce(SUM)` for normalisation denominator.
- Final `O = softmax(S) × V`: repeat Cannon for this second matmul.
- Communication topology: **ring** within Cartesian sub-communicator.

### M3 — Hybrid Integration

- `MPI_Comm_split` splits 4 ranks into 2 sub-communicators of 2 ranks each.
  - Sub-comm 0: ranks {0, 1} → handles heads {0, 1}
  - Sub-comm 1: ranks {2, 3} → handles heads {2, 3}
- Within each sub-comm, M2-style 1×2 Tensor Parallelism on the assigned heads.
- After sub-comm computation, `MPI_Gather` into `MPI_COMM_WORLD` rank 0
  to concatenate all head outputs.

> **Critical**: NEVER pass `MPI_COMM_WORLD` to `MPI_Allreduce` inside sub-comm
> operations. Every collective inside a sub-comm uses only `sub_comm`.

---

## Tasks

### Task 0 — Resolve blockers before writing code (≤ 30 min)

- **Action**: On all 4 nodes, run `nproc` and record physical core count. Agree
  on `SEQ_LEN` starting value (suggest 512 for smoke test). Confirm float32.
- **Validate**: All 4 nodes respond to `ssh nodeX nproc`. Team has agreed on
  `D_MODEL`, `NUM_HEADS`, `D_K`.

---

### Task 1 — Makefile + tensor + data_gen (≤ 1 h)

- **Action**: Write `Makefile` with `mpicc -O2 -Wall -std=c11 -lm`. Implement
  `Tensor` struct with `alloc_tensor`, `free_tensor`, `matmul`, `transpose`,
  `softmax_rows` (sequential, numerically stable via row-max subtraction).
  Write `data_gen.c` using `srand(42)` so all ranks generate identical data.
- **Validate**: `make && mpirun -np 1 ./hybrid_attention --test-tensor` prints
  "TENSOR OK".

---

### Task 2 — Sequential baseline + correctness harness (≤ 1 h)

- **Action**: Implement `attention_seq(Q, K, V, out, seq_len, d_k)` in
  `attention.c`. In `main.c`, when `rank == 0 || np == 1`, run seq baseline and
  store result. After any parallel mode, broadcast parallel result to rank 0 and
  compare element-wise with `max_err < 1e-4f`.
- **Validate**: `mpirun -np 1 ./hybrid_attention --seq` prints result.
  `mpirun -np 4 --mode head` prints "CORRECTNESS PASS" or shows max error.

---

### Task 3 — M1: Head Parallelism (≤ 2 h)

- **Action**: In `head_parallel.c` implement:
  1. Root splits Q/K/V into `NUM_HEADS` head-slices (shape `[seq_len, d_k]` each).
  2. `MPI_Scatter` one head per rank (for `np == NUM_HEADS`; handle uneven case
     generically).
  3. Each rank calls `attention_seq` on its head slice.
  4. `MPI_Gather` head output slices back to root.
  5. Root concatenates along the head dimension.
- **Validate**:
  ```bash
  mpirun -np 4 ./hybrid_attention --mode head --seq-len 512
  # Expected: "CORRECTNESS PASS  max_err=..."
  ```

---

### Task 4 — M2: Tensor Parallelism — Cannon + Distributed Softmax (≤ 3 h)

This is the hardest task. Implement in `tensor_parallel.c`:

**Step 4a — Cartesian topology**
```c
int dims[2] = {2, 2};
int periods[2] = {1, 1};   // wrap-around for ring shifts
MPI_Cart_create(comm, 2, dims, periods, 0, &cart_comm);
MPI_Cart_coords(cart_comm, rank, 2, coords);
```

**Step 4b — Cannon alignment**
```c
// Shift A blocks left by coords[0], B blocks up by coords[1]
// Use MPI_Cart_shift to get left/right and up/down neighbours
MPI_Sendrecv_replace(A_block, block_size, MPI_FLOAT,
                     left_rank, 0, right_rank, 0, cart_comm, &status);
MPI_Sendrecv_replace(B_block, block_size, MPI_FLOAT,
                     up_rank, 1, down_rank, 1, cart_comm, &status);
```

**Step 4c — Cannon main loop (√P = 2 iterations)**
```c
for (int step = 0; step < sqrt_p; step++) {
    local_matmul_accumulate(C_block, A_block, B_block);
    shift_A_left_one(A_block, cart_comm, left_rank, right_rank);
    shift_B_up_one(B_block, cart_comm, up_rank, down_rank);
}
```

**Step 4d — Distributed Softmax (numerically stable)**
```c
float local_max = row_max(S_block);
float global_max;
MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, sub_comm);
subtract_scalar(S_block, global_max);
exp_inplace(S_block);
float local_sum = row_sum(S_block);
float global_sum;
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, sub_comm);
divide_scalar(S_block, global_sum);
```

**Step 4e — Second Cannon pass for `softmax(S) × V`**

- **Validate**:
  ```bash
  mpirun -np 4 ./hybrid_attention --mode tensor --seq-len 512
  # Expected: "CORRECTNESS PASS  max_err=..."
  ```

---

### Task 5 — M3: Hybrid Integration (≤ 2 h)

- **Action**: In `hybrid.c`:
  1. `MPI_Comm_split(MPI_COMM_WORLD, rank / 2, rank, &sub_comm)` — ranks {0,1}
     → color 0, ranks {2,3} → color 1.
  2. Each sub-comm computes M2 tensor-parallel attention on its assigned heads
     (sub-comm rank 0 = head 0 assignment, sub-comm rank 1 = head 1 assignment
     within the group).
  3. After sub-comm finishes, `MPI_Gather` partial results into rank 0 of
     `MPI_COMM_WORLD`.
  4. Rank 0 concatenates head outputs.
- **Critical guard**: Every `MPI_Allreduce` inside `tensor_parallel.c` must
  accept a `comm` argument — never hardcode `MPI_COMM_WORLD`.
- **Validate**:
  ```bash
  mpirun -np 4 ./hybrid_attention --mode hybrid --seq-len 512
  # Expected: "CORRECTNESS PASS  max_err=..."
  ```

---

### Task 6 — Profiler + benchmark scripts (≤ 1 h)

- **Action**: In `profiler.c`, maintain per-rank arrays:
  `t_io[rank]`, `t_compute[rank]`, `t_comm[rank]`. Use `MPI_Wtime()` delimiters.
  Add `--csv` flag to `main.c` that prints `rank,seq_len,t_io,t_compute,t_comm`
  to stdout. Write three shell scripts:

  `bench_n_sizing.sh`:
  ```bash
  for N in 128 256 512 1024 2048 4096 8192; do
      mpirun -np $TOTAL_PROCS ./hybrid_attention --mode hybrid --seq-len $N --csv
  done
  ```

  `bench_granularity.sh`:
  ```bash
  mpirun -np $TOTAL_PROCS ./hybrid_attention --mode hybrid --seq-len $N --csv
  # Produces one row per rank; check if any two rows differ by > 25% in total time
  ```

  `bench_speedup.sh`:
  ```bash
  for P in 1 2 4 8 ...; do
      mpirun -np $P ./hybrid_attention --mode hybrid --seq-len $((2*N)) --csv
  done
  ```

- **Validate**: `bash scripts/bench_n_sizing.sh > results/n_sizing.csv` produces
  parseable CSV.

---

## Day-by-Day Schedule

| Time slot | Work |
|---|---|
| Day 1, h 0–0.5 | Task 0 (resolve blockers, agree dims) |
| Day 1, h 0.5–1.5 | Task 1 (Makefile, tensor, data_gen) |
| Day 1, h 1.5–2.5 | Task 2 (sequential baseline + correctness harness) |
| Day 1, h 2.5–4.5 | Task 3 (M1 Head Parallelism) |
| Day 1, h 4.5–7.5 | Task 4 (M2 Tensor Parallelism — allow most time here) |
| Day 1, h 7.5–9.5 | Task 5 (M3 Hybrid) |
| Day 1, h 9.5–10.5 | Task 6 (Profiler + bench scripts) |
| Day 2, h 0–1 | Run N-sizing experiment; pick N (target: ~2–3 min runtime) |
| Day 2, h 1–2 | Run granularity experiment; fix if load imbalance > 25% |
| Day 2, h 2–3 | Run speedup experiment |
| Day 2, h 3–12 | Write report (10–20 pages) |

---

## Validation

```bash
# Smoke test all three modes
mpirun -np 4 ./hybrid_attention --mode head   --seq-len 512
mpirun -np 4 ./hybrid_attention --mode tensor --seq-len 512
mpirun -np 4 ./hybrid_attention --mode hybrid --seq-len 512

# Each must print: CORRECTNESS PASS  max_err=<value < 1e-4>

# Benchmark (fill N and TOTAL_PROCS after Task 0)
bash scripts/bench_n_sizing.sh  > results/n_sizing.csv
bash scripts/bench_granularity.sh > results/granularity.csv
bash scripts/bench_speedup.sh   > results/speedup.csv
```

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cannon needs √P × √P = perfect square | Medium | With P=4, √P=2 ✓. For P=2 fall back to 1D strip partition. Add `assert(sqrt_p * sqrt_p == P)` at startup. |
| Wrong `comm` in `MPI_Allreduce` at M3 | High | Make every collective in `tensor_parallel.c` accept `comm` as parameter. Never use `MPI_COMM_WORLD` inside. |
| Distributed softmax overflow (large N) | Medium | Always subtract global max before exp. Test with seq_len=8192. |
| Time runs out before M3 | Medium | M1+M2 alone satisfy most rubric criteria. If M3 is incomplete, document the architecture and show it in pseudo-code. |
| Load imbalance > 25% | Medium | Adjust `heads_per_rank` or block size. Have bench_granularity.sh flag violations automatically. |
| `nproc` unknown until Day 1 start | Low | Benchmark scripts parameterised by `$TOTAL_PROCS` env var; set it once cores are known. |

---

## Report Outline (10–20 pages)

1. **Introduction** (1 page) — problem statement, motivation
2. **Algorithm Description** (2 pages) — Attention formula, decomposition
3. **Parallelism Type + Decomposition Technique** (2 pages) — answers grader
   section 1 & 2: Hybrid (task + data); technique = data + task decomposition
4. **Parallelization Method** (3 pages) — answers grader section 3:
   - Mapping: 1D head slice (M1), 2D n/√p × n/√p block (M2), split (M3)
   - Communication: master–worker (M1), Cannon ring (M2), both (M3)
   - Load balancing: granularity chart analysis
   - Pseudo-code for each mode
5. **Results** (4–6 pages) — answers grader section 4:
   - Correctness verification (sequential == parallel diff table)
   - N-sizing chart (find N for 2–3 min)
   - Granularity chart (per-process compute vs. comm bars)
   - Speedup chart (time with/without comm + speedup curve)
6. **Discussion** (1 page) — bottlenecks, VLAN limitations, future work
7. **Conclusion** (0.5 page)

---

## Acceptance

- [x] All three modes compile without warnings (`-Wall -Wextra`) — verified clean build
- [x] All three modes pass `CORRECTNESS PASS` — verified at seq-len 256/512/1024, np 1/2/4/8
- [x] Cannon path (`--cannon`) passes — `max_err≈1.5e-7` (float rounding, expected)
- [x] No `MPI_COMM_WORLD` inside sub-communicator operations — hybrid passes `sub_comm` throughout
- [x] Cluster tooling matches instruction (SSH/NFS/OpenMPI/`mpi-prime`) — `test_cluster.sh`, `make mpi-prime`, 4-node hostfile
- [ ] `bench_n_sizing.csv` has ≥ 5 data points — *smoke-tested only; real run pending*
- [ ] `bench_granularity.csv` shows load balance ≤ 25% — *smoke-tested (9.6%); real N run pending*
- [ ] `bench_speedup.csv` covers ≥ 4 process counts — *smoke-tested; real run pending*
- [ ] Report is 10–20 pages with all four graded sections and three charts — *not started*

### Status summary (2026-06-22)
**DONE:** all code (M1/M2/M3 + Cannon), compiles clean, correctness verified, cluster tooling.
**REMAINING:** (1) run benchmarks for real chart data, (2) write the 10–20 page report.
