# ✅ What I Need To Do — Manual Checklist

This is the list of steps **you (human)** must do, in order. The code is already
written and verified — what remains is cluster setup, running benchmarks, and the
report.

Legend:  👤 = only you can do it (needs the physical machines / your password)
         🤖 = Claude can do it for you (just ask)

---

## PHASE A — Cluster setup (👤 on all 4 machines)

> Goal: 4 machines (`master`, `slave1`, `slave2`, `slave3`) that can talk to each
> other and run MPI. Do this once.

### Part 1 — Passwordless SSH keys
- [ ] **On `slave1`:**
  ```bash
  mv ~/.ssh/id_rsa_slave1 ~/.ssh/id_rsa
  mv ~/.ssh/id_rsa_slave1.pub ~/.ssh/id_rsa.pub
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/id_rsa*
  chmod 600 ~/.ssh/authorized_keys
  ```
- [ ] **On `slave2`:** same, with `id_rsa_slave2`
- [ ] **On `slave3`:** same, with `id_rsa_slave3`
- [ ] **On `master`:** same, with `id_rsa_master`
- [ ] **Map names → IPs:** edit `/etc/hosts` on **every** machine so `master`,
      `slave1`, `slave2`, `slave3` resolve to their VLAN/LAN IPs.
- [ ] **Test:** from `master`, `ssh slave1 hostname` must work with **no password**
      (repeat for slave2, slave3).

### Part 2 — NFS shared folder
- [ ] **On `master` (NFS server):**
  ```bash
  sudo apt install -y nfs-kernel-server
  mkdir -p ~/mpi_shared
  # add this line to /etc/exports (adjust username/path & IPs):
  #   /home/<user>/mpi_shared slave1(rw,sync,no_subtree_check) slave2(...) slave3(...)
  sudo exportfs -ra
  sudo systemctl restart nfs-kernel-server
  ```
- [ ] **On each `slaveN` (NFS client):**
  ```bash
  sudo apt install -y nfs-common
  mkdir -p ~/mpi_shared
  sudo mount master:/home/<user>/mpi_shared ~/mpi_shared
  ```
- [ ] **Test:** create a file in `~/mpi_shared` on `master` → it appears on all slaves.
- [ ] **Put this project inside `~/mpi_shared`** so the compiled binary is shared.

### Part 3 — OpenMPI (same version on every node)
- [ ] Install OpenMPI on **all 4 machines**. Simplest (works fine):
  ```bash
  sudo apt install -y build-essential openmpi-bin libopenmpi-dev
  ```
  *(Or build from source per the instruction — if you do, remove the apt
  packages first and re-export `$PATH` / `$LD_LIBRARY_PATH`.)*
- [ ] **Verify identical version everywhere:** `mpirun --version` on all 4 — the
      version number **must match**.

### Part 4 — Prove the cluster works
- [ ] Set `slots=` in [`hostfile`](hostfile): run `nproc` on each machine, put the
      core count for each node.
- [ ] Run the smoke test:
  ```bash
  bash scripts/test_cluster.sh
  ```
- [ ] ✋ **Do not continue until it prints `cluster OK`.** If it fails, the script
      tells you what to check (SSH → /etc/hosts → OpenMPI version → NFS).

---

## PHASE B — Build & run the project (👤 on `master`)

- [ ] `make` — builds `hybrid_attention` and `mpi-prime`
- [ ] Confirm correctness on the real cluster:
  ```bash
  mpirun --hostfile hostfile -np <total_cores> ./hybrid_attention --mode hybrid --seq-len 1024
  ```
  Expect: `CORRECTNESS PASS`.

---

## PHASE C — Benchmarks (🤖 Claude can run these, or 👤 you on the cluster)

> Chart data for the report. Best run on the real 4-node cluster (the network
> numbers only appear there). Set `TOTAL_PROCS` = your total core count.

- [ ] **Chart B (find N for ~2–3 min runtime):**
  ```bash
  TOTAL_PROCS=<cores> MODE=hybrid bash scripts/bench_n_sizing.sh
  ```
- [ ] **Chart C (load balance — use the N you found):**
  ```bash
  TOTAL_PROCS=<cores> N=<chosen_N> MODE=hybrid bash scripts/bench_granularity.sh
  ```
- [ ] **Chart D (speedup, 1→2X processes):**
  ```bash
  TOTAL_PROCS=<2x cores> N=<chosen_N> MODE=hybrid bash scripts/bench_speedup.sh
  ```
- [ ] Results land in `results/*.csv`. Hand these to Claude to make the charts.

---

## PHASE D — Report (🤖 Claude can draft, 👤 you review & submit)

- [ ] Ask Claude to draft the 10–20 page report (all four graded sections).
- [ ] Plug in the real benchmark charts from Phase C.
- [ ] Review, adjust wording, submit.

---

## Quick status

| Milestone | State |
|---|---|
| M1/M2/M3 code | ✅ done & verified |
| Cluster tooling | ✅ ready (`test_cluster.sh`, `make mpi-prime`, 4-node hostfile) |
| **Phase A** cluster setup | ⬜ **your turn** |
| **Phase C** benchmark data | ⬜ pending (Claude or you) |
| **Phase D** report | ⬜ pending (Claude drafts) |

> Full detail: requirements in [`.claude/prds/distributed-attention.prd.md`](.claude/prds/distributed-attention.prd.md),
> implementation plan in [`.claude/plans/distributed-attention.plan.md`](.claude/plans/distributed-attention.plan.md).
