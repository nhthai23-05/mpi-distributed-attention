#!/usr/bin/env python3
"""bench_scaling.py — sweep N and compare sequential vs head-parallel performance.

Demonstrates that communication overhead is amortised as N grows: for small N
the parallel speedup is low (MPI overhead dominates), but it rises toward the
ideal P× speedup as the O(N²) compute work dominates.

Usage
-----
  python3 scripts/bench_scaling.py [options]

  --binary PATH    path to the hybrid_attention binary  (default: ./hybrid_attention)
  --procs LIST     comma-separated process counts       (default: 1,2,4)
  --heads N        number of attention heads            (default: 4)
  --d-model N      model dimension                      (default: 512)
  --sizes LIST     comma-separated N (seq_len) values   (default: 64,128,256,512,1024,2048,4096)
  --repeats N      repetitions per (N, procs) pair      (default: 3; best-of is used)
  --results-dir D  output directory                     (default: results)
  --no-plot        skip matplotlib chart generation
  --hostfile PATH  MPI hostfile (default: none; omit for local single-machine runs)
  --oversubscribe  pass --oversubscribe to mpirun (needed on single machines)

Output
------
  results/scaling.csv
  results/chart-scaling.png
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary",      default="./hybrid_attention")
    ap.add_argument("--procs",       default="1,2,4",
                    help="comma-separated list of MPI process counts")
    ap.add_argument("--heads",       type=int, default=4)
    ap.add_argument("--d-model",     type=int, default=512, dest="d_model")
    ap.add_argument("--sizes",       default="64,128,256,512,1024,2048,4096",
                    help="comma-separated seq_len values to sweep")
    ap.add_argument("--repeats",     type=int, default=3,
                    help="runs per (N,P) pair; best (min) time is kept")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--no-plot",     action="store_true")
    ap.add_argument("--hostfile",    default=None,
                    help="MPI hostfile; omit for local single-machine runs")
    ap.add_argument("--oversubscribe", action="store_true", default=True,
                    help="pass --oversubscribe to mpirun (default: on)")
    ap.add_argument("--no-oversubscribe", action="store_false", dest="oversubscribe")
    return ap.parse_args()

# ── benchmark runner ──────────────────────────────────────────────────────────

def mpirun_cmd(args, np, binary_args):
    cmd = ["mpirun"]
    if args.oversubscribe:
        cmd += ["--oversubscribe"]
    cmd += ["-np", str(np)]
    if args.hostfile and os.path.exists(args.hostfile):
        cmd += ["--hostfile", args.hostfile]
    cmd += [args.binary] + binary_args
    return cmd


def run_seq(args, N):
    """Return best sequential time over `repeats` runs (seconds)."""
    best = float("inf")
    cmd = mpirun_cmd(args, 1, [
        "--mode", "seq",
        "--seq-len", str(N),
        "--heads",   str(args.heads),
        "--d-model", str(args.d_model),
    ])
    for _ in range(args.repeats):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                          timeout=300).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        m = re.search(r"\[SEQ\] time=([0-9.]+)s", out)
        if m:
            best = min(best, float(m.group(1)))
    return best if best < float("inf") else None


def run_parallel(args, N, np):
    """Return best head-parallel wall time over `repeats` runs (seconds)."""
    best = float("inf")
    # Clamp procs to num_heads: head mode assigns at least 1 head per rank
    if np > args.heads:
        return None
    cmd = mpirun_cmd(args, np, [
        "--mode",    "head",
        "--seq-len", str(N),
        "--heads",   str(args.heads),
        "--d-model", str(args.d_model),
        "--no-check",
    ])
    for _ in range(args.repeats):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                          timeout=300).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        m = re.search(r"\[HEAD\] wall=([0-9.]+)s", out)
        if m:
            best = min(best, float(m.group(1)))
    return best if best < float("inf") else None

# ── CSV helpers ───────────────────────────────────────────────────────────────

HEADER = ["seq_len", "procs", "mode", "time_s", "speedup"]

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

# ── plotting ──────────────────────────────────────────────────────────────────

PALETTE = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

def make_plot(rows, out_path, heads, d_model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not found — skipping chart (pip install matplotlib)")
        return

    # Organise by (procs, N)
    by_proc = defaultdict(dict)  # by_proc[procs][N] = time_s
    for r in rows:
        by_proc[r["procs"]][r["seq_len"]] = r["time_s"]

    all_procs = sorted(by_proc.keys())
    seq_times = by_proc.get(1, {})
    all_N     = sorted({r["seq_len"] for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: absolute runtime vs N ───────────────────────────────────────
    for idx, p in enumerate(all_procs):
        times = by_proc[p]
        Ns    = sorted(times)
        label = "sequential (P=1)" if p == 1 else f"head-parallel (P={p})"
        style = "-o" if p == 1 else ("--s" if p == 2 else ("-.^" if p == 4 else ":D"))
        ax1.plot(Ns, [times[n] for n in Ns], style,
                 label=label, color=PALETTE[idx], linewidth=2, markersize=6)

    # Theoretical ideal lines for each P > 1
    for idx, p in enumerate(all_procs):
        if p == 1:
            continue
        Ns_seq = sorted(seq_times)
        if not Ns_seq:
            continue
        ideal = [seq_times[n] / p for n in Ns_seq if n in by_proc[p]]
        Ns_both = [n for n in Ns_seq if n in by_proc[p]]
        if ideal:
            ax1.plot(Ns_both, ideal, ":", color=PALETTE[all_procs.index(p)],
                     alpha=0.4, label=f"ideal P={p} (T_seq/{p})")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Sequence length N", fontsize=12)
    ax1.set_ylabel("Wall time  (s)", fontsize=12)
    ax1.set_title(f"Runtime vs N\n(heads={heads}, d_model={d_model})", fontsize=12)
    ax1.set_xticks(all_N)
    ax1.set_xticklabels([str(n) for n in all_N], rotation=30, ha="right")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(fontsize=9)

    # ── Panel 2: speedup vs N ────────────────────────────────────────────────
    for idx, p in enumerate(all_procs):
        if p == 1:
            continue
        times  = by_proc[p]
        Ns     = sorted(n for n in times if n in seq_times and times[n] and seq_times[n])
        speeds = [seq_times[n] / times[n] for n in Ns]
        style  = "--s" if p == 2 else ("-.^" if p == 4 else ":D")
        ax2.plot(Ns, speeds, style,
                 label=f"P={p} speedup", color=PALETTE[idx], linewidth=2, markersize=6)
        # Ideal
        ax2.axhline(p, color=PALETTE[idx], linestyle=":", alpha=0.4,
                    label=f"ideal P={p}")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence length N", fontsize=12)
    ax2.set_ylabel("Speedup  T(1) / T(P)", fontsize=12)
    ax2.set_title(f"Speedup vs N\n(parallel efficiency rises with N)", fontsize=12)
    ax2.set_xticks(all_N)
    ax2.set_xticklabels([str(n) for n in all_N], rotation=30, ha="right")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle("Head-parallel MHA: parallel becomes better as N grows",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart written to {out_path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    sizes  = [int(x) for x in args.sizes.split(",")]
    procs  = sorted({int(x) for x in args.procs.split(",")})

    if 1 not in procs:
        procs = [1] + procs   # always include sequential baseline

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path  = os.path.join(args.results_dir, "scaling.csv")
    plot_path = os.path.join(args.results_dir, "chart-scaling.png")

    print(f"Binary  : {args.binary}")
    print(f"Sizes   : {sizes}")
    print(f"Procs   : {procs}")
    print(f"Heads   : {args.heads}   d_model={args.d_model}")
    print(f"Repeats : {args.repeats}  (best-of kept)")
    print()

    rows = []

    for N in sizes:
        # --- sequential baseline ---
        print(f"  N={N:5d}  P=1 (seq) ... ", end="", flush=True)
        t0 = time.time()
        t_seq = run_seq(args, N)
        elapsed = time.time() - t0
        if t_seq is None:
            print("FAILED")
        else:
            print(f"{t_seq:.4f}s  (wall {elapsed:.1f}s)")
            rows.append({"seq_len": N, "procs": 1, "mode": "seq",
                         "time_s": t_seq, "speedup": 1.0})

        # --- parallel runs ---
        for p in procs:
            if p == 1:
                continue
            if p > args.heads:
                print(f"  N={N:5d}  P={p} skipped (P > num_heads={args.heads})")
                continue
            print(f"  N={N:5d}  P={p} (head) ... ", end="", flush=True)
            t0 = time.time()
            t_par = run_parallel(args, N, p)
            elapsed = time.time() - t0
            if t_par is None:
                print("FAILED")
            else:
                speedup = (t_seq / t_par) if (t_seq and t_par) else float("nan")
                eff     = speedup / p * 100
                print(f"{t_par:.4f}s  speedup={speedup:.2f}x  eff={eff:.0f}%  (wall {elapsed:.1f}s)")
                rows.append({"seq_len": N, "procs": p, "mode": "head",
                             "time_s": t_par, "speedup": round(speedup, 4)})
        print()

    write_csv(csv_path, rows)
    print(f"Results written to {csv_path}")

    if not args.no_plot and rows:
        make_plot(rows, plot_path, args.heads, args.d_model)

    # Print summary table
    print()
    print("Summary (speedup = T_seq / T_head_par):")
    print(f"{'N':>6}  {'P':>4}  {'time':>8}  {'speedup':>8}  {'eff%':>6}")
    print("-" * 42)
    for r in rows:
        sp = f"{r['speedup']:.2f}x" if r['procs'] > 1 else "baseline"
        ef = f"{r['speedup']/r['procs']*100:.0f}%" if r['procs'] > 1 else ""
        print(f"{r['seq_len']:>6}  {r['procs']:>4}  {r['time_s']:>8.4f}s  {sp:>8}  {ef:>6}")


if __name__ == "__main__":
    main()
