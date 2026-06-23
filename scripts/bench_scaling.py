#!/usr/bin/env python3
"""bench_scaling.py — sweep N and compare sequential vs parallel performance.

Demonstrates that communication overhead is amortised as N grows: for small N
the parallel speedup is low (MPI overhead dominates), but it rises toward the
ideal P× speedup as the O(N²) compute work dominates.

Mode selection per process count P:
  P == 1            → sequential (--mode seq)
  1 < P <= heads    → head-parallel (--mode head)
  P > heads         → hybrid (--mode hybrid, needs --groups)

Usage
-----
  python3 scripts/bench_scaling.py [options]

  --binary PATH    path to the hybrid_attention binary  (default: ./hybrid_attention)
  --procs LIST     comma-separated process counts       (default: 1,2,4)
  --heads N        number of attention heads            (default: 4)
  --groups N       head groups for hybrid mode          (default: same as --heads)
                   P must be divisible by groups; heads must be divisible by groups
  --d-model N      model dimension                      (default: 512)
  --sizes LIST     comma-separated N (seq_len) values   (default: 64,128,256,512,1024,2048,4096)
  --repeats N      repetitions per (N, procs) pair      (default: 3; best-of is used)
  --results-dir D  output directory                     (default: results)
  --no-plot        skip matplotlib chart generation
  --hostfile PATH  MPI hostfile (default: none; omit for local single-machine runs)
  --oversubscribe  pass --oversubscribe to mpirun (needed on single machines)

Cluster example (60 procs, 4 heads, 4 groups → 15 procs per head via tensor-par):
  python3 scripts/bench_scaling.py \\
    --procs 1,4,8,16,32,60 --heads 4 --groups 4 --d-model 512 \\
    --sizes 256,512,1024,2048,4096,8192 \\
    --hostfile hostfile --no-oversubscribe

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
    ap.add_argument("--groups",      type=int, default=None,
                    help="groups for hybrid mode (default: same as --heads)")
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
    """Return best head-parallel wall time over `repeats` runs (seconds).
    Returns (time, mode_label) or (None, None) if not applicable."""
    best = float("inf")

    # Choose mode: head when P fits within num_heads, hybrid for larger P
    if np <= args.heads:
        mode = "head"
        extra = []
        pattern = r"\[HEAD\] wall=([0-9.]+)s"
    else:
        groups = args.groups if args.groups is not None else args.heads
        # Validate hybrid constraints
        if np % groups != 0 or args.heads % groups != 0:
            return None, None
        mode = "hybrid"
        extra = ["--groups", str(groups)]
        pattern = r"\[HYBRID\] wall=([0-9.]+)s"

    cmd = mpirun_cmd(args, np, [
        "--mode",    mode,
        "--seq-len", str(N),
        "--heads",   str(args.heads),
        "--d-model", str(args.d_model),
        "--no-check",
    ] + extra)

    for _ in range(args.repeats):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                          timeout=300).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        m = re.search(pattern, out)
        if m:
            best = min(best, float(m.group(1)))
    return (best, mode) if best < float("inf") else (None, None)

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
    by_proc = defaultdict(dict)   # by_proc[procs][N] = time_s
    mode_of = {}                  # mode_of[procs] = "seq"|"head"|"hybrid"
    for r in rows:
        by_proc[r["procs"]][r["seq_len"]] = r["time_s"]
        mode_of[r["procs"]] = r["mode"]

    all_procs = sorted(by_proc.keys())
    seq_times = by_proc.get(1, {})
    all_N     = sorted({r["seq_len"] for r in rows})

    STYLES = ["-o", "--s", "-.^", ":D", "-v", "--P", "-.h"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: absolute runtime vs N ───────────────────────────────────────
    for idx, p in enumerate(all_procs):
        times = by_proc[p]
        Ns    = sorted(times)
        m     = mode_of.get(p, "")
        if p == 1:
            label = "sequential (P=1)"
        elif m == "hybrid":
            label = f"hybrid (P={p})"
        else:
            label = f"head-parallel (P={p})"
        style = STYLES[idx % len(STYLES)]
        ax1.plot(Ns, [times[n] for n in Ns], style,
                 label=label, color=PALETTE[idx % len(PALETTE)],
                 linewidth=2, markersize=6)

    # Theoretical ideal lines for each P > 1
    for idx, p in enumerate(all_procs):
        if p == 1:
            continue
        Ns_seq = sorted(seq_times)
        if not Ns_seq:
            continue
        ideal    = [seq_times[n] / p for n in Ns_seq if n in by_proc[p]]
        Ns_both  = [n for n in Ns_seq if n in by_proc[p]]
        if ideal:
            ax1.plot(Ns_both, ideal, ":", color=PALETTE[idx % len(PALETTE)],
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
        m      = mode_of.get(p, "")
        Ns     = sorted(n for n in times if n in seq_times and times[n] and seq_times[n])
        speeds = [seq_times[n] / times[n] for n in Ns]
        slabel = f"hybrid P={p}" if m == "hybrid" else f"head P={p}"
        style  = STYLES[idx % len(STYLES)]
        ax2.plot(Ns, speeds, style,
                 label=slabel, color=PALETTE[idx % len(PALETTE)],
                 linewidth=2, markersize=6)
        ax2.axhline(p, color=PALETTE[idx % len(PALETTE)], linestyle=":", alpha=0.4,
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
            groups = args.groups if args.groups is not None else args.heads
            # Validate before running so we print a clear skip reason
            if p > args.heads:
                if p % groups != 0 or args.heads % groups != 0:
                    print(f"  N={N:5d}  P={p} skipped "
                          f"(hybrid needs P%groups==0 and heads%groups==0; "
                          f"P={p} groups={groups} heads={args.heads})")
                    continue
                mode_label = f"hybrid groups={groups}"
            else:
                mode_label = "head"

            print(f"  N={N:5d}  P={p} ({mode_label}) ... ", end="", flush=True)
            t0 = time.time()
            t_par, mode_used = run_parallel(args, N, p)
            elapsed = time.time() - t0
            if t_par is None:
                print("FAILED")
            else:
                speedup = (t_seq / t_par) if (t_seq and t_par) else float("nan")
                eff     = speedup / p * 100
                print(f"{t_par:.4f}s  speedup={speedup:.2f}x  eff={eff:.0f}%  (wall {elapsed:.1f}s)")
                rows.append({"seq_len": N, "procs": p, "mode": mode_used,
                             "time_s": t_par, "speedup": round(speedup, 4)})
        print()

    write_csv(csv_path, rows)
    print(f"Results written to {csv_path}")

    if not args.no_plot and rows:
        make_plot(rows, plot_path, args.heads, args.d_model)

    # Print summary table
    print()
    print("Summary (speedup = T_seq / T_par):")
    print(f"{'N':>6}  {'P':>4}  {'mode':>8}  {'time':>10}  {'speedup':>8}  {'eff%':>6}")
    print("-" * 54)
    for r in rows:
        sp = f"{r['speedup']:.2f}x" if r['procs'] > 1 else "baseline"
        ef = f"{r['speedup']/r['procs']*100:.0f}%" if r['procs'] > 1 else ""
        print(f"{r['seq_len']:>6}  {r['procs']:>4}  {r['mode']:>8}  "
              f"{r['time_s']:>10.4f}s  {sp:>8}  {ef:>6}")


if __name__ == "__main__":
    main()
