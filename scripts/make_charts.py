#!/usr/bin/env python3
"""Turn the benchmark CSVs in results/ into the three charts the report needs.

Inputs (produced by the bench_*.sh scripts):
  results/n_sizing.csv     rank,seq_len,t_io,t_compute,t_comm        (all ranks, many N)
  results/granularity.csv  rank,seq_len,t_io,t_compute,t_comm        (all ranks, one N)
  results/speedup.csv      procs,seq_len,t_wall_with_comm,t_wall_no_comm,speedup_with,speedup_no

Outputs (PNG, 150 dpi) into results/:
  chart-b-n-sizing.png     execution time vs input size N (with / without comm)
  chart-c-granularity.png  per-process compute+comm stacked bars (+ load-imbalance %)
  chart-d-speedup.png      runtime vs P and speedup vs P (with / without comm)

Usage:
  python3 scripts/make_charts.py                  # read everything in results/
  python3 scripts/make_charts.py --results-dir X  # custom dir

Only the charts whose CSV exists are generated; missing ones are skipped with a note.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: works over SSH / on the cluster
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib   (or: sudo apt install python3-matplotlib)")


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def chart_b(results_dir):
    """Chart B: program execution time vs input size N, with and without comm.

    'Execution time from start to finish' is approximated by the slowest rank,
    i.e. max over ranks of (compute+comm) for the with-comm curve and max over
    ranks of compute for the without-comm curve.
    """
    path = os.path.join(results_dir, "n_sizing.csv")
    if not os.path.exists(path):
        print(f"skip Chart B: {path} not found")
        return
    with_comm, without_comm = defaultdict(float), defaultdict(float)
    for r in _read(path):
        n = int(r["seq_len"])
        comp, comm = float(r["t_compute"]), float(r["t_comm"])
        with_comm[n] = max(with_comm[n], comp + comm)
        without_comm[n] = max(without_comm[n], comp)
    ns = sorted(with_comm)
    if not ns:
        print("skip Chart B: no data rows")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [with_comm[n] for n in ns], "o-", label="with communication", color="#1f77b4")
    ax.plot(ns, [without_comm[n] for n in ns], "s--", label="computation only", color="#ff7f0e")
    ax.axhspan(120, 180, color="green", alpha=0.08, label="2-3 min target")
    ax.set_xlabel("Input size N (seq_len)")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Chart B — Execution time vs input size N")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    out = os.path.join(results_dir, "chart-b-n-sizing.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_c(results_dir):
    """Chart C: per-process compute + comm stacked bars; report load imbalance."""
    path = os.path.join(results_dir, "granularity.csv")
    if not os.path.exists(path):
        print(f"skip Chart C: {path} not found")
        return
    rows = sorted(_read(path), key=lambda r: int(r["rank"]))
    if not rows:
        print("skip Chart C: no data rows")
        return
    ranks = [int(r["rank"]) for r in rows]
    comp = [float(r["t_compute"]) for r in rows]
    comm = [float(r["t_comm"]) for r in rows]
    totals = [c + m for c, m in zip(comp, comm)]

    fig, ax = plt.subplots(figsize=(max(8, len(ranks) * 0.5), 5))
    ax.bar(ranks, comp, label="computation", color="#1f77b4")
    ax.bar(ranks, comm, bottom=comp, label="communication", color="#ff7f0e")
    ax.set_xlabel("Process rank")
    ax.set_ylabel("Time (s)")
    n = rows[0]["seq_len"]
    ax.set_title(f"Chart C — Per-process compute vs comm  (N={n}, P={len(ranks)})")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend()

    lo, hi = min(totals), max(totals)
    imbalance = (hi - lo) / lo * 100 if lo > 0 else 0.0
    verdict = "BALANCED (<=25%)" if imbalance <= 25 else "IMBALANCED (>25%) — adjust granularity"
    ax.text(0.5, 0.97, f"load imbalance (max-min)/min = {imbalance:.1f}%  -> {verdict}",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    out = os.path.join(results_dir, "chart-c-granularity.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}  (load imbalance {imbalance:.1f}%)")


def chart_d(results_dir):
    """Chart D: runtime vs P and speedup vs P, with and without comm."""
    path = os.path.join(results_dir, "speedup.csv")
    if not os.path.exists(path):
        print(f"skip Chart D: {path} not found")
        return
    rows = sorted(_read(path), key=lambda r: int(r["procs"]))
    if not rows:
        print("skip Chart D: no data rows")
        return
    procs = [int(r["procs"]) for r in rows]
    t_with = [float(r["t_wall_with_comm"]) for r in rows]
    t_no = [float(r["t_wall_no_comm"]) for r in rows]

    def _f(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return float("nan")
    sp_with = [_f(r["speedup_with"]) for r in rows]
    sp_no = [_f(r["speedup_no"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(procs, t_with, "o-", label="with communication", color="#1f77b4")
    ax1.plot(procs, t_no, "s--", label="computation only", color="#ff7f0e")
    ax1.set_xlabel("Processes P"); ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Chart D(i) — Runtime vs processes")
    ax1.set_xscale("log", base=2); ax1.set_xticks(procs); ax1.set_xticklabels(procs)
    ax1.grid(True, which="both", ls=":", alpha=0.5); ax1.legend()

    ax2.plot(procs, procs, "k:", label="ideal (linear)")
    ax2.plot(procs, sp_with, "o-", label="with communication", color="#1f77b4")
    ax2.plot(procs, sp_no, "s--", label="computation only", color="#ff7f0e")
    ax2.set_xlabel("Processes P"); ax2.set_ylabel("Speedup  T(1)/T(P)")
    ax2.set_title("Chart D(ii) — Speedup vs processes")
    ax2.set_xscale("log", base=2); ax2.set_xticks(procs); ax2.set_xticklabels(procs)
    ax2.grid(True, which="both", ls=":", alpha=0.5); ax2.legend()

    out = os.path.join(results_dir, "chart-d-speedup.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot the benchmark CSVs into report charts.")
    ap.add_argument("--results-dir", default="results", help="directory holding the *.csv files")
    args = ap.parse_args()
    if not os.path.isdir(args.results_dir):
        sys.exit(f"results dir not found: {args.results_dir}")
    chart_b(args.results_dir)
    chart_c(args.results_dir)
    chart_d(args.results_dir)


if __name__ == "__main__":
    main()
