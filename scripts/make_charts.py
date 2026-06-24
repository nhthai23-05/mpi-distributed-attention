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


def _optf(row, key, default=0.0):
    """Read an optional numeric CSV field (older CSVs may lack the C2 columns)."""
    try:
        v = row.get(key, default)
        return float(v) if v not in (None, "") else default
    except (ValueError, TypeError):
        return default


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
    wait = [_optf(r, "t_wait") for r in rows]
    msgs = [int(_optf(r, "msgs")) for r in rows]
    byts = [int(_optf(r, "bytes")) for r in rows]
    has_wait = any(w > 0 for w in wait)
    totals = [c + m + w for c, m, w in zip(comp, comm, wait)]

    fig, ax = plt.subplots(figsize=(max(8, len(ranks) * 0.6), 5))
    ax.bar(ranks, comp, label="computation", color="#1f77b4")
    ax.bar(ranks, comm, bottom=comp, label="communication", color="#ff7f0e")
    if has_wait:
        base = [c + m for c, m in zip(comp, comm)]
        ax.bar(ranks, wait, bottom=base, label="wait (idle at sync)", color="#d62728")
    ax.set_xlabel("Process rank")
    ax.set_ylabel("Time (s)")
    n = rows[0]["seq_len"]
    ax.set_title(f"Chart C — Per-process compute / comm / wait  (N={n}, P={len(ranks)})")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend()

    # Load imbalance is measured on the WORK each rank does (compute+comm), not on
    # the total wall (compute+comm+wait): with --profile-wait the barriers equalize
    # the wall time (the wait segment just fills the gap), so total-wall imbalance is
    # trivially ~0. The brief's question — does the *idle* time differ by >25%? — is
    # answered by how uneven the work is, which the red wait segments then visualize.
    work = [c + m for c, m in zip(comp, comm)]
    lo, hi = min(work), max(work)
    imbalance = (hi - lo) / lo * 100 if lo > 0 else 0.0
    verdict = "BALANCED (<=25%)" if imbalance <= 25 else "IMBALANCED (>25%) — adjust granularity"
    note = f"work (compute+comm) imbalance (max-min)/min = {imbalance:.1f}%  -> {verdict}"
    if has_wait:
        note += "\nred = idle wait (fast ranks blocked at barriers for slow ones)"
    if sum(msgs) or sum(byts):
        note += (f"\ntransfers this run: {sum(msgs)} msgs, "
                 f"{sum(byts)/1e6:.1f} MB total across {len(ranks)} ranks")
    ax.text(0.5, 0.97, note,
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

    # C2: communication volume / link latency per P (optional columns).
    has_comm = "bytes" in rows[0] or "latency_us" in rows[0]
    byts = [_optf(r, "bytes") for r in rows]
    lat  = [_optf(r, "latency_us") for r in rows]

    ncols = 3 if has_comm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))
    ax1, ax2 = axes[0], axes[1]

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

    if has_comm:
        ax3 = axes[2]
        ax3.plot(procs, [b / 1e6 for b in byts], "o-", color="#9467bd",
                 label="bytes moved (rank 0)")
        ax3.set_xlabel("Processes P"); ax3.set_ylabel("Communication volume (MB)")
        ax3.set_title("Chart D(iii) — Comm volume & link latency vs P")
        ax3.set_xscale("log", base=2); ax3.set_xticks(procs); ax3.set_xticklabels(procs)
        ax3.grid(True, which="both", ls=":", alpha=0.5)
        ax3b = ax3.twinx()
        ax3b.plot(procs, lat, "s--", color="#8c564b", label="link latency (µs)")
        ax3b.set_ylabel("One-way link latency (µs)")
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax3b.get_legend_handles_labels()
        ax3.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    out = os.path.join(results_dir, "chart-d-speedup.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_threads(results_dir):
    """OpenMP thread-scaling (from bench_threads.sh): runtime and speedup vs threads."""
    path = os.path.join(results_dir, "threads.csv")
    if not os.path.exists(path):
        print(f"skip Chart Threads: {path} not found")
        return
    rows = sorted(_read(path), key=lambda r: int(r["threads"]))
    if not rows:
        print("skip Chart Threads: no data rows")
        return
    th = [int(r["threads"]) for r in rows]
    t  = [float(r["time_s"]) for r in rows]
    sp = [_optf(r, "speedup") for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(th, t, "o-", color="#1f77b4")
    ax1.set_xlabel("OpenMP threads (1 MPI rank)"); ax1.set_ylabel("Runtime (s)")
    ax1.set_title("OpenMP — runtime vs threads")
    ax1.set_xscale("log", base=2); ax1.set_xticks(th); ax1.set_xticklabels(th)
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    ax2.plot(th, th, "k:", label="ideal (linear)")
    ax2.plot(th, sp, "o-", color="#2ca02c", label="measured")
    ax2.set_xlabel("OpenMP threads (1 MPI rank)"); ax2.set_ylabel("Speedup  T(1)/T(t)")
    ax2.set_title("OpenMP — speedup vs threads\n(plateaus: kernel is memory-bandwidth bound)")
    ax2.set_xscale("log", base=2); ax2.set_xticks(th); ax2.set_xticklabels(th)
    ax2.grid(True, which="both", ls=":", alpha=0.5); ax2.legend()

    out = os.path.join(results_dir, "chart-threads.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_ring(results_dir):
    """Ring vs 1D (from bench_ring_vs_1d.sh).

    Left  : measured total time (compute+comm+wait) vs P.
    Right : peak K,V rows resident per rank — the key win. 1D broadcasts ALL keys
            so every rank holds N rows; ring shards them, so each rank holds only
            ceil(N/P). (Total bytes moved are ~equal — ring trades one broadcast
            for P-1 smaller shifts — so the advantage is memory + overlap, not
            volume; on a single shared-memory node ring can even be slightly
            slower because comm is nearly free there.)
    """
    path = os.path.join(results_dir, "ring_vs_1d.csv")
    if not os.path.exists(path):
        print(f"skip Chart Ring: {path} not found")
        return
    rows = _read(path)
    if not rows:
        print("skip Chart Ring: no data rows")
        return
    by_var = defaultdict(dict)            # by_var[variant][P] = row
    for r in rows:
        by_var[r["variant"]][int(r["procs"])] = r
    colors = {"1d": "#ff7f0e", "ring": "#1f77b4"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for var in sorted(by_var):
        Ps    = sorted(by_var[var])
        total = [float(by_var[var][p]["t_compute"]) + float(by_var[var][p]["t_comm"])
                 + _optf(by_var[var][p], "t_wait") for p in Ps]
        # peak resident K,V rows per rank: 1D holds all N, ring holds ceil(N/P)
        peak  = []
        for p in Ps:
            N = int(by_var[var][p]["seq_len"])
            peak.append(N if var == "1d" else -(-N // p))   # ceil div for ring
        ax1.plot(Ps, total, "o-", label=var, color=colors.get(var))
        ax2.plot(Ps, peak, "o-", label=var, color=colors.get(var))

    allP = sorted({int(r["procs"]) for r in rows})
    ax1.set_xlabel("Processes P"); ax1.set_ylabel("Total time compute+comm+wait (s)")
    ax1.set_title("Ring vs 1D — measured total time")
    ax2.set_xlabel("Processes P"); ax2.set_ylabel("Peak K,V rows resident per rank")
    ax2.set_title("Ring vs 1D — memory footprint per rank\n(1D: O(N) replicated   ring: O(N/P) sharded)")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2); ax.set_xticks(allP); ax.set_xticklabels(allP)
        ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()

    out = os.path.join(results_dir, "chart-ring-vs-1d.png")
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
    chart_threads(args.results_dir)
    chart_ring(args.results_dir)


if __name__ == "__main__":
    main()
