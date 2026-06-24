#!/bin/bash
# N-sizing benchmark: find N where runtime ≈ 2-3 minutes.
# Usage: TOTAL_PROCS=4 MODE=hybrid bash scripts/bench_n_sizing.sh
#   TARGET_S=180  -> stop once a single N takes this many wall-seconds (default 180)
#
# Output: CSV to results/n_sizing.csv
#   rank,seq_len,t_io,t_compute,t_comm
#
# Progress: prints measured wall time per N, cumulative sweep time, and a rough
# estimate for the next N (work grows ~4x when N doubles, since cost is O(N^2)).

# Force C locale so awk/printf use '.' decimals — a ',' decimal (some locales)
# would corrupt the comma-separated CSV and break the arithmetic below.
export LC_ALL=C
# Keep this MPI-scaling benchmark single-threaded by default so the numbers
# reflect pure MPI process scaling. Override with OMP_NUM_THREADS=N for hybrid runs.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

BINARY="${BINARY:-./hybrid_attention}"
MODE="${MODE:-hybrid}"
TOTAL_PROCS="${TOTAL_PROCS:-4}"
HOSTFILE="${HOSTFILE:-hostfile}"
TARGET_S="${TARGET_S:-180}"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/n_sizing.csv"
TMP="$RESULTS_DIR/.n_sizing_run.csv"   # holds ONE run's output, parsed per-iteration

echo "rank,seq_len,t_io,t_compute,t_comm,t_wait,msgs,bytes,latency_us" > "$OUT"

echo "N-sizing sweep: mode=$MODE  procs=$TOTAL_PROCS  stop-at >= ${TARGET_S}s"
echo ""

sweep_start=$SECONDS

# Sweep N — start small to calibrate, grow until a step crosses the target window.
# The streaming online-softmax kernel uses O(N) scratch (no N x N score matrix),
# so N can now go well past 8192 without the old quadratic-memory wall.
for N in 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do
    echo "  [N=$N] running ..."
    step_start=$SECONDS

    # Capture THIS run's per-rank CSV into a fresh temp file (stderr carries
    # MPI noise + optional --progress, which we drop here for clean CSV).
    mpirun --oversubscribe --prefix /usr \
           --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
           --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
           -np "$TOTAL_PROCS" --hostfile "$HOSTFILE" \
           "$BINARY" --mode "$MODE" --seq-len "$N" --csv --no-check 2>/dev/null \
        | grep -v "^rank" > "$TMP"

    step_wall=$(( SECONDS - step_start ))
    cat "$TMP" >> "$OUT"

    # Rank 0's compute/comm for THIS run only (from the per-run temp file).
    read -r t_comp t_comm <<<"$(awk -F',' '$1==0 {print $4, $5; exit}' "$TMP")"
    sweep_wall=$(( SECONDS - sweep_start ))

    printf "    done: wall=%ss  (rank0 compute=%ss comm=%ss)  | sweep elapsed=%ss\n" \
           "$step_wall" "${t_comp:-?}" "${t_comm:-?}" "$sweep_wall"

    # Stop once a single N reaches the target window (≈2-3 min); that's the N
    # you want for the granularity / speedup charts.
    if [ "$step_wall" -ge "$TARGET_S" ]; then
        echo "  [stop] N=$N hit ${step_wall}s (>= ${TARGET_S}s target). Use N=$N for the report."
        break
    fi

    # Rough estimate for the next step: doubling N ~ 4x the O(N^2) work.
    if [ "$N" -lt 65536 ]; then
        est=$(( step_wall * 4 ))
        echo "    next: N=$(( N * 2 )) est ~${est}s (O(N^2): ~4x per doubling)"
    fi
done

echo ""
echo "Results written to $OUT  (total sweep $(( SECONDS - sweep_start ))s)"
echo ""
echo "--- Summary (rank 0 rows) ---"
grep "^0," "$OUT" | awk -F',' '{
    wall=$4+$5
    printf "  N=%-5s  compute=%.3fs  comm=%.3fs  total=%.3fs\n", $2, $4, $5, wall
}'

rm -f "$TMP"
