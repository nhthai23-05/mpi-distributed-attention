#!/bin/bash
# N-sizing benchmark: find N where runtime ≈ 2-3 minutes
# Usage: TOTAL_PROCS=4 MODE=hybrid bash scripts/bench_n_sizing.sh
#
# Output: CSV to results/n_sizing.csv
#   rank,seq_len,t_io,t_compute,t_comm

BINARY="./hybrid_attention"
MODE="${MODE:-hybrid}"
TOTAL_PROCS="${TOTAL_PROCS:-4}"
HOSTFILE="${HOSTFILE:-hostfile}"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/n_sizing.csv"

echo "rank,seq_len,t_io,t_compute,t_comm" > "$OUT"

# Sweep N — start small to calibrate, go up to find the 2-3 min window
for N in 64 128 256 512 1024 2048 4096 8192; do
    echo "  Running N=$N with $TOTAL_PROCS procs ..."
    # Capture per-rank timings (skip CSV header line)
    mpirun --oversubscribe --prefix /usr \
           --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
           --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
           -np "$TOTAL_PROCS" --hostfile "$HOSTFILE" \
           "$BINARY" --mode "$MODE" --seq-len "$N" --csv --no-check 2>/dev/null \
        | grep -v "^rank" >> "$OUT"

    # Estimate total wall time from rank 0's compute+comm
    WALL=$(grep "^0," "$RESULTS_DIR/n_sizing_tmp.csv" 2>/dev/null | \
           awk -F',' '{printf "%.2f", $4+$5}')
    echo "    N=$N  approx_wall=${WALL}s"

    # Stop if we've exceeded 5 minutes (no point going further)
    if awk -v w="$WALL" 'BEGIN{exit (w+0 > 300) ? 0 : 1}'; then
        echo "  Stopping: N=$N takes >5 min"
        break
    fi
done

echo ""
echo "Results written to $OUT"
echo ""
echo "--- Summary (rank 0 rows) ---"
grep "^0," "$OUT" | awk -F',' '{
    wall=$4+$5
    printf "  N=%-5s  compute=%.3fs  comm=%.3fs  total=%.3fs\n", $2, $4, $5, wall
}'
