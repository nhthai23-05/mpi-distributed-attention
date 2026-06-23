#!/bin/bash
# Granularity benchmark: per-process compute vs communication breakdown.
# Use the N found from bench_n_sizing.sh (target: ~2-3 min runtime).
#
# Usage: TOTAL_PROCS=4 N=2048 MODE=hybrid bash scripts/bench_granularity.sh
#
# Output: results/granularity.csv
#   rank,seq_len,t_io,t_compute,t_comm

BINARY="./hybrid_attention"
MODE="${MODE:-hybrid}"
TOTAL_PROCS="${TOTAL_PROCS:-4}"
HOSTFILE="${HOSTFILE:-hostfile}"
N="${N:-1024}"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/granularity.csv"

echo "rank,seq_len,t_io,t_compute,t_comm" > "$OUT"

echo "Running granularity benchmark: N=$N, procs=$TOTAL_PROCS, mode=$MODE"
mpirun --oversubscribe --prefix /usr \
       --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
       --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
       -np "$TOTAL_PROCS" --hostfile "$HOSTFILE" \
       "$BINARY" --mode "$MODE" --seq-len "$N" --csv --no-check 2>/dev/null \
    | grep -v "^rank" >> "$OUT"

echo ""
echo "Results written to $OUT"
echo ""
echo "--- Per-rank breakdown ---"
grep -v "^rank" "$OUT" | awk -F',' '{
    total = $4 + $5
    printf "  rank=%s  compute=%.3fs  comm=%.3fs  total=%.3fs\n", $1, $4, $5, total
    totals[$1] = total
}
END {
    # Check load balance: find min/max total time
    min_t = 1e18; max_t = 0
    for (r in totals) {
        if (totals[r] < min_t) min_t = totals[r]
        if (totals[r] > max_t) max_t = totals[r]
    }
    if (min_t > 0) {
        imbalance = (max_t - min_t) / min_t * 100
        printf "\n  Load imbalance: %.1f%%", imbalance
        if (imbalance > 25)
            printf "  WARNING: exceeds 25%% threshold — adjust granularity\n"
        else
            printf "  OK (< 25%%)\n"
    }
}'
