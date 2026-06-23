#!/bin/bash
# Strong-scaling speedup: vary process count from 1 to TOTAL_PROCS at input 2*N.
#
# Usage: TOTAL_PROCS=4 N=1024 MODE=tensor bash scripts/bench_speedup.sh
#
# Output: results/speedup.csv
#   procs,seq_len,t_wall_with_comm,t_wall_no_comm,speedup_with,speedup_no
#
# Why tensor mode: a valid speedup curve needs the SAME kernel at every P,
# including the P=1 anchor. Tensor mode runs the identical tp_allgather kernel
# for P=1..N. Hybrid CANNOT run at P=1 (it needs P >= --groups), so it cannot
# anchor an absolute-speedup chart — use bench_granularity.sh to characterize
# hybrid instead. T(1) is measured with the same --csv profiler as T(P), so the
# "with comm" / "no comm" speedups are apples-to-apples.

# Force C locale so awk/printf use '.' decimals — a ',' decimal (some locales)
# would corrupt the comma-separated CSV and break the speedup arithmetic.
export LC_ALL=C

BINARY="./hybrid_attention"
MODE="${MODE:-tensor}"
TOTAL_PROCS="${TOTAL_PROCS:-4}"
HOSTFILE="${HOSTFILE:-hostfile}"
N="${N:-1024}"
INPUT_N=$((N * 2))   # Speedup chart uses 2*N
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/speedup.csv"
TMP="$RESULTS_DIR/.speedup_tmp.csv"

echo "procs,seq_len,t_wall_with_comm,t_wall_no_comm,speedup_with,speedup_no" > "$OUT"

echo "Running speedup benchmark: N=$INPUT_N (=2*$N), mode=$MODE"

# Baseline: the SAME kernel/mode at P=1 (comm ~ 0). Measured with --csv so the
# compute/comm split matches the parallel runs below.
echo "  p=1 (serial baseline, --mode $MODE)..."
step_start=$SECONDS
mpirun --oversubscribe --prefix /usr \
       --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
       --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
       -np 1 --hostfile "$HOSTFILE" \
       "$BINARY" --mode "$MODE" --seq-len "$INPUT_N" --csv --no-check 2>/dev/null \
    | grep "^0," > "$TMP"
echo "    (wall $(( SECONDS - step_start ))s)"
T1_COMPUTE=$(awk -F',' 'NR==1{print $4}' "$TMP")
T1_COMM=$(awk -F',' 'NR==1{print $5}' "$TMP")
if [ -z "$T1_COMPUTE" ]; then
    echo "ERROR: no P=1 baseline row. Mode '$MODE' likely can't run at -np 1" >&2
    echo "       (hybrid needs P >= --groups). Re-run with MODE=tensor."          >&2
    exit 1
fi
T1_WITH=$(awk "BEGIN{printf \"%.6f\", ${T1_COMPUTE:-0} + ${T1_COMM:-0}}")
T1_NO="$T1_COMPUTE"
echo "    t1_compute=${T1_COMPUTE}s  t1_comm=${T1_COMM}s"
echo "1,$INPUT_N,$T1_WITH,$T1_NO,1.00,1.00" >> "$OUT"

# Parallel runs: vary p = 2, 4, 8, ..., up to TOTAL_PROCS
P=2
while [ "$P" -le "$TOTAL_PROCS" ]; do
    echo "  p=$P ..."
    step_start=$SECONDS
    mpirun --oversubscribe --prefix /usr \
           --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
           --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
           -np "$P" --hostfile "$HOSTFILE" \
           "$BINARY" --mode "$MODE" --seq-len "$INPUT_N" --csv --no-check 2>/dev/null \
        | grep "^0," > "$TMP"   # take rank 0's row
    echo "    (wall $(( SECONDS - step_start ))s)"

    T_IO=$(awk -F',' 'NR==1{print $3}' "$TMP")
    T_COMPUTE=$(awk -F',' 'NR==1{print $4}' "$TMP")
    T_COMM=$(awk -F',' 'NR==1{print $5}' "$TMP")

    T_WITH=$(awk "BEGIN{printf \"%.6f\", ${T_COMPUTE:-0} + ${T_COMM:-0}}")
    T_NO=$(awk   "BEGIN{printf \"%.6f\", ${T_COMPUTE:-0}}")

    SP_WITH=$(awk "BEGIN{if($T_WITH>0) printf \"%.4f\", $T1_WITH/$T_WITH; else print \"N/A\"}")
    SP_NO=$(awk   "BEGIN{if($T_NO>0)   printf \"%.4f\", $T1_NO/$T_NO;     else print \"N/A\"}")

    echo "    t_wall(with_comm)=${T_WITH}s  speedup=${SP_WITH}"
    echo "    t_wall(no_comm)  =${T_NO}s    speedup=${SP_NO}"
    echo "$P,$INPUT_N,$T_WITH,$T_NO,$SP_WITH,$SP_NO" >> "$OUT"

    P=$((P * 2))
done

rm -f "$TMP"
echo ""
echo "Results written to $OUT"
echo ""
echo "--- Speedup table ---"
cat "$OUT"
