#!/bin/bash
# Speedup benchmark: vary process count from 1 to TOTAL_PROCS at input size 2*N.
#
# Usage: TOTAL_PROCS=4 N=1024 MODE=hybrid bash scripts/bench_speedup.sh
#
# Output: results/speedup.csv
#   procs,seq_len,t_wall_with_comm,t_wall_no_comm,speedup_with,speedup_no

BINARY="./hybrid_attention"
MODE="${MODE:-hybrid}"
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

# Sequential baseline (1 process)
echo "  p=1 (sequential baseline)..."
mpirun --oversubscribe --prefix /usr \
       --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
       --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
       -np 1 --hostfile "$HOSTFILE" \
       "$BINARY" --mode seq --seq-len "$INPUT_N" --no-check 2>/dev/null \
    > "$TMP" 2>&1
T1_WITH=$(grep "\[SEQ\]" "$TMP" | grep -oP 'time=\K[0-9.]+' | head -1)
T1_WITH="${T1_WITH:-0}"
T1_NO="$T1_WITH"  # sequential has no separate comm time
echo "    t_seq=${T1_WITH}s"
echo "1,$INPUT_N,$T1_WITH,$T1_NO,1.00,1.00" >> "$OUT"

# Parallel runs: vary p = 2, 4, 8, ..., up to TOTAL_PROCS
P=2
while [ "$P" -le "$TOTAL_PROCS" ]; do
    echo "  p=$P ..."
    mpirun --oversubscribe --prefix /usr \
           --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
           --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
           -np "$P" --hostfile "$HOSTFILE" \
           "$BINARY" --mode "$MODE" --seq-len "$INPUT_N" --csv --no-check 2>/dev/null \
        | grep "^0," > "$TMP"   # take rank 0's row

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
