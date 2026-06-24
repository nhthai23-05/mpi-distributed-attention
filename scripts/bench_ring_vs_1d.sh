#!/bin/bash
# Algorithm comparison: 1D (broadcast full K,V) vs RING (sharded K,V, streamed
# with non-blocking overlap + online softmax). Same single-head problem at a
# fixed N, swept over the process count P. The ring path holds only O(N/P) of
# K,V per rank and overlaps the shift with compute, so its *exposed* comm+wait
# is lower; this script captures the evidence for that.
#
# Usage: TOTAL_PROCS=8 N=2048 bash scripts/bench_ring_vs_1d.sh
#
# Output: results/ring_vs_1d.csv
#   variant,procs,seq_len,t_compute,t_comm,t_wait,bytes

export LC_ALL=C
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

BINARY="${BINARY:-./hybrid_attention}"
TOTAL_PROCS="${TOTAL_PROCS:-4}"
HOSTFILE="${HOSTFILE:-hostfile}"
N="${N:-2048}"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/ring_vs_1d.csv"
echo "variant,procs,seq_len,t_compute,t_comm,t_wait,bytes" > "$OUT"

run() {  # $1=extra flags  $2=label  $3=procs
    local flags="$1" label="$2" P="$3" row comp comm wait bytes
    row=$(mpirun --oversubscribe --prefix /usr \
            --mca btl_tcp_if_include "${NET:-192.168.0.0/24}" \
            --mca oob_tcp_if_include "${NET:-192.168.0.0/24}" \
            -np "$P" --hostfile "$HOSTFILE" \
            "$BINARY" --mode tensor $flags --seq-len "$N" --csv --no-check --profile-wait 2>/dev/null \
          | grep "^0,")
    comp=$(echo "$row"  | awk -F',' '{print $4}')
    comm=$(echo "$row"  | awk -F',' '{print $5}')
    wait=$(echo "$row"  | awk -F',' '{print $6}')
    bytes=$(echo "$row" | awk -F',' '{print $8}')
    printf "  %-5s P=%-3s compute=%ss comm=%ss wait=%ss bytes=%s\n" \
           "$label" "$P" "${comp:-?}" "${comm:-?}" "${wait:-?}" "${bytes:-?}"
    echo "$label,$P,$N,${comp:-0},${comm:-0},${wait:-0},${bytes:-0}" >> "$OUT"
}

echo "Ring vs 1D comparison: N=$N, procs 2..$TOTAL_PROCS"
P=2
while [ "$P" -le "$TOTAL_PROCS" ]; do
    run ""       "1d"   "$P"
    run "--ring" "ring" "$P"
    P=$((P * 2))
done

echo ""
echo "Results written to $OUT"
cat "$OUT"
