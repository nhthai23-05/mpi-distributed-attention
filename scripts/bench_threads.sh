#!/bin/bash
# OpenMP thread-scaling: fix MPI at a single rank and vary OMP_NUM_THREADS. This
# isolates the intra-node (shared-memory) speedup of the attention kernel from
# MPI. Uses --mode seq so it is pure compute (no communication).
#
# Note: the kernel re-reads K,V for every query row, so it is memory-bandwidth
# bound — expect speedup to plateau well before #cores. That plateau is itself a
# reportable result (and motivates the sharded ring path, where each rank's K,V
# block is only N/P rows and stays closer to cache).
#
# Usage: MAX_THREADS=8 N=2048 bash scripts/bench_threads.sh

export LC_ALL=C

BINARY="./hybrid_attention"
N="${N:-2048}"
MAX_THREADS="${MAX_THREADS:-$(nproc)}"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/threads.csv"
echo "threads,seq_len,time_s,speedup" > "$OUT"

echo "OpenMP thread-scaling sweep: N=$N  up to $MAX_THREADS threads"

T1=""
T=1
while [ "$T" -le "$MAX_THREADS" ]; do
    secs=$(OMP_NUM_THREADS=$T mpirun --oversubscribe -np 1 "$BINARY" \
               --mode seq --seq-len "$N" 2>/dev/null \
           | grep -oE 'time=[0-9.]+' | head -1 | cut -d= -f2)
    if [ -z "$secs" ]; then
        echo "  threads=$T FAILED"; T=$((T * 2)); continue
    fi
    [ -z "$T1" ] && T1="$secs"
    sp=$(awk "BEGIN{if($secs>0) printf \"%.3f\", $T1/$secs; else print \"NA\"}")
    printf "  threads=%-3s time=%ss  speedup=%sx\n" "$T" "$secs" "$sp"
    echo "$T,$N,$secs,$sp" >> "$OUT"
    T=$((T * 2))
done

echo ""
echo "Results written to $OUT"
