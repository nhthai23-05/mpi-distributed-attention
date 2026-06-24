#!/bin/bash
# Assemble the final, consistent speedup.csv:
#   p=1,2,4,8  : original run (master-only, no network, no memory pressure -> clean)
#   p=16,22    : optimized-binary re-run (scratchpad/speedup_clean_1622.csv)
#   p=32,44    : optimized-binary run    (scratchpad/newpoints.csv)
cd /home/thach/HUST/Parallel/mpi-distributed-attention
SCR=/tmp/claude-1000/-home-thach-HUST-Parallel-mpi-distributed-attention/d9b9c3fe-5a23-4d7b-8546-7c633bf5c4f2/scratchpad
OLD=results/speedup.csv
OUT=results/speedup.csv.new

head -1 "$OLD" > "$OUT"                              # header
awk -F',' '$1==1||$1==2||$1==4||$1==8' "$OLD" >> "$OUT"   # clean low-p
cat "$SCR/speedup_clean_1622.csv" >> "$OUT"          # p16,22 (optimized)
cat "$SCR/newpoints.csv" >> "$OUT"                   # p32,44 (optimized)

mv "$OUT" "$OLD"
echo "=== assembled results/speedup.csv ==="
cat "$OLD"
