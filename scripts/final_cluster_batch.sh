#!/bin/bash
# Final cluster batch (optimized binary):
#  1. tensor-1D granularity at N*=32768, p=22 (for the load-balance comparison)
#  2. clean speedup re-run of p=16 and p=22 at 2N*=65536 (replace the
#     memory-pressured old-binary rows so the upper curve is consistent)
cd /home/thach/HUST/Parallel/mpi-distributed-attention
export OMP_NUM_THREADS=1 LC_ALL=C
B=/tmp/mpiattn/hybrid_attention
SCR=/tmp/claude-1000/-home-thach-HUST-Parallel-mpi-distributed-attention/d9b9c3fe-5a23-4d7b-8546-7c633bf5c4f2/scratchpad
MPI="mpirun --oversubscribe --prefix /usr --mca btl_tcp_if_include 192.168.124.0/24 --mca oob_tcp_if_include 192.168.124.0/24 --hostfile hostfile"

echo "[$(date +%H:%M:%S)] (1) tensor-1D granularity N=32768 p=22 ..."
$MPI -np 22 "$B" --mode tensor --seq-len 32768 --csv --no-check --profile-wait 2>/dev/null \
  | grep -v "^rank" > "$SCR/granularity_tensor.csv"
echo "  rows: $(wc -l < "$SCR/granularity_tensor.csv")"
awk -F',' '{tot=$4+$5; if(tot>mx)mx=tot; if(min==""||tot<min)min=tot}
  END{if(min>0) printf "  tensor imbalance (max-min)/min = %.1f%%  (min=%.1fs max=%.1fs)\n",(mx-min)/min*100,min,mx}' \
  "$SCR/granularity_tensor.csv"

# Clean speedup re-runs (optimized binary). T1 baseline reused from speedup.csv.
T1_WITH=401.694945; T1_NO=401.650985; INPUT_N=65536
: > "$SCR/speedup_clean_1622.csv"
for P in 16 22; do
  echo "[$(date +%H:%M:%S)] (2) speedup p=$P at N=$INPUT_N (optimized) ..."
  t0=$SECONDS
  row=$($MPI -np "$P" "$B" --mode tensor --seq-len $INPUT_N --csv --no-check 2>/dev/null | grep "^0,")
  echo "    wall=$((SECONDS-t0))s  row=$row"
  comp=$(echo "$row"|cut -d, -f4); comm=$(echo "$row"|cut -d, -f5)
  msgs=$(echo "$row"|cut -d, -f7); byt=$(echo "$row"|cut -d, -f8); lat=$(echo "$row"|cut -d, -f9)
  tw=$(awk "BEGIN{printf \"%.6f\",$comp+$comm}"); tn=$comp
  spw=$(awk "BEGIN{printf \"%.4f\",$T1_WITH/$tw}"); spn=$(awk "BEGIN{printf \"%.4f\",$T1_NO/$tn}")
  echo "    => with=${tw}s sp=$spw | no=${tn}s sp=$spn | bytes=$byt lat=${lat}us"
  echo "$P,$INPUT_N,$tw,$tn,$spw,$spn,$msgs,$byt,$lat" >> "$SCR/speedup_clean_1622.csv"
done
echo "[$(date +%H:%M:%S)] BATCH DONE"
echo "--- tensor granularity per-rank ---"; cat "$SCR/granularity_tensor.csv"
echo "--- clean p16/p22 ---"; cat "$SCR/speedup_clean_1622.csv"
