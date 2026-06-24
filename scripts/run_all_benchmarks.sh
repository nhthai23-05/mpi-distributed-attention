#!/bin/bash
# Driver: run every report benchmark in sequence on the real 3-node cluster.
# Cluster: master(8) + slave1(4) + slave2(10) = 22 physical cores, WiFi LAN.
# Binary deployed to /tmp/mpiattn on every node (no NFS; usernames differ).
#
# Charts produced (CSV in results/):
#   C  granularity   hybrid  N*=32768  P=22   (load balance)
#   B  n-sizing      hybrid          P=22     (runtime vs N, stop ~N*)
#   D  speedup       tensor  2N*=65536  P=1..44
#   E  ring vs 1d    tensor  N=16384  P=2..16
#   F  openmp        seq     N=8192   threads 1..16  (master only)
set -u
cd "$(dirname "$0")/.."

export BINARY=/tmp/mpiattn/hybrid_attention
export NET=192.168.124.0/24
export HOSTFILE=hostfile
export OMP_NUM_THREADS=1
export LC_ALL=C

log() { echo ""; echo "############################################################"; echo "# $* — $(date +%H:%M:%S)"; echo "############################################################"; }

SUITE_START=$SECONDS

log "CHART C — granularity / load balance (hybrid, N=32768, P=22)"
TOTAL_PROCS=22 N=32768 MODE=hybrid bash scripts/bench_granularity.sh

log "CHART B — runtime vs input size N (hybrid, P=22, stop ~N*=32768)"
TOTAL_PROCS=22 MODE=hybrid TARGET_S=140 bash scripts/bench_n_sizing.sh

log "CHART D — speedup 1..44 (tensor, 2N*=65536, mem-safe oversubscribe)"
TOTAL_PROCS=44 N=32768 MODE=tensor \
  PLIST="2 4 8 16 22 44" PHYS_CORES=22 HOSTFILE_BIG=hostfile_big \
  bash scripts/bench_speedup.sh

log "CHART E — ring vs 1D (tensor, N=16384, P=2..16)"
TOTAL_PROCS=16 N=16384 bash scripts/bench_ring_vs_1d.sh

log "CHART F — OpenMP thread scaling (master, N=8192, threads 1..16)"
MAX_THREADS=16 N=8192 bash scripts/bench_threads.sh

log "ALL BENCHMARKS DONE — total $(( SECONDS - SUITE_START ))s"
echo "CSV files:"; ls -la results/*.csv
