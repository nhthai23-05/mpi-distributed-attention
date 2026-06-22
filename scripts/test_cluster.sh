#!/bin/bash
# Part 4 of the cluster setup: prove SSH + NFS + OpenMPI work across all nodes
# BEFORE running the real project. Compiles and runs mpi-prime.c on every host
# listed in the hostfile.
#
# Usage:
#   bash scripts/test_cluster.sh                 # auto-detects total slots
#   NP=12 bash scripts/test_cluster.sh           # force 12 processes
#   HOSTFILE=hostfile bash scripts/test_cluster.sh

set -u

HOSTFILE="${HOSTFILE:-hostfile}"
BIN="mpi-prime"

echo "================================================================"
echo " CLUSTER SMOKE TEST  (mpi-prime across all nodes)"
echo "================================================================"

# --- 0. sanity checks ---
if ! command -v mpicc >/dev/null 2>&1; then
    echo "ERROR: mpicc not found. Is OpenMPI installed and on \$PATH?"
    exit 1
fi
if [ ! -f "$HOSTFILE" ]; then
    echo "ERROR: hostfile '$HOSTFILE' not found."
    exit 1
fi

# --- 1. compile mpi-prime ---
echo "[1/4] Compiling $BIN ..."
mpicc -O2 -std=c11 -o "$BIN" mpi-prime.c -lm || { echo "compile failed"; exit 1; }
echo "      OK"

# --- 2. show the hostfile (the machines we expect to use) ---
echo "[2/4] Nodes declared in $HOSTFILE:"
grep -vE '^\s*#|^\s*$' "$HOSTFILE" | sed 's/^/        /'

# --- 3. compute total processes = sum of slots, unless NP is given ---
if [ -z "${NP:-}" ]; then
    NP=$(grep -vE '^\s*#|^\s*$' "$HOSTFILE" \
         | grep -oE 'slots=[0-9]+' | cut -d= -f2 \
         | awk '{s+=$1} END{print s}')
    NP="${NP:-1}"
fi
echo "[3/4] Launching $BIN with $NP processes ..."

# --- 4. run; --map-by node spreads ranks across machines so we exercise the
#         network (otherwise MPI may pack all ranks onto the first host) ---
echo "[4/4] Output:"
echo "----------------------------------------------------------------"
mpirun --hostfile "$HOSTFILE" -np "$NP" --map-by node ./"$BIN"
STATUS=$?
echo "----------------------------------------------------------------"

if [ "$STATUS" -eq 0 ]; then
    echo "RESULT: cluster OK — mpirun completed across the declared nodes."
    echo "        You can now run ./hybrid_attention the same way:"
    echo "        mpirun --hostfile $HOSTFILE -np $NP ./hybrid_attention --mode hybrid --seq-len 1024"
else
    echo "RESULT: FAILED (exit $STATUS)."
    echo "        Check, in order:  passwordless SSH  ->  /etc/hosts names  ->"
    echo "        same OpenMPI version on every node  ->  NFS-shared binary path."
fi
exit "$STATUS"
