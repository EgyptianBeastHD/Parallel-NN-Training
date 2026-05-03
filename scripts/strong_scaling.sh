#!/usr/bin/env bash
# Strong-scaling sweep: fix the global batch and dataset, vary rank count,
# record per-epoch time. Run from the project root.
#
#   ./scripts/strong_scaling.sh                   # defaults: ranks 1 2 4 8, ring
#   ALGO=hd RANKS="1 2 4 8 16" ./scripts/strong_scaling.sh
#
# Output: scaling_<algo>.csv with columns: ranks, epoch, loss, acc, time_s, samples_per_s

set -euo pipefail

ALGO=${ALGO:-ring}
RANKS=${RANKS:-"1 2 4 8"}
EPOCHS=${EPOCHS:-3}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
EXTRA=${EXTRA:-""}
OUT="scaling_${ALGO}.csv"

if [[ ! -x build/train_mpi ]]; then
    echo "build/train_mpi missing. Run 'make mpi' first." >&2
    exit 1
fi

echo "ranks,epoch,loss,acc,time_s,samples_per_s" > "$OUT"

for P in $RANKS; do
    echo "=== ranks=$P algo=$ALGO ==="
    LOG=$(mktemp)
    mpirun -np "$P" --oversubscribe ./build/train_mpi \
        --algo "$ALGO" --epochs "$EPOCHS" --global-batch "$GLOBAL_BATCH" \
        --warmup 5 $EXTRA | tee "$LOG"

    # Parse "Epoch X/Y | Loss: L | Test Acc: A% | Time: Ts | S samples/s"
    grep -E '^Epoch ' "$LOG" | awk -v p="$P" '{
        split($2, e, "/");
        loss = $5;
        acc  = $9; gsub(/%/, "", acc);
        time = $12; gsub(/s/, "", time);
        sps  = $14;
        printf "%d,%d,%s,%s,%s,%s\n", p, e[1], loss, acc, time, sps
    }' >> "$OUT"
    rm "$LOG"
    echo
done

echo "wrote $OUT"
