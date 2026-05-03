#!/usr/bin/env bash
# Compare all-reduce algorithms at a fixed rank count.
# Logs the per-iteration timing summary (compute / comm / total) for each.
#
#   ./scripts/compare_algos.sh             # defaults: P=8, all algorithms
#   P=4 ./scripts/compare_algos.sh
#   ALGOS="ring hd" P=8 ./scripts/compare_algos.sh
#
# Also runs the non-blocking pipelined variant for comparison against blocking
# ring (useful for showing the comm/compute overlap effect).

set -euo pipefail

P=${P:-8}
EPOCHS=${EPOCHS:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
ALGOS=${ALGOS:-"tree ring hd mpi"}
OUT="compare_p${P}.csv"

if [[ ! -x build/train_mpi ]]; then
    echo "build/train_mpi missing. Run 'make mpi' first." >&2
    exit 1
fi

echo "algo,nonblocking,compute_ms,allreduce_ms,iter_ms,allreduce_pct" > "$OUT"

run_one() {
    local algo="$1"
    local nb="$2"
    local label="$algo$([ "$nb" = "1" ] && echo " (nonblocking)" || echo "")"
    echo "=== $label  P=$P ==="

    local extra=""
    [[ "$nb" = "1" ]] && extra="--nonblocking"

    local LOG; LOG=$(mktemp)
    mpirun -np "$P" --oversubscribe ./build/train_mpi \
        --algo "$algo" --epochs "$EPOCHS" --global-batch "$GLOBAL_BATCH" \
        --warmup 10 $extra | tee "$LOG"

    # Grab the timing summary block. Output format:
    #   avg compute / iter:      16.479 ms                    -> field 5
    #   avg all-reduce/iter:     46.059 ms  (70.2% of iter)   -> field 3, pct in field 5
    #   avg total / iter:        65.649 ms                    -> field 5
    local comp comm iter pct
    comp=$(awk '/avg compute/    { print $5 }' "$LOG")
    comm=$(awk '/avg all-reduce/ { print $3 }' "$LOG")
    iter=$(awk '/avg total/      { print $5 }' "$LOG")
    pct=$( awk '/avg all-reduce/ { gsub(/[()%]/,"",$5); print $5 }' "$LOG")
    echo "$algo,$nb,$comp,$comm,$iter,$pct" >> "$OUT"
    rm "$LOG"
    echo
}

for algo in $ALGOS; do
    run_one "$algo" 0
done

# Non-blocking pipelined variant -- the algorithm field is informational,
# under the hood it uses MPI_Iallreduce.
run_one "ring" 1

echo "wrote $OUT"
