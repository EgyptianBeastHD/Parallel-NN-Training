#!/usr/bin/env bash
# Microbenchmark each all-reduce algorithm across a range of message sizes
# at a fixed rank count. Useful for the alpha-beta model fit in the report.
#
#   ./scripts/microbench.sh
#   P=16 ./scripts/microbench.sh
#
# Output: microbench_p<P>.csv

set -euo pipefail

P=${P:-8}
SIZES=${SIZES:-"1024,8192,65536,262144,1048576,4194304"}
REPS=${REPS:-30}
OUT="microbench_p${P}.csv"

if [[ ! -x build/allreduce_bench ]]; then
    echo "build/allreduce_bench missing. Run 'make bench' first." >&2
    exit 1
fi

echo "ranks,size,algo,median_ms,min_ms,gbs_eff" > "$OUT"

LOG=$(mktemp)
mpirun -np "$P" --oversubscribe ./build/allreduce_bench \
    --sizes "$SIZES" --reps "$REPS" | tee "$LOG"

# Lines look like:   1024       tree         0.1234       0.1100        0.123
awk -v p="$P" '
    /^[[:space:]]+[0-9]/ {
        printf "%d,%s,%s,%s,%s,%s\n", p, $1, $2, $3, $4, $5
    }' "$LOG" >> "$OUT"

rm "$LOG"
echo "wrote $OUT"
