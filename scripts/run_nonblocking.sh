#!/usr/bin/env bash
# Collect non-blocking vs blocking comparison data on Perlmutter at P=8.
# Run this inside a salloc/srun-capable session (e.g. after `salloc -N 1 -C cpu ...`).
#
# Writes nonblocking_results.csv with columns:
#   config,params,mode,compute_ms,allreduce_ms,iter_ms,allreduce_pct
#
# After this completes, run plot_nonblocking.py to render the figure.
#
# IMPORTANT: this script assumes ./build/train_mpi accepts a --sizes CLI argument
# that specifies network architecture as a space-separated list of layer sizes.
# If your binary instead requires recompilation per network size, see the
# "OPTION B" comment below.

set -euo pipefail
export OMP_NUM_THREADS=1

OUT=nonblocking_results.csv
echo "config,params,mode,compute_ms,allreduce_ms,iter_ms,allreduce_pct" > "$OUT"

# (config_name, param_count, layer_sizes)
declare -a CONFIGS=(
    "50K|50000|784 64 32 10"
    "235K|235146|784 256 128 10"
    "567K|567306|784 512 256 128 10"
    "1.46M|1462282|784 1024 512 256 10"
)

run_one() {
    local config_name=$1
    local params=$2
    local sizes=$3
    local mode=$4

    local flag=""
    [[ "$mode" == "nonblocking" ]] && flag="--nonblocking"

    local LOG
    LOG=$(mktemp)

    # OPTION A: binary accepts --sizes flag
    srun -n 8 -c 32 --cpu-bind=cores ./build/train_mpi \
        --algo ring \
        --epochs 3 \
        --global-batch 64 \
        --warmup 10 \
        --sizes $sizes \
        $flag > "$LOG" 2>&1

    # OPTION B (if your binary doesn't take --sizes, comment Option A out and
    # uncomment this block instead. You'll need to recompile train_mpi with the
    # right hardcoded sizes between runs.)
    # srun -n 8 -c 32 --cpu-bind=cores ./build/train_mpi \
    #     --algo ring --epochs 3 --global-batch 64 --warmup 10 \
    #     $flag > "$LOG" 2>&1

    # Parse the timing summary printed by train_mpi at end of run.
    # Expected format:
    #   avg compute / iter:    16.479 ms
    #   avg all-reduce/iter:   46.059 ms  (70.2% of iter)
    #   avg total / iter:      65.649 ms
    local comp comm iter pct
    comp=$(awk '/avg compute/    { print $5 }' "$LOG")
    comm=$(awk '/avg all-reduce/ { print $3 }' "$LOG")
    iter=$(awk '/avg total/      { print $5 }' "$LOG")
    pct=$( awk '/avg all-reduce/ { gsub(/[()%]/,"",$5); print $5 }' "$LOG")

    if [[ -z "$comp" || -z "$comm" || -z "$iter" ]]; then
        echo "WARN: could not parse timing summary for $config_name/$mode. Log saved to $LOG"
        echo "$config_name,$params,$mode,NA,NA,NA,NA" >> "$OUT"
    else
        echo "$config_name,$params,$mode,$comp,$comm,$iter,$pct" >> "$OUT"
        rm "$LOG"
    fi
}

for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%|*}"
    rest="${cfg#*|}"
    params="${rest%%|*}"
    sizes="${rest##*|}"

    echo "================================================================"
    echo "Running ${name} (${sizes}, ${params} params)"
    echo "================================================================"
    for mode in blocking nonblocking; do
        echo "  --- ${mode} ---"
        run_one "$name" "$params" "$sizes" "$mode"
    done
    echo
done

echo "Done. Results saved to ${OUT}"
echo
echo "CSV contents:"
cat "$OUT"
