#!/usr/bin/env bash
set -euo pipefail

# Batch runner for self-consistency sweeps on MATH-500
#
# Usage:
#   bash run_batch_self_consistency_math500.sh
#

WORKDIR="${WORKDIR:-$(pwd)}"

OUTDIR="self_consistency_math500_runs"
mkdir -p "$OUTDIR"

MAIN_LOG="$OUTDIR/all_runs.txt"

echo "=== self-consistency batch start: $(date -Is)" | tee "$MAIN_LOG"
echo "=== WORKDIR: $WORKDIR" | tee -a "$MAIN_LOG"
echo "=== OUTDIR:  $OUTDIR" | tee -a "$MAIN_LOG"
echo | tee -a "$MAIN_LOG"

run_sc () {
    local row="$1"; shift
    local tag="$1"; shift

    local outfile="$OUTDIR/${row}__${tag}.txt"

    echo "=== ${row} (${tag})" | tee -a "$MAIN_LOG"
    echo "=== Output -> ${outfile}" | tee -a "$MAIN_LOG"
    echo "Command: uv run self_consistency_math500.py $*" | tee -a "$MAIN_LOG"

    pushd "$WORKDIR" >/dev/null

    PYTHONUNBUFFERED=1 uv run python -u self_consistency_math500.py "$@" \
        2>&1 \
    | awk 'BEGIN{RS="\r|\n"; ORS="\n"} {print; fflush()}' \
    | tee "$outfile" \
    | tee -a "$MAIN_LOG"

    popd >/dev/null

    echo | tee -a "$MAIN_LOG"
}

# -----------------------------------------------------------------------------
# Rows 4-12
# -----------------------------------------------------------------------------

# Row 4
run_sc "row04" "base_t0.9_p0.9_n1" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500

# Row 5
run_sc "row05" "base_t0.9_p0.9_n3" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500

# Row 6
run_sc "row06" "base_t0.9_p0.9_n5" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500

# Row 7
run_sc "row07" "base_t0.9_p0.9_n10" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500

# Row 8
run_sc "row08" "base_t0.9_p0.9_n1_explain" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500 \
    --prompt_suffix "\\n\\nExplain step by step."

# Row 9
run_sc "row09" "base_t0.9_p0.9_n3_explain" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\\n\\nExplain step by step."

# Row 10
run_sc "row10" "base_t0.9_p0.9_n5_explain" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500 \
    --prompt_suffix "\\n\\nExplain step by step."

# Row 11
run_sc "row11" "base_t0.9_p0.9_n10_explain" \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500 \
    --prompt_suffix "\\n\\nExplain step by step."

# Row 12
run_sc "row12" "reasoning_t0.9_p0.9_n3_explain" \
    --which_model "reasoning" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\\n\\nExplain step by step."

echo "=== self-consistency batch done: $(date -Is)" | tee -a "$MAIN_LOG"
