#!/bin/bash
# Robustness check: original occluded-object task, but occluders are blue-filled
# with a yellow border. Runs the 3 frontier models, then judges with the same
# two judges used for the main results (gpt-5.4 + claude-opus-4-6).
set -e
cd "$(dirname "$0")/../.."

DATA_DIR="data/combined_output_blue"
OUT_DIR="data/eval_results_blue"
mkdir -p "$OUT_DIR"

models=("gpt-5.4" "claude-opus-4-6" "gemini-3.1-pro-preview")

echo "=== Running evals (blue occluders) ==="
for m in "${models[@]}"; do
    echo ">>> $m"
    python -u sec3_occludebench/eval_script.py --model "$m" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR"
done

echo "=== Judging ==="
python -u sec3_occludebench/judger_script.py \
    --results_dir "$OUT_DIR" \
    --output_dir "$OUT_DIR" \
    --judge_models gpt-5.4 claude-opus-4-6

echo "=== Blue robustness run complete ==="
