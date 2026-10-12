#!/bin/bash
# Cue removal: the base occluded query on direct images whose shadow/reflection
# was removed in Photoshop (data/physical_occluded_combined/), all five models.
set -e
# Run from the repo root (data and result paths are root-relative)
cd "$(dirname "$0")/../.."

DATA_DIR="data/physical_occluded_combined"
OUT_DIR="data/eval_results_physical"

models=(
    "gpt-5.4"
    "claude-opus-4-6"
    "gemini-3.1-pro-preview"
    "cocoscilab/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c"
    "cocoscilab/Qwen/Qwen3-VL-8B-Instruct-6dd00d41"
)

for m in "${models[@]}"; do
    (python -u sec3_occludebench/eval_script.py --model "$m" --data_dir "$DATA_DIR" --output_dir "$OUT_DIR") &
done
wait

python -u sec3_occludebench/judger_script.py \
    --results_dir "$OUT_DIR" \
    --output_dir "$OUT_DIR" \
    --judge_models gpt-5.4 claude-opus-4-6
