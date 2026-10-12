#!/bin/bash
set -e
# Run from the repo root (data and result paths are root-relative)
cd "$(dirname "$0")/../.."

OUTPUT_DIR="${OUTPUT_DIR:-data/eval_results_tool_use_thinking}"

echo "=========================================="
echo "RUNNING TOOL USE EVALS — parallelized by model"
echo "=========================================="

# --- GPT-5.4 pipeline ---
(
  python3 sec4_3_improvements/tool_use/eval_tool_use_thinking.py --model gpt-5.4 --output_dir $OUTPUT_DIR
) &

# --- Claude Opus 4.6 pipeline ---
(
  python3 sec4_3_improvements/tool_use/eval_tool_use_thinking.py --model claude-opus-4-6 --output_dir $OUTPUT_DIR
) &

wait

echo ""
echo "=========================================="
echo "JUDGING"
echo "=========================================="

python3 sec3_occludebench/judger_script.py --results_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR --judge_models gpt-5.4 claude-opus-4-6

echo ""
echo "=========================================="
echo "ALL DONE — results in $OUTPUT_DIR/"
echo "=========================================="
