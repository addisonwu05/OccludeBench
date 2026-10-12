#!/bin/bash
set -e
# Run from the repo root (data and result paths are root-relative)
cd "$(dirname "$0")/../.."

BASE_DIR="data/eval_results_occlusion_nudge"
THINKING_DIR="data/eval_results_thinking_occlusion_nudge"
TOOL_DIR="data/eval_results_tool_use_thinking_occlusion_nudge"

# echo "=========================================="
# echo "OCCLUSION NUDGE EVALS — base (all models)"
# echo "=========================================="

# # run_base_model() {
# #     local model="$1"
# #     echo "Running occluded (occlusion nudge): $model"
# #     python -u sec3_occludebench/eval_script.py --model "$model" --nudge occlusion --output_dir "$BASE_DIR"
# # }

# #(run_base_model "gpt-5.4") &
# #(run_base_model "claude-opus-4-6") &
# #(run_base_model "gemini-3.1-pro-preview") &
# #(run_base_model "cocoscilab/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c") &
# #(run_base_model "cocoscilab/Qwen/Qwen3-VL-8B-Instruct-6dd00d41") &

# wait

echo ""
echo "=========================================="
echo "OCCLUSION NUDGE EVALS — thinking (GPT-5.4 + Claude)"
echo "=========================================="

(python -u sec4_3_improvements/extended_reasoning/eval_script_thinking.py --model gpt-5.4 --nudge occlusion --output_dir "$THINKING_DIR") &
(python -u sec4_3_improvements/extended_reasoning/eval_script_thinking.py --model claude-opus-4-6 --nudge occlusion --output_dir "$THINKING_DIR") &

wait

echo ""
echo "=========================================="
echo "OCCLUSION NUDGE EVALS — tool use (GPT-5.4 + Claude)"
echo "=========================================="

(python -u sec4_3_improvements/tool_use/eval_tool_use_thinking.py --model gpt-5.4 --nudge occlusion --output_dir "$TOOL_DIR") &
(python -u sec4_3_improvements/tool_use/eval_tool_use_thinking.py --model claude-opus-4-6 --nudge occlusion --output_dir "$TOOL_DIR") &

wait

echo ""
echo "=========================================="
echo "JUDGING"
echo "=========================================="

#python -u sec3_occludebench/judger_script.py --results_dir "$BASE_DIR"     --output_dir "$BASE_DIR"     --judge_models gpt-5.4 claude-opus-4-6
python -u sec3_occludebench/judger_script.py --results_dir "$THINKING_DIR" --output_dir "$THINKING_DIR" --judge_models gpt-5.4 claude-opus-4-6
python -u sec3_occludebench/judger_script.py --results_dir "$TOOL_DIR"     --output_dir "$TOOL_DIR"     --judge_models gpt-5.4 claude-opus-4-6

echo ""
echo "=========================================="
echo "ALL DONE"
#echo "  Base:     $BASE_DIR/"
echo "  Thinking: $THINKING_DIR/"
echo "  Tool use: $TOOL_DIR/"
echo "=========================================="
