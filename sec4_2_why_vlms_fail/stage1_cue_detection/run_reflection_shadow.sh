#!/bin/bash
# REBUTTAL: reflection/shadow-presence question on the physical occluded images.
# Runs one process per model family in parallel. Llama-4-Maverick excluded.
set -euo pipefail
cd "$(dirname "$0")"

openai_model="gpt-5.4"
claude_model="claude-opus-4-6"
gemini_model="gemini-3.1-pro-preview"
qwen_model="cocoscilab/Qwen/Qwen3-VL-8B-Instruct-6dd00d41"

run_one() { echo "Running: $1"; python -u eval_reflection_shadow.py --model "$1"; }

(run_one "$openai_model") &
(run_one "$claude_model") &
(run_one "$gemini_model") &
(run_one "$qwen_model") &

wait
echo "All REBUTTAL reflection/shadow evals complete."
