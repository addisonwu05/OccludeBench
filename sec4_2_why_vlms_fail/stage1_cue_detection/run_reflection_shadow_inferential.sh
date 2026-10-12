#!/bin/bash
# REBUTTAL control: reflection/shadow-presence question on the INFERENTIAL images
# (both physical and non-physical). Expected answer is mostly "no" if these are
# truly solved by context rather than a direct reflection/shadow cue.
# 3 models (Qwen endpoint down, Llama excluded).
set -euo pipefail
cd "$(dirname "$0")"

for model in "gpt-5.4" "claude-opus-4-6" "gemini-3.1-pro-preview"; do
  echo "Running (inferential): $model"
  python -u eval_reflection_shadow.py --model "$model" --select inferential &
done
wait
echo "All REBUTTAL inferential reflection/shadow evals complete."
