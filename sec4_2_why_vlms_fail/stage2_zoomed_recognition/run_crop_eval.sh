#!/bin/bash
# Mass eval over the manually-saved crops, all 3 models (run AFTER cropping).
set -euo pipefail
cd "$(dirname "$0")"
for model in "gpt-5.4" "claude-opus-4-6" "gemini-3.1-pro-preview"; do
  echo "=== crop eval: $model ==="
  python -u ask_crops.py --model "$model" &
done
wait
echo "All crop evals complete."
