#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for m in "gpt-5.4" "claude-opus-4-6" "gemini-3.1-pro-preview"; do
  echo "=== reverse map eval: $m ==="
  python -u reverse_map_eval.py --model "$m" &
done
wait
echo "All reverse map evals complete."
