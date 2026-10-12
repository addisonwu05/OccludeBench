#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for m in "gpt-5.4" "claude-opus-4-6" "gemini-3.1-pro-preview"; do
  echo "=== identify-highlight eval: $m ==="
  python -u identify_highlight_eval.py --model "$m" &
done
wait
echo "All identify-highlight evals complete."
