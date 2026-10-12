#!/bin/bash
cd "$(dirname "$0")"
# wait until gemini eval reaches 153 (or ask_crops process exits), then judge to completion
for i in $(seq 1 120); do
  ec=$(python3 -c "import json;print(len(json.load(open('crop_eval_gemini-3.1-pro-preview_results.json'))))" 2>/dev/null || echo 0)
  echo "wait $i: gemini eval $ec/153"
  if [ "$ec" -ge 153 ] || ! pgrep -f "ask_crops.py" >/dev/null; then
    echo "eval done (or stopped). Final judging pass..."
    python judge_crops.py
    break
  fi
  sleep 15
done
