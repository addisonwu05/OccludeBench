#!/bin/bash
# Run the crop judger concurrently with the eval: keep judging (resumable) until
# every eval response has a consensus judgement. Tolerant of transient read
# errors caused by the eval writing its results file mid-judge.
cd "$(dirname "$0")"
for i in $(seq 1 60); do
  python judge_crops.py 2>/dev/null || echo "  (transient error, retrying)"
  # done when each judged file count == its eval file count
  remaining=$(python3 - <<'PY'
import json, glob, os
rem=0
for f in glob.glob('crop_eval_*_results.json'):
    if '_judged_' in f: continue
    jf=f.replace('_results.json','_judged_results.json')
    n=len(json.load(open(f)))
    j=len(json.load(open(jf))) if os.path.exists(jf) else 0
    rem += max(0, n-j)
print(rem)
PY
)
  echo "iteration $i: $remaining responses still unjudged"
  [ "$remaining" = "0" ] && { echo "All judged."; break; }
  sleep 15
done
