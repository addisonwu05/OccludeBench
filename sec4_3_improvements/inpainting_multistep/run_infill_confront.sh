#!/bin/bash
# REBUTTAL: inpaint-then-confront, gpt-5.4 and gemini-3.1-pro-preview.
# Subset: physical=True items the model got WRONG on the baseline occluded task.
# Stage 1 regenerates the infill; stage 2 confronts the model with it in a fresh call.
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${1:-all}"   # generate | confront | all

run_one() {
    echo "Running: $1 (stage: $STAGE)"
    python -u infill_confront_eval.py --provider "$1" --stage "$STAGE"
}

(run_one gpt    > gpt.log    2>&1) &
(run_one gemini > gemini.log 2>&1) &

wait
echo "Infill-confront eval complete. Logs: gpt.log, gemini.log"
