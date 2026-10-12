#!/bin/bash
# Judge the stage-2 (confront) answers with the same judges used for the baseline,
# so the numbers drop straight into the existing comparison.
set -euo pipefail
cd "$(dirname "$0")"

python -u ../../sec3_occludebench/judger_script.py \
    --results_dir . \
    --output_dir . \
    --judge_models gpt-5.4 claude-opus-4-6
