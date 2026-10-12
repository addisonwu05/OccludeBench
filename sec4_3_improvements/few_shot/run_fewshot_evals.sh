#!/bin/bash
# Run from the repo root (data and result paths are root-relative)
cd "$(dirname "$0")/../.."

# ===== MODEL FAMILIES =====
openai_models=("gpt-5.4")
claude_models=("claude-opus-4-6")
gemini_models=("gemini-3.1-pro-preview")
llama_models=("cocoscilab/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c")
qwen_models=("cocoscilab/Qwen/Qwen3-VL-8B-Instruct-6dd00d41")

# ===== TASK FUNCTIONS =====
run_model() {
    local model="$1"
    echo "Running few-shot: $model"
    python -u sec4_3_improvements/few_shot/eval_fewshot.py --model "$model"
}

run_family_serially() {
    local models=("$@")
    for model in "${models[@]}"; do
        run_model "$model"
    done
}

# ===== PARALLEL EXECUTION: One process per family =====
(run_family_serially "${openai_models[@]}") &
(run_family_serially "${claude_models[@]}") &
(run_family_serially "${gemini_models[@]}") &
(run_family_serially "${llama_models[@]}") &
(run_family_serially "${qwen_models[@]}") &

wait

echo "All few-shot evals complete."

# ===== JUDGE EVALS =====
echo "Running judge evals..."
python -u sec3_occludebench/judger_script.py \
    --results_dir "data/eval_results_fewshot" \
    --output_dir "data/eval_results_fewshot" \
    --judge_models gpt-5.4 claude-opus-4-6

echo "All judge evals complete."
