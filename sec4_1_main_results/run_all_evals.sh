#!/bin/bash
# Run from the repo root (data and result paths are root-relative)
cd "$(dirname "$0")/.."

# ===== MODEL FAMILIES =====
openai_models=("gpt-5.4")
claude_models=("claude-opus-4-6")
gemini_models=("gemini-3.1-pro-preview")
llama_models=("cocoscilab/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c")
qwen_models=("cocoscilab/Qwen/Qwen3-VL-8B-Instruct-6dd00d41")

# ===== TASK FUNCTIONS (per single model) =====
run_model() {
    local model="$1"
    echo "Running: $model"
    python -u sec3_occludebench/eval_script.py --model "$model"
}

run_model_transparent() {
    local model="$1"
    echo "Running transparent: $model"
    python -u sec4_1_main_results/localization_unmasked/eval_transparent.py --model "$model"
}

run_all_tasks_for_model() {
    local model="$1"
    run_model "$model"
    run_model_transparent "$model"
}

# ===== EXECUTION HELPERS =====
run_family_serially() {
    local models=("$@")
    for model in "${models[@]}"; do
        run_all_tasks_for_model "$model"
    done
}

# ===== PARALLEL EXECUTION: One process per family =====
(run_family_serially "${openai_models[@]}") &
(run_family_serially "${claude_models[@]}") &
(run_family_serially "${gemini_models[@]}") &
(run_family_serially "${llama_models[@]}") &
(run_family_serially "${qwen_models[@]}") &

wait

echo "All evals complete."
