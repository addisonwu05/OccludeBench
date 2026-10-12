#!/bin/bash
# Codex (GPT-5.4) as a full agentic harness on the direct images.
# Requires OPENAI_API_KEY in the environment.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"
# Run from the repo root (data and result paths are root-relative)
cd "$(dirname "$0")/../.."
for img in data/combined_output/*_occluded.png; do
  id=$(basename "$img" _occluded.png)

  # Skip if already evaluated
  if [ -f "data/eval_results_coding_agent/${id}.json" ]; then
    echo "Skipping $id (already done)"
    continue
  fi

  is_physical=$(python3 -c "
import json
with open('data/combined_output/annotations.json') as f:
    entries = json.load(f)
entry = next((e for e in entries if e['id'] == '$id'), None)
print(entry.get('physical', False) if entry else False)
")

  if [ "$is_physical" = "True" ]; then
    codex exec --model gpt-5.4 --full-auto --sandbox workspace-write --ephemeral \
      "Look at the image at $img. What is the primary object that is completely occluded in this image? Be as specific as you can. Write your answer to data/eval_results_coding_agent/${id}.json as {\"id\": \"$id\", \"response\": \"<your answer>\"}."
  fi
done