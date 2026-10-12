"""
For each model's wrong_transparent results, judge whether the ground truth object
is mentioned/present in the model's scene description.
"""

import argparse
import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")

JUDGE_PROMPT = (
    "A vision model was given an image and produced the following scene description:\n\n"
    "\"{description}\"\n\n"
    "The ground truth answer for the highlighted object in that image is: \"{ground_truth}\"\n\n"
    "Is the ground truth object mentioned or present as an object in the model's description? "
    "Answer only 'yes' or 'no'."
)

EVAL_MODELS = [
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
    "Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c",
    "Qwen3-VL-8B-Instruct",
]


def judge(description, ground_truth, judge_model, client):
    prompt = JUDGE_PROMPT.format(description=description, ground_truth=ground_truth)
    verdict = client.chat.completions.create(
        model=judge_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().lower()
    return verdict.startswith("yes")


def run(args):
    client = OpenAI(api_key=OpenAI_API_KEY)
    os.makedirs(args.output_dir, exist_ok=True)

    models = [args.model] if args.model else EVAL_MODELS

    for model in models:
        input_path = os.path.join(args.input_dir, f"{model}_wrong_transparent.json")
        if not os.path.exists(input_path):
            print(f"[SKIP] No results for {model}")
            continue

        output_path = os.path.join(args.output_dir, f"{model}_wrong_transparent_judged.json")

        if os.path.exists(output_path):
            with open(output_path) as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
        else:
            results = []
            done_ids = set()

        with open(input_path) as f:
            entries = json.load(f)

        remaining = [e for e in entries if e["id"] not in done_ids]
        print(f"\n[{model}] Total: {len(entries)} | Done: {len(done_ids)} | Remaining: {len(remaining)}")

        for entry in remaining:
            print(f"  id={entry['id']} gt={entry['ground_truth']!r} ...", end=" ", flush=True)
            try:
                mentioned = judge(
                    entry["transparent_description"],
                    entry["ground_truth"],
                    args.judge_model,
                    client
                )
                print("yes" if mentioned else "no")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            results.append({
                "id": entry["id"],
                "ground_truth": entry["ground_truth"],
                "ground_truth_mentioned": mentioned,
                "transparent_description": entry["transparent_description"],
                "occluded_response": entry.get("occluded_response"),
            })

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        # Summary
        if results:
            n_mentioned = sum(1 for r in results if r["ground_truth_mentioned"])
            print(f"  Ground truth mentioned: {n_mentioned}/{len(results)} ({100*n_mentioned/len(results):.1f}%)")
        print(f"  Saved -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Run a single model (default: all)")
    parser.add_argument("--input_dir", default="data/eval_results_wrong_transparent")
    parser.add_argument("--output_dir", default="data/eval_results_wrong_transparent")
    parser.add_argument("--judge_model", default="gpt-5.4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
