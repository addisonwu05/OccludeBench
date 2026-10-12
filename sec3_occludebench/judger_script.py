import argparse
import json
import os
import re
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")

JUDGE_CORRECT_PROMPT = (
    "You are evaluating whether a vision model correctly identified an occluded object.\n\n"
    "Ground truth: \"{ground_truth}\"\n\n"
    "Model response: \"{response}\"\n\n"
    "Does the model's response correctly identify the same object as the ground truth? "
    "Allow for reasonable variation in phrasing, specificity, or synonyms "
    "(e.g., 'barber pole' matches 'barber's pole', 'wing mirror' matches 'side mirror', "
    "'cat' matches 'pet' if the ground truth is 'Reasonable Pet'). "
    "The response should identify the core object — partial credit guesses that get the "
    "category right but miss specifics should be marked correct only if the category-level "
    "answer is what the ground truth describes. "
    "Exception: if the ground truth explicitly refers to a reflection or shadow of an object "
    "(e.g., 'pen shadow', 'reflection of a car'), the response must identify both the shadow/reflection "
    "nature AND the underlying object to be marked correct — identifying only the object (e.g., 'pen') "
    "is not sufficient. "
    "Answer only 'yes' or 'no'."
)


def judge_correct(ground_truth, response, judge_model, clients):
    prompt = JUDGE_CORRECT_PROMPT.format(
        ground_truth=ground_truth.strip(),
        response=response.strip()
    )
    if judge_model.startswith("claude"):
        verdict = clients["anthropic"].messages.create(
            model=judge_model,
            temperature=0,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text.strip().lower()
    else:
        verdict = clients["openai"].chat.completions.create(
            model=judge_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content.strip().lower()
    return verdict.startswith("yes")


def get_effective_response(entry):
    """Pick the best response from an occluded-eval entry."""
    for key in ("mask_followup_response", "followup_response", "response"):
        val = entry.get(key)
        if val:
            return val, key
    return "", "response"


def discover_model_stems(results_dir):
    """Return all unique model stems found in the results directory."""
    stems = set()
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json") or fname.endswith("_judged_results.json"):
            continue
        for suffix in ("_vetted_results.json", "_transparent_results.json", "_results.json"):
            if fname.endswith(suffix):
                stems.add(fname[: -len(suffix)])
                break
    return sorted(stems)


def load_results(results_dir, stem):
    """Load occluded results: prefer vetted, fall back to plain results."""
    vetted_path = os.path.join(results_dir, f"{stem}_vetted_results.json")
    plain_path = os.path.join(results_dir, f"{stem}_results.json")
    if os.path.exists(vetted_path):
        with open(vetted_path) as f:
            return json.load(f), "vetted"
    elif os.path.exists(plain_path):
        with open(plain_path) as f:
            return json.load(f), "plain"
    return None, None


def load_transparent_results(results_dir, stem):
    path = os.path.join(results_dir, f"{stem}_transparent_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_judger(args):
    clients = {
        "openai": OpenAI(api_key=OpenAI_API_KEY),
        "anthropic": anthropic.Anthropic(api_key=Anthropic_API_KEY),
    }
    os.makedirs(args.output_dir, exist_ok=True)

    judge_models = args.judge_models

    stems = discover_model_stems(args.results_dir)
    if not stems:
        print("No result files found.")
        return

    print(f"Found models: {', '.join(stems)}\n")
    print(f"Judge models: {', '.join(judge_models)}\n")

    for stem in stems:
        occluded_results, source_type = load_results(args.results_dir, stem)
        transparent_results = load_transparent_results(args.results_dir, stem)

        has_occluded = occluded_results is not None
        has_transparent = transparent_results is not None

        if not has_occluded and not has_transparent:
            print(f"[{stem}] No usable result files found, skipping.")
            continue

        output_path = os.path.join(args.output_dir, f"{stem}_judged_results.json")

        # Load existing judged output for resume support
        if os.path.exists(output_path):
            with open(output_path) as f:
                judged = json.load(f)
            judged_by_id = {r["id"]: r for r in judged}
            print(f"[{stem}] Resuming: {len(judged_by_id)} already judged.")
        else:
            judged_by_id = {}

        # Build lookup for transparent results
        transparent_by_id = {}
        if has_transparent:
            for entry in transparent_results:
                transparent_by_id[entry["id"]] = entry

        # Collect all IDs to judge
        all_ids = set()
        occluded_by_id = {}
        if has_occluded:
            for entry in occluded_results:
                occluded_by_id[entry["id"]] = entry
                all_ids.add(entry["id"])
        all_ids.update(transparent_by_id.keys())

        updated = False
        for entry_id in sorted(all_ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
            existing = judged_by_id.get(entry_id, {"id": entry_id})

            # Ensure verdict dicts exist
            if not isinstance(existing.get("occluded_correct"), dict):
                existing["occluded_correct"] = {}
            if not isinstance(existing.get("transparent_correct"), dict):
                existing["transparent_correct"] = {}

            # Always sync tool_calls from raw results
            if entry_id in occluded_by_id and "tool_calls" in occluded_by_id[entry_id]:
                existing["tool_calls"] = occluded_by_id[entry_id]["tool_calls"]

            # Determine which judge models still need to run
            pending_occ = entry_id in occluded_by_id and [
                jm for jm in judge_models if jm not in existing["occluded_correct"]
            ]
            pending_trans = entry_id in transparent_by_id and [
                jm for jm in judge_models if jm not in existing["transparent_correct"]
            ]

            if not pending_occ and not pending_trans:
                continue

            print(f"[{stem}] Judging {entry_id}...")

            # --- Occluded eval ---
            if pending_occ:
                occ = occluded_by_id[entry_id]
                ground_truth = occ.get("ground_truth", "").strip()
                effective_response, response_key = get_effective_response(occ)

                existing["ground_truth"] = ground_truth
                existing["occluded_source"] = source_type
                existing["occluded_response_used"] = response_key
                existing["occluded_response"] = effective_response

                for jm in pending_occ:
                    if ground_truth and effective_response:
                        correct = judge_correct(ground_truth, effective_response, jm, clients)
                        existing["occluded_correct"][jm] = correct
                        print(f"  occluded ({response_key}) [{jm}]: {'CORRECT' if correct else 'WRONG'}")
                    else:
                        existing["occluded_correct"][jm] = None
                        print(f"  occluded [{jm}]: skipped (missing ground truth or response)")

            # --- Transparent eval ---
            if pending_trans:
                trans = transparent_by_id[entry_id]
                ground_truth = trans.get("ground_truth", "").strip()
                trans_response = trans.get("response", "")

                if "ground_truth" not in existing:
                    existing["ground_truth"] = ground_truth
                existing["transparent_response"] = trans_response

                for jm in pending_trans:
                    if ground_truth and trans_response:
                        correct = judge_correct(ground_truth, trans_response, jm, clients)
                        existing["transparent_correct"][jm] = correct
                        print(f"  transparent [{jm}]: {'CORRECT' if correct else 'WRONG'}")
                    else:
                        existing["transparent_correct"][jm] = None
                        print(f"  transparent [{jm}]: skipped (missing ground truth or response)")

            judged_by_id[entry_id] = existing
            updated = True

            with open(output_path, "w") as f:
                json.dump(list(judged_by_id.values()), f, indent=2)

        if not updated and judged_by_id:
            with open(output_path, "w") as f:
                json.dump(list(judged_by_id.values()), f, indent=2)

        # Summary per judge model
        all_judged = list(judged_by_id.values())
        print(f"\n[{stem}] Summary:")
        for jm in judge_models:
            occ_judged = [r for r in all_judged if isinstance(r.get("occluded_correct"), dict) and r["occluded_correct"].get(jm) is not None]
            trans_judged = [r for r in all_judged if isinstance(r.get("transparent_correct"), dict) and r["transparent_correct"].get(jm) is not None]

            occ_correct = sum(1 for r in occ_judged if r["occluded_correct"][jm])
            trans_correct = sum(1 for r in trans_judged if r["transparent_correct"][jm])

            print(f"  [{jm}]")
            if occ_judged:
                print(f"    Occluded  ({source_type}): {occ_correct}/{len(occ_judged)} correct "
                      f"({100 * occ_correct / len(occ_judged):.1f}%)")
            if trans_judged:
                print(f"    Transparent:              {trans_correct}/{len(trans_judged)} correct "
                      f"({100 * trans_correct / len(trans_judged):.1f}%)")
        print(f"  Output: {output_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge model responses against ground truth for occluded and transparent evals."
    )
    parser.add_argument("--results_dir", type=str, default="data/eval_results",
                        help="Directory containing *_results.json / *_vetted_results.json / *_transparent_results.json files.")
    parser.add_argument("--output_dir", type=str, default="data/eval_results",
                        help="Directory to write *_judged_results.json files.")
    parser.add_argument("--judge_models", type=str, nargs="+", default=["gpt-5.4"],
                        help="One or more OpenAI models to use for judging.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_judger(args)
