import argparse
import json
import os
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")

JUDGE_CORRECT_PROMPT = (
    "You are evaluating whether a human participant correctly identified an occluded object.\n\n"
    "Ground truth: \"{ground_truth}\"\n\n"
    "Human response: \"{response}\"\n\n"
    "Does the human's response correctly identify the same object as the ground truth? "
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


def run_judger(args):
    clients = {
        "openai": OpenAI(api_key=OpenAI_API_KEY),
        "anthropic": anthropic.Anthropic(api_key=Anthropic_API_KEY),
    }

    judge_models = args.judge_models
    input_path = args.input
    output_path = args.output

    with open(input_path) as f:
        entries = json.load(f)

    # Resume support: load existing judged output
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_judged = json.load(f)
        judged_by_id = {r["id"]: r for r in existing_judged}
        print(f"Resuming: {len(judged_by_id)} already judged.")
    else:
        judged_by_id = {}

    print(f"Total entries: {len(entries)}")
    print(f"Judge models: {', '.join(judge_models)}\n")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    for entry in entries:
        entry_id = entry["id"]
        ground_truth = str(entry.get("ground_truth") or "").strip()
        response = str(entry.get("response") or "").strip()

        existing = judged_by_id.get(entry_id, {"id": entry_id})

        # Migrate old format: occluded_correct -> correct
        if "occluded_correct" in existing and "correct" not in existing:
            existing["correct"] = existing.pop("occluded_correct")

        # Carry over metadata fields
        for field in ("image_id", "prolific_pid", "is_physical", "response_time_ms", "ground_truth", "response"):
            if field in entry:
                existing[field] = entry[field]

        if not isinstance(existing.get("correct"), dict):
            existing["correct"] = {}

        pending = [jm for jm in judge_models if jm not in existing["correct"]]
        if not pending:
            judged_by_id[entry_id] = existing
            continue

        print(f"Judging {entry_id} ...")

        for jm in pending:
            if ground_truth and response:
                correct = judge_correct(ground_truth, response, jm, clients)
                existing["correct"][jm] = correct
                print(f"  [{jm}]: {'CORRECT' if correct else 'WRONG'}  |  gt={ground_truth!r}  resp={response!r}")
            else:
                existing["correct"][jm] = None
                print(f"  [{jm}]: skipped (missing ground truth or response)")

        # Consensus: True only if all judge models agree correct
        verdicts = [v for v in existing["correct"].values() if v is not None]
        existing["consensus_correct"] = all(verdicts) if verdicts else None

        judged_by_id[entry_id] = existing

        with open(output_path, "w") as f:
            json.dump(list(judged_by_id.values()), f, indent=2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(list(judged_by_id.values()), f, indent=2)

    # Summary
    all_judged = list(judged_by_id.values())
    print(f"\n===== Summary =====")
    for jm in judge_models:
        judged = [r for r in all_judged if isinstance(r.get("correct"), dict) and r["correct"].get(jm) is not None]
        correct = sum(1 for r in judged if r["correct"][jm])
        print(f"[{jm}]: {correct}/{len(judged)} correct ({100 * correct / len(judged):.1f}%)" if judged else f"[{jm}]: no results")

    consensus = [r for r in all_judged if r.get("consensus_correct") is not None]
    consensus_correct = sum(1 for r in consensus if r["consensus_correct"])
    print(f"[consensus]: {consensus_correct}/{len(consensus)} correct ({100 * consensus_correct / len(consensus):.1f}%)" if consensus else "[consensus]: no results")

    # Per-image accuracy (consensus)
    image_scores = {}
    for r in all_judged:
        img_id = r.get("image_id", "unknown")
        if img_id not in image_scores:
            image_scores[img_id] = {"correct": 0, "total": 0}
        if r.get("consensus_correct") is not None:
            image_scores[img_id]["total"] += 1
            if r["consensus_correct"]:
                image_scores[img_id]["correct"] += 1

    per_image_accs = [v["correct"] / v["total"] for v in image_scores.values() if v["total"] > 0]
    if per_image_accs:
        avg = sum(per_image_accs) / len(per_image_accs)
        print(f"\nPer-image accuracy (consensus, averaged across images): {avg * 100:.1f}%  ({len(per_image_accs)} images)")

    # Physical vs digital breakdown (consensus)
    for is_phys, label in [(True, "physical"), (False, "digital")]:
        subset = [r for r in all_judged if r.get("is_physical") == is_phys and r.get("consensus_correct") is not None]
        if subset:
            n_correct = sum(1 for r in subset if r["consensus_correct"])
            print(f"[{label}] consensus: {n_correct}/{len(subset)} correct ({100 * n_correct / len(subset):.1f}%)")

    print(f"\nOutput saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge human prolific baseline responses against ground truth."
    )
    parser.add_argument(
        "--input", type=str, default="data/eval_results_human/human_prolific_results.json",
        help="Path to the converted human baseline results JSON."
    )
    parser.add_argument(
        "--output", type=str, default="data/eval_results_human/human_prolific_judged_results.json",
        help="Path to write judged results JSON."
    )
    parser.add_argument(
        "--judge_models", type=str, nargs="+", default=["gpt-5.4", "claude-opus-4-6"],
        help="Judge models to use (consensus requires all to agree)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_judger(args)
