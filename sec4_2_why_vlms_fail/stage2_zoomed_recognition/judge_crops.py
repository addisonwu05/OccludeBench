"""
Judge the crop-eval responses for correctness (OBJECT-ONLY criterion).

Because the crop prompt already tells the model the region is a shadow/reflection
and asks only for the underlying object, a response is correct if it names the
right underlying object (e.g. 'mountain' matches 'Reflection of a mountain') —
it need NOT restate that it is a shadow/reflection.

2-judge consensus (gpt-5.4 + claude-opus-4-6), matching the physical baseline.
Writes crop_eval_<model>_judged_results.json and prints an accuracy summary.

Usage:
    python judge_crops.py                      # judge all crop_eval_*_results.json
"""
import glob
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))

CLIENTS = {
    "openai": OpenAI(api_key=os.getenv("OpenAI_API_KEY")),
    "anthropic": anthropic.Anthropic(api_key=os.getenv("Anthropic_API_KEY")),
}
JUDGES = ["gpt-5.4", "claude-opus-4-6"]

JUDGE_OBJECT_ONLY = (
    "You are evaluating whether a vision model correctly identified an object.\n\n"
    "The model was shown a cropped region that is a shadow or reflection of some object, "
    "and was asked to name the underlying object.\n\n"
    "Ground truth: \"{ground_truth}\"\n\n"
    "Model response: \"{response}\"\n\n"
    "Does the response correctly identify the same underlying object as the ground truth? "
    "The ground truth may be phrased as 'reflection of a car' or 'car shadow'; the response "
    "only needs to identify the underlying OBJECT (e.g. 'car') and does NOT need to state "
    "that it is a shadow or reflection. Allow reasonable variation in phrasing, specificity, "
    "or synonyms (e.g. 'wing mirror' matches 'side mirror'). "
    "Answer only 'yes' or 'no'."
)


def judge_correct(ground_truth, response, judge_model):
    prompt = JUDGE_OBJECT_ONLY.format(ground_truth=(ground_truth or "").strip(),
                                      response=(response or "").strip())
    if judge_model.startswith("claude"):
        v = CLIENTS["anthropic"].messages.create(
            model=judge_model, temperature=0, max_tokens=10,
            messages=[{"role": "user", "content": prompt}]).content[0].text
    else:
        v = CLIENTS["openai"].chat.completions.create(
            model=judge_model, temperature=0,
            messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    return v.strip().lower().startswith("yes")


def judge_file(path):
    results = json.load(open(path))
    model = os.path.basename(path).replace("crop_eval_", "").replace("_results.json", "")
    out = path.replace("_results.json", "_judged_results.json")
    if os.path.exists(out):
        judged = json.load(open(out))
        by_key = {r["crop_file"]: r for r in judged}
    else:
        judged, by_key = [], {}

    for r in results:
        rec = by_key.get(r["crop_file"])
        if rec is None:
            rec = dict(r); rec["correct"] = {}
            judged.append(rec); by_key[r["crop_file"]] = rec
        rec.setdefault("correct", {})
        for jm in JUDGES:
            if jm in rec["correct"]:
                continue
            rec["correct"][jm] = judge_correct(r["ground_truth"], r["response"], jm)
        votes = list(rec["correct"].values())
        rec["consensus_correct"] = bool(votes) and any(votes)
        with open(out, "w") as f:
            json.dump(judged, f, indent=2)

    n = len(judged)
    corr = sum(1 for r in judged if r.get("consensus_correct"))
    print(f"{model:26} crop accuracy (object-only, consensus): {corr}/{n} = {corr/n:.1%}")
    return model, {r["id"]: r.get("consensus_correct") for r in judged}


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(HERE, "crop_eval_*_results.json")))
    files = [f for f in files if "_judged_" not in f]
    print(f"Judging {len(files)} crop-eval files with {JUDGES} (object-only)...\n")
    per_model = {}
    for f in files:
        m, acc = judge_file(f)
        per_model[m] = acc
    print("\nDone. Per-model judged files written as crop_eval_<model>_judged_results.json")
