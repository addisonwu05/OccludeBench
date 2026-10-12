"""
Classify each stage-2 confront response along two axes that the correctness
judge does not capture:

  physics_verdict : did the model judge its own infill physically coherent?
                    plausible | implausible | mixed
  names_occluder  : did its final answer name the mask/annotation/censor box
                    itself rather than a real hidden object?

Cross-tabbing physics_verdict against the correctness verdict is the point: it
tests whether "this infill violates reflection/shadow physics" is what actually
drives a model to a better answer, or whether models call their own fills
coherent and still miss the object.

Writes <model>_verdicts.json and prints the cross-tab.
"""
import json
import os

from openai import OpenAI
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))

CLASSIFIER_MODEL = "gpt-5.4"

MODELS = ["gpt-5.4", "gemini-3.1-pro-preview"]

CLASSIFY_PROMPT = (
    "A vision model was shown an image with an occluded region and a generated "
    "infill of that region. It was asked whether the infill made sense physically "
    "given the properties of reflections and shadows, and then asked what the "
    "occluded object is.\n\n"
    "Here is its response:\n\n\"\"\"{response}\"\"\"\n\n"
    "Answer with a JSON object only, no other text, with exactly these keys:\n"
    "  \"physics_verdict\": one of \"plausible\", \"implausible\", \"mixed\" — "
    "what the model concluded about whether the infill is physically coherent. "
    "Use \"mixed\" if it said partly coherent, was unclear, or did not commit.\n"
    "  \"names_occluder\": true or false — true if the model's FINAL answer about "
    "the occluded object identifies the mask/occluder itself (a black box, red "
    "region, censor bar, annotation rectangle, editing overlay) rather than a real "
    "object in the depicted scene. False if it names a real-world object.\n"
)


def classify(client, response):
    out = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(response=response)}],
    ).choices[0].message.content.strip()
    if out.startswith("```"):
        out = out.split("```")[1].lstrip("json").strip()
    return json.loads(out)


def correctness(entry):
    vals = [v for v in (entry.get("occluded_correct") or {}).values() if v is not None]
    return any(vals) if vals else None


def main():
    client = OpenAI(api_key=os.getenv("OpenAI_API_KEY"))

    for model in MODELS:
        results_path = os.path.join(HERE, f"{model}_results.json")
        judged_path = os.path.join(HERE, f"{model}_judged_results.json")
        out_path = os.path.join(HERE, f"{model}_verdicts.json")

        with open(results_path) as f:
            results = json.load(f)
        with open(judged_path) as f:
            correct_by_id = {e["id"]: correctness(e) for e in json.load(f)}

        if os.path.exists(out_path):
            with open(out_path) as f:
                verdicts = json.load(f)
            done = {v["id"] for v in verdicts}
        else:
            verdicts, done = [], set()

        for r in results:
            if r["id"] in done:
                continue
            try:
                c = classify(client, r["response"])
            except Exception as e:
                print(f"  [ERROR] {r['id']}: {e}")
                continue
            verdicts.append({
                "id": r["id"],
                "ground_truth": r["ground_truth"],
                "physics_verdict": c.get("physics_verdict"),
                "names_occluder": c.get("names_occluder"),
                "correct": correct_by_id.get(r["id"]),
            })
            with open(out_path, "w") as f:
                json.dump(verdicts, f, indent=2)
            print(f"[{model}] {r['id']}: {c.get('physics_verdict')}, "
                  f"names_occluder={c.get('names_occluder')}, correct={correct_by_id.get(r['id'])}")

        # ---- cross-tab ----
        print("\n" + "=" * 72)
        print(f"{model}  (n={len(verdicts)})")
        drift = [v for v in verdicts if v["names_occluder"]]
        print(f"  answered with the mask/occluder instead of a real object: "
              f"{len(drift)}/{len(verdicts)} ({100*len(drift)/len(verdicts):.1f}%)")
        print(f"\n  physics verdict x correctness:")
        print(f"    {'verdict':<14}{'n':>5}{'correct':>10}{'rate':>9}")
        for pv in ["plausible", "implausible", "mixed"]:
            grp = [v for v in verdicts if v["physics_verdict"] == pv]
            if not grp:
                continue
            n_ok = sum(1 for v in grp if v["correct"])
            print(f"    {pv:<14}{len(grp):>5}{n_ok:>10}{100*n_ok/len(grp):>8.1f}%")
        print()


if __name__ == "__main__":
    main()
