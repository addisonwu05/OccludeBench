"""
Analysis for the inpaint-then-confront eval.

Every item in this experiment was WRONG at baseline by construction, so the
headline number is a recovery rate: of the previously-wrong physical items, how
many does the model get right after being shown its own infill and asked whether
it is physically coherent.

Reads the judged output of run_judge.sh and prints a per-model summary plus the
list of recovered items.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "gpt":    "gpt-5.4",
    "gemini": "gemini-3.1-pro-preview",
}


def verdict(entry, judge=None):
    """True/False/None for an item, majority-free: `judge` picks one, else 'any judge'."""
    verdicts = entry.get("occluded_correct") or {}
    if judge:
        return verdicts.get(judge)
    vals = [v for v in verdicts.values() if v is not None]
    if not vals:
        return None
    return any(vals)


def main():
    for provider, model in MODELS.items():
        judged_path = os.path.join(HERE, f"{model}_judged_results.json")
        if not os.path.exists(judged_path):
            print(f"[{model}] no judged results yet — run ./run_judge.sh first.\n")
            continue

        with open(judged_path) as f:
            judged = json.load(f)
        with open(os.path.join(HERE, f"{model}_results.json")) as f:
            raw = {r["id"]: r for r in json.load(f)}

        scored = [e for e in judged if verdict(e) is not None]
        recovered = [e for e in scored if verdict(e)]

        print("=" * 72)
        print(f"{model}  —  physical items wrong at baseline: {len(scored)}")
        print(f"  recovered after confront : {len(recovered)}/{len(scored)} "
              f"({100 * len(recovered) / len(scored):.1f}%)" if scored else "  no scored items")

        # Per-judge breakdown, so the number is not an artifact of 'any judge'.
        judges = sorted({j for e in scored for j in (e.get("occluded_correct") or {})})
        for j in judges:
            vals = [verdict(e, j) for e in scored]
            vals = [v for v in vals if v is not None]
            if vals:
                print(f"    [{j}] {sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.1f}%)")

        if recovered:
            print("\n  Recovered items:")
            for e in recovered:
                r = raw.get(e["id"], {})
                print(f"    {e['id']:<28} GT: {e.get('ground_truth','')}")
                print(f"      baseline : {(r.get('baseline_response') or '')[:110]}")
                print(f"      confront : {(e.get('occluded_response') or '')[:110]}")
        print()


if __name__ == "__main__":
    main()
