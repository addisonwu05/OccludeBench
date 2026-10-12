"""
Retry the infills the original eval_image_gen.py run never produced.

That run caught exceptions and continued without recording anything, so there is
no record of why these failed — transient API error vs a permanent refusal like
IMAGE_RECITATION. This retries them and logs the outcome per item.

Writes PNGs to sec4_3_improvements/inpainting_multistep/generated_retry/<provider>/ rather than
into data/eval_results_image_gen/, so the original run's artifacts are left untouched.
Outcomes go to retry_outcomes.json.
"""
import json
import os

from google import genai
from google.genai import types
from openai import OpenAI

import infill_confront_eval as m

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "generated_retry")
OUTCOMES = os.path.join(HERE, "retry_outcomes.json")

# Items with physical=True that the original run never produced an infill for.
MISSING = {
    "gpt":    ["img1070"],
    "gemini": ["img1018", "img1019", "img1026", "img1027",
               "img1041", "img1066", "img1093", "img1099"],
}


def generate_gemini_verbose(client, image_bytes):
    """Like generate_gemini but returns (bytes|None, finish_reason)."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model=m.GEN_MODEL["gemini"],
        contents=[image_part, m.GEN_PROMPT],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    cand = response.candidates[0]
    reason = str(cand.finish_reason)
    if cand.content is None or cand.content.parts is None:
        return None, reason
    for part in cand.content.parts:
        if part.inline_data is not None:
            return part.inline_data.data, reason
    return None, reason


def main():
    ann = {str(a["id"]): a for a in json.load(open(m.ANNOTATIONS_PATH))}
    outcomes = json.load(open(OUTCOMES)) if os.path.exists(OUTCOMES) else []
    done = {(o["provider"], o["id"]) for o in outcomes if o["status"] == "generated"}

    for provider, ids in MISSING.items():
        out_dir = os.path.join(OUT_ROOT, provider)
        os.makedirs(out_dir, exist_ok=True)

        if provider == "gemini":
            client = genai.Client(api_key=m.Gemini_API_KEY)
        else:
            client = OpenAI(api_key=m.OpenAI_API_KEY)

        for item_id in ids:
            if (provider, item_id) in done:
                print(f"[{provider}] {item_id}: already retried OK, skipping")
                continue
            entry = ann.get(item_id)
            if entry is None:
                print(f"[{provider}] {item_id}: not in annotations, skipping")
                continue

            occluded = os.path.join(
                m.DATA_DIR,
                os.path.splitext(entry["original_file"])[0] + "_occluded.png",
            )
            img = m.encode_image_bytes(occluded)

            status, reason, err = "failed", None, None
            data = None
            try:
                if provider == "gemini":
                    data, reason = generate_gemini_verbose(client, img)
                else:
                    data = m.generate_gpt(client, img)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            if data:
                with open(os.path.join(out_dir, f"{item_id}_generated.png"), "wb") as f:
                    f.write(data)
                status = "generated"

            outcomes = [o for o in outcomes
                        if not (o["provider"] == provider and o["id"] == item_id)]
            outcomes.append({
                "provider": provider,
                "id": item_id,
                "ground_truth": entry.get("answer", ""),
                "status": status,
                "finish_reason": reason,
                "error": err,
            })
            with open(OUTCOMES, "w") as f:
                json.dump(outcomes, f, indent=2)

            print(f"[{provider}] {item_id}: {status}"
                  f"{f' (finish_reason={reason})' if reason else ''}"
                  f"{f' [{err}]' if err else ''}")

    print("\n--- summary ---")
    for provider in MISSING:
        got = [o for o in outcomes if o["provider"] == provider and o["status"] == "generated"]
        bad = [o for o in outcomes if o["provider"] == provider and o["status"] != "generated"]
        print(f"{provider}: recovered {len(got)}/{len(MISSING[provider])}")
        for o in bad:
            print(f"   still missing {o['id']}: "
                  f"{o.get('finish_reason') or o.get('error')}")


if __name__ == "__main__":
    main()
