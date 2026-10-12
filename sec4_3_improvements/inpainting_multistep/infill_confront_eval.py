"""
REBUTTAL eval: inpaint-then-confront.

Two stages, run on the physical=True items where the model got the BASELINE
occluded task wrong (no judge in data/eval_results/<model>_judged_results.json marked
occluded_correct):

  Stage 1 (generate)  Give the model the occluded image and ask it to fill the
                      masked region with the hidden object made visible, using
                      the same GEN_PROMPT as appendix/image_generation/eval_image_gen.py.
                      GPT  -> gpt-5.4 with the image_generation tool
                      Gemini -> gemini-3-pro-image-preview
                      By default this REUSES the infills already produced by
                      that earlier partial run (data/eval_results_image_gen/)
                      and only generates what is missing; --no_reuse forces a
                      fresh generation for every item.

  Stage 2 (confront)  FRESH call to the model that the rest of the paper
                      measures (gpt-5.4 / gemini-3.1-pro-preview), with BOTH
                      the occluded original and the model's own infill
                      attached, asking whether the infill is physically
                      coherent and then re-asking for the occluded object.

The fresh call in stage 2 is deliberate: Gemini's generator
(gemini-3-pro-image-preview) is a different model from the one evaluated
everywhere else (gemini-3.1-pro-preview), so an in-thread continuation would
measure a model with no comparable baseline. Both arms use a fresh call so they
stay symmetric.

Output: sec4_3_improvements/inpainting_multistep/<model>_results.json, with `response` holding
the stage-2 answer so sec3_occludebench/judger_script.py can score it unmodified.
Generated images: sec4_3_improvements/inpainting_multistep/generated/<provider>/<id>_generated.png
"""
import argparse
import base64
import io
import json
import os
import shutil

from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

load_dotenv(os.path.join(ROOT, ".env"))
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Gemini_API_KEY = os.getenv("Gemini_API_KEY")

# ---- models -----------------------------------------------------------------
# provider -> (generation model, reasoning model evaluated elsewhere in the paper)
GEN_MODEL = {
    "gpt":    "gpt-5.4",
    "gemini": "gemini-3-pro-image-preview",
}
REASON_MODEL = {
    "gpt":    "gpt-5.4",
    "gemini": "gemini-3.1-pro-preview",
}
BASELINE_JUDGED = {
    "gpt":    os.path.join(ROOT, "data/eval_results", "gpt-5.4_judged_results.json"),
    "gemini": os.path.join(ROOT, "data/eval_results", "gemini-3.1-pro-preview_judged_results.json"),
}

ANNOTATIONS_PATH = os.path.join(ROOT, "data/combined_output", "annotations.json")
DATA_DIR = os.path.join(ROOT, "data/combined_output")

# Infills from the earlier partial run of appendix/image_generation/eval_image_gen.py, which used the
# identical GEN_PROMPT. Reused by default so stage 1 costs 2 generations, not 179.
PRIOR_GEN_DIR = {
    "gpt":    os.path.join(ROOT, "data/eval_results_image_gen", "gpt"),
    "gemini": os.path.join(ROOT, "data/eval_results_image_gen", "gemini"),
}

# Infills recovered by retry_missing_infills.py for items the original run never
# produced. Searched after PRIOR_GEN_DIR.
RETRY_GEN_DIR = {
    "gpt":    os.path.join(HERE, "generated_retry", "gpt"),
    "gemini": os.path.join(HERE, "generated_retry", "gemini"),
}

# ---- prompts ----------------------------------------------------------------
# Stage 1: verbatim from appendix/image_generation/eval_image_gen.py so the inpaint arm is a rerun,
# not a new condition.
GEN_PROMPT = (
    "This image has a black region covering something. "
    "Generate a new version of this image with the hidden object "
    "naturally visible in place of the black region, keeping everything "
    "else in the image exactly the same."
)

# Stage 2: the confrontation.
CONFRONT_PROMPT = (
    "The first image is the original: a region of it is covered by a solid mask, "
    "hiding an object. The second image is a generated version of that image, "
    "where the hidden object has been filled in.\n\n"
    "Does this infill make sense physically, given the properties of reflections "
    "and shadows? Now, what is the occluded object? Be as specific as you can."
)


# ---- item selection ---------------------------------------------------------
def load_qualifying(provider, select="wrong"):
    """
    physical=True items, filtered by baseline outcome.

      select="wrong"  items where NO judge marked the baseline answer correct
                      (the recovery arm)
      select="all"    every physical item, so the post-confront accuracy on the
                      full physical set is measured rather than projected from
                      an assumption that already-correct items stay correct
    """
    with open(BASELINE_JUDGED[provider]) as f:
        judged = json.load(f)
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    phys = {str(a["id"]) for a in annotations if a.get("physical")}
    orig = {str(a["id"]): a["original_file"] for a in annotations}

    items = []
    for it in judged:
        item_id = str(it["id"])
        if item_id not in phys:
            continue
        verdicts = [v for v in (it.get("occluded_correct") or {}).values() if v is not None]
        if not verdicts:
            continue  # unjudged at baseline
        baseline_correct = any(verdicts)
        if select == "wrong" and baseline_correct:
            continue
        items.append({
            "id": item_id,
            "original_file": orig.get(item_id, ""),
            "ground_truth": (it.get("ground_truth") or "").strip(),
            "baseline_correct": baseline_correct,
            "baseline_response": it.get("occluded_response", ""),
        })
    return items


# ---- image helpers ----------------------------------------------------------
def encode_image_bytes(image_path, max_bytes=3.5 * 1024 * 1024):
    with open(image_path, "rb") as f:
        data = f.read()
    if len(data) <= max_bytes:
        return data
    img = Image.open(io.BytesIO(data))
    scale = 0.9
    while True:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def as_data_url(image_bytes):
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")


# ---- stage 1: generate ------------------------------------------------------
def generate_gemini(client, image_bytes):
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model=GEN_MODEL["gemini"],
        contents=[image_part, GEN_PROMPT],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    candidate = response.candidates[0]
    if candidate.content is None or candidate.content.parts is None:
        # e.g. IMAGE_RECITATION — the generator declines to produce the image.
        print(f"  [BLOCKED] no image parts (finish_reason={candidate.finish_reason})")
        return None
    for part in candidate.content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    return None


def generate_gpt(client, image_bytes):
    response = client.responses.create(
        model=GEN_MODEL["gpt"],
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": as_data_url(image_bytes)},
                {"type": "input_text",  "text": GEN_PROMPT},
            ],
        }],
        tools=[{"type": "image_generation"}],
    )
    for item in response.output:
        if item.type == "image_generation_call":
            if not item.result:
                # Tool call present but no image — e.g. the request was refused.
                print(f"  [BLOCKED] image_generation_call returned no result "
                      f"(status={getattr(item, 'status', None)})")
                return None
            return base64.b64decode(item.result)
    return None


def prior_infill_path(provider, item_id):
    """Path to an existing infill for this item: original run first, then retries."""
    for root in (PRIOR_GEN_DIR[provider], RETRY_GEN_DIR[provider]):
        path = os.path.join(root, f"{item_id}_generated.png")
        if os.path.exists(path):
            return path
    return None


def run_generate(provider, out_dir, reuse=True, select="wrong"):
    items = load_qualifying(provider, select)
    gen_dir = os.path.join(out_dir, "generated", provider)
    os.makedirs(gen_dir, exist_ok=True)
    meta_path = os.path.join(gen_dir, "metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        done = {m["id"] for m in metadata}
        print(f"[{provider}/generate] resuming: {len(done)} done, {len(items) - len(done)} left")
    else:
        metadata, done = [], set()

    client = generate_fn = None  # built lazily: reuse may mean zero API calls

    print(f"[{provider}/generate] {len(items)} qualifying items (physical, baseline wrong)"
          f"{' — reusing prior infills where available' if reuse else ' — regenerating all'}")

    for item in items:
        if item["id"] in done:
            continue
        base = os.path.splitext(item["original_file"])[0]
        occluded_path = os.path.join(DATA_DIR, f"{base}_occluded.png")
        if not os.path.exists(occluded_path):
            print(f"  [SKIP] missing {occluded_path}")
            continue

        dest = os.path.join(gen_dir, f"{item['id']}_generated.png")
        prior = prior_infill_path(provider, item["id"]) if reuse else None

        if prior:
            shutil.copyfile(prior, dest)
            source = "reused"
            print(f"[{provider}/generate] {item['id']} reused from partial run")
        else:
            if generate_fn is None:
                if provider == "gemini":
                    client = genai.Client(api_key=Gemini_API_KEY)
                    generate_fn = generate_gemini
                else:
                    client = OpenAI(api_key=OpenAI_API_KEY)
                    generate_fn = generate_gpt
            print(f"[{provider}/generate] {item['id']} generating (GT: {item['ground_truth']})")
            try:
                generated = generate_fn(client, encode_image_bytes(occluded_path))
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue
            if generated is None:
                print(f"  [WARN] no image returned for {item['id']}")
                continue
            with open(dest, "wb") as f:
                f.write(generated)
            source = "generated"

        metadata.append({
            "id": item["id"],
            "ground_truth": item["ground_truth"],
            "original_file": item["original_file"],
            "generated_file": f"{item['id']}_generated.png",
            "infill_source": source,
            "baseline_correct": item["baseline_correct"],
            "baseline_response": item["baseline_response"],
        })
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    n_reused = sum(1 for m in metadata if m.get("infill_source") == "reused")
    print(f"[{provider}/generate] done: {len(metadata)} images in {gen_dir}/ "
          f"({n_reused} reused, {len(metadata) - n_reused} newly generated)")
    return metadata


# ---- stage 2: confront ------------------------------------------------------
def confront_client(provider):
    if provider == "gemini":
        return OpenAI(
            api_key=Gemini_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI(api_key=OpenAI_API_KEY)


def confront(client, provider, occluded_bytes, generated_bytes):
    return client.chat.completions.create(
        model=REASON_MODEL[provider],
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": as_data_url(occluded_bytes)}},
                {"type": "image_url", "image_url": {"url": as_data_url(generated_bytes)}},
                {"type": "text", "text": CONFRONT_PROMPT},
            ],
        }],
    ).choices[0].message.content


def run_confront(provider, out_dir):
    gen_dir = os.path.join(out_dir, "generated", provider)
    meta_path = os.path.join(gen_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[{provider}/confront] no generations yet — run stage 'generate' first.")
        return []
    with open(meta_path) as f:
        metadata = json.load(f)

    # Named <reasoning-model>_results.json so sec3_occludebench/judger_script.py picks it up.
    out_path = os.path.join(out_dir, f"{REASON_MODEL[provider]}_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        done = {r["id"] for r in results}
        print(f"[{provider}/confront] resuming: {len(done)} done, {len(metadata) - len(done)} left")
    else:
        results, done = [], set()

    client = confront_client(provider)
    print(f"[{provider}/confront] {len(metadata)} items, model {REASON_MODEL[provider]}")

    for m in metadata:
        if m["id"] in done:
            continue
        base = os.path.splitext(m["original_file"])[0]
        occluded_path = os.path.join(DATA_DIR, f"{base}_occluded.png")
        generated_path = os.path.join(gen_dir, m["generated_file"])
        if not (os.path.exists(occluded_path) and os.path.exists(generated_path)):
            print(f"  [SKIP] missing image for {m['id']}")
            continue

        print(f"[{provider}/confront] {m['id']} (GT: {m['ground_truth']})")
        try:
            response = confront(
                client, provider,
                encode_image_bytes(occluded_path),
                encode_image_bytes(generated_path),
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        print(f"  Response: {response}")

        results.append({
            "id": m["id"],
            "ground_truth": m["ground_truth"],
            "response": response,            # stage-2 answer; judger_script.py reads this
            "confront_response": response,   # explicit alias for analysis
            "baseline_correct": m.get("baseline_correct"),
            "baseline_response": m.get("baseline_response", ""),
            "generated_file": m["generated_file"],
            "followup_response": None,
        })
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"[{provider}/confront] done: {len(results)} results in {out_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", type=str, default="gpt", choices=["gpt", "gemini"])
    p.add_argument("--stage", type=str, default="all", choices=["generate", "confront", "all"])
    p.add_argument("--output_dir", type=str, default=HERE)
    p.add_argument("--no_reuse", action="store_true",
                   help="Regenerate every infill instead of reusing the earlier partial run.")
    p.add_argument("--select", type=str, default="wrong", choices=["wrong", "all"],
                   help="'wrong': only items the model failed at baseline (recovery arm). "
                        "'all': every physical item, for a measured full-set accuracy.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.stage in ("generate", "all"):
        run_generate(args.provider, args.output_dir, reuse=not args.no_reuse, select=args.select)
    if args.stage in ("confront", "all"):
        run_confront(args.provider, args.output_dir)
