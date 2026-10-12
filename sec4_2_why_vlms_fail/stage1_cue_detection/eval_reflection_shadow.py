"""
REBUTTAL eval: reflection/shadow presence on the PHYSICAL occluded images.

Asks each model a single yes/no-style question about every physical=True entry
in annotations.json, using the occluded PNG from data/combined_output/.

Prompt (verbatim, as requested):
    "Is there the presence of a reflection or a shadow in this image?"

Models run: gpt-5.4, claude-opus-4-6, gemini-3.1-pro-preview, Qwen3-VL-8B.
(Llama-4-Maverick intentionally excluded for now.)

Results are written next to this script as <model>_reflection_shadow_results.json.
Adapted from sec3_occludebench/eval_script.py (client init + image encoding + chat_completion).
"""
import argparse
import base64
import io
import json
import os

from openai import OpenAI
from together import Together
import anthropic
from dotenv import load_dotenv
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")
Gemini_API_KEY = os.getenv("Gemini_API_KEY")
Together_API_KEY = os.getenv("Together_API_KEY")

PROMPT = "Is there the presence of a reflection or a shadow in this image?"


def init_client(model):
    if "gpt" in model or model[0] == "o":
        return OpenAI(api_key=OpenAI_API_KEY)
    elif "gemini" in model:
        return OpenAI(
            api_key=Gemini_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif "claude" in model:
        return anthropic.Anthropic(api_key=Anthropic_API_KEY)
    else:
        return Together(api_key=Together_API_KEY)


def encode_image(image_path, max_bytes=3.5 * 1024 * 1024):
    with open(image_path, "rb") as f:
        data = f.read()
    if len(data) <= max_bytes:
        return base64.b64encode(data).decode("utf-8")
    img = Image.open(io.BytesIO(data))
    scale = 0.9
    while True:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            break
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def chat_completion(model, client, image_path, prompt=PROMPT):
    b64 = encode_image(image_path)
    media_type = "image/png"

    if "claude" in model:
        return client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        ).content[0].text
    else:
        return client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        ).choices[0].message.content


def select_ids(select, cat_file):
    """Return the set of image ids (as str) to include, or None for 'all'."""
    if select in ("inferential", "direct"):
        import pandas as pd
        x = pd.read_excel(cat_file)
        ids = x[x["direct or inferential"] == select]["image id"]
        return {str(i) for i in ids}
    return None  # physical/all handled via the physical flag


def run_eval(args, client):
    with open(os.path.join(args.data_dir, "annotations.json")) as f:
        entries = json.load(f)

    if args.select == "physical":
        entries = [e for e in entries if e.get("physical", False)]
        tag = "reflection_shadow"
    elif args.select == "all":
        tag = "reflection_shadow_all"
    else:  # inferential or direct
        keep = select_ids(args.select, args.cat_file)
        entries = [e for e in entries if str(e["id"]) in keep]
        tag = f"reflection_shadow_{args.select}"
    print(f"Selection '{args.select}': {len(entries)} candidate entries")

    model_name = args.model.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_name}_{tag}_results.json")

    # Resume support: skip ids already done.
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} already done, "
              f"{len(entries) - len(done_ids)} remaining.")
    else:
        results = []
        done_ids = set()

    os.makedirs(args.output_dir, exist_ok=True)

    for entry in entries:
        if entry["id"] in done_ids:
            continue

        base = os.path.splitext(entry["original_file"])[0]
        occluded_path = os.path.join(args.image_dir, f"{base}_occluded.png")
        if not os.path.exists(occluded_path):
            print(f"  [skip] no occluded image for id={entry['id']} ({base})")
            continue

        print(f"Running {entry['id']} ({base})...")
        response = chat_completion(args.model, client, occluded_path, prompt=PROMPT)
        print(f"  Response: {response}")

        results.append({
            "id": entry["id"],
            "original_file": entry["original_file"],
            "ground_truth": entry.get("answer", ""),
            "prompt": PROMPT,
            "response": response,
        })

        # Save after each entry so progress is never lost.
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results, output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data/combined_output"),
                        help="Directory containing annotations.json.")
    parser.add_argument("--image_dir", type=str, default=os.path.join(ROOT, "data/combined_output"),
                        help="Directory containing the *_occluded.png images.")
    parser.add_argument("--output_dir", type=str, default=HERE,
                        help="Where to write results (this folder).")
    parser.add_argument("--select", type=str, default="physical",
                        choices=["physical", "inferential", "direct", "all"],
                        help="Which image subset to ask the reflection/shadow question about.")
    parser.add_argument("--cat_file", type=str, default=os.path.join(ROOT, "data/Image categories (3).xlsx"),
                        help="Category spreadsheet for direct/inferential selection.")
    return parser.parse_args()


def main(args):
    client = init_client(args.model)
    results, output_path = run_eval(args, client)
    print(f"Done. {len(results)} results in {output_path}")


if __name__ == "__main__":
    main(parse_args())
