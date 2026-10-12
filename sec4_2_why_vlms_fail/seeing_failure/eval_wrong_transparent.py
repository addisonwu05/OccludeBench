"""
For each model in data/eval_results/, find images where both judges marked occluded as wrong,
filter to only images in the annotated/ folder, then have THAT SAME MODEL describe the
transparent image comprehensively.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

from openai import OpenAI
from together import Together
import anthropic
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")
Gemini_API_KEY = os.getenv("Gemini_API_KEY")
Together_API_KEY = os.getenv("Together_API_KEY")

PROMPT = (
    "This image contains a thick-bordered transparent bounding box (the border may be yellow, red, or green) "
    "highlighting a region of the scene. Provide a comprehensive, detailed description of the entire scene: "
    "describe every object, person, animal, text, surface, background, lighting, colors, textures, "
    "spatial relationships, and any other visual details you can observe. Be exhaustive and precise."
)

EVAL_MODELS = [
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
    "Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c",
    "Qwen3-VL-8B-Instruct",
]

# Map from file stem (used in result filenames) to actual API model name
MODEL_API_NAME = {
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c": "cocoscilab/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c",
}


def init_client(model):
    if "gpt" in model or model[0] == 'o':
        return OpenAI(api_key=OpenAI_API_KEY)
    elif "gemini" in model:
        return OpenAI(api_key=Gemini_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
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


PIL_TO_MEDIA_TYPE = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}

def detect_media_type(image_path):
    with Image.open(image_path) as img:
        return PIL_TO_MEDIA_TYPE.get(img.format, "image/png")

def query_model(model, client, image_path):
    b64 = encode_image(image_path)
    media_type = detect_media_type(image_path)

    if "claude" in model:
        return client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": PROMPT}
                ]
            }]
        ).content[0].text
    else:
        return client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": PROMPT}
                ]
            }]
        ).choices[0].message.content


def get_wrong_annotated_entries(judged_path, annotated_stems):
    """Return entries where all judges said wrong AND image is in annotated/."""
    with open(judged_path) as f:
        data = json.load(f)
    return [
        e for e in data
        if all(v is False for v in e["occluded_correct"].values())
        and e["id"] in annotated_stems
    ]


def run(args):
    # Map stem -> full path for every image in annotated/
    annotated_map = {
        Path(f).stem: os.path.join(args.annotated_dir, f)
        for f in os.listdir(args.annotated_dir)
        if not f.endswith(".json")
    }
    annotated_stems = set(annotated_map.keys())
    print(f"Annotated images: {len(annotated_stems)}")

    os.makedirs(args.output_dir, exist_ok=True)

    models = [args.model] if args.model else EVAL_MODELS
    for model in models:
        judged_path = os.path.join(args.eval_results_dir, f"{model}_judged_results.json")
        if not os.path.exists(judged_path):
            print(f"[SKIP] No judged results for {model}")
            continue

        output_path = os.path.join(args.output_dir, f"{model}_wrong_transparent.json")

        if os.path.exists(output_path):
            with open(output_path) as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
        else:
            results = []
            done_ids = set()

        wrong_entries = get_wrong_annotated_entries(judged_path, annotated_stems)
        remaining = [e for e in wrong_entries if e["id"] not in done_ids]
        print(f"\n[{model}] Wrong & annotated: {len(wrong_entries)} | Done: {len(done_ids)} | Remaining: {len(remaining)}")

        if not remaining:
            continue

        api_model = MODEL_API_NAME.get(model, model)
        client = init_client(api_model)

        for entry in remaining:
            image_path = annotated_map.get(entry["id"])
            if not image_path:
                print(f"  [SKIP] No annotated image for id={entry['id']}")
                continue

            print(f"  id={entry['id']} gt={entry['ground_truth']!r} ...", end=" ", flush=True)
            try:
                response = query_model(api_model, client, image_path)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            results.append({
                "id": entry["id"],
                "ground_truth": entry["ground_truth"],
                "transparent_description": response,
                "occluded_correct": entry["occluded_correct"],
                "occluded_response": entry.get("occluded_response"),
            })

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        print(f"  Saved {len(results)} results -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Run a single model (default: all)")
    parser.add_argument("--annotated_dir", default="data/annotated")
    parser.add_argument("--eval_results_dir", default="data/eval_results")
    parser.add_argument("--output_dir", default="data/eval_results_wrong_transparent")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
