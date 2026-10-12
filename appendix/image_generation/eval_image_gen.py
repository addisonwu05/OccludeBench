import argparse
import base64
import io
import json
import os
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
Gemini_API_KEY = os.getenv("Gemini_API_KEY")
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")

GEMINI_GEN_MODEL = "gemini-3-pro-image-preview"

JUDGE_RESULTS = {
    "gemini": "data/eval_results/gemini-3.1-pro-preview_judged_results.json",
    "gpt":    "data/eval_results/gpt-5.4_judged_results.json",
}

ANNOTATIONS_PATH = "data/combined_output/annotations.json"

GEN_PROMPT = (
    "This image has a black region covering something. "
    "Generate a new version of this image with the hidden object "
    "naturally visible in place of the black region, keeping everything "
    "else in the image exactly the same."
)


def load_qualifying_ids(judge_results_path, annotations_path):
    with open(judge_results_path) as f:
        judged = json.load(f)
    with open(annotations_path) as f:
        annotations = json.load(f)

    phys_map = {str(a["id"]): a["physical"] for a in annotations}
    orig_map = {str(a["id"]): a["original_file"] for a in annotations}

    qualifying = []
    for item in judged:
        item_id = str(item["id"])
        if phys_map.get(item_id):
            qualifying.append({
                "id": item_id,
                "original_file": orig_map.get(item_id, ""),
                "ground_truth": item["ground_truth"],
            })
    return qualifying


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


def generate_gemini(client, image_bytes):
    prompt = GEN_PROMPT
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model=GEMINI_GEN_MODEL,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    return None


def generate_gpt(client, image_bytes):
    prompt = GEN_PROMPT
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.responses.create(
        model="gpt-5.4",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                {"type": "input_text",  "text": prompt},
            ],
        }],
        tools=[{"type": "image_generation"}],
    )
    for item in response.output:
        if item.type == "image_generation_call":
            return base64.b64decode(item.result)
    return None


def run_eval(args):
    provider = args.model
    judge_results_path = JUDGE_RESULTS[provider]
    qualifying = load_qualifying_ids(judge_results_path, args.annotations)
    print(f"[{provider}] {len(qualifying)} qualifying items.")

    output_dir = os.path.join(args.output_dir, provider)
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        done_ids = {r["id"] for r in metadata}
        print(f"Resuming: {len(done_ids)} done, {len(qualifying) - len(done_ids)} remaining.")
    else:
        metadata = []
        done_ids = set()

    if provider == "gemini":
        client = genai.Client(api_key=Gemini_API_KEY)
        generate_fn = generate_gemini
    else:
        client = OpenAI(api_key=OpenAI_API_KEY)
        generate_fn = generate_gpt

    for item in qualifying:
        item_id = item["id"]
        if item_id in done_ids:
            continue

        base = os.path.splitext(item["original_file"])[0]
        occluded_path = os.path.join(args.data_dir, f"{base}_occluded.png")

        if not os.path.exists(occluded_path):
            print(f"  [SKIP] Not found: {occluded_path}")
            continue

        print(f"Generating for {item_id} (GT: {item['ground_truth']})...")
        image_bytes = encode_image_bytes(occluded_path)

        try:
            generated_bytes = generate_fn(client, image_bytes)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        if generated_bytes is None:
            print(f"  [WARN] No image returned for {item_id}")
            continue

        out_path = os.path.join(output_dir, f"{item_id}_generated.png")
        with open(out_path, "wb") as f:
            f.write(generated_bytes)
        print(f"  Saved: {out_path}")

        metadata.append({
            "id": item_id,
            "ground_truth": item["ground_truth"],
            "original_file": item["original_file"],
            "generated_file": f"{item_id}_generated.png",
        })

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nDone. {len(metadata)} images in {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="gemini", choices=["gemini", "gpt"])
    parser.add_argument("--data_dir",    type=str, default="data/combined_output")
    parser.add_argument("--annotations", type=str, default=ANNOTATIONS_PATH)
    parser.add_argument("--output_dir",  type=str, default="data/eval_results_image_gen")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
