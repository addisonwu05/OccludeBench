import argparse
import base64
import io
import json
import os
from openai import OpenAI
from together import Together
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Together_API_KEY = os.getenv("Together_API_KEY")

PROMPT = "What is the primary object that is completely occluded in this image? Be as specific as you can."

JUDGE_MASK_PROMPT = (
    "A vision model was asked to identify the primary occluded object in an image. "
    "In the image, a black or dark filled shape (rectangle, polygon, etc.) with a yellow or colored border "
    "is used as an annotation marker to hide (occlude) the real object underneath it. "
    "Here is the model's response:\n\n\"{response}\"\n\n"
    "Does this response identify the annotation marker itself — the black/dark shape, rectangle, or polygon "
    "with the yellow/colored border or outline — as the occluded object, rather than correctly identifying "
    "what real-world object is hidden underneath the marker? "
    "Answer only 'yes' or 'no'."
)

MASK_FOLLOWUP_PROMPT = (
    "The black shape with the yellow border in this image is just an annotation marker used to indicate "
    "where an object has been hidden — it is not the occluded object itself. "
    "Looking carefully at the surrounding context of the image, what specific real-world object do you "
    "think is actually hidden underneath that marker?"
)

# Maps result filename stem -> full Together model ID
MODELS = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c": (
        "cocoscilab/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-9940db5c"
    ),
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
}


def judge(response, judge_model):
    client = OpenAI(api_key=OpenAI_API_KEY)
    prompt = JUDGE_MASK_PROMPT.format(response=response)
    verdict = client.chat.completions.create(
        model=judge_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().lower()
    return verdict.startswith("yes")


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


def mask_followup_completion(model_id, image_path, initial_response):
    client = Together(api_key=Together_API_KEY)
    b64 = encode_image(image_path)
    media_type = "image/png"

    return client.chat.completions.create(
        model=model_id,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": PROMPT}
                ]
            },
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": MASK_FOLLOWUP_PROMPT}
        ]
    ).choices[0].message.content


def run_vetter(args):
    with open(os.path.join(args.data_dir, "annotations.json")) as f:
        annotations = json.load(f)
    id_to_file = {e["id"]: e["original_file"] for e in annotations}

    os.makedirs(args.output_dir, exist_ok=True)

    for stem, model_id in MODELS.items():
        input_path = os.path.join(args.results_dir, f"{stem}_results.json")
        output_path = os.path.join(args.output_dir, f"{stem}_vetted_results.json")

        if not os.path.exists(input_path):
            print(f"[{stem}] Result file not found: {input_path}, skipping.")
            continue

        with open(input_path) as f:
            results = json.load(f)

        # Load existing vetted output for resume support
        if os.path.exists(output_path):
            with open(output_path) as f:
                vetted = json.load(f)
            vetted_by_id = {r["id"]: r for r in vetted}
            print(f"[{stem}] Resuming: {len(vetted_by_id)} already vetted.")
        else:
            vetted_by_id = {}

        updated = False
        for result in results:
            entry_id = result["id"]

            # Skip immediately if already in vetted output
            if entry_id in vetted_by_id:
                continue

            vetted_entry = dict(result)
            if "mask_followup_response" not in vetted_entry:
                vetted_entry["mask_followup_response"] = None

            response = result.get("response", "")
            print(f"[{stem}] Judging {entry_id}...")

            describes_mask = judge(response, args.judge_model)
            if describes_mask:
                original_file = id_to_file.get(entry_id)
                if not original_file:
                    print(f"  [!] No image mapping for id {entry_id}, skipping follow-up.")
                    vetted_by_id[entry_id] = vetted_entry
                    continue

                base = os.path.splitext(original_file)[0]
                occluded_path = os.path.join(args.data_dir, f"{base}_occluded.png")

                print(f"  [Judge] Describes mask — sending follow-up...")
                followup = mask_followup_completion(model_id, occluded_path, response)
                print(f"  Follow-up: {followup}")
                vetted_entry["mask_followup_response"] = followup
            else:
                vetted_entry["mask_followup_response"] = None

            vetted_by_id[entry_id] = vetted_entry
            updated = True

            # Save after each entry
            with open(output_path, "w") as f:
                json.dump(list(vetted_by_id.values()), f, indent=2)

        if not updated:
            # Write output even if nothing was updated (first run with all already done)
            with open(output_path, "w") as f:
                json.dump(list(vetted_by_id.values()), f, indent=2)

        flagged = sum(
            1 for r in vetted_by_id.values() if r.get("mask_followup_response") is not None
        )
        print(f"[{stem}] Done. {flagged}/{len(results)} entries had mask follow-ups.")
        print(f"  Output: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vetter: detect and follow up when a model describes the occlusion marker itself."
    )
    parser.add_argument("--data_dir", type=str, default="data/combined_output",
                        help="Directory containing annotations.json and occluded images.")
    parser.add_argument("--results_dir", type=str, default="data/eval_results",
                        help="Directory containing the original *_results.json files.")
    parser.add_argument("--output_dir", type=str, default="data/eval_results",
                        help="Directory to write *_vetted_results.json files.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o",
                        help="OpenAI model to use for judging responses.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vetter(args)
