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

load_dotenv()
OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
Anthropic_API_KEY = os.getenv("Anthropic_API_KEY")
Gemini_API_KEY = os.getenv("Gemini_API_KEY")
Together_API_KEY = os.getenv("Together_API_KEY")

PROMPT = "What is the object outlined by the yellow border in this image? Be as specific as you can."

def init_client(args):
    if "gpt" in args.model or args.model[0] == 'o':
        return OpenAI(api_key=OpenAI_API_KEY)
    elif "gemini" in args.model:
        return OpenAI(api_key=Gemini_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif "claude" in args.model:
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

def chat_completion(args, client, image_path):
    b64 = encode_image(image_path)
    media_type = "image/png"

    if "claude" in args.model:
        return client.messages.create(
            model=args.model,
            max_tokens=500,
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
            model=args.model,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": PROMPT}
                ]
            }]
        ).choices[0].message.content

def run_eval(args, client):
    with open(os.path.join(args.data_dir, "annotations.json")) as f:
        entries = json.load(f)

    model_name = args.model.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_name}_transparent_results.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} already done, {len(entries) - len(done_ids)} remaining.")
    else:
        results = []
        done_ids = set()

    for entry in entries:
        if entry["id"] in done_ids:
            continue

        base = os.path.splitext(entry["original_file"])[0]
        transparent_path = os.path.join(args.data_dir, f"{base}_transparent.png")

        if not os.path.exists(transparent_path):
            print(f"Missing transparent image for {entry['id']}, skipping.")
            continue

        print(f"Running {entry['id']}...")
        response = chat_completion(args, client, transparent_path)
        print(f"  {response}")

        results.append({
            "id": entry["id"],
            "ground_truth": entry.get("answer", ""),
            "response": response
        })

        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--data_dir", type=str, default="data/combined_output")
    parser.add_argument("--output_dir", type=str, default="data/eval_results")
    return parser.parse_args()

def main(args):
    client = init_client(args)
    os.makedirs(args.output_dir, exist_ok=True)
    results = run_eval(args, client)
    model_name = args.model.split("/")[-1]
    print(f"Done. {len(results)} results in {os.path.join(args.output_dir, f'{model_name}_transparent_results.json')}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
