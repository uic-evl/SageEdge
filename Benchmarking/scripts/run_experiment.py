"""
run_experiment.py

Runs the semantic quality experiment on a single device.
For each of 100 images, runs the model 5 times and collects:
  - The text output
  - The Ollama embedding of that output

Output: outputs/semantic_experiment/results_{DEVICE_NAME}.jsonl

Run on BOTH Thor and Dell GB10 independently, then copy both
JSONL files to one machine for analysis.

Usage:
    # Set your device name and model before running
    python scripts/run_experiment.py --device thor --model moondream2
    python scripts/run_experiment.py --device dell_gb10 --model moondream2

Requirements:
    pip install ollama
    ollama pull nomic-embed-text   # embedding model
    ollama pull <your-model>       # inference model (if using Ollama for inference)

Note:
    If you're using HuggingFace for inference instead of Ollama,
    replace the `generate_response()` function with your existing
    model loading pattern from your bench_*.py scripts.
"""

import json
import os
import argparse
import time
from pathlib import Path

import ollama


SAMPLE_FILE   = "data/testsets/semantic_experiment_100.json"
OUTPUT_DIR    = "outputs/semantic_experiment"
EMBED_MODEL   = "nomic-embed-text"
NUM_RUNS      = 5

PROMPT = "Describe what you see in this image."


parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True, help="Device name (e.g. thor, dell_gb10)")
parser.add_argument("--model",  required=True, help="Ollama model name (e.g. moondream2)")
args = parser.parse_args()

DEVICE_NAME  = args.device
MODEL_NAME   = args.model
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, f"results_{DEVICE_NAME}.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)


completed_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                completed_ids.add(rec["image_id"])
            except json.JSONDecodeError:
                pass
    print(f"Resuming — {len(completed_ids)} images already done")


with open(SAMPLE_FILE) as f:
    samples = json.load(f)

remaining = [s for s in samples if s["image_id"] not in completed_ids]
print(f"Images to process: {len(remaining)} / {len(samples)}")
print(f"Device: {DEVICE_NAME}  |  Model: {MODEL_NAME}  |  Runs per image: {NUM_RUNS}")
print(f"Output: {OUTPUT_FILE}\n")


def generate_response(image_path: str, prompt: str) -> str:
    """
    Generate a response using Ollama.
    Replace this function body if you prefer HuggingFace inference —
    just keep the same signature (returns a string).
    """
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }]
    )
    return response["message"]["content"].strip()


def get_embedding(text: str) -> list:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


print("Warming up model...")
try:
    _ = generate_response(samples[0]["full_path"], PROMPT)
    _ = get_embedding("warmup text")
    print("Warmup complete.\n")
except Exception as e:
    print(f"Warmup failed: {e}")
    print("Check that Ollama is running and the model is pulled.")
    exit(1)


start_time = time.time()

with open(OUTPUT_FILE, "a") as out_f:
    for idx, sample in enumerate(remaining):
        image_id   = sample["image_id"]
        image_path = sample["full_path"]
        expected   = sample["expected_entities"]

        if not os.path.exists(image_path):
            print(f"[{idx+1}/{len(remaining)}] SKIP — file not found: {image_path}")
            continue

        print(f"[{idx+1}/{len(remaining)}] image_id={image_id}  ({Path(image_path).name})")

        run_outputs = []
        for run_i in range(NUM_RUNS):
            t0   = time.time()
            text = generate_response(image_path, PROMPT)
            emb  = get_embedding(text)
            elapsed = time.time() - t0

            run_outputs.append({
                "run":       run_i,
                "text":      text,
                "embedding": emb,
                "latency_s": round(elapsed, 3)
            })
            print(f"  run {run_i+1}/{NUM_RUNS}  ({elapsed:.2f}s)  \"{text[:60]}...\"")

        record = {
            "image_id":          image_id,
            "file_name":         sample["file_name"],
            "device":            DEVICE_NAME,
            "model":             MODEL_NAME,
            "prompt":            PROMPT,
            "expected_entities": expected,
            "outputs":           run_outputs
        }

        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        # Progress estimate
        elapsed_total = time.time() - start_time
        done = idx + 1
        rate = done / elapsed_total
        eta  = (len(remaining) - done) / rate if rate > 0 else 0
        print(f"  → saved  |  ETA: {eta/60:.1f} min remaining\n")

print(f"Done! Results saved to {OUTPUT_FILE}")