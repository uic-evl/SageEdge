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
import hashlib
import platform
import sys

import ollama


SAMPLE_FILE   = "data/testsets/semantic_experiment_100.json"
OUTPUT_DIR    = "outputs/semantic_experiment"
EMBED_MODEL   = "nomic-embed-text"
NUM_RUNS      = 5

PROMPT = "Describe what you see in this image."

# Generation options for reproducibility
TEMPERATURE   = 0.0
TOP_P         = 1.0
MAX_TOKENS    = 256


parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True, help="Device name (e.g. thor, dell_gb10)")
parser.add_argument("--model",  required=True, help="Ollama model name (e.g. moondream2)")
args = parser.parse_args()

DEVICE_NAME  = args.device
MODEL_NAME   = args.model

SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_").replace(":", "-")

EXPERIMENT_DATE = time.strftime("%Y%m%d")

EXPERIMENT_ID = hashlib.md5(
    f"{MODEL_NAME}|{PROMPT}|{TEMPERATURE}|{TOP_P}|{MAX_TOKENS}|{EMBED_MODEL}".encode()
).hexdigest()[:10]

OUTPUT_FILE  = os.path.join(
    OUTPUT_DIR,
    f"results_{DEVICE_NAME}_{EXPERIMENT_DATE}_{SAFE_MODEL_NAME}_{EXPERIMENT_ID}.jsonl",
)

RUNTIME_INFO = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "ollama_python_package": getattr(ollama, "__version__", "unknown"),
}

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
print(f"Experiment ID: {EXPERIMENT_ID}")
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
        }],
        options={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": MAX_TOKENS
        }
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
            text = None
            emb = None
            gen_latency = None
            emb_latency = None
            total_latency = None
            error = None

            for attempt in range(2):
                try:
                    t_gen = time.time()
                    text = generate_response(image_path, PROMPT)
                    gen_latency = time.time() - t_gen

                    t_emb = time.time()
                    emb = get_embedding(text)
                    emb_latency = time.time() - t_emb

                    total_latency = gen_latency + emb_latency
                    break
                except Exception as e:
                    error = str(e)
                    print(f"  run {run_i+1}/{NUM_RUNS} attempt {attempt+1} failed: {e}")
                    if attempt == 0:
                        print("    retrying once...")
                        time.sleep(2)

            if total_latency is not None:
                preview = (text or "").replace("\n", " ")[:60]
                print(f"  run {run_i+1}/{NUM_RUNS}  ({total_latency:.2f}s)  \"{preview}...\"")
            else:
                print(f"  run {run_i+1}/{NUM_RUNS}  FAILED after retries")

            run_outputs.append({
                "run": run_i,
                "text": text,
                "text_lower": text.lower() if text is not None else None,
                "embedding": emb,
                "gen_latency_s": gen_latency,
                "embed_latency_s": emb_latency,
                "total_latency_s": total_latency,
                "gen_latency_s_3dp": round(gen_latency, 3) if gen_latency is not None else None,
                "embed_latency_s_3dp": round(emb_latency, 3) if emb_latency is not None else None,
                "total_latency_s_3dp": round(total_latency, 3) if total_latency is not None else None,
                "error": error,
            })

        # Compute per-image total latency
        image_total_latency = sum(
            (o["total_latency_s"] for o in run_outputs if o["total_latency_s"] is not None),
            0.0,
        )

        record = {
            "image_id":           image_id,
            "file_name":          sample["file_name"],
            "device":             DEVICE_NAME,
            "model":              MODEL_NAME,
            "embedding_model":    EMBED_MODEL,
            "experiment_id":      EXPERIMENT_ID,
            "prompt":             PROMPT,
            "expected_entities":  expected,
            "generation_options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS
            },
            "runtime":            RUNTIME_INFO,
            "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_total_latency_s": round(image_total_latency, 3),
            "outputs":            run_outputs
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