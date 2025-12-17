#!/usr/bin/env python3
import os
import json
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# avoid torchvision on jetson / thor
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

MODEL_ID = "vikhyatk/moondream2"

# repo root: Benchmarking/
BASE_DIR = Path(__file__).resolve().parent.parent

# image path: env override OR repo-relative default
image_path_str = os.environ.get("IMAGE_PATH", "images/image2.jpg")
IMAGE_PATH = Path(image_path_str)
if not IMAGE_PATH.is_absolute():
    IMAGE_PATH = (BASE_DIR / IMAGE_PATH).resolve()

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

# outputs: env override OR repo-relative outputs/
output_dir_str = os.environ.get("BENCH_OUTPUT_DIR", "outputs")
OUTPUT_DIR = Path(output_dir_str)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (BASE_DIR / OUTPUT_DIR).resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "moondream_caption.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"loading {MODEL_ID} on {device} (dtype={dtype})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,          # keep since it worked for you
    device_map="auto",          # safer than device_map=device
    trust_remote_code=True,
)

image = Image.open(IMAGE_PATH).convert("RGB")
question = "Describe this image in detail."

with torch.no_grad():
    image_emb = model.encode_image(image)
    answer = model.answer_question(image_emb, question, tokenizer)

print("\n=== MOONDREAM CAPTION ===")
print(answer)

result = {
    "model_id": MODEL_ID,
    "image_path": str(IMAGE_PATH),
    "device": device,
    "dtype": str(dtype),
    "question": question,
    "caption": answer,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nsaved caption json to: {OUTPUT_JSON}")
print("=== END ===")
