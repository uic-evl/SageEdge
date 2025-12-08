#!/usr/bin/env python3
import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# avoid torchvision on jetson / thor
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

MODEL_ID = "vikhyatk/moondream2"
IMAGE_PATH = os.environ.get(
    "IMAGE_PATH",
    "/home/thorwaggle1/Desktop/SageEdge/Benchmarking/images/test.jpg",
)

OUTPUT_DIR = "/home/thorwaggle1/Desktop/SageEdge/Benchmarking/outputs"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "moondream_caption.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"loading {MODEL_ID} on {device} (dtype={dtype})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
)

image = Image.open(IMAGE_PATH).convert("RGB")
question = "Describe this image in detail."

with torch.no_grad():
    # moondream’s remote code exposes these helpers
    image_emb = model.encode_image(image)
    answer = model.answer_question(image_emb, question, tokenizer)

print("\n=== MOONDREAM CAPTION ===")
print(answer)

os.makedirs(OUTPUT_DIR, exist_ok=True)
result = {
    "model_id": MODEL_ID,
    "image_path": IMAGE_PATH,
    "device": device,
    "dtype": str(dtype),
    "question": question,
    "caption": answer,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved caption JSON to: {OUTPUT_JSON}")
print("=== END ===")