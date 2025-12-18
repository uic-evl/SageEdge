#!/usr/bin/env python3
import os
import json
import torch
import warnings
from PIL import Image
from pathlib import Path

# -----------------------------
# QUIET MODE SETUP
# -----------------------------

# Suppress HF + PyTorch warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="Found GPU0")
warnings.filterwarnings("ignore", category=UserWarning)

# Disable HF progress bars + logging noise
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# -----------------------------
# LLaVA imports (after quiet mode)
# -----------------------------

from llavamini.model.builder import load_pretrained_model
from llavamini.mm_utils import get_model_name_from_path, process_images
from llavamini.constants import IMAGE_TOKEN_INDEX
from llavamini.mm_utils import tokenizer_image_token

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "ICTNLP/llava-mini-llama-3.1-8b"

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = Path(
    os.environ.get("IMAGE_PATH", "images/image3.jpg")
)
if not IMAGE_PATH.is_absolute():
    IMAGE_PATH = (BASE_DIR / IMAGE_PATH).resolve()

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

OUTPUT_DIR = Path(
    os.environ.get("BENCH_OUTPUT_DIR", "outputs")
)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (BASE_DIR / OUTPUT_DIR).resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "llava_caption.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"loading LLaVA-Mini on {device}")

# -----------------------------
# Load model (quiet)
# -----------------------------

model_name = get_model_name_from_path(MODEL_PATH)

tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_PATH,
    None,
    model_name,
    torch_dtype=dtype,
    device_map="auto",
)

model.eval()

# -----------------------------
# Load image
# -----------------------------

image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = process_images(
    [image],
    image_processor,
    model.config,
).unsqueeze(1).to(device, dtype=dtype)

# -----------------------------
# Build prompt
# -----------------------------

question = "Describe this image in detail."
prompt = f"<image>\n{question}"

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors="pt",
).unsqueeze(0).to(device)

# -----------------------------
# Inference
# -----------------------------

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=100,
        do_sample=False,
    )

caption = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True,
).strip()

print("\n=== LLAVA CAPTION ===")
print(caption)

# -----------------------------
# Save result
# -----------------------------

result = {
    "model_id": MODEL_PATH,
    "image_path": str(IMAGE_PATH),
    "device": device,
    "dtype": str(dtype),
    "question": question,
    "caption": caption,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nsaved caption json to: {OUTPUT_JSON}")
