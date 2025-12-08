import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# avoid torchvision on Jetson / Thor
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

MODEL_ID = "google/gemma-3n-E4B-it"
IMAGE_PATH = os.environ.get(
    "IMAGE_PATH",
    "/home/thorwaggle1/Desktop/SageEdge/Benchmarking/images/test.jpg",
)

# where to save JSON output
OUTPUT_DIR = os.path.join(
    "/home/thorwaggle1/Desktop/SageEdge/Benchmarking", "outputs"
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "gemma_caption.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"loading {MODEL_ID} on {device} (dtype={dtype})")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,   # HF still supports this, even if it's "deprecated"
    device_map=device,
)

# 1) load image
img = Image.open(IMAGE_PATH).convert("RGB")

# 2) build prompt that matches your chat template
tokenizer = processor.tokenizer
bos = tokenizer.bos_token or ""

prompt = (
    f"{bos}"
    "<start_of_turn>user\n"
    "<image_soft_token>Describe this image in detail.<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# 3) processor builds text + image tensors
inputs = processor(
    text=prompt,
    images=[img],
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=80)

# 4) slice off the prompt tokens
input_len = inputs["input_ids"].shape[-1]
generated_ids = outputs[0, input_len:]

caption = processor.decode(generated_ids, skip_special_tokens=True)

print("\n=== CAPTION ===")
print(caption)

# 5) save to JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)

result = {
    "model_id": MODEL_ID,
    "image_path": IMAGE_PATH,
    "device": device,
    "dtype": str(dtype),
    "caption": caption,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved caption JSON to: {OUTPUT_JSON}")
