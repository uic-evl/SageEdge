import os
os.environ.setdefault('TRANSFORMERS_NO_TORCHVISION', '1')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import warnings
warnings.filterwarnings('ignore')

MODEL_ID = 'google/gemma-3n-E4B-it'
img_path = '/home/dgxwaggle/Desktop/SageEdge/Benchmarking/data/coco/val2017/000000000139.jpg'

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='cuda',
)
model.eval()

image = Image.open(img_path).convert('RGB')
print(f"Image size: {image.size}")

messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image'},
            {'type': 'text', 'text': 'Describe this image in one sentence.'},
        ],
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(f"\nPrompt ends with: ...{prompt[-100:]}")

inputs = processor(text=prompt, images=[image], return_tensors='pt')
inputs = {k: v.to('cuda') for k, v in inputs.items()}

print(f"Input shape: {inputs['input_ids'].shape}")

# Try different generation strategies
print("\n" + "="*60)
print("TEST 1: Basic generation (no special params)")
print("="*60)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=30)
gen_ids = outputs[0, inputs['input_ids'].shape[-1]:]
print(f"Generated tokens: {gen_ids.tolist()[:20]}")
text = processor.decode(gen_ids, skip_special_tokens=True).strip()
print(f"Result: '{text}'")

print("\n" + "="*60)
print("TEST 2: With temperature sampling")
print("="*60)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
gen_ids = outputs[0, inputs['input_ids'].shape[-1]:]
print(f"Generated tokens: {gen_ids.tolist()[:20]}")
text = processor.decode(gen_ids, skip_special_tokens=True).strip()
print(f"Result: '{text}'")

print("\n" + "="*60)
print("TEST 3: Force minimum tokens")
print("="*60)
with torch.inference_mode():
    outputs = model.generate(**inputs, min_new_tokens=10, max_new_tokens=30)
gen_ids = outputs[0, inputs['input_ids'].shape[-1]:]
print(f"Generated tokens: {gen_ids.tolist()[:20]}")
text = processor.decode(gen_ids, skip_special_tokens=True).strip()
print(f"Result: '{text}'")
