from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")
img = Image.open("data/images/coco_val2017/000000397133.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Write one sentence caption describing the image."},
        ],
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("PROMPT LENGTH:", len(prompt))
print("PROMPT:")
print(repr(prompt))
