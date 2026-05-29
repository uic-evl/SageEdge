from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def caption_gemma3n_hf(
    image_path: str,
    model_id: str = "google/gemma-3n-E2B-it",
    max_new_tokens: int = 80,
) -> str:
    image_path = str(Path(image_path).resolve())
    image = Image.open(image_path).convert("RGB")

    prompt = (
        "Describe this image in one short factual sentence. "
        "Mention only clearly visible people and objects. Do not guess."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_id)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

    if prompt in decoded:
        decoded = decoded.split(prompt, 1)[-1].strip()

    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    if lines and lines[0].lower() == "model":
        lines = lines[1:]

    decoded = " ".join(lines).strip()
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="google/gemma-3n-E2B-it")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    caption = caption_gemma3n_hf(
        image_path=args.image,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    print(caption)


if __name__ == "__main__":
    main()
