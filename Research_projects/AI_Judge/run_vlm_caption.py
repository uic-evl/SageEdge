from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from ultralytics import YOLO


# -------------------------
# YOLO
# -------------------------
def yolo_detect(image_path: str, model_name: str = "yolov8n.pt", conf: float = 0.25) -> List[Dict[str, Any]]:
    model = YOLO(model_name)
    res = model(image_path, conf=conf, verbose=False)[0]

    dets: List[Dict[str, Any]] = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        dets.append(
            {
                "class": res.names[cls_id],
                "conf": float(b.conf.item()),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            }
        )
    return dets


# -------------------------
# Moondream2 (Transformers)
# -------------------------
def caption_moondream2(image: Image.Image, prompt: str) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "vikhyatk/moondream2"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,   # CPU-safe
    ).eval()

    with torch.inference_mode():
        emb = model.encode_image(image)
        text = model.answer_question(emb, prompt, tok)

    return str(text).strip()


# -------------------------
# Gemma 3n (Transformers)
# -------------------------
def caption_gemma3n(image: Image.Image, prompt: str, device: str = "cpu", max_new_tokens: int = 80) -> str:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "google/gemma-3n-E4B-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # Transformers docs: prompt must include <image_soft_token> where image goes
    full_prompt = f"<image_soft_token>{prompt}"

    inputs = processor(text=full_prompt, images=[image], return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # decode full output; keep it simple
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return str(text).strip()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="outputs/vlm_caption.json")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--skip_gemma", action="store_true")
    p.add_argument("--skip_moondream", action="store_true")
    p.add_argument("--skip_yolo", action="store_true")
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    image_path = str(Path(args.image).resolve())
    img = Image.open(image_path).convert("RGB")

    prompt = (
        "Write a detailed caption of what you see in the image. "
        "Mention people, objects, and what they are doing. "
        "Do not guess. If unsure about an action, describe the scene without guessing."
    )

    out: Dict[str, Any] = {"image_path": image_path, "results": {}, "latency_ms": {}}

    if not args.skip_yolo:
        t0 = time.time()
        dets = yolo_detect(image_path)
        out["results"]["yolo_detections"] = dets
        out["latency_ms"]["yolo"] = int((time.time() - t0) * 1000)

    if not args.skip_moondream:
        t0 = time.time()
        out["results"]["moondream2_caption"] = caption_moondream2(img, prompt)
        out["latency_ms"]["moondream2"] = int((time.time() - t0) * 1000)

    if not args.skip_gemma:
        t0 = time.time()
        out["results"]["gemma3n_caption"] = caption_gemma3n(img, prompt, device=args.device)
        out["latency_ms"]["gemma3n"] = int((time.time() - t0) * 1000)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print("Saved:", args.out)

    if "moondream2_caption" in out["results"]:
        print("\nMoondream2:", out["results"]["moondream2_caption"])
    if "gemma3n_caption" in out["results"]:
        print("\nGemma3n:", out["results"]["gemma3n_caption"])
    if "yolo_detections" in out["results"]:
        print("\nYOLO classes:", sorted({d["class"] for d in out["results"]["yolo_detections"]}))


if __name__ == "__main__":
    main()
