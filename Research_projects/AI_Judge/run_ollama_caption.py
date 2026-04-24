from __future__ import annotations

import json, time
from pathlib import Path
from typing import Any, Dict, List

import ollama
from ultralytics import YOLO


def yolo_detect(image_path: str, model_name: str = "yolov8n.pt", conf: float = 0.25) -> List[Dict[str, Any]]:
    model = YOLO(model_name)
    res = model(image_path, conf=conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        dets.append({
            "class": res.names[cls_id],
            "conf": float(b.conf.item()),
            "xyxy": [float(x) for x in b.xyxy[0].tolist()],
        })
    return dets


def caption_ollama(image_path: str, model: str = "moondream") -> str:
    prompt = (
        "Write a detailed caption describing what you see in the image. "
        "Mention people, objects, and what they are doing if it is clearly visible. "
        "Do NOT guess. If unsure, say you are unsure."
    )
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        images=[image_path],
        stream=False,
    )
    return resp["response"].strip()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="outputs/ollama_caption.json")
    p.add_argument("--model", default="moondream")  # or llava, llama3.2-vision
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    image_path = str(Path(args.image).resolve())

    out: Dict[str, Any] = {"image_path": image_path, "stages": {}, "caption_raw": "", "detections": []}

    t0 = time.time()
    cap = caption_ollama(image_path, model=args.model)
    out["caption_raw"] = cap
    out["stages"]["caption_ms"] = int((time.time() - t0) * 1000)

    t1 = time.time()
    dets = yolo_detect(image_path)
    out["detections"] = dets
    out["stages"]["yolo_ms"] = int((time.time() - t1) * 1000)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print("Saved:", args.out)
    print("Caption:", out["caption_raw"])
    print("Detected classes:", sorted({d["class"] for d in dets}))


if __name__ == "__main__":
    main()

