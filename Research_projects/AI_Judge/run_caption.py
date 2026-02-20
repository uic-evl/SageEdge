from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO


def yolo_detect(image_path: str, model_name: str = "yolov8n.pt", conf: float = 0.25) -> List[Dict]:
    model = YOLO(model_name)
    res = model(image_path, conf=conf, verbose=False)[0]

    dets = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        dets.append({
            "class": res.names[cls_id],
            "conf": float(b.conf.item())
        })
    return dets


def make_caption(detections: List[Dict]) -> str:
    if not detections:
        return "I could not detect any recognizable objects."

    counts = {}
    for d in detections:
        cls = d["class"]
        counts[cls] = counts.get(cls, 0) + 1

    parts = []
    for cls, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if cls == "person":
            parts.append(f"{n} {'person' if n == 1 else 'people'}")
        else:
            parts.append(f"{n} {cls if n == 1 else cls + 's'}")

    return "The image shows " + ", ".join(parts) + "."


def run(image_path: str):
    image_path = str(Path(image_path).resolve())
    detections = yolo_detect(image_path)
    caption = make_caption(detections)

    print("Caption:")
    print(caption)
    print("\nDetections:")
    for d in detections:
        print("-", d)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    args = p.parse_args()

    run(args.image)
