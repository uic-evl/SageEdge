"""
Detector ablation: run YOLOv8 n / s / m / x once at a LOW confidence (0.05) and
save every box. Any higher threshold can then be applied offline, so one run
covers all thresholds.

Usage (from the AI_Judge folder):
  python run_detector_ablation.py --list coco_1000.txt --out runs/v3/detections
  # quick test:  --limit 20 --models yolov8n.pt

Weights for s/m/x download automatically the first time (needs internet).
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", default="runs/v3/detections")
    ap.add_argument("--models", nargs="+",
                    default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    print("CUDA available:", torch.cuda.is_available())

    imgs = [str(Path(l.strip()).resolve()) for l in open(a.list) if l.strip()]
    if a.limit:
        imgs = imgs[:a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)

    for w in a.models:
        model = YOLO(w)
        model(imgs[0], verbose=False)                      # warm-up (excluded from timing)
        out = Path(a.out) / f"{Path(w).stem}.jsonl"
        with out.open("w") as f:
            for i, img in enumerate(imgs, 1):
                t0 = time.perf_counter()
                r = model(img, conf=a.conf, verbose=False)[0]
                ms = (time.perf_counter() - t0) * 1000
                dets = [{"class": r.names[int(b.cls)], "conf": round(float(b.conf), 4),
                         "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()]} for b in r.boxes]
                f.write(json.dumps({"image": img, "model": w, "conf_min": a.conf,
                                    "latency_ms": round(ms, 1), "detections": dets}) + "\n")
                if i % 100 == 0:
                    print(f"{w}: {i}/{len(imgs)}")
        print("saved", out)


if __name__ == "__main__":
    main()
