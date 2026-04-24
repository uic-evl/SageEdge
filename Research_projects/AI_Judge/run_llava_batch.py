from __future__ import annotations
import json, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
from ultralytics import YOLO

from judge.rule_judge import rule_based_judge


def yolo_detect(image_path: str, model_name: str = "yolov8n.pt", conf: float = 0.25) -> Dict[str, Any]:
    model = YOLO(model_name)
    res = model(image_path, conf=conf, verbose=False)[0]
    dets: List[Dict[str, Any]] = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        dets.append({
            "class": res.names[cls_id],
            "conf": float(b.conf.item()),
            "xyxy": [float(x) for x in b.xyxy[0].tolist()],
        })
    return {"detections": dets, "model": model_name, "conf": conf}


def caption_ollama(image_path: str, model: str, prompt: str) -> str:
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        images=[image_path],
        stream=False,
    )
    return resp["response"].strip()


def safe(fn, fallback):
    t0 = time.time()
    try:
        return True, fn(), None, int((time.time() - t0) * 1000)
    except Exception as e:
        return False, fallback, {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }, int((time.time() - t0) * 1000)


def run_one(image_path: str, ollama_model: str, prompt: str) -> Dict[str, Any]:
    image_path = str(Path(image_path).resolve())
    out: Dict[str, Any] = {"image": image_path, "stages": {}}

    ok_det, det_out, det_err, det_ms = safe(
        lambda: yolo_detect(image_path, conf=0.25),
        {"detections": [], "model": None, "conf": 0.25},
    )
    out["stages"]["detect"] = {"ok": ok_det, "latency_ms": det_ms, "error": det_err, "data": det_out}
    dets = det_out.get("detections", [])

    ok_cap, cap_text, cap_err, cap_ms = safe(
        lambda: caption_ollama(image_path, model=ollama_model, prompt=prompt),
        "",
    )
    out["stages"]["caption_vlm"] = {"ok": ok_cap, "latency_ms": cap_ms, "error": cap_err, "data": {"model": ollama_model}}
    out["caption"] = cap_text

    rb = rule_based_judge(out["caption"], dets)

    record = {
        "image": image_path,
        "caption": out["caption"],
        "detections": dets,
        "hallucination": rb.get("hallucination_score"),
        "grounding": rb.get("grounding_score"),
        "violations": rb.get("violations", []),
        "claims": (rb.get("claims") or {}).get("objects", []),
        "detected_classes": rb.get("detected_classes", []),
        "vlm_model": ollama_model,
        "yolo_conf": 0.25,
        "prompt": prompt,
    }
    return record


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, help="Text file with one image path per line")
    p.add_argument("--out_jsonl", default="runs/llava_run/predictions.jsonl")
    p.add_argument("--model", default="llava")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    prompt = "Describe the image in one sentence and list the visible objects."

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # clear output each run (Week 8 controlled experiment)
    if out_path.exists():
        out_path.unlink()

    n = 0
    with open(args.list, "r") as f_in, open(out_path, "a") as f_out:
        for line in f_in:
            img = line.strip()
            if not img:
                continue
            rec = run_one(img, ollama_model=args.model, prompt=prompt)
            f_out.write(json.dumps(rec) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"processed {n}")
            if args.limit and n >= args.limit:
                break

    print("Saved:", str(out_path), "N=", n)


if __name__ == "__main__":
    main()
