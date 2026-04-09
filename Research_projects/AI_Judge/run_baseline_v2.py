from __future__ import annotations
import json
from pathlib import Path

import json, time, traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from run_ollama_caption import caption_ollama

from judge.rule_judge_v2 import rule_based_judge_v2


@dataclass
class StageResult:
    ok: bool
    data: Dict[str, Any]
    error: Optional[Dict[str, str]]
    latency_ms: int


def safe_call(name: str, fn, fallback_data: Dict[str, Any]) -> StageResult:
    t0 = time.time()
    try:
        out = fn()
        return StageResult(True, out, None, int((time.time() - t0) * 1000))
    except Exception as e:
        return StageResult(
            False,
            fallback_data,
            {
                "stage": name,
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
            int((time.time() - t0) * 1000),
        )


def yolo_detect(image_path: str, model_name: str = "yolov8n.pt", conf: float = 0.25) -> Dict[str, Any]:
    from ultralytics import YOLO

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

    return {"detections": dets, "model": model_name, "conf": conf}


def _plural(noun: str, n: int) -> str:
    if n == 1:
        return noun
    if noun.endswith("s"):
        return noun
    return noun + "s"


def caption_from_yolo(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    # grounded caption (no action guessing)
    if not detections:
        return {"caption_raw": "I could not detect any recognizable objects."}

    counts: Dict[str, int] = {}
    for d in detections:
        cls = (d.get("class") or "object").lower()
        counts[cls] = counts.get(cls, 0) + 1

    # most frequent first
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    parts = []
    for cls, n in items:
        if cls == "person":
            parts.append(f"{n} {'person' if n==1 else 'people'}")
        else:
            parts.append(f"{n} {_plural(cls, n)}")

    return {"caption_raw": "The image shows " + ", ".join(parts) + "."}


def caption_with_moondream2(image_path: str, prompt: str = "") -> Dict[str, Any]:
    # richer caption using Moondream2 (CPU-friendly)
    from PIL import Image
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "vikhyatk/moondream2"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()

    img = Image.open(image_path).convert("RGB")

    if not prompt:
        prompt = (
            "Write a detailed caption describing what is visible in the image. "
            "Only mention things you are confident are present. "
            "If you are unsure about an action, describe the scene without guessing."
        )

    with torch.inference_mode():
        emb = model.encode_image(img)
        text = model.answer_question(emb, prompt, tok)

    return {"caption_raw": str(text).strip(), "model": model_id}


def run(image_path: str, use_vlm: bool = True) -> Dict[str, Any]:
    image_path = str(Path(image_path).resolve())
    out: Dict[str, Any] = {"image_path": image_path, "stages": {}}

    det = safe_call("detect", lambda: yolo_detect(image_path), {"detections": [], "model": None, "conf": None})
    out["stages"]["detect"] = asdict(det)
    out["detections"] = out["stages"]["detect"]["data"].get("detections", [])

    cap_text = ""
    if use_vlm:
        cap_vlm = safe_call(
            "caption_vlm",
            lambda: {"caption_raw": caption_ollama(image_path, model="moondream")},
            {"caption_raw": ""}
        )
        out["stages"]["caption_vlm"] = asdict(cap_vlm)
        cap_text = out["stages"]["caption_vlm"]["data"].get("caption_raw", "").strip()
    if not cap_vlm.ok:
        print("VLM ERROR:")
        print(cap_vlm.error["type"])
        print(cap_vlm.error["message"])
        print(cap_vlm.error["traceback"])

    if not cap_text:
        raise RuntimeError("VLM caption failed; stopping to avoid YOLO-caption leakage.")
    else:
        out["stages"]["caption"] = out["stages"]["caption_vlm"]
        out["caption_raw"] = cap_text

    judge = safe_call(
    "judge_rule_v2",
    lambda: {"rule_based_v2": rule_based_judge_v2(out["caption_raw"], out["detections"])},
    {"rule_based_v2": {"claims": {}, "violations": [], "hallucination_score": None, "grounding_score": None}},
    )
    
    out["stages"]["judge_rule"] = asdict(judge)
    out["judge"] = out["stages"]["judge_rule"]["data"]

    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="outputs/result.json")
    p.add_argument("--no_vlm", action="store_true", help="Disable VLM captioning; use YOLO-only caption.")
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    result = run(args.image, use_vlm=(not args.no_vlm))
    # --- FAKE CAPTION TEST ---
    from judge.rule_judge import rule_based_judge

    #fake = "A person."
    #result["caption_raw"] = fake
    #result["judge"] = {"rule_based": rule_based_judge(fake, result.get("detections", []))}
    # ---- Week7 benchmark logging ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rb = (result.get("judge") or {}).get("rule_based_v2", {})

    record = {
        "image": args.image,
        "caption": result.get("caption_raw"),
        "detections": result.get("detections", []),

        "hallucination": rb.get("hallucination_score"),
        "grounding": rb.get("grounding_score"),
        "violations": rb.get("violations", []),

        "claims": ((rb.get("claims") or {}).get("objects", [])),
        "detected_classes": rb.get("detected_classes", []),
        "detected_classes_raw": rb.get("detected_classes_raw", []),

        "checked_claim_count": rb.get("checked_claim_count"),
        "supported_claim_count": rb.get("supported_claim_count"),
        "unsupported_claim_count": rb.get("unsupported_claim_count"),
    }

    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    # ---------------------------------
    
    print("Saved:", args.out)
    print("Caption:", result.get("caption_raw"))
    print("Num detections:", len(result.get("detections", [])))
