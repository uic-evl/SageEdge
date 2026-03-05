from __future__ import annotations
import json
import argparse
from collections import Counter, defaultdict

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_coco_class_set(yolo_weights: str = "yolov8n.pt") -> set[str]:
    # Uses Ultralytics model metadata to get COCO class names.
    # Safe: does not run inference, just loads names.
    from ultralytics import YOLO
    m = YOLO(yolo_weights)
    names = m.names
    return {str(v).lower() for v in names.values()}

def categorize_violation(claim: str, caption: str, detected_classes: list[str], coco_classes: set[str]) -> str:
    c = (claim or "").lower().strip()
    cap = (caption or "").lower()

    # 1) Claim isn't even in the detector label space
    if c not in coco_classes:
        return "out_of_label_space"

    # 2) Common semantic mismatch patterns (cheap but useful)
    if c == "car" and ("train" in cap or "rail" in cap or "tram" in cap):
        return "semantic_mismatch_train_car"
    if c == "person" and ("no people" in cap or "no person" in cap or "nobody" in cap):
        return "caption_negation_conflict"

    # 3) Near-miss vehicle type confusion (detector sees a related vehicle)
    vehicle_family = {"car", "truck", "bus", "motorcycle", "bicycle"}
    det = set((x or "").lower() for x in (detected_classes or []))
    if c in vehicle_family and len(vehicle_family.intersection(det)) > 0 and c not in det:
        return "vehicle_type_confusion"

    # 4) Otherwise: could be true hallucination OR detector miss
    return "unverified_object"

def summarize(path: str, model_name: str, coco_classes: set[str]) -> dict:
    rows = list(load_jsonl(path))
    cat_counts = Counter()
    claim_counts = Counter()
    examples = defaultdict(list)

    for r in rows:
        caption = r.get("caption", "")
        det_classes = r.get("detected_classes", []) or []
        for v in (r.get("violations") or []):
            if not isinstance(v, dict): 
                continue
            if v.get("type") != "object_not_detected":
                continue
            claim = (v.get("claim") or "").lower().strip()
            cat = categorize_violation(claim, caption, det_classes, coco_classes)
            cat_counts[cat] += 1
            claim_counts[claim] += 1

            if len(examples[cat]) < 5:  # keep a few examples per bucket
                examples[cat].append({
                    "image": r.get("image"),
                    "claim": claim,
                    "caption": caption,
                    "detected_classes": det_classes
                })

    return {
        "model": model_name,
        "N": len(rows),
        "total_object_not_detected": sum(cat_counts.values()),
        "category_counts": dict(cat_counts),
        "top_claims": claim_counts.most_common(10),
        "examples": dict(examples),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llava", required=True)
    ap.add_argument("--moondream", required=True)
    ap.add_argument("--out", default="outputs/week9_violation_report.json")
    ap.add_argument("--yolo", default="yolov8n.pt")
    args = ap.parse_args()

    coco_classes = get_coco_class_set(args.yolo)

    report = {
        "llava": summarize(args.llava, "llava", coco_classes),
        "moondream": summarize(args.moondream, "moondream", coco_classes),
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved:", args.out)
    print("\nLLaVA category_counts:", report["llava"]["category_counts"])
    print("Moondream category_counts:", report["moondream"]["category_counts"])
    print("\nLLaVA top_claims:", report["llava"]["top_claims"])
    print("Moondream top_claims:", report["moondream"]["top_claims"])

if __name__ == "__main__":
    main()
