"""
AI Judge: check the object claims of a VLM caption against the image.

How it works
  1. parse_caption_v3 extracts the objects the caption claims (handles negation
     like "no people", compounds like "table lamp", synonyms like "man" -> person).
  2. YOLOv8 detects the objects in the image (default YOLOv8x, confidence >= 0.25).
  3. Each claim that is a COCO class is "supported" if YOLO found that class,
     otherwise "unsupported" (= likely hallucination). Other words (sky, wall ...)
     are "out of vocabulary": YOLO cannot check them.
  hallucination_score = unsupported / checked claims (None if nothing is checkable).

Usage (from v3/, venv with ultralytics active):
  # one image
  python judge.py --image val2017/000000387387.jpg --caption "A yellow bus parked next to a man with a dog."
  # many captions (output of run_captions_unified.py) -> JSONL that evaluate.py can score
  python judge.py --captions results/llava.jsonl --out runs/llava_judged.jsonl
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from judge_v3.rule_judge_v3 import rule_based_judge_v3


class Judge:
    def __init__(self, weights="yolov8x.pt", conf=0.25, device=None):
        from ultralytics import YOLO
        self.model, self.conf, self.device = YOLO(weights), conf, device
        self.weights = weights

    def detect(self, image):
        r = self.model(image, conf=self.conf, device=self.device, verbose=False)[0]
        return [{"class": r.names[int(b.cls)], "conf": round(float(b.conf), 4)} for b in r.boxes]

    def __call__(self, image, caption):
        t0 = time.perf_counter()
        dets = self.detect(image)
        t1 = time.perf_counter()
        j = rule_based_judge_v3(caption, dets, min_conf=self.conf)
        t2 = time.perf_counter()
        return {"image": str(image), "caption": caption, "detector": self.weights, "min_conf": self.conf,
                "detector_latency_ms": round((t1 - t0) * 1000, 1), "judge_latency_ms": round((t2 - t1) * 1000, 3),
                **{k: v for k, v in j.items() if k != "claims"},
                "claims": j["claims"]["objects"], "negated_claims": j["claims"]["negated"]}


def show(rec):
    print(f"\n{Path(rec['image']).name}: {rec['caption']}")
    print(f"  YOLO found: {', '.join(rec['detected_classes']) or '-'}")
    for c in rec["claim_results"]:
        mark = {"supported": "OK ", "unsupported_by_verifier": "NO ", "out_of_yolo_vocabulary": " ? "}[c["status"]]
        print(f"  [{mark}] {c['claim_normalized']}")
    if rec["negated_claims"]:
        print(f"  negated (not checked): {', '.join(rec['negated_claims'])}")
    hs = rec["hallucination_score"]
    print(f"  hallucination score: {'n/a (no checkable claim)' if hs is None else f'{hs:.2f}'}"
          f"  ({rec['unsupported_claim_count']}/{rec['checked_claim_count']} checked claims unsupported)"
          f"  | YOLO {rec['detector_latency_ms']} ms, judge {rec['judge_latency_ms']} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image"); ap.add_argument("--caption")
    ap.add_argument("--captions", help="JSONL from run_captions_unified.py (fields: image, caption)")
    ap.add_argument("--out", help="output JSONL for --captions")
    ap.add_argument("--detector", default="yolov8x.pt", help="yolov8n/s/m/x.pt (downloads on first use)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: auto)")
    a = ap.parse_args()

    judge = Judge(a.detector, a.conf, a.device)
    if a.image and a.caption:
        show(judge(a.image, a.caption))
    elif a.captions and a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        caps = [json.loads(l) for l in open(a.captions) if l.strip()]
        n_chk = n_uns = 0
        with out.open("w") as f:
            for i, c in enumerate(caps, 1):
                img = c["image"] if Path(c["image"]).exists() else str(Path("val2017") / Path(c["image"]).name)
                rec = judge(img, c.get("caption") or "")
                rec["captioner"] = (c.get("run_info") or {}).get("model")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chk += rec["checked_claim_count"]; n_uns += rec["unsupported_claim_count"]
                if i % 100 == 0:
                    print(f"{i}/{len(caps)}")
        print(f"judged {len(caps)} captions | checked claims {n_chk} | flagged {n_uns / max(n_chk, 1):.3f} | saved {out}")
    else:
        ap.error("give --image and --caption, or --captions and --out")


if __name__ == "__main__":
    main()
