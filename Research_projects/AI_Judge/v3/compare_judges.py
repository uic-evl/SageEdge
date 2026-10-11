"""
Compare the YOLO rule judge (YOLOv8 n/s/m/x) with the Qwen3-VL-32B judge on the
100 images that Qwen judged (4 captioners -> 400 captions), using COCO GT as truth.

Rule-judge verdicts are recomputed here from the caption and detection files
(deterministic, < 1 s), so no intermediate "judged" files are needed.
Confidence intervals: 95% percentile bootstrap that resamples captions (image x captioner).

Usage (from v3/):
  python compare_judges.py --coco_ann annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse, json, random
from collections import defaultdict
from pathlib import Path

from evaluate import image_id, load_coco_gt, prf_kappa
from judge_v3.parse_caption_v3 import COCO_SET
from judge_v3.rule_judge_v3 import rule_based_judge_v3

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
DETECTORS = ["n", "s", "m", "x"]


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_judges(results="results", min_conf=0.25):
    """Return rows: one per (captioner, image, claim) judged by Qwen, with every judge's flag.

    flag = True means the judge says the object is NOT in the image (hallucination).
    For OOV claims the YOLO flags are None (YOLO cannot check them).
    """
    res = Path(results)
    dets = {d: {Path(r["image"]).name: r["detections"] for r in jl(res / "detections" / f"yolov8{d}.jsonl")}
            for d in DETECTORS}
    rows = []
    for m in CAPTIONERS:
        caps = {Path(c["image"]).name: c.get("caption") or "" for c in jl(res / f"{m}.jsonl")}
        for q in jl(res / f"judge_qwen_{m}.jsonl"):
            if not q.get("ok", True):
                continue
            b = Path(q["image"]).name
            yolo = {}
            for d in DETECTORS:
                j = rule_based_judge_v3(caps[b], dets[d][b], min_conf=min_conf)
                yolo[d] = {c["claim_normalized"]: c["status"] == "unsupported_by_verifier"
                           for c in j["claim_results"] if c["in_yolo_vocabulary"]}
            for o, v in (q.get("verdicts") or {}).items():
                if v not in ("yes", "no"):
                    continue
                rows.append({"cap": m, "b": b, "iid": image_id(b), "o": o, "coco": o in COCO_SET,
                             "qwen": v == "no", **{d: yolo[d].get(o) for d in DETECTORS}})
    return rows


def boot_f1(rows, key, truth, rng, B=1000):
    by = defaultdict(list)
    for r in rows:
        by[(r["cap"], r["b"])].append((r[key], truth(r)))
    ks, v = list(by), []
    for _ in range(B):
        f = prf_kappa([p for k in rng.choices(ks, k=len(ks)) for p in by[k]])["f1"]
        if f == f:
            v.append(f)
    v.sort()
    return v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]


def paired_diff(rows, a, b, truth, rng, B=2000):
    by = defaultdict(list)
    for r in rows:
        by[(r["cap"], r["b"])].append(r)
    ks, d = list(by), []
    for _ in range(B):
        s = [r for k in rng.choices(ks, k=len(ks)) for r in by[k]]
        d.append(prf_kappa([(r[a], truth(r)) for r in s])["f1"] - prf_kappa([(r[b], truth(r)) for r in s])["f1"])
    d.sort()
    return sum(d) / len(d), d[int(.025 * B)], d[int(.975 * B) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", default="annotations/instances_val2017.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--min_conf", type=float, default=0.25)
    a = ap.parse_args()

    gt = load_coco_gt(a.coco_ann)
    rows = [r for r in load_judges(a.results, a.min_conf) if r["coco"]]
    truth = lambda r: r["o"] not in gt[r["iid"]]
    rng = random.Random(0)

    print(f"COCO-class claims: {len(rows)} (100 images x {len(CAPTIONERS)} captioners), "
          f"GT-hallucinated {sum(truth(r) for r in rows)}")
    for key, name in [("n", "YOLOv8n"), ("s", "YOLOv8s"), ("m", "YOLOv8m"), ("x", "YOLOv8x"), ("qwen", "Qwen3-VL-32B")]:
        p = [(r[key], truth(r)) for r in rows]
        k = prf_kappa(p); lo, hi = boot_f1(rows, key, truth, rng)
        print(f"  {name:13s} flag {sum(f for f, _ in p) / len(p):.3f}  P {k['precision']:.3f}  R {k['recall']:.3f}  "
              f"F1 {k['f1']:.3f} [{lo:.3f}, {hi:.3f}]  kappa {k['kappa']:.3f}")
    agree = sum(r["x"] == r["qwen"] for r in rows) / len(rows)
    mean, lo, hi = paired_diff(rows, "x", "qwen", truth, rng)
    print(f"  YOLOv8x vs Qwen: verdict agreement {agree:.3f}; F1 difference {mean:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
