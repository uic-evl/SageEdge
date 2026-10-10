"""
Score captions with the YOLO rule judge (v3), offline.

Inputs:
  --captions   output of run_captions_unified.py      (runs/v3/<model>.jsonl)
  --detections output of run_detector_ablation.py     (runs/v3/detections/yolov8n.jsonl)
  --min_conf   YOLO confidence threshold applied here (default 0.25)

Usage:
  python score_rule_judge.py --captions runs/v3/llava.jsonl \
      --detections runs/v3/detections/yolov8n.jsonl --out runs/v3/judged/llava_yolov8n.jsonl

Prints a summary: claims, verifiable images, judge-flagged hallucination rate.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from judge_v3.rule_judge_v3 import rule_based_judge_v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_conf", type=float, default=0.25)
    a = ap.parse_args()

    dets = {}
    for l in open(a.detections):
        if l.strip():
            r = json.loads(l)
            dets[str(Path(r["image"]).name)] = r
    caps = [json.loads(l) for l in open(a.captions) if l.strip()]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_chk = n_uns = n_ver = n_missing = 0
    with out.open("w") as f:
        for c in caps:
            d = dets.get(Path(c["image"]).name)
            if d is None:
                n_missing += 1
                continue
            t0 = time.perf_counter()
            j = rule_based_judge_v3(c.get("caption") or "", d["detections"], min_conf=a.min_conf)
            judge_ms = (time.perf_counter() - t0) * 1000
            rec = {"image": c["image"], "caption": c.get("caption"),
                   "captioner": (c.get("run_info") or {}).get("model"),
                   "detector": d.get("model"), "min_conf": a.min_conf,
                   "detector_latency_ms": d.get("latency_ms"), "judge_latency_ms": round(judge_ms, 3),
                   **{k: v for k, v in j.items() if k != "claims"},
                   "claims": j["claims"]["objects"], "negated_claims": j["claims"]["negated"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_chk += j["checked_claim_count"]; n_uns += j["unsupported_claim_count"]; n_ver += j["verifiable"]
    print(f"images scored {len(caps) - n_missing} (missing detections: {n_missing}) | "
          f"verifiable {n_ver} | checked claims {n_chk} | "
          f"judge-flagged hallucination rate {n_uns / max(n_chk, 1):.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
