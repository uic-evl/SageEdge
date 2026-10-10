"""
Evaluate hallucination judges against COCO ground truth (and optionally human labels).

Works for both judge outputs:
  * rule judge  (score_rule_judge.py)  -> claim flagged if status == "unsupported_by_verifier"
  * VLM judge   (run_vlm_judge.py)     -> claim flagged if verdict == "no"

Only COCO-class claims are evaluated (the only ones COCO GT can verify).

Reports, per input file:
  CHAIR-i  : share of checked claims that are truly absent (COCO GT)   -> caption model quality
  CHAIR-s  : share of captions with >= 1 truly absent claim            -> caption model quality
  flag rate, precision, recall, F1, Cohen's kappa of the judge vs GT   -> judge quality
  95% bootstrap CIs (resampling images), optional agreement with human labels.

Usage:
  python evaluate.py --coco_ann annotations/instances_val2017.json \
      --files runs/v3/judged/*.jsonl runs/v3/judge_qwen_*.jsonl \
      [--human human_validation_100_labeled_clean.csv] --out results/eval_summary.csv
"""
from __future__ import annotations

import argparse, csv, json, os, random
from collections import defaultdict
from pathlib import Path

try:
    from judge_v3.parse_caption_v3 import COCO_SET
except ImportError:  # pragma: no cover
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent / "judge_v3"))
    from parse_caption_v3 import COCO_SET


def image_id(path: str) -> int:
    return int(Path(path).stem)


def load_coco_gt(ann_path: str) -> dict[int, set[str]]:
    ann = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    gt: dict[int, set[str]] = defaultdict(set)
    for a in ann["annotations"]:                 # crowd annotations count as present
        gt[a["image_id"]].add(cats[a["category_id"]])
    for im in ann["images"]:
        gt.setdefault(im["id"], set())
    return gt


def claims_from_record(rec: dict) -> list[tuple[str, bool]]:
    """Return [(coco_claim, flagged_as_hallucination)] for one judged caption."""
    if "claim_results" in rec:                                   # rule judge
        return [(c["claim_normalized"], c["status"] == "unsupported_by_verifier")
                for c in rec["claim_results"] if c.get("in_yolo_vocabulary")]
    if "verdicts" in rec:                                        # VLM judge
        if not rec.get("ok", True):
            return []
        return [(c, v == "no") for c, v in (rec.get("verdicts") or {}).items()
                if c in COCO_SET and v in ("yes", "no")]
    raise ValueError("unknown record format")


def prf_kappa(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    tp = sum(f and t for f, t in pairs); fp = sum(f and not t for f, t in pairs)
    fn = sum(t and not f for f, t in pairs); tn = sum(not f and not t for f, t in pairs)
    n = len(pairs) or 1
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "kappa": kappa}


def summarize(per_image: dict[int, list[tuple[bool, bool]]]) -> dict[str, float]:
    pairs = [x for v in per_image.values() for x in v]
    checked = [v for v in per_image.values() if v]
    out = {"images_verifiable": len(checked), "checked_claims": len(pairs),
           "chair_i": sum(t for _, t in pairs) / max(len(pairs), 1),
           "chair_s": sum(any(t for _, t in v) for v in checked) / max(len(checked), 1),
           "judge_flag_rate": sum(f for f, _ in pairs) / max(len(pairs), 1)}
    out.update(prf_kappa(pairs))
    return out


def bootstrap(per_image, keys, n_boot=1000, seed=0):
    rng = random.Random(seed)
    ids = list(per_image)
    samples = defaultdict(list)
    for _ in range(n_boot):
        s = summarize({i: per_image[ids[j]] for i, j in enumerate(rng.choices(range(len(ids)), k=len(ids)))})
        for k in keys:
            if s[k] == s[k]:
                samples[k].append(s[k])
    ci = {}
    for k in keys:
        v = sorted(samples[k])
        ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (float("nan"),) * 2
    return ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--human", help="CSV with image, model, claim_normalized, human_label")
    ap.add_argument("--out", default="results/eval_summary.csv")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()

    gt = load_coco_gt(a.coco_ann)
    human = {}
    if a.human:
        for r in csv.DictReader(open(a.human)):
            if r.get("human_label") in ("supported", "unsupported"):
                human[(image_id(r["image"]), r["claim_normalized"])] = r["human_label"] == "unsupported"

    rows = []
    for fpath in a.files:
        per_image: dict[int, list[tuple[bool, bool]]] = {}
        hum_pairs = []
        for line in open(fpath):
            if not line.strip():
                continue
            rec = json.loads(line)
            iid = image_id(rec["image"])
            pairs = []
            for claim, flagged in claims_from_record(rec):
                pairs.append((flagged, claim not in gt[iid]))
                if (iid, claim) in human:
                    hum_pairs.append((flagged, human[(iid, claim)]))
            per_image[iid] = pairs
        s = summarize(per_image)
        ci = bootstrap(per_image, ["chair_i", "judge_flag_rate", "precision", "recall", "f1"], a.n_boot)
        row = {"file": os.path.basename(fpath), **{k: round(v, 4) for k, v in s.items()}}
        for k, (lo, hi) in ci.items():
            row[f"{k}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        if human:
            h = prf_kappa(hum_pairs)
            row.update({"human_n": len(hum_pairs), **{f"human_{k}": round(v, 4) for k, v in h.items()}})
        rows.append(row)
        print(f"{row['file']}: CHAIRi {row['chair_i']:.3f} {row['chair_i_ci95']} | CHAIRs {row['chair_s']:.3f} | "
              f"judge P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f} {row['f1_ci95']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "file", k)))
        w.writeheader(); w.writerows(rows)
    print("saved", a.out)


if __name__ == "__main__":
    main()
