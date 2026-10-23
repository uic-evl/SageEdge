"""
Paired captioner comparison for grounding (precision, coverage, grounded-F1).

Reads runs/grounding/per_caption_yolov8x.csv (written by grounding_metrics.py) and,
for every pair of captioners, computes the difference of the corpus (micro) metric
on the same 1,000 images, with a paired bootstrap 95% CI (resampling images,
B = 2000, seed 0). Both the COCO-GT version and the label-free judge (YOLOv8x)
version are reported, so we can see whether the judge reproduces the GT ranking.

Usage (from v3/):
  python grounding/captioner_pairs_grounding.py
Output: runs/grounding/captioner_pairs.csv
"""
from __future__ import annotations

import argparse, csv, itertools, random
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
NAN = float("nan")
METRICS = {  # name: (numerator, denominator)
    "prec_gt": ("sup_gt", "n"), "cov_gt": ("gt_hit", "gt_n"), "cov_gt_salient": ("sal_hit", "sal_n"),
    "prec_judge": ("sup_judge", "n"), "cov_judge": ("det_hit", "det_n"),
}


def corpus(rows):
    S = lambda k: sum(r[k] for r in rows)
    out = {m: (S(a) / S(b) if S(b) else NAN) for m, (a, b) in METRICS.items()}
    f = lambda p, c: 2 * p * c / (p + c) if p == p and c == c and p + c else NAN
    out["gf1_gt"] = f(out["prec_gt"], out["cov_gt"])
    out["gf1_judge"] = f(out["prec_judge"], out["cov_judge"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_caption", default=str(V3 / "runs/grounding/per_caption_yolov8x.csv"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=str(V3 / "runs/grounding/captioner_pairs.csv"))
    a = ap.parse_args()

    data: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in csv.DictReader(open(a.per_caption)):
        data[r["captioner"]][r["image"]] = {k: int(v) for k, v in r.items() if k not in ("captioner", "image")}
    caps = sorted(data)
    keys = list(METRICS) + ["gf1_gt", "gf1_judge"]

    print("Corpus values (%):")
    print("captioner | " + "  ".join(f"{k:>14s}" for k in keys))
    for c in caps:
        v = corpus(list(data[c].values()))
        print(f"{c:9s} | " + "  ".join(f"{100 * v[k]:14.1f}" for k in keys))

    rows = []
    print("\nPaired differences A - B (pp, 95% CI); * = CI excludes 0")
    for c1, c2 in itertools.combinations(caps, 2):
        ids = sorted(set(data[c1]) & set(data[c2]))
        A, B = corpus([data[c1][i] for i in ids]), corpus([data[c2][i] for i in ids])
        rng = random.Random(0)
        boots = defaultdict(list)
        for _ in range(a.n_boot):
            pick = rng.choices(ids, k=len(ids))
            x, y = corpus([data[c1][i] for i in pick]), corpus([data[c2][i] for i in pick])
            for k in keys:
                d = x[k] - y[k]
                if d == d:
                    boots[k].append(d)
        row = {"A": c1, "B": c2, "images": len(ids)}
        parts = []
        for k in keys:
            v = sorted(boots[k]); lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
            d = A[k] - B[k]
            row[f"d_{k}"] = round(d, 4); row[f"d_{k}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
            if k in ("prec_gt", "cov_gt", "gf1_gt", "gf1_judge"):
                parts.append(f"{k} {100 * d:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '}")
        rows.append(row)
        print(f"{c1:9s} - {c2:9s}: " + " | ".join(parts))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
