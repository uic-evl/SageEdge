"""
Paired differences in CHAIR-i between captioners (same 1,000 images, COCO GT).

CHAIR-i only depends on the claims and COCO GT (not on the detector), so the
claims are taken from the parser directly. 95% CI: paired bootstrap over images.

Usage (from v3/):
  python captioner_pairs.py --coco_ann annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

from evaluate import image_id, load_coco_gt
from judge_v3.parse_caption_v3 import COCO_SET, parse_caption

PAIRS = [("gemma4", "gemma3n"), ("moondream", "gemma4"), ("llava", "moondream"),
         ("moondream", "gemma3n"), ("llava", "gemma4"), ("llava", "gemma3n")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", default="annotations/instances_val2017.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    gt = load_coco_gt(a.coco_ann)
    hal = {}   # captioner -> {image_id: [is_hallucinated per COCO claim]}
    for m in {x for p in PAIRS for x in p}:
        hal[m] = {}
        for l in open(Path(a.results) / f"{m}.jsonl"):
            c = json.loads(l); i = image_id(c["image"])
            claims = [o for o in parse_caption(c.get("caption") or "")["objects"] if o in COCO_SET]
            hal[m][i] = [o not in gt[i] for o in claims]
    ids = sorted(set.intersection(*[set(v) for v in hal.values()]))

    def chair(m, sel):
        t = [x for i in sel for x in hal[m][i]]
        return 100 * sum(t) / len(t)

    rng = random.Random(0)
    print(f"{'comparison':22s} delta CHAIR-i (pp)  95% CI")
    for x, y in PAIRS:
        d = sorted(chair(x, s) - chair(y, s) for s in (rng.choices(ids, k=len(ids)) for _ in range(a.n_boot)))
        lo, hi = d[int(.025 * a.n_boot)], d[int(.975 * a.n_boot) - 1]
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"{x + ' - ' + y:22s} {chair(x, ids) - chair(y, ids):+5.1f}  [{lo:+.1f}, {hi:+.1f}] {sig}")


if __name__ == "__main__":
    main()
