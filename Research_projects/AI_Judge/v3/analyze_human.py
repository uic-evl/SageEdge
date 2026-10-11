"""
Blinded human study: COCO GT, the YOLO rule judge and the Qwen3-VL-32B judge
against human labels (100 images, every object claimed by any captioner).

Human labels: data/human_labels_100.csv (pair_id, image, object, label, annotator),
label in {present, absent, unsure}. A claim is a human-labeled hallucination if its
object was labeled absent; "unsure" pairs are excluded. Because a label is an
image-level fact, it applies to every caption that makes that claim.

Confidence intervals: 95% percentile bootstrap that resamples captions
(image x captioner), the same unit as compare_judges.py.

Usage (from v3/):
  python analyze_human.py --coco_ann annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse, collections, csv, random

from compare_judges import CAPTIONERS, load_judges
from evaluate import image_id, load_coco_gt, prf_kappa
from judge_v3.parse_caption_v3 import COCO_SET


def by_caption(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["cap"], r["b"])].append(r)
    return g


def boot(rows, stat, rng, B=1000):
    g = by_caption(rows); ks = list(g); v = []
    for _ in range(B):
        x = stat([r for k in rng.choices(ks, k=len(ks)) for r in g[k]])
        if x == x:
            v.append(x)
    v.sort()
    return v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_ann", default="annotations/instances_val2017.json")
    ap.add_argument("--labels", default="data/human_labels_100.csv")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    gt = load_coco_gt(a.coco_ann)
    H = {(r["image"], r["object"]): r["label"] for r in csv.DictReader(open(a.labels))}
    rng = random.Random(0)
    print(f"human labels: {len(H)} pairs, {len({k[0] for k in H})} images, "
          + ", ".join(f"{l} {sum(v == l for v in H.values())}" for l in ("present", "absent", "unsure")))

    # ---------- 1. COCO GT vs human on unique (image, object) pairs
    pairs = [(k, v) for k, v in H.items() if k[1] in COCO_SET and v != "unsure"]
    g = [k[1] not in gt[image_id(k[0])] for k, _ in pairs]
    h = [v == "absent" for _, v in pairs]
    k = prf_kappa(list(zip(g, h)))
    seen = sum(gi and not hi for gi, hi in zip(g, h)); missed = sum(hi and not gi for gi, hi in zip(g, h))
    print(f"\n[1] unique COCO-class pairs {len(pairs)}: GT-absent {sum(g)}, human-absent {sum(h)}; "
          f"kappa(GT, human) {k['kappa']:.3f}")
    print(f"    GT-absent but seen by human: {seen};  GT-present but labeled absent: {missed}")

    # ---------- 2. claim level: judges vs human
    rows = []
    for r in load_judges(a.results):
        lab = H.get((r["b"], r["o"]))
        if lab is None or lab == "unsure":
            continue
        r["hum"] = lab == "absent"
        r["gt"] = (r["o"] not in gt[r["iid"]]) if r["coco"] else None
        rows.append(r)
    coco = [r for r in rows if r["coco"]]
    print(f"\n[2] COCO-class claims with a human label: {len(coco)}; human-hallucinated "
          f"{sum(r['hum'] for r in coco)} ({sum(r['hum'] for r in coco) / len(coco):.3f})")
    f1 = lambda key: (lambda s: prf_kappa([(r[key], r["hum"]) for r in s])["f1"])
    for key, name in [("n", "YOLOv8n"), ("x", "YOLOv8x"), ("qwen", "Qwen3-VL-32B"), ("gt", "COCO GT")]:
        p = [(r[key], r["hum"]) for r in coco]; kk = prf_kappa(p); lo, hi = boot(coco, f1(key), rng)
        print(f"    {name:13s} flag {sum(x for x, _ in p) / len(p):.3f}  acc {sum(x == y for x, y in p) / len(p):.3f}  "
              f"P {kk['precision']:.3f}  R {kk['recall']:.3f}  F1 {kk['f1']:.3f} [{lo:.3f}, {hi:.3f}]  kappa {kk['kappa']:.3f}")
    diff = lambda s: f1("x")(s) - f1("qwen")(s)
    lo, hi = boot(coco, diff, rng, B=2000)
    print(f"    F1(YOLOv8x) - F1(Qwen): {diff(coco):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ---------- 3. OOV claims (only Qwen can judge them)
    oov = [r for r in rows if not r["coco"]]
    p = [(r["qwen"], r["hum"]) for r in oov]; kk = prf_kappa(p)
    print(f"\n[3] OOV claims with a human label: {len(oov)}; human-hallucinated {sum(r['hum'] for r in oov)} "
          f"({sum(r['hum'] for r in oov) / len(oov):.3f})")
    print(f"    Qwen3-VL-32B  flag {sum(x for x, _ in p) / len(p):.3f}  acc {sum(x == y for x, y in p) / len(p):.3f}  "
          f"P {kk['precision']:.3f}  R {kk['recall']:.3f}  F1 {kk['f1']:.3f}  kappa {kk['kappa']:.3f}")

    # ---------- 4. captioner hallucination by human labels
    print("\n[4] captioner hallucination rate by human labels (%), 95% CI")
    rate = lambda s: 100 * sum(r["hum"] for r in s) / len(s) if s else float("nan")
    for m in CAPTIONERS:
        c = [r for r in coco if r["cap"] == m]; al = [r for r in rows if r["cap"] == m]
        lc, hc = boot(c, rate, rng); la, ha = boot(al, rate, rng)
        gtr = 100 * sum(r["gt"] for r in c) / len(c)
        print(f"    {m:9s} COCO claims {rate(c):5.1f} [{lc:.1f}, {hc:.1f}] (COCO GT {gtr:.1f})   "
              f"all claims {rate(al):5.1f} [{la:.1f}, {ha:.1f}]")


if __name__ == "__main__":
    main()
