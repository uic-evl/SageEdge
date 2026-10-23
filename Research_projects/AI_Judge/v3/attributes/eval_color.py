"""
Attribute hallucination, step 3: evaluate the color judge against the blinded human color labels.

Reference: human label per item (present / absent; unsure items are dropped). Positive class = hallucinated
color claim (human: absent). Each human-absent item is split with COCO GT into
"object absent" (class not annotated in the image) and "wrong color" (object annotated, color wrong).

Judges compared (all flag a claim as hallucinated when it is not supported):
  object only   : YOLOv8x has a box of the class (conf >= 0.25); color ignored = the object judge of the paper
  color, crop   : object box AND the claimed color is the top color of a box crop (the color judge)
  color, image  : the claimed color is the top color for "a photo of a {color} {object}" on the whole image
Metrics: P / R / F1 for hallucinated claims, Cohen's kappa, AUC of the continuous score, 95% bootstrap CI over images.
Also: color hallucination rate per captioner (human labels, and judge estimate on all 1,000 images).

Usage (from v3/):  python attributes/eval_color.py --labels color_labels_mahsa.csv --coco_ann ../annotations/instances_val2017.json
                   [--labels2 color_labels_other.csv  for inter-annotator agreement]
Output: printed tables + runs/attributes/color_eval_<model>.json
"""
from __future__ import annotations

import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
NAN = float("nan")


def read_labels(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get("label") in ("present", "absent", "unsure"):
            out[int(r["pair_id"])] = r["label"]
    return out


def kappa(a, b):
    n = len(a)
    if not n:
        return NAN
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else NAN


def auc(y, s):
    pos = [v for v, t in zip(s, y) if t]; neg = [v for v, t in zip(s, y) if not t]
    if not pos or not neg:
        return NAN
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def prf(y, p):
    tp = sum(a and b for a, b in zip(y, p)); fp = sum((not a) and b for a, b in zip(y, p))
    fn = sum(a and (not b) for a, b in zip(y, p))
    P = tp / (tp + fp) if tp + fp else NAN; R = tp / (tp + fn) if tp + fn else NAN
    F = 2 * P * R / (P + R) if P == P and R == R and P + R else NAN
    return P, R, F


def boot(items, fn, n_boot, seed=0):
    by_img = defaultdict(list)
    for it in items:
        by_img[it["image"]].append(it)
    imgs = list(by_img); rng = random.Random(seed); vals = []
    for _ in range(n_boot):
        s = [it for im in (rng.choice(imgs) for _ in imgs) for it in by_img[im]]
        v = fn(s)
        if v == v:
            vals.append(v)
    vals.sort()
    return (vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]) if vals else (NAN, NAN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--labels2", default="")
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--model", default="siglip")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    ann = json.load(open(a.coco_ann))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    fname = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = defaultdict(set)
    for x in ann["annotations"]:
        gt[fname[x["image_id"]]].add(cats[x["category_id"]])

    scores = {(r["image"], r["color"], r["object"]): r
              for r in map(json.loads, open(V3 / f"runs/attributes/color_scores_{a.model}.jsonl"))}
    pairs = list(csv.DictReader(open(V3 / "data/color_pairs.csv")))
    lab = read_labels(a.labels)
    print(f"labels: {len(lab)} of {len(pairs)} items "
          f"({sum(v == 'present' for v in lab.values())} present, {sum(v == 'absent' for v in lab.values())} absent, "
          f"{sum(v == 'unsure' for v in lab.values())} unsure)")
    res = {"model": a.model, "n_items": len(pairs), "n_labeled": len(lab)}

    if a.labels2:
        lab2 = read_labels(a.labels2)
        common = sorted(set(lab) & set(lab2))
        k3 = kappa([lab[i] for i in common], [lab2[i] for i in common])
        both = [i for i in common if "unsure" not in (lab[i], lab2[i])]
        k2 = kappa([lab[i] for i in both], [lab2[i] for i in both])
        print(f"inter-annotator: {len(common)} common items, kappa (3 labels) {k3:.3f}, "
              f"kappa present/absent ({len(both)} items) {k2:.3f}")
        res["inter_annotator"] = {"n": len(common), "kappa3": k3, "kappa2": k2, "n2": len(both)}

    items = []
    for p in pairs:
        pid = int(p["pair_id"]); key = (p["image"], p["color"], p["coco_class"])
        if lab.get(pid) not in ("present", "absent"):
            continue
        s = scores.get(key)
        if s is None:
            raise SystemExit(f"no judge score for {key}: run color_judge.py first")
        items.append({"image": p["image"], "y": lab[pid] == "absent", "caps": p["captioners"].split(";"),
                      "obj_in_gt": p["coco_class"] in gt[p["image"]], "in100": p.get("in_human100") == "1",
                      "obj": s["n_crops"] > 0, "crop": bool(s["crop_support"]), "full": s["full_support"],
                      "s_obj": max((b["conf"] for b in s["boxes"]), default=0.0),
                      "s_crop": s["crop_score"], "s_full": s["full_score"]})
    y = [it["y"] for it in items]
    n_abs = sum(y)
    n_wrong = sum(it["y"] and it["obj_in_gt"] for it in items)
    print(f"evaluated: {len(items)} items on {len({it['image'] for it in items})} images; hallucinated (absent) "
          f"{n_abs} = {100 * n_abs / len(items):.1f}%: object absent {n_abs - n_wrong}, wrong color {n_wrong}")
    res.update({"n_eval": len(items), "n_halluc": n_abs, "n_wrong_color": n_wrong, "n_object_absent": n_abs - n_wrong})

    judges = [("object only (YOLOv8x)", "obj", "s_obj"), ("color, crop (judge)", "crop", "s_crop"),
              ("color, whole image", "full", "s_full")]
    print(f"\n== hallucinated color claims vs human (positive = absent); 95% CI over images")
    res["judges"] = {}
    for nm, k, sk in judges:
        pred = [not it[k] for it in items]
        P, R, F = prf(y, pred)
        K = kappa(y, pred)
        A = auc(y, [-it[sk] for it in items])
        fci = boot(items, lambda s: prf([i["y"] for i in s], [not i[k] for i in s])[2], a.n_boot)
        kci = boot(items, lambda s: kappa([i["y"] for i in s], [not i[k] for i in s]), a.n_boot)
        r_obj = sum(p and t and not it["obj_in_gt"] for p, t, it in zip(pred, y, items)) / max(1, n_abs - n_wrong)
        r_col = sum(p and t and it["obj_in_gt"] for p, t, it in zip(pred, y, items)) / max(1, n_wrong)
        print(f"  {nm:24s} P {P:.3f} R {R:.3f} F1 {F:.3f} [{fci[0]:.3f}, {fci[1]:.3f}]  kappa {K:.3f} "
              f"[{kci[0]:.3f}, {kci[1]:.3f}]  AUC {A:.3f} | recall: object absent {r_obj:.2f}, wrong color {r_col:.2f}")
        res["judges"][k] = {"name": nm, "P": P, "R": R, "F1": F, "F1_ci": fci, "kappa": K, "kappa_ci": kci, "AUC": A,
                            "recall_object_absent": r_obj, "recall_wrong_color": r_col}
    d = boot(items, lambda s: prf([i["y"] for i in s], [not i["crop"] for i in s])[2]
             - prf([i["y"] for i in s], [not i["obj"] for i in s])[2], a.n_boot)
    print(f"  F1(color crop) - F1(object only): 95% CI [{d[0]:+.3f}, {d[1]:+.3f}]")
    res["dF1_crop_minus_obj_ci"] = d

    print("\n== color hallucination rate per captioner")
    allc = list(csv.DictReader(open(V3 / "data/color_claims_all.csv")))
    res["captioners"] = {}
    for c in CAPTIONERS:
        mine = [it for it in items if c in it["caps"]]
        hr = sum(it["y"] for it in mine) / len(mine) if mine else NAN
        cl = [scores[(r["image"], r["color"], r["object"])] for r in allc if r["captioner"] == c]
        jr = sum(not s["crop_support"] for s in cl) / len(cl) if cl else NAN
        wc = [s for s, r in zip(cl, [r for r in allc if r["captioner"] == c]) if r["object"] in gt[r["image"]]]
        jw = sum(not s["crop_support"] for s in wc if s["n_crops"]) / max(1, sum(s["n_crops"] > 0 for s in wc))
        print(f"  {c:10s} human: {hr * 100:5.1f}% of {len(mine)} labeled | judge, 1,000 images: {len(cl)} claims, "
              f"{jr * 100:5.1f}% unsupported; wrong color where the object is in GT and detected {jw * 100:5.1f}%")
        res["captioners"][c] = {"human_rate": hr, "n_labeled": len(mine), "n_claims_1000": len(cl),
                                "judge_unsupported": jr, "judge_wrong_color_detected": jw}
    out = V3 / f"runs/attributes/color_eval_{a.model}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
