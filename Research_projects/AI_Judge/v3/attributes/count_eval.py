"""
Attribute hallucination, counts: can the judge check number claims such as "two dogs" or "three people"?

Count claims: a number word (two .. ten) before a COCO object, with at most two
modifiers between them ("two small dogs", "three white sheep"); the object must not be negated and must
not modify a following noun ("two ski poles"). "Two slices of pizza" is not a pizza count (the count
belongs to "slices"). Vague words (several, many, a few, a couple, a pair) are not counts. "One" is left out:
in the captions it mostly picks one of several objects ("one man is wearing ..."), and an over-count of one
object is the same as an object hallucination. Digits are left out because they are mostly times ("6 o'clock").

Reference (COCO ground truth): the number of instance annotations of the class in the image.
  - claims on images with a crowd annotation (iscrowd = 1) of that class are left out (count unknown);
  - a claim is an OVER-COUNT (hallucinated) when the image has fewer instances than claimed
    (this includes an absent object). Saying fewer than COCO annotates is reported as UNDER-COUNT but is not
    counted as a hallucination, because captions often describe only some of the instances
    ("two people sit on a bench" in a busy street);
  - pure count error: over-count while the object is present (at least one instance).
Judge (label-free): the number of boxes of the class with conf >= tau. The claim "n objects" is supported when
the detector finds at least n, i.e. when the n-th most confident box of the class has conf >= tau; the
continuous score is that n-th confidence (0 when there are fewer than n boxes). The judge flags an over-count
when the score is below tau.
Reported for YOLOv8n/s/m/x and tau 0.25 / 0.40 / 0.50: P, R, F1, kappa against COCO over-counts, the AUC of the
score, and the same split into count errors only (object present); per captioner the claim counts, the COCO
over-count rate and the judge's estimate. 95% bootstrap CIs over images (B = 2000, seed 0).

Usage (from v3/, any venv; CPU only, no new model runs):
  python attributes/count_eval.py --coco_ann ../annotations/instances_val2017.json
Output: runs/attributes/count_claims.csv (one row per claim), runs/attributes/count_eval.csv
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from evaluate import prf_kappa                                                               # noqa: E402
from judge_v3.parse_caption_v3 import COCO_SET, _match_phrases, _negated_positions, _tokenize  # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
DETECTORS = ["yolov8n", "yolov8s", "yolov8m", "yolov8x"]
TAUS = [0.25, 0.40, 0.50]
NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
PUNCT = set(".,;:!?")
STOP = {"a", "an", "the", "and", "or", "with", "of", "on", "in", "at", "to", "by", "from", "near", "next", "under",
        "over", "behind", "is", "are", "was", "were", "has", "have", "its", "their", "his", "her", "this", "that",
        "while", "as", "who", "which", "wearing", "holding", "than", "about", "around", "only", "least", "most"}
# nouns that take the count when they follow the object word ("two ski poles", "three bus stops")
HEADS = {"pole", "poles", "boot", "boots", "stop", "stops", "lane", "lanes", "track", "tracks", "station", "stations",
         "seat", "seats", "wheel", "wheels", "door", "doors", "window", "windows", "car", "cars", "rack", "racks",
         "slice", "slices", "piece", "pieces", "leg", "legs", "plate", "plates", "sign", "signs", "light", "lights",
         "player", "players", "mask", "masks", "handle", "handles", "box", "boxes", "bag", "bags", "shop", "shops"}


def count_claims(caption: str):
    """Return [(n, coco_class, phrase)] for number words directly before a COCO object."""
    toks = _tokenize(caption)
    neg = _negated_positions(toks)
    spans = _match_phrases(toks)
    in_span = {p for s, e, _ in spans for p in range(s, e)}
    out = []
    for s, e, label in spans:
        if label not in COCO_SET or any(p in neg for p in range(s, e)):
            continue
        if e < len(toks) and toks[e] in HEADS:
            continue
        j, gap, n = s - 1, 0, None
        while j >= 0:
            t = toks[j]
            if t in NUM:
                n = NUM[t]
                if j > 0 and toks[j - 1] in {"of", "than", "at"}:      # "two of the dogs", "more than two"
                    n = None
                break
            if t in PUNCT or t in STOP or j in in_span or gap >= 2:
                break
            gap += 1
            j -= 1
        if n is not None:
            if label == "orange" and toks[s] == "orange":
                continue
            out.append((n, label, " ".join(toks[j:e])))
    return list(dict.fromkeys(out))


def load_counts(ann_path):
    ann = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    cnt, crowd = defaultdict(Counter), defaultdict(set)
    for a in ann["annotations"]:
        f = f"{a['image_id']:012d}.jpg"
        if a.get("iscrowd", 0):
            crowd[f].add(cats[a["category_id"]])
        else:
            cnt[f][cats[a["category_id"]]] += 1
    return cnt, crowd


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--outdir", default=str(V3 / "runs/attributes"))
    a = ap.parse_args()

    import numpy as np
    from sklearn.metrics import roc_auc_score

    gt_cnt, crowd = load_counts(a.coco_ann)
    confs = {}
    for dn in DETECTORS:
        confs[dn] = {}
        for r in jl(V3 / f"results/detections/{dn}.jsonl"):
            m = defaultdict(list)
            for x in r["detections"]:
                m[x["class"]].append(float(x["conf"]))
            confs[dn][Path(r["image"]).name] = {k: sorted(v, reverse=True) for k, v in m.items()}

    rows, n_crowd = [], 0
    for cap in CAPTIONERS:
        for r in jl(V3 / f"results/{cap}.jsonl"):
            f = Path(r["image"]).name
            for n, cl, phrase in count_claims(r.get("caption") or ""):
                if cl in crowd[f]:
                    n_crowd += 1
                    continue
                g = gt_cnt[f][cl]
                row = {"image": f, "captioner": cap, "class": cl, "n": n, "phrase": phrase, "gt_count": g,
                       "over": int(g < n), "under": int(g > n), "exact": int(g == n), "present": int(g > 0)}
                for dn in DETECTORS:
                    c = confs[dn].get(f, {}).get(cl, [])
                    row[f"score_{dn}"] = round(c[n - 1], 4) if len(c) >= n else 0.0
                    row[f"count25_{dn}"] = sum(x >= 0.25 for x in c)
                rows.append(row)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "count_claims.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    y = np.array([r["over"] for r in rows]); pres = np.array([r["present"] for r in rows])
    imgs = sorted({r["image"] for r in rows})
    by_img = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["image"]].append(i)
    rng = random.Random(0)
    boots = [np.array([i for k in rng.choices(imgs, k=len(imgs)) for i in by_img[k]]) for _ in range(a.n_boot)]
    ci = lambda v: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1])
    nv = Counter(r["n"] for r in rows)
    print(f"{len(rows)} count claims on {len(imgs)} images ({n_crowd} left out: crowd annotation of the class); "
          f"numbers: " + ", ".join(f"{k}: {nv[k]}" for k in sorted(nv)))
    print(f"COCO: exact {np.mean([r['exact'] for r in rows]):.1%}, over-count {y.mean():.1%} "
          f"(object absent {np.mean([r['over'] and not r['present'] for r in rows]):.1%}, "
          f"present but fewer {np.mean([r['over'] and r['present'] for r in rows]):.1%}), "
          f"under-count {np.mean([r['under'] for r in rows]):.1%}")

    out = []
    for cap in ["all"] + CAPTIONERS:
        sel = [i for i, r in enumerate(rows) if cap == "all" or r["captioner"] == cap]
        ys = y[sel]
        rate_b = [y[b][np.isin(b, sel)].mean() for b in boots[:500]] if cap != "all" else [y[b].mean() for b in boots[:500]]
        jx = np.array([rows[i]["score_yolov8x"] < 0.25 for i in sel])
        out.append({"part": "captioner", "captioner": cap, "claims": len(sel),
                    "exact": round(float(np.mean([rows[i]["exact"] for i in sel])), 4),
                    "over_gt": round(float(ys.mean()), 4), "over_gt_ci95": "[%.3f, %.3f]" % ci(rate_b),
                    "over_gt_present": round(float(np.mean([rows[i]["over"] and rows[i]["present"] for i in sel])), 4),
                    "under_gt": round(float(np.mean([rows[i]["under"] for i in sel])), 4),
                    "judge_v8x_flag_rate": round(float(jx.mean()), 4)})
        x = out[-1]
        print(f"  {cap:10s} claims {x['claims']:4d} | exact {x['exact']:.1%} | over-count {x['over_gt']:.1%} {x['over_gt_ci95']} "
              f"(object present {x['over_gt_present']:.1%}) | under-count {x['under_gt']:.1%} | judge v8x flags {x['judge_v8x_flag_rate']:.1%}")

    print("\njudge vs COCO over-counts (positive = over-count)")
    for dn in DETECTORS:
        s = np.array([r[f"score_{dn}"] for r in rows])
        auc = roc_auc_score(y, -s)
        auc_p = roc_auc_score(y[pres == 1], -s[pres == 1]) if 0 < y[pres == 1].sum() < (pres == 1).sum() else float("nan")
        for tau in TAUS:
            fl = s < tau
            m = prf_kappa(list(zip(fl.tolist(), (y == 1).tolist())))
            f1b = [prf_kappa(list(zip(fl[b].tolist(), (y[b] == 1).tolist())))["f1"] for b in boots]
            mp = prf_kappa(list(zip(fl[pres == 1].tolist(), (y[pres == 1] == 1).tolist())))
            out.append({"part": "judge", "detector": dn, "tau": tau, "claims": len(rows), "flag_rate": round(float(fl.mean()), 4),
                        **{k: round(v, 4) for k, v in m.items()}, "f1_ci95": "[%.3f, %.3f]" % ci([v for v in f1b if v == v]),
                        "auc": round(auc, 4), "present_only_claims": int(pres.sum()),
                        "present_only_precision": round(mp["precision"], 4), "present_only_recall": round(mp["recall"], 4),
                        "present_only_f1": round(mp["f1"], 4), "present_only_auc": round(auc_p, 4)})
            x = out[-1]
            print(f"  {dn} tau {tau:.2f}: flag {x['flag_rate']:.3f} P {x['precision']:.3f} R {x['recall']:.3f} "
                  f"F1 {x['f1']:.3f} {x['f1_ci95']} kappa {x['kappa']:.3f} AUC {x['auc']:.3f} | object present only "
                  f"(n {x['present_only_claims']}): P {x['present_only_precision']:.3f} R {x['present_only_recall']:.3f} "
                  f"F1 {x['present_only_f1']:.3f} AUC {x['present_only_auc']:.3f}")
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(outdir / "count_eval.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(f"saved {outdir}/count_claims.csv, count_eval.csv")


if __name__ == "__main__":
    main()
