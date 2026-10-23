"""
Step 4b: does image retrieval add signal to the rule judge?  (rule-only vs rule+retrieval ablation)

For every COCO-class claim (image i, class c) of the four frozen V3 captioners:
  yolo_conf   max YOLO confidence for class c in image i (results/detections, conf >= 0.05; 0 if absent)
  prior_k     share of the k nearest bank images (COCO train2017, image embedding, cosine) whose GT contains c
  prior_k_w   same, weighted by cosine similarity
Truth: hallucination = c not in the COCO GT of image i (secondary: blinded human labels, 100 images).

Judges (flag = predicted hallucination), all evaluated on the same claims:
  rule        YOLO class detected with conf >= 0.25 (the paper judge; no training)
  yolo-lr     logistic regression on yolo_conf            -> isolates "learned threshold" gains
  retr-lr     logistic regression on the retrieval prior only
  yolo+retr   logistic regression on yolo_conf, 1[detected], prior, prior x yolo_conf
Learned judges: 5-fold cross-validation grouped by image (all claims of an image in one fold),
the decision threshold is chosen on the training folds (max F1), out-of-fold predictions are pooled.
Retrieval adds value only if yolo+retr beats yolo-lr (not just the fixed-threshold rule).

Reported: F1/precision/recall/kappa, ROC-AUC and average precision; paired bootstrap over images
(B = 2000, seed 0) for F1 and AUC differences; sensitivity over k in {5, 10, 20, 50}; F1 vs human labels.

Usage (from v3/, venv_gpu or any venv with numpy, scikit-learn, faiss):
  python retrieval/eval_retrieval.py --coco_ann ../annotations/instances_val2017.json --model siglip
Output: runs/retrieval/<model>/ablation.csv, k_sensitivity.csv
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from evaluate import load_coco_gt, prf_kappa                        # noqa: E402
from judge_v3.parse_caption_v3 import COCO_SET, parse_caption      # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
KS = [5, 10, 20, 50]
MAIN_K = 10


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--model", default="siglip")
    ap.add_argument("--detectors", nargs="+", default=["yolov8n", "yolov8x"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--suffix", default="", help="e.g. _limit200 for the test run")
    a = ap.parse_args()

    import numpy as np, faiss
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    d = V3 / "runs/retrieval" / a.model
    bank = np.load(d / f"bank_emb{a.suffix}.npy"); ev = np.load(d / f"eval_emb{a.suffix}.npy")
    bank_files = (d / f"bank_files{a.suffix}.txt").read_text().split()
    eval_files = (d / f"eval_files{a.suffix}.txt").read_text().split()
    meta = {m["file"]: set(m["classes"]) for m in jl(V3 / "runs/retrieval/bank_meta.jsonl")}
    bank_cls = [meta[f] for f in bank_files]
    assert not set(bank_files) & set(eval_files), "bank and evaluation images overlap"

    index = faiss.IndexFlatIP(bank.shape[1]); index.add(bank)
    sims, nn = index.search(ev, max(KS))
    neigh = {f: (sims[i], nn[i]) for i, f in enumerate(eval_files)}

    def prior(f, c, k, weighted):
        s, idx = neigh[f]
        hit = np.array([c in bank_cls[j] for j in idx[:k]], dtype=float)
        if not weighted:
            return float(hit.mean())
        w = np.clip(s[:k], 1e-6, None)
        return float((w * hit).sum() / w.sum())

    gt = load_coco_gt(a.coco_ann)
    human = {(r["image"], r["object"]): r["label"] for r in csv.DictReader(open(V3 / "data/human_labels_100.csv"))}
    det = {}
    for dn in a.detectors:
        det[dn] = {}
        for r in jl(V3 / f"results/detections/{dn}.jsonl"):
            m: dict[str, float] = {}
            for x in r["detections"]:
                m[x["class"]] = max(m.get(x["class"], 0.0), x["conf"])
            det[dn][Path(r["image"]).name] = m

    claims = []
    for cap in CAPTIONERS:
        for r in jl(V3 / f"results/{cap}.jsonl"):
            f = Path(r["image"]).name
            if f not in neigh:
                continue
            for c in parse_caption(r.get("caption") or "")["objects"]:
                if c not in COCO_SET:
                    continue
                h = human.get((f, c))
                claims.append({"cap": cap, "img": f, "cls": c, "y": int(c not in gt[int(Path(f).stem)]),
                               "human": {"absent": 1, "present": 0}.get(h),
                               **{f"conf_{dn}": det[dn].get(f, {}).get(c, 0.0) for dn in a.detectors},
                               **{f"p{k}": prior(f, c, k, False) for k in KS},
                               **{f"p{k}w": prior(f, c, k, True) for k in KS}})
    y = np.array([c["y"] for c in claims])
    imgs = sorted({c["img"] for c in claims})
    rng = random.Random(0); rng.shuffle(imgs)
    fold_of = {f: i % a.folds for i, f in enumerate(imgs)}
    fold = np.array([fold_of[c["img"]] for c in claims])
    print(f"model {a.model}: bank {len(bank)} images, eval {len(eval_files)} images, claims {len(claims)} "
          f"(GT-hallucinated {y.sum()} = {y.mean():.1%}), {a.folds}-fold CV grouped by image")

    def feats(kind, dn, pk):
        conf = np.array([c[f"conf_{dn}"] for c in claims]) if dn else None
        p = np.array([c[pk] for c in claims]) if pk else None
        if kind == "yolo":
            return conf[:, None]
        if kind == "retr":
            return p[:, None]
        return np.stack([conf, (conf > 0).astype(float), p, p * conf], 1)

    def best_threshold(score, truth):
        best, thr = -1.0, 0.5
        for t in np.unique(np.round(score, 4)):
            pred = score >= t
            tp = (pred & (truth == 1)).sum(); fp = (pred & (truth == 0)).sum(); fn = (~pred & (truth == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best:
                best, thr = f1, t
        return thr

    def cross_val(X):
        """Out-of-fold probability of hallucination and flags (threshold chosen on training folds)."""
        prob = np.zeros(len(y)); flag = np.zeros(len(y), dtype=bool)
        for k in range(a.folds):
            tr, te = fold != k, fold == k
            lr = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr])
            ptr = lr.predict_proba(X[tr])[:, 1]
            thr = best_threshold(ptr, y[tr])
            prob[te] = lr.predict_proba(X[te])[:, 1]
            flag[te] = prob[te] >= thr
        return prob, flag

    def scores(flag, prob):
        m = prf_kappa(list(zip(flag.tolist(), (y == 1).tolist())))
        m["auc"] = roc_auc_score(y, prob); m["ap"] = average_precision_score(y, prob)
        hm = [(bool(f), c["human"] == 1) for f, c in zip(flag, claims) if c["human"] is not None]
        m["f1_human"] = prf_kappa(hm)["f1"]; m["n_human"] = len(hm)
        return m

    by_img = defaultdict(list)
    for i, c in enumerate(claims):
        by_img[c["img"]].append(i)
    keys = list(by_img)

    def paired(fa, pa, fb, pb):
        """Bootstrap over images: F1(a)-F1(b) and AUC(a)-AUC(b)."""
        r = random.Random(0); df, da = [], []
        for _ in range(a.n_boot):
            idx = np.array([i for k in r.choices(keys, k=len(keys)) for i in by_img[k]])
            yy = y[idx]
            f = lambda fl: prf_kappa(list(zip(fl[idx].tolist(), (yy == 1).tolist())))["f1"]
            df.append(f(fa) - f(fb))
            if 0 < yy.sum() < len(yy):
                da.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
        q = lambda v: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1])
        return q(df), q(da)

    rows = []
    for dn in a.detectors:
        conf = np.array([c[f"conf_{dn}"] for c in claims])
        rule_flag = conf < 0.25
        rule_prob = 1 - conf
        pk = f"p{MAIN_K}w"
        p_y, f_y = cross_val(feats("yolo", dn, None))
        p_r, f_r = cross_val(feats("retr", None, pk))
        p_b, f_b = cross_val(feats("both", dn, pk))
        res = {"rule (tau 0.25)": (rule_flag, rule_prob), "yolo-lr": (f_y, p_y),
               f"retr-lr (k={MAIN_K}, w)": (f_r, p_r), f"yolo+retr (k={MAIN_K}, w)": (f_b, p_b)}
        print(f"\n== {dn}: F1 / AUC vs COCO GT ({len(y)} claims); paired 95% CI over images")
        for name, (fl, pr) in res.items():
            m = scores(fl, pr)
            (dfr, dar) = paired(fl, pr, rule_flag, rule_prob)
            (dfy, day) = paired(fl, pr, f_y, p_y)
            rows.append({"detector": dn, "judge": name, **{k: round(v, 4) for k, v in m.items()},
                         "dF1_vs_rule": round(m["f1"] - scores(rule_flag, rule_prob)["f1"], 4),
                         "dF1_vs_rule_ci95": f"[{dfr[0]:.3f}, {dfr[1]:.3f}]",
                         "dF1_vs_yolo_lr": round(m["f1"] - scores(f_y, p_y)["f1"], 4),
                         "dF1_vs_yolo_lr_ci95": f"[{dfy[0]:.3f}, {dfy[1]:.3f}]",
                         "dAUC_vs_yolo_lr": round(m["auc"] - scores(f_y, p_y)["auc"], 4),
                         "dAUC_vs_yolo_lr_ci95": f"[{day[0]:.3f}, {day[1]:.3f}]"})
            x = rows[-1]
            print(f"  {name:22s} F1 {x['f1']:.3f} P {x['precision']:.3f} R {x['recall']:.3f} AUC {x['auc']:.3f} "
                  f"AP {x['ap']:.3f} | human F1 {x['f1_human']:.3f} | vs rule dF1 {x['dF1_vs_rule']:+.3f} "
                  f"{x['dF1_vs_rule_ci95']} | vs yolo-lr dF1 {x['dF1_vs_yolo_lr']:+.3f} {x['dF1_vs_yolo_lr_ci95']} "
                  f"dAUC {x['dAUC_vs_yolo_lr']:+.3f} {x['dAUC_vs_yolo_lr_ci95']}")

    # ---------------- k sensitivity (threshold-free AUC, so no selection on the test data)
    ks_rows = []
    print(f"\n== k sensitivity: AUC of the raw prior (retrieval only) and of yolo+retr (CV), {a.model}")
    for k in KS:
        for w in ("", "w"):
            pk = f"p{k}{w}"
            p = np.array([c[pk] for c in claims])
            row = {"k": k, "weighted": bool(w), "auc_prior_only": round(roc_auc_score(y, 1 - p), 4)}
            for dn in a.detectors:
                pb, _ = cross_val(feats("both", dn, pk))
                row[f"auc_yolo+retr_{dn}"] = round(roc_auc_score(y, pb), 4)
            ks_rows.append(row)
            print("  " + "  ".join(f"{kk} {vv}" for kk, vv in row.items()))

    for name, rs in (("ablation.csv", rows), ("k_sensitivity.csv", ks_rows)):
        with open(d / name.replace(".csv", f"{a.suffix}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    print(f"\nsaved {d}/ablation{a.suffix}.csv, k_sensitivity{a.suffix}.csv")


if __name__ == "__main__":
    main()
