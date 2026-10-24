"""
Step 4e: does the caption side of the Judge memory help? (joint image + caption retrieval of prior judgments)

Uses the memory built by build_memory.py and the features of eval_retrieval_v2.py. New features per claim:
  sent      image-sentence agreement in the joint space: cosine between the image and the sentence that
            states the claim, raw and relative to the whole caption (cos(image, sentence) - cos(image, caption))
  m_judge   prior judgments: the k most similar past claims of the SAME class from other images
            (similarity = joint [image, sentence] embedding), similarity-weighted share that the judge
            (YOLOv8x, tau 0.25) flagged. Label-free: this is what a Sage node could store.
  m_gt      the same neighbours, but the share that is absent in COCO GT (a labeled memory, for comparison)
Leakage control: 5-fold CV grouped by image (same folds as eval_retrieval_v2.py). For a test claim the memory
holds only training-fold claims; for a training claim it holds training-fold claims of other images.
Judges (logistic regression, threshold chosen on the training folds), YOLOv8n and YOLOv8x:
  yolo-lr | B0 = +text +judge bank (best label-free judge of v2) | +sent | +mem judge | +mem GT |
  +sent +mem judge | B0 +sent +mem judge (all label-free) | B0 +sent +mem GT
Paired bootstrap over images (B = 2000, seed 0) against yolo-lr and against B0.
Caption level (the plan's "aggregate the neighbours' scores"): k nearest past captions of other folds
(joint [image, caption]), weighted mean of their judge scores; Spearman with the caption's COCO-GT
hallucination rate, compared with the caption's own judge score.

Usage (from v3/, any venv with numpy, scipy, faiss, scikit-learn; no GPU):
  python retrieval/eval_memory.py --coco_ann ../annotations/instances_val2017.json --model siglip
Output: runs/memory/<model>/ablation_memory.csv, caption_level.csv
"""
from __future__ import annotations

import argparse, csv, json, random, sqlite3, sys
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from evaluate import load_coco_gt, prf_kappa                                 # noqa: E402
from judge_v3.parse_caption_v3 import COCO_CLASSES, COCO_SET, parse_caption   # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--model", default="siglip", choices=["siglip", "clip"])
    ap.add_argument("--detectors", nargs="+", default=["yolov8n", "yolov8x"])
    ap.add_argument("--retrdir", default=str(V3 / "runs/retrieval"))
    ap.add_argument("--memdir", default=str(V3 / "runs/memory"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--jtau", type=float, default=0.25)
    a = ap.parse_args()
    K = a.k

    import numpy as np, faiss
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    rd = Path(a.retrdir) / a.model; md = Path(a.memdir) / a.model
    gt = load_coco_gt(a.coco_ann)
    human = {(r["image"], r["object"]): r["label"] for r in csv.DictReader(open(V3 / "data/human_labels_100.csv"))}

    # ---------------- memory
    con = sqlite3.connect(md / "memory.sqlite"); con.row_factory = sqlite3.Row
    caps = [dict(r) for r in con.execute("SELECT * FROM captions ORDER BY row_id")]
    cls = [dict(r) for r in con.execute("SELECT * FROM claims ORDER BY row_id")]
    con.close()
    cap_emb = np.load(md / "caption_emb.npy"); claim_emb = np.load(md / "claim_emb.npy"); img_emb = np.load(md / "image_emb.npy")
    assert len(cap_emb) == len(caps) == len(img_emb) and len(claim_emb) == len(cls)

    # ---------------- features of eval_retrieval_v2 (image retrieval prior, image-class text score)
    bank = np.load(rd / "bank_emb.npy"); ev = np.load(rd / "eval_emb.npy")
    bank_files = (rd / "bank_files.txt").read_text().split(); eval_files = (rd / "eval_files.txt").read_text().split()
    assert not set(bank_files) & set(eval_files)
    meta = {m["file"]: set(m["classes"]) for m in jl(Path(a.retrdir) / "bank_meta.jsonl")}
    bank_gt = [meta[f] for f in bank_files]
    bdet = {r["image"]: r["detections"] for r in jl(Path(a.retrdir) / "bank_detections_yolov8x.jsonl")}
    bank_judge = [{x["class"] for x in bdet[f] if x["conf"] >= a.jtau} for f in bank_files]
    cos = np.load(rd / "text_cos.npy")
    zs = np.exp(100 * (cos - cos.max(1, keepdims=True))); zs /= zs.sum(1, keepdims=True)
    cidx = {c: i for i, c in enumerate(COCO_CLASSES)}
    index = faiss.IndexFlatIP(bank.shape[1]); index.add(bank)
    sims, nn = index.search(ev, 10)                 # k = 10 as in eval_retrieval_v2 (image-bank prior unchanged)
    row_of = {f: i for i, f in enumerate(eval_files)}
    assert np.allclose(img_emb, ev[[row_of[c["image"]] for c in caps]]), "memory image embeddings != retrieval eval_emb"

    # the claim set must be exactly the one of eval_retrieval_v2 (same order)
    v2_claims = []
    for cap in CAPTIONERS:
        for r in jl(V3 / f"results/{cap}.jsonl"):
            f = Path(r["image"]).name
            if f in row_of:
                v2_claims += [(f, cap, o) for o in parse_caption(r.get("caption") or "")["objects"] if o in COCO_SET]
    assert v2_claims == [(c["image"], c["captioner"], c["class"]) for c in cls], "memory claims differ from v2 claims"

    def prior(i, c, labels):
        w = np.clip(sims[i], 1e-6, None)
        return float((w * np.array([c in labels[j] for j in nn[i]], dtype=float)).sum() / w.sum())

    det = {}
    for dn in a.detectors:
        det[dn] = {}
        for r in jl(V3 / f"results/detections/{dn}.jsonl"):
            m: dict[str, float] = {}
            for x in r["detections"]:
                m[x["class"]] = max(m.get(x["class"], 0.0), x["conf"])
            det[dn][Path(r["image"]).name] = m

    cap_cos = {c["row_id"]: c["img_cap_cos"] for c in caps}
    claims = []
    for c in cls:
        f, o = c["image"], c["class"]; i = row_of[f]
        claims.append({"img": f, "cap_row": c["caption_row"], "class": o, "y": int(o not in gt[int(Path(f).stem)]),
                       "human": {"absent": 1, "present": 0}.get(human.get((f, o))),
                       **{f"conf_{dn}": det[dn].get(f, {}).get(o, 0.0) for dn in a.detectors},
                       "cos": float(cos[i, cidx[o]]), "zs": float(zs[i, cidx[o]]),
                       "p_gt": prior(i, o, bank_gt), "p_judge": prior(i, o, bank_judge),
                       "sent": c["img_sent_cos"], "sent_rel": c["img_sent_cos"] - cap_cos[c["caption_row"]],
                       "flag": c["flagged"]})
    n = len(claims)
    y = np.array([c["y"] for c in claims]); flag_j = np.array([c["flag"] for c in claims], dtype=float)
    col = lambda k: np.array([c[k] for c in claims], dtype=float)
    imgs = sorted({c["img"] for c in claims}); rng = random.Random(0); rng.shuffle(imgs)
    fold_of = {f: k % a.folds for k, f in enumerate(imgs)}
    fold = np.array([fold_of[c["img"]] for c in claims])
    img_id = np.array([row_of[c["img"]] for c in claims])
    print(f"model {a.model}: {n} claims ({y.mean():.1%} GT-hallucinated, {flag_j.mean():.1%} judge-flagged), "
          f"{len(caps)} captions in memory, k = {K}")

    # ---------------- memory priors (same class, joint [image, sentence] similarity)
    J = (np.concatenate([ev[img_id], claim_emb], 1) / np.sqrt(2)).astype("float32")
    by_class = defaultdict(list)
    for i, c in enumerate(claims):
        by_class[c["class"]].append(i)
    by_class = {k: np.array(v) for k, v in by_class.items()}

    def mem_prior(q_idx, m_idx, k, labels, exclude_same_img):
        """similarity-weighted share of `labels` among the k most similar memory claims (rows m_idx)."""
        out = np.full(len(q_idx), np.nan)
        if len(m_idx) == 0:
            return out
        S = J[q_idx] @ J[m_idx].T
        if exclude_same_img:
            S[img_id[q_idx][:, None] == img_id[m_idx][None, :]] = -np.inf
        kk = min(k, len(m_idx))
        top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        s = np.take_along_axis(S, top, 1); lab = labels[m_idx][top]
        w = np.where(np.isfinite(s), np.clip(s, 1e-6, None), 0.0)
        den = w.sum(1)
        return np.where(den > 0, (w * lab).sum(1) / np.where(den > 0, den, 1), np.nan)

    def fold_mem(k_fold, k=K):
        """memory features for one CV fold: training rows (other training images), test rows (training memory)."""
        mj = np.full(n, np.nan); mg = np.full(n, np.nan)
        for cl, idx in by_class.items():
            tr = idx[fold[idx] != k_fold]; te = idx[fold[idx] == k_fold]
            for labels, arr in ((flag_j, mj), (y.astype(float), mg)):
                if len(te):
                    arr[te] = mem_prior(te, tr, k, labels, False)
                if len(tr):
                    arr[tr] = mem_prior(tr, tr, k, labels, True)
        trm = fold != k_fold
        for arr, labels in ((mj, flag_j), (mg, y.astype(float))):
            arr[np.isnan(arr)] = labels[trm].mean()          # class never seen in other training images
        return {"m_judge": mj, "m_gt": mg}

    mems = [fold_mem(k) for k in range(a.folds)]

    def feats(dn, parts, k_fold):
        conf = col(f"conf_{dn}"); X = [conf, (conf > 0).astype(float)]
        if "text" in parts:
            X += [col("cos"), col("zs"), col("zs") * conf]
        if "p_judge" in parts:
            X += [col("p_judge"), col("p_judge") * conf]
        if "sent" in parts:
            X += [col("sent"), col("sent_rel"), col("sent") * conf]
        for p in ("m_judge", "m_gt"):
            if p in parts:
                X += [mems[k_fold][p], mems[k_fold][p] * conf]
        return np.stack(X, 1)

    def best_thr(score, truth):
        best, thr = -1.0, 0.5
        for t in np.unique(np.round(score, 4)):
            pr = score >= t; tp = (pr & (truth == 1)).sum(); fp = (pr & (truth == 0)).sum(); fn = (~pr & (truth == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best:
                best, thr = f1, t
        return thr

    def cv(dn, parts):
        prob = np.zeros(n); fl = np.zeros(n, dtype=bool)
        for k in range(a.folds):
            X = feats(dn, parts, k); tr, te = fold != k, fold == k
            lr = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
            thr = best_thr(lr.predict_proba(X[tr])[:, 1], y[tr])
            prob[te] = lr.predict_proba(X[te])[:, 1]; fl[te] = prob[te] >= thr
        return prob, fl

    by_img = defaultdict(list)
    for i, c in enumerate(claims):
        by_img[c["img"]].append(i)
    keys = list(by_img)

    def paired(fa, pa, fb, pb):
        r = random.Random(0); df, da = [], []
        for _ in range(a.n_boot):
            idx = np.array([i for kk in r.choices(keys, k=len(keys)) for i in by_img[kk]]); yy = y[idx]
            f1 = lambda fl: prf_kappa(list(zip(fl[idx].tolist(), (yy == 1).tolist())))["f1"]
            df.append(f1(fa) - f1(fb))
            if 0 < yy.sum() < len(yy):
                da.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
        q = lambda v: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1])
        return q(df), q(da)

    def scores(fl, prob):
        m = prf_kappa(list(zip(fl.tolist(), (y == 1).tolist())))
        hm = [(bool(f), c["human"] == 1) for f, c in zip(fl, claims) if c["human"] is not None]
        return {**m, "auc": roc_auc_score(y, prob), "ap": average_precision_score(y, prob), "f1_human": prf_kappa(hm)["f1"]}

    B0 = ("text", "p_judge")
    variants = [("yolo-lr", ()), ("B0 = +text +judge bank", B0), ("+sent", ("sent",)), ("+mem judge", ("m_judge",)),
                ("+mem GT", ("m_gt",)), ("+sent +mem judge", ("sent", "m_judge")),
                ("B0 +sent +mem judge (label-free)", B0 + ("sent", "m_judge")), ("B0 +sent +mem GT", B0 + ("sent", "m_gt"))]
    rows = []
    zero = ((0.0, 0.0), (0.0, 0.0))
    for dn in a.detectors:
        res = {name: cv(dn, parts) for name, parts in variants}
        bp, bf = res["yolo-lr"]; b0p, b0f = res["B0 = +text +judge bank"]
        sb, sb0 = scores(bf, bp), scores(b0f, b0p)
        print(f"\n== {dn} ({a.model}); deltas vs yolo-lr and vs B0, paired 95% CI over images")
        for name, parts in variants:
            p, fl = res[name]; m = scores(fl, p)
            (df, da) = paired(fl, p, bf, bp) if parts else zero
            (df0, da0) = paired(fl, p, b0f, b0p) if parts != B0 else zero
            rows.append({"detector": dn, "model": a.model, "judge": name, **{k: round(v, 4) for k, v in m.items()},
                         "dF1_vs_yololr": round(m["f1"] - sb["f1"], 4), "dF1_vs_yololr_ci95": f"[{df[0]:.3f}, {df[1]:.3f}]",
                         "dAUC_vs_yololr": round(m["auc"] - sb["auc"], 4), "dAUC_vs_yololr_ci95": f"[{da[0]:.3f}, {da[1]:.3f}]",
                         "dF1_vs_B0": round(m["f1"] - sb0["f1"], 4), "dF1_vs_B0_ci95": f"[{df0[0]:.3f}, {df0[1]:.3f}]",
                         "dAUC_vs_B0": round(m["auc"] - sb0["auc"], 4), "dAUC_vs_B0_ci95": f"[{da0[0]:.3f}, {da0[1]:.3f}]"})
            x = rows[-1]
            print(f"  {name:34s} F1 {x['f1']:.3f} AUC {x['auc']:.3f} AP {x['ap']:.3f} human F1 {x['f1_human']:.3f} | "
                  f"vs yolo-lr dF1 {x['dF1_vs_yololr']:+.3f} {x['dF1_vs_yololr_ci95']} dAUC {x['dAUC_vs_yololr']:+.3f} "
                  f"{x['dAUC_vs_yololr_ci95']} | vs B0 dF1 {x['dF1_vs_B0']:+.3f} {x['dF1_vs_B0_ci95']} "
                  f"dAUC {x['dAUC_vs_B0']:+.3f} {x['dAUC_vs_B0_ci95']}", flush=True)

    # single features (memory priors taken out-of-fold: each claim from the fold where it is a test claim)
    oof = lambda mm, key: sum(np.where(fold == k, mm[k][key], 0.0) for k in range(a.folds))
    single = {"image-sentence cos": roc_auc_score(y, -col("sent")),
              "image-sentence cos (relative)": roc_auc_score(y, -col("sent_rel")),
              "memory prior, judge verdicts": roc_auc_score(y, oof(mems, "m_judge")),
              "memory prior, GT labels": roc_auc_score(y, oof(mems, "m_gt")),
              "image-class text zero-shot (v2)": roc_auc_score(y, -col("zs")),
              "judge-bank image prior (v2)": roc_auc_score(y, -col("p_judge"))}
    print("\nsingle-feature AUC: " + " | ".join(f"{k} {v:.3f}" for k, v in single.items()))
    ks = {}
    for k in (5, 10, 20, 50):
        mk = mems if k == K else [fold_mem(f, k) for f in range(a.folds)]
        ks[k] = (roc_auc_score(y, oof(mk, "m_judge")), roc_auc_score(y, oof(mk, "m_gt")))
    print("k sensitivity (AUC of memory prior, judge / GT): " + " | ".join(f"k {k}: {v[0]:.3f} / {v[1]:.3f}" for k, v in ks.items()))
    rows.append({"detector": "-", "model": a.model, "judge": "single-feature AUC", **{k: round(v, 4) for k, v in single.items()},
                 **{f"auc_mem_judge_k{k}": round(v[0], 4) for k, v in ks.items()}, **{f"auc_mem_gt_k{k}": round(v[1], 4) for k, v in ks.items()}})
    keys_ = list(dict.fromkeys(k for r in rows for k in r))
    with open(md / "ablation_memory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys_); w.writeheader(); w.writerows(rows)

    # ---------------- caption level: aggregate the scores of the k nearest past captions
    cap_fold = np.array([fold_of.get(c["image"], -1) for c in caps])
    JC = (np.concatenate([img_emb, cap_emb], 1) / np.sqrt(2)).astype("float32")
    js = np.array([c["judge_score"] if c["judge_score"] is not None else np.nan for c in caps], dtype=float)
    gt_rate = np.full(len(caps), np.nan); chk = defaultdict(list)
    for c in claims:
        chk[c["cap_row"]].append(c["y"])
    for r, v in chk.items():
        gt_rate[r] = np.mean(v)
    mem_cap = np.full(len(caps), np.nan); mem_cap_gt = np.full(len(caps), np.nan)
    for k in range(a.folds):
        te = np.where((cap_fold == k) & ~np.isnan(js))[0]; tr = np.where((cap_fold != k) & (cap_fold >= 0) & ~np.isnan(js))[0]
        S = JC[te] @ JC[tr].T
        top = np.argpartition(-S, K - 1, axis=1)[:, :K]; w = np.clip(np.take_along_axis(S, top, 1), 1e-6, None)
        mem_cap[te] = (w * js[tr][top]).sum(1) / w.sum(1)
        mem_cap_gt[te] = (w * gt_rate[tr][top]).sum(1) / w.sum(1)
    ok = ~np.isnan(gt_rate) & ~np.isnan(js) & ~np.isnan(mem_cap)
    crow = []
    print(f"\n== caption level ({ok.sum()} captions with COCO-class claims): Spearman with the COCO-GT hallucination rate")
    for cap in ["all"] + CAPTIONERS:
        sel = ok & np.array([cap == "all" or c["captioner"] == cap for c in caps])
        r = {"model": a.model, "captioner": cap, "n": int(sel.sum()),
             "judge_score": spearman(js[sel], gt_rate[sel]),
             "memory_judge_scores": spearman(mem_cap[sel], gt_rate[sel]),
             "memory_gt_rates": spearman(mem_cap_gt[sel], gt_rate[sel]),
             "judge_plus_memory": spearman(js[sel] + mem_cap[sel], gt_rate[sel])}
        crow.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print(f"  {cap:10s} n {r['n']:4d} | own judge score {r['judge_score']:.3f} | k-NN memory of judge scores "
              f"{r['memory_judge_scores']:.3f} | k-NN memory of GT rates {r['memory_gt_rates']:.3f} | judge + memory {r['judge_plus_memory']:.3f}")
    with open(md / "caption_level.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crow[0])); w.writeheader(); w.writerows(crow)
    print(f"saved {md}/ablation_memory.csv, caption_level.csv")


if __name__ == "__main__":
    main()
