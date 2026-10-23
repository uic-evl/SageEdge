"""
Step 4c: retrieval closer to the project plan.

The plan asked for (1) a joint image + caption embedding space and (2) a vector DB that stores
prior JUDGMENTS. eval_retrieval.py used image embeddings and a bank labeled with COCO ground truth.
This script adds:
  text   zero-shot image-claim agreement in the same CLIP/SigLIP space: the claimed class is
         embedded as text ("a photo of a <class>.") and compared with the image embedding
         (cosine, and the softmax over the 80 COCO classes)
  jbank  a memory bank of prior judge verdicts instead of labels: YOLOv8x is run once on the
         10,000 bank images, and the prior counts how many of the k nearest bank images the
         judge saw the class in (conf >= 0.25). No human labels are needed, so this memory
         could be built on a Sage node from its own past images.
Judges (logistic regression, 5-fold CV grouped by image, threshold chosen on training folds):
  yolo-lr | +text | +retr(GT bank) | +retr(judge bank) | +text+judge bank (label-free) | +text+GT bank
compared with yolo-lr (paired bootstrap over images, B = 2000, seed 0), for YOLOv8n and YOLOv8x.

Usage (from v3/, venv_gpu active; the first run detects objects in the bank, ~5-10 min on GPU):
  python retrieval/eval_retrieval_v2.py --coco_ann ../annotations/instances_val2017.json --model siglip
  python retrieval/eval_retrieval_v2.py --coco_ann ../annotations/instances_val2017.json --model clip
Outputs: runs/retrieval/bank_detections_yolov8x.jsonl (resumable), runs/retrieval/<model>/text_cos.npy,
         runs/retrieval/<model>/ablation_v2.csv
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from evaluate import load_coco_gt, prf_kappa                        # noqa: E402
from judge_v3.parse_caption_v3 import COCO_CLASSES, COCO_SET, parse_caption   # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
MODELS = {"siglip": ("ViT-B-16-SigLIP", "webli"), "clip": ("ViT-B-32-quickgelu", "openai")}
K = 10


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def detect_bank(files, img_dir: Path, out: Path, weights: Path, batch=32):
    """YOLOv8x on the bank images (resumable JSONL, boxes with conf >= 0.05)."""
    done = {json.loads(l)["image"] for l in open(out)} if out.exists() else set()
    todo = [f for f in files if f not in done]
    print(f"bank detections: {len(done)} done, {len(todo)} to run")
    if not todo:
        return
    from ultralytics import YOLO
    model = YOLO(str(weights))
    with open(out, "a") as fh:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            res = model([str(img_dir / f) for f in chunk], conf=0.05, device=0, verbose=False)
            for f, r in zip(chunk, res):
                dets = [{"class": r.names[int(b.cls)], "conf": round(float(b.conf), 4)} for b in r.boxes]
                fh.write(json.dumps({"image": f, "detections": dets}) + "\n")
            fh.flush()
            if (i // batch) % 50 == 0:
                print(f"  {min(i + batch, len(todo))}/{len(todo)}")


def text_cosines(model_key: str, eval_emb, out: Path):
    """Cosine between every evaluation image and the 80 class prompts in the model's joint space."""
    import numpy as np
    if out.exists():
        return np.load(out)
    import torch, open_clip
    name, pre = MODELS[model_key]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev)
    tok = open_clip.get_tokenizer(name)
    with torch.inference_mode():
        t = model.encode_text(tok([f"a photo of a {c}." for c in COCO_CLASSES]).to(dev))
        t = torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()
    cos = (eval_emb @ t.T).astype("float32")
    np.save(out, cos)
    return cos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--model", default="siglip", choices=list(MODELS))
    ap.add_argument("--detectors", nargs="+", default=["yolov8n", "yolov8x"])
    ap.add_argument("--train_dir", default=str(V3.parent / "train2017_bank"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--jtau", type=float, default=0.25, help="judge threshold for the judge-verdict bank")
    a = ap.parse_args()

    import numpy as np, faiss
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    d = V3 / "runs/retrieval" / a.model
    bank = np.load(d / "bank_emb.npy"); ev = np.load(d / "eval_emb.npy")
    bank_files = (d / "bank_files.txt").read_text().split()
    eval_files = (d / "eval_files.txt").read_text().split()
    assert not set(bank_files) & set(eval_files), "bank and evaluation images overlap"
    meta = {m["file"]: set(m["classes"]) for m in jl(V3 / "runs/retrieval/bank_meta.jsonl")}
    bank_gt = [meta[f] for f in bank_files]

    bdet_path = V3 / "runs/retrieval/bank_detections_yolov8x.jsonl"
    detect_bank(bank_files, Path(a.train_dir), bdet_path, V3 / "yolov8x.pt")
    bdet = {r["image"]: r["detections"] for r in jl(bdet_path)}
    bank_judge = [{x["class"] for x in bdet[f] if x["conf"] >= a.jtau} for f in bank_files]
    print(f"bank: {len(bank_files)} images; judge-verdict bank agrees with GT on "
          f"{np.mean([len(g & j) / max(1, len(g | j)) for g, j in zip(bank_gt, bank_judge)]):.3f} (mean Jaccard)")

    cos = text_cosines(a.model, ev, d / "text_cos.npy")
    zs = np.exp(100 * (cos - cos.max(1, keepdims=True))); zs /= zs.sum(1, keepdims=True)
    cidx = {c: i for i, c in enumerate(COCO_CLASSES)}

    index = faiss.IndexFlatIP(bank.shape[1]); index.add(bank)
    sims, nn = index.search(ev, K)
    row_of = {f: i for i, f in enumerate(eval_files)}

    def prior(i, c, labels):
        w = np.clip(sims[i], 1e-6, None)
        hit = np.array([c in labels[j] for j in nn[i]], dtype=float)
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
            if f not in row_of:
                continue
            i = row_of[f]
            for c in parse_caption(r.get("caption") or "")["objects"]:
                if c not in COCO_SET:
                    continue
                claims.append({"img": f, "y": int(c not in gt[int(Path(f).stem)]),
                               "human": {"absent": 1, "present": 0}.get(human.get((f, c))),
                               **{f"conf_{dn}": det[dn].get(f, {}).get(c, 0.0) for dn in a.detectors},
                               "cos": float(cos[i, cidx[c]]), "zs": float(zs[i, cidx[c]]),
                               "p_gt": prior(i, c, bank_gt), "p_judge": prior(i, c, bank_judge)})
    y = np.array([c["y"] for c in claims])
    imgs = sorted({c["img"] for c in claims}); rng = random.Random(0); rng.shuffle(imgs)
    fold_of = {f: k % a.folds for k, f in enumerate(imgs)}
    fold = np.array([fold_of[c["img"]] for c in claims])
    print(f"model {a.model}: {len(claims)} claims ({y.mean():.1%} GT-hallucinated), k = {K}")

    col = lambda k: np.array([c[k] for c in claims])

    def feats(dn, parts):
        conf = col(f"conf_{dn}"); X = [conf, (conf > 0).astype(float)]
        if "text" in parts:
            X += [col("cos"), col("zs"), col("zs") * conf]
        for p in ("p_gt", "p_judge"):
            if p in parts:
                X += [col(p), col(p) * conf]
        return np.stack(X, 1)

    def best_thr(score, truth):
        best, thr = -1.0, 0.5
        for t in np.unique(np.round(score, 4)):
            pr = score >= t; tp = (pr & (truth == 1)).sum(); fp = (pr & (truth == 0)).sum(); fn = (~pr & (truth == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best:
                best, thr = f1, t
        return thr

    def cv(X):
        prob = np.zeros(len(y)); flag = np.zeros(len(y), dtype=bool)
        for k in range(a.folds):
            tr, te = fold != k, fold == k
            lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            thr = best_thr(lr.predict_proba(X[tr])[:, 1], y[tr])
            prob[te] = lr.predict_proba(X[te])[:, 1]; flag[te] = prob[te] >= thr
        return prob, flag

    by_img = defaultdict(list)
    for i, c in enumerate(claims):
        by_img[c["img"]].append(i)
    keys = list(by_img)

    def paired(fa, pa, fb, pb):
        r = random.Random(0); df, da = [], []
        for _ in range(a.n_boot):
            idx = np.array([i for k in r.choices(keys, k=len(keys)) for i in by_img[k]]); yy = y[idx]
            f1 = lambda fl: prf_kappa(list(zip(fl[idx].tolist(), (yy == 1).tolist())))["f1"]
            df.append(f1(fa) - f1(fb))
            if 0 < yy.sum() < len(yy):
                da.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
        q = lambda v: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1])
        return q(df), q(da)

    def scores(flag, prob):
        m = prf_kappa(list(zip(flag.tolist(), (y == 1).tolist())))
        hm = [(bool(f), c["human"] == 1) for f, c in zip(flag, claims) if c["human"] is not None]
        return {**m, "auc": roc_auc_score(y, prob), "ap": average_precision_score(y, prob), "f1_human": prf_kappa(hm)["f1"]}

    variants = [("yolo-lr", ()), ("+text", ("text",)), ("+retr GT bank", ("p_gt",)),
                ("+retr judge bank", ("p_judge",)), ("+text +judge bank (label-free)", ("text", "p_judge")),
                ("+text +GT bank", ("text", "p_gt"))]
    rows = []
    for dn in a.detectors:
        base_p, base_f = cv(feats(dn, ()))
        print(f"\n== {dn} ({a.model}); deltas vs yolo-lr, paired 95% CI over images")
        for name, parts in variants:
            p, fl = (base_p, base_f) if not parts else cv(feats(dn, parts))
            m = scores(fl, p)
            (df, da) = paired(fl, p, base_f, base_p) if parts else ((0.0, 0.0), (0.0, 0.0))
            b = scores(base_f, base_p)
            rows.append({"detector": dn, "model": a.model, "judge": name, **{k: round(v, 4) for k, v in m.items()},
                         "dF1": round(m["f1"] - b["f1"], 4), "dF1_ci95": f"[{df[0]:.3f}, {df[1]:.3f}]",
                         "dAUC": round(m["auc"] - b["auc"], 4), "dAUC_ci95": f"[{da[0]:.3f}, {da[1]:.3f}]"})
            x = rows[-1]
            print(f"  {name:32s} F1 {x['f1']:.3f} AUC {x['auc']:.3f} AP {x['ap']:.3f} human F1 {x['f1_human']:.3f} | "
                  f"dF1 {x['dF1']:+.3f} {x['dF1_ci95']}  dAUC {x['dAUC']:+.3f} {x['dAUC_ci95']}")
    zs_auc = roc_auc_score(y, -col("zs")); jb_auc = roc_auc_score(y, -col("p_judge")); gb_auc = roc_auc_score(y, -col("p_gt"))
    print(f"\nsingle-feature AUC: text zero-shot {zs_auc:.3f} | judge-bank prior {jb_auc:.3f} | GT-bank prior {gb_auc:.3f}")
    rows.append({"detector": "-", "model": a.model, "judge": "single-feature AUC",
                 "auc_text_zeroshot": round(zs_auc, 4), "auc_prior_judge_bank": round(jb_auc, 4), "auc_prior_gt_bank": round(gb_auc, 4)})
    keys_ = list(dict.fromkeys(k for r in rows for k in r))
    with open(d / "ablation_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys_); w.writeheader(); w.writerows(rows)
    print(f"saved {d}/ablation_v2.csv")


if __name__ == "__main__":
    main()
