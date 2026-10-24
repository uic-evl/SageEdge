"""
Step 4g: can the Judge memory also estimate GROUNDING (precision and coverage) of a caption?

The plan's retrieval layer should "retrieve nearest neighbors from the vector database (with known judgments)
and aggregate their scores to estimate hallucination risk and grounding". eval_memory.py tested the
hallucination side; this script tests the grounding side at caption level, with the definitions of
grounding/grounding_metrics.py (paper RQ6):
  precision = supported COCO-class claims / COCO-class claims; coverage = COCO classes in the image that
  the caption names / COCO classes in the image. GT: COCO instance annotations. Judge (label-free): YOLOv8x,
  tau 0.25 for both (the detected classes stand in for the classes in the image).
For every caption, the k = 10 nearest past captions in the joint [image, caption] space of the memory
(runs/memory/<model>/, FAISS joint_caption vectors), taken only from the other CV folds (5 folds grouped by
image, so no caption of the same image), give
  memory (judge): similarity-weighted mean of the neighbours' judge estimates (label-free)
  memory (GT)   : the same with the neighbours' GT values (labeled memory, for comparison)
Reported per captioner: Spearman correlation with the GT value of the caption's own judge estimate, of the two
memory estimates and of judge + memory (sum), with 95% bootstrap CIs over images (B = 1000, seed 0), and
the mean of each estimate next to the mean GT value (does the memory reproduce the captioners' levels?).
Folds: the 1,000 images sorted, shuffled with random.Random(0), image i -> fold i mod 5.

Usage (from v3/, any venv with numpy, scipy; no GPU; after build_memory.py):
  python retrieval/eval_memory_grounding.py --coco_ann ../annotations/instances_val2017.json --model siglip
Output: runs/memory/<model>/grounding_memory.csv
"""
from __future__ import annotations

import argparse, csv, random, sqlite3, sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3)); sys.path.insert(0, str(V3 / "grounding"))
from grounding_metrics import caption_row, load_det, load_gt   # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--model", default="siglip", choices=["siglip", "clip"])
    ap.add_argument("--memdir", default=str(V3 / "runs/memory"))
    ap.add_argument("--detector", default="yolov8x")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import spearmanr

    md = Path(a.memdir) / a.model
    con = sqlite3.connect(md / "memory.sqlite"); con.row_factory = sqlite3.Row
    caps = [dict(r) for r in con.execute("SELECT row_id, image, captioner, caption FROM captions ORDER BY row_id")]
    con.close()
    assert len(caps) == 4000, f"expected the 4,000 evaluated captions, found {len(caps)} (use the original memory)"
    img_emb = np.load(md / "image_emb.npy"); cap_emb = np.load(md / "caption_emb.npy")
    JC = (np.concatenate([img_emb, cap_emb], 1) / np.sqrt(2)).astype("float32")

    gt, sal = load_gt(a.coco_ann, 0.01)
    det = load_det(V3 / f"results/detections/{a.detector}.jsonl")
    nan = float("nan")
    vals = {k: np.full(len(caps), nan) for k in ("prec_gt", "prec_judge", "cov_gt", "cov_judge")}
    for i, c in enumerate(caps):
        r = caption_row(c["image"], c["caption"] or "", det, a.tau, a.tau, gt, sal, {})
        if r["n"]:
            vals["prec_gt"][i] = r["sup_gt"] / r["n"]; vals["prec_judge"][i] = r["sup_judge"] / r["n"]
        if r["gt_n"]:
            vals["cov_gt"][i] = r["gt_hit"] / r["gt_n"]
        if r["det_n"]:
            vals["cov_judge"][i] = r["det_hit"] / r["det_n"]

    imgs = sorted({c["image"] for c in caps}); rng = random.Random(0); rng.shuffle(imgs)
    fold_of = {f: k % a.folds for k, f in enumerate(imgs)}
    fold = np.array([fold_of[c["image"]] for c in caps])

    def knn_mean(src):
        """similarity-weighted mean of src over the k nearest captions of other folds that have a value."""
        out = np.full(len(caps), nan)
        for k in range(a.folds):
            te = np.where(fold == k)[0]; tr = np.where((fold != k) & ~np.isnan(src))[0]
            S = JC[te] @ JC[tr].T
            top = np.argpartition(-S, a.k - 1, axis=1)[:, :a.k]
            w = np.clip(np.take_along_axis(S, top, 1), 1e-6, None)
            out[te] = (w * src[tr][top]).sum(1) / w.sum(1)
        return out

    rows = []
    for q in ("prec", "cov"):
        g, j = vals[f"{q}_gt"], vals[f"{q}_judge"]
        mj, mg = knn_mean(j), knn_mean(g)
        ests = {"own judge": j, "memory (judge)": mj, "memory (GT)": mg, "judge + memory (judge)": j + mj}
        print(f"\n== {a.model}: {'precision' if q == 'prec' else 'coverage'} per caption, Spearman with GT "
              f"(95% CI over images) | mean estimate vs mean GT")
        for cap in ["all"] + CAPTIONERS:
            sel = np.array([cap == "all" or c["captioner"] == cap for c in caps])
            ok = sel & ~np.isnan(g) & ~np.isnan(j) & ~np.isnan(mj)
            idx = np.where(ok)[0]
            by_img = {}
            for i in idx:
                by_img.setdefault(caps[i]["image"], []).append(i)
            keys = list(by_img); rb = random.Random(0)
            boots = [np.array([i for kk in rb.choices(keys, k=len(keys)) for i in by_img[kk]]) for _ in range(a.n_boot)]
            row = {"model": a.model, "quantity": "precision" if q == "prec" else "coverage", "captioner": cap,
                   "n": int(ok.sum()), "mean_gt": round(float(g[ok].mean()), 4)}
            line = []
            for name, e in ests.items():
                rho = float(spearmanr(e[ok], g[ok]).correlation)
                bs = sorted(float(spearmanr(e[b], g[b]).correlation) for b in boots)
                lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]
                key = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
                row[f"rho_{key}"] = round(rho, 4); row[f"rho_{key}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
                if name != "judge + memory (judge)":
                    row[f"mean_{key}"] = round(float(e[ok].mean()), 4)
                line.append(f"{name} {rho:.3f} [{lo:.3f}, {hi:.3f}]")
            rows.append(row)
            print(f"  {cap:10s} n {row['n']:4d} | " + " | ".join(line) +
                  f" | mean GT {row['mean_gt']:.3f}, judge {row['mean_own_judge']:.3f}, "
                  f"memory (judge) {row['mean_memory_judge']:.3f}, memory (GT) {row['mean_memory_GT']:.3f}")
    keys_ = list(dict.fromkeys(k for r in rows for k in r))
    with open(md / "grounding_memory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys_); w.writeheader(); w.writerows(rows)
    print(f"saved {md}/grounding_memory.csv")


if __name__ == "__main__":
    main()
