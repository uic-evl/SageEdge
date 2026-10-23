"""
Grounding of a caption = precision + coverage (Step C of the plan).

The rule judge's `grounding_score` = supported / checked is exactly
1 - hallucination_score, so on its own it adds nothing new. We therefore report
grounding as two complementary quantities, each measured with labels and estimated
label-free by the judge:

  precision  share of the caption's COCO claims that are present
               judge: claim class detected (conf >= --tau)       (= grounding_score)
               GT   : claim class in COCO instance annotations   (= 1 - CHAIR-i)
               human: claim object labeled present (100 images; COCO-only and all claims)
  coverage   share of the objects present in the image that the caption mentions
               judge: share of detected classes (conf >= --tau_cov) the caption names
               GT   : share of COCO GT classes the caption names
               GT-salient: same, only GT classes whose largest instance covers >= 1%
                           of the image (small background objects are rarely captioned)
  grounded-F1  harmonic mean of corpus precision and coverage

Per captioner and detector: corpus (micro) values with 95% bootstrap CIs over images,
the judge-minus-GT estimation error, and the per-caption agreement between judge and
reference (Spearman / Pearson, bootstrap CIs). Human agreement is reported on the 100
human-labeled images (precision only: the human labels list claimed objects, not all
objects in the image, so human coverage cannot be computed).

Usage (from v3/, any venv; no GPU):
  python grounding/grounding_metrics.py --coco_ann ../annotations/instances_val2017.json
Outputs: runs/grounding/summary.csv, runs/grounding/per_caption.csv (+ printed tables)
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from judge_v3.parse_caption_v3 import COCO_SET, parse_caption      # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
DETECTORS = ["yolov8n", "yolov8s", "yolov8m", "yolov8x"]
NAN = float("nan")


# ----------------------------------------------------------------- loading
def load_gt(path: str, salient_frac: float):
    ann = json.load(open(path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    ims = {im["id"]: im for im in ann["images"]}
    gt = {im["file_name"]: set() for im in ims.values()}
    big: dict[str, dict[str, float]] = defaultdict(dict)
    for a in ann["annotations"]:
        im = ims[a["image_id"]]; n = im["file_name"]; c = cats[a["category_id"]]
        gt[n].add(c)
        frac = a.get("area", 0) / (im["width"] * im["height"])
        big[n][c] = max(big[n].get(c, 0.0), frac)
    sal = {n: {c for c, f in big[n].items() if f >= salient_frac} for n in gt}
    return gt, sal


def load_det(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    for r in map(json.loads, open(path)):
        m: dict[str, float] = {}
        for d in r["detections"]:
            m[d["class"]] = max(m.get(d["class"], 0.0), d["conf"])
        out[Path(r["image"]).name] = m
    return out


def load_human(path: Path) -> dict[tuple[str, str], str]:
    return {(r["image"], r["object"]): r["label"] for r in csv.DictReader(open(path))}


# ----------------------------------------------------------------- stats
def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return NAN
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else NAN


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def ci(vals):
    v = sorted(x for x in vals if x == x)
    return (v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]) if v else (NAN, NAN)


# ----------------------------------------------------------------- per caption
def caption_row(name, caption, det, tau, tau_cov, gt, sal, human):
    objs = parse_caption(caption)["objects"]
    coco = [c for c in objs if c in COCO_SET]
    dset = {c for c, s in det.get(name, {}).items() if s >= tau}
    dcov = {c for c, s in det.get(name, {}).items() if s >= tau_cov}
    g, s = gt[name], sal[name]
    hum_all = [human[(name, c)] for c in objs if human.get((name, c)) in ("present", "absent")]
    hum_coco = [human[(name, c)] for c in coco if human.get((name, c)) in ("present", "absent")]
    return {"n": len(coco), "sup_judge": sum(c in dset for c in coco), "sup_gt": sum(c in g for c in coco),
            "det_n": len(dcov), "det_hit": len(set(coco) & dcov),
            "gt_n": len(g), "gt_hit": len(set(coco) & g), "sal_n": len(s), "sal_hit": len(set(coco) & s),
            "h_n": len(hum_coco), "h_pres": sum(x == "present" for x in hum_coco),
            "ha_n": len(hum_all), "ha_pres": sum(x == "present" for x in hum_all)}


def corpus(rows):
    S = lambda k: sum(r[k] for r in rows)
    d = lambda a, b: S(a) / S(b) if S(b) else NAN
    out = {"prec_judge": d("sup_judge", "n"), "prec_gt": d("sup_gt", "n"),
           "cov_judge": d("det_hit", "det_n"), "cov_gt": d("gt_hit", "gt_n"), "cov_gt_salient": d("sal_hit", "sal_n"),
           "prec_human": d("h_pres", "h_n"), "prec_human_all": d("ha_pres", "ha_n")}
    f = lambda p, c: 2 * p * c / (p + c) if p == p and c == c and p + c else NAN
    out["gf1_judge"] = f(out["prec_judge"], out["cov_judge"])
    out["gf1_gt"] = f(out["prec_gt"], out["cov_gt"])
    return out


def per_caption_pairs(rows, a_num, a_den, b_num, b_den):
    xs, ys = [], []
    for r in rows:
        if r[a_den] and r[b_den]:
            xs.append(r[a_num] / r[a_den]); ys.append(r[b_num] / r[b_den])
    return xs, ys


AGREEMENTS = [  # (label, judge num/den, reference num/den)
    ("prec judge~GT", "sup_judge", "n", "sup_gt", "n"),
    ("cov judge~GT", "det_hit", "det_n", "gt_hit", "gt_n"),
    ("cov judge~GT-sal", "det_hit", "det_n", "sal_hit", "sal_n"),
    ("prec judge~human", "sup_judge", "n", "h_pres", "h_n"),
    ("prec GT~human", "sup_gt", "n", "h_pres", "h_n"),
]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--tau", type=float, default=0.25, help="judge threshold (paper: 0.25)")
    ap.add_argument("--tau_cov", type=float, default=0.25, help="detection threshold for label-free coverage")
    ap.add_argument("--salient_frac", type=float, default=0.01)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--outdir", default=str(V3 / "runs/grounding"))
    a = ap.parse_args()

    gt, sal = load_gt(a.coco_ann, a.salient_frac)
    human = load_human(V3 / "data/human_labels_100.csv")
    human_imgs = {Path(l.strip()).name for l in open(V3 / "data/human_validation_100.txt") if l.strip()}
    dets = {d: load_det(V3 / f"results/detections/{d}.jsonl") for d in DETECTORS}
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    summary, per_cap = [], []
    for cap in CAPTIONERS:
        caps = {Path(r["image"]).name: r.get("caption") or "" for r in map(json.loads, open(V3 / f"results/{cap}.jsonl"))}
        for dname in DETECTORS:
            rows = {n: caption_row(n, c, dets[dname], a.tau, a.tau_cov, gt, sal, human) for n, c in caps.items()}
            ids = sorted(rows)
            C = corpus([rows[i] for i in ids])
            boots = defaultdict(list)
            for _ in range(a.n_boot):
                pick = [rows[i] for i in rng.choices(ids, k=len(ids))]
                b = corpus(pick)
                for k, v in b.items():
                    boots[k].append(v)
                boots["err_prec"].append(b["prec_judge"] - b["prec_gt"])
                boots["err_cov"].append(b["cov_judge"] - b["cov_gt"])
            row = {"captioner": cap, "detector": dname, "tau": a.tau, "tau_cov": a.tau_cov, "images": len(ids),
                   "captions_verifiable": sum(rows[i]["n"] > 0 for i in ids),
                   **{k: round(v, 4) for k, v in C.items()},
                   "err_prec": round(C["prec_judge"] - C["prec_gt"], 4), "err_cov": round(C["cov_judge"] - C["cov_gt"], 4)}
            for k in list(C) + ["err_prec", "err_cov"]:
                lo, hi = ci(boots[k]); row[f"{k}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
            # per-caption agreement (all 1,000 images for GT; 100 images for human)
            for lab, an, ad, bn, bd in AGREEMENTS:
                sub = ids if "human" not in lab else [i for i in ids if i in human_imgs]
                xs, ys = per_caption_pairs([rows[i] for i in sub], an, ad, bn, bd)
                key = lab.replace(" ", "_").replace("~", "_vs_")
                row[f"{key}_n"] = len(xs)
                row[f"{key}_spearman"] = round(spearman(xs, ys), 4)
                row[f"{key}_pearson"] = round(pearson(xs, ys), 4)
                bs = []
                for _ in range(a.n_boot):
                    p = rng.choices(range(len(xs)), k=len(xs))
                    bs.append(spearman([xs[j] for j in p], [ys[j] for j in p]))
                lo, hi = ci(bs); row[f"{key}_spearman_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
            summary.append(row)
            if dname == "yolov8x":
                for i in ids:
                    per_cap.append({"captioner": cap, "image": i, **rows[i]})

    # ---------------- print
    print(f"Grounding = precision + coverage (judge tau {a.tau}, coverage tau {a.tau_cov}, "
          f"salient >= {a.salient_frac:.0%} of image). Values in %; [95% CI].")
    for dname in DETECTORS:
        print(f"\n== {dname}")
        print("captioner | prec judge  prec GT   err  | cov judge  cov GT  cov GT-sal  err | gF1 judge gF1 GT | "
              "Spearman prec J~GT  cov J~GT")
        for r in (x for x in summary if x["detector"] == dname):
            print(f"{r['captioner']:9s} |  {100*r['prec_judge']:5.1f}    {100*r['prec_gt']:5.1f}  {100*r['err_prec']:+5.1f} | "
                  f" {100*r['cov_judge']:5.1f}    {100*r['cov_gt']:5.1f}    {100*r['cov_gt_salient']:5.1f}   {100*r['err_cov']:+5.1f} | "
                  f"  {100*r['gf1_judge']:5.1f}   {100*r['gf1_gt']:5.1f}  |  "
                  f"{r['prec_judge_vs_GT_spearman']:.3f} "
                  f"{r['prec_judge_vs_GT_spearman_ci95']}  {r['cov_judge_vs_GT_spearman']:.3f} {r['cov_judge_vs_GT_spearman_ci95']}")
    print("\n== human labels (100 images, YOLOv8x judge): per-caption precision agreement")
    for r in (x for x in summary if x["detector"] == "yolov8x"):
        print(f"{r['captioner']:9s} | human prec COCO {100*r['prec_human']:5.1f} {r['prec_human_ci95']}  all {100*r['prec_human_all']:5.1f} | "
              f"Spearman judge~human {r['prec_judge_vs_human_spearman']:.3f} {r['prec_judge_vs_human_spearman_ci95']} "
              f"(n={r['prec_judge_vs_human_n']})  GT~human {r['prec_GT_vs_human_spearman']:.3f} {r['prec_GT_vs_human_spearman_ci95']}")

    def write(name, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    write("summary.csv", summary)
    write("per_caption_yolov8x.csv", per_cap)
    print(f"\nsaved {out}/summary.csv, per_caption_yolov8x.csv")


if __name__ == "__main__":
    main()
