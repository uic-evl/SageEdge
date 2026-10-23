"""
Evaluate the feedback loop (runs/refine/<captioner>__<condition>.jsonl).

For every file and every round r = 0..max_rounds we score the caption the image
has AFTER r rounds: if the loop stopped earlier (nothing flagged) or a rewrite
failed, the last valid caption is carried forward. Round 0 is the frozen V3 caption.

Metrics (COCO-class claims only, unless noted):
  chair_i        share of claims absent in COCO GT (micro over claims)
  chair_s        share of verifiable captions with >= 1 GT-absent claim
  judge_flag     share of claims the YOLO judge flags (label-free)
  coverage_gt    share of GT classes of the image that the caption mentions (micro)
  coverage_det   same, but vs YOLO classes (conf >= --conf): label-free estimate
  claims         COCO claims per caption;  oov  out-of-vocabulary claims per caption
  words          mean caption length in words
  unverifiable   captions with no COCO claim
  human_hall     share of human-labeled claims (COCO + OOV) labeled absent;
                 human_cov = share of claims that have a human label
Loop statistics: triggered (flagged at round 0), stopped clean, mean rounds,
unchanged rewrites (model returned the same caption = kept the flagged object),
extra rewrite latency (from the run logs; NOT a clean benchmark).

Paired bootstrap (resampling images, B = 2000, seed 0):
  * every round vs round 0 of the same file
  * final round between conditions of the same captioner (self vs control, ...)

Outputs (in --outdir, default runs/refine/eval/):
  rounds.csv, deltas_vs_round0.csv, conditions.csv, new_pairs_to_label.csv

Usage (from v3/):
  python refine/eval_refine.py --coco_ann ../annotations/instances_val2017.json
  python refine/eval_refine.py --coco_ann ... --files runs/refine/llava__self.jsonl
  python refine/eval_refine.py --coco_ann ... --subset data/human_validation_100.txt
"""
from __future__ import annotations

import argparse, csv, glob, json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from judge_v3.parse_caption_v3 import COCO_SET, parse_caption      # noqa: E402

CONDITION_PAIRS = [("self", "control"), ("oracle", "self"), ("moondream", "self"), ("oracle", "control"),
                   ("selfB", "control"), ("selfB", "self")]
DELTA_KEYS = ["chair_i", "chair_s", "judge_flag", "coverage_gt", "coverage_det", "claims", "words",
              "human_hall"]


# ----------------------------------------------------------------- loading
def load_gt(path: str) -> dict[str, set[str]]:
    ann = json.load(open(path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    name = {im["id"]: im["file_name"] for im in ann["images"]}
    gt: dict[str, set[str]] = {n: set() for n in name.values()}
    for a in ann["annotations"]:                    # crowd annotations count as present
        gt[name[a["image_id"]]].add(cats[a["category_id"]])
    return gt


def load_det(path: str, conf: float) -> dict[str, set[str]]:
    out = {}
    for r in map(json.loads, open(path)):
        out[Path(r["image"]).name] = {d["class"] for d in r["detections"] if d["conf"] >= conf}
    return out


def load_human(path: str) -> dict[tuple[str, str], str]:
    return {(r["image"], r["object"]): r["label"] for r in csv.DictReader(open(path))}


def captions_by_round(rec: dict, max_rounds: int) -> list[str]:
    """Caption after r rounds, r = 0..max_rounds (carried forward)."""
    ok = {x["round"]: x["caption"] for x in rec["rounds"] if x.get("ok", True)}
    caps, cur = [], ok[0]
    for r in range(max_rounds + 1):
        cur = ok.get(r, cur)
        caps.append(cur)
    return caps


# ----------------------------------------------------------------- scoring
def score_caption(caption: str, name: str, gt, det, human) -> dict:
    p = parse_caption(caption)
    coco = [c for c in p["objects"] if c in COCO_SET]
    oov = [c for c in p["objects"] if c not in COCO_SET]
    g, d = gt[name], det.get(name, set())
    hl = [human[(name, c)] for c in p["objects"] if human.get((name, c)) in ("present", "absent")]
    unl = [c for c in p["objects"] if (name, c) not in human]
    return {"n": len(coco), "absent": sum(c not in g for c in coco), "flag": sum(c not in d for c in coco),
            "gt_n": len(g), "gt_hit": len(set(coco) & g), "det_n": len(d), "det_hit": len(set(coco) & d),
            "oov": len(oov), "words": len(caption.split()),
            "h_n": len(hl), "h_abs": sum(x == "absent" for x in hl), "all_claims": len(p["objects"]),
            "unlabeled": unl}


def aggregate(rows: list[dict]) -> dict:
    """rows: one score_caption() dict per image."""
    s = lambda k: sum(r[k] for r in rows)
    ver = [r for r in rows if r["n"]]
    nan = float("nan")
    return {"images": len(rows),
            "chair_i": s("absent") / s("n") if s("n") else nan,
            "chair_s": sum(r["absent"] > 0 for r in ver) / len(ver) if ver else nan,
            "judge_flag": s("flag") / s("n") if s("n") else nan,
            "coverage_gt": s("gt_hit") / s("gt_n") if s("gt_n") else nan,
            "coverage_det": s("det_hit") / s("det_n") if s("det_n") else nan,
            "claims": s("n") / len(rows), "oov": s("oov") / len(rows),
            "words": s("words") / len(rows), "unverifiable": len(rows) - len(ver),
            "human_hall": s("h_abs") / s("h_n") if s("h_n") else nan,
            "human_cov": s("h_n") / s("all_claims") if s("all_claims") else nan}


def paired_boot(a: dict[str, dict], b: dict[str, dict], keys, B=2000, seed=0):
    """a, b: image -> score row. Returns {key: (delta, lo, hi)} for agg(a) - agg(b)."""
    ids = sorted(set(a) & set(b))
    A, Bb = aggregate([a[i] for i in ids]), aggregate([b[i] for i in ids])
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(B):
        pick = rng.choices(ids, k=len(ids))
        x, y = aggregate([a[i] for i in pick]), aggregate([b[i] for i in pick])
        for k in keys:
            v = x[k] - y[k]
            if v == v:
                samples[k].append(v)
    out = {}
    for k in keys:
        v = sorted(samples[k])
        lo, hi = (v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]) if v else (float("nan"),) * 2
        out[k] = (A[k] - Bb[k], lo, hi)
    return out, len(ids)


def fmt(x, pct=False):
    if x != x:
        return "  -  "
    return f"{100 * x:5.1f}" if pct else f"{x:6.2f}"


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--files", nargs="*", help="default: runs/refine/*__*.jsonl (dry runs skipped)")
    ap.add_argument("--subset", help="only images in this list (e.g. data/human_validation_100.txt)")
    ap.add_argument("--detector", default=str(V3 / "results/detections/yolov8x.jsonl"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--human", default=str(V3 / "data/human_labels_100.csv"))
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--outdir", default=str(V3 / "runs/refine/eval"))
    a = ap.parse_args()

    files = a.files or sorted(f for f in glob.glob(str(V3 / "runs/refine/*__*.jsonl")) if "__dry" not in f)
    if not files:
        sys.exit("no refine files found")
    gt, det, human = load_gt(a.coco_ann), load_det(a.detector, a.conf), load_human(a.human)
    subset = {Path(l.strip()).name for l in open(a.subset) if l.strip()} if a.subset else None
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    round_rows, delta_rows, cond_rows = [], [], []
    final: dict[tuple[str, str], dict[str, dict]] = {}       # (captioner, condition) -> image -> score
    new_pairs: dict[tuple[str, str], set] = defaultdict(set)  # (image, object) -> {file}

    for f in files:
        cap_name, cond = Path(f).stem.split("__")[:2]
        recs = [json.loads(l) for l in open(f) if l.strip()]
        recs = [r for r in recs if subset is None or r["image"] in subset]
        if not recs:
            continue
        per_round = [dict() for _ in range(a.max_rounds + 1)]
        for rec in recs:
            for r, c in enumerate(captions_by_round(rec, a.max_rounds)):
                per_round[r][rec["image"]] = sc = score_caption(c, rec["image"], gt, det, human)
                if r == 0:
                    r0_objects = set(parse_caption(c)["objects"])
                for o in sc["unlabeled"]:      # objects that appear only in rewritten captions
                    if r > 0 and o not in r0_objects:
                        new_pairs[(rec["image"], o)].add(f"{cap_name}__{cond}")
        final[(cap_name, cond)] = per_round[-1]

        # loop statistics
        trig = [r for r in recs if r["rounds"][0]["feedback"]]
        lat = [sum(x.get("latency_ms", 0) for x in r["rounds"][1:]) for r in recs]
        stats = {"triggered": len(trig), "stopped_clean": sum(r["stopped_clean"] for r in recs),
                 "mean_rounds_triggered": statistics.mean(r["final_round"] for r in trig) if trig else 0,
                 "rewrite_errors": sum(not x.get("ok", True) for r in recs for x in r["rounds"]),
                 "rewrites": sum(1 for r in recs for x in r["rounds"][1:]),
                 "unchanged_rewrites": sum(1 for r in recs for prev, x in zip(r["rounds"], r["rounds"][1:])
                                           if x.get("ok", True) and x["caption"].strip() == prev.get("caption", "").strip()),
                 "extra_s_per_image": statistics.mean(lat) / 1000,
                 "extra_s_per_triggered": (sum(lat) / len(trig) / 1000) if trig else 0}
        trig_ids = {r["image"] for r in trig}

        for r in range(a.max_rounds + 1):
            for part, ids in (("all", None), ("triggered", trig_ids)):
                rows = [v for k, v in per_round[r].items() if ids is None or k in ids]
                if not rows:
                    continue
                row = {"captioner": cap_name, "condition": cond, "subset": part, "round": r,
                       **{k: round(v, 4) if isinstance(v, float) else v for k, v in aggregate(rows).items()}}
                if r == 0 and part == "all":
                    row.update({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
                round_rows.append(row)
            if r > 0:
                d, n = paired_boot(per_round[r], per_round[0], DELTA_KEYS, a.n_boot)
                delta_rows.append({"captioner": cap_name, "condition": cond, "round": r, "images": n,
                                   **{f"d_{k}": round(v[0], 4) for k, v in d.items()},
                                   **{f"d_{k}_ci95": f"[{v[1]:.4f}, {v[2]:.4f}]" for k, v in d.items()}})

        r0, rf = aggregate(list(per_round[0].values())), aggregate(list(per_round[-1].values()))
        print(f"\n== {cap_name} / {cond}: {len(recs)} images, triggered {stats['triggered']}, "
              f"stopped clean {stats['stopped_clean']}, mean rounds (triggered) {stats['mean_rounds_triggered']:.2f}, "
              f"errors {stats['rewrite_errors']}, unchanged {stats['unchanged_rewrites']}/{stats['rewrites']} rewrites, extra {stats['extra_s_per_triggered']:.1f} s per triggered image")
        print("   round | CHAIR-i CHAIR-s | flag | cov-GT cov-det | claims  oov  words | unver | human-hall (cov)")
        for r in range(a.max_rounds + 1):
            x = aggregate(list(per_round[r].values()))
            print(f"   {r:5d} | {fmt(x['chair_i'], 1)}  {fmt(x['chair_s'], 1)} | {fmt(x['judge_flag'], 1)} | "
                  f"{fmt(x['coverage_gt'], 1)}  {fmt(x['coverage_det'], 1)} | {fmt(x['claims'])} {fmt(x['oov'])} "
                  f"{x['words']:6.1f} | {x['unverifiable']:5d} | {fmt(x['human_hall'], 1)} ({fmt(x['human_cov'], 1)})")
        dd, _ = paired_boot(per_round[-1], per_round[0], ("chair_i", "chair_s", "coverage_gt"), a.n_boot)
        print("   final - round 0 (pp, paired 95% CI): " + "; ".join(
            f"{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" for k, v in dd.items()))

    # ---------------- conditions compared at the final round (same captioner, same images)
    print("\n== conditions, final round, paired by image")
    for cap_name in sorted({k[0] for k in final}):
        for c1, c2 in CONDITION_PAIRS:
            if (cap_name, c1) in final and (cap_name, c2) in final:
                d, n = paired_boot(final[(cap_name, c1)], final[(cap_name, c2)], DELTA_KEYS, a.n_boot)
                cond_rows.append({"captioner": cap_name, "comparison": f"{c1} - {c2}", "images": n,
                                  **{f"d_{k}": round(v[0], 4) for k, v in d.items()},
                                  **{f"d_{k}_ci95": f"[{v[1]:.4f}, {v[2]:.4f}]" for k, v in d.items()}})
                print(f"   {cap_name:9s} {c1:>9s} - {c2:<8s} (n={n}): CHAIR-i {100 * d['chair_i'][0]:+.1f} "
                      f"[{100 * d['chair_i'][1]:+.1f}, {100 * d['chair_i'][2]:+.1f}]  coverage "
                      f"{100 * d['coverage_gt'][0]:+.1f} [{100 * d['coverage_gt'][1]:+.1f}, {100 * d['coverage_gt'][2]:+.1f}]")

    # ---------------- write
    def write(name, rows, keys=None):
        keys = keys or list(dict.fromkeys(k for r in rows for k in r))
        if keys:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    write("rounds.csv", round_rows)
    write("deltas_vs_round0.csv", delta_rows)
    write("conditions.csv", cond_rows)
    labelled_imgs = {k[0] for k in human}
    np_rows = [{"pair_id": i, "image": im, "object": o, "coco_class": o in COCO_SET,
                "in_human_100": im in labelled_imgs, "files": ";".join(sorted(fs))}
               for i, ((im, o), fs) in enumerate(sorted(new_pairs.items()), 1)]
    write("new_pairs_to_label.csv", np_rows, ["pair_id", "image", "object", "coco_class", "in_human_100", "files"])
    n_in = sum(r["in_human_100"] for r in np_rows)
    print(f"\nnew (image, object) pairs without a human label in rewritten captions: {len(np_rows)} "
          f"({sum(r['coco_class'] for r in np_rows)} COCO), {n_in} of them on the 100 labeled images")
    print(f"saved {out}/rounds.csv, deltas_vs_round0.csv, conditions.csv, new_pairs_to_label.csv")


if __name__ == "__main__":
    main()
