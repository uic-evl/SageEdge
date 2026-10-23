"""
Edge-cloud cascade judge (Step 3): YOLO on the edge, Qwen3-VL-32B only for uncertain claims.

Policy (lo, hi) for a COCO-class claim with YOLO max confidence s for that class
(s = 0 if the class was not detected at conf >= 0.05):
    s >= hi       -> supported      (edge decides)
    s <  lo       -> hallucinated   (edge decides)
    lo <= s < hi  -> escalated: the Qwen3-VL-32B verdict is used
  YOLO-only (paper)  = (0.25, 0.25)   verify-flags = (0, 0.25)   Qwen-only = (0, inf)

Cost model (Jetson AGX Thor, 120 W, measured in V3, results/benchmark.json + energy.json):
  every caption pays the edge judge (YOLO + parse/match); a caption with >= 1 escalated
  claim also pays ONE Qwen call (all its escalated claims in one prompt, as in V3):
    time   = edge_ms + frac_captions_escalated * 72,869 ms (median Qwen call)
    energy = edge_J  + frac_captions_escalated * 4,143 J   (VIN rail, total)
  Assumption (state in paper): the V3 Qwen verdict of a claim is reused when only a
  subset of the caption's claims is escalated, and the Qwen call cost does not depend on
  the number of claims in the prompt.

Data: the 400 captions (100 images x 4 captioners) that Qwen judged in V3; truth = COCO GT
(889 COCO-class claims) and, secondarily, the blinded human labels.

Evaluation:
  1. full (lo, hi) grid -> F1 vs escalation, Pareto frontier (descriptive, in-sample)
  2. named policies with bootstrap CIs (resampling captions) and paired F1 differences
  3. honest operating points: 5-fold cross-validation grouped by image; for an escalation
     budget (share of captions sent to Qwen) the thresholds are chosen on 4 folds and
     applied to the 5th; pooled out-of-fold F1 with CI and paired difference vs YOLO-only / Qwen-only
  4. OOV claims (sky, wall ...): a COCO detector cannot check them; cost of also sending them to Qwen

Usage (from v3/, any venv, no GPU):
  python cascade/cascade_eval.py --coco_ann ../annotations/instances_val2017.json
Outputs: runs/cascade/grid.csv, policies.csv, cv_budgets.csv
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from evaluate import load_coco_gt, prf_kappa                       # noqa: E402
from judge_v3.parse_caption_v3 import COCO_SET                     # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
DETECTORS = ["yolov8n", "yolov8s", "yolov8m", "yolov8x"]
INF = float("inf")
LOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
HIS = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, INF]
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]


# ----------------------------------------------------------------- data
def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_rows(gt, human):
    det = {d: {} for d in DETECTORS}
    for d in DETECTORS:
        for r in jl(V3 / f"results/detections/{d}.jsonl"):
            m: dict[str, float] = {}
            for x in r["detections"]:
                m[x["class"]] = max(m.get(x["class"], 0.0), x["conf"])
            det[d][Path(r["image"]).name] = m
    rows = []
    for cap in CAPTIONERS:
        for q in jl(V3 / f"results/judge_qwen_{cap}.jsonl"):
            if not q.get("ok", True):
                continue
            b = Path(q["image"]).name
            for o, v in (q.get("verdicts") or {}).items():
                if v not in ("yes", "no"):
                    continue
                h = human.get((b, o))
                rows.append({"cap": cap, "b": b, "o": o, "coco": o in COCO_SET, "qwen": v == "no",
                             "gt": (o not in gt[int(Path(b).stem)]) if o in COCO_SET else None,
                             "human": {"absent": True, "present": False}.get(h),
                             **{d: det[d].get(b, {}).get(o, 0.0) for d in DETECTORS}})
    return rows


def costs():
    b = json.load(open(V3 / "results/benchmark.json"))
    e = json.load(open(V3 / "results/energy.json"))["phases"]
    parse_ms = b["rule_judge_parse_match"]["median_ms"]
    edge_ms = {d: b["yolo"][f"{d}@cuda:0"]["median_ms"] + parse_ms for d in DETECTORS}
    edge_j = {d: (e[d]["energy_J_per_caption_total"]["VIN"] if d in e else float("nan")) for d in DETECTORS}
    return edge_ms, edge_j, b["vlm_judge"]["median_ms"], e["qwen3-vl-32b"]["energy_J_per_caption_total"]["VIN"]


# ----------------------------------------------------------------- policy
def decide(r, d, lo, hi):
    """Return (flag, escalated) for one COCO-class claim."""
    s = r[d]
    if s >= hi:
        return False, False
    if s < lo:
        return True, False
    return r["qwen"], True


def evaluate(rows, d, lo, hi, truth="gt"):
    pairs, esc_caps, caps, esc_claims = [], set(), set(), 0
    for r in rows:
        f, e = decide(r, d, lo, hi)
        caps.add((r["cap"], r["b"]))
        if e:
            esc_caps.add((r["cap"], r["b"])); esc_claims += 1
        if r[truth] is not None:
            pairs.append((f, r[truth]))
    m = prf_kappa(pairs)
    m.update({"n": len(pairs), "flag_rate": sum(f for f, _ in pairs) / max(len(pairs), 1),
              "esc_claims": esc_claims / max(len(rows), 1), "esc_captions": len(esc_caps) / max(len(caps), 1)})
    return m


def by_caption(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["cap"], r["b"])].append(r)
    return g


def boot(rows, fn, B, seed=0):
    g = by_caption(rows); ks = list(g); rng = random.Random(seed); v = []
    for _ in range(B):
        x = fn([r for k in rng.choices(ks, k=len(ks)) for r in g[k]])
        if x == x:
            v.append(x)
    v.sort()
    return (v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]) if v else (float("nan"),) * 2


def f1_of(flags_truth):
    return prf_kappa(flags_truth)["f1"]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--outdir", default=str(V3 / "runs/cascade"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    gt = load_coco_gt(a.coco_ann)
    human = {(r["image"], r["object"]): r["label"] for r in csv.DictReader(open(V3 / "data/human_labels_100.csv"))}
    allrows = load_rows(gt, human)
    rows = [r for r in allrows if r["coco"]]
    oov = [r for r in allrows if not r["coco"]]
    edge_ms, edge_j, q_ms, q_j = costs()
    n_caps = len(by_caption(allrows))
    print(f"captions {n_caps}, COCO-class claims {len(rows)} (GT-hallucinated {sum(r['gt'] for r in rows)}), "
          f"OOV claims {len(oov)}")
    print(f"cost model: Qwen call {q_ms / 1000:.1f} s, {q_j:.0f} J; edge (ms / J): "
          + ", ".join(f"{d} {edge_ms[d]:.1f}/{edge_j[d]:.2f}" for d in DETECTORS))

    def cost(d, frac):
        return edge_ms[d] + frac * q_ms, edge_j[d] + frac * q_j

    # ---------------- 1. grid
    grid = []
    for d in DETECTORS:
        for lo in LOS:
            for hi in HIS:
                if lo > hi:
                    continue
                m = evaluate(rows, d, lo, hi)
                mh = evaluate(rows, d, lo, hi, "human")
                t, e = cost(d, m["esc_captions"])
                grid.append({"detector": d, "lo": lo, "hi": hi, "f1_gt": m["f1"], "precision": m["precision"],
                             "recall": m["recall"], "kappa": m["kappa"], "f1_human": mh["f1"],
                             "esc_claims": m["esc_claims"], "esc_captions": m["esc_captions"],
                             "ms_per_caption": t, "J_per_caption": e})
    for d in DETECTORS:   # Pareto frontier (in-sample): no other point has >= F1 at <= escalation
        pts = [g for g in grid if g["detector"] == d]
        for g in pts:
            g["pareto"] = not any(o["f1_gt"] > g["f1_gt"] and o["esc_captions"] <= g["esc_captions"] for o in pts)

    # ---------------- 2. named policies
    named = [("YOLO-only (paper)", 0.25, 0.25), ("verify flags", 0.0, 0.25), ("verify accepts", 0.25, INF),
             ("band 0.10-0.50", 0.10, 0.50), ("band 0.05-0.70", 0.05, 0.70), ("Qwen-only", 0.0, INF)]
    pol_rows = []
    print("\n== named policies vs COCO GT (889 claims) | F1 vs human | share of captions sent to Qwen | cost per caption")
    for d in ("yolov8n", "yolov8x"):
        for name, lo, hi in named:
            m, mh = evaluate(rows, d, lo, hi), evaluate(rows, d, lo, hi, "human")
            lo_ci, hi_ci = boot(rows, lambda s: evaluate(s, d, lo, hi)["f1"], a.n_boot)
            dd = boot(rows, lambda s: evaluate(s, d, lo, hi)["f1"] - evaluate(s, d, 0.25, 0.25)["f1"], a.n_boot)
            dq = boot(rows, lambda s: evaluate(s, d, lo, hi)["f1"] - evaluate(s, d, 0.0, INF)["f1"], a.n_boot)
            t, e = cost(d, m["esc_captions"])
            pol_rows.append({"detector": d, "policy": name, "lo": lo, "hi": hi, "f1_gt": round(m["f1"], 4),
                             "f1_gt_ci95": f"[{lo_ci:.3f}, {hi_ci:.3f}]", "precision": round(m["precision"], 4),
                             "recall": round(m["recall"], 4), "kappa": round(m["kappa"], 4),
                             "f1_human": round(mh["f1"], 4), "esc_claims": round(m["esc_claims"], 4),
                             "esc_captions": round(m["esc_captions"], 4),
                             "d_f1_vs_yolo_only": round(m["f1"] - evaluate(rows, d, .25, .25)["f1"], 4),
                             "d_f1_vs_yolo_only_ci95": f"[{dd[0]:.3f}, {dd[1]:.3f}]",
                             "d_f1_vs_qwen_only": round(m["f1"] - evaluate(rows, d, 0, INF)["f1"], 4),
                             "d_f1_vs_qwen_only_ci95": f"[{dq[0]:.3f}, {dq[1]:.3f}]",
                             "s_per_caption": round(t / 1000, 3), "J_per_caption": round(e, 1)})
            p = pol_rows[-1]
            print(f"{d:8s} {name:18s} F1 {p['f1_gt']:.3f} {p['f1_gt_ci95']}  human {p['f1_human']:.3f} | "
                  f"escalate {100 * p['esc_captions']:5.1f}% capt ({100 * p['esc_claims']:5.1f}% claims) | "
                  f"{p['s_per_caption']:7.2f} s {p['J_per_caption']:7.1f} J | vs YOLO-only {p['d_f1_vs_yolo_only']:+.3f} "
                  f"{p['d_f1_vs_yolo_only_ci95']} vs Qwen {p['d_f1_vs_qwen_only']:+.3f} {p['d_f1_vs_qwen_only_ci95']}")

    # ---------------- 3. cross-validated budgets (thresholds chosen out-of-sample)
    images = sorted({r["b"] for r in rows})
    rng = random.Random(0); rng.shuffle(images)
    fold = {b: i % a.folds for i, b in enumerate(images)}
    cv_rows = []
    print(f"\n== {a.folds}-fold CV (grouped by image): thresholds chosen on training folds under a budget "
          "(share of captions sent to Qwen)")
    for d in ("yolov8n", "yolov8x"):
        for budget in BUDGETS:
            oof, chosen = [], []
            for k in range(a.folds):
                tr = [r for r in rows if fold[r["b"]] != k]; te = [r for r in rows if fold[r["b"]] == k]
                best = max(((evaluate(tr, d, lo, hi)["f1"], -evaluate(tr, d, lo, hi)["esc_captions"], lo, hi)
                            for lo in LOS for hi in HIS
                            if lo <= hi and evaluate(tr, d, lo, hi)["esc_captions"] <= budget),
                           default=None)
                lo, hi = (best[2], best[3]) if best else (0.25, 0.25)
                chosen.append((lo, hi))
                for r in te:
                    f, e = decide(r, d, lo, hi)
                    oof.append({**r, "pol_flag": f, "pol_esc": e})
            fl = lambda s: f1_of([(r["pol_flag"], r["gt"]) for r in s])
            m = prf_kappa([(r["pol_flag"], r["gt"]) for r in oof])
            esc_c = len({(r["cap"], r["b"]) for r in oof if r["pol_esc"]}) / n_caps
            ci_ = boot(oof, fl, a.n_boot)
            dy = boot(oof, lambda s: fl(s) - f1_of([(decide(r, d, .25, .25)[0], r["gt"]) for r in s]), a.n_boot)
            dq = boot(oof, lambda s: fl(s) - f1_of([(r["qwen"], r["gt"]) for r in s]), a.n_boot)
            f_y = f1_of([(decide(r, d, .25, .25)[0], r["gt"]) for r in oof])
            f_q = f1_of([(r["qwen"], r["gt"]) for r in oof])
            t, e = cost(d, esc_c)
            cv_rows.append({"detector": d, "budget": budget, "f1_gt": round(m["f1"], 4), "f1_ci95": f"[{ci_[0]:.3f}, {ci_[1]:.3f}]",
                            "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
                            "esc_captions": round(esc_c, 4), "s_per_caption": round(t / 1000, 3), "J_per_caption": round(e, 1),
                            "d_vs_yolo_only": round(m["f1"] - f_y, 4), "d_vs_yolo_only_ci95": f"[{dy[0]:.3f}, {dy[1]:.3f}]",
                            "d_vs_qwen_only": round(m["f1"] - f_q, 4), "d_vs_qwen_only_ci95": f"[{dq[0]:.3f}, {dq[1]:.3f}]",
                            "thresholds_per_fold": " ".join(f"({x:g},{y:g})" for x, y in chosen)})
            c = cv_rows[-1]
            print(f"{d:8s} budget {100 * budget:4.0f}% -> F1 {c['f1_gt']:.3f} {c['f1_ci95']} | escalated {100 * esc_c:5.1f}% | "
                  f"{c['s_per_caption']:6.2f} s {c['J_per_caption']:7.1f} J | vs YOLO-only {c['d_vs_yolo_only']:+.3f} "
                  f"{c['d_vs_yolo_only_ci95']} | vs Qwen-only {c['d_vs_qwen_only']:+.3f} {c['d_vs_qwen_only_ci95']} | "
                  f"{c['thresholds_per_fold']}")

    # ---------------- 4. OOV claims
    oov_caps = {(r["cap"], r["b"]) for r in oov}
    mh = prf_kappa([(r["qwen"], r["human"]) for r in oov if r["human"] is not None])
    print(f"\n== OOV: {len(oov)} claims in {len(oov_caps)}/{n_caps} captions ({100 * len(oov_caps) / n_caps:.0f}%). "
          f"Sending them to Qwen too means escalating that share of captions. "
          f"Qwen on OOV vs human: P {mh['precision']:.3f} R {mh['recall']:.3f} F1 {mh['f1']:.3f} kappa {mh['kappa']:.3f}")

    def write(name, rs):
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rs)
    write("grid.csv", [{k: (round(v, 4) if isinstance(v, float) and v != INF else ("inf" if v == INF else v))
                        for k, v in g.items()} for g in grid])
    write("policies.csv", pol_rows)
    write("cv_budgets.csv", cv_rows)
    print(f"\nsaved {out}/grid.csv, policies.csv, cv_budgets.csv")


if __name__ == "__main__":
    main()
