"""
Synthetic hallucination stress test (proposal: "augment with synthetic hallucination examples").

For every caption of the four frozen V3 captioners we insert ONE object that is absent from the
image (not in the COCO instance annotations) and not already claimed by the caption, as a new
sentence at the end:  "The image also shows a <word>."  (an unfinished caption is closed with a period first)
The object is chosen in three ways (POPE-style):
  random       uniform over the absent classes
  popular      the most frequent COCO class (train2017 instances) that is absent
  adversarial  the absent class that co-occurs most often with the classes present in the image
               (co-occurrence counted on train2017, so no information from the evaluation images)
The word is a random surface form of the class from the parser's synonym list (e.g. "man" for
person, "sofa" for couch), so the test also checks that the parser maps it to the class.

Negation check: we also insert "The image does not show any <word>." for the random object; the parser
should NOT turn it into a claim (a claim here would be a parser error).

Reported per strategy and detector (YOLOv8 n/s/m/x, tau = 0.25), 95% bootstrap CI over captions:
  parsed     share of inserted objects that the parser extracts as a claim of the right class
  caught     share of parsed inserted claims the judge flags (= judge recall on synthetic hallucinations)
  missed_by_false_detection  share of misses where the detector reported the absent class (>= tau)
and the negation leak rate (compare "caught" with the V3 judge recall on real hallucinations, 0.79-0.85 for YOLOv8x).

Usage (from v3/, any venv, no GPU, ~1-2 min):
  python synthetic/synthetic_test.py --coco_ann ../annotations/instances_val2017.json \
      --coco_train ../annotations/instances_train2017.json
Outputs: runs/synthetic/summary.csv, runs/synthetic/inserted.jsonl
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from judge_v3.parse_caption_v3 import COCO_SET, SYNONYMS, BLOCKED_COMPOUNDS, parse_caption   # noqa: E402
from judge_v3.rule_judge_v3 import rule_based_judge_v3                                       # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
DETECTORS = ["yolov8n", "yolov8s", "yolov8m", "yolov8x"]
STRATEGIES = ["random", "popular", "adversarial"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_gt(path):
    ann = json.load(open(path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    name = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = {n: set() for n in name.values()}
    for a in ann["annotations"]:
        gt[name[a["image_id"]]].add(cats[a["category_id"]])
    return gt


def surface_forms() -> dict[str, list[str]]:
    """COCO class -> surface words that the parser maps to it (single words and phrases)."""
    forms: dict[str, list[str]] = defaultdict(list)
    for w, c in SYNONYMS.items():
        if c in COCO_SET and w not in BLOCKED_COMPOUNDS and w.replace(" ", "").isalpha():
            forms[c].append(w)
    for c in COCO_SET:
        if c not in forms[c]:
            forms[c].append(c)
    return {c: sorted(set(v)) for c, v in forms.items()}


def article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def ci(vals, B, rng):
    """Bootstrap CI of the mean of a 0/1 list."""
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals); m = sum(vals) / n; bs = []
    for _ in range(B):
        bs.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    bs.sort()
    return m, bs[int(.025 * B)], bs[int(.975 * B) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_ann", required=True)
    ap.add_argument("--coco_train", required=True, help="instances_train2017.json (frequencies, co-occurrence)")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--outdir", default=str(V3 / "runs/synthetic"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    gt = load_gt(a.coco_ann)
    train = load_gt(a.coco_train)
    freq = Counter(c for s in train.values() for c in s)
    co = defaultdict(Counter)
    for s in train.values():
        for p in s:
            for c in s:
                if c != p:
                    co[p][c] += 1
    forms = surface_forms()
    det = {d: {} for d in DETECTORS}
    for d in DETECTORS:
        for r in jl(V3 / f"results/detections/{d}.jsonl"):
            det[d][Path(r["image"]).name] = r["detections"]

    recs = []
    for cap in CAPTIONERS:
        for r in jl(V3 / f"results/{cap}.jsonl"):
            name = Path(r["image"]).name
            caption = (r.get("caption") or "").strip()
            if caption and caption[-1] not in ".!?":
                caption += "."        # 160-token captions can stop mid-sentence; close it so a
                                      # negation scope cannot spill into the inserted sentence
            claimed = set(parse_caption(caption)["objects"])
            absent = sorted(c for c in COCO_SET if c not in gt[name] and c not in claimed and c in forms)
            if not absent:
                continue
            choice = {"random": rng.choice(absent),
                      "popular": max(absent, key=lambda c: (freq[c], c))}
            if gt[name]:
                choice["adversarial"] = max(absent, key=lambda c: (sum(co[p][c] / max(freq[p], 1) for p in gt[name]), c))
            for strat, cls in choice.items():
                word = rng.choice(forms[cls])
                new = f"{caption} The image also shows {article(word)} {word}."
                parsed = cls in parse_caption(new)["objects"]
                row = {"captioner": cap, "image": name, "strategy": strat, "class": cls, "word": word, "parsed": parsed}
                for d in DETECTORS:
                    j = rule_based_judge_v3(new, det[d][name], min_conf=a.tau)
                    flag = any(c["claim_normalized"] == cls and c["status"] == "unsupported_by_verifier"
                               for c in j["claim_results"])
                    falsedet = any(x["class"] == cls and x["conf"] >= a.tau for x in det[d][name])
                    row[f"caught_{d}"] = flag
                    row[f"falsedet_{d}"] = falsedet
                if strat == "random":
                    neg = f"{caption} The image does not show any {word}."
                    row["negation_leak"] = cls in parse_caption(neg)["objects"]
                recs.append(row)

    with open(out / "inserted.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    summary = []
    print(f"inserted objects: {len(recs)} ({', '.join(f'{s} {sum(r['strategy'] == s for r in recs)}' for s in STRATEGIES)}); "
          f"tau = {a.tau}; values in %, [95% CI]")
    for s in STRATEGIES:
        R = [r for r in recs if r["strategy"] == s]
        p = ci([int(r["parsed"]) for r in R], a.n_boot, rng)
        P = [r for r in R if r["parsed"]]
        line = f"\n== {s:11s} n={len(R)}  parsed {100 * p[0]:5.1f} [{100 * p[1]:.1f}, {100 * p[2]:.1f}]"
        print(line)
        for d in DETECTORS:
            c = ci([int(r[f"caught_{d}"]) for r in P], a.n_boot, rng)
            miss = [r for r in P if not r[f"caught_{d}"]]
            fd = sum(r[f"falsedet_{d}"] for r in miss) / len(miss) if miss else float("nan")
            per = {cap: sum(r[f"caught_{d}"] for r in P if r["captioner"] == cap) / max(1, sum(r["captioner"] == cap for r in P))
                   for cap in CAPTIONERS}
            summary.append({"strategy": s, "detector": d, "n_inserted": len(R), "n_parsed": len(P),
                            "parsed": round(p[0], 4), "caught": round(c[0], 4), "caught_ci95": f"[{c[1]:.4f}, {c[2]:.4f}]",
                            "misses": len(miss), "misses_due_to_false_detection": round(fd, 4),
                            **{f"caught_{k}": round(v, 4) for k, v in per.items()}})
            print(f"   {d}: caught {100 * c[0]:5.1f} [{100 * c[1]:.1f}, {100 * c[2]:.1f}]  misses {len(miss):4d} "
                  f"({100 * fd:4.1f}% because the detector reported the absent class)  | "
                  + "  ".join(f"{k} {100 * v:.1f}" for k, v in per.items()))
    neg = [int(r["negation_leak"]) for r in recs if r["strategy"] == "random"]
    nl = ci(neg, a.n_boot, rng)
    print(f"\nnegation leak (\"does not show any X\" parsed as a claim): {100 * nl[0]:.1f}% [{100 * nl[1]:.1f}, {100 * nl[2]:.1f}] of {len(neg)}")
    bad = Counter((r["word"], r["class"]) for r in recs if not r["parsed"])
    if bad:
        print("surface words the parser did not map (word -> class, count):", bad.most_common(10))
    summary.append({"strategy": "negation_leak", "detector": "-", "n_inserted": len(neg), "parsed": round(nl[0], 4)})
    keys = list(dict.fromkeys(k for r in summary for k in r))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    print(f"saved {out}/summary.csv, inserted.jsonl")


if __name__ == "__main__":
    main()
