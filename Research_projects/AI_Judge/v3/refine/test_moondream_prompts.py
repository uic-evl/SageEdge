"""
Find ONE feedback prompt that every rewriter (moondream, llava, gemma4) can follow.

With the main feedback prompt, 84% of moondream rewrites are empty or < 10 words ("urn"),
while the control prompt ("Look again carefully ...") works (median 62 words). This script
tries prompt variants on the same flagged images and reports, per variant:
  broken   empty or < 10 words
  removed  share of flagged objects no longer claimed (judge on the rewrite)
  kept     share of the other COCO claims still present (did it keep everything else?)
  words    median length
Cases (rewriter <- captions): moondream <- moondream, llava <- llava, gemma4 <- gemma4 (self-refine),
moondream <- llava (refiner condition); the first --n images flagged at round 0 for each. T = 0, seed 42, 160 tokens, as in the loop.

Usage (from v3/, venv with ollama):
  python refine/test_moondream_prompts.py --n 10      (~200 generations, ~25 min)
Output: runs/refine/moondream_prompt_test.jsonl (+ printed table)
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "refine"))
from run_refine_loop import feedback_text, find_image, judge, load_jsonl_by_name   # noqa: E402

VARIANTS = {
    "A_current": ('Here is a description of this image:\n"{caption}"\n\nThe image does not contain: {objects}. '
                  "Rewrite the description without these objects. Keep everything else. Reply with the new description only."),
    "B_control+": ('Here is a description of this image:\n"{caption}"\n\nLook again carefully and describe the image. '
                   "Do not mention {objects}. Reply with the new description only."),
    "C_control+no": ('Here is a description of this image:\n"{caption}"\n\nLook again carefully and describe the image. '
                     "There is no {objects} in the image."),
    "D_short": "Describe this image. Do not mention {objects}.",
    "E_rewrite": 'Rewrite this description so that it does not mention {objects}:\n"{caption}"',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--images_dir", action="append",
                    default=[str(V3.parent / "val2017"), str(V3 / "val2017"), "val2017"])
    a = ap.parse_args()
    import ollama

    dets = load_jsonl_by_name(V3 / "results/detections/yolov8x.jsonl")
    TAG = {"moondream": "moondream:latest", "llava": "llava:latest", "gemma4": "gemma4:e2b"}
    cases = []
    for rewriter, capsrc in (("moondream", "moondream"), ("llava", "llava"), ("gemma4", "gemma4"), ("moondream", "llava")):
        recs = [json.loads(l) for l in open(V3 / f"runs/refine/{capsrc}__control.jsonl")]
        cases += [(f"{rewriter}<-{capsrc}", TAG[rewriter], r["image"], r["rounds"][0]["caption"])
                  for r in recs if r["rounds"][0]["feedback"]][:a.n]

    out = V3 / "runs/refine/moondream_prompt_test.jsonl"
    rows = []
    with out.open("w") as f:
        for src, model, name, cap in cases:
            d = dets[name]["detections"]
            j0, fb = judge(cap, d, 0.25, None)
            others = [c for c in j0["checked"] if c not in fb]
            img = find_image(name, a.images_dir)
            for vname, tpl in VARIANTS.items():
                prompt = tpl.format(caption=cap, objects=feedback_text(cap, fb))
                t0 = time.perf_counter()
                kw = dict(model=model, prompt=prompt, images=[img], stream=False,
                          options={"temperature": 0, "seed": 42, "num_predict": 160})
                try:
                    r = ollama.generate(**kw, think=False)
                except Exception as e:                  # model without a thinking switch
                    if "think" not in str(e).lower():
                        raise
                    r = ollama.generate(**kw)
                new = (r.get("response") or "").strip().strip('"').strip()
                j1, _ = judge(new, d, 0.25, None)
                row = {"source": src, "image": name, "variant": vname, "feedback": fb, "words": len(new.split()),
                       "broken": len(new.split()) < 10,
                       "removed": sum(c not in j1["checked"] for c in fb) / len(fb),
                       "kept": (sum(c in j1["checked"] for c in others) / len(others)) if others else None,
                       "latency_ms": int((time.perf_counter() - t0) * 1000), "caption": new}
                rows.append(row); f.write(json.dumps(row) + "\n"); f.flush()
            print(f"done {src} {name}")

    srcs = list(dict.fromkeys(c[0] for c in cases))
    print(f"\n{len(cases)} flagged captions ({a.n} per case); judge = YOLOv8x @ 0.25")
    print("variant        case                broken  removed  kept   median words")
    for v in VARIANTS:
        for src in srcs:
            R = [r for r in rows if r["variant"] == v and r["source"] == src]
            ok = [r for r in R if not r["broken"]]
            kept = [r["kept"] for r in ok if r["kept"] is not None]
            print(f"{v:14s} {src:18s}  {sum(r['broken'] for r in R):2d}/{len(R):<2d}   "
                  f"{statistics.mean(r['removed'] for r in ok) if ok else float('nan'):.2f}    "
                  f"{statistics.mean(kept) if kept else float('nan'):.2f}   {statistics.median(r['words'] for r in R)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
