"""
Attribute hallucination, step 2: a color judge for color + object claims.

For every unique (image, color phrase, object) in data/color_claims_all.csv:
  1. take the stored YOLOv8x boxes of that class with conf >= --tau (top --max_boxes by confidence)
  2. crop each box (10% padding) and score it zero-shot with an open_clip model against
     "a photo of a {color} {object}." for the 11 colors (grey merged into gray), softmax over colors
  3. a box supports the claim when the claimed colors are its top-n colors (n = number of claimed colors,
     so "red car" needs red as top-1, "black and white dog" needs black and white as the top-2);
     the claim is supported when at least one box supports it.
     score = max over boxes of the summed probability of the claimed colors (0 when there is no box)
Also stored, as ablations: the same zero-shot test on the whole image (no detector).

Usage (from v3/, venv_gpu):  python attributes/color_judge.py --model siglip     (then --model clip)
Output: runs/attributes/color_scores_<model>.jsonl (resumable), runs/attributes/color_bench_<model>.json
"""
from __future__ import annotations

import argparse, csv, json, statistics, time
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
MODELS = {"siglip": ("ViT-B-16-SigLIP", "webli"), "clip": ("ViT-B-32-quickgelu", "openai")}
COLORS = ["red", "blue", "green", "yellow", "black", "white", "gray", "brown", "orange", "pink", "purple"]


def boxes_by_image(path: Path, tau: float):
    out = {}
    for r in map(json.loads, open(path)):
        out[Path(r["image"]).name] = [d for d in r["detections"] if d["conf"] >= tau]
    return out


def crop(img, xyxy, pad=0.10, min_side=8):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(img.width, x2 + pad * w), min(img.height, y2 + pad * h)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return img.crop((int(x1), int(y1), int(round(x2)), int(round(y2))))


def judge(probs, claimed):
    """probs: list of 11 color probabilities; claimed: list of colors. Returns (support, score, top colors)."""
    order = sorted(range(len(COLORS)), key=lambda i: -probs[i])
    top = [COLORS[i] for i in order[:len(claimed)]]
    return set(top) == set(claimed), float(sum(probs[COLORS.index(c)] for c in claimed)), [COLORS[i] for i in order[:3]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="siglip")
    ap.add_argument("--images", default=str(V3.parent / "val2017"))
    ap.add_argument("--detector", default="yolov8x")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--max_boxes", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="test: only the first N claims")
    a = ap.parse_args()

    import torch, open_clip
    from PIL import Image

    claims = list(dict.fromkeys((r["image"], r["color"], r["object"])
                                for r in csv.DictReader(open(V3 / "data/color_claims_all.csv"))))
    if a.limit:
        claims = claims[:a.limit]
    outdir = V3 / "runs/attributes"; outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"color_scores_{a.model}{'_limit' + str(a.limit) if a.limit else ''}.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["image"], r["color"], r["object"]) for r in map(json.loads, open(out_path))}
    todo = [c for c in claims if c not in done]
    print(f"{len(claims)} unique color claims: {len(done)} done, {len(todo)} to run")

    dets = boxes_by_image(V3 / f"results/detections/{a.detector}.jsonl", a.tau)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name, pre = MODELS[a.model]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(name)
    scale = float(model.logit_scale.exp())
    text_cache = {}

    def text_emb(obj):
        if obj not in text_cache:
            with torch.inference_mode():
                t = model.encode_text(tok([f"a photo of a {c} {obj}." for c in COLORS]).to(dev))
            text_cache[obj] = torch.nn.functional.normalize(t.float(), dim=-1)
        return text_cache[obj]

    def color_probs(ims, obj):
        x = torch.stack([preprocess(im) for im in ims]).to(dev)
        with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
            e = model.encode_image(x)
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        return (scale * e @ text_emb(obj).T).softmax(-1).cpu().tolist()

    lat = []
    with open(out_path, "a") as fh:
        for i, (im_name, color, obj) in enumerate(todo, 1):
            claimed = color.split(" and ")
            img = Image.open(Path(a.images) / im_name).convert("RGB")
            boxes = sorted((d for d in dets.get(im_name, []) if d["class"] == obj), key=lambda d: -d["conf"])
            crops = [(d, c) for d in boxes[:a.max_boxes] if (c := crop(img, d["xyxy"])) is not None]
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            per_box = []
            if crops:
                for (d, _), p in zip(crops, color_probs([c for _, c in crops], obj)):
                    sup, sc, top = judge(p, claimed)
                    per_box.append({"conf": d["conf"], "support": sup, "score": round(sc, 4), "top3": top})
            if dev == "cuda":
                torch.cuda.synchronize()
            if crops:
                lat.append((time.perf_counter() - t) * 1000)
            fsup, fsc, ftop = judge(color_probs([img], obj)[0], claimed)
            rec = {"image": im_name, "color": color, "object": obj, "n_boxes": len(boxes), "n_crops": len(crops),
                   "crop_support": any(b["support"] for b in per_box) if per_box else None,
                   "crop_score": max((b["score"] for b in per_box), default=0.0), "boxes": per_box,
                   "full_support": fsup, "full_score": round(fsc, 4), "full_top3": ftop}
            fh.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                fh.flush(); print(f"  {i}/{len(todo)}")

    rows = [json.loads(l) for l in open(out_path)]
    nb = sum(r["n_crops"] > 0 for r in rows)
    print(f"{len(rows)} claims scored: {nb} with a {a.detector} box of the class; "
          f"color supported on {sum(bool(r['crop_support']) for r in rows)} of them")
    if lat and len(lat) > 20:
        lat = lat[10:]
        bench = {"model": name, "pretrained": pre, "device": dev, "claims_timed": len(lat),
                 "ms_per_claim_median": round(statistics.median(lat), 2),
                 "ms_per_claim_p90": round(sorted(lat)[int(.9 * len(lat)) - 1], 2)}
        (outdir / f"color_bench_{a.model}.json").write_text(json.dumps(bench, indent=2))
        print("benchmark:", bench)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
