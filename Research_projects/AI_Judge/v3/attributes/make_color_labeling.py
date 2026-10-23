"""
Attribute hallucination, step 1: extract color + object claims and build a blinded labeling page.

A color claim is a COCO-class object in a caption with color words right before it
("a red double-decker bus", "a black and white dog"). Colors come from the parser's list
(red, blue, green, yellow, black, white, gray, brown, orange, pink, purple). Negated objects
are skipped. Up to three tokens may sit between the color and the object ("red double-decker bus").

The labeling set has one item per unique (image, color phrase, object) over all captioners: every item on the
100 images of the object study (data/human_validation_100.txt), then whole extra images from the 1,000
(seeded) until --n_items (default 200). The page shows only the image
and the question "Is there a <color phrase> <object> in the image?" (present / absent / unsure),
no caption, model or judge verdict, in random order.

Usage (from v3/, any venv with Pillow):
  python attributes/make_color_labeling.py --images ../val2017
Outputs: data/color_pairs.csv (items to label), data/color_claims_all.csv (all color claims, 1,000 images),
         color_label_tool.html (offline labeling page; open it in a browser)
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from collections import OrderedDict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from judge_v3.parse_caption_v3 import (COCO_SET, COLORS, _match_phrases, _negated_positions,   # noqa: E402
                                       _tokenize)
import make_labeling_page as mlp                                                              # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
PUNCT = set(".,;:!?")
HEADS = {"boots", "boot", "mask", "jacket", "pants", "suit", "helmet", "goggles", "poles", "pole", "lift", "shirt",
         "cover", "case", "bag", "rack", "stand", "holder", "sign", "seat", "box", "track", "stop", "wheel", "wheels",
         "cap", "hat", "gloves", "glove", "lane", "station", "shop", "shelter", "door", "frame", "handle", "cushion",
         "collar", "leash", "saddle", "bed", "tie", "cloth", "lid", "screen", "plate"}
SHADES = {"-", "light", "dark", "bright", "pale", "deep", "colored", "-colored"}
STOP = {"a", "an", "the", "and", "or", "with", "of", "on", "in", "at", "to", "by", "from", "near", "next", "under",
        "over", "behind", "is", "are", "was", "were", "has", "have", "its", "their", "his", "her", "this", "that",
        "two", "three", "four", "several", "some", "many", "few", "while", "as", "who", "which", "wearing", "holding"}
GUIDE = """
<b>Question:</b> is there at least one object of this kind <b>with this color</b> in the photo?<br>
<b>Present</b>: you can see such an object and its main color matches (for two colors, both must be clearly part of it).<br>
<b>Absent</b>: there is no such object, or it has a clearly different color.<br>
<b>Unsure</b>: the object is too small or dark, or the color is borderline (e.g. gray vs. silver vs. white in shadow). Use sparingly.<br>
Judge only from the image. Objects inside a picture or screen do not count.
"""


def color_claims(caption: str):
    """Return [(color_phrase, coco_class)] for color words directly before a COCO object."""
    toks = _tokenize(caption)
    neg = _negated_positions(toks)
    spans = _match_phrases(toks)
    in_span = {p for s, e, _ in spans for p in range(s, e)}
    out = []
    for s, e, label in spans:
        if label not in COCO_SET or any(p in neg for p in range(s, e)):
            continue
        if e < len(toks) and toks[e] in HEADS:
            continue                  # "red ski boots", "white bus stop": the color belongs to the next noun
        cols, j, gap = [], s - 1, 0
        while j >= 0:
            t = toks[j]
            parts = [p for p in t.split("-") if p]
            hy = len(parts) > 1 and set(parts) <= COLORS | {"and", "colored", "light", "dark"} and set(parts) & COLORS
            if t in COLORS:
                cols.append("gray" if t == "grey" else t)
            elif hy:                  # "black-and-white", "red-colored"
                cols += ["gray" if p == "grey" else p for p in reversed(parts) if p in COLORS]
            elif t in SHADES or (t == "and" and cols):
                pass
            elif cols or t in PUNCT or t in STOP or j in in_span or gap >= 2:
                break                 # colors end here, or the gap reached another noun / function word
            else:
                gap += 1              # modifiers between color and object ("red double-decker bus")
            j -= 1
        if cols:
            if label == "orange" and toks[s] == "orange":
                continue              # the color itself, not the fruit
            out.append((" and ".join(reversed(cols)), label))
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(V3.parent / "val2017"))
    ap.add_argument("--n_items", type=int, default=200,
                    help="labeling set size: every item on the 100 human images, then whole extra images (seeded) up to this")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    human_imgs = {Path(l.strip()).name for l in open(V3 / "data/human_validation_100.txt") if l.strip()}
    allrows, items = [], OrderedDict()
    for cap in CAPTIONERS:
        for r in map(json.loads, open(V3 / f"results/{cap}.jsonl")):
            name = Path(r["image"]).name
            for col, obj in color_claims(r.get("caption") or ""):
                allrows.append({"captioner": cap, "image": name, "color": col, "object": obj})
                items.setdefault((name, col, obj), []).append(cap)
    with open(V3 / "data/color_claims_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["captioner", "image", "color", "object"]); w.writeheader(); w.writerows(allrows)

    by_img: "OrderedDict[str, list]" = OrderedDict()
    for k in items:
        by_img.setdefault(k[0], []).append(k)
    rng = random.Random(a.seed)
    chosen = sorted(im for im in by_img if im in human_imgs)
    extra = sorted(im for im in by_img if im not in human_imgs)
    rng.shuffle(extra)
    n = sum(len(by_img[im]) for im in chosen)
    for im in extra:
        if n >= a.n_items:
            break
        chosen.append(im); n += len(by_img[im])
    keys = [k for im in chosen for k in by_img[im]]
    rng.shuffle(keys)
    with open(V3 / "data/color_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair_id", "image", "object", "color", "coco_class", "captioners", "in_human100"])
        for i, (im, col, obj) in enumerate(keys, 1):
            w.writerow([i, im, f"{col} {obj}", col, obj, ";".join(items[(im, col, obj)]), int(im in human_imgs)])

    groups: "OrderedDict[str, list]" = OrderedDict()
    for i, (im, col, obj) in enumerate(keys, 1):
        groups.setdefault(im, []).append({"id": i, "obj": f"{col} {obj}", "hint": ""})
    order = list(groups); rng.shuffle(order)
    data = [{"image": im, "b64": mlp.img_b64(Path(a.images) / im), "items": groups[im]} for im in order]
    html = (mlp.TEMPLATE.replace("aijudge_labels_v2", "aijudge_color_labels_v1")
            .replace("AI Judge - blinded object labeling", "AI Judge - blinded color labeling")
            .replace("<b>Blinded object labeling</b>", "<b>Blinded color labeling</b>")
            .replace("labels_${who}.csv", "color_labels_${who}.csv")
            .replace("__DATA__", json.dumps(data)).replace("__GUIDE__", GUIDE).replace("__N__", str(len(keys))))
    (V3 / "color_label_tool.html").write_text(html, encoding="utf-8")
    per_cap = {c: sum(r["captioner"] == c for r in allrows) for c in CAPTIONERS}
    n_h = sum(k[0] in human_imgs for k in keys)
    print(f"color claims on 1,000 images: {len(allrows)} {per_cap}; {len(items)} unique (image, color, object)")
    print(f"labeling set: {len(keys)} items on {len(groups)} images ({n_h} on the 100 human images)")
    print("saved data/color_claims_all.csv, data/color_pairs.csv, color_label_tool.html")


if __name__ == "__main__":
    main()
