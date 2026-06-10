#!/usr/bin/env python3
"""
compute_clipscore.py

Adds CLIPScore (image-text alignment via CLIP embeddings) to rescored runs.jsonl
files. CLIPScore is reference-free: it scores the model output directly against
the image without needing human captions.

Reference: Hessel et al. 2021, "CLIPScore: A Reference-free Evaluation Metric
for Image Captioning" (EMNLP 2021).

Note on the constant: the original CLIPScore paper multiplies cosine similarity
by 2.5 (clamped at 0) so scores fall in a more interpretable [0, 2.5] range.
We store both the raw cosine and the scaled CLIPScore.

For each output in each rescored runs.jsonl, adds:
    clipscore_cos    = raw cosine similarity between image and output text embeddings
    clipscore        = max(0, 2.5 * cosine) — the canonical Hessel et al. formula

Image embeddings are computed once and cached in memory (500 images × 1 embedding).

USAGE:
    python compute_clipscore.py \\
        --image_dir data/coco/val2017 \\
        --manifest data/testsets/coco_val2017_500.txt \\
        --rescored MODEL:DEVICE:PATH \\
        ... \\
        [--model_id openai/clip-vit-large-patch14] \\
        [--device cuda] \\
        [--batch_size 32]

OUTPUT: For each input rescored.jsonl, writes a sibling *_clipscored.jsonl with
the new fields added (does not overwrite the input).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image_map(image_dir, manifest_path):
    """
    Returns {image_id_int: full_path_str} for every image in the manifest.
    The manifest contains one full path per line (e.g. /.../val2017/000000000139.jpg).
    image_id is the int parsed out of the filename.
    """
    image_dir = Path(image_dir).resolve()
    image_map = {}
    with open(manifest_path) as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            # filename like 000000000139.jpg → image_id 139
            name = Path(p).stem
            try:
                img_id = int(name)
            except ValueError:
                continue
            # Prefer the path as-given in the manifest if it exists
            if os.path.exists(p):
                image_map[img_id] = p
            else:
                # Fallback: build from image_dir + filename
                fallback = str(image_dir / Path(p).name)
                if os.path.exists(fallback):
                    image_map[img_id] = fallback
    return image_map


def embed_images(image_map, model, processor, device, batch_size):
    """Embed every image in the manifest. Returns {image_id: np.ndarray (n_dim,)}."""
    img_ids = sorted(image_map.keys())
    print(f"[clip] embedding {len(img_ids)} images...")

    embeddings = {}
    t0 = time.time()

    for batch_start in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[batch_start:batch_start + batch_size]
        batch_paths = [image_map[i] for i in batch_ids]

        pil_images = []
        for p in batch_paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  ERROR loading {p}: {e}")
                pil_images.append(None)

        # Replace any None with a blank image (keeps batch shape; we'll skip these)
        valid_indices = [i for i, im in enumerate(pil_images) if im is not None]
        if not valid_indices:
            continue
        pil_images = [im for im in pil_images if im is not None]
        batch_ids = [batch_ids[i] for i in valid_indices]

        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # transformers 5.x may return a dataclass; extract the tensor
        if hasattr(image_features, "image_embeds"):
            image_features = image_features.image_embeds
        elif hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        elif hasattr(image_features, "last_hidden_state"):
            # fallback — average pool
            image_features = image_features.last_hidden_state.mean(dim=1)

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        for img_id, emb in zip(batch_ids, image_features.cpu().numpy()):
            embeddings[img_id] = emb.astype(np.float32)

        if (batch_start // batch_size) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  ... {batch_start + len(batch_ids)}/{len(img_ids)} "
                  f"in {elapsed:.1f}s")

    print(f"[clip] done embedding images in {time.time() - t0:.1f}s, "
          f"{len(embeddings)} successful")
    return embeddings


def embed_texts_batch(texts, model, processor, device, batch_size):
    """Embed a list of strings. Returns np.ndarray (n_texts, dim) normalized."""
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)

    # CLIP's text encoder caps at 77 tokens (~50-60 words). We truncate long
    # outputs to fit; this is standard practice for CLIPScore on long captions.
    all_embs = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        inputs = processor(
            text=batch, return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        ).to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        if hasattr(text_features, "text_embeds"):
            text_features = text_features.text_embeds
        elif hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        elif hasattr(text_features, "last_hidden_state"):
            text_features = text_features.last_hidden_state.mean(dim=1)

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        all_embs.append(text_features.cpu().numpy())

    return np.concatenate(all_embs, axis=0).astype(np.float32)


def score_one_file(path, image_embeddings, model, processor, device,
                   batch_size, out_path):
    """For each output in rescored.jsonl, compute clipscore and write rescored file."""
    print(f"\n[score] reading {path}")

    # Pass 1 — collect all (text, image_id) pairs
    records = []
    flat_texts = []
    flat_meta = []  # (rec_idx, out_idx, img_id, valid)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            rec_idx = len(records) - 1
            img_id = rec.get("image_id")
            try:
                img_id_int = int(img_id)
            except (TypeError, ValueError):
                img_id_int = None

            for o_idx, out in enumerate(rec.get("outputs", [])):
                txt = out.get("text") or ""
                if (not out.get("error") and txt and
                        img_id_int is not None and img_id_int in image_embeddings):
                    flat_texts.append(txt)
                    flat_meta.append((rec_idx, o_idx, img_id_int, True))
                else:
                    flat_meta.append((rec_idx, o_idx, img_id_int, False))

    n_to_score = sum(1 for m in flat_meta if m[3])
    print(f"[score] records: {len(records)}, outputs to score: {n_to_score} "
          f"(skipping {len(flat_meta) - n_to_score})")

    # Pass 2 — embed all texts in batches
    print(f"[score] embedding {len(flat_texts)} texts...")
    t0 = time.time()
    text_embs = embed_texts_batch(flat_texts, model, processor, device, batch_size)
    print(f"[score] embedded in {time.time() - t0:.1f}s")

    # Pass 3 — compute scores
    scores_cos = []
    scores_clip = []
    valid_iter = iter(range(len(text_embs)))

    for rec_idx, o_idx, img_id, valid in flat_meta:
        out = records[rec_idx]["outputs"][o_idx]
        if not valid:
            continue
        i = next(valid_iter)
        img_emb = image_embeddings[img_id]
        txt_emb = text_embs[i]
        cos = float(np.dot(img_emb, txt_emb))  # already normalized
        clip = max(0.0, 2.5 * cos)
        out["clipscore_cos"] = cos
        out["clipscore"] = clip
        scores_cos.append(cos)
        scores_clip.append(clip)

    # Pass 4 — write output
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    if scores_cos:
        print(f"[score] clipscore_cos:  mean={np.mean(scores_cos):.4f}  "
              f"median={np.median(scores_cos):.4f}  "
              f"min={np.min(scores_cos):.4f}  max={np.max(scores_cos):.4f}")
        print(f"[score] clipscore (scaled): mean={np.mean(scores_clip):.4f}  "
              f"median={np.median(scores_clip):.4f}")
    print(f"[score] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True,
                    help="Directory containing COCO val2017 images")
    ap.add_argument("--manifest", required=True,
                    help="Manifest file: one image path per line")
    ap.add_argument("--rescored", action="append", required=True,
                    help="Path to a rescored runs.jsonl (repeat for multiple)")
    ap.add_argument("--model_id", default="openai/clip-vit-large-patch14",
                    help="HuggingFace CLIP model id")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.image_dir):
        print(f"ERROR: image_dir not found: {args.image_dir}")
        sys.exit(1)
    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}")
        sys.exit(1)

    print(f"[clip] loading CLIP model: {args.model_id}")
    from transformers import CLIPModel, CLIPProcessor
    t0 = time.time()
    model = CLIPModel.from_pretrained(args.model_id).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_id)
    print(f"[clip] loaded in {time.time() - t0:.1f}s")

    # Load image map and embed all images once
    print(f"[clip] loading image map from {args.manifest}")
    image_map = load_image_map(args.image_dir, args.manifest)
    print(f"[clip] {len(image_map)} images located on disk")

    image_embeddings = embed_images(
        image_map, model, processor, args.device, args.batch_size)

    # Score each rescored file
    for path in args.rescored:
        if not os.path.exists(path):
            print(f"  WARNING: file not found, skipping: {path}")
            continue
        out_path = path.replace("_rescored.jsonl", "_clipscored.jsonl")
        if out_path == path:
            out_path = path.rsplit(".jsonl", 1)[0] + "_clipscored.jsonl"
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite)")
            continue
        score_one_file(path, image_embeddings, model, processor,
                       args.device, args.batch_size, out_path)

    print("\n[done] all files scored")


if __name__ == "__main__":
    main()