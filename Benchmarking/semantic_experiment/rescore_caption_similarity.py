#!/usr/bin/env python3
"""
rescore_caption_similarity.py

Adds caption-similarity quality scores to enriched runs.jsonl files.

For each output in an enriched runs.jsonl, computes:
    caption_similarity_max   = max cosine sim between the output's mpnet embedding
                               and the embeddings of the 5 COCO reference captions
                               for that image.
    caption_similarity_mean  = mean cosine sim (avg over the 5 references)

Reads:
    - captions_val2017.json (COCO captions annotation file)
    - one or more enriched runs.jsonl files (must already have text_embedding fields,
      i.e. output of compute_quality_metrics.py)

Writes:
    - For each input <name>_enriched.jsonl, produces <name>_rescored.jsonl
      with the new fields added to each output dict.

USAGE:
    python rescore_caption_similarity.py \\
        --captions data/annotations/captions_val2017.json \\
        --enriched outputs/.../moondream_thor_*/runs_enriched.jsonl \\
        --enriched outputs/.../smolvlm2_thor_*/runs_enriched.jsonl \\
        [--embedding_model sentence-transformers/all-mpnet-base-v2] \\
        [--device cuda]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_captions(path):
    """Load COCO captions JSON, return {image_id -> [caption_str, ...]}."""
    print(f"[captions] loading {path}")
    with open(path) as f:
        data = json.load(f)

    img_to_captions = defaultdict(list)
    for ann in data["annotations"]:
        img_to_captions[ann["image_id"]].append(ann["caption"])

    total_caps = sum(len(c) for c in img_to_captions.values())
    print(f"[captions] {len(img_to_captions)} images, {total_caps} captions "
          f"(avg {total_caps / len(img_to_captions):.1f} per image)")
    return dict(img_to_captions)


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_all_captions(img_to_captions, model_name, device):
    """
    Embed every caption once. Returns:
        {image_id -> np.ndarray of shape (n_captions, 768)}
    """
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print(f"[embed] loaded on {device}")

    # Flatten for batch encoding
    image_ids = []
    flat_captions = []
    for img_id, caps in img_to_captions.items():
        for c in caps:
            image_ids.append(img_id)
            flat_captions.append(c)

    print(f"[embed] encoding {len(flat_captions)} captions...")
    flat_embs = model.encode(
        flat_captions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    print(f"[embed] done, shape={flat_embs.shape}")

    # Re-group by image_id
    by_image = defaultdict(list)
    for img_id, emb in zip(image_ids, flat_embs):
        by_image[img_id].append(emb)
    by_image = {k: np.stack(v) for k, v in by_image.items()}

    return by_image


def rescore_one_file(runs_path, captions_emb_by_image, out_path):
    """
    For each output in runs_path, compute caption similarity scores.
    Writes a new JSONL with the rescored records.
    """
    print(f"\n[rescore] reading {runs_path}")
    n_records = 0
    n_outputs = 0
    n_rescored = 0
    n_missing_ref = 0
    n_missing_emb = 0

    sims_max = []
    sims_mean = []

    with open(runs_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_records += 1

            img_id = rec.get("image_id")
            # COCO image_ids in annotations are ints; manifest might encode as str
            try:
                img_id_int = int(img_id)
            except (TypeError, ValueError):
                img_id_int = img_id

            ref_embs = captions_emb_by_image.get(img_id_int)
            if ref_embs is None:
                # Maybe the manifest IDs are zero-padded strings; try string key
                ref_embs = captions_emb_by_image.get(img_id)

            for out in rec.get("outputs", []):
                n_outputs += 1
                emb = out.get("embedding") or out.get("text_embedding")
                if emb is None:
                    n_missing_emb += 1
                    continue
                if ref_embs is None:
                    n_missing_ref += 1
                    continue

                # Compute cosine to each reference, take max and mean
                emb = np.asarray(emb, dtype=np.float32)
                # ref_embs shape: (n_caps, 768)
                # normalize
                emb_n = emb / (np.linalg.norm(emb) + 1e-12)
                ref_n = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-12)
                sims = ref_n @ emb_n   # shape (n_caps,)

                s_max = float(np.max(sims))
                s_mean = float(np.mean(sims))
                out["caption_similarity_max"] = s_max
                out["caption_similarity_mean"] = s_mean

                sims_max.append(s_max)
                sims_mean.append(s_mean)
                n_rescored += 1

            f_out.write(json.dumps(rec) + "\n")

    print(f"[rescore] records: {n_records}")
    print(f"[rescore] outputs: {n_outputs}")
    print(f"[rescore] rescored: {n_rescored}")
    if n_missing_ref > 0:
        print(f"[rescore] WARNING: {n_missing_ref} outputs had no matching reference captions")
    if n_missing_emb > 0:
        print(f"[rescore] WARNING: {n_missing_emb} outputs had no text_embedding")
    if sims_max:
        print(f"[rescore] caption_similarity_max:  mean={np.mean(sims_max):.3f}  "
              f"median={np.median(sims_max):.3f}  min={np.min(sims_max):.3f}  max={np.max(sims_max):.3f}")
        print(f"[rescore] caption_similarity_mean: mean={np.mean(sims_mean):.3f}  "
              f"median={np.median(sims_mean):.3f}")
    print(f"[rescore] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True,
                    help="Path to COCO captions_val2017.json")
    ap.add_argument("--enriched", action="append", required=True,
                    help="Path to an enriched runs.jsonl (repeat for multiple)")
    ap.add_argument("--embedding_model",
                    default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing _rescored.jsonl files")
    args = ap.parse_args()

    # Sanity check inputs
    if not os.path.exists(args.captions):
        print(f"ERROR: captions file not found: {args.captions}")
        sys.exit(1)
    for p in args.enriched:
        if not os.path.exists(p):
            print(f"ERROR: enriched file not found: {p}")
            sys.exit(1)

    # Load captions and embed them once
    img_to_captions = load_captions(args.captions)
    captions_emb_by_image = embed_all_captions(
        img_to_captions, args.embedding_model, args.device)

    # Rescore each input file
    for runs_path in args.enriched:
        out_path = runs_path.replace("_enriched.jsonl", "_rescored.jsonl")
        if out_path == runs_path:
            # Wasn't named _enriched; just append _rescored
            out_path = runs_path.rsplit(".jsonl", 1)[0] + "_rescored.jsonl"
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite to redo)")
            continue
        rescore_one_file(runs_path, captions_emb_by_image, out_path)

    print("\n[done] all files rescored")


if __name__ == "__main__":
    main()