#!/usr/bin/env python3
"""
compute_quality_metrics.py

Post-hoc enrichment script for the semantic extension.

Reads raw runs.jsonl produced by bench_semantic_hf.py and writes a sibling
runs_enriched.jsonl containing:
  - expected_entities       (per record, loaded from COCO instances_val2017.json)
  - expected_counts         (per record, dict of category -> ground-truth count)
  - per-output additions:
      embedding             (sentence-transformer mpnet vector)
      entity_recall         (fraction of expected entities mentioned)
      count_accuracy        (exact-match rate on objects_and_counts task only)
      gen_len               (backfilled for Moondream via tokenization)
      tokens_per_sec        (backfilled for Moondream)

Records are otherwise unchanged so analyze_results.py / check_hallucination.py
still work on the enriched file.

Why this is a separate script:
  - Enrichment uses ML models (sentence-transformer, tokenizer) that would
    contaminate hardware telemetry if loaded during the bench run.
  - You can re-run enrichment any time with different metrics / encoders
    without redoing the expensive (hours-long) bench run.

Usage:
    # enrich a single run
    python compute_quality_metrics.py \\
        --runs outputs/semantic_extension/moondream/<run_group>/runs.jsonl \\
        --coco_annotations data/annotations/instances_val2017.json \\
        --output outputs/semantic_extension/moondream/<run_group>/runs_enriched.jsonl

    # enrich every runs.jsonl under a parent dir, in place (writes runs_enriched.jsonl alongside each)
    python compute_quality_metrics.py \\
        --runs_glob 'outputs/semantic_extension/*/*/runs.jsonl' \\
        --coco_annotations data/annotations/instances_val2017.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# COCO annotation loading
# ──────────────────────────────────────────────────────────────────────────────

def load_coco_annotations(path: Path) -> dict:
    """
    Load instances_val2017.json and build two lookups per image_id:
      - expected_entities: sorted list of unique COCO category names in the image
      - expected_counts:   dict[category_name] -> count (number of instances)

    Returns:
        {
          "entities_by_image": {image_id: [category names sorted]},
          "counts_by_image":   {image_id: {category_name: count}},
          "categories":        [list of all COCO category names],
        }
    """
    print(f"[coco] loading {path}")
    t0 = time.perf_counter()
    with open(path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    counts_by_image: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name:
            counts_by_image[img_id][cat_name] += 1

    entities_by_image = {
        img_id: sorted(cats.keys())
        for img_id, cats in counts_by_image.items()
    }
    counts_by_image_plain = {
        img_id: dict(cats) for img_id, cats in counts_by_image.items()
    }

    t1 = time.perf_counter()
    print(f"[coco] loaded {len(entities_by_image)} images, "
          f"{len(cat_id_to_name)} categories, in {t1-t0:.1f}s")

    return {
        "entities_by_image": entities_by_image,
        "counts_by_image": counts_by_image_plain,
        "categories": sorted(cat_id_to_name.values()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metric: entity recall (substring match against COCO categories)
# ──────────────────────────────────────────────────────────────────────────────

# Map of COCO category aliases — colloquial words models actually use.
# Conservative; only obvious mappings. Keeps the metric defensible.
#
# The `person` aliases are the most consequential — without them, captions
# like "A woman is standing..." score 0 for person recall even when correct.
# Vehicle and animal aliases are similarly common.
COCO_ALIASES = {
    # People — critical, models almost never say "person"
    "person": ["person", "man", "woman", "boy", "girl", "child", "kid",
               "people", "individual", "human", "lady", "gentleman", "guy"],

    # Vehicles
    "car":         ["car", "vehicle", "automobile", "sedan", "suv"],
    "truck":       ["truck", "pickup", "lorry"],
    "motorcycle":  ["motorcycle", "motorbike", "moped"],
    "airplane":    ["airplane", "plane", "aircraft", "jet", "airliner"],
    "bicycle":     ["bicycle", "bike", "cycle"],
    "boat":        ["boat", "ship", "vessel", "yacht", "sailboat"],
    "bus":         ["bus", "coach"],
    "train":       ["train", "locomotive"],

    # Common items where models use synonyms
    "tv":             ["tv", "television", "monitor", "screen"],
    "sofa":           ["sofa", "couch"],
    "couch":          ["sofa", "couch"],
    "dining table":   ["dining table", "table", "dinner table"],
    "potted plant":   ["potted plant", "plant", "houseplant", "pot plant"],
    "cell phone":     ["cell phone", "cellphone", "mobile phone", "phone", "smartphone"],
    "wine glass":     ["wine glass", "wineglass", "glass of wine"],
    "hair drier":     ["hair drier", "hair dryer", "blow dryer", "hairdryer"],
    "donut":          ["donut", "doughnut"],
    "frisbee":        ["frisbee", "frisby"],
    "skis":           ["skis", "ski"],
    "sports ball":    ["sports ball", "ball", "soccer ball", "basketball", "tennis ball"],
    "baseball bat":   ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "tennis racket":  ["tennis racket", "racket", "racquet"],
    "fire hydrant":   ["fire hydrant", "hydrant"],
    "stop sign":      ["stop sign"],
    "parking meter":  ["parking meter"],
    "traffic light":  ["traffic light", "stoplight", "traffic signal"],

    # Animals — usually OK by exact match, but a few aliases
    "cow":   ["cow", "cattle", "bull", "calf"],
    "sheep": ["sheep", "lamb"],
    "bird":  ["bird"],
    "cat":   ["cat", "kitten", "kitty"],
    "dog":   ["dog", "puppy"],

    # Furniture & objects
    "bottle":  ["bottle"],
    "cup":     ["cup", "mug"],
    "remote":  ["remote", "remote control"],
    "keyboard": ["keyboard"],
    "laptop":  ["laptop", "computer", "notebook computer"],
    "mouse":   ["mouse", "computer mouse"],
    "scissors": ["scissors"],
    "toothbrush": ["toothbrush"],
}


def entity_in_text(category: str, text_lower: str) -> bool:
    """Check if a COCO category appears in text. Uses aliases for common variants."""
    aliases = COCO_ALIASES.get(category, [category])
    for alias in aliases:
        # word-boundary match for short words to avoid false positives
        # (e.g. "car" matching "card"); but allow multi-word phrases as substring
        if " " in alias:
            if alias in text_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(alias)}s?\b", text_lower):
                return True
    return False


def compute_entity_recall(text: str, expected_entities: list[str]) -> float | None:
    """Fraction of expected entities that appear in the text."""
    if not expected_entities:
        return None
    if not text:
        return 0.0
    tl = text.lower()
    hits = sum(1 for e in expected_entities if entity_in_text(e, tl))
    return round(hits / len(expected_entities), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Metric: count accuracy (objects_and_counts task only)
# ──────────────────────────────────────────────────────────────────────────────

# Lines like "table: 1", "vase: 3", "- chair: 2", "* dog: 1", etc.
COUNT_LINE_RE = re.compile(
    r"^[\s\-\*•·]*([A-Za-z][A-Za-z\s_-]+?)\s*[:=]\s*(\d+)\s*$",
    re.MULTILINE,
)


def parse_predicted_counts(text: str) -> dict[str, int]:
    """Extract object: count pairs from the model's text output."""
    counts = {}
    if not text:
        return counts
    for cat, n in COUNT_LINE_RE.findall(text):
        cat = cat.strip().lower()
        if cat and cat not in counts:  # take first occurrence (looped outputs)
            counts[cat] = int(n)
    return counts


def normalize_category(name: str) -> str:
    """Lowercase, strip, replace separators. For matching to COCO categories."""
    return re.sub(r"[\s_-]+", " ", name.strip().lower())


def compute_count_accuracy(
    text: str,
    expected_counts: dict[str, int],
) -> dict | None:
    """
    Compute count accuracy on objects_and_counts output.

    Returns:
        {
          "n_categories_mentioned":  int,
          "n_categories_correct":    int (exact count match),
          "n_categories_present":    int (mentioned a real expected category at all),
          "count_exact_accuracy":    float in [0, 1] over expected categories,
          "count_mae":               mean absolute count error over mentioned categories,
        }
        or None if expected_counts is empty.
    """
    if not expected_counts:
        return None

    predicted = parse_predicted_counts(text)
    if not predicted:
        return {
            "n_categories_mentioned": 0,
            "n_categories_correct": 0,
            "n_categories_present": 0,
            "count_exact_accuracy": 0.0,
            "count_mae": None,
        }

    # Normalize expected category names for matching
    expected_norm = {normalize_category(k): v for k, v in expected_counts.items()}

    n_mentioned = len(predicted)
    n_correct = 0
    n_present = 0
    abs_errs = []

    for pred_cat, pred_n in predicted.items():
        norm = normalize_category(pred_cat)
        # exact match or alias hit
        true_n = expected_norm.get(norm)
        if true_n is None:
            # try aliases for COCO categories
            for canon, aliases in COCO_ALIASES.items():
                if norm in aliases and normalize_category(canon) in expected_norm:
                    true_n = expected_norm[normalize_category(canon)]
                    break
        if true_n is not None:
            n_present += 1
            abs_errs.append(abs(pred_n - true_n))
            if pred_n == true_n:
                n_correct += 1

    return {
        "n_categories_mentioned": n_mentioned,
        "n_categories_correct": n_correct,
        "n_categories_present": n_present,
        "count_exact_accuracy": round(n_correct / len(expected_counts), 4),
        "count_mae": round(float(np.mean(abs_errs)), 4) if abs_errs else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingEncoder:
    """Lazily-loaded sentence transformer. mpnet-base-v2 by default."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            print(f"[embed] loading {self.model_name}")
            t0 = time.perf_counter()
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            t1 = time.perf_counter()
            print(f"[embed] loaded in {t1-t0:.1f}s")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Batched encoding. Returns list of vectors (lists), not numpy."""
        self._ensure_loaded()
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        return [emb.tolist() for emb in embeddings]


# ──────────────────────────────────────────────────────────────────────────────
# Moondream backfill (tokenize output text post-hoc to recover gen_len)
# ──────────────────────────────────────────────────────────────────────────────

class MoondreamTokenizer:
    """Lazily-loaded Moondream tokenizer for backfilling gen_len."""

    def __init__(self, model_id: str = "vikhyatk/moondream2"):
        self.model_id = model_id
        self.tokenizer = None
        self.failed = False

    def _ensure_loaded(self):
        if self.tokenizer is not None or self.failed:
            return
        try:
            from transformers import AutoTokenizer
            print(f"[tok] loading Moondream tokenizer ({self.model_id})")
            t0 = time.perf_counter()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True,
            )
            t1 = time.perf_counter()
            print(f"[tok] loaded in {t1-t0:.1f}s")
        except Exception as e:
            print(f"[tok] WARN: failed to load Moondream tokenizer: {e}")
            print(f"[tok] gen_len will remain null for Moondream records")
            self.failed = True

    def count(self, text: str) -> int | None:
        self._ensure_loaded()
        if self.tokenizer is None:
            return None
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Enrichment over one runs.jsonl
# ──────────────────────────────────────────────────────────────────────────────

def enrich_file(
    runs_path: Path,
    output_path: Path,
    coco: dict,
    encoder: EmbeddingEncoder,
    md_tokenizer: MoondreamTokenizer,
    overwrite: bool = False,
) -> dict:
    """
    Read runs_path (JSONL), enrich each record, write to output_path.
    Returns summary stats.
    """
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} exists (use --overwrite to redo)")
        return {"skipped": True, "path": str(output_path)}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: read all records, collect all texts for batched embedding
    records = []
    all_texts = []  # (record_idx, output_idx, text)
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line: {e}")
                continue
            records.append(rec)
            for o_idx, o in enumerate(rec.get("outputs", []) or []):
                text = o.get("text") or ""
                if text:
                    all_texts.append((len(records) - 1, o_idx, text))

    if not records:
        print(f"[warn] no records in {runs_path}")
        return {"n_records": 0, "path": str(output_path)}

    print(f"[enrich] {runs_path.name}: {len(records)} records, "
          f"{len(all_texts)} non-empty outputs")

    # Pass 2: batched embedding
    texts_only = [t[2] for t in all_texts]
    if texts_only:
        t0 = time.perf_counter()
        embeddings = encoder.encode(texts_only)
        t1 = time.perf_counter()
        print(f"[enrich] embedded {len(texts_only)} texts in {t1-t0:.1f}s")
    else:
        embeddings = []

    # Map back: (rec_idx, output_idx) -> embedding
    emb_map = {}
    for (rec_idx, o_idx, _), emb in zip(all_texts, embeddings):
        emb_map[(rec_idx, o_idx)] = emb

    # Pass 3: per-record enrichment
    is_moondream = any(r.get("model_key") == "moondream" for r in records[:5])

    entities_by_image = coco["entities_by_image"]
    counts_by_image = coco["counts_by_image"]

    n_with_expected = 0
    n_outputs_total = 0
    n_outputs_with_text = 0
    n_outputs_with_recall = 0

    with open(output_path, "w") as fout:
        for rec_idx, rec in enumerate(records):
            image_id = rec.get("image_id")
            task = rec.get("task")

            # Attach COCO ground truth
            expected_entities = entities_by_image.get(image_id, []) if image_id else []
            expected_counts = counts_by_image.get(image_id, {}) if image_id else {}

            rec["expected_entities"] = expected_entities
            rec["expected_counts"] = expected_counts
            if expected_entities:
                n_with_expected += 1

            # Per-output enrichment
            for o_idx, o in enumerate(rec.get("outputs", []) or []):
                n_outputs_total += 1
                text = o.get("text") or ""
                if not text:
                    continue
                n_outputs_with_text += 1

                # embedding
                emb = emb_map.get((rec_idx, o_idx))
                if emb is not None:
                    o["embedding"] = emb

                # entity recall
                recall = compute_entity_recall(text, expected_entities)
                if recall is not None:
                    o["entity_recall"] = recall
                    n_outputs_with_recall += 1

                # count accuracy (only on objects_and_counts task)
                if task == "objects_and_counts":
                    count_metrics = compute_count_accuracy(text, expected_counts)
                    if count_metrics is not None:
                        o["count_accuracy"] = count_metrics

                # Moondream backfill: tokenize to recover gen_len, then tps
                if is_moondream and o.get("gen_len") is None:
                    n_tok = md_tokenizer.count(text)
                    if n_tok is not None:
                        o["gen_len"] = n_tok
                        gen_lat = o.get("gen_latency_s")
                        if gen_lat and gen_lat > 0:
                            o["tokens_per_sec"] = round(n_tok / gen_lat, 3)

            fout.write(json.dumps(rec) + "\n")

    summary = {
        "n_records": len(records),
        "n_records_with_expected_entities": n_with_expected,
        "n_outputs_total": n_outputs_total,
        "n_outputs_with_text": n_outputs_with_text,
        "n_outputs_with_recall": n_outputs_with_recall,
        "path": str(output_path),
    }
    print(f"[done] {output_path.name}: {summary['n_outputs_with_recall']} / "
          f"{summary['n_outputs_with_text']} outputs got entity_recall")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", help="Path to a single runs.jsonl")
    p.add_argument("--runs_glob",
                   help="Glob for multiple runs.jsonl (e.g. 'outputs/**/runs.jsonl')")
    p.add_argument("--output",
                   help="Output path (only valid with --runs). Default: alongside input "
                        "as runs_enriched.jsonl")
    p.add_argument("--coco_annotations", required=True,
                   help="Path to instances_val2017.json")
    p.add_argument("--embedding_model",
                   default="sentence-transformers/all-mpnet-base-v2",
                   help="Sentence transformer model name (default mpnet-base-v2)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing runs_enriched.jsonl files")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.runs and not args.runs_glob:
        sys.exit("must provide --runs or --runs_glob")
    if args.runs and args.runs_glob:
        sys.exit("--runs and --runs_glob are mutually exclusive")

    coco_path = Path(args.coco_annotations).resolve()
    if not coco_path.exists():
        sys.exit(f"coco annotations not found: {coco_path}")
    coco = load_coco_annotations(coco_path)

    encoder = EmbeddingEncoder(args.embedding_model)
    md_tokenizer = MoondreamTokenizer()

    if args.runs:
        runs_path = Path(args.runs).resolve()
        if not runs_path.exists():
            sys.exit(f"runs not found: {runs_path}")
        if args.output:
            output_path = Path(args.output).resolve()
        else:
            output_path = runs_path.parent / "runs_enriched.jsonl"
        enrich_file(runs_path, output_path, coco, encoder, md_tokenizer,
                    overwrite=args.overwrite)
    else:
        # glob mode
        paths = sorted(glob(args.runs_glob, recursive=True))
        if not paths:
            sys.exit(f"no files matched: {args.runs_glob}")
        print(f"[main] found {len(paths)} runs files")
        for p in paths:
            runs_path = Path(p).resolve()
            output_path = runs_path.parent / "runs_enriched.jsonl"
            print()
            enrich_file(runs_path, output_path, coco, encoder, md_tokenizer,
                        overwrite=args.overwrite)


if __name__ == "__main__":
    main()