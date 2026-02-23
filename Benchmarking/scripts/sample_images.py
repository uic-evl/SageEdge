"""
sample_images.py

One-time script to sample 100 COCO val2017 images and extract
expected entity labels from ground truth annotations.

Output: data/testsets/semantic_experiment_100.json
Run once, then copy to both Thor and Dell GB10 so the experiment
uses the exact same image set on both devices.

Usage:
    python scripts/sample_images.py
"""

import json
import random
import os

# ── Paths ────────────────────────────────────────────────────────────────────
COCO_DIR        = "data/coco"
ANNOTATIONS     = os.path.join(COCO_DIR, "annotations/annotations/instances_val2017.json")
IMAGES_DIR      = os.path.join(COCO_DIR, "val2017")
OUTPUT_PATH     = "data/testsets/semantic_experiment_100.json"

NUM_IMAGES      = 100
RANDOM_SEED     = 42

# ── Load COCO annotations ────────────────────────────────────────────────────
print("Loading COCO annotations...")
with open(ANNOTATIONS) as f:
    coco = json.load(f)

# Build category id -> name map
cat_map = {c["id"]: c["name"] for c in coco["categories"]}

# Sample images deterministically
random.seed(RANDOM_SEED)
sampled_images = random.sample(coco["images"], NUM_IMAGES)
sampled_ids    = {img["id"] for img in sampled_images}

print(f"Sampled {len(sampled_images)} images (seed={RANDOM_SEED})")

# ── Build expected entities per image ────────────────────────────────────────
# Maps image_id -> sorted list of unique COCO category names present
id_to_entities = {}
for ann in coco["annotations"]:
    if ann["image_id"] in sampled_ids:
        id_to_entities.setdefault(ann["image_id"], set()).add(
            cat_map[ann["category_id"]]
        )

# ── Build output records ─────────────────────────────────────────────────────
records = []
for img in sampled_images:
    file_name = img["file_name"]
    full_path = os.path.join(IMAGES_DIR, file_name)

    if not os.path.exists(full_path):
        print(f"  WARNING: image not found: {full_path}")

    records.append({
        "image_id":        img["id"],
        "file_name":       file_name,
        "full_path":       full_path,
        "expected_entities": sorted(id_to_entities.get(img["id"], []))
    })

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(records, f, indent=2)

print(f"Saved {len(records)} records to {OUTPUT_PATH}")

# Quick sanity check
entity_counts = [len(r["expected_entities"]) for r in records]
print(f"Avg entities per image: {sum(entity_counts)/len(entity_counts):.1f}")
print(f"Sample record:\n{json.dumps(records[0], indent=2)}")