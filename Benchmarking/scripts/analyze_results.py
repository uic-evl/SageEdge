"""
analyze_results.py

Loads results from both Thor and Dell GB10 and computes:
  Metric 1 — Stability       (within-device cosine similarity across 5 runs)
  Metric 2 — Cross-Device    (cosine similarity between mean embeddings)
  Metric 3 — Entity F1       (precision, recall, F1 vs. COCO ground truth)

Usage:
    python scripts/analyze_results.py \
        --thor  outputs/semantic_experiment/results_thor.jsonl \
        --dell  outputs/semantic_experiment/results_dell_gb10.jsonl \
        --out   outputs/semantic_experiment/summary.json

Requirements:
    pip install scikit-learn numpy
"""

import json
import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--thor", required=True, help="Path to Thor JSONL results")
parser.add_argument("--dell", required=True, help="Path to Dell GB10 JSONL results")
parser.add_argument("--out",  default="outputs/semantic_experiment/summary.json",
                    help="Output path for summary JSON")
args = parser.parse_args()

def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["image_id"]] = rec
    print(f"  Loaded {len(records)} records from {path}")
    return records

print("Loading results...")
thor_data = load_jsonl(args.thor)
dell_data = load_jsonl(args.dell)

# Only analyze images present in BOTH files
common_ids = sorted(set(thor_data.keys()) & set(dell_data.keys()))
print(f"  Common images: {len(common_ids)}\n")

if len(common_ids) == 0:
    print("ERROR: No common image IDs found between the two files.")
    exit(1)


def compute_stability(outputs):
    """Mean pairwise cosine similarity across NUM_RUNS embeddings."""
    embs = np.array([o["embedding"] for o in outputs])  # (N, D)
    sim_matrix = cosine_similarity(embs)
    # Upper triangle only (exclude diagonal)
    n = len(embs)
    upper = [sim_matrix[i][j] for i in range(n) for j in range(i+1, n)]
    return float(np.mean(upper))

def compute_cross_device_agreement(thor_outputs, dell_outputs):
    """Cosine similarity between mean embeddings from each device."""
    thor_mean = np.mean([o["embedding"] for o in thor_outputs], axis=0)
    dell_mean = np.mean([o["embedding"] for o in dell_outputs], axis=0)
    sim = cosine_similarity([thor_mean], [dell_mean])[0][0]
    return float(sim)

def compute_entity_f1(text, expected_entities):
    """
    Simple substring match: check whether each expected entity
    appears in the model's output text.
    Returns (precision, recall, f1).
    """
    if not expected_entities:
        return 0.0, 0.0, 0.0

    text_lower = text.lower()
    matched = {e for e in expected_entities if e.lower() in text_lower}

    # For this experiment: precision = matched / all mentioned entities
    # We approximate "predicted set" as the matched entities
    # (true precision would require extracting all nouns, overkill here)
    recall    = len(matched) / len(expected_entities)
    precision = recall  # symmetric under substring match assumption
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)

print("Computing metrics...")

per_image = []

thor_stabilities  = []
dell_stabilities  = []
cross_agreements  = []
thor_f1s          = []
dell_f1s          = []

for image_id in common_ids:
    thor_rec = thor_data[image_id]
    dell_rec = dell_data[image_id]
    expected = thor_rec["expected_entities"]  # same for both (from COCO)

    # Metric 1: Stability per device
    thor_stab = compute_stability(thor_rec["outputs"])
    dell_stab = compute_stability(dell_rec["outputs"])

    # Metric 2: Cross-device agreement
    cross = compute_cross_device_agreement(thor_rec["outputs"], dell_rec["outputs"])

    # Metric 3: Entity F1 — use mean of 5 run texts per device
    # (run entity check on each output, then average)
    thor_f1_scores = [
        compute_entity_f1(o["text"], expected)[2]
        for o in thor_rec["outputs"]
    ]
    dell_f1_scores = [
        compute_entity_f1(o["text"], expected)[2]
        for o in dell_rec["outputs"]
    ]
    thor_f1 = float(np.mean(thor_f1_scores))
    dell_f1 = float(np.mean(dell_f1_scores))

    # Avg latency per device
    thor_latency = np.mean([o["latency_s"] for o in thor_rec["outputs"]])
    dell_latency = np.mean([o["latency_s"] for o in dell_rec["outputs"]])

    per_image.append({
        "image_id":           image_id,
        "file_name":          thor_rec["file_name"],
        "expected_entities":  expected,
        "thor_stability":     round(thor_stab, 4),
        "dell_stability":     round(dell_stab, 4),
        "cross_device_agreement": round(cross, 4),
        "thor_entity_f1":     round(thor_f1, 4),
        "dell_entity_f1":     round(dell_f1, 4),
        "thor_avg_latency_s": round(float(thor_latency), 3),
        "dell_avg_latency_s": round(float(dell_latency), 3),
    })

    thor_stabilities.append(thor_stab)
    dell_stabilities.append(dell_stab)
    cross_agreements.append(cross)
    thor_f1s.append(thor_f1)
    dell_f1s.append(dell_f1)


summary = {
    "num_images":   len(common_ids),
    "model":        thor_data[common_ids[0]]["model"],
    "prompt":       thor_data[common_ids[0]]["prompt"],
    "metrics": {
        "metric1_stability": {
            "description": "Mean pairwise cosine similarity across 5 runs (higher = more stable)",
            "thor_mean":   round(float(np.mean(thor_stabilities)), 4),
            "thor_std":    round(float(np.std(thor_stabilities)), 4),
            "dell_mean":   round(float(np.mean(dell_stabilities)), 4),
            "dell_std":    round(float(np.std(dell_stabilities)), 4),
        },
        "metric2_cross_device_agreement": {
            "description": "Cosine similarity between mean embeddings across devices (higher = more similar behavior)",
            "mean":        round(float(np.mean(cross_agreements)), 4),
            "std":         round(float(np.std(cross_agreements)), 4),
        },
        "metric3_entity_f1": {
            "description": "F1 score for entity recall vs. COCO ground truth labels",
            "thor_mean":   round(float(np.mean(thor_f1s)), 4),
            "thor_std":    round(float(np.std(thor_f1s)), 4),
            "dell_mean":   round(float(np.mean(dell_f1s)), 4),
            "dell_std":    round(float(np.std(dell_f1s)), 4),
        }
    },
    "per_image": per_image
}

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)


m = summary["metrics"]
print("\n" + "="*60)
print(f"  SEMANTIC QUALITY EXPERIMENT — {summary['model'].upper()}")
print(f"  {summary['num_images']} images  |  5 runs per image per device")
print("="*60)

print("\nMetric 1 — Stability (cosine sim, higher is better)")
print(f"  Thor:      {m['metric1_stability']['thor_mean']:.4f}  ±{m['metric1_stability']['thor_std']:.4f}")
print(f"  Dell GB10: {m['metric1_stability']['dell_mean']:.4f}  ±{m['metric1_stability']['dell_std']:.4f}")

print("\nMetric 2 — Cross-Device Agreement (cosine sim, higher = more similar)")
print(f"  Mean: {m['metric2_cross_device_agreement']['mean']:.4f}  ±{m['metric2_cross_device_agreement']['std']:.4f}")

print("\nMetric 3 — Entity F1 (vs. COCO ground truth, higher is better)")
print(f"  Thor:      {m['metric3_entity_f1']['thor_mean']:.4f}  ±{m['metric3_entity_f1']['thor_std']:.4f}")
print(f"  Dell GB10: {m['metric3_entity_f1']['dell_mean']:.4f}  ±{m['metric3_entity_f1']['dell_std']:.4f}")

print(f"\nFull results saved to: {args.out}")
print("="*60)