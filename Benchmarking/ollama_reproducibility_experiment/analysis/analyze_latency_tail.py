"""
analyze_latency_tail.py

Computes per-(model, device, task) latency tail statistics from the experiment
JSONL files. Reports p50, p95, p99, max, and tail_ratio (p99/p50) — the standard
metrics for understanding deployment-time latency predictability.

Distinguishes generation latency (model inference only) from total latency
(generation + embedding overhead). For deployment latency analysis, you want
generation latency alone — embedding overhead is part of the analysis pipeline
on the device, not the inference pipeline.

Usage:
    # 5-task data, both devices
    python scripts/analyze_latency_tail.py \\
        --input 'outputs/semantic_experiment_5task/*.jsonl' \\
        --out outputs/semantic_experiment_5task/latency_tail.json

    # Single-prompt data, both devices. Files live under thor/ and dell/.
    python scripts/analyze_latency_tail.py \\
        --input 'outputs/semantic_experiment/**/*.jsonl' \\
        --out outputs/semantic_experiment/latency_tail.json

    # Equivalent single-prompt command without a recursive glob
    python scripts/analyze_latency_tail.py \\
        --input 'outputs/semantic_experiment/thor/*.jsonl' \\
        --input 'outputs/semantic_experiment/dell/*.jsonl' \\
        --out outputs/semantic_experiment/latency_tail.json

    # Filter to one task
    python scripts/analyze_latency_tail.py \\
        --input 'outputs/semantic_experiment_5task/*.jsonl' \\
        --tasks caption_brief \\
        --out outputs/latency_tail_caption.json

Output: per-(model, device, task) row containing:
    n_samples, p50, p95, p99, max, mean, std, tail_ratio (p99/p50)

Quote your globs so the shell doesn't expand them.
"""

import argparse
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, action="append",
                    help="Glob(s) of JSONL files. Pass multiple times to merge.")
parser.add_argument("--out", default="outputs/latency_tail.json",
                    help="Output JSON path")
parser.add_argument("--tasks", nargs="+", default=None,
                    help="Restrict to specific task names")
parser.add_argument("--latency-field", default="gen_latency_s",
                    choices=["gen_latency_s", "total_latency_s", "latency_s"],
                    help="Which latency field to analyze (default: gen_latency_s — generation only)")
parser.add_argument("--exclude-empty", action="store_true", default=True,
                    help="Exclude runs with empty text (default: True)")
args = parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def expand_globs(patterns):
    files = []
    for p in patterns:
        m = sorted(glob(p, recursive=True))
        if not m:
            print(f"  WARN: no files match: {p}")
        files.extend(m)
    seen, deduped = set(), []
    for f in files:
        if f not in seen:
            deduped.append(f); seen.add(f)
    return deduped


def get_latency(o, field):
    """Try the requested field; fall back to alternatives."""
    val = o.get(field)
    if val is not None:
        return val
    # Fallback chain
    for alt in ["gen_latency_s", "total_latency_s", "latency_s"]:
        v = o.get(alt)
        if v is not None:
            return v
    return None


# ── Load all latencies, grouped by (model, device, task) ─────────────────────
print("Resolving file globs...")
files = expand_globs(args.input)
print(f"  Files: {len(files)}")

if not files:
    print("\nERROR: no input files.")
    print("  If you are analyzing single-prompt semantic_experiment data, use:")
    print("    --input 'outputs/semantic_experiment/**/*.jsonl'")
    print("  or pass the device folders separately:")
    print("    --input 'outputs/semantic_experiment/thor/*.jsonl' \\")
    print("    --input 'outputs/semantic_experiment/dell/*.jsonl'")
    exit(1)

groups = defaultdict(list)
n_total_runs = 0
n_excluded_empty = 0
n_excluded_no_lat = 0

for path in files:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            model    = rec.get("model")
            device   = rec.get("device")
            task     = rec.get("task")  # None for single-prompt

            if args.tasks and task not in args.tasks:
                continue

            for o in rec.get("outputs", []):
                n_total_runs += 1
                text = (o.get("text") or "").strip()
                if args.exclude_empty and not text:
                    n_excluded_empty += 1
                    continue
                lat = get_latency(o, args.latency_field)
                if lat is None or lat <= 0:
                    n_excluded_no_lat += 1
                    continue
                groups[(model, device, task)].append(lat)

print(f"\nLoaded {n_total_runs} run-records total")
print(f"  Excluded (empty text):   {n_excluded_empty}")
print(f"  Excluded (no latency):   {n_excluded_no_lat}")
print(f"  Groups (model,device,task): {len(groups)}\n")

if not groups:
    print("ERROR: no usable latency data.")
    exit(1)


# ── Compute per-group tail statistics ────────────────────────────────────────
def tail_stats(lats):
    arr = np.array(lats)
    return {
        "n_samples":  len(arr),
        "p50":        round(float(np.percentile(arr, 50)), 4),
        "p95":        round(float(np.percentile(arr, 95)), 4),
        "p99":        round(float(np.percentile(arr, 99)), 4),
        "max":        round(float(np.max(arr)), 4),
        "mean":       round(float(np.mean(arr)), 4),
        "std":        round(float(np.std(arr)), 4),
        "tail_ratio": round(float(np.percentile(arr, 99) / np.percentile(arr, 50)), 3),
    }


results = []
for (model, device, task), lats in sorted(groups.items()):
    stats = tail_stats(lats)
    results.append({
        "model":   model,
        "device":  device,
        "task":    task,
        **stats,
    })


# ── Write JSON ───────────────────────────────────────────────────────────────
output = {
    "n_files":          len(files),
    "latency_field":    args.latency_field,
    "exclude_empty":    args.exclude_empty,
    "tasks_filter":     args.tasks,
    "n_groups":         len(results),
    "groups":           results,
}

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(output, f, indent=2)


# ── Headline table ───────────────────────────────────────────────────────────
print("=" * 116)
print(f"  LATENCY TAIL ANALYSIS — field: {args.latency_field}")
print("=" * 116)

# Sort by tail_ratio descending so the worst-tail groups show first
results_sorted = sorted(results, key=lambda r: -r["tail_ratio"])

header = (f"{'model':30s} {'device':10s} {'task':22s} {'n':>5s} "
          f"{'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s} "
          f"{'mean':>7s} {'tail':>6s}")
print("\n" + header)
print("-" * len(header))
for r in results_sorted:
    task_label = r["task"] or "—"
    flag = ""
    if r["tail_ratio"] > 2.0:
        flag = " ⚠"
    print(f"{r['model']:30s} {r['device']:10s} {task_label:22s} "
          f"{r['n_samples']:>5d} "
          f"{r['p50']:>7.3f} {r['p95']:>7.3f} {r['p99']:>7.3f} {r['max']:>7.3f} "
          f"{r['mean']:>7.3f} {r['tail_ratio']:>6.2f}{flag}")

# Per-model summary across tasks/devices
print("\n" + "=" * 90)
print("  PER-MODEL TAIL SUMMARY (mean across all tasks/devices)")
print("=" * 90)
by_model = defaultdict(list)
for r in results:
    by_model[r["model"]].append(r["tail_ratio"])

print(f"\n{'model':30s} {'mean tail_ratio':>20s} {'max tail_ratio':>20s} {'groups':>8s}")
print("-" * 80)
for model, ratios in sorted(by_model.items(), key=lambda x: -np.mean(x[1])):
    print(f"{model:30s} {np.mean(ratios):>20.2f} {max(ratios):>20.2f} {len(ratios):>8d}")

print(f"\nFull results saved to: {args.out}")
print("=" * 116)
