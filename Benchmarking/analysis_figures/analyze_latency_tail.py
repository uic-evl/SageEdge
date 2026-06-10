"""
analyze_latency_tail.py

Computes per-(model, device, task) latency tail statistics from experiment JSONL files.
"""

import argparse
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np


def expand_globs(patterns):
    files = []
    for pattern in patterns:
        matches = sorted(glob(pattern, recursive=True))
        if not matches:
            print(f"  WARN: no files match: {pattern}")
        files.extend(matches)

    deduped = []
    seen = set()
    for path in files:
        if path in seen:
            continue
        deduped.append(path)
        seen.add(path)
    return deduped


def get_latency(obj, field):
    val = obj.get(field)
    if val is None and field != "gen_latency_s":
        val = obj.get("gen_latency_s")
    if val is None and field != "total_latency_s":
        val = obj.get("total_latency_s")
    if val is None and field != "latency_s":
        val = obj.get("latency_s")
    return val


def tail_stats(values):
    arr = np.asarray(values, dtype=float)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return {
        "n_samples": int(arr.size),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "tail_ratio": float(p99 / p50) if p50 else None,
        "gt_5s": int(np.sum(arr > 5)),
        "gt_6s": int(np.sum(arr > 6)),
        "gt_7s": int(np.sum(arr > 7)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, action="append", help="Glob(s) of JSONL files. Pass multiple times to merge.")
    parser.add_argument("--out", default="outputs/latency_tail.json", help="Output JSON path")
    parser.add_argument("--tasks", nargs="+", default=None, help="Restrict to specific task names")
    parser.add_argument(
        "--latency-field",
        default="gen_latency_s",
        choices=["gen_latency_s", "total_latency_s", "latency_s"],
        help="Which latency field to analyze (default: gen_latency_s - generation only)",
    )
    parser.add_argument("--exclude-empty", action="store_true", default=True, help="Exclude runs with empty text (default: True)")
    args = parser.parse_args()

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
        raise SystemExit(1)

    groups = defaultdict(list)
    n_total_runs = 0
    n_excluded_empty = 0
    n_excluded_no_lat = 0

    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                model = record.get("model") or record.get("model_name") or record.get("model_tag") or "unknown"
                device = record.get("device") or record.get("device_name") or "unknown"
                task = record.get("task") or "caption"
                if args.tasks and task not in args.tasks:
                    continue
                for out in record.get("outputs", []):
                    n_total_runs += 1
                    text = out.get("text", "")
                    if args.exclude_empty and not text:
                        n_excluded_empty += 1
                        continue
                    latency = get_latency(out, args.latency_field)
                    if latency is None:
                        n_excluded_no_lat += 1
                        continue
                    groups[(model, device, task)].append(float(latency))

    print(f"\nLoaded {n_total_runs} run-records total")
    print(f"  Excluded (empty text):   {n_excluded_empty}")
    print(f"  Excluded (no latency):   {n_excluded_no_lat}")
    print(f"  Groups (model,device,task): {len(groups)}\n")
    if not groups:
        print("ERROR: no usable latency data.")
        raise SystemExit(1)

    results = []
    for (model, device, task), values in sorted(groups.items()):
        row = {"model": model, "device": device, "task": task, **tail_stats(values)}
        results.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 116)
    print(f"  LATENCY TAIL ANALYSIS - field: {args.latency_field}")
    print("=" * 116)
    print(f"{'model':30s} {'device':10s} {'task':22s} {'n':>5s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s} {'mean':>7s} {'tail':>6s}")
    for r in results:
        warning = " !" if r["tail_ratio"] and r["tail_ratio"] > 2 else ""
        print(
            f"{r['model'][:30]:30s} {r['device'][:10]:10s} {r['task'][:22]:22s} "
            f"{r['n_samples']:5d} {r['p50']:7.3f} {r['p95']:7.3f} {r['p99']:7.3f} "
            f"{r['max']:7.3f} {r['mean']:7.3f} {r['tail_ratio']:6.2f}{warning}"
        )
    print(f"\nFull results saved to: {args.out}")


if __name__ == "__main__":
    main()
