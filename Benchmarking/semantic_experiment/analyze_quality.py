#!/usr/bin/env python3
"""
analyze_quality.py  (v2)

Aggregates enriched runs.jsonl files across all (model, device) pairs and
writes merged CSVs ready for plotting.

Reads these fields from each output:
    embedding                         (mpnet vector, 768 floats)
    entity_recall                     (float 0-1, COCO string-match recall)
    caption_similarity_max            (float, cosine to nearest COCO reference caption)
    caption_similarity_mean           (float, mean cosine across the 5 COCO refs)
    count_accuracy                    (dict with count_exact_accuracy, count_mae, ...)
    gen_latency_s, gen_len, tokens_per_sec
    text, error

USAGE:
    python analyze_quality.py \\
        --enriched moondream:thor:/path/to/runs_enriched.jsonl \\
        --enriched smolvlm2:thor:/path/to/runs_enriched.jsonl \\
        --enriched moondream:dell:/path/to/runs_enriched.jsonl \\
        ... \\
        --output_dir analysis_output/all_devices

OUTPUTS (in --output_dir):
    all_models_devices_summary.csv   # 1 row per (model, device) — slide 14 scatter, slide 18 table
    per_task_recall.csv              # 1 row per (model, device, task) — slide 15 heatmap
    within_device_stability.csv      # 1 row per (model, device) — slide 16 numbers
    cross_device_pairs.csv           # 1 row per (model, image, task) cosine across devices — slide 17
    determinism_examples.json        # all (model, device) low-stability examples — slide 16
    talk_numbers.txt                 # human-readable summary, paste into speaker notes
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ─── Loading ────────────────────────────────────────────────────────────────

def load_enriched(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_outputs(records):
    for rec in records:
        img_id = rec.get("image_id")
        task = rec.get("task")
        for i, out in enumerate(rec.get("outputs", [])):
            yield img_id, task, i, out, rec


def get_embedding(out):
    """Embedding field can be `embedding` (current) or `text_embedding` (legacy)."""
    return out.get("embedding") or out.get("text_embedding")


# ─── Stats helpers ──────────────────────────────────────────────────────────

def mean_safe(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pct(arr, p):
    if not arr:
        return None
    return float(np.percentile(arr, p))


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def within_image_stability(valid_outputs):
    embs = [get_embedding(o) for o in valid_outputs]
    embs = [e for e in embs if e]
    if len(embs) < 2:
        return None
    sims = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sims.append(cosine_sim(embs[i], embs[j]))
    return sum(sims) / len(sims)


# ─── Per-(model, device) summary ───────────────────────────────────────────

def summarize_model_device(model, device, records):
    all_recall = []
    all_count_exact = []
    all_count_mae = []
    all_cap_max = []
    all_cap_mean = []
    all_clipscore = []
    all_latency_ms = []
    all_gen_len = []
    all_tps = []
    # Hardware telemetry — one number per record (not per output)
    all_power_avg = []
    all_power_peak = []
    all_energy_j = []
    all_gpu_mem_gb = []
    all_duration_s = []
    all_cpu_pct = []
    hw_source = None

    # First pass: pull hw_stats once per record
    seen_records = set()
    for img_id, task, i, out, rec in collect_outputs(records):
        rec_id = id(rec)
        if rec_id in seen_records:
            continue
        seen_records.add(rec_id)
        hw = rec.get("hw_stats") or {}
        if hw.get("power_watts_avg") is not None:
            all_power_avg.append(hw["power_watts_avg"])
        if hw.get("power_watts_peak") is not None:
            all_power_peak.append(hw["power_watts_peak"])
        if hw.get("energy_joules_est") is not None:
            all_energy_j.append(hw["energy_joules_est"])
        if hw.get("gpu_mem_alloc_gb_peak") is not None:
            all_gpu_mem_gb.append(hw["gpu_mem_alloc_gb_peak"])
        if hw.get("duration_seconds") is not None:
            all_duration_s.append(hw["duration_seconds"])
        if hw.get("cpu_percent_avg") is not None:
            all_cpu_pct.append(hw["cpu_percent_avg"])
        if hw_source is None and hw.get("source"):
            hw_source = hw["source"]

    # Second pass: per-output metrics
    for img_id, task, i, out, rec in collect_outputs(records):
        if out.get("error"):
            continue
        if out.get("entity_recall") is not None:
            all_recall.append(out["entity_recall"])
        if task == "objects_and_counts" and isinstance(out.get("count_accuracy"), dict):
            ca = out["count_accuracy"]
            if ca.get("count_exact_accuracy") is not None:
                all_count_exact.append(ca["count_exact_accuracy"])
            if ca.get("count_mae") is not None:
                all_count_mae.append(ca["count_mae"])
        if out.get("caption_similarity_max") is not None:
            all_cap_max.append(out["caption_similarity_max"])
        if out.get("caption_similarity_mean") is not None:
            all_cap_mean.append(out["caption_similarity_mean"])
        if out.get("clipscore_cos") is not None:
            all_clipscore.append(out["clipscore_cos"])
        if out.get("gen_latency_s") is not None:
            all_latency_ms.append(out["gen_latency_s"] * 1000)
        if out.get("gen_len") is not None:
            all_gen_len.append(out["gen_len"])
        tps = out.get("tokens_per_sec") or out.get("tokens_per_second")
        if tps is not None:
            all_tps.append(tps)
        # Energy/power telemetry lives at the RECORD level, not per-output.
        # We pick it up once per record below (in the wrapper).

    return {
        "model": model,
        "device": device,
        "n_records": len(records),
        "n_outputs": sum(1 for _ in collect_outputs(records)),
        "n_recall_valid": len(all_recall),
        "entity_recall_mean": mean_safe(all_recall),
        "count_exact_mean": mean_safe(all_count_exact),
        "count_mae_mean": mean_safe(all_count_mae),
        "caption_sim_max_mean": mean_safe(all_cap_max),
        "caption_sim_max_median": float(np.median(all_cap_max)) if all_cap_max else None,
        "caption_sim_mean_mean": mean_safe(all_cap_mean),
        "clipscore_cos_mean": mean_safe(all_clipscore),
        "latency_ms_mean": mean_safe(all_latency_ms),
        "latency_ms_p50": pct(all_latency_ms, 50),
        "latency_ms_p90": pct(all_latency_ms, 90),
        "latency_ms_p99": pct(all_latency_ms, 99),
        "gen_len_mean": mean_safe(all_gen_len),
        "tokens_per_sec_mean": mean_safe(all_tps),
        # Energy + memory
        "power_watts_avg_mean": mean_safe(all_power_avg),
        "power_watts_peak_mean": mean_safe(all_power_peak),
        "energy_joules_mean": mean_safe(all_energy_j),
        "energy_per_token_mj_mean": (
            1000 * mean_safe(all_energy_j) / mean_safe(all_gen_len)
            if mean_safe(all_energy_j) is not None and mean_safe(all_gen_len)
            else None
        ),
        "gpu_mem_gb_peak_mean": mean_safe(all_gpu_mem_gb),
        "duration_seconds_mean": mean_safe(all_duration_s),
        "cpu_percent_avg_mean": mean_safe(all_cpu_pct),
        "hw_source": hw_source,
    }


# ─── Per-task breakdown ─────────────────────────────────────────────────────

def per_task_breakdown(model, device, records):
    by_task = defaultdict(lambda: {"recall": [], "count_exact": [],
                                    "count_mae": [], "cap_max": [], "cap_mean": []})
    for img_id, task, i, out, rec in collect_outputs(records):
        if out.get("error"):
            continue
        if out.get("entity_recall") is not None:
            by_task[task]["recall"].append(out["entity_recall"])
        if task == "objects_and_counts" and isinstance(out.get("count_accuracy"), dict):
            ca = out["count_accuracy"]
            if ca.get("count_exact_accuracy") is not None:
                by_task[task]["count_exact"].append(ca["count_exact_accuracy"])
            if ca.get("count_mae") is not None:
                by_task[task]["count_mae"].append(ca["count_mae"])
        if out.get("caption_similarity_max") is not None:
            by_task[task]["cap_max"].append(out["caption_similarity_max"])
        if out.get("caption_similarity_mean") is not None:
            by_task[task]["cap_mean"].append(out["caption_similarity_mean"])

    rows = []
    for task, d in sorted(by_task.items()):
        rows.append({
            "model": model,
            "device": device,
            "task": task,
            "n": len(d["recall"]),
            "entity_recall_mean": mean_safe(d["recall"]),
            "count_exact_mean": mean_safe(d["count_exact"]),
            "count_mae_mean": mean_safe(d["count_mae"]),
            "caption_sim_max_mean": mean_safe(d["cap_max"]),
            "caption_sim_mean_mean": mean_safe(d["cap_mean"]),
        })
    return rows


# ─── Within-device stability ────────────────────────────────────────────────

def stability_breakdown(model, device, records):
    stabilities = []
    examples = []

    for rec in records:
        img_id = rec.get("image_id")
        task = rec.get("task")
        outputs = rec.get("outputs", [])
        if not outputs:
            continue
        valid = [o for o in outputs
                 if not o.get("error") and get_embedding(o)]
        if len(valid) < 2:
            continue
        s = within_image_stability(valid)
        if s is None:
            continue
        stabilities.append((img_id, task, s, valid))

        if s < 0.85:
            examples.append({
                "image_id": img_id,
                "task": task,
                "stability": s,
                "texts": [o.get("text", "") for o in valid],
            })

    if not stabilities:
        return None, []

    s_vals = [s for _, _, s, _ in stabilities]
    s_arr = np.array(s_vals)

    examples.sort(key=lambda e: e["stability"])
    examples = examples[:10]

    summary = {
        "model": model,
        "device": device,
        "n_image_task_pairs": len(s_vals),
        "mean_stability": float(np.mean(s_arr)),
        "median_stability": float(np.median(s_arr)),
        "min_stability": float(np.min(s_arr)),
        "p5_stability": float(np.percentile(s_arr, 5)),
        "p10_stability": float(np.percentile(s_arr, 10)),
        "frac_below_0_85": float(np.mean(s_arr < 0.85)),
        "frac_below_0_90": float(np.mean(s_arr < 0.90)),
        "frac_below_0_95": float(np.mean(s_arr < 0.95)),
    }
    return summary, examples


# ─── Cross-device agreement ────────────────────────────────────────────────

def cross_device_agreement(model, by_device_records):
    devices = sorted(by_device_records.keys())
    if len(devices) < 2:
        return []

    lookup = defaultdict(dict)
    for device, records in by_device_records.items():
        for rec in records:
            img_id = rec.get("image_id")
            task = rec.get("task")
            outputs = rec.get("outputs", [])
            if not outputs:
                continue
            o = outputs[0]  # repeat 0 as representative
            emb = get_embedding(o)
            if o.get("error") or not emb:
                continue
            lookup[(img_id, task)][device] = (emb, o.get("text", ""))

    rows = []
    for (img_id, task), per_dev in lookup.items():
        present = sorted(per_dev.keys())
        if len(present) < 2:
            continue
        sims = []
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                e_i = per_dev[present[i]][0]
                e_j = per_dev[present[j]][0]
                sims.append(cosine_sim(e_i, e_j))
        mean_sim = sum(sims) / len(sims)

        row = {
            "model": model,
            "image_id": img_id,
            "task": task,
            "devices_present": "|".join(present),
            "n_pairs": len(sims),
            "mean_cross_device_sim": mean_sim,
        }
        for d in present:
            row[f"text_{d}"] = per_dev[d][1][:200]
        rows.append(row)
    return rows


# ─── Output writers ────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path} ({len(rows)} rows)")


def write_text_summary(path, summary_rows, stability_rows, examples):
    lines = []
    lines.append("=" * 78)
    lines.append("PAISE 2026 TALK — Quality Metrics Summary")
    lines.append("=" * 78)
    lines.append("")

    lines.append("HEADLINE — quality + latency by (model, device):")
    lines.append("")
    lines.append(f"  {'model':14s} {'device':6s} {'recall':>7s} {'cap_sim':>8s} "
                 f"{'clip':>7s} {'cnt_ex':>7s} {'lat_ms':>8s} {'p99_ms':>8s} {'tok/s':>7s}")
    lines.append("  " + "-" * 80)
    for r in sorted(summary_rows, key=lambda x: (x["device"], x["model"])):
        recall = f"{r['entity_recall_mean']:.3f}" if r.get("entity_recall_mean") is not None else "—"
        cap = f"{r['caption_sim_max_mean']:.3f}" if r.get("caption_sim_max_mean") is not None else "—"
        clip = f"{r['clipscore_cos_mean']:.3f}" if r.get("clipscore_cos_mean") is not None else "—"
        cex = f"{r['count_exact_mean']:.3f}" if r.get("count_exact_mean") is not None else "—"
        lat = f"{r['latency_ms_mean']:.0f}" if r.get("latency_ms_mean") is not None else "—"
        p99 = f"{r['latency_ms_p99']:.0f}" if r.get("latency_ms_p99") is not None else "—"
        tps = f"{r['tokens_per_sec_mean']:.1f}" if r.get("tokens_per_sec_mean") is not None else "—"
        lines.append(f"  {r['model']:14s} {r['device']:6s} {recall:>7s} {cap:>8s} "
                     f"{clip:>7s} {cex:>7s} {lat:>8s} {p99:>8s} {tps:>7s}")
    lines.append("")

    # Energy + memory section
    if any(r.get("energy_joules_mean") is not None for r in summary_rows):
        lines.append("ENERGY + MEMORY — per generation:")
        lines.append("")
        lines.append(f"  {'model':14s} {'device':6s} {'pwr_W':>6s} {'energy_J':>9s} "
                     f"{'mJ/tok':>7s} {'mem_GB':>7s} {'source':>14s}")
        lines.append("  " + "-" * 70)
        for r in sorted(summary_rows, key=lambda x: (x["device"], x["model"])):
            pwr = f"{r['power_watts_avg_mean']:.1f}" if r.get("power_watts_avg_mean") is not None else "—"
            ene = f"{r['energy_joules_mean']:.1f}" if r.get("energy_joules_mean") is not None else "—"
            mjt = f"{r['energy_per_token_mj_mean']:.1f}" if r.get("energy_per_token_mj_mean") is not None else "—"
            mem = f"{r['gpu_mem_gb_peak_mean']:.2f}" if r.get("gpu_mem_gb_peak_mean") is not None else "—"
            src = r.get("hw_source") or "—"
            lines.append(f"  {r['model']:14s} {r['device']:6s} {pwr:>6s} {ene:>9s} "
                         f"{mjt:>7s} {mem:>7s} {src:>14s}")
        lines.append("")

    if stability_rows:
        lines.append("STABILITY — within-device, 5-repeat semantic agreement:")
        lines.append("")
        lines.append(f"  {'model':14s} {'device':6s} {'mean':>6s} {'min':>6s} {'<0.85':>7s} {'<0.95':>7s}")
        lines.append("  " + "-" * 56)
        for s in sorted(stability_rows, key=lambda x: (x["device"], x["model"])):
            lines.append(f"  {s['model']:14s} {s['device']:6s} "
                         f"{s['mean_stability']:>6.3f} {s['min_stability']:>6.3f} "
                         f"{s['frac_below_0_85']:>7.1%} {s['frac_below_0_95']:>7.1%}")
        lines.append("")

    if examples:
        lines.append("DETERMINISM EXAMPLES — lowest-stability inputs (top 5):")
        lines.append("")
        for ex in examples[:5]:
            lines.append(f"  model={ex.get('model','?')} device={ex.get('device','?')} "
                         f"image={ex['image_id']} task={ex['task']} stability={ex['stability']:.3f}")
            for i, t in enumerate(ex["texts"]):
                lines.append(f"    Run {i}: {t[:90]}")
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[write] {path}")
    print("")
    print("\n".join(lines))


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", action="append", required=True,
                    help="MODEL:DEVICE:PATH (repeat for each run)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs_by_model_device = {}
    for spec in args.enriched:
        try:
            model, device, path = spec.split(":", 2)
        except ValueError:
            print(f"ERROR: --enriched must be MODEL:DEVICE:PATH, got: {spec}")
            sys.exit(1)
        path = path.strip()
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        records = list(load_enriched(path))
        print(f"[load] {model}:{device} → {len(records)} records from {path}")
        runs_by_model_device[(model, device)] = records

    summary_rows = []
    per_task_rows = []
    stability_rows = []
    all_examples = []

    for (model, device), records in sorted(runs_by_model_device.items()):
        print(f"[analyze] {model} on {device}")
        summary_rows.append(summarize_model_device(model, device, records))
        per_task_rows.extend(per_task_breakdown(model, device, records))
        stab, exs = stability_breakdown(model, device, records)
        if stab is not None:
            stability_rows.append(stab)
            for ex in exs:
                ex["model"] = model
                ex["device"] = device
            all_examples.extend(exs)

    cross_device_rows_all = []
    models = sorted(set(m for (m, d) in runs_by_model_device))
    for model in models:
        by_dev = {d: r for (m, d), r in runs_by_model_device.items() if m == model}
        if len(by_dev) < 2:
            continue
        rows = cross_device_agreement(model, by_dev)
        print(f"[cross-device] {model}: {len(rows)} matched (image, task) pairs across {len(by_dev)} devices")
        cross_device_rows_all.extend(rows)

    all_examples.sort(key=lambda e: e["stability"])

    write_csv(out / "all_models_devices_summary.csv", summary_rows)
    write_csv(out / "per_task_recall.csv", per_task_rows)
    write_csv(out / "within_device_stability.csv", stability_rows)
    write_csv(out / "cross_device_pairs.csv", cross_device_rows_all)

    with open(out / "determinism_examples.json", "w") as f:
        json.dump(all_examples, f, indent=2)
    print(f"[write] {out / 'determinism_examples.json'} ({len(all_examples)} examples)")

    write_text_summary(out / "talk_numbers.txt",
                       summary_rows, stability_rows, all_examples)


if __name__ == "__main__":
    main()