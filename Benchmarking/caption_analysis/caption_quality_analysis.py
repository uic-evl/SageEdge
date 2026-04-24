#!/usr/bin/env python3
"""
Analyze caption quality and speed characteristics for vision models.

Generates plots for:
- Speed: latency distributions, percentiles, consistency
- Quality: caption lengths, task performance, error rates
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# Set style
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200


def read_json(p: Path):
    """Read JSON file."""
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_runs_jsonl(filepath):
    """Load all records from runs.jsonl file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def infer_device_label(run_dir: Path):
    """Infer device name from directory path."""
    device_mapping = {
        'dell_output': 'Dell',
        'thor_output': 'Thor',
        'orin_output': 'Orin',
    }
    for part in run_dir.parts:
        if part in device_mapping:
            return device_mapping[part]
    return "unknown"


def infer_dtype_short(dtype: str):
    """Extract short dtype name."""
    if not dtype:
        return "na"
    d = str(dtype).lower()
    if "bfloat16" in d or "bf16" in d:
        return "bf16"
    if "float16" in d or "fp16" in d:
        return "fp16"
    if "float32" in d or "fp32" in d:
        return "fp32"
    return d


def extract_model_name(model_key: str):
    """Extract clean model name."""
    name = model_key.replace('_', ' ').title()
    name = name.replace('Vlm', 'VLM')
    name = name.replace('Vl', 'VL')
    name = name.replace('2b', '2B')
    name = name.replace('3n', '3N')
    return name


def normalize_model_name(model_name: str):
    """Normalize model names across devices (Internvl and Internvl2 2b → Internvl 2b)."""
    if model_name in ['Internvl', 'Internvl2 2b']:
        return 'Internvl 2b'
    return model_name


def find_run_dirs(roots):
    """Find all benchmark run directories."""
    run_dirs = []
    for r in roots:
        root = Path(r)
        for meta in root.rglob("run_meta.json"):
            run_dir = meta.parent
            if (run_dir / "summary.json").exists() and (run_dir / "runs.jsonl").exists():
                run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def analyze_run(run_dir: Path):
    """Analyze a single run for speed and quality metrics."""
    meta = read_json(run_dir / "run_meta.json")
    summ = read_json(run_dir / "summary.json")
    
    # Skip failed runs
    num_errors = summ.get('num_errors', 0)
    total_runs = summ.get('total_runs', 0)
    if num_errors == total_runs and total_runs > 0:
        return None
    
    runs = load_runs_jsonl(run_dir / "runs.jsonl")
    
    model_key = meta.get('model_key', run_dir.parts[-2])
    model_name = normalize_model_name(extract_model_name(model_key))
    device = infer_device_label(run_dir)
    dtype = infer_dtype_short(meta.get('dtype', ''))
    
    # Speed metrics
    latencies = []
    latencies_by_task = defaultdict(list)
    
    # Quality metrics
    caption_lengths = []
    caption_lengths_by_task = defaultdict(list)
    task_counts = defaultdict(int)
    error_counts = defaultdict(int)
    
    for record in runs:
        task = record.get('task', 'unknown')
        latency = record.get('latency_ms', 0)
        
        if record.get('error') is None:
            # Successful run
            latencies.append(latency)
            latencies_by_task[task].append(latency)
            
            output_text = record.get('output_text', '')
            caption_len = len(output_text)
            caption_lengths.append(caption_len)
            caption_lengths_by_task[task].append(caption_len)
            task_counts[task] += 1
        else:
            error_counts[task] += 1
    
    if not latencies:
        return None
    
    return {
        'run_dir': str(run_dir),
        'model_name': model_name,
        'model_key': model_key,
        'device': device,
        'dtype': dtype,
        
        # Speed metrics
        'latency_mean': np.mean(latencies),
        'latency_median': np.median(latencies),
        'latency_std': np.std(latencies),
        'latency_p90': np.percentile(latencies, 90),
        'latency_p95': np.percentile(latencies, 95),
        'latency_p99': np.percentile(latencies, 99),
        'latency_min': np.min(latencies),
        'latency_max': np.max(latencies),
        'latencies': latencies,
        'latencies_by_task': dict(latencies_by_task),
        
        # Quality metrics
        'caption_length_mean': np.mean(caption_lengths) if caption_lengths else 0,
        'caption_length_median': np.median(caption_lengths) if caption_lengths else 0,
        'caption_length_std': np.std(caption_lengths) if caption_lengths else 0,
        'caption_lengths': caption_lengths,
        'caption_lengths_by_task': dict(caption_lengths_by_task),
        
        # Task performance
        'task_counts': dict(task_counts),
        'error_counts': dict(error_counts),
        'total_tasks': sum(task_counts.values()),
        'total_errors': sum(error_counts.values()),
        'success_rate': sum(task_counts.values()) / (sum(task_counts.values()) + sum(error_counts.values())) if (sum(task_counts.values()) + sum(error_counts.values())) > 0 else 0,
    }


def style_axes(ax):
    """Apply consistent styling to plot axes."""
    ax.grid(True, axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_latency_distributions(analyses, output_dir):
    """Plot latency distributions for each model/device/dtype combination."""
    print("Generating latency distribution plots...")
    
    # Box plot comparing latencies
    fig, ax = plt.subplots(figsize=(16, 8))
    
    data_for_plot = []
    labels = []
    
    for analysis in sorted(analyses, key=lambda x: (x['model_name'], x['device'], x['dtype'])):
        data_for_plot.append(analysis['latencies'])
        labels.append(f"{analysis['model_name']}\n{analysis['device']} {analysis['dtype']}")
    
    bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True, showfliers=False)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distributions Across Models and Devices', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_latency_distributions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved speed_latency_distributions.png")


def plot_latency_percentiles(analyses, output_dir):
    """Plot latency percentiles comparison."""
    print("Generating latency percentiles plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data
    models = sorted(set(a['model_name'] for a in analyses))
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(set(a['device'] for a in analyses)):
        for dtype in sorted(set(a['dtype'] for a in analyses)):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        p90_values = []
        p95_values = []
        p99_values = []
        
        for model in models:
            matching = [a for a in analyses if a['model_name'] == model and a['device'] == device and a['dtype'] == dtype]
            if matching:
                a = matching[0]
                p90_values.append(a['latency_p90'])
            else:
                p90_values.append(0)
        
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, p90_values, width, label=f'{device} {dtype} (P90)', alpha=0.8)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('P90 Latency Comparison Across Models and Devices', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=7, ncol=3)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_latency_p90.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved speed_latency_p90.png")


def plot_latency_consistency(analyses, output_dir):
    """Plot latency consistency (std dev / mean)."""
    print("Generating latency consistency plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate coefficient of variation (CV = std/mean)
    for analysis in analyses:
        analysis['latency_cv'] = (analysis['latency_std'] / analysis['latency_mean']) * 100 if analysis['latency_mean'] > 0 else 0
    
    models = sorted(set(a['model_name'] for a in analyses))
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(set(a['device'] for a in analyses)):
        for dtype in sorted(set(a['dtype'] for a in analyses)):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        cv_values = []
        
        for model in models:
            matching = [a for a in analyses if a['model_name'] == model and a['device'] == device and a['dtype'] == dtype]
            if matching:
                cv_values.append(matching[0]['latency_cv'])
            else:
                cv_values.append(0)
        
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, cv_values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Latency Consistency (Lower is More Consistent)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_latency_consistency.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved speed_latency_consistency.png")


def plot_caption_lengths(analyses, output_dir):
    """Plot caption length statistics."""
    print("Generating caption length plots...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = sorted(set(a['model_name'] for a in analyses))
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(set(a['device'] for a in analyses)):
        for dtype in sorted(set(a['dtype'] for a in analyses)):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        length_values = []
        
        for model in models:
            matching = [a for a in analyses if a['model_name'] == model and a['device'] == device and a['dtype'] == dtype]
            if matching:
                length_values.append(matching[0]['caption_length_mean'])
            else:
                length_values.append(0)
        
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, length_values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Average Caption Length (characters)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Average Caption Length by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_caption_lengths.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved quality_caption_lengths.png")


def plot_task_performance(analyses, output_dir):
    """Plot performance by task type."""
    print("Generating task performance plots...")
    
    # Aggregate task latencies across all runs
    task_data = defaultdict(list)
    
    for analysis in analyses:
        for task, latencies in analysis['latencies_by_task'].items():
            task_data[task].extend(latencies)
    
    if not task_data:
        print("  ⚠ No task data available")
        return
    
    # Box plot by task
    fig, ax = plt.subplots(figsize=(12, 7))
    
    tasks = sorted(task_data.keys())
    data_for_plot = [task_data[task] for task in tasks]
    
    bp = ax.boxplot(data_for_plot, labels=tasks, patch_artist=True, showfliers=False)
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(tasks)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Task Type', fontsize=12, fontweight='bold')
    ax.set_title('Latency by Task Type (All Models)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_by_task_type.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved speed_by_task_type.png")
    
    # Caption lengths by task
    caption_task_data = defaultdict(list)
    for analysis in analyses:
        for task, lengths in analysis['caption_lengths_by_task'].items():
            caption_task_data[task].extend(lengths)
    
    if caption_task_data:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        tasks = sorted(caption_task_data.keys())
        data_for_plot = [caption_task_data[task] for task in tasks]
        
        bp = ax.boxplot(data_for_plot, labels=tasks, patch_artist=True, showfliers=False)
        
        colors = plt.cm.Pastel2(np.linspace(0, 1, len(tasks)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Caption Length (characters)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Task Type', fontsize=12, fontweight='bold')
        ax.set_title('Caption Length by Task Type (All Models)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        style_axes(ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_caption_length_by_task.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved quality_caption_length_by_task.png")


def plot_success_rates(analyses, output_dir):
    """Plot success rates for each configuration."""
    print("Generating success rate plots...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = sorted(set(a['model_name'] for a in analyses))
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(set(a['device'] for a in analyses)):
        for dtype in sorted(set(a['dtype'] for a in analyses)):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        success_values = []
        
        for model in models:
            matching = [a for a in analyses if a['model_name'] == model and a['device'] == device and a['dtype'] == dtype]
            if matching:
                success_values.append(matching[0]['success_rate'] * 100)
            else:
                success_values.append(0)
        
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, success_values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Task Success Rate by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_success_rates.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved quality_success_rates.png")


def plot_speed_vs_quality_scatter(analyses, output_dir):
    """Plot speed vs quality tradeoff."""
    print("Generating speed vs quality scatter plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use different markers for different devices
    device_markers = {'Dell': 'o', 'Thor': 's', 'Orin': '^'}
    dtype_colors = {'bf16': 'blue', 'fp16': 'red'}
    
    for analysis in analyses:
        device = analysis['device']
        dtype = analysis['dtype']
        
        x_val = analysis['latency_mean']  # Speed (lower is faster)
        y_val = analysis['caption_length_mean']  # Quality proxy (longer might be more detailed)
        
        marker = device_markers.get(device, 'o')
        color = dtype_colors.get(dtype, 'gray')
        
        ax.scatter(x_val, y_val, marker=marker, c=color, s=100, alpha=0.6, 
                  edgecolors='black', linewidth=0.5,
                  label=f'{analysis["model_name"]} ({device} {dtype})')
        
        # Add model name as annotation
        ax.annotate(analysis['model_name'], (x_val, y_val), 
                   fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Mean Latency (ms) - Lower is Faster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Caption Length (chars) - Proxy for Detail', fontsize=12, fontweight='bold')
    ax.set_title('Speed vs Caption Detail Tradeoff', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for device, marker in device_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', 
                                     markersize=8, label=device))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                 markersize=8, label='bf16'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                 markersize=8, label='fp16'))
    
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_vs_quality_tradeoff.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved speed_vs_quality_tradeoff.png")


def generate_analysis_report(analyses, output_dir):
    """Generate detailed analysis report."""
    print("Generating analysis report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CAPTION QUALITY & SPEED ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Speed Analysis
    report_lines.append("-" * 80)
    report_lines.append("SPEED ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Fastest models
    fastest = sorted(analyses, key=lambda x: x['latency_mean'])[:5]
    report_lines.append("Top 5 Fastest (Lowest Mean Latency):")
    for i, a in enumerate(fastest, 1):
        report_lines.append(f"  {i}. {a['model_name']:20s} on {a['device']:10s} ({a['dtype']}): "
                          f"{a['latency_mean']:7.1f}ms mean, {a['latency_p90']:7.1f}ms P90")
    
    report_lines.append("")
    
    # Most consistent
    most_consistent = sorted(analyses, key=lambda x: x['latency_cv'])[:5]
    report_lines.append("Top 5 Most Consistent (Lowest CV):")
    for i, a in enumerate(most_consistent, 1):
        report_lines.append(f"  {i}. {a['model_name']:20s} on {a['device']:10s} ({a['dtype']}): "
                          f"CV={a['latency_cv']:5.1f}%")
    
    report_lines.append("")
    
    # Quality Analysis
    report_lines.append("-" * 80)
    report_lines.append("CAPTION QUALITY ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Most detailed captions
    most_detailed = sorted(analyses, key=lambda x: x['caption_length_mean'], reverse=True)[:5]
    report_lines.append("Top 5 Most Detailed Captions (Longest):")
    for i, a in enumerate(most_detailed, 1):
        report_lines.append(f"  {i}. {a['model_name']:20s} on {a['device']:10s} ({a['dtype']}): "
                          f"{a['caption_length_mean']:6.1f} chars avg")
    
    report_lines.append("")
    
    # Highest success rates
    most_reliable = sorted(analyses, key=lambda x: x['success_rate'], reverse=True)[:5]
    report_lines.append("Top 5 Most Reliable (Highest Success Rate):")
    for i, a in enumerate(most_reliable, 1):
        report_lines.append(f"  {i}. {a['model_name']:20s} on {a['device']:10s} ({a['dtype']}): "
                          f"{a['success_rate']*100:5.1f}% success ({a['total_tasks']} successful, {a['total_errors']} errors)")
    
    report_lines.append("")
    
    # Best speed/quality tradeoff
    report_lines.append("-" * 80)
    report_lines.append("SPEED vs QUALITY TRADEOFF")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Normalize and calculate combined score
    max_latency = max(a['latency_mean'] for a in analyses)
    max_length = max(a['caption_length_mean'] for a in analyses)
    
    for a in analyses:
        # Lower latency is better, higher length is better (as quality proxy)
        speed_score = 1 - (a['latency_mean'] / max_latency)
        quality_score = a['caption_length_mean'] / max_length
        a['combined_score'] = (speed_score + quality_score) / 2
    
    best_tradeoff = sorted(analyses, key=lambda x: x['combined_score'], reverse=True)[:5]
    report_lines.append("Top 5 Best Speed/Quality Balance:")
    for i, a in enumerate(best_tradeoff, 1):
        report_lines.append(f"  {i}. {a['model_name']:20s} on {a['device']:10s} ({a['dtype']}): "
                          f"{a['latency_mean']:6.1f}ms, {a['caption_length_mean']:6.1f} chars, "
                          f"score={a['combined_score']:.3f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    report_file = output_dir / 'caption_quality_speed_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Saved analysis report to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze caption quality and speed')
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Top-level output directories')
    parser.add_argument('--out', required=True,
                       help='Output directory for plots and data')
    args = parser.parse_args()
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CAPTION QUALITY & SPEED ANALYSIS")
    print("=" * 80)
    print(f"\nScanning directories: {', '.join(args.roots)}")
    
    run_dirs = find_run_dirs(args.roots)
    print(f"Found {len(run_dirs)} benchmark runs\n")
    
    if not run_dirs:
        print("ERROR: No benchmark results found!")
        return
    
    print("Analyzing runs...")
    analyses = []
    for run_dir in run_dirs:
        try:
            analysis = analyze_run(run_dir)
            if analysis:
                analyses.append(analysis)
                print(f"  ✓ {analysis['model_name']:20s} on {analysis['device']:10s} ({analysis['dtype']})")
            else:
                print(f"  ⊘ Skipped {run_dir.parts[-2]:15s} (failed run)")
        except Exception as e:
            print(f"  ✗ Error processing {run_dir}: {e}")
    
    if not analyses:
        print("\nERROR: No valid analyses!")
        return
    
    print(f"\n✓ Analyzed {len(analyses)} runs successfully\n")
    
    # Generate all plots
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    print()
    
    plot_latency_distributions(analyses, output_dir)
    plot_latency_percentiles(analyses, output_dir)
    plot_latency_consistency(analyses, output_dir)
    plot_caption_lengths(analyses, output_dir)
    plot_task_performance(analyses, output_dir)
    plot_success_rates(analyses, output_dir)
    plot_speed_vs_quality_scatter(analyses, output_dir)
    
    print()
    print("=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    print()
    
    generate_analysis_report(analyses, output_dir)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print(f"  - Plots: {len(list(output_dir.glob('*.png')))} PNG files")
    print(f"  - Report: caption_quality_speed_report.txt")


if __name__ == '__main__':
    main()
