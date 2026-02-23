#!/usr/bin/env python3
"""
Batch process all benchmark results and generate complete LaTeX table (v2 with dtype).

Usage:
    python batch_process_results_v2.py --output-dir /path/to/benchmark/outputs
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


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


def calculate_metrics(runs_data, summary_data, meta_data):
    """Calculate all metrics needed for the table."""
    
    model_key = meta_data.get('model_key', 'unknown')
    host = meta_data.get('host', 'unknown')
    
    # Map hostnames to device names
    device_mapping = {
        'thorwaggle': 'Thor',
        'thorwaggle1': 'Thor',
        'dellgb10': 'Dell GB10',
        'dgxwaggle': 'Dell GB10',
        'jetson': 'Orin',
        'orin': 'Orin',
    }
    device = device_mapping.get(host.lower(), host)
    
    # Extract dtype (check bfloat16 BEFORE float16 to avoid substring match)
    dtype_raw = meta_data.get('dtype', 'unknown')
    if 'bfloat16' in str(dtype_raw) or 'bf16' in str(dtype_raw):
        dtype = 'bf16'
    elif 'float16' in str(dtype_raw) or 'fp16' in str(dtype_raw):
        dtype = 'fp16'
    elif 'float32' in str(dtype_raw) or 'fp32' in str(dtype_raw):
        dtype = 'fp32'
    else:
        dtype = str(dtype_raw).split('.')[-1] if '.' in str(dtype_raw) else str(dtype_raw)
    
    duration_min = summary_data.get('total_elapsed_minutes', 0)
    latency_ms_mean = summary_data.get('latency_ms_mean', 0)
    throughput = summary_data.get('images_per_second_mean', 0)
    
    # Calculate total tokens
    total_prompt_tokens = 0
    total_response_tokens = 0
    
    for record in runs_data:
        if record.get('error') is None:
            prompt_tokens = record.get('prompt_tokens') or record.get('input_len', 0)
            response_tokens = record.get('response_tokens_est') or record.get('gen_len', 0)
            total_prompt_tokens += prompt_tokens
            total_response_tokens += response_tokens
    
    total_tokens = total_prompt_tokens + total_response_tokens
    
    # Calculate average RAM usage (convert MB to GB)
    ram_samples = []
    for record in runs_data:
        sys_after = record.get('sys_after', {})
        ram_used_mb = sys_after.get('ram_used_mb', 0)
        if ram_used_mb > 0:
            ram_samples.append(ram_used_mb / 1024.0)
    
    avg_ram_gb = sum(ram_samples) / len(ram_samples) if ram_samples else 0
    
    # Calculate average GPU utilization
    gpu_util_samples = []
    for record in runs_data:
        cuda_stats = record.get('cuda_stats', {})
        gpu_util = cuda_stats.get('gpu_utilization_percent')
        if gpu_util is not None:
            gpu_util_samples.append(gpu_util)
    
    avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples) if gpu_util_samples else None
    
    # Calculate average CPU utilization
    cpu_util_samples = []
    for record in runs_data:
        sys_after = record.get('sys_after', {})
        cpu_percent = sys_after.get('cpu_percent')
        if cpu_percent is not None and cpu_percent > 0:
            cpu_util_samples.append(cpu_percent)
    
    avg_cpu_util = sum(cpu_util_samples) / len(cpu_util_samples) if cpu_util_samples else None
    
    # Calculate average power and total energy
    power_samples = []
    total_energy_joules = 0
    
    for record in runs_data:
        power_stats = record.get('power_stats')
        if power_stats:
            power_avg = power_stats.get('power_watts_avg')
            if power_avg:
                power_samples.append(power_avg)
            
            energy = power_stats.get('energy_joules_est', 0)
            total_energy_joules += energy
    
    avg_power_watts = sum(power_samples) / len(power_samples) if power_samples else 0
    
    # Calculate derived metrics
    duration_seconds = duration_min * 60
    tokens_per_sec = total_tokens / duration_seconds if duration_seconds > 0 else 0
    tps_per_watt = tokens_per_sec / avg_power_watts if avg_power_watts > 0 else 0
    joules_per_token = total_energy_joules / total_tokens if total_tokens > 0 else 0
    
    return {
        'model': model_key,
        'device': device,
        'dtype': dtype,
        'total_tokens': int(total_tokens),
        'latency_ms': round(latency_ms_mean, 1),
        'throughput': round(throughput, 3),
        'gpu_util': round(avg_gpu_util, 1) if avg_gpu_util is not None else None,
        'cpu_util': round(avg_cpu_util, 1) if avg_cpu_util is not None else None,
        'ram_gb': round(avg_ram_gb, 1),
        'power_w': round(avg_power_watts, 1),
        'tokens_per_sec': round(tokens_per_sec, 1),
        'tps_per_watt': round(tps_per_watt, 2),
        'joules_per_token': round(joules_per_token, 2)
    }


def find_benchmark_runs(base_dir):
    """Find all benchmark run directories."""
    base_path = Path(base_dir)
    results = []
    
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            runs_file = run_dir / 'runs.jsonl'
            summary_file = run_dir / 'summary.json'
            meta_file = run_dir / 'run_meta.json'
            
            if all(f.exists() for f in [runs_file, summary_file, meta_file]):
                results.append({
                    'model': model_dir.name,
                    'run_group': run_dir.name,
                    'runs_file': runs_file,
                    'summary_file': summary_file,
                    'meta_file': meta_file
                })
    
    return results


def generate_latex_table(all_metrics):
    """Generate complete LaTeX table from all metrics with dtype."""
    
    # Organize by model, device, and dtype
    models = sorted(set(m['model'] for m in all_metrics))
    devices = ['Dell GB10', 'Thor', 'Orin']
    dtypes = ['bf16', 'fp16']
    
    # Create a lookup: metrics_lookup[(model, device, dtype)] = metrics
    metrics_lookup = {}
    for m in all_metrics:
        key = (m['model'], m['device'], m['dtype'])
        metrics_lookup[key] = m
    
    # Generate table
    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance metrics for compact VLMs across three edge systems. "
                 r"All models tested on 500 COCO validation images with 5 tasks per image (2500 total inferences). "
                 r"Each model tested with both fp16 and bf16 precision.}")
    lines.append(r"\label{tab:benchmark-results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}lcccccccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{System} & \textbf{Dtype} & "
                 r"\textbf{Latency} & \textbf{Throughput} & \textbf{GPU} & \textbf{CPU} & "
                 r"\textbf{RAM} & \textbf{Power} & \textbf{Tokens/} & \textbf{TPS/} & \textbf{J/} & \textbf{Total} \\")
    lines.append(r"& & & \textbf{(ms)} & \textbf{(img/s)} & \textbf{(\%)} & \textbf{(\%)} & "
                 r"\textbf{(GB)} & \textbf{(W)} & \textbf{sec} & \textbf{Watt} & \textbf{Token} & \textbf{Tokens} \\")
    lines.append(r"\midrule")
    
    # Add rows for each model/device/dtype combination
    for i, model in enumerate(models):
        for device in devices:
            for dtype in dtypes:
                key = (model, device, dtype)
                if key in metrics_lookup:
                    m = metrics_lookup[key]
                    gpu_str = str(m['gpu_util']) if m['gpu_util'] is not None else "---"
                    cpu_str = str(m['cpu_util']) if m['cpu_util'] is not None else "---"
                    
                    row = (f"{m['model']} & {m['device']} & {m['dtype']} & "
                          f"{m['latency_ms']} & {m['throughput']} & "
                          f"{gpu_str} & {cpu_str} & "
                          f"{m['ram_gb']} & {m['power_w']} & {m['tokens_per_sec']} & "
                          f"{m['tps_per_watt']} & {m['joules_per_token']} & {m['total_tokens']} \\\\")
                else:
                    row = f"{model} & {device} & {dtype} & --- & --- & --- & --- & --- & --- & --- & --- & --- & --- \\\\"
                lines.append(row)
        
        if i < len(models) - 1:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize")
    lines.append(r"\textbf{Metrics:} Latency = mean per-image inference time. Throughput = images per second. "
                 r"GPU\% = average GPU compute utilization (via pynvml on Dell GB10; not available on Thor/Orin). "
                 r"CPU\% = average CPU utilization. RAM = average system memory used. "
                 r"Power = average power draw (Thor/Orin use tegrastats VDD\_GPU, Dell uses pynvml). "
                 r"Tokens/sec = total tokens / duration. TPS/Watt = efficiency (Tokens/sec per Watt). "
                 r"J/Token = energy cost. --- = Not tested or not available.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Batch process benchmark results (v2 with dtype)')
    parser.add_argument('--output-dir', required=True, help='Base directory containing benchmark outputs')
    parser.add_argument('--output-file', default='complete_benchmark_table.tex', 
                       help='Output LaTeX file')
    args = parser.parse_args()
    
    print(f"Scanning {args.output_dir} for benchmark results...")
    runs = find_benchmark_runs(args.output_dir)
    print(f"Found {len(runs)} benchmark runs")
    
    if not runs:
        print("No benchmark results found!")
        return
    
    all_metrics = []
    for run_info in runs:
        print(f"Processing {run_info['model']} - {run_info['run_group']}...")
        
        try:
            runs_data = load_runs_jsonl(run_info['runs_file'])
            
            with open(run_info['summary_file'], 'r') as f:
                summary_data = json.load(f)
            
            with open(run_info['meta_file'], 'r') as f:
                meta_data = json.load(f)
            
            metrics = calculate_metrics(runs_data, summary_data, meta_data)
            all_metrics.append(metrics)
            
            print(f"  ✓ {metrics['model']} on {metrics['device']} ({metrics['dtype']}): "
                  f"{metrics['latency_ms']}ms, {metrics['throughput']} img/s")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if not all_metrics:
        print("No valid metrics calculated!")
        return
    
    print(f"\nGenerating LaTeX table...")
    latex_table = generate_latex_table(all_metrics)
    
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✓ LaTeX table saved to: {output_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for m in sorted(all_metrics, key=lambda x: (x['model'], x['device'], x['dtype'])):
        print(f"{m['model']:15s} on {m['device']:12s} ({m['dtype']}): "
              f"{m['latency_ms']:7.1f}ms, {m['throughput']:.3f} img/s, "
              f"{m['tps_per_watt']:.2f} TPS/W, {m['joules_per_token']:.2f} J/tok")


if __name__ == '__main__':
    main()