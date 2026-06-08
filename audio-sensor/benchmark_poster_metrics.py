#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)

TIMING_RE = re.compile(r"whisper_print_timings:\s+total time\s+=\s+([0-9.]+)\s+ms")
TRANSCRIPT_TS_RE = re.compile(r"^\[[0-9:.,\s\-]+-->\s*[0-9:.,\s\-]+\]\s*(.*)$")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GR3D_RE = re.compile(r"GR3D_FREQ\s+([0-9.]+)%")
CPU_BLOCK_RE = re.compile(r"CPU\s+\[([^\]]+)\]")
CPU_CORE_RE = re.compile(r"([0-9.]+)%@")


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        curr = [i]
        for j, token_b in enumerate(b, 1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (token_a != token_b)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def accuracy_percent(reference, hypothesis):
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()
    if not ref_words:
        return 0.0
    wer = levenshtein(ref_words, hyp_words) / len(ref_words)
    return max(0.0, 100.0 * (1.0 - wer))


class TegraStats:
    def __init__(self, interval_ms):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.cpu_samples = []
        self.gpu_samples = []

    def start(self):
        if not shutil_which("tegrastats"):
            return
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.proc:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def _read(self):
        for line in self.proc.stdout:
            gpu = parse_gpu(line)
            cpu = parse_cpu(line)
            if gpu is not None:
                self.gpu_samples.append(gpu)
            if cpu is not None:
                self.cpu_samples.append(cpu)

    @property
    def avg_gpu(self):
        return average(self.gpu_samples)

    @property
    def avg_cpu(self):
        return average(self.cpu_samples)


def shutil_which(name):
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def average(values):
    if not values:
        return None
    return sum(values) / len(values)


def parse_gpu(line):
    match = GR3D_RE.search(line)
    if not match:
        return None
    return float(match.group(1))


def parse_cpu(line):
    match = CPU_BLOCK_RE.search(line)
    if not match:
        return None
    values = [float(item) for item in CPU_CORE_RE.findall(match.group(1))]
    if not values:
        return None
    return sum(values) / len(values)


def extract_transcript(output):
    lines = []
    for raw in output.splitlines():
        line = ANSI_RE.sub("", raw).strip()
        if not line:
            continue
        if line.startswith("whisper_") or line.startswith("system_info:") or line.startswith("main:"):
            continue
        if "whisper_print_timings:" in line:
            continue
        match = TRANSCRIPT_TS_RE.match(line)
        if match:
            lines.append(match.group(1).strip())
        elif not line.startswith("[") and not line.endswith("]"):
            lines.append(line)
    return " ".join(lines).strip()


def parse_total_time(output):
    matches = TIMING_RE.findall(output)
    if not matches:
        return None
    return float(matches[-1])


def run_once(binary, model, audio, threads, monitor_interval_ms, no_gpu):
    cmd = [
        str(binary),
        "-m",
        str(model),
        "-f",
        str(audio),
        "-t",
        str(threads),
        "-l",
        "en",
    ]
    if no_gpu:
        cmd.append("-ng")

    monitor = TegraStats(monitor_interval_ms)
    monitor.start()
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - start) * 1000.0
    monitor.stop()

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"{model.name} failed with code {proc.returncode}\n{output}")

    return {
        "wall_time_ms": wall_ms,
        "whisper_time_ms": parse_total_time(output),
        "transcript": extract_transcript(output),
        "avg_gpu_percent": monitor.avg_gpu,
        "avg_cpu_percent": monitor.avg_cpu,
    }


def summarize_model(name, model, args, reference):
    runs = []
    for idx in range(args.runs):
        print(f"Running {name} pass {idx + 1}/{args.runs}...")
        result = run_once(
            args.binary,
            model,
            args.audio,
            args.threads,
            args.monitor_interval_ms,
            args.no_gpu,
        )
        result["accuracy_percent"] = accuracy_percent(reference, result["transcript"])
        runs.append(result)

    return {
        "model": name,
        "accuracy_percent": average([item["accuracy_percent"] for item in runs]),
        "avg_whisper_time_ms": average([item["whisper_time_ms"] for item in runs if item["whisper_time_ms"] is not None]),
        "avg_wall_time_ms": average([item["wall_time_ms"] for item in runs]),
        "avg_gpu_percent": average([item["avg_gpu_percent"] for item in runs if item["avg_gpu_percent"] is not None]),
        "avg_cpu_percent": average([item["avg_cpu_percent"] for item in runs if item["avg_cpu_percent"] is not None]),
        "runs": runs,
    }


def fmt(value, suffix="", decimals=1):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def write_outputs(results, out_prefix):
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    json_path = out_prefix.with_suffix(".json")

    rows = []
    for result in results:
        rows.append({
            "Model": result["model"],
            "Accuracy (%)": fmt(result["accuracy_percent"], decimals=1),
            "Average Whisper Time (ms)": fmt(result["avg_whisper_time_ms"], decimals=1),
            "Average GPU Usage (%)": fmt(result["avg_gpu_percent"], decimals=1),
            "Average CPU Usage (%)": fmt(result["avg_cpu_percent"], decimals=1),
        })

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as f:
        headers = list(rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[item] for item in headers) + " |\n")

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    print()
    print(md_path.read_text().strip())
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper models for poster metrics")
    parser.add_argument("--audio", type=Path, default=ROOT / "samples" / "jfk.wav")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--tiny-model", type=Path, default=ROOT / "models" / "ggml-tiny.en.bin")
    parser.add_argument("--base-model", type=Path, default=ROOT / "models" / "ggml-base.en.bin")
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "bin" / "whisper-cli")
    parser.add_argument("--cpu-binary", type=Path, default=ROOT / "build" / "bin" / "whisper-cli")
    parser.add_argument("--gpu-binary", type=Path, default=ROOT / "build-cuda-orin" / "bin" / "whisper-cli")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--monitor-interval-ms", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "poster_metrics")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--mode", choices=["default", "cpu", "gpu", "both"], default="default")
    args = parser.parse_args()

    reference = args.reference_file.read_text().strip() if args.reference_file else args.reference
    jobs = []
    if args.mode == "both":
        jobs = [
            ("tiny.en CPU", args.tiny_model, args.cpu_binary, True),
            ("base.en CPU", args.base_model, args.cpu_binary, True),
            ("tiny.en GPU", args.tiny_model, args.gpu_binary, False),
            ("base.en GPU", args.base_model, args.gpu_binary, False),
        ]
    elif args.mode == "cpu":
        jobs = [
            ("tiny.en CPU", args.tiny_model, args.cpu_binary, True),
            ("base.en CPU", args.base_model, args.cpu_binary, True),
        ]
    elif args.mode == "gpu":
        jobs = [
            ("tiny.en GPU", args.tiny_model, args.gpu_binary, False),
            ("base.en GPU", args.base_model, args.gpu_binary, False),
        ]
    else:
        jobs = [
            ("tiny.en", args.tiny_model, args.binary, args.no_gpu),
            ("base.en", args.base_model, args.binary, args.no_gpu),
        ]

    results = []
    for name, model, binary, no_gpu in jobs:
        args.binary = binary
        args.no_gpu = no_gpu
        results.append(summarize_model(name, model, args, reference))
    write_outputs(results, args.out)


if __name__ == "__main__":
    main()
