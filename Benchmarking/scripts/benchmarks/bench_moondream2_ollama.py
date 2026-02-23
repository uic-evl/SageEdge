#!/usr/bin/env python3
# benchmark moondream via ollama (image->text) with a standardized 5-task suite
# logs per task per image to jsonl + summary json
# designed to be launched by the orchestrator inside a python venv
#
# requires:
#   pip install requests
# and an ollama vision model pulled, e.g.:
#   ollama pull moondream
#   ollama pull llava

import argparse
import base64
import gc
import json
import os
import platform
import threading
import time
import subprocess
import re
from datetime import datetime
from pathlib import Path

import psutil
import requests
import torch
from PIL import Image

MODEL_KEY = "moondream2"
# for ollama, model id is the ollama model name, e.g. "moondream" or "llava"
DEFAULT_OLLAMA_MODEL = "moondream"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# try to import pynvml for gpu monitoring (dell / nvidia)
try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    PYNVML_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="text file with one image path per line")
    p.add_argument("--output_dir", required=True, help="base outputs dir")
    p.add_argument("--prompt", default="Write one detailed sentence describing the image.")
    p.add_argument("--max_new_tokens", type=int, default=128)  # mapped to ollama num_predict
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)

    # kept for compatibility, but ignored for ollama precision
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")

    p.add_argument("--run_group", required=True, help="experiment tag + timestamp from orchestrator")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="resume from existing runs.jsonl")
    p.add_argument("--power_sample_interval_ms", type=int, default=100, help="power sampling interval in ms")
    p.add_argument("--num_crops", type=int, default=4, help="not used (kept for compatibility)")

    # ollama controls
    p.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST, help="ollama host, e.g. http://127.0.0.1:11434")
    p.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL, help="ollama model name, e.g. moondream or llava")
    p.add_argument("--ollama_timeout_s", type=int, default=600, help="http timeout for ollama requests")
    p.add_argument("--temperature", type=float, default=0.0, help="ollama temperature")

    # optional: manual label if auto-detection fails
    p.add_argument(
        "--precision_label",
        default="",
        help="optional label to record precision/format (e.g., q8_0, q4_0). overrides auto-detect if provided.",
    )
    return p.parse_args()


def read_manifest(path: Path) -> list[str]:
    paths = []
    with open(path, "r") as f:
        for line in f:
            pth = line.strip()
            if pth:
                paths.append(pth)
    return paths


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(fp, rec: dict):
    fp.write(json.dumps(rec) + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def load_completed_images(runs_path: Path, task_names: set[str]) -> set[str]:
    if not runs_path.exists():
        return set()

    per_image_tasks: dict[str, set[str]] = {}
    with open(runs_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            img_path = obj.get("image_path")
            task = obj.get("task")
            if not img_path or not task:
                continue
            per_image_tasks.setdefault(img_path, set()).add(task)

    return {img for img, ts in per_image_tasks.items() if task_names.issubset(ts)}


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def system_snapshot() -> dict:
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0),
        "ram_used_mb": round(mem.used / (1024**2), 1),
        "ram_available_mb": round(mem.available / (1024**2), 1),
        "ram_percent": mem.percent,
    }


def resolve_device_dtype(dtype_flag: str):
    # kept mostly for logging compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        return device, torch.float32
    if dtype_flag == "fp16":
        return device, torch.float16
    if dtype_flag == "bf16":
        return device, torch.bfloat16
    return device, torch.float32


def reset_cuda_peaks():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_utilization() -> float | None:
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


def cuda_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}

    snapshot = {
        "cuda": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_mem_alloc_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
        "gpu_max_mem_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "gpu_max_mem_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
    }

    gpu_util = get_gpu_utilization()
    if gpu_util is not None:
        snapshot["gpu_utilization_percent"] = round(gpu_util, 1)

    return snapshot


def detect_ollama_quantization(model: str) -> str | None:
    """
    best-effort detection of weight format / quantization for ollama models.
    tries:
      - exact q tags embedded in model string (e.g., llama3.2-vision:q8_0)
      - `ollama show <model>` output regex (common in modelfiles/gguf metadata)
    returns q8_0/q4_0/etc or None if unknown.
    """
    # 1) if the user/model tag already contains quantization info after a colon
    #    e.g., "llama3.2-vision:q8_0"
    m = re.search(r":(q\d+_(?:0|1|k_[a-z]+(?:_[a-z]+)?))\b", model.lower())
    if m:
        return m.group(1)

    # 2) shell out to ollama show
    try:
        out = subprocess.check_output(["ollama", "show", model], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None

    # look for gguf quant types like q4_0, q8_0, q4_k_m, q5_k_m, etc
    m2 = re.search(r"\b(q\d+_(?:0|1|k_[a-z]+(?:_[a-z]+)?))\b", out.lower())
    if m2:
        return m2.group(1)

    # some outputs might include "quantization: Q4_0" etc
    m3 = re.search(r"quant(?:ization)?\s*[:=]\s*(q\d+_[^\s]+)", out.lower())
    if m3:
        return m3.group(1).strip()

    return None


def timed_call(fn, device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def build_tasks():
    return {
        "caption_brief": {
            "prompt": "Write one sentence caption describing the image.",
            "purpose": "low-latency captioning",
        },
        "objects_and_counts": {
            "prompt": (
                "List up to 8 main objects with approximate counts. "
                "Use format 'object: count' on separate lines."
            ),
            "purpose": "object recognition (approximate)",
        },
        "spatial_relationships": {
            "prompt": (
                "Write 2-3 sentences describing spatial relationships between the main objects "
                "(left/right, foreground/background, near/far)."
            ),
            "purpose": "spatial grounding",
        },
        "scene_context": {
            "prompt": "Write exactly 2 sentences describing the overall scene and setting (where it is and what is happening).",
            "purpose": "scene understanding",
        },
        "attributes": {
            "prompt": "Write exactly 2 sentences describing notable visual attributes (colors, lighting, materials, weather).",
            "purpose": "fine-grained perception",
        },
    }


def image_to_base64_jpeg(image_path: str) -> str:
    # ollama expects base64 image bytes
    # we normalize to jpeg to avoid weird png modes
    img = Image.open(image_path).convert("RGB")
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ollama_generate(
    ollama_host: str,
    ollama_model: str,
    image_b64: str,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    timeout_s: int,
) -> tuple[str, str | None, int | None, int | None]:
    """
    use ollama /api/generate with images=[base64] and prompt=user_text
    returns: text, err, prompt_tokens, response_tokens_est
    """
    url = ollama_host.rstrip("/") + "/api/generate"
    payload = {
        "model": ollama_model,
        "prompt": user_text,
        "images": [image_b64],
        "stream": False,
        "options": {
            "num_predict": int(max_new_tokens),
            "temperature": float(temperature),
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return "", f"ollama_http_{r.status_code}: {r.text[:200]}", None, None
        data = r.json()
        text = data.get("response", "") or ""

        # token counts are not always provided; keep schema compatible by estimating
        prompt_tokens = None
        response_tokens_est = None
        if isinstance(data.get("prompt_eval_count"), int):
            prompt_tokens = int(data["prompt_eval_count"])
        if isinstance(data.get("eval_count"), int):
            response_tokens_est = int(data["eval_count"])

        return text, None, prompt_tokens, response_tokens_est
    except Exception as e:
        return "", str(e), None, None


class PowerMonitor:
    """Monitor GPU power draw using nvml (dell/nvidia)."""

    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.samples = []
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.available = False

        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetPowerUsage(handle)
                self.handle = handle
                self.available = True
            except Exception:
                self.available = False

    def _monitor_loop(self):
        while self.monitoring:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append(power_mw / 1000.0)
            except Exception:
                pass
            time.sleep(self.sample_interval_ms / 1000.0)

    def start(self):
        if not self.available:
            return
        self.samples = []
        self.monitoring = True
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.available:
            return None
        self.monitoring = False
        self.end_time = time.perf_counter()

        if self.thread:
            self.thread.join(timeout=1.0)

        if not self.samples:
            return None

        duration_s = self.end_time - self.start_time
        avg_watts = sum(self.samples) / len(self.samples)
        peak_watts = max(self.samples)
        energy_joules = avg_watts * duration_s

        return {
            "power_watts_samples": [round(s, 2) for s in self.samples],
            "power_watts_avg": round(avg_watts, 2),
            "power_watts_peak": round(peak_watts, 2),
            "energy_joules_est": round(energy_joules, 2),
            "sample_count": len(self.samples),
            "duration_seconds": round(duration_s, 3),
        }

    def cleanup(self):
        if self.available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def get_image_resolution(image: Image.Image) -> dict:
    w, h = image.size
    return {"image_width": w, "image_height": h, "image_resolution": f"{w}x{h}"}


def main():
    args = parse_args()

    manifest = Path(args.manifest).resolve()
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")

    image_paths = read_manifest(manifest)
    image_paths = [str(Path(p).resolve()) for p in image_paths]

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit("no images in manifest")

    host = platform.node() or "unknown_host"
    device, _dtype_for_device = resolve_device_dtype(args.dtype)

    power_monitor = PowerMonitor(sample_interval_ms=args.power_sample_interval_ms)

    out_dir = Path(args.output_dir).resolve() / MODEL_KEY / args.run_group
    ensure_dir(out_dir)

    # runtime / precision tracking
    runtime = "ollama"
    precision_mode = "quantized"
    quantization = None
    if args.precision_label.strip():
        quantization = args.precision_label.strip()
    else:
        quantization = detect_ollama_quantization(args.ollama_model) or "unknown"

    run_meta = {
        "model_key": MODEL_KEY,
        "model_id": f"ollama:{args.ollama_model}",
        "run_group": args.run_group,
        "host": host,
        "device": device,

        # dtype is not user-controlled in ollama; record as NA to avoid confusion
        "dtype": "na_ollama",

        "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
        "manifest": str(manifest),
        "num_images": len(image_paths),
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "num_crops": args.num_crops,
        "suite": "week5_5tasks_bounded",
        "n_tiles": 1,

        # method tracking
        "runtime": runtime,
        "precision_mode": precision_mode,
        "quantization": quantization,
        "ollama_model_tag": args.ollama_model,

        "power_monitoring_available": power_monitor.available,
        "power_sample_interval_ms": args.power_sample_interval_ms if power_monitor.available else None,
        "ollama_host": args.ollama_host,
        "ollama_model": args.ollama_model,
        "ollama_timeout_s": args.ollama_timeout_s,
        "ollama_temperature": args.temperature,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"using ollama model {args.ollama_model} at {args.ollama_host}")
    print(f"precision: {precision_mode} ({quantization})")
    print(f"power monitoring: {'enabled' if power_monitor.available else 'not available'}")

    tasks = build_tasks()
    tasks["caption_brief"]["prompt"] = args.prompt
    task_names = set(tasks.keys())
    runs_path = out_dir / "runs.jsonl"

    if args.resume:
        completed_images = load_completed_images(runs_path, task_names)
        if completed_images:
            print(f"resume enabled: {len(completed_images)} images already complete (all tasks logged)")
        image_paths = [p for p in image_paths if str(p) not in completed_images]
        if not image_paths:
            print("nothing left to do (all images already complete).")
            return

    # warmup
    if args.warmup > 0:
        try:
            img0_path = image_paths[0]
            img0_b64 = image_to_base64_jpeg(img0_path)
            first_task = next(iter(tasks.values()))
            for _ in range(args.warmup):
                _ = ollama_generate(
                    ollama_host=args.ollama_host,
                    ollama_model=args.ollama_model,
                    image_b64=img0_b64,
                    user_text=first_task["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    timeout_s=args.ollama_timeout_s,
                )
            for task_cfg in tasks.values():
                _ = ollama_generate(
                    ollama_host=args.ollama_host,
                    ollama_model=args.ollama_model,
                    image_b64=img0_b64,
                    user_text=task_cfg["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    timeout_s=args.ollama_timeout_s,
                )
            print(f"warmup complete: {args.warmup} runs + 1 full pass")
        except Exception as e:
            print(f"warmup failed (continuing anyway): {e}")

    latencies = []
    throughputs = []
    errors = 0
    total_runs = 0

    benchmark_start = time.perf_counter()

    with open(runs_path, "a") as f:
        global_i = 0
        for batch_i, batch in enumerate(chunked(image_paths, args.batch_size), start=1):
            print(f"\n[{MODEL_KEY}] batch {batch_i} ({len(batch)} images)")
            for img_path in batch:
                img_i = global_i
                global_i += 1

                img_name = Path(img_path).name

                try:
                    # base64 once per image; reuse for all tasks
                    img_b64 = image_to_base64_jpeg(img_path)
                    # also load just for resolution logging
                    img = Image.open(img_path).convert("RGB")
                    img_resolution = get_image_resolution(img)
                except Exception as e:
                    for task_name in tasks.keys():
                        rec = {
                            "runtime": runtime,
                            "precision_mode": precision_mode,
                            "quantization": quantization,
                            "model_id": f"ollama:{args.ollama_model}",

                            "image_index": img_i,
                            "image_name": img_name,
                            "image_path": str(img_path),
                            "task": task_name,
                            "latency_ms": None,
                            "images_per_second": None,
                            "output_text": "",
                            "error": f"image_load_failed: {e}",
                            "sys_before": system_snapshot(),
                            "sys_after": system_snapshot(),
                            "cuda_stats": cuda_snapshot(),
                            "power_stats": None,
                            "n_tiles": 1,
                            "timestamp": datetime.now().isoformat(),
                        }
                        append_jsonl(f, rec)
                        total_runs += 1
                        errors += 1
                    continue

                for task_name, task_cfg in tasks.items():
                    latencies_for_task = []
                    prompt_tokens = None
                    response_tokens_est = None
                    txt = ""
                    err = None
                    before_sys = system_snapshot()
                    reset_cuda_peaks()

                    power_monitor.start()

                    for repeat in range(args.repeats):
                        def call():
                            return ollama_generate(
                                ollama_host=args.ollama_host,
                                ollama_model=args.ollama_model,
                                image_b64=img_b64,
                                user_text=task_cfg["prompt"],
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                timeout_s=args.ollama_timeout_s,
                            )

                        (txt_rep, err_rep, prompt_tok_rep, resp_tok_rep), ms = timed_call(call, device=device)
                        latencies_for_task.append(ms)

                        if repeat == 0:
                            txt, err = txt_rep, err_rep
                            prompt_tokens, response_tokens_est = prompt_tok_rep, resp_tok_rep

                        if err_rep is None:
                            latencies.append(ms)

                    power_stats = power_monitor.stop()

                    after_sys = system_snapshot()
                    cuda_stats = cuda_snapshot()

                    if err is not None:
                        errors += 1

                    mean_latency = sum(latencies_for_task) / len(latencies_for_task) if latencies_for_task else None
                    images_per_sec = 1000.0 / mean_latency if mean_latency and mean_latency > 0 else None
                    if images_per_sec is not None and err is None:
                        throughputs.append(images_per_sec)

                    rec = {
                        "runtime": runtime,
                        "precision_mode": precision_mode,
                        "quantization": quantization,
                        "model_id": f"ollama:{args.ollama_model}",

                        "image_index": img_i,
                        "image_name": img_name,
                        "image_path": str(img_path),
                        **img_resolution,
                        "task": task_name,
                        "task_purpose": task_cfg.get("purpose"),
                        "task_prompt": task_cfg.get("prompt"),
                        "latency_ms": round(mean_latency, 3) if mean_latency else None,
                        "latencies_ms": [round(l, 3) for l in latencies_for_task],
                        "images_per_second": round(images_per_sec, 3) if images_per_sec else None,
                        "prompt_tokens": prompt_tokens,
                        "response_tokens_est": response_tokens_est,
                        "n_tiles": 1,
                        "output_text": txt,
                        "error": err,
                        "sys_before": before_sys,
                        "sys_after": after_sys,
                        "cuda_stats": cuda_stats,
                        "power_stats": power_stats,
                        "timestamp": datetime.now().isoformat(),
                    }
                    append_jsonl(f, rec)
                    total_runs += 1

                    status = "ok" if err is None else "error"
                    throughput_str = f" ({images_per_sec:.2f} img/s)" if images_per_sec else ""
                    if mean_latency:
                        print(f"[{MODEL_KEY}] img {img_i+1}/{len(image_paths)} task {task_name:25s} {status} {mean_latency:.1f} ms{throughput_str}")
                    else:
                        print(f"[{MODEL_KEY}] img {img_i+1}/{len(image_paths)} task {task_name:25s} {status}")

    lat_sorted = sorted(latencies)
    throughput_sorted = sorted(throughputs)

    benchmark_end = time.perf_counter()
    elapsed_seconds = benchmark_end - benchmark_start
    elapsed_minutes = elapsed_seconds / 60.0

    summary = {
        "model_key": MODEL_KEY,
        "model_id": f"ollama:{args.ollama_model}",
        "run_group": args.run_group,
        "host": host,
        "device": device,

        # dtype is not controlled for ollama
        "dtype": "na_ollama",

        "runtime": runtime,
        "precision_mode": precision_mode,
        "quantization": quantization,
        "ollama_model_tag": args.ollama_model,

        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "num_images": len(image_paths),
        "num_tasks": len(tasks),
        "total_runs": total_runs,
        "num_errors": errors,
        "total_elapsed_seconds": round(elapsed_seconds, 2),
        "total_elapsed_minutes": round(elapsed_minutes, 2),
        "latency_ms_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "latency_ms_min": round(min(latencies), 3) if latencies else None,
        "latency_ms_max": round(max(latencies), 3) if latencies else None,
        "latency_ms_p50": round(percentile(lat_sorted, 50), 3) if lat_sorted else None,
        "latency_ms_p90": round(percentile(lat_sorted, 90), 3) if lat_sorted else None,
        "images_per_second_mean": round(sum(throughputs) / len(throughputs), 3) if throughputs else None,
        "images_per_second_min": round(min(throughputs), 3) if throughputs else None,
        "images_per_second_max": round(max(throughputs), 3) if throughputs else None,
        "images_per_second_p50": round(percentile(throughput_sorted, 50), 3) if throughput_sorted else None,
        "images_per_second_p90": round(percentile(throughput_sorted, 90), 3) if throughput_sorted else None,
        "power_monitoring_available": power_monitor.available,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("Benchmark complete!")
    print(f"Total elapsed time: {elapsed_minutes:.2f} minutes ({elapsed_seconds:.1f} seconds)")
    print(f"Total runs: {total_runs}, Errors: {errors}")
    if latencies:
        print(f"Latency (mean): {summary['latency_ms_mean']:.1f} ms")
        print(f"Latency (p50): {summary['latency_ms_p50']:.1f} ms")
    if throughputs:
        print(f"Throughput (mean): {summary['images_per_second_mean']:.2f} img/s")
    print(f"Results saved to: {out_dir}")
    print("=" * 60)

    power_monitor.cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
