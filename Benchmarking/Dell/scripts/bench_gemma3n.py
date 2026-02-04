#!/usr/bin/env python3
# benchmark gemma-3n (image->text) with a standardized 5-task suite
# logs per task per image to jsonl + summary json
# designed to be launched by the orchestrator inside the gemma venv

import argparse
import gc
import json
import os
import platform
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import psutil
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# avoid torchvision on jetson / thor (safe elsewhere too)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

MODEL_KEY = "gemma3n"
MODEL_ID = "google/gemma-3n-E4B-it"

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def parse_args():b
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="text file with one image path per line")
    p.add_argument("--output_dir", required=True, help="base outputs dir")
    p.add_argument("--prompt", default="Write one detailed sentence describing the image.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--run_group", required=True, help="experiment tag + timestamp from orchestrator")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="resume from existing runs.jsonl")
    p.add_argument("--power_sample_interval_ms", type=int, default=100, help="power sampling interval in ms")
    return p.parse_args()


def read_manifest(path: Path) -> list[str]:
    paths = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    return paths


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(fp, rec: dict):
    fp.write(json.dumps(rec) + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def load_completed_images(runs_path: Path, task_names: set[str]) -> set[str]:
    """
    Returns a set of image_path strings that already have ALL tasks logged.
    """
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

    completed = {img for img, ts in per_image_tasks.items() if task_names.issubset(ts)}
    return completed


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def system_snapshot() -> dict:
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0),
        "ram_used_mb": round(mem.used / (1024**2), 1),
        "ram_available_mb": round(mem.available / (1024**2), 1),
        "ram_percent": mem.percent,
    }


def resolve_device_dtype(dtype_flag: str):
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


def get_gpu_utilization() -> float:
    """Get GPU compute utilization percentage using pynvml"""
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
    
    # Add GPU utilization
    gpu_util = get_gpu_utilization()
    if gpu_util is not None:
        snapshot["gpu_utilization_percent"] = round(gpu_util, 1)
    
    return snapshot


class PowerMonitor:
    """Monitor GPU power draw during inference"""
    
    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.samples = []
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.end_time = None
        
        # Check if power monitoring is available
        self.available = False
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # Test if power draw is supported
                pynvml.nvmlDeviceGetPowerUsage(handle)
                self.handle = handle
                self.available = True
            except Exception:
                self.available = False
    
    def _monitor_loop(self):
        """Background thread that samples power draw"""
        while self.monitoring:
            try:
                # Power usage is in milliwatts, convert to watts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                power_w = power_mw / 1000.0
                self.samples.append(power_w)
            except Exception:
                pass
            time.sleep(self.sample_interval_ms / 1000.0)
    
    def start(self):
        """Start monitoring power draw"""
        if not self.available:
            return
        
        self.samples = []
        self.monitoring = True
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return power statistics"""
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
        """Cleanup NVML resources"""
        if self.available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def get_image_resolution(image: Image.Image) -> dict:
    """Get image resolution information"""
    width, height = image.size
    return {
        "image_width": width,
        "image_height": height,
        "image_resolution": f"{width}x{height}",
    }


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


def std_dev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_val = sum(values) / len(values)
    var = sum((v - mean_val) ** 2 for v in values) / len(values)
    return var ** 0.5


def gpu_mem_gb_from_cuda_stats(cuda_stats: dict) -> float | None:
    if not cuda_stats or not cuda_stats.get("cuda"):
        return None
    max_alloc = cuda_stats.get("gpu_max_mem_alloc_mb")
    max_reserved = cuda_stats.get("gpu_max_mem_reserved_mb")
    candidates = [v for v in [max_alloc, max_reserved] if isinstance(v, (int, float))]
    if not candidates:
        return None
    peak_mb = max(candidates)
    return round(peak_mb / 1024.0, 3)


def build_tasks():
    # bounded outputs, no <end> tokens that models may ignore
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


def gemma_generate(
    processor,
    model,
    image: Image.Image,
    user_text: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, str | None, int, int]:
    """
    Generate text from image using Gemma 3n model.
    
    IMPORTANT: Do not pass pad_token_id or eos_token_id to generate().
    Gemma 3n generates only PAD tokens when these are explicitly set.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    if device == "cuda":
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(device="cuda", dtype=model.dtype)
            else:
                inputs[k] = v.to("cuda")

    # CRITICAL: Do NOT pass pad_token_id or eos_token_id
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = int(inputs["input_ids"].shape[-1])
    gen_ids = outputs[0, input_len:]
    gen_len = int(gen_ids.numel())

    # Decode with skip_special_tokens=True for clean output
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    err = None
    if text == "":
        err = "empty_generation"

    return text, err, input_len, gen_len


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
    device, dtype = resolve_device_dtype(args.dtype)

    # Initialize power monitor
    power_monitor = PowerMonitor(sample_interval_ms=args.power_sample_interval_ms)

    # outputs/<model_key>/<run_group>/
    out_dir = Path(args.output_dir).resolve() / MODEL_KEY / args.run_group
    ensure_dir(out_dir)

    run_meta = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "run_group": args.run_group,
        "host": host,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
        "manifest": str(manifest),
        "num_images": len(image_paths),
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "suite": "week5_5tasks_bounded",
        "power_monitoring_available": power_monitor.available,
        "power_sample_interval_ms": args.power_sample_interval_ms if power_monitor.available else None,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"loading {MODEL_ID} on {device} (dtype={dtype})")
    print(f"power monitoring: {'enabled' if power_monitor.available else 'not available'}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="cuda" if device == "cuda" else None,
        attn_implementation="eager",
    )
    model.eval()

    tasks = build_tasks()
    tasks["caption_brief"]["prompt"] = args.prompt

    task_names = set(tasks.keys())

    runs_path = out_dir / "runs.jsonl"

    completed_images = set()
    if args.resume:
        completed_images = load_completed_images(runs_path, task_names)
        if completed_images:
            print(f"resume enabled: {len(completed_images)} images already complete (all tasks logged)")

    if args.resume:
        image_paths = [p for p in image_paths if str(p) not in completed_images]
        if not image_paths:
            print("nothing left to do (all images already complete).")
            return

    # warmup (first image + first task, then full pass through all tasks)
    if args.warmup > 0:
        try:
            img0 = Image.open(image_paths[0]).convert("RGB")
            first_task = next(iter(tasks.values()))
            for _ in range(args.warmup):
                _ = gemma_generate(
                    processor=processor,
                    model=model,
                    image=img0,
                    user_text=first_task["prompt"],
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            # plus one pass of all tasks on first image
            for task_name, task_cfg in tasks.items():
                _ = gemma_generate(
                    processor=processor,
                    model=model,
                    image=img0,
                    user_text=task_cfg["prompt"],
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            print(f"warmup complete: {args.warmup} runs + 1 full pass")
        except Exception as e:
            print(f"warmup failed (continuing anyway): {e}")

    latencies = []
    throughputs = []
    errors = 0
    total_runs = 0
    
    # Track total benchmark time
    benchmark_start = time.perf_counter()

    with open(runs_path, "a") as f:
        global_i = 0
        for batch_i, batch in enumerate(chunked(image_paths, args.batch_size), start=1):
            # Print batch header (keeping Dell's format)
            print()
            print(f"[{MODEL_KEY}] batch {batch_i} ({len(batch)} images)")
            
            for img_path in batch:
                img_i = global_i
                global_i += 1

                img_name = Path(img_path).name

                # load once per image; reuse for all tasks
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_resolution = get_image_resolution(img)
                except Exception as e:
                    # log one record per task even if the image failed to load
                    for task_name in tasks.keys():
                        rec = {
                            "image_index": img_i,
                            "image_name": img_name,
                            "image_path": str(img_path),
                            "task": task_name,
                            "latency_ms": None,
                            "images_per_second": None,
                            "power_watts": None,
                            "power_std": None,
                            "gpu_mem_gb": None,
                            "output_text": "",
                            "error": f"image_load_failed: {e}",
                            "sys_before": system_snapshot(),
                            "sys_after": system_snapshot(),
                            "cuda_stats": cuda_snapshot(),
                            "power_stats": None,
                            "n_tiles": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                        append_jsonl(f, rec)
                        total_runs += 1
                        errors += 1
                    continue

                for task_name, task_cfg in tasks.items():
                    total_runs += 1
                    latencies_for_task = []
                    input_len = None
                    gen_len = None
                    txt = ""
                    err = None
                    before_sys = system_snapshot()
                    reset_cuda_peaks()

                    # Start power monitoring
                    power_monitor.start()

                    for repeat in range(args.repeats):
                        def call():
                            try:
                                result = gemma_generate(
                                    processor=processor,
                                    model=model,
                                    image=img,
                                    user_text=task_cfg["prompt"],
                                    device=device,
                                    max_new_tokens=args.max_new_tokens,
                                )
                                return result
                            except Exception as e:
                                return "", str(e), None, None

                        (txt_rep, err_rep, input_len_rep, gen_len_rep), ms = timed_call(call, device=device)
                        latencies_for_task.append(ms)
                        if repeat == 0:
                            txt, err, input_len, gen_len = txt_rep, err_rep, input_len_rep, gen_len_rep
                        if err_rep is None:
                            latencies.append(ms)

                    # Stop power monitoring
                    power_stats = power_monitor.stop()

                    after_sys = system_snapshot()
                    cuda_stats = cuda_snapshot()
                    gpu_mem_gb = gpu_mem_gb_from_cuda_stats(cuda_stats)

                    power_watts = None
                    power_std = None
                    if power_stats:
                        power_watts = power_stats.get("power_watts_avg")
                        samples = power_stats.get("power_watts_samples") or []
                        power_std = std_dev(samples)

                    mean_latency = sum(latencies_for_task) / len(latencies_for_task) if latencies_for_task else None

                    # Calculate throughput
                    images_per_sec = 1000.0 / mean_latency if mean_latency and mean_latency > 0 else None

                    if err is not None:
                        errors += 1
                    else:
                        if images_per_sec is not None:
                            throughputs.append(images_per_sec)

                    rec = {
                        "image_index": img_i,
                        "image_name": img_name,
                        "image_path": str(img_path),
                        **img_resolution,  # Add resolution info
                        "task": task_name,
                        "task_purpose": task_cfg.get("purpose"),
                        "task_prompt": task_cfg.get("prompt"),
                        "latency_ms": round(mean_latency, 3) if mean_latency else None,
                        "latencies_ms": [round(l, 3) for l in latencies_for_task],
                        "images_per_second": round(images_per_sec, 3) if images_per_sec else None,
                        "input_len": input_len,
                        "gen_len": gen_len,
                        "power_watts": round(power_watts, 3) if power_watts is not None else None,
                        "power_std": round(power_std, 3) if power_std is not None else None,
                        "gpu_mem_gb": gpu_mem_gb,
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

                    # Keep Dell's preferred terminal format
                    status = "ok" if err is None else "error"
                    throughput_str = f" ({images_per_sec:.2f} img/s)" if images_per_sec else ""
                    if mean_latency:
                        print(f"[{MODEL_KEY}] img {img_i+1}/{len(image_paths)} task {task_name:30} {status} {mean_latency:.1f} ms{throughput_str}")
                    else:
                        print(f"[{MODEL_KEY}] img {img_i+1}/{len(image_paths)} task {task_name:30} {status}")

    lat_sorted = sorted(latencies)
    throughput_sorted = sorted(throughputs)
    
    # Calculate total elapsed time
    benchmark_end = time.perf_counter()
    elapsed_seconds = benchmark_end - benchmark_start
    elapsed_minutes = elapsed_seconds / 60.0
    
    summary = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "run_group": args.run_group,
        "host": host,
        "device": device,
        "dtype": str(dtype),
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
        "latency_ms_p99": round(percentile(lat_sorted, 99), 3) if lat_sorted else None,
        "images_per_second_mean": round(sum(throughputs) / len(throughputs), 3) if throughputs else None,
        "images_per_second_min": round(min(throughputs), 3) if throughputs else None,
        "images_per_second_max": round(max(throughputs), 3) if throughputs else None,
        "images_per_second_p50": round(percentile(throughput_sorted, 50), 3) if throughput_sorted else None,
        "images_per_second_p90": round(percentile(throughput_sorted, 90), 3) if throughput_sorted else None,
        "power_monitoring_available": power_monitor.available,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
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

    # cleanup
    power_monitor.cleanup()
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
