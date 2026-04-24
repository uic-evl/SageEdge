#!/usr/bin/env python3
# benchmark llava (image->text) with a standardized 5-task suite
# logs per task per image to jsonl + summary json
# designed to be launched by the orchestrator inside the llava venv
# Thor version: manual device placement (no device_map="auto")
# Thor note: includes tegrastats support for power monitoring

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import psutil
from PIL import Image

# -----------------------------
# QUIET MODE SETUP
# -----------------------------

# Suppress HF + PyTorch warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="Found GPU0")
warnings.filterwarnings("ignore", category=UserWarning)

# Disable HF progress bars + logging noise
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# -----------------------------
# LLaVA imports (after quiet mode)
# -----------------------------

from llavamini.model.builder import load_pretrained_model
from llavamini.mm_utils import get_model_name_from_path, process_images
from llavamini.constants import IMAGE_TOKEN_INDEX
from llavamini.mm_utils import tokenizer_image_token

MODEL_KEY = "llava"
MODEL_ID = "ICTNLP/llava-mini-llama-3.1-8b"

# Try to import pynvml for GPU power monitoring (DGX)
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="text file with one image path per line")
    p.add_argument("--output_dir", required=True, help="base outputs dir")
    p.add_argument("--prompt", default="Write one detailed sentence describing the image.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
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
    """Get GPU compute utilization - returns None if not available"""
    # NVML not available on Jetson, utilize tegrastats via PowerMonitor instead
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
    
    # Add GPU utilization if available
    gpu_util = get_gpu_utilization()
    if gpu_util is not None:
        snapshot["gpu_utilization_percent"] = round(gpu_util, 1)
    
    return snapshot


def check_tegrastats_available() -> bool:
    """Check if tegrastats is available (Jetson platform)"""
    try:
        result = subprocess.run(
            ["which", "tegrastats"],
            capture_output=True,
            timeout=1
        )
        return result.returncode == 0
    except Exception:
        return False


def parse_tegrastats_line(line: str) -> dict:
    """
    Parse a line from tegrastats output.
    Extract power values in mW (milliwatts)
    """
    power_data = {}
    
    # Extract all power readings (format: NAME XXXXmW/XXXXmW)
    power_pattern = r'(\w+)\s+(\d+)mW/\d+mW'
    matches = re.findall(power_pattern, line)
    
    for name, value_str in matches:
        try:
            power_mw = int(value_str)
            power_data[name] = power_mw / 1000.0  # Convert to watts
        except ValueError:
            continue
    
    return power_data


class PowerMonitor:
    """Monitor GPU power draw during inference (supports both pynvml/DGX and tegrastats/Jetson)"""

    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.samples = []
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.method = None
        self.available = False
        
        # Try pynvml first (DGX, NVIDIA GPUs)
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # Test if power draw is supported
                pynvml.nvmlDeviceGetPowerUsage(handle)
                self.handle = handle
                self.available = True
                self.method = "pynvml"
                print("Power monitoring: using pynvml (NVIDIA GPU)")
                return
            except Exception:
                pass
        
        # Fall back to tegrastats (Jetson)
        if check_tegrastats_available():
            try:
                subprocess.run(
                    ["sudo", "-n", "tegrastats", "--interval", "1000", "--stop"],
                    capture_output=True,
                    timeout=2,
                    text=True,
                )
                self.available = True
                self.method = "tegrastats"
                self.tegrastats_process = None
                print("Power monitoring: using tegrastats (Jetson)")
            except Exception:
                self.available = False

    def _monitor_loop_pynvml(self):
        """Background thread that samples power draw using pynvml"""
        while self.monitoring:
            try:
                # Power usage is in milliwatts, convert to watts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                power_w = power_mw / 1000.0
                self.samples.append(power_w)
            except Exception:
                pass
            time.sleep(self.sample_interval_ms / 1000.0)

    def _monitor_loop_tegrastats(self):
        try:
            self.tegrastats_process = subprocess.Popen(
                ["sudo", "tegrastats", "--interval", str(self.sample_interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            for line in self.tegrastats_process.stdout:
                if not self.monitoring:
                    break
                power_data = parse_tegrastats_line(line)
                if power_data:
                    self.samples.append(power_data)
        except Exception:
            pass
        finally:
            if self.tegrastats_process:
                self.tegrastats_process.terminate()
                try:
                    self.tegrastats_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.tegrastats_process.kill()

    def start(self):
        if not self.available:
            return
        self.samples = []
        self.monitoring = True
        self.start_time = time.perf_counter()
        
        if self.method == "pynvml":
            self.thread = threading.Thread(target=self._monitor_loop_pynvml, daemon=True)
        elif self.method == "tegrastats":
            self.thread = threading.Thread(target=self._monitor_loop_tegrastats, daemon=True)
        
        if self.thread:
            self.thread.start()

    def stop(self):
        if not self.available:
            return None
        self.monitoring = False
        self.end_time = time.perf_counter()
        
        # Stop tegrastats process if running
        if self.method == "tegrastats" and self.tegrastats_process:
            self.tegrastats_process.terminate()
            try:
                self.tegrastats_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.tegrastats_process.kill()
        
        if self.thread:
            self.thread.join(timeout=2.0)
        if not self.samples:
            return None

        duration_s = self.end_time - self.start_time
        
        if self.method == "pynvml":
            # pynvml samples are floats (watts)
            avg_watts = sum(self.samples) / len(self.samples)
            peak_watts = max(self.samples)
            min_watts = min(self.samples)
            energy_joules = avg_watts * duration_s
            
            return {
                "method": "pynvml",
                "power_watts_avg": round(avg_watts, 2),
                "power_watts_peak": round(peak_watts, 2),
                "power_watts_min": round(min_watts, 2),
                "energy_joules_est": round(energy_joules, 2),
                "sample_count": len(self.samples),
                "duration_seconds": round(duration_s, 3),
            }
        
        elif self.method == "tegrastats":
            # tegrastats samples are dicts with multiple power rails
            power_rails = {}
            all_rails = set()
            for sample in self.samples:
                all_rails.update(sample.keys())
            for rail in all_rails:
                vals = [s[rail] for s in self.samples if rail in s]
                if vals:
                    power_rails[rail] = {
                        "avg": round(sum(vals) / len(vals), 2),
                        "peak": round(max(vals), 2),
                        "min": round(min(vals), 2),
                    }
            # Use VDD_GPU_SOC as primary GPU power metric for Jetson
            gpu_avg = power_rails.get("VDD_GPU_SOC", {}).get("avg")
            if gpu_avg is None:
                # Fall back to VDD_GPU if VDD_GPU_SOC not available
                gpu_avg = power_rails.get("VDD_GPU", {}).get("avg")
            
            return {
                "method": "tegrastats",
                "power_rails": power_rails,
                "power_watts_avg": gpu_avg,
                "energy_joules_est": round(gpu_avg * duration_s, 2) if gpu_avg else None,
                "sample_count": len(self.samples),
                "duration_seconds": round(duration_s, 3),
            }
        
        return None

    def cleanup(self):
        if self.method == "pynvml" and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        elif self.method == "tegrastats" and self.tegrastats_process:
            try:
                self.tegrastats_process.terminate()
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


def llava_generate(model, tokenizer, image_processor, image: Image.Image, user_text: str, device: str, max_new_tokens: int) -> tuple[str, str | None, int | None, int | None]:
    """
    Generate text from image using LLaVA model.
    
    ACTUALLY FIXED: Decode full output, then use string matching to remove the user prompt.
    This handles image token expansion correctly.
    """
    try:
        # Process image and text
        image_tensor = process_images(
            [image],
            image_processor,
            model.config,
        ).unsqueeze(1)

        prompt = f"<image>\n{user_text}"

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)

        # Store the text token length (before image expansion)
        text_token_len = input_ids.shape[1]

        inputs = {
            "input_ids": input_ids,
            "images": image_tensor,
        }

        # Move to device with proper dtype handling
        if device == "cuda":
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(device="cuda", dtype=model.dtype)
                else:
                    inputs[k] = v.to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                images=inputs["images"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # THE REAL FIX: Decode the full output (prompt + response)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # Remove the user prompt text using string matching
        # The <image> token doesn't appear in decoded text, so we only need to remove user_text
        if user_text in full_text:
            # Find and remove the prompt
            idx = full_text.find(user_text)
            answer = full_text[idx + len(user_text):].strip()
        else:
            # Fallback: if prompt not found, the full text IS the answer
            answer = full_text

        # Token counts
        full_output_len = output_ids.shape[1]
        prompt_tokens = text_token_len
        response_tokens_est = full_output_len - text_token_len
        
        return answer, None, prompt_tokens, response_tokens_est
        
    except Exception as e:
        return "", str(e), None, None


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
        "power_monitoring_method": power_monitor.method,
        "power_sample_interval_ms": args.power_sample_interval_ms if power_monitor.available else None,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"loading {MODEL_ID} on {device} (dtype={dtype})")
    print(f"power monitoring: {'enabled' if power_monitor.available else 'not available'}")

    model_name = get_model_name_from_path(MODEL_ID)

    # Load with device_map=None to prevent auto-offloading, then manually move to GPU
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_ID,
        None,
        model_name,
        device_map=None,     # Disable automatic device mapping
        torch_dtype=dtype,   # Pass dtype to load function
    )
    
    # CRITICAL: Explicitly move model to GPU after loading
    # device_map=None loads to CPU, so we must manually move it
    if device == "cuda":
        print(f"Moving model to {device} with dtype {dtype}...")
        model = model.to(device=device, dtype=dtype)
        print(f"Model successfully moved to {device}")
        print(f"GPU Memory after move: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
    
    print(f"Model loaded with device={device}, dtype={dtype}")

    # Try to force eager attention after loading
    try:
        if hasattr(model.config, '_attn_implementation'):
            model.config._attn_implementation = 'eager'
            print("Set attention implementation to: eager")
        elif hasattr(model.config, 'attn_implementation'):
            model.config.attn_implementation = 'eager'
            print("Set attention implementation to: eager")
        else:
            print("Warning: Could not set attention implementation (using default)")
    except Exception as e:
        print(f"Warning: Could not set attention implementation: {e}")

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
                _ = llava_generate(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image=img0,
                    user_text=first_task["prompt"],
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            # plus one pass of all tasks on first image
            for task_name, task_cfg in tasks.items():
                _ = llava_generate(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
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
            # Print batch header
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
                    prompt_tokens = None
                    response_tokens_est = None
                    txt = ""
                    err = None
                    before_sys = system_snapshot()
                    reset_cuda_peaks()

                    # Start power monitoring
                    power_monitor.start()

                    for repeat in range(args.repeats):
                        def call():
                            try:
                                result = llava_generate(
                                    model=model,
                                    tokenizer=tokenizer,
                                    image_processor=image_processor,
                                    image=img,
                                    user_text=task_cfg["prompt"],
                                    device=device,
                                    max_new_tokens=args.max_new_tokens,
                                )
                                return result
                            except Exception as e:
                                return "", str(e), None, None

                        (txt_rep, err_rep, prompt_tokens_rep, response_tokens_est_rep), ms = timed_call(call, device=device)
                        latencies_for_task.append(ms)
                        if repeat == 0:
                            txt, err, prompt_tokens, response_tokens_est = txt_rep, err_rep, prompt_tokens_rep, response_tokens_est_rep
                        if err_rep is None:
                            latencies.append(ms)

                    # Stop power monitoring
                    power_stats = power_monitor.stop()

                    after_sys = system_snapshot()
                    cuda_stats = cuda_snapshot()

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

                    # Terminal output
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
        "images_per_second_mean": round(sum(throughputs) / len(throughputs), 3) if throughputs else None,
        "images_per_second_min": round(min(throughputs), 3) if throughputs else None,
        "images_per_second_max": round(max(throughputs), 3) if throughputs else None,
        "images_per_second_p50": round(percentile(throughput_sorted, 50), 3) if throughput_sorted else None,
        "images_per_second_p90": round(percentile(throughput_sorted, 90), 3) if throughput_sorted else None,
        "power_monitoring_available": power_monitor.available,
        "power_monitoring_method": power_monitor.method,
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
    print("=" * 60)

    # cleanup
    power_monitor.cleanup()
    del model
    del tokenizer
    del image_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()