#!/usr/bin/env python3
# benchmark internvl2-2b (image->text) with a standardized 5-task suite
# logs per task per image to jsonl + summary json
# designed to be launched by the orchestrator inside the internvl venv
# 
# Visual preprocessing: InternVL2 with max_num_tiles=12 and use_thumbnail=False
# (follows HF recommended settings for consistent visual-budget across images)

import argparse
import gc
import json
import os
import platform
import time
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import psutil
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_KEY = "internvl2_2b"
MODEL_ID = "OpenGVLab/InternVL2-2B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="text file with one image path per line")
    p.add_argument("--output_dir", required=True, help="base outputs dir")
    p.add_argument("--prompt", default="Write one detailed sentence describing the image.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--limit", type=int, default=10)
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
    return snapshot


def check_tegrastats_available() -> bool:
    """Check if tegrastats is available (Jetson platform)"""
    try:
        result = subprocess.run(["which", "tegrastats"], capture_output=True, timeout=1)
        return result.returncode == 0
    except Exception:
        return False


def parse_tegrastats_line(line: str) -> dict:
    """
    Parse a line from tegrastats output.
    
    Example line:
    01-08-2026 15:36:50 RAM 25686/125772MB (lfb 1234x4MB) SWAP 0/0MB (cached 0MB) 
    CPU [15%@2265,10%@2265,12%@2265,8%@2265] EMC_FREQ 0%@3199 GR3D_FREQ 99%@1300 
    AO@45C GPU@48C Tboard@44C VDD_GPU_SOC 15000mW/15000mW VDD_CPU_CV 3500mW/3500mW 
    VIN_SYS_5V0 18500mW/18500mW
    """
    parsed = {}
    
    # gpu utilization: tegrastats exposes it as "GR3D_FREQ XX%@YYYY"
    m = re.search(r"GR3D_FREQ\s+(\d+)%@", line)
    if m:
        try:
            parsed["gpu_utilization_percent"] = float(m.group(1))
        except Exception:
            pass
    
    # power rails (format: NAME XXXXmW/XXXXmW)
    power_pattern = r"(\w+)\s+(\d+)mW/\d+mW"
    matches = re.findall(power_pattern, line)
    for name, value_str in matches:
        try:
            power_mw = int(value_str)
            parsed[name] = power_mw / 1000.0
        except ValueError:
            continue
    return parsed


class PowerMonitor:
    """Monitor power draw during inference using tegrastats (Jetson)"""

    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.samples = []
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.method = None  # 'tegrastats'
        self.available = False

        # Use tegrastats on Jetson
        if check_tegrastats_available():
            try:
                subprocess.run([
                    "sudo", "-n", "tegrastats", "--interval", "1000", "--stop"
                ], capture_output=True, timeout=2, text=True)
                self.available = True
                self.method = 'tegrastats'
                self.tegrastats_process = None
            except Exception as e:
                pass

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
        except Exception as e:
            print(f"Error in tegrastats monitoring: {e}")
        finally:
            if hasattr(self, "tegrastats_process") and self.tegrastats_process:
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
        if self.method == 'tegrastats':
            self.thread = threading.Thread(target=self._monitor_loop_tegrastats, daemon=True)
        if self.thread:
            self.thread.start()

    def stop(self):
        if not self.available:
            return None
        self.monitoring = False
        self.end_time = time.perf_counter()
        if self.method == 'tegrastats' and hasattr(self, "tegrastats_process") and self.tegrastats_process:
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
        if self.method == 'tegrastats':
            vdd_gpu_soc_values = [s.get('VDD_GPU_SOC', 0) for s in self.samples if isinstance(s, dict)]
            vdd_cpu_cv_values = [s.get('VDD_CPU_CV', 0) for s in self.samples if isinstance(s, dict)]
            vin_sys_5v0_values = [s.get('VIN_SYS_5V0', 0) for s in self.samples if isinstance(s, dict)]
            
            # gpu utilization (%): collected from parse_tegrastats_line via GR3D_FREQ
            gpu_util_values = [s.get("gpu_utilization_percent", None) for s in self.samples if isinstance(s, dict)]
            gpu_util_values = [v for v in gpu_util_values if v is not None]
            
            result = {}
            if vdd_gpu_soc_values:
                result['power_gpu_soc_mean_watts'] = round(sum(vdd_gpu_soc_values) / len(vdd_gpu_soc_values), 3)
            if vdd_cpu_cv_values:
                result['power_cpu_cv_mean_watts'] = round(sum(vdd_cpu_cv_values) / len(vdd_cpu_cv_values), 3)
            if vin_sys_5v0_values:
                result['power_sys_5v0_mean_watts'] = round(sum(vin_sys_5v0_values) / len(vin_sys_5v0_values), 3)
            
            if gpu_util_values:
                result["gpu_utilization_percent_mean"] = round(sum(gpu_util_values) / len(gpu_util_values), 3)
            
            return result
        return None

    def cleanup(self):
        if self.method == "tegrastats" and hasattr(self, "tegrastats_process") and self.tegrastats_process:
            try:
                self.tegrastats_process.terminate()
                self.tegrastats_process.wait(timeout=1)
            except Exception:
                pass


def get_image_resolution(image: Image.Image) -> dict:
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


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448):
    """
    InternVL's dynamic image preprocessing with adaptive tiling.
    use_thumbnail is disabled to ensure fixed visual budget per image.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    return processed_images


def load_image_tiles(image_file: str, input_size: int = 448, max_num_tiles: int = 12):
    """
    Load and preprocess image for InternVL with natural variable tiling.
    Tile count varies based on image aspect ratio (per HF dynamic_preprocess).
    """
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num_tiles)
    
    pixel_values = [transform(t) for t in tiles]
    pixel_values = torch.stack(pixel_values)
    img_resolution = get_image_resolution(image)
    return pixel_values, len(tiles), img_resolution


def internvl_generate(model, tokenizer, pixel_values, user_text: str, max_new_tokens: int):
    # internvl expects <image> token in question for image+text
    question = f"<image>\n{user_text}"
    generation_config = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    try:
        with torch.inference_mode():
            text = model.chat(tokenizer, pixel_values, question, generation_config)
        prompt_tokens_est = len(tokenizer.encode(question))
        response_tokens_est = len(tokenizer.encode(text))
        return text, None, prompt_tokens_est, response_tokens_est
    except Exception as e:
        return "", str(e), None, None


def main():
    args = parse_args()

    manifest = Path(args.manifest).resolve()
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")

    image_paths = [str(Path(p).resolve()) for p in read_manifest(manifest)]
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

    # Fixed preprocessing settings for reproducibility
    max_num_tiles = 12  # HF recommended value
    use_thumbnail = False  # Disabled for consistent visual budget

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
        "max_num_tiles": max_num_tiles,
        "use_thumbnail": use_thumbnail,
        "power_monitoring_available": power_monitor.available,
        "power_monitoring_method": power_monitor.method,
        "power_sample_interval_ms": args.power_sample_interval_ms if power_monitor.available else None,
        "preprocessing_note": "HF default settings: max_num_tiles=12, use_thumbnail=False. Tile count varies naturally per image aspect ratio.",
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"loading {MODEL_ID} on {device} (dtype={dtype})")
    if device == "cuda" and args.dtype == "bf16":
        print("warning: bf16 can be flaky on some jetson setups; fp16 is usually the safest default")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype if device == "cuda" else torch.float32,
            attn_implementation="eager",
        ).eval()
    except Exception as e:
        print(f"Warning: Could not set attn_implementation='eager', using default: {e}")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype if device == "cuda" else torch.float32,
        ).eval()

    if device == "cuda":
        model = model.cuda()

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
            pv0, _, _ = load_image_tiles(image_paths[0], max_num_tiles=max_num_tiles)
            pv0 = pv0.to(dtype=dtype if device == "cuda" else torch.float32)
            if device == "cuda":
                pv0 = pv0.cuda()

            first_task = next(iter(tasks.values()))
            for _ in range(args.warmup):
                _ = internvl_generate(model, tokenizer, pv0, first_task["prompt"], args.max_new_tokens)
            for _, task_cfg in tasks.items():
                _ = internvl_generate(model, tokenizer, pv0, task_cfg["prompt"], args.max_new_tokens)
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
                    pixel_values, n_tiles, img_resolution = load_image_tiles(img_path, max_num_tiles=max_num_tiles)
                    pixel_values = pixel_values.to(dtype=dtype if device == "cuda" else torch.float32)
                    if device == "cuda":
                        pixel_values = pixel_values.cuda()
                except Exception as e:
                    for task_name in tasks.keys():
                        rec = {
                            "image_index": img_i,
                            "image_name": img_name,
                            "image_path": str(img_path),
                            "task": task_name,
                            "latency_ms": None,
                            "images_per_second": None,
                            "output_text": "",
                            "error": f"image_preprocess_failed: {e}",
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
                    latencies_for_task = []
                    prompt_tokens_est = None
                    response_tokens_est = None
                    txt = ""
                    err = None
                    before_sys = system_snapshot()
                    reset_cuda_peaks()

                    # Start power monitoring
                    power_monitor.start()

                    for repeat in range(args.repeats):
                        def call():
                            return internvl_generate(
                                model=model,
                                tokenizer=tokenizer,
                                pixel_values=pixel_values,
                                user_text=task_cfg["prompt"],
                                max_new_tokens=args.max_new_tokens,
                            )

                        (txt_rep, err_rep, pt_rep, rt_rep), ms = timed_call(call, device=device)
                        latencies_for_task.append(ms)
                        if repeat == 0:
                            txt, err, prompt_tokens_est, response_tokens_est = txt_rep, err_rep, pt_rep, rt_rep
                        if err_rep is None:
                            latencies.append(ms)

                    # Stop power monitoring
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
                        "prompt_tokens": prompt_tokens_est,
                        "response_tokens_est": response_tokens_est,
                        "n_tiles": int(n_tiles),
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

    # Calculate total elapsed time
    benchmark_end = time.perf_counter()
    elapsed_seconds = benchmark_end - benchmark_start
    elapsed_minutes = elapsed_seconds / 60.0

    lat_sorted = sorted(latencies)
    throughput_sorted = sorted(throughputs)
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
        "power_sample_interval_ms": args.power_sample_interval_ms if power_monitor.available else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # cleanup
    power_monitor.cleanup()

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


if __name__ == "__main__":
    main()
