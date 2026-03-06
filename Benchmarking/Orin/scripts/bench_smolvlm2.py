#!/usr/bin/env python3
# benchmark smolvlm2 (image->text) with a standardized 5-task suite
# logs per task per image to jsonl + summary json
# designed to be launched by the orchestrator inside the smolvlm2 venv
#
# jetson orin agx notes:
# - do not set TRANSFORMERS_NO_TORCHVISION=1 (smolvlm2 processor imports torchvision.transforms.v2)
# - prefer sdpa attention on cuda, fallback to eager

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_KEY = "smolvlm2"
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def parse_args():
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
    returns a set of image_path strings that already have all tasks logged.
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
        yield lst[i:i + n]


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
    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.samples = []
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.method = None
        self.available = False

        if check_tegrastats_available():
            try:
                subprocess.run([
                    "sudo", "-n", "tegrastats", "--interval", "1000", "--stop"
                ], capture_output=True, timeout=2, text=True)
                self.available = True
                self.method = "tegrastats"
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
        if self.method == "tegrastats":
            self.thread = threading.Thread(target=self._monitor_loop_tegrastats, daemon=True)
        if self.thread:
            self.thread.start()

    def stop(self):
        if not self.available:
            return None
        self.monitoring = False
        self.end_time = time.perf_counter()
        if self.method == "tegrastats" and hasattr(self, "tegrastats_process") and self.tegrastats_process:
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
        if self.method == "tegrastats":
            vdd_gpu_soc_values = [s.get("VDD_GPU_SOC", 0) for s in self.samples if isinstance(s, dict)]
            vdd_cpu_cv_values = [s.get("VDD_CPU_CV", 0) for s in self.samples if isinstance(s, dict)]
            vin_sys_5v0_values = [s.get("VIN_SYS_5V0", 0) for s in self.samples if isinstance(s, dict)]
            
            # gpu utilization (%): collected from parse_tegrastats_line via GR3D_FREQ
            gpu_util_values = [s.get("gpu_utilization_percent", None) for s in self.samples if isinstance(s, dict)]
            gpu_util_values = [v for v in gpu_util_values if v is not None]
            
            result = {}
            if vdd_gpu_soc_values:
                result["power_gpu_soc_mean_watts"] = round(sum(vdd_gpu_soc_values) / len(vdd_gpu_soc_values), 3)
            if vdd_cpu_cv_values:
                result["power_cpu_cv_mean_watts"] = round(sum(vdd_cpu_cv_values) / len(vdd_cpu_cv_values), 3)
            if vin_sys_5v0_values:
                result["power_sys_5v0_mean_watts"] = round(sum(vin_sys_5v0_values) / len(vin_sys_5v0_values), 3)
            
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
            "prompt": (
                "Write exactly 2 sentences describing the overall scene and setting "
                "(where it is and what is happening)."
            ),
            "purpose": "scene understanding",
        },
        "attributes": {
            "prompt": "Write exactly 2 sentences describing notable visual attributes (colors, lighting, materials, weather).",
            "purpose": "fine-grained perception",
        },
    }


def smolvlm2_generate(
    processor,
    model,
    image: Image.Image,
    user_text: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, str | None, int | None, int | None]:
    # important: use the model’s chat template so image placeholders match image count
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False
    }

    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
                **generation_args,
            )

        input_len = inputs["input_ids"].shape[1]
        out = out[:, input_len:]
        text = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        prompt_tokens = int(inputs["input_ids"].shape[1])
        response_tokens_est = len(processor.tokenizer.encode(text))
        return text, None, prompt_tokens, response_tokens_est
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
    if device == "cuda" and args.dtype == "bf16":
        print("running on bf16 on cuda (orin agx)")

    # Migrate away from deprecated TRANSFORMERS_CACHE: set HF_HOME and unset TRANSFORMERS_CACHE
    _tc = os.environ.get("TRANSFORMERS_CACHE")
    if _tc:
        os.environ.setdefault("HF_HOME", _tc)
        os.environ.pop("TRANSFORMERS_CACHE", None)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # attention choice: sdpa is usually the best default on jetson, but be ready to fallback
    # attn_impl = "eager"
    # if device == "cuda":
    #     attn_impl = "sdpa"

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=dtype if device == "cuda" else torch.float32,
        _attn_implementation= "eager",
    ).to(device)
    model.eval()

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
            img0 = Image.open(image_paths[0]).convert("RGB")
            first_task = next(iter(tasks.values()))
            for _ in range(args.warmup):
                _ = smolvlm2_generate(
                    processor=processor,
                    model=model,
                    image=img0,
                    user_text=first_task["prompt"],
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            for _, task_cfg in tasks.items():
                _ = smolvlm2_generate(
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
                    img = Image.open(img_path).convert("RGB")
                    img_resolution = get_image_resolution(img)
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
                            return smolvlm2_generate(
                                processor=processor,
                                model=model,
                                image=img,
                                user_text=task_cfg["prompt"],
                                device=device,
                                max_new_tokens=args.max_new_tokens,
                            )

                        (txt_rep, err_rep, prompt_tok_rep, resp_tok_rep), ms = timed_call(call, device=device)
                        latencies_for_task.append(ms)
                        if repeat == 0:
                            txt, err, prompt_tokens, response_tokens_est = txt_rep, err_rep, prompt_tok_rep, resp_tok_rep
                        if err_rep is None:
                            latencies.append(ms)

                    # Stop power monitoring
                    power_stats = power_monitor.stop()

                    after_sys = system_snapshot()
                    cuda_stats = cuda_snapshot()

                    # map jetson gpu% into cuda_stats so schema matches dell/nvml output
                    if power_stats and isinstance(power_stats, dict):
                        gpu_u = power_stats.get("gpu_utilization_percent_mean")
                        if gpu_u is not None and "gpu_utilization_percent" not in cuda_stats:
                            cuda_stats["gpu_utilization_percent"] = gpu_u

                    if err is not None:
                        errors += 1

                    mean_latency = sum(latencies_for_task) / len(latencies_for_task) if latencies_for_task else None

                    # Calculate throughput (images per second)
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
                        print(f"\n[{MODEL_KEY}] batch {batch_i} ({len(batch)} images)") if False else None
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
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # cleanup
    power_monitor.cleanup()
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


if __name__ == "__main__":
    main()
