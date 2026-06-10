
import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import ollama

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    pynvml = None
    HAS_PYNVML = False

BENCHMARKING_DIR = Path(__file__).resolve().parents[1]
EMBED_MODEL = "nomic-embed-text"
TEMPERATURE = 0.0
TOP_P = 0.9
MAX_TOKENS = 128
DEVICE_NAME = "dell_gb10"


class NvmlMonitor:
    """Background NVML sampler for discrete NVIDIA GPUs."""

    def __init__(self, interval_ms: int = 100, enabled: bool = True, device_index: int = 0):
        self.interval_ms = interval_ms
        self.device_index = device_index
        self.available = bool(enabled and HAS_PYNVML)
        self._handle = None
        self._info = {}
        self._samples = []
        self._running = False
        self._thread = None
        self._t0 = None
        self._t1 = None
        if not self.available:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._info = {
                "gpu_name": pynvml.nvmlDeviceGetName(self._handle).decode() if isinstance(pynvml.nvmlDeviceGetName(self._handle), bytes) else pynvml.nvmlDeviceGetName(self._handle),
                "driver_version": pynvml.nvmlSystemGetDriverVersion().decode() if isinstance(pynvml.nvmlSystemGetDriverVersion(), bytes) else pynvml.nvmlSystemGetDriverVersion(),
                "total_memory_mb": round(mem.total / 1048576, 1),
            }
            try:
                self._info["power_limit_watts"] = round(pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0, 2)
            except Exception:
                self._info["power_limit_watts"] = None
        except Exception as e:
            self.available = False
            self._info = {"error": str(e)}

    def start(self):
        if not self.available:
            return
        self._samples = []
        self._running = True
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        sleep_s = max(self.interval_ms / 1000.0, 0.01)
        while self._running:
            sample = {}
            try:
                sample["gpu_power_watts"] = round(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0, 2)
            except Exception:
                pass
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                sample["gpu_util_percent"] = util.gpu
                sample["mem_util_percent"] = util.memory
            except Exception:
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                sample["gpu_mem_used_mb"] = round(mem.used / 1048576, 1)
            except Exception:
                pass
            try:
                sample["gpu_sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
            except Exception:
                pass
            try:
                sample["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            self._samples.append(sample)
            time.sleep(sleep_s)

    def stop(self):
        if not self.available:
            return {"available": False, **self._info}
        self._running = False
        self._t1 = time.perf_counter()
        if self._thread:
            self._thread.join(timeout=1.0)
        duration = max((self._t1 or time.perf_counter()) - (self._t0 or time.perf_counter()), 0.0)

        def vals(key):
            return [s[key] for s in self._samples if s.get(key) is not None]

        powers = vals("gpu_power_watts")
        utils = vals("gpu_util_percent")
        mems = vals("gpu_mem_used_mb")
        clocks = vals("gpu_sm_clock_mhz")
        temps = vals("gpu_temp_c")
        avg_power = round(sum(powers) / len(powers), 2) if powers else None
        return {
            "available": True,
            **self._info,
            "gpu_power_watts_avg": avg_power,
            "gpu_power_watts_peak": round(max(powers), 2) if powers else None,
            "gpu_util_percent_avg": round(sum(utils) / len(utils), 2) if utils else None,
            "gpu_util_percent_peak": max(utils) if utils else None,
            "gpu_mem_used_mb_peak": round(max(mems), 1) if mems else None,
            "gpu_sm_clock_mhz_avg": round(sum(clocks) / len(clocks), 1) if clocks else None,
            "gpu_temp_c_peak": max(temps) if temps else None,
            "energy_joules_est": round(avg_power * duration, 2) if avg_power is not None else None,
            "sample_count": len(self._samples),
            "duration_seconds": round(duration, 3),
        }

    def shutdown(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _system_snapshot():
    if not HAS_PSUTIL:
        return {"psutil_enabled": False}
    mem = psutil.virtual_memory()
    return {
        "psutil_enabled": True,
        "cpu_percent": psutil.cpu_percent(interval=0),
        "ram_used_mb": round(mem.used / 1048576, 1),
        "ram_available_mb": round(mem.available / 1048576, 1),
        "ram_percent": mem.percent,
    }


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def _model_identity(model_tag):
    try:
        info = ollama.show(model_tag)
        get = info.get if isinstance(info, dict) else lambda k: getattr(info, k, None)
        return _json_safe({
            "digest": get("digest"),
            "modelfile": get("modelfile"),
            "details": get("details"),
        })
    except Exception as e:
        return {"error": str(e)}


def generate_response(image_path, prompt):
    msg = {"role": "user", "content": prompt, "images": [str(image_path)]}
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[msg],
        options={"temperature": TEMPERATURE, "top_p": TOP_P, "num_predict": MAX_TOKENS},
    )
    text = response["message"]["content"].strip() if isinstance(response, dict) else response.message.content.strip()

    def _g(k):
        if isinstance(response, dict):
            return response.get(k)
        return getattr(response, k, None)

    eval_duration_s = ((_g("eval_duration") or 0) / 1e9) or None
    return {
        "text": text,
        "prompt_tokens": _g("prompt_eval_count"),
        "response_tokens": _g("eval_count"),
        "eval_duration_s": eval_duration_s,
    }


def get_embedding(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"] if isinstance(response, dict) else response.embedding


def _sanitize_model(model):
    return model.replace("/", "_").replace(":", "-")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(_json_safe(record)) + "\n")


def _run_once(image_path, prompt, monitor):
    sys_before = _system_snapshot()
    monitor.start()
    t0 = time.perf_counter()
    response = generate_response(image_path, prompt)
    gen_latency_s = time.perf_counter() - t0
    hw_stats = monitor.stop()

    t1 = time.perf_counter()
    embedding = get_embedding(response["text"])
    embed_latency_s = time.perf_counter() - t1
    total_latency_s = gen_latency_s + embed_latency_s
    response_tokens = response.get("response_tokens")
    tokens_per_sec = response_tokens / gen_latency_s if response_tokens and gen_latency_s else None
    return {
        "text": response["text"],
        "text_lower": response["text"].lower(),
        "embedding": embedding,
        "prompt_tokens": response.get("prompt_tokens"),
        "response_tokens": response_tokens,
        "eval_duration_s": response.get("eval_duration_s"),
        "gen_latency_s": gen_latency_s,
        "embed_latency_s": embed_latency_s,
        "total_latency_s": total_latency_s,
        "latency_s": total_latency_s,
        "gen_latency_s_3dp": round(gen_latency_s, 3),
        "embed_latency_s_3dp": round(embed_latency_s, 3),
        "total_latency_s_3dp": round(total_latency_s, 3),
        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
        "hw_stats": hw_stats,
        "sys_before": sys_before,
        "sys_after": _system_snapshot(),
        "error": None,
    }

SAMPLE_FILE = Path("data/testsets/semantic_experiment_100.json")
OUTPUT_DIR = Path("outputs/semantic_experiment")
PROMPT = "Describe this image. List the key objects present."
NUM_RUNS = 5
MODEL_NAME = None


def _completed_ids(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    return done


def main():
    global MODEL_NAME, NUM_RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model tag. Pin Q4_K_M (e.g. 'moondream:1.8b-v2-q4_K_M').")
    parser.add_argument("--num_runs", type=int, default=NUM_RUNS)
    parser.add_argument("--no-nvml", action="store_true", help="Skip pynvml hardware capture")
    parser.add_argument("--nvml-interval-ms", type=int, default=100, help="NVML sample interval (default 100ms)")
    parser.add_argument("--gpu-index", type=int, default=0, help="Which GPU to monitor (default 0)")
    args = parser.parse_args()

    MODEL_NAME = args.model
    NUM_RUNS = args.num_runs
    sample_file = SAMPLE_FILE if SAMPLE_FILE.is_absolute() else BENCHMARKING_DIR / SAMPLE_FILE
    output_dir = OUTPUT_DIR if OUTPUT_DIR.is_absolute() else BENCHMARKING_DIR / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_id = f"{DEVICE_NAME}_{_sanitize_model(MODEL_NAME)}_{datetime.now().strftime('%Y%m%d')}"
    output_path = output_dir / f"results_{exp_id}.jsonl"

    monitor = NvmlMonitor(args.nvml_interval_ms, enabled=not args.no_nvml, device_index=args.gpu_index)
    print(f"NVML: {'on' if monitor.available else 'off'}  |  psutil: {'on' if HAS_PSUTIL else 'off'}")
    if monitor._info.get("gpu_name"):
        print(f"  GPU: {monitor._info.get('gpu_name')} ({monitor._info.get('total_memory_mb')} MB total, driver {monitor._info.get('driver_version')})")
    print(f"Fetching model identity for {MODEL_NAME}...")
    model_identity = _model_identity(MODEL_NAME)
    print(f"  digest: {model_identity.get('digest', 'unknown')}")

    samples = _load_json(sample_file)
    completed = _completed_ids(output_path)
    if completed:
        print(f"Resuming - {len(completed)} images already done")
    remaining = [s for s in samples if s.get("image_id") not in completed]
    print(f"Images to process: {len(remaining)} / {len(samples)}")
    print(f"Device: {DEVICE_NAME}  |  Model: {MODEL_NAME}  |  Runs/image: {NUM_RUNS}")
    print(f"Experiment ID: {exp_id}")
    print(f"Output: {output_path}\n")

    try:
        print("Warming up model...")
        generate_response(remaining[0]["full_path"] if remaining else samples[0]["full_path"], "warmup text")
        print("Warmup complete.\n")
    except Exception as e:
        print(f"Warmup failed: {e}")
        print("Check that Ollama is running and the model is pulled.")

    started = time.perf_counter()
    for idx, sample in enumerate(remaining, 1):
        image_path = Path(sample["full_path"])
        if not image_path.is_absolute():
            image_path = BENCHMARKING_DIR / image_path
            sample["full_path"] = str(image_path)
        if not image_path.exists():
            print(f"[{idx}] SKIP - not found: {image_path}")
            continue
        print(f"[{idx}] image_id={sample.get('image_id')}")
        outputs = []
        for run in range(1, NUM_RUNS + 1):
            for attempt in range(1, 3):
                try:
                    out = _run_once(image_path, PROMPT, monitor)
                    out["run"] = run
                    outputs.append(out)
                    power = out["hw_stats"].get("gpu_power_watts_avg")
                    power_s = f"  {power:.1f}W" if power is not None else ""
                    tps = f"  {out['tokens_per_sec']:.2f} tok/s" if out.get("tokens_per_sec") else ""
                    print(f"  run {run}{power_s}{tps} ({out['gen_latency_s']:.2f}s)  \"{out['text'][:80]}...\"")
                    break
                except Exception as e:
                    print(f"  run {run} attempt {attempt} failed: {e}")
                    if attempt == 2:
                        outputs.append({"run": run, "text": "", "error": str(e)})
                    else:
                        print("    retrying once...")
        record = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "device": DEVICE_NAME,
            "model": MODEL_NAME,
            "model_identity": model_identity,
            "image_id": sample.get("image_id"),
            "file_name": sample.get("file_name"),
            "full_path": sample.get("full_path"),
            "expected_entities": sample.get("expected_entities", []),
            "prompt": PROMPT,
            "outputs": outputs,
        }
        _append_jsonl(output_path, record)
        elapsed = time.perf_counter() - started
        eta_min = (elapsed / idx) * (len(remaining) - idx) / 60 if idx else 0
        print(f"  -> saved  |  ETA: {eta_min:.1f} min remaining\n")

    monitor.shutdown()
    print(f"Done! Results saved to {output_path}")
    print(f"Next: python check_hallucination.py --input {output_path}")
    print("Then: python analyze_results.py --thor ... --dell ...")


if __name__ == "__main__":
    main()
