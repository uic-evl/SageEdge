"""
run_experiment_dell_5task.py  (Dell GB10, 5-task workload, pynvml capture)

Combines:
  - The 5-task workload from collect_ollama.py (caption_brief, objects_and_counts,
    spatial_relationships, scene_context, attributes)
  - The pynvml hardware capture from run_experiment_dell.py (GPU power, util,
    memory, clock, temperature, energy)

JSONL shape: each record is one (image, task) pair, containing N runs.
Image-level identifiers (image_id, file_name, expected_entities) are repeated
across the 5 task records for that image, with `task` and `prompt` distinguishing
them. This means decide_framing.py and check_hallucination.py work without
modification — they'll just see (image_id, task) tuples instead of bare image_ids.

Default workload: 5 tasks × 500 images × 5 runs × 1 model = 12,500 generations.
Across 5 models that's 62,500 generations per device — overnight scale on Dell GB10.

Usage:
    pip install ollama pynvml psutil
    ollama pull nomic-embed-text
    ollama pull <your pinned model tag>

    python run_experiment_dell_5task.py --model moondream:1.8b-v2-q4_K_M
    python run_experiment_dell_5task.py --model qwen3-vl:2b-instruct-q4_K_M

    # Run only a subset of tasks
    python run_experiment_dell_5task.py --model moondream:1.8b-v2-q4_K_M \\
        --tasks caption_brief objects_and_counts

    # Resume a partially-completed run (skips (image_id, task) pairs already done)
    python run_experiment_dell_5task.py --model moondream:1.8b-v2-q4_K_M --resume

    # Use a 100-image testset instead of 500
    python run_experiment_dell_5task.py --model moondream:1.8b-v2-q4_K_M \\
        --sample_file data/testsets/semantic_experiment_100.json
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import ollama

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


# ── Config ────────────────────────────────────────────────────────────────────
BENCHMARKING_DIR    = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_FILE = BENCHMARKING_DIR / "data/testsets/semantic_experiment_500.json"
OUTPUT_DIR          = BENCHMARKING_DIR / "outputs/semantic_experiment_5task"
EMBED_MODEL         = "nomic-embed-text"
NUM_RUNS            = 5

TEMPERATURE = 0.0
TOP_P       = 1.0
MAX_TOKENS  = 256

DEVICE_NAME = "dell_gb10"

TASKS = {
    "caption_brief": {
        "prompt":  "Write one detailed sentence describing the image.",
        "purpose": "low-latency captioning",
    },
    "objects_and_counts": {
        "prompt":  "List up to 8 main objects with approximate counts. "
                   "Use format 'object: count' on separate lines.",
        "purpose": "object recognition",
    },
    "spatial_relationships": {
        "prompt":  "Write 2-3 sentences describing spatial relationships "
                   "between the main objects (left/right, foreground/"
                   "background, near/far).",
        "purpose": "spatial grounding",
    },
    "scene_context": {
        "prompt":  "Write exactly 2 sentences describing the overall scene "
                   "and setting (where it is and what is happening).",
        "purpose": "scene understanding",
    },
    "attributes": {
        "prompt":  "Write exactly 2 sentences describing notable visual "
                   "attributes (colors, lighting, materials, weather).",
        "purpose": "fine-grained perception",
    },
}


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="Ollama model tag (pin Q4_K_M)")
parser.add_argument("--sample_file", default=DEFAULT_SAMPLE_FILE,
                    help=f"Path to image testset JSON (default: {DEFAULT_SAMPLE_FILE})")
parser.add_argument("--num_runs", type=int, default=NUM_RUNS)
parser.add_argument("--tasks", nargs="+", default=None,
                    help="Optional subset of task names to run")
parser.add_argument("--resume", action="store_true",
                    help="Skip (image_id, task) pairs already in the output file")
parser.add_argument("--no-nvml", action="store_true",
                    help="Skip pynvml hardware capture")
parser.add_argument("--nvml-interval-ms", type=int, default=100,
                    help="NVML sample interval (default 100ms)")
parser.add_argument("--gpu-index", type=int, default=0,
                    help="Which GPU to monitor (default 0)")
args = parser.parse_args()

MODEL_NAME  = args.model
SAMPLE_FILE = Path(args.sample_file)
if not SAMPLE_FILE.is_absolute():
    SAMPLE_FILE = Path.cwd() / SAMPLE_FILE
NUM_RUNS    = args.num_runs

# Filter tasks if subset requested
if args.tasks:
    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        print(f"ERROR: unknown task names: {unknown}")
        print(f"Available: {list(TASKS.keys())}")
        sys.exit(1)
    TASKS = {k: TASKS[k] for k in args.tasks}

SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_").replace(":", "-")
EXPERIMENT_DATE = time.strftime("%Y%m%d")

# Experiment ID changes if you change tasks or generation params, so re-runs
# with different config produce different output files.
EXPERIMENT_ID = hashlib.md5(
    (f"{MODEL_NAME}|{sorted(TASKS.keys())}|{TEMPERATURE}|{TOP_P}|"
     f"{MAX_TOKENS}|{EMBED_MODEL}").encode()
).hexdigest()[:10]

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"results_{DEVICE_NAME}_{EXPERIMENT_DATE}_{SAFE_MODEL_NAME}_5task_{EXPERIMENT_ID}.jsonl",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── NVML monitor ──────────────────────────────────────────────────────────────
class NvmlMonitor:
    """Background NVML sampler (same shape as run_experiment_dell.py)."""

    def __init__(self, interval_ms: int = 100, enabled: bool = True,
                 device_index: int = 0):
        self.interval_ms  = interval_ms
        self.device_index = device_index
        self.available    = enabled and HAS_PYNVML
        self._handle      = None
        self._info        = {}
        self._samples     = []
        self._running     = False
        self._thread      = None
        self._t0 = self._t1 = None

        if not self.available:
            return

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

            name = pynvml.nvmlDeviceGetName(self._handle)
            self._info["gpu_name"] = (name.decode() if isinstance(name, bytes)
                                      else name)

            try:
                drv = pynvml.nvmlSystemGetDriverVersion()
                self._info["driver_version"] = (drv.decode()
                                                if isinstance(drv, bytes) else drv)
            except Exception:
                self._info["driver_version"] = None

            try:
                mem_total = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total
                self._info["total_memory_mb"] = round(mem_total / 1024**2, 1)
            except Exception:
                self._info["total_memory_mb"] = None

            try:
                cap = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)
                self._info["power_limit_watts"] = round(cap / 1000.0, 2)
            except Exception:
                self._info["power_limit_watts"] = None

        except Exception as e:
            print(f"  NVML init failed: {e}")
            self.available = False

    def start(self):
        if not self.available:
            return
        self._samples = []
        self._running = True
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        period = self.interval_ms / 1000.0
        h = self._handle
        while self._running:
            sample = {}
            try:
                sample["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                sample["power_w"] = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                sample["gpu_util_pct"] = util.gpu
                sample["mem_util_pct"] = util.memory
            except Exception:
                sample["gpu_util_pct"] = None
                sample["mem_util_pct"] = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                sample["mem_used_mb"] = round(mem.used / 1024**2, 1)
            except Exception:
                sample["mem_used_mb"] = None
            try:
                sample["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                    h, pynvml.NVML_CLOCK_SM)
            except Exception:
                sample["sm_clock_mhz"] = None
            try:
                sample["temp_c"] = pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                sample["temp_c"] = None

            self._samples.append(sample)
            time.sleep(period)

    def stop(self):
        if not self.available:
            return None
        self._running = False
        self._t1 = time.perf_counter()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return None

        duration = self._t1 - self._t0

        def _vals(key):
            return [s[key] for s in self._samples if s.get(key) is not None]

        powers   = _vals("power_w")
        utils    = _vals("gpu_util_pct")
        mems     = _vals("mem_used_mb")
        clocks   = _vals("sm_clock_mhz")
        temps    = _vals("temp_c")

        gpu_avg  = round(sum(powers) / len(powers), 3) if powers else None
        gpu_peak = round(max(powers), 3) if powers else None

        return {
            "method":              "pynvml",
            "gpu_name":            self._info.get("gpu_name"),
            "driver_version":      self._info.get("driver_version"),
            "total_memory_mb":     self._info.get("total_memory_mb"),
            "power_limit_watts":   self._info.get("power_limit_watts"),
            "gpu_power_watts_avg":  gpu_avg,
            "gpu_power_watts_peak": gpu_peak,
            "gpu_freq_mhz_mean":    round(sum(clocks)/len(clocks), 1) if clocks else None,
            "gpu_freq_mhz_peak":    max(clocks) if clocks else None,
            "gpu_util_pct_mean":    round(sum(utils)/len(utils), 1) if utils else None,
            "gpu_util_pct_peak":    max(utils) if utils else None,
            "ram_used_mb_mean":     round(sum(mems)/len(mems), 1) if mems else None,
            "ram_used_mb_peak":     max(mems) if mems else None,
            "temp_c_mean":          round(sum(temps)/len(temps), 1) if temps else None,
            "temp_c_peak":          max(temps) if temps else None,
            "energy_joules_est":    round(gpu_avg * duration, 2) if gpu_avg else None,
            "sample_count":         len(self._samples),
            "duration_seconds":     round(duration, 3),
        }

    def shutdown(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _system_snapshot():
    if not HAS_PSUTIL:
        return None
    mem = psutil.virtual_memory()
    return {
        "cpu_percent":      psutil.cpu_percent(interval=0),
        "ram_used_mb":      round(mem.used      / 1024**2, 1),
        "ram_available_mb": round(mem.available / 1024**2, 1),
        "ram_percent":      mem.percent,
    }


# ── Model identity ────────────────────────────────────────────────────────────
def _json_safe(value):
    """Convert SDK response objects into values json.dumps can encode."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    return str(value)


def _model_identity(model_tag: str) -> dict:
    try:
        info = ollama.show(model_tag)
        get = (lambda k: info.get(k)) if isinstance(info, dict) \
              else (lambda k: getattr(info, k, None))
        return {
            "digest":    _json_safe(get("digest") or
                                     get("modelfile_digest") or "unknown"),
            "modelfile": str(get("modelfile") or "")[:1000],
            "details":   _json_safe(get("details")),
        }
    except Exception as e:
        return {"digest": f"error: {e}", "modelfile": "", "details": None}


# ── Build runtime block ───────────────────────────────────────────────────────
monitor = NvmlMonitor(
    interval_ms=args.nvml_interval_ms,
    enabled=not args.no_nvml,
    device_index=args.gpu_index,
)

print(f"NVML: {'on' if monitor.available else 'off'}  "
      f"|  psutil: {'on' if HAS_PSUTIL else 'off'}")
if monitor.available:
    print(f"  GPU: {monitor._info.get('gpu_name')}  "
          f"({monitor._info.get('total_memory_mb')} MB total, "
          f"driver {monitor._info.get('driver_version')})")

print(f"Fetching model identity for {MODEL_NAME}...")
MODEL_IDENTITY = _model_identity(MODEL_NAME)
print(f"  digest: {MODEL_IDENTITY['digest']}")

RUNTIME_INFO = {
    "python":                  sys.version.split()[0],
    "platform":                platform.platform(),
    "host":                    platform.node(),
    "ollama_python_package":   getattr(ollama, "__version__", "unknown"),
    "model_digest":            MODEL_IDENTITY["digest"],
    "model_modelfile_excerpt": MODEL_IDENTITY["modelfile"],
    "model_details":           MODEL_IDENTITY["details"],
    "nvml_enabled":            monitor.available,
    "nvml_interval_ms":        args.nvml_interval_ms if monitor.available else None,
    "gpu_name":                monitor._info.get("gpu_name") if monitor.available else None,
    "driver_version":          monitor._info.get("driver_version") if monitor.available else None,
    "total_gpu_memory_mb":     monitor._info.get("total_memory_mb") if monitor.available else None,
    "power_limit_watts":       monitor._info.get("power_limit_watts") if monitor.available else None,
    "psutil_enabled":          HAS_PSUTIL,
}


# ── Resume support ────────────────────────────────────────────────────────────
# Resume granularity: (image_id, task) tuples. If a pair is in the output file,
# we skip it. Lets you Ctrl+C and restart without losing progress.
completed_pairs = set()
if args.resume and os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                completed_pairs.add((rec["image_id"], rec.get("task")))
            except json.JSONDecodeError:
                pass
    print(f"Resuming — {len(completed_pairs)} (image, task) pairs already done")


# ── Load samples ──────────────────────────────────────────────────────────────
if not os.path.exists(SAMPLE_FILE):
    print(f"ERROR: sample file not found: {SAMPLE_FILE}")
    print(f"  Run prepare_sample.py to generate it, or pass --sample_file")
    monitor.shutdown()
    sys.exit(1)

with open(SAMPLE_FILE) as f:
    samples = json.load(f)

for sample in samples:
    image_path = Path(sample["full_path"])
    if not image_path.is_absolute():
        sample["full_path"] = str(BENCHMARKING_DIR / image_path)

# Build the (image, task) work list, skipping already-completed pairs
work_items = []
for sample in samples:
    for task_name in TASKS:
        if (sample["image_id"], task_name) not in completed_pairs:
            work_items.append((sample, task_name))

total_pairs = len(samples) * len(TASKS)
print(f"Images: {len(samples)}  |  Tasks: {len(TASKS)}  |  "
      f"Pairs to process: {len(work_items)} / {total_pairs}")
print(f"Device: {DEVICE_NAME}  |  Model: {MODEL_NAME}  |  Runs/pair: {NUM_RUNS}")
print(f"Total generations to do: {len(work_items) * NUM_RUNS:,}")
print(f"Experiment ID: {EXPERIMENT_ID}")
print(f"Output: {OUTPUT_FILE}\n")


# ── Inference helpers ─────────────────────────────────────────────────────────
def generate_response(image_path: str, prompt: str) -> dict:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        options={
            "temperature": TEMPERATURE,
            "top_p":       TOP_P,
            "num_predict": MAX_TOKENS,
        },
    )
    msg = response["message"] if isinstance(response, dict) else response.message
    text = (msg["content"] if isinstance(msg, dict) else msg.content).strip()

    def _g(k):
        return response.get(k) if isinstance(response, dict) \
               else getattr(response, k, None)

    return {
        "text":            text,
        "prompt_tokens":   _g("prompt_eval_count"),
        "response_tokens": _g("eval_count"),
        "eval_duration_s": (_g("eval_duration") or 0) / 1e9 or None,
    }


def get_embedding(text: str) -> list:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"] if isinstance(response, dict) \
           else response.embedding


# ── Warmup ────────────────────────────────────────────────────────────────────
# Warm up with the longest task prompt so the model is loaded under realistic
# conditions before timing begins.
print("Warming up model...")
try:
    warmup_task = max(TASKS.values(), key=lambda t: len(t["prompt"]))
    _ = generate_response(samples[0]["full_path"], warmup_task["prompt"])
    _ = get_embedding("warmup text")
    print("Warmup complete.\n")
except Exception as e:
    print(f"Warmup failed: {e}")
    print("Check that Ollama is running and the model is pulled.")
    monitor.shutdown()
    sys.exit(1)


# ── Main loop ─────────────────────────────────────────────────────────────────
start_time = time.time()

try:
    with open(OUTPUT_FILE, "a") as out_f:
        for idx, (sample, task_name) in enumerate(work_items):
            image_id   = sample["image_id"]
            image_path = sample["full_path"]
            expected   = sample["expected_entities"]
            task_cfg   = TASKS[task_name]
            prompt     = task_cfg["prompt"]

            if not os.path.exists(image_path):
                print(f"[{idx+1}/{len(work_items)}] SKIP — not found: {image_path}")
                continue

            print(f"[{idx+1}/{len(work_items)}] image_id={image_id}  "
                  f"task={task_name}  ({Path(image_path).name})")

            run_outputs = []
            for run_i in range(NUM_RUNS):
                text = emb = None
                gen_latency = emb_latency = total_latency = None
                prompt_tokens = response_tokens = None
                tokens_per_sec = None
                error = None

                sys_before = _system_snapshot()
                monitor.start()

                for attempt in range(2):
                    try:
                        t_gen = time.time()
                        gen_result = generate_response(image_path, prompt)
                        gen_latency = time.time() - t_gen

                        text = gen_result["text"]
                        prompt_tokens   = gen_result["prompt_tokens"]
                        response_tokens = gen_result["response_tokens"]
                        if response_tokens and gen_result["eval_duration_s"]:
                            tokens_per_sec = round(
                                response_tokens / gen_result["eval_duration_s"], 2)

                        t_emb = time.time()
                        emb = get_embedding(text)
                        emb_latency = time.time() - t_emb

                        total_latency = gen_latency + emb_latency
                        break
                    except Exception as e:
                        error = str(e)
                        print(f"  run {run_i+1}/{NUM_RUNS} attempt {attempt+1} "
                              f"failed: {e}")
                        if attempt == 0:
                            print("    retrying once...")
                            time.sleep(2)

                hw_stats = monitor.stop()
                sys_after = _system_snapshot()

                if total_latency is not None:
                    preview = (text or "").replace("\n", " ")[:50]
                    pw_str = ""
                    if hw_stats and hw_stats.get("gpu_power_watts_avg"):
                        pw_str = f"  {hw_stats['gpu_power_watts_avg']:.1f}W"
                    tps_str = f"  {tokens_per_sec:.1f} tok/s" if tokens_per_sec else ""
                    print(f"  run {run_i+1}/{NUM_RUNS}  "
                          f"({total_latency:.2f}s){pw_str}{tps_str}  "
                          f'"{preview}..."')
                else:
                    print(f"  run {run_i+1}/{NUM_RUNS}  FAILED after retries")

                run_outputs.append({
                    "run":             run_i,
                    "text":            text,
                    "text_lower":      text.lower() if text is not None else None,
                    "embedding":       emb,
                    "gen_latency_s":   gen_latency,
                    "embed_latency_s": emb_latency,
                    "total_latency_s": total_latency,
                    "latency_s":       total_latency,
                    "gen_latency_s_3dp":   round(gen_latency, 3) if gen_latency else None,
                    "embed_latency_s_3dp": round(emb_latency, 3) if emb_latency else None,
                    "total_latency_s_3dp": round(total_latency, 3) if total_latency else None,
                    "prompt_tokens":   prompt_tokens,
                    "response_tokens": response_tokens,
                    "tokens_per_sec":  tokens_per_sec,
                    "hw_stats":        hw_stats,
                    "sys_before":      sys_before,
                    "sys_after":       sys_after,
                    "error":           error,
                })

            image_total_latency = sum(
                (o["total_latency_s"] for o in run_outputs
                 if o["total_latency_s"] is not None),
                0.0,
            )

            record = {
                "image_id":           image_id,
                "file_name":          sample["file_name"],
                "device":             DEVICE_NAME,
                "model":              MODEL_NAME,
                "embedding_model":    EMBED_MODEL,
                "experiment_id":      EXPERIMENT_ID,
                "task":               task_name,
                "task_purpose":       task_cfg["purpose"],
                "prompt":             prompt,
                "expected_entities":  expected,
                "generation_options": {
                    "temperature": TEMPERATURE,
                    "top_p":       TOP_P,
                    "max_tokens":  MAX_TOKENS,
                },
                "runtime":            RUNTIME_INFO,
                "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
                "image_total_latency_s": round(image_total_latency, 3),
                "outputs":            run_outputs,
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            # ETA based on (image, task) pair rate
            elapsed_total = time.time() - start_time
            done = idx + 1
            rate = done / elapsed_total
            eta  = (len(work_items) - done) / rate if rate > 0 else 0
            eta_h = eta / 3600
            print(f"  → saved  |  ETA: {eta_h:.1f} hours remaining\n")
finally:
    monitor.shutdown()

print(f"Done! Results saved to {OUTPUT_FILE}")
print(f"Next: python check_hallucination.py --input {OUTPUT_FILE}")
print(f"Then: per-task analysis (analyze by 'task' field for cross-task comparison)")
