"""
run_experiment_thor_5task.py  (Jetson Thor, 5-task workload, tegrastats capture)

Thor counterpart to run_experiment_dell_5task.py. Same JSONL schema, same
5-task workload (caption_brief, objects_and_counts, spatial_relationships,
scene_context, attributes), but uses tegrastats instead of pynvml for
hardware capture. decide_framing.py and check_hallucination.py read both
device's outputs uniformly because the hw_stats schemas align.

What's captured per generation:
  - GPU power (W) — VDD_GPU rail via tegrastats
  - GPU clock (MHz) — GR3D_FREQ
  - CPU utilization (%) — averaged across cores
  - Unified RAM (MB) — Jetson shares CPU/GPU memory
  - Energy estimate (J) = avg power × wall-clock duration

Captured once at startup:
  - Model digest + Ollama modelfile excerpt
  - Tegrastats availability and interval

Usage:
    pip install ollama psutil
    ollama pull nomic-embed-text
    ollama pull <your pinned model tag>
    NOPASSWD sudo for tegrastats (needed by the background sampler)

    python run_experiment_thor_5task.py --device thor --model moondream:1.8b-v2-q4_K_M
    python run_experiment_thor_5task.py --model moondream:1.8b-v2-q4_K_M
    python run_experiment_thor_5task.py --model qwen3-vl:2b-instruct-q4_K_M

    # Subset of tasks
    python run_experiment_thor_5task.py --model moondream:1.8b-v2-q4_K_M \\
        --tasks caption_brief objects_and_counts

    # Resume a partially-completed run
    python run_experiment_thor_5task.py --model moondream:1.8b-v2-q4_K_M --resume

    # Skip hardware capture (still produces analyzable JSONL)
    python run_experiment_thor_5task.py --model moondream:1.8b-v2-q4_K_M --no-tegrastats
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
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


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_FILE = "data/testsets/semantic_experiment_100.json"
OUTPUT_DIR          = "outputs/semantic_experiment_5task"
EMBED_MODEL         = "nomic-embed-text"
NUM_RUNS            = 5

TEMPERATURE = 0.0
TOP_P       = 1.0
MAX_TOKENS  = 256

DEVICE_NAME = "thor"

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
parser.add_argument("--device", default=DEVICE_NAME,
                    help=f"Device label for output records/files (default: {DEVICE_NAME})")
parser.add_argument("--model", required=True,
                    help="Ollama model tag (pin Q4_K_M)")
parser.add_argument("--sample_file", default=DEFAULT_SAMPLE_FILE,
                    help=f"Path to image testset JSON (default: {DEFAULT_SAMPLE_FILE})")
parser.add_argument("--num_runs", type=int, default=NUM_RUNS)
parser.add_argument("--tasks", nargs="+", default=None,
                    help="Optional subset of task names to run")
parser.add_argument("--resume", action="store_true",
                    help="Skip (image_id, task) pairs already in the output file")
parser.add_argument("--no-tegrastats", action="store_true",
                    help="Skip tegrastats hardware capture (debugging only)")
parser.add_argument("--tegrastats-interval-ms", type=int, default=100,
                    help="Tegrastats sample interval (default 100ms)")
args = parser.parse_args()

DEVICE_NAME = args.device
MODEL_NAME  = args.model
SAMPLE_FILE = args.sample_file
NUM_RUNS    = args.num_runs

if args.tasks:
    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        print(f"ERROR: unknown task names: {unknown}")
        print(f"Available: {list(TASKS.keys())}")
        sys.exit(1)
    TASKS = {k: TASKS[k] for k in args.tasks}

SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_").replace(":", "-")
EXPERIMENT_DATE = time.strftime("%Y%m%d")

EXPERIMENT_ID = hashlib.md5(
    (f"{MODEL_NAME}|{sorted(TASKS.keys())}|{TEMPERATURE}|{TOP_P}|"
     f"{MAX_TOKENS}|{EMBED_MODEL}").encode()
).hexdigest()[:10]

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"results_{DEVICE_NAME}_{EXPERIMENT_DATE}_{SAFE_MODEL_NAME}_5task_{EXPERIMENT_ID}.jsonl",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Tegrastats parsing ────────────────────────────────────────────────────────
def _parse_tegrastats_line(line: str) -> dict:
    """Parse one tegrastats line. Returns dict of available fields."""
    out = {"power_rails": {}, "gpu_freq_mhz": None, "cpu_util_pct": None,
           "ram_used_mb": None}

    m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if m:
        out["ram_used_mb"] = int(m.group(1))

    for name, mw in re.findall(r"(\w+)\s+(\d+)mW/\d+mW", line):
        out["power_rails"][name] = round(int(mw) / 1000.0, 3)

    m = re.search(r"GR3D_FREQ\s+@\[([^\]]+)\]", line)
    if m:
        freqs = [int(f) for f in m.group(1).split(",") if f.strip().isdigit()]
        if freqs:
            out["gpu_freq_mhz"] = round(sum(freqs)/len(freqs), 1)

    m = re.search(r"CPU \[([^\]]+)\]", line)
    if m:
        pcts = re.findall(r"(\d+)%", m.group(1))
        if pcts:
            out["cpu_util_pct"] = round(sum(int(p) for p in pcts)/len(pcts), 1)

    return out


def _tegrastats_available() -> bool:
    try:
        return subprocess.run(["which", "tegrastats"],
                              capture_output=True, timeout=2).returncode == 0
    except Exception:
        return False


class TegrastatsMonitor:
    """Background tegrastats sampler. Auto-disables if tegrastats missing."""

    def __init__(self, interval_ms: int = 100, enabled: bool = True):
        self.interval_ms = interval_ms
        self.available = enabled and _tegrastats_available()
        self._samples = []
        self._running = False
        self._thread = None
        self._process = None
        self._t0 = self._t1 = None

    def start(self):
        if not self.available:
            return
        self._samples = []
        self._running = True
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            self._process = subprocess.Popen(
                ["sudo", "tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
            for line in self._process.stdout:
                if not self._running:
                    break
                parsed = _parse_tegrastats_line(line)
                if parsed["power_rails"] or parsed["ram_used_mb"] is not None:
                    self._samples.append(parsed)
        except Exception:
            pass
        finally:
            self._kill()

    def _kill(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception:
                pass
            self._process = None

    def stop(self):
        if not self.available:
            return None
        self._running = False
        self._t1 = time.perf_counter()
        self._kill()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return None

        duration = self._t1 - self._t0

        rails: dict = {}
        for s in self._samples:
            for rail, w in s["power_rails"].items():
                rails.setdefault(rail, []).append(w)
        rails_summary = {
            r: {"avg": round(sum(v)/len(v), 3), "peak": round(max(v), 3)}
            for r, v in rails.items()
        }

        gpu_rail = (rails_summary.get("VDD_GPU") or
                    rails_summary.get("VDD_GPU_SOC") or
                    rails_summary.get("VDD_GPU_CV"))
        gpu_avg = gpu_rail["avg"]  if gpu_rail else None
        gpu_peak = gpu_rail["peak"] if gpu_rail else None

        gpu_freqs = [s["gpu_freq_mhz"] for s in self._samples
                     if s["gpu_freq_mhz"] is not None]
        cpu_utils = [s["cpu_util_pct"] for s in self._samples
                     if s["cpu_util_pct"] is not None]
        rams = [s["ram_used_mb"] for s in self._samples
                if s["ram_used_mb"] is not None]

        return {
            "method": "tegrastats",
            "power_rails": rails_summary,
            "gpu_power_watts_avg":  gpu_avg,
            "gpu_power_watts_peak": gpu_peak,
            "gpu_freq_mhz_mean":    round(sum(gpu_freqs)/len(gpu_freqs), 1)
                                    if gpu_freqs else None,
            "gpu_freq_mhz_peak":    max(gpu_freqs) if gpu_freqs else None,
            "cpu_util_pct_mean":    round(sum(cpu_utils)/len(cpu_utils), 1)
                                    if cpu_utils else None,
            "ram_used_mb_mean":     round(sum(rams)/len(rams), 1) if rams else None,
            "ram_used_mb_peak":     max(rams) if rams else None,
            "energy_joules_est":    round(gpu_avg * duration, 2) if gpu_avg else None,
            "sample_count":         len(self._samples),
            "duration_seconds":     round(duration, 3),
        }


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


def _json_safe(value):
    """Convert client/library objects into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    return str(value)


# ── Model identity ────────────────────────────────────────────────────────────
def _model_identity(model_tag: str) -> dict:
    try:
        info = ollama.show(model_tag)
        get = (lambda k: info.get(k)) if isinstance(info, dict) \
              else (lambda k: getattr(info, k, None))
        return {
            "digest":    get("digest") or get("modelfile_digest") or "unknown",
            "modelfile": (get("modelfile") or "")[:1000],
            "details":   _json_safe(get("details")),
        }
    except Exception as e:
        return {"digest": f"error: {e}", "modelfile": "", "details": None}


def _installed_ollama_models() -> list[str]:
    try:
        listed = ollama.list()
        models = listed.get("models") if isinstance(listed, dict) \
                 else getattr(listed, "models", [])
        names = []
        for model in models:
            name = model.get("name") if isinstance(model, dict) \
                   else getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(name)
        return sorted(names)
    except Exception:
        return []


# ── Build runtime block ───────────────────────────────────────────────────────
monitor = TegrastatsMonitor(
    interval_ms=args.tegrastats_interval_ms,
    enabled=not args.no_tegrastats,
)

print(f"Tegrastats: {'on' if monitor.available else 'off'}  "
      f"|  psutil: {'on' if HAS_PSUTIL else 'off'}")

print(f"Fetching model identity for {MODEL_NAME}...")
MODEL_IDENTITY = _model_identity(MODEL_NAME)
print(f"  digest: {MODEL_IDENTITY['digest']}")
if str(MODEL_IDENTITY["digest"]).startswith("error:"):
    print(f"ERROR: Ollama model not found or unavailable: {MODEL_NAME}")
    installed = _installed_ollama_models()
    if installed:
        print("Installed Ollama models:")
        for name in installed:
            print(f"  - {name}")
    print("  Pull the model first or pass one of the installed model tags above.")
    sys.exit(1)

RUNTIME_INFO = {
    "python":                  sys.version.split()[0],
    "platform":                platform.platform(),
    "host":                    platform.node(),
    "ollama_python_package":   getattr(ollama, "__version__", "unknown"),
    "model_digest":            MODEL_IDENTITY["digest"],
    "model_modelfile_excerpt": MODEL_IDENTITY["modelfile"],
    "model_details":           MODEL_IDENTITY["details"],
    "tegrastats_enabled":      monitor.available,
    "tegrastats_interval_ms":  args.tegrastats_interval_ms if monitor.available else None,
    "psutil_enabled":          HAS_PSUTIL,
}


# ── Resume support ────────────────────────────────────────────────────────────
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
    print(f"  Pass --sample_file with a valid path")
    sys.exit(1)

with open(SAMPLE_FILE) as f:
    samples = json.load(f)

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
# Use the longest task prompt for warmup so the model is loaded under realistic
# conditions. Pick from whatever subset of TASKS the user requested.
warmup_task = next(
    (t for t in ["spatial_relationships", "scene_context", "attributes",
                 "objects_and_counts", "caption_brief"] if t in TASKS),
    list(TASKS.keys())[0],
)
print(f"Warming up model with '{warmup_task}' prompt...")
try:
    _ = generate_response(samples[0]["full_path"], TASKS[warmup_task]["prompt"])
    _ = get_embedding("warmup text")
    print("Warmup complete.\n")
except Exception as e:
    print(f"Warmup failed: {e}")
    print("Check that Ollama is running and the model is pulled.")
    sys.exit(1)


# ── Main loop ─────────────────────────────────────────────────────────────────
start_time = time.time()

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

        out_f.write(json.dumps(_json_safe(record)) + "\n")
        out_f.flush()

        elapsed_total = time.time() - start_time
        done = idx + 1
        rate = done / elapsed_total
        eta  = (len(work_items) - done) / rate if rate > 0 else 0
        eta_h = eta / 3600
        print(f"  → saved  |  ETA: {eta_h:.1f} hours remaining\n")

print(f"Done! Results saved to {OUTPUT_FILE}")
print(f"Next: python check_hallucination.py --input {OUTPUT_FILE}")
print(f"Then: per-task analysis (analyze by 'task' field for cross-task comparison)")
