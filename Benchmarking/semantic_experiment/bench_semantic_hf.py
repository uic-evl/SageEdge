#!/usr/bin/env python3
"""
bench_semantic_hf.py

Unified HF-transformers benchmark for compact VLMs on edge devices.
Runs the 5-task suite over 500 COCO images with N repeats per (image, task).
Auto-detects device (Thor JP7 vs Dell with pynvml) and uses the right telemetry.

Outputs JSONL records in the schema expected by analyze_results.py and
check_hallucination.py:

    {
      "model": "...", "model_key": "...", "device": "...", "dtype": "...",
      "image_id": int, "image_path": str, "image_name": str,
      "image_width": int, "image_height": int,
      "task": str, "task_prompt": str, "task_purpose": str,
      "outputs": [{"run": int, "text": str, "gen_latency_s": float,
                   "input_len": int, "gen_len": int, "error": str|null}, ...],
      "hw_stats": {...},
      "expected_entities": null,    # filled in by post-hoc script
      "timestamp": str, "run_group": str
    }

NO embeddings, NO entity recall, NO COCO annotation loading happens in this
script. Those are post-hoc in compute_quality_metrics.py to avoid contaminating
hardware measurements and to allow re-analysis with different metrics later.

Usage (per-model env activated separately):

    source envs/moondream-env/bin/activate
    python bench_semantic_hf.py \\
        --model moondream \\
        --manifest data/testsets/coco_val2017_500.txt \\
        --output_dir outputs/semantic_extension \\
        --run_group "moondream_thor_$(date +%Y%m%d_%H%M)" \\
        --repeats 5 \\
        --max_new_tokens 200 \\
        --dtype bf16

Resume:
    Add --resume to skip (image, task) pairs already complete in runs.jsonl.

Device override (rare, mostly testing):
    --device thor | dell | orin
"""

import argparse
import gc
import json
import os
import platform
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from PIL import Image

# avoid torchvision on jetson / thor
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


# ──────────────────────────────────────────────────────────────────────────────
# Device detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    """Auto-detect which device this script is running on."""
    nv_tegra = Path("/etc/nv_tegra_release")
    if nv_tegra.exists():
        text = nv_tegra.read_text()
        if "R38" in text:
            return "thor"   # JetPack 7
        if "R36" in text:
            return "orin"   # JetPack 6
        if "R35" in text:
            return "orin"   # JetPack 5 (also Orin family)
        return "jetson_unknown"
    # No /etc/nv_tegra_release → not a Jetson. Assume Dell-class with pynvml.
    if torch.cuda.is_available():
        return "dell"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry — base class + per-device implementations
# ──────────────────────────────────────────────────────────────────────────────

class TelemetryCollector:
    """Base class. Each subclass implements start/stop/snapshot."""

    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.samples: list[dict] = []
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.end_time = None

    def start(self):
        self.samples = []
        self.monitoring = True
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict | None:
        self.monitoring = False
        self.end_time = time.perf_counter()
        if self.thread:
            self.thread.join(timeout=2.0)
        return self._summarize()

    def _loop(self):
        while self.monitoring:
            try:
                sample = self._read_one()
                if sample:
                    self.samples.append(sample)
            except Exception:
                pass
            time.sleep(self.sample_interval_ms / 1000.0)

    def _read_one(self) -> dict | None:
        raise NotImplementedError

    def _summarize(self) -> dict | None:
        raise NotImplementedError

    def cleanup(self):
        pass


class DellTelemetry(TelemetryCollector):
    """pynvml-based telemetry for Dell-class systems (NVIDIA discrete GPUs)."""

    def __init__(self, sample_interval_ms: int = 100):
        super().__init__(sample_interval_ms)
        self.available = False
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # test power read works
            pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.available = True
        except Exception as e:
            print(f"[telemetry] pynvml unavailable: {e}")

    def _read_one(self) -> dict | None:
        if not self.available:
            return None
        # Wrap each pynvml call individually — some platforms (e.g. NVIDIA GB10
        # / Grace-Blackwell consumer drivers) return NVMLError_NotSupported for
        # nvmlDeviceGetMemoryInfo but still support power and utilization. Don't
        # let one unsupported call drop the entire sample.
        sample: dict = {"t": time.perf_counter()}

        try:
            sample["power_w"] = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            pass

        try:
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            sample["gpu_util_pct"] = util.gpu
        except Exception:
            pass

        try:
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            sample["gpu_mem_used_mb"] = mem.used / (1024 ** 2)
        except Exception:
            # GB10 / Grace-Blackwell driver doesn't expose memory info via NVML.
            # Skip silently; downstream consumers should fall back to
            # torch.cuda.max_memory_allocated() for GPU memory on these systems.
            pass

        try:
            sample["cpu_pct"] = psutil.cpu_percent(interval=0)
        except Exception:
            pass

        # Return None only if we got nothing useful (only the timestamp)
        return sample if len(sample) > 1 else None

    def _summarize(self) -> dict | None:
        if not self.samples:
            return {"source": "pynvml", "available": False}

        powers = [s["power_w"] for s in self.samples if s.get("power_w") is not None]
        utils = [s["gpu_util_pct"] for s in self.samples if s.get("gpu_util_pct") is not None]
        mems = [s["gpu_mem_used_mb"] for s in self.samples if s.get("gpu_mem_used_mb") is not None]
        cpus = [s["cpu_pct"] for s in self.samples if s.get("cpu_pct") is not None]

        duration = (self.end_time or 0) - (self.start_time or 0)

        avg_w = sum(powers) / len(powers) if powers else None
        peak_w = max(powers) if powers else None
        energy_j = avg_w * duration if avg_w and duration else None

        return {
            "power_watts_avg": round(avg_w, 3) if avg_w is not None else None,
            "power_watts_peak": round(peak_w, 3) if peak_w is not None else None,
            "power_watts_samples": [round(p, 3) for p in powers],
            "power_rails": {
                "pynvml_gpu_mean_w": round(avg_w, 3) if avg_w is not None else None,
                "total": round(avg_w, 3) if avg_w is not None else None,
            },
            "gpu_utilization_percent_mean": round(sum(utils) / len(utils), 1) if utils else None,
            "gpu_utilization_percent_peak": max(utils) if utils else None,
            "gpu_mem_used_mb_peak": round(max(mems), 1) if mems else None,
            "cpu_percent_avg": round(sum(cpus) / len(cpus), 1) if cpus else None,
            "cpu_percent_peak": round(max(cpus), 1) if cpus else None,
            "energy_joules_est": round(energy_j, 2) if energy_j is not None else None,
            "duration_seconds": round(duration, 3),
            "sample_count": len(self.samples),
            "source": "pynvml",
        }

    def cleanup(self):
        if self.available:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass


# GR3D_FREQ formats observed across JetPack versions:
#   Thor JP7:        "GR3D_FREQ @[314,314,314]"     (no %, 3 engines)
#   Orin JP6.4:      "GR3D_FREQ 0%@[0,0]"           (%, 2 engines as array)
#   Older JP6/JP5:   "GR3D_FREQ 45%@318"            (%, single freq)
# This unified regex captures optional util% and a list of one or more freqs.
GR3D_RE = re.compile(
    r"GR3D_FREQ\s+(?:(\d+)%)?\s*@?\s*\[?\s*([\d,\s]+?)\s*\]?(?=\s|$)"
)

# Regex for tegrastats parsing (used across Thor / Orin / older Jetsons)
TEGRA_RE = {
    "vin_mw":             re.compile(r"\bVIN\s+(\d+)mW/(\d+)mW"),
    "vin_sys_5v0_mw":     re.compile(r"\bVIN_SYS_5V0\s+(\d+)mW/(\d+)mW"),
    "vdd_cpu_soc_mss_mw": re.compile(r"\bVDD_CPU_SOC_MSS\s+(\d+)mW/(\d+)mW"),
    "vdd_gpu_soc_mw":     re.compile(r"\bVDD_GPU_SOC\s+(\d+)mW/(\d+)mW"),
    "vdd_cpu_cv_mw":      re.compile(r"\bVDD_CPU_CV\s+(\d+)mW/(\d+)mW"),
    "ram_used_mb":        re.compile(r"\bRAM\s+(\d+)/(\d+)MB"),
    "emc_freq_pct":       re.compile(r"EMC_FREQ\s+(\d+)%"),
    "tj_celsius":         re.compile(r"\btj@([\d.]+)C"),
    "cpu_freqs":          re.compile(r"CPU\s+\[([^\]]+)\]"),
}


class TegrastatsTelemetry(TelemetryCollector):
    """
    Tegrastats-based telemetry for Jetson devices (Thor JP7, Orin JP6/5).
    Spawns tegrastats as a subprocess and parses lines from its stdout.

    Requires passwordless sudo for /usr/bin/tegrastats. Configure via:
        sudo visudo
        # add: <user> ALL=(ALL) NOPASSWD: /usr/bin/tegrastats, /usr/bin/pkill
    """

    def __init__(self, device_kind: str, sample_interval_ms: int = 100):
        super().__init__(sample_interval_ms)
        self.device_kind = device_kind   # "thor" or "orin"
        self.proc = None
        self.reader_thread = None
        self.available = self._check_available()
        if not self.available:
            return
        # Defensive: kill any stale tegrastats from prior runs
        subprocess.run(
            ["sudo", "-n", "pkill", "-9", "-f", "tegrastats"],
            check=False, capture_output=True
        )

    def _check_available(self) -> bool:
        try:
            r = subprocess.run(
                ["sudo", "-n", "tegrastats", "--help"],
                capture_output=True, timeout=3, text=True
            )
            return r.returncode in (0, 1)  # --help may exit nonzero
        except Exception as e:
            print(f"[telemetry] tegrastats unavailable: {e}")
            return False

    def start(self):
        if not self.available:
            return
        self.samples = []
        self.monitoring = True
        self.start_time = time.perf_counter()
        self.proc = subprocess.Popen(
            ["sudo", "-n", "tegrastats", "--interval", str(self.sample_interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def _reader_loop(self):
        try:
            for line in self.proc.stdout:
                if not self.monitoring:
                    break
                sample = self._parse_line(line)
                if sample:
                    self.samples.append(sample)
        except Exception:
            pass

    def _parse_line(self, line: str) -> dict | None:
        s: dict = {"t": time.perf_counter()}

        m = TEGRA_RE["vin_mw"].search(line)
        if m:
            s["vin_w"] = int(m.group(1)) / 1000.0

        m = TEGRA_RE["vin_sys_5v0_mw"].search(line)
        if m:
            s["vin_sys_5v0_w"] = int(m.group(1)) / 1000.0

        m = TEGRA_RE["vdd_cpu_soc_mss_mw"].search(line)
        if m:
            s["vdd_cpu_soc_mss_w"] = int(m.group(1)) / 1000.0

        m = TEGRA_RE["vdd_gpu_soc_mw"].search(line)
        if m:
            s["vdd_gpu_soc_w"] = int(m.group(1)) / 1000.0

        m = TEGRA_RE["vdd_cpu_cv_mw"].search(line)
        if m:
            s["vdd_cpu_cv_w"] = int(m.group(1)) / 1000.0

        # GPU activity — handles all observed formats:
        #   Thor JP7:   "GR3D_FREQ @[314,314,314]"     → freqs only
        #   Orin JP6.4: "GR3D_FREQ 0%@[0,0]"           → util + freqs array
        #   Older JP6:  "GR3D_FREQ 45%@318"            → util + single freq
        m = GR3D_RE.search(line)
        if m:
            util_str, freqs_str = m.group(1), m.group(2)
            if util_str is not None:
                s["gpu_util_pct"] = int(util_str)
            if freqs_str:
                freq_vals = [int(x.strip()) for x in freqs_str.split(",") if x.strip().isdigit()]
                if freq_vals:
                    s["gpu_freq_mhz_mean"] = sum(freq_vals) / len(freq_vals)
                    s["gpu_freq_mhz_peak"] = max(freq_vals)

        m = TEGRA_RE["ram_used_mb"].search(line)
        if m:
            s["ram_used_mb"] = int(m.group(1))

        m = TEGRA_RE["emc_freq_pct"].search(line)
        if m:
            s["emc_freq_pct"] = int(m.group(1))

        m = TEGRA_RE["tj_celsius"].search(line)
        if m:
            s["tj_c"] = float(m.group(1))

        m = TEGRA_RE["cpu_freqs"].search(line)
        if m:
            cores = m.group(1).split(",")
            cpu_pcts, cpu_mhzs = [], []
            for c in cores:
                cm = re.match(r"\s*(\d+)%@(\d+)", c)
                if cm:
                    cpu_pcts.append(int(cm.group(1)))
                    cpu_mhzs.append(int(cm.group(2)))
            if cpu_pcts:
                s["cpu_pct_mean"] = sum(cpu_pcts) / len(cpu_pcts)
                s["cpu_freq_mhz_mean"] = sum(cpu_mhzs) / len(cpu_mhzs)

        return s if len(s) > 1 else None  # >1 because "t" is always there

    def stop(self) -> dict | None:
        self.monitoring = False
        self.end_time = time.perf_counter()

        if self.proc:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.proc = None

        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)

        return self._summarize()

    def _summarize(self) -> dict | None:
        if not self.samples:
            return {"source": f"tegrastats_{self.device_kind}", "available": False}

        def mean(key: str) -> float | None:
            vals = [s[key] for s in self.samples if key in s]
            return sum(vals) / len(vals) if vals else None

        def peak(key: str) -> float | None:
            vals = [s[key] for s in self.samples if key in s]
            return max(vals) if vals else None

        duration = (self.end_time or 0) - (self.start_time or 0)

        vin_mean = mean("vin_w")
        vin_peak = peak("vin_w")
        vin_sys_5v0_mean = mean("vin_sys_5v0_w")
        vdd_cpu_soc_mss_mean = mean("vdd_cpu_soc_mss_w")
        vdd_gpu_soc_mean = mean("vdd_gpu_soc_w")
        vdd_cpu_cv_mean = mean("vdd_cpu_cv_w")

        # Headline power figure: VIN (total board input) on Thor;
        # sum of GPU+CPU+SYS rails on Orin; fall back as available.
        if vin_mean is not None:
            headline_w = vin_mean
        elif vdd_gpu_soc_mean is not None and vdd_cpu_cv_mean is not None:
            headline_w = vdd_gpu_soc_mean + vdd_cpu_cv_mean + (vin_sys_5v0_mean or 0)
        else:
            headline_w = vin_sys_5v0_mean

        energy_j = headline_w * duration if headline_w and duration else None

        rails = {}
        if vin_mean is not None:
            rails["vin_mean_w"] = round(vin_mean, 3)
        if vin_sys_5v0_mean is not None:
            rails["vin_sys_5v0_mean_w"] = round(vin_sys_5v0_mean, 3)
        if vdd_cpu_soc_mss_mean is not None:
            rails["vdd_cpu_soc_mss_mean_w"] = round(vdd_cpu_soc_mss_mean, 3)
        if vdd_gpu_soc_mean is not None:
            rails["vdd_gpu_soc_mean_w"] = round(vdd_gpu_soc_mean, 3)
        if vdd_cpu_cv_mean is not None:
            rails["vdd_cpu_cv_mean_w"] = round(vdd_cpu_cv_mean, 3)
        if headline_w is not None:
            rails["total"] = round(headline_w, 3)

        notes = []
        if self.device_kind == "thor":
            notes.append("vdd_gpu_excluded_unreliable_on_jp7")
            notes.append("gr3d_reports_freq_not_util_on_jp7")

        return {
            "power_watts_avg": round(headline_w, 3) if headline_w is not None else None,
            "power_watts_peak": round(vin_peak, 3) if vin_peak is not None else None,
            "power_rails": rails,
            "gpu_freq_mhz_mean": round(mean("gpu_freq_mhz_mean") or 0, 1)
                                  if mean("gpu_freq_mhz_mean") else None,
            "gpu_freq_mhz_peak": peak("gpu_freq_mhz_peak"),
            "gpu_utilization_percent_mean": round(mean("gpu_util_pct") or 0, 1)
                                            if mean("gpu_util_pct") else None,
            "cpu_percent_avg": round(mean("cpu_pct_mean") or 0, 1)
                                if mean("cpu_pct_mean") else None,
            "cpu_freq_mhz_mean": round(mean("cpu_freq_mhz_mean") or 0, 1)
                                  if mean("cpu_freq_mhz_mean") else None,
            "ram_used_mb_peak": peak("ram_used_mb"),
            "tj_celsius_peak": round(peak("tj_c") or 0, 2) if peak("tj_c") else None,
            "emc_freq_pct_mean": round(mean("emc_freq_pct") or 0, 1)
                                  if mean("emc_freq_pct") else None,
            "energy_joules_est": round(energy_j, 2) if energy_j is not None else None,
            "duration_seconds": round(duration, 3),
            "sample_count": len(self.samples),
            "source": f"tegrastats_{self.device_kind}",
            "notes": "; ".join(notes) if notes else None,
        }

    def cleanup(self):
        if self.proc:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        # final defensive sweep
        subprocess.run(
            ["sudo", "-n", "pkill", "-9", "-f", "tegrastats"],
            check=False, capture_output=True
        )


def make_telemetry(device_kind: str) -> TelemetryCollector:
    if device_kind == "dell":
        return DellTelemetry()
    if device_kind in ("thor", "orin"):
        return TegrastatsTelemetry(device_kind)
    # cpu / jetson_unknown — return a no-op collector
    class NoOpTelemetry(TelemetryCollector):
        def _read_one(self): return None
        def _summarize(self): return {"source": "none", "available": False}
    return NoOpTelemetry()


# ──────────────────────────────────────────────────────────────────────────────
# CUDA snapshot (works on both Thor and Dell via PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def reset_cuda_peaks():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def gpu_mem_peak_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return round(max(alloc, reserved), 3)


# ──────────────────────────────────────────────────────────────────────────────
# Task suite
# ──────────────────────────────────────────────────────────────────────────────

TASKS = {
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


# ──────────────────────────────────────────────────────────────────────────────
# Per-model generation dispatch
# ──────────────────────────────────────────────────────────────────────────────
#
# Each generator takes:
#   processor, model, image (PIL.Image), user_text (str), device, max_new_tokens
# and returns (text, error_or_None, input_len, gen_len).
#
# Add models here as you expand. Each is isolated to its own venv.
# ──────────────────────────────────────────────────────────────────────────────

def moondream_generate(processor, model, image, user_text, device, max_new_tokens):
    """Moondream2 uses its own .query() API via trust_remote_code."""
    try:
        result = model.query(image, user_text)
        text = (result.get("answer") if isinstance(result, dict) else str(result)).strip()
        if not text:
            return "", "empty_generation", None, None
        return text, None, None, None
    except Exception as e:
        return "", f"generate_error: {type(e).__name__}: {e}", None, None


def hf_chat_template_generate(processor, model, image, user_text, device, max_new_tokens):
    """
    Generic HF chat-template path. Works for InternVL2, SmolVLM2, Gemma3n,
    Phi-3.5-Vision when their processors support apply_chat_template with images.
    """
    try:
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
        if device == "cuda" or (torch.cuda.is_available() and str(model.device).startswith("cuda")):
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(device="cuda", dtype=model.dtype)
                else:
                    inputs[k] = v.to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        input_len = int(inputs["input_ids"].shape[-1])
        gen_ids = outputs[0, input_len:]
        gen_len = int(gen_ids.numel())
        text = processor.decode(gen_ids, skip_special_tokens=True).strip()

        if not text:
            return "", "empty_generation", input_len, gen_len
        return text, None, input_len, gen_len
    except Exception as e:
        return "", f"generate_error: {type(e).__name__}: {e}", None, None


# ──────────────────────────────────────────────────────────────────────────────
# InternVL2-specific generation
# ──────────────────────────────────────────────────────────────────────────────
#
# InternVL2 predates the standard HF chat-template + image-input convention.
# It requires:
#   - A separate AutoTokenizer (not processor.tokenizer)
#   - Custom image preprocessing with dynamic 448x448 tiling
#   - A model-specific .chat(tokenizer, pixel_values, question, gen_config) method
#
# Image preprocessing reference: official InternVL2 HuggingFace model card.
# ──────────────────────────────────────────────────────────────────────────────

_INTERNVL_TOKENIZER = None  # cached at module level — loaded once per process
_INTERNVL_TRANSFORM = None  # cached at module level

_INTERNVL_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_INTERNVL_IMAGENET_STD = (0.229, 0.224, 0.225)
_INTERNVL_INPUT_SIZE = 448
_INTERNVL_MAX_TILES = 12


def _internvl_build_transform():
    """Build the standard InternVL2 image transform (448x448, ImageNet normalize)."""
    global _INTERNVL_TRANSFORM
    if _INTERNVL_TRANSFORM is not None:
        return _INTERNVL_TRANSFORM
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    _INTERNVL_TRANSFORM = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((_INTERNVL_INPUT_SIZE, _INTERNVL_INPUT_SIZE),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_INTERNVL_IMAGENET_MEAN, std=_INTERNVL_IMAGENET_STD),
    ])
    return _INTERNVL_TRANSFORM


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find target grid layout closest to image aspect ratio."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Tile image into an NxM grid of 448x448 patches based on aspect ratio."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )

    target_w = image_size * target_aspect_ratio[0]
    target_h = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = image.resize((target_w, target_h))
    cols = target_w // image_size
    tiles = []
    for i in range(blocks):
        x0 = (i % cols) * image_size
        y0 = (i // cols) * image_size
        tiles.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def _internvl_preprocess_image(image, dtype):
    """Convert a PIL image to InternVL2's expected pixel_values tensor (cuda)."""
    transform = _internvl_build_transform()
    tiles = _internvl_dynamic_preprocess(
        image, image_size=_INTERNVL_INPUT_SIZE,
        use_thumbnail=True, max_num=_INTERNVL_MAX_TILES,
    )
    pixel_values = torch.stack([transform(t) for t in tiles])
    pixel_values = pixel_values.to(dtype=dtype, device="cuda")
    return pixel_values


def _internvl_get_tokenizer(model_id: str):
    """Lazy-loaded tokenizer for InternVL2 generation."""
    global _INTERNVL_TOKENIZER
    if _INTERNVL_TOKENIZER is None:
        from transformers import AutoTokenizer
        _INTERNVL_TOKENIZER = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False,
        )
    return _INTERNVL_TOKENIZER


def internvl_generate(processor, model, image, user_text, device, max_new_tokens):
    """
    InternVL2 uses model.chat(tokenizer, pixel_values, question, gen_config).

    Conservative implementation: single 448x448 image, no dynamic tiling, no
    extra kwargs. Falls back to the simplest documented InternVL2 invocation.

    If anything fails, the full traceback is included in the returned error so
    we can debug what exactly broke inside model.chat().
    """
    try:
        import traceback as _tb

        # Resolve tokenizer (cached at module level)
        model_id = getattr(model.config, "_name_or_path", None) or "OpenGVLab/InternVL2-2B"
        tokenizer = _internvl_get_tokenizer(model_id)

        # Simplest possible preprocessing: single 448x448 patch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        pixel_values = transform(image).unsqueeze(0)  # (1, 3, 448, 448)
        pixel_values = pixel_values.to(dtype=model.dtype, device="cuda")

        # Minimal prompt + gen config
        question = f"<image>\n{user_text}"
        gen_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values, question, gen_config)

        # Some InternVL2 versions return a tuple even without return_history
        if isinstance(response, tuple):
            response = response[0]

        text = (response or "").strip()
        if not text:
            return "", "empty_generation", None, None

        # Post-hoc tokenization for gen_len (best-effort)
        try:
            gen_len = len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            gen_len = None

        return text, None, None, gen_len

    except Exception as e:
        import traceback as _tb
        tb = _tb.format_exc()
        # Truncate the traceback to keep the JSONL record manageable
        tb_short = tb[-1500:] if len(tb) > 1500 else tb
        return "", f"generate_error: {type(e).__name__}: {e}\n--- traceback ---\n{tb_short}", None, None


GENERATORS = {
    "moondream":    moondream_generate,
    "gemma3n":      hf_chat_template_generate,
    "internvl":     internvl_generate,
    "internvl2_2b": internvl_generate,
    "smolvlm":      hf_chat_template_generate,
    "smolvlm2":     hf_chat_template_generate,
    "llava_mini":   hf_chat_template_generate,
    "phi":          hf_chat_template_generate,
    "phi35v":       hf_chat_template_generate,
}

# Per-model HF identifiers
MODEL_IDS = {
    "moondream":    "vikhyatk/moondream2",
    "gemma3n":      "google/gemma-3n-E4B-it",
    "internvl":     "OpenGVLab/InternVL2-2B",
    "internvl2_2b": "OpenGVLab/InternVL2-2B",
    "smolvlm":      "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "smolvlm2":     "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "llava_mini":   "ICTNLP/llava-mini-llama-3.1-8b",
    "phi":          "microsoft/Phi-3.5-vision-instruct",
    "phi35v":       "microsoft/Phi-3.5-vision-instruct",
}


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_manifest(path: Path) -> list[str]:
    paths = []
    with open(path) as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    return paths


def image_id_from_path(path: str) -> int:
    """COCO uses zero-padded integer filenames: 000000000139.jpg → 139."""
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        # fallback: hash so at least it's deterministic
        return abs(hash(stem)) % (10 ** 9)


def append_jsonl(fp, rec: dict):
    fp.write(json.dumps(rec) + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def load_completed_keys(runs_path: Path) -> set[tuple[int, str]]:
    """Returns set of (image_id, task) tuples already complete in runs.jsonl."""
    if not runs_path.exists():
        return set()
    done = set()
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("image_id") is not None and rec.get("task"):
                    done.add((rec["image_id"], rec["task"]))
            except Exception:
                continue
    return done


def resolve_dtype(dtype_flag: str):
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    return torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(GENERATORS.keys()),
                   help="Model key (each is isolated to its own venv)")
    p.add_argument("--manifest", required=True,
                   help="Text file with one absolute image path per line")
    p.add_argument("--output_dir", required=True, help="Base outputs directory")
    p.add_argument("--run_group", required=True,
                   help="Run tag (e.g. moondream_thor_20260514_1823)")
    p.add_argument("--device", default=None,
                   help="Override device detection (thor|dell|orin|cpu)")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--repeats", type=int, default=5,
                   help="Repeats per (image, task) for stability metric")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--warmup", type=int, default=3,
                   help="Warmup generations on first image before measurement")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of images (for testing). 0 = no limit.")
    p.add_argument("--resume", action="store_true",
                   help="Skip (image, task) pairs already complete")
    p.add_argument("--telemetry_interval_ms", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()

    # device
    device_kind = args.device or detect_device()
    print(f"[setup] device kind: {device_kind}")

    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(args.dtype)
    if cuda_device == "cpu":
        dtype = torch.float32

    # manifest
    manifest = Path(args.manifest).resolve()
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    image_paths = read_manifest(manifest)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise SystemExit("no images in manifest")
    print(f"[setup] {len(image_paths)} images from manifest")

    # output dir
    out_dir = Path(args.output_dir).resolve() / args.model / args.run_group
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"

    # resume
    completed_keys: set[tuple[int, str]] = set()
    if args.resume:
        completed_keys = load_completed_keys(runs_path)
        print(f"[setup] resume: {len(completed_keys)} (image, task) pairs already complete")

    # telemetry
    telemetry = make_telemetry(device_kind)
    print(f"[setup] telemetry: {telemetry.__class__.__name__}")

    # run metadata
    model_id = MODEL_IDS.get(args.model, args.model)
    host = platform.node() or "unknown"
    run_meta = {
        "model_key": args.model,
        "model_id": model_id,
        "run_group": args.run_group,
        "host": host,
        "device_kind": device_kind,
        "cuda_device": cuda_device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": _get_transformers_version(),
        "manifest": str(manifest),
        "num_images": len(image_paths),
        "num_tasks": len(TASKS),
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "telemetry_interval_ms": args.telemetry_interval_ms,
        "suite": "semantic_extension_5task_v1",
        "timestamp_start": datetime.now().isoformat(),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # model load
    print(f"[load] loading {model_id} on {cuda_device} (dtype={dtype})")
    t_load_start = time.perf_counter()
    processor, model = _load_model(args.model, model_id, cuda_device, dtype)
    t_load_end = time.perf_counter()
    print(f"[load] loaded in {t_load_end - t_load_start:.1f}s")

    generate_fn = GENERATORS[args.model]

    # warmup
    if args.warmup > 0:
        try:
            img0 = Image.open(image_paths[0]).convert("RGB")
            first_prompt = next(iter(TASKS.values()))["prompt"]
            for _ in range(args.warmup):
                _ = generate_fn(processor, model, img0, first_prompt,
                                cuda_device, args.max_new_tokens)
            # one pass through all tasks on first image
            for tname, tcfg in TASKS.items():
                _ = generate_fn(processor, model, img0, tcfg["prompt"],
                                cuda_device, args.max_new_tokens)
            print(f"[warmup] {args.warmup} runs + 1 full-task pass")
        except Exception as e:
            print(f"[warmup] failed (continuing): {e}")

    # main loop
    t_bench_start = time.perf_counter()
    total_records = 0
    total_errors = 0
    total_repeats = 0
    total_repeat_errors = 0

    with open(runs_path, "a") as fout:
        for img_i, img_path in enumerate(image_paths):
            image_id = image_id_from_path(img_path)
            img_name = Path(img_path).name

            # load image once per image; reuse across tasks
            try:
                img = Image.open(img_path).convert("RGB")
                img_w, img_h = img.size
            except Exception as e:
                print(f"[err] image_load_failed {img_path}: {e}")
                # still log one record per task so resume works
                for tname, tcfg in TASKS.items():
                    if (image_id, tname) in completed_keys:
                        continue
                    rec = _make_error_record(
                        args, model_id, device_kind, dtype,
                        image_id, img_path, img_name, None, None,
                        tname, tcfg, f"image_load_failed: {e}",
                    )
                    append_jsonl(fout, rec)
                    total_records += 1
                    total_errors += 1
                continue

            for tname, tcfg in TASKS.items():
                if (image_id, tname) in completed_keys:
                    continue

                reset_cuda_peaks()
                telemetry.start()

                outputs_list = []
                for run_idx in range(args.repeats):
                    t0 = time.perf_counter()
                    if cuda_device == "cuda":
                        torch.cuda.synchronize()
                    text, err, input_len, gen_len = generate_fn(
                        processor, model, img, tcfg["prompt"],
                        cuda_device, args.max_new_tokens,
                    )
                    if cuda_device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    gen_lat_s = t1 - t0
                    # End-to-end TPS: total tokens / total latency.
                    # gen_len is None for Moondream (uses .query() API) — backfilled
                    # post-hoc in compute_quality_metrics.py by tokenizing output_text.
                    tps = None
                    if gen_len is not None and gen_len > 0 and gen_lat_s > 0:
                        tps = round(gen_len / gen_lat_s, 3)

                    outputs_list.append({
                        "run": run_idx,
                        "text": text,
                        "gen_latency_s": round(gen_lat_s, 4),
                        "input_len": input_len,
                        "gen_len": gen_len,
                        "tokens_per_sec": tps,
                        "error": err,
                    })
                    total_repeats += 1
                    if err is not None:
                        total_repeat_errors += 1

                hw_stats = telemetry.stop() or {}
                hw_stats["gpu_mem_alloc_gb_peak"] = gpu_mem_peak_gb()

                rec = {
                    "model": model_id,
                    "model_key": args.model,
                    "device": device_kind,
                    "host": host,
                    "dtype": str(dtype).replace("torch.", ""),
                    "image_id": image_id,
                    "image_path": img_path,
                    "image_name": img_name,
                    "image_width": img_w,
                    "image_height": img_h,
                    "task": tname,
                    "task_prompt": tcfg["prompt"],
                    "task_purpose": tcfg["purpose"],
                    "outputs": outputs_list,
                    "hw_stats": hw_stats,
                    "expected_entities": None,   # filled by post-hoc script
                    "timestamp": datetime.now().isoformat(),
                    "run_group": args.run_group,
                }
                append_jsonl(fout, rec)
                total_records += 1

                # one-line status
                mean_lat = sum(o["gen_latency_s"] for o in outputs_list) / len(outputs_list)
                n_err = sum(1 for o in outputs_list if o["error"] is not None)
                err_tag = f" err={n_err}" if n_err > 0 else ""
                print(f"[{args.model}] img {img_i+1}/{len(image_paths)} "
                      f"task {tname:22} mean {mean_lat*1000:7.1f} ms{err_tag}")
                if n_err > 0:
                    total_errors += 1

    t_bench_end = time.perf_counter()
    elapsed_s = t_bench_end - t_bench_start

    # summary
    summary = {
        **run_meta,
        "timestamp_end": datetime.now().isoformat(),
        "total_elapsed_seconds": round(elapsed_s, 1),
        "total_elapsed_minutes": round(elapsed_s / 60, 2),
        "total_records": total_records,
        "total_records_with_errors": total_errors,
        "total_repeats": total_repeats,
        "total_repeat_errors": total_repeat_errors,
        "model_load_seconds": round(t_load_end - t_load_start, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 64)
    print(f"  done — {args.model} on {device_kind}")
    print(f"  records: {total_records}  (errors in record: {total_errors})")
    print(f"  repeats: {total_repeats}  (errors in repeat: {total_repeat_errors})")
    print(f"  elapsed: {elapsed_s/60:.2f} min")
    print(f"  output:  {out_dir}")
    print("=" * 64)

    # cleanup
    telemetry.cleanup()
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


def _load_model(model_key: str, model_id: str, cuda_device: str, dtype):
    """Per-model loading. Each env has the deps the model needs."""
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText

    device_map = {"": 0} if cuda_device == "cuda" else None

    if model_key == "moondream":
        # Moondream2 uses trust_remote_code path
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            dtype=dtype, device_map=device_map,
        )
    elif model_key in ("gemma3n",):
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device_map,
            attn_implementation="eager",
        )
    else:
        # InternVL, SmolVLM, LLaVA-Mini, Phi-3.5-Vision
        # Try AutoModelForImageTextToText first (modern HF integration).
        # If that fails (e.g. older models with custom code like InternVL2),
        # fall back to AutoModelForCausalLM with trust_remote_code.
        # Some older custom-code models don't accept the newer `dtype=` kwarg
        # and need the legacy `torch_dtype=` instead — try both.
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = None
        last_err = None
        for model_cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
            if model is not None:
                break
            for dtype_kwarg in ("dtype", "torch_dtype"):
                kwargs = {
                    "trust_remote_code": True,
                    "device_map": device_map,
                    dtype_kwarg: dtype,
                }
                # attn_implementation only goes on the AutoModelForImageTextToText path
                if model_cls is AutoModelForImageTextToText:
                    kwargs["attn_implementation"] = "eager"
                try:
                    model = model_cls.from_pretrained(model_id, **kwargs)
                    print(f"[load] used {model_cls.__name__} with {dtype_kwarg}=")
                    break
                except (TypeError, ValueError) as e:
                    last_err = e
                    continue
        if model is None:
            raise RuntimeError(
                f"Could not load {model_id} via either "
                f"AutoModelForImageTextToText or AutoModelForCausalLM "
                f"with either dtype= or torch_dtype=. Last error: {last_err}"
            )

    model.eval()
    return processor, model


def _make_error_record(args, model_id, device_kind, dtype, image_id, img_path,
                       img_name, img_w, img_h, tname, tcfg, err_str):
    return {
        "model": model_id,
        "model_key": args.model,
        "device": device_kind,
        "dtype": str(dtype).replace("torch.", ""),
        "image_id": image_id,
        "image_path": img_path,
        "image_name": img_name,
        "image_width": img_w,
        "image_height": img_h,
        "task": tname,
        "task_prompt": tcfg["prompt"],
        "task_purpose": tcfg["purpose"],
        "outputs": [{"run": 0, "text": "", "gen_latency_s": None,
                     "input_len": None, "gen_len": None,
                     "tokens_per_sec": None, "error": err_str}],
        "hw_stats": None,
        "expected_entities": None,
        "timestamp": datetime.now().isoformat(),
        "run_group": args.run_group,
    }


if __name__ == "__main__":
    main()