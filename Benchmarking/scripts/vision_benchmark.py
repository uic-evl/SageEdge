#!/usr/bin/env python3
# benchmark multiple vision-language models + yolo locally
# usage:
#   python3 vision_benchmark.py gemma3n moondream yolo

import os
import sys
import time
import json
from datetime import datetime
import gc

import psutil
import torch
from pathlib import Path
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,   # gemma3n
    AutoTokenizer,                 # moondream
    AutoModelForCausalLM,          # moondream
)

from ultralytics import YOLO      # yolo object detector

# -------------------- cli args --------------------

if len(sys.argv) < 2:
    print("usage: python3 vision_benchmark.py <model1> <model2> ...")
    print("example: python3 vision_benchmark.py gemma3n moondream yolo")
    sys.exit(1)

MODEL_KEYS = sys.argv[1:]

# -------------------- config --------------------

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root (Benchmarking/)
# allow overrides, otherwise use repo-relative folders
IMAGE_DIR = Path(os.getenv("BENCH_IMAGE_DIR", BASE_DIR / "images")).resolve()
OUTPUT_DIR = Path(os.getenv("BENCH_OUTPUT_DIR", BASE_DIR / "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# prompts we will run for every *vlm* (gemma / moondream)
VISION_TASKS = {
    "image_captioning": "Describe the content of this image.",
    "object_detection_like": "List the main objects you can see in this image.",
    "scene_understanding": "Describe the scene in this image in detail.",
}

RAM_THRESHOLD_GB = 1.0

# path to your yolo weights; change this to what you actually have
YOLO_WEIGHTS = "yolo11n.pt"   # automatically downloads if missing

def load_yolo_model(config, device):
    print(f"\n🔧 loading YOLO model '{config['weights']}' on {device}")
    model = YOLO(config["weights"])
    model.to(device)
    return model

MODEL_CONFIG = {
    "gemma3n": {
        "type": "gemma3n",
        "hf_id": "google/gemma-3n-E4B-it",
        "max_new_tokens": 80,
    },
    "moondream": {
        "type": "moondream",
        "hf_id": "vikhyatk/moondream2",
        "max_new_tokens": 80,
    },
    "yolo": {
        "type": "yolo",
        "weights": "yolo11n.pt",
    },
}


# -------------------- helpers --------------------
def get_image_file_list(image_dir):
    """return a sorted list of image paths from the given folder."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_dir = Path(image_dir)

    if not image_dir.is_dir():
        print(f"⚠️ image directory not found: {image_dir}")
        return []

    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])

    if not files:
        print(f"⚠️ no images found in {image_dir}")
        return []

    print(f"📂 found {len(files)} images in {image_dir}")
    return [str(p) for p in files]  # keep rest of your code unchanged



def get_system_metrics():
    """lightweight snapshot of system usage."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu = psutil.cpu_percent(interval=0)
    return {
        "cpu_percent": cpu,
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_percent": mem.percent,
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "swap_percent": swap.percent,
        "ram_available_gb": round(mem.available / (1024**3), 2),
    }


def load_hf_model(config, device, dtype):
    """
    load processor/tokenizer + model according to config['type'].
    returns (processor_like, model).
    """
    model_type = config["type"]
    hf_id = config.get("hf_id")

    print(f"\n🔧 loading model '{hf_id}' as type '{model_type}' on {device} (dtype={dtype})")

    if model_type == "gemma3n":
        processor = AutoProcessor.from_pretrained(hf_id)
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device,
        )
        return processor, model

    elif model_type == "moondream":
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        return tokenizer, model

    else:
        raise ValueError(f"unsupported model type for hf model loader: {model_type}")


def run_inference_on_image(config, processor_like, model, image_path, prompt, device):
    """
    run a single prompt + image through the given vlm (gemma / moondream).
    returns generated text.
    """
    model_type = config["type"]
    img = Image.open(image_path).convert("RGB")

    if model_type == "gemma3n":
        processor = processor_like
        tokenizer = processor.tokenizer
        bos = tokenizer.bos_token or ""

        full_prompt = (
            f"{bos}"
            "<start_of_turn>user\n"
            f"<image_soft_token>{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        inputs = processor(
            text=full_prompt,
            images=[img],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.get("max_new_tokens", 80),
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0, input_len:]
        caption = processor.decode(generated_ids, skip_special_tokens=True)
        return caption

    elif model_type == "moondream":
        tokenizer = processor_like
        with torch.no_grad():
            image_emb = model.encode_image(img)
            answer = model.answer_question(image_emb, prompt, tokenizer)
        return answer

    else:
        raise ValueError(f"unsupported model type for vlm inference: {model_type}")

def run_yolo_on_image(model, image_path, device):
    """
    run yolo on a single image.
    returns:
      - num_detections
      - per_class_counts dict
      - summary string
    """
    # run prediction (no saving, no verbose)
    # passing path is easiest and keeps pre-processing inside yolo
    results = model.predict(
        source=image_path,
        device=device,
        verbose=False,
    )

    # results is a list, we only passed one image
    r = results[0]

    num_detections = 0
    per_class_counts = {}

    if r.boxes is not None and r.boxes.cls is not None:
        cls_indices = r.boxes.cls.detach().cpu().tolist()
        num_detections = len(cls_indices)
        names = model.names  # mapping idx -> class name
        for idx in cls_indices:
            cls_name = names.get(int(idx), str(int(idx)))
            per_class_counts[cls_name] = per_class_counts.get(cls_name, 0) + 1

    if per_class_counts:
        parts = [f"{cls}: {count}" for cls, count in per_class_counts.items()]
        summary = ", ".join(parts)
    else:
        summary = "no detections"

    return num_detections, per_class_counts, summary


# -------------------- main loop --------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    image_files = get_image_file_list(IMAGE_DIR)
    if not image_files:
        print("no images to run on, exiting.")
        sys.exit(1)

    for key in MODEL_KEYS:
        if key not in MODEL_CONFIG:
            print(f"\n⚠️ unknown model key '{key}'. valid keys: {list(MODEL_CONFIG.keys())}")
            continue

        config = MODEL_CONFIG[key]
        model_type = config["type"]

        available_ram = get_system_metrics()["ram_available_gb"]
        if available_ram < RAM_THRESHOLD_GB:
            print(f"\n❌ skipping '{key}' due to low available ram: {available_ram} gb")
            continue

        results = []
        num_runs = 0
        num_errors = 0
        durations = []

        # -------------------------
        # yolo branch
        # -------------------------
        if model_type == "yolo":
            try:
                yolo_model = load_yolo_model(config, device)
            except Exception as e:
                print(f"\n❌ failed to load yolo model '{key}': {e}")
                continue

            for image_path in image_files:
                image_name = os.path.basename(image_path)
                num_runs += 1
                print(f"\n🧠 model '{key}' (yolo) | image '{image_name}'")

                if not os.path.exists(image_path):
                    print(f"⚠️ image not found: {image_path}")
                    num_errors += 1
                    continue

                sys_before = get_system_metrics()
                start_time = time.time()

                try:
                    num_det, class_counts, summary = run_yolo_on_image(
                        model=yolo_model,
                        image_path=image_path,
                        device=device,
                    )
                except Exception as e:
                    num_errors += 1
                    summary = ""
                    class_counts = {}
                    num_det = 0
                    print(f"❌ failed to run yolo for image '{image_name}': {e}")

                end_time = time.time()
                sys_after = get_system_metrics()

                duration = end_time - start_time
                durations.append(duration)

                record = {
                    "model_key": key,
                    "model_type": "yolo",
                    "task": "yolo_detection",
                    "image_path": image_path,
                    "image_name": image_name,
                    "num_detections": num_det,
                    "class_counts": class_counts,
                    "duration_sec": round(duration, 3),
                    "sys_before": sys_before,
                    "sys_after": sys_after,
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary,
                }

                print(f"✅ yolo done in {duration:.3f}s | detections={num_det} | {summary}")
                results.append(record)

        # -------------------------
        # vlm (gemma / moondream) branch
        # -------------------------
        else:
            try:
                processor_like, model = load_hf_model(config, device, dtype)
            except Exception as e:
                print(f"\n❌ failed to load model '{key}': {e}")
                continue

            for image_path in image_files:
                image_name = os.path.basename(image_path)

                for task_name, prompt in VISION_TASKS.items():
                    num_runs += 1
                    print(f"\n🧠 model '{key}' | image '{image_name}' | task '{task_name}'")

                    if not os.path.exists(image_path):
                        print(f"⚠️ image not found: {image_path}")
                        num_errors += 1
                        continue

                    sys_before = get_system_metrics()
                    start_time = time.time()

                    try:
                        output_text = run_inference_on_image(
                            config=config,
                            processor_like=processor_like,
                            model=model,
                            image_path=image_path,
                            prompt=prompt,
                            device=device,
                        )
                        error_msg = None
                    except Exception as e:
                        num_errors += 1
                        error_msg = str(e)
                        output_text = ""
                        print(
                            f"❌ failed to run task '{task_name}' "
                            f"for model '{key}' on '{image_name}': {e}"
                        )

                    end_time = time.time()
                    sys_after = get_system_metrics()

                    duration = end_time - start_time
                    durations.append(duration)

                    tokens = len(output_text.split()) if output_text else 0
                    tps = tokens / duration if duration > 0 and tokens > 0 else 0.0

                    record = {
                        "model_key": key,
                        "model_type": model_type,
                        "hf_id": config.get("hf_id"),
                        "task": task_name,
                        "prompt": prompt,
                        "image_path": image_path,
                        "image_name": image_name,
                        "tokens": tokens,
                        "duration_sec": round(duration, 3),
                        "tps": round(tps, 3),
                        "sys_before": sys_before,
                        "sys_after": sys_after,
                        "timestamp": datetime.now().isoformat(),
                        "output": output_text,
                        "error": error_msg,
                    }

                    print(f"✅ done in {duration:.2f}s ({tokens} tokens, tps={tps:.2f})")
                    if output_text:
                        print(f"   sample output: {output_text[:200]}...")
                    results.append(record)

        # --------------------------
        # save results with timestamp
        # --------------------------
        timestamp_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"{key}_vision_results_{timestamp_date}.json"
        summary_path = str(OUTPUT_DIR / output_filename)

        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        avg_duration = sum(durations) / len(durations) if durations else 0.0

        print(f"\n📁 saved results to: {summary_path}")
        print(
            f"✅ benchmark complete for '{key}'. "
            f"total runs: {num_runs}, errors: {num_errors}, avg duration: {avg_duration:.3f}s"
        )

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ torch.cuda.empty_cache() failed: {e}")


if __name__ == "__main__":
    main()
