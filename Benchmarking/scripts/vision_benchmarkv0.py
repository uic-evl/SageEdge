#!/usr/bin/env python3
# benchmark multiple vision-language models locally with huggingface
# usage:
#   python3 benchmark_hf_vision_models.py gemma3n moondream

import os
import sys
import time
import json
from datetime import datetime
import gc

import psutil
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,   # gemma3n
    AutoTokenizer,                 # moondream
    AutoModelForCausalLM,          # moondream
)

# -------------------- cli args --------------------

if len(sys.argv) < 2:
    print("usage: python3 benchmark_hf_vision_models.py <model1> <model2> ...")
    print("example: python3 benchmark_hf_vision_models.py gemma3n moondream")
    sys.exit(1)

MODEL_KEYS = sys.argv[1:]

# -------------------- config --------------------

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# folder with your 5 test images (change this path to your actual folder)
IMAGE_DIR = "/home/thorwaggle1/Desktop/SageEdge/Benchmarking/images"

# three prompts we will run for every image
VISION_TASKS = {
    "image_captioning": "Describe the content of this image.",
    "object_detection_like": "List the main objects you can see in this image.",
    "scene_understanding": "Describe the scene in this image in detail.",
}

RAM_THRESHOLD_GB = 1.0

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
}

# -------------------- helpers --------------------

def get_image_file_list(image_dir):
    """return a sorted list of image paths from the given folder."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    if not os.path.isdir(image_dir):
        print(f"⚠️ image directory not found: {image_dir}")
        return files
    for name in sorted(os.listdir(image_dir)):
        if name.lower().endswith(exts):
            files.append(os.path.join(image_dir, name))
    if not files:
        print(f"⚠️ no images found in {image_dir}")
    else:
        print(f"📂 found {len(files)} images in {image_dir}")
    return files


def get_system_metrics():
    # note: cpu_percent(interval=0) avoids a 1s pause each call
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
        # here "processor" is actually the tokenizer; we treat it uniformly
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        return tokenizer, model

    else:
        raise ValueError(f"unsupported model type: {model_type}")


def run_inference_on_image(config, processor_like, model, image_path, prompt, device):
    """
    run a single prompt + image through the given model.
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
        # here processor_like is the tokenizer
        tokenizer = processor_like
        with torch.no_grad():
            image_emb = model.encode_image(img)
            answer = model.answer_question(image_emb, prompt, tokenizer)
        return answer

    else:
        raise ValueError(f"unsupported model type for inference: {model_type}")


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
        model_name_safe = key.replace("/", "_")
        log_dir = os.path.join("hf_benchmark_logs", f"{model_name_safe}_vision")
        os.makedirs(log_dir, exist_ok=True)

        available_ram = get_system_metrics()["ram_available_gb"]
        if available_ram < RAM_THRESHOLD_GB:
            print(f"\n❌ skipping '{key}' due to low available ram: {available_ram} gb")
            continue

        try:
            processor_like, model = load_hf_model(config, device, dtype)
        except Exception as e:
            print(f"\n❌ failed to load model '{key}': {e}")
            continue

        results = []
        num_runs = 0
        num_errors = 0
        durations = []

        # loop over every image and every task
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
                except Exception as e:
                    num_errors += 1
                    output_text = f"ERROR: {e}"
                    print(f"❌ failed to run task '{task_name}' for model '{key}' on '{image_name}': {e}")

                end_time = time.time()
                sys_after = get_system_metrics()

                duration = end_time - start_time
                tokens = len(output_text.split())
                tps = tokens / duration if duration > 0 else 0.0
                durations.append(duration)

                record = {
                    "model_key": key,
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
                }

                print(f"✅ done in {duration:.2f}s ({tokens} tokens, tps={tps:.2f})")
                print(f"   sample output: {output_text[:200]}...")

                results.append(record)

        # save summary for this model
        summary_path = os.path.join(log_dir, "vision_benchmark_results.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        print(f"\n✅ vision benchmark complete for '{key}'. results saved in: {log_dir}")
        print(f"   total runs: {num_runs}, errors: {num_errors}, avg duration: {avg_duration:.2f}s")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
