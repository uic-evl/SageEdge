"""
Unified captioning run for all VLMs — same prompt, deterministic decoding,
version + latency logging, resumable.

Usage (from ~/Desktop/SageEdge/Research_projects/AI_Judge):
  python run_captions_unified.py --model moondream:latest --list coco_1000.txt --out runs/v3/moondream.jsonl
  python run_captions_unified.py --model llava:latest     --list coco_1000.txt --out runs/v3/llava.jsonl
  python run_captions_unified.py --model gemma3n-hf       --list coco_1000.txt --out runs/v3/gemma.jsonl

Quick sanity check first (10 images):   add  --limit 10
Captions only: YOLO + judging are done afterwards, offline, so the judge can be
changed without re-running the VLMs.
"""
from __future__ import annotations

import argparse, json, platform, time
from pathlib import Path

# Neutral prompt shared by ALL models. Short "one sentence" prompts make
# moondream (Ollama) emit broken captions ("urn of water...", "iced plane...").
PROMPT = "Describe this image."
SEED, MAX_TOKENS = 42, 160


# ----------------------------------------------------------------- backends
def ollama_backend(model: str):
    import ollama
    digest = next((m.get("digest") for m in ollama.list().get("models", [])
                   if m.get("model", m.get("name")) == model), None)

    # Thinking models (e.g. gemma4, qwen3) spend the token budget on hidden
    # "thinking" and return an empty response unless thinking is switched off.
    state = {"think_flag": True}

    def caption(path: str) -> dict:
        kw = dict(model=model, prompt=PROMPT, images=[path], stream=False,
                  options={"temperature": 0, "seed": SEED, "num_predict": MAX_TOKENS})
        if state["think_flag"]:
            try:
                r = ollama.generate(**kw, think=False)
            except Exception as e:              # model has no thinking switch
                if "think" not in str(e).lower():
                    raise
                state["think_flag"] = False
                r = ollama.generate(**kw)
        else:
            r = ollama.generate(**kw)
        return {"caption": (r.get("response") or "").strip(),
                "thinking_chars": len(r.get("thinking") or ""),
                "gen_tokens": r.get("eval_count"),
                "ollama_total_ms": int((r.get("total_duration") or 0) / 1e6)}
    return caption, {"backend": "ollama", "model": model, "digest": digest, "think": False}


def gemma_backend(model_id: str = "google/gemma-3n-E2B-it"):
    import torch, transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda, "| torch", torch.__version__)
    if not cuda:
        print("WARNING: running Gemma on CPU — this is why captions take ~30 s each.")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16 if cuda else torch.float32, device_map="auto")

    def caption(path: str) -> dict:
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(path).convert("RGB")},
                                             {"type": "text", "text": PROMPT}]}]
        inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(model.device, dtype=model.dtype)
        n_in = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=MAX_TOKENS, do_sample=False)
        text = processor.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return {"caption": text, "gen_tokens": int(out.shape[-1] - n_in)}
    return caption, {"backend": "huggingface", "model": model_id, "cuda": cuda,
                     "torch": torch.__version__, "transformers": transformers.__version__}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ollama tag (e.g. llava:latest) or gemma3n-hf")
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    caption, info = gemma_backend() if a.model == "gemma3n-hf" else ollama_backend(a.model)
    info.update({"prompt": PROMPT, "seed": SEED, "max_tokens": MAX_TOKENS,
                 "host": platform.node(), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("Run info:", info)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():                                   # resume instead of deleting
        done = {json.loads(l)["image"] for l in out.open() if l.strip()}
        print(f"Resuming: {len(done)} images already done")

    imgs = [str(Path(l.strip()).resolve()) for l in open(a.list) if l.strip()]
    if a.limit:
        imgs = imgs[:a.limit]

    with out.open("a") as f:
        for i, img in enumerate(imgs, 1):
            if img in done:
                continue
            t0 = time.perf_counter()
            try:
                rec = {"image": img, "ok": True, **caption(img)}
                if not rec["caption"]:
                    rec["ok"], rec["error"] = False, "empty caption"
            except Exception as e:                      # logged, never silently replaced
                rec = {"image": img, "ok": False, "caption": "", "error": f"{type(e).__name__}: {e}"}
            rec["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            rec["run_info"] = info
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{i}/{len(imgs)}] {rec['latency_ms']} ms  {rec['caption'][:150]!r}")


if __name__ == "__main__":
    main()
