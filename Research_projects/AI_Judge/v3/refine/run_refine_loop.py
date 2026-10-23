"""
Feedback + self-improvement loop (paper section "Judge feedback").

    round 0 caption (frozen, from results/<captioner>.jsonl)
      -> judge (YOLOv8x @ 0.25, precomputed detections) -> flagged objects
      -> rewrite prompt (image + previous caption + feedback) -> new caption
      -> judge again ... up to --max_rounds, or stop when nothing is flagged.

Conditions (--condition):
  self     : the same captioner rewrites its own caption, with judge feedback
  moondream: moondream:latest rewrites (image + caption + judge feedback)
  control  : the same captioner rewrites WITHOUT feedback ("look again").
             It is triggered by the same judge flags and runs the same rounds,
             so the only difference to `self` is the content of the feedback.
  oracle   : like `self`, but the feedback lists claims absent in COCO GT
             (upper bound). Needs --coco_ann.

Round 0 is NOT regenerated: it is the frozen V3 caption, so round 0 numbers
equal the paper numbers. Detections come from results/detections/*.jsonl
(same image -> same detections), so no YOLO is run here.

Usage (from v3/, venv_gpu active, Ollama running):
  python refine/run_refine_loop.py --captioner llava --condition self --limit 10
  python refine/run_refine_loop.py --captioner llava --condition self
  python refine/run_refine_loop.py --captioner llava --condition oracle \
      --coco_ann ../annotations/instances_val2017.json
Output: runs/refine/<captioner>__<condition>.jsonl (resumable, one line per image).
Test without Ollama:  add --dry_run (the "model" just repeats the old caption).
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from judge_v3.rule_judge_v3 import rule_based_judge_v3          # noqa: E402
from judge_v3.parse_caption_v3 import _match_phrases, _negated_positions, _tokenize  # noqa: E402

OLLAMA_TAG = {"llava": "llava:latest", "moondream": "moondream:latest", "gemma4": "gemma4:e2b"}
SEED, MAX_TOKENS = 42, 160

FEEDBACK_PROMPT = (
    'Here is a description of this image:\n"{caption}"\n\n'
    "The image does not contain: {objects}. Rewrite the description without these objects. "
    "Keep everything else. Reply with the new description only.")
# Prompt-robustness variant B = the control prompt plus one feedback sentence
# (chosen from refine/test_moondream_prompts.py). Default stays A (the main prompt).
FEEDBACK_PROMPT_B = (
    'Here is a description of this image:\n"{caption}"\n\n'
    "Look again carefully and describe the image. Do not mention {objects}. Reply with the new description only.")
CONTROL_PROMPT = (
    'Here is a description of this image:\n"{caption}"\n\n'
    "Look again carefully and describe the image. Reply with the new description only.")


# ----------------------------------------------------------------- helpers
def load_jsonl_by_name(path) -> dict:
    return {Path(r["image"]).name: r for r in map(json.loads, open(path)) if r.get("image")}


def load_coco_gt(ann_path: str) -> dict[str, set[str]]:
    ann = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    gt: dict[str, set[str]] = defaultdict(set)
    for a in ann["annotations"]:
        gt[names[a["image_id"]]].add(cats[a["category_id"]])
    return gt


def find_image(name: str, dirs: list[str]) -> str:
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(f"{name} not found in {dirs} (use --images_dir)")


def surface_words(caption: str) -> dict[str, list[str]]:
    """COCO class -> the words the caption itself used for it ("potted plant" -> ["plants"]).
    The feedback names these words, because the model cannot remove "potted plant"
    if its caption says "plants". Uses the same tokenizer/matcher as the parser."""
    toks = _tokenize(caption)
    neg = _negated_positions(toks)
    out: dict[str, list[str]] = defaultdict(list)
    for s, e, label in _match_phrases(toks):
        w = " ".join(toks[s:e])
        if label and not any(p in neg for p in range(s, e)) and w not in out[label]:
            out[label].append(w)
    return out


def feedback_text(caption: str, classes: list[str]) -> str:
    sw = surface_words(caption)
    words: list[str] = []
    for c in classes:
        for w in sw.get(c) or [c]:
            if w not in words:
                words.append(w)
    return ", ".join(words)


def judge(caption: str, dets: list, conf: float, gt: set | None):
    """Return (judge record, COCO classes to put in the feedback)."""
    j = rule_based_judge_v3(caption, dets, min_conf=conf)
    checked = [c["claim_normalized"] for c in j["claim_results"] if c["in_yolo_vocabulary"]]
    flagged = [c["claim_normalized"] for c in j["claim_results"]
               if c["status"] == "unsupported_by_verifier"]
    feedback = [c for c in checked if c not in gt] if gt is not None else flagged
    return {"checked": checked, "judge_flagged": flagged,
            "oov": [c["claim_normalized"] for c in j["claim_results"] if not c["in_yolo_vocabulary"]],
            "negated": j["claims"]["negated"],
            "hallucination_score": j["hallucination_score"]}, feedback


def ollama_rewriter(model: str):
    import ollama
    digest = next((m.get("digest") for m in ollama.list().get("models", [])
                   if m.get("model", m.get("name")) == model), None)
    state = {"think_flag": True}

    def rewrite(image: str, prompt: str) -> dict:
        kw = dict(model=model, prompt=prompt, images=[image], stream=False,
                  options={"temperature": 0, "seed": SEED, "num_predict": MAX_TOKENS})
        if state["think_flag"]:
            try:
                r = ollama.generate(**kw, think=False)
            except Exception as e:                      # model has no thinking switch
                if "think" not in str(e).lower():
                    raise
                state["think_flag"] = False
                r = ollama.generate(**kw)
        else:
            r = ollama.generate(**kw)
        text = (r.get("response") or "").strip().strip('"').strip()
        return {"caption": text, "gen_tokens": r.get("eval_count"),
                "ollama_total_ms": int((r.get("total_duration") or 0) / 1e6)}
    return rewrite, {"model": model, "digest": digest}


def hf_gemma3n_rewriter(model_id: str = "google/gemma-3n-E2B-it"):
    """Same HF setup as run_captions_unified.gemma_backend (greedy, 160 new tokens)."""
    import torch, transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda, "| torch", torch.__version__)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16 if cuda else torch.float32, device_map="auto")

    def rewrite(image: str, prompt: str) -> dict:
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(image).convert("RGB")},
                                             {"type": "text", "text": prompt}]}]
        inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(model.device, dtype=model.dtype)
        n_in = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=MAX_TOKENS, do_sample=False)
        text = processor.decode(out[0][n_in:], skip_special_tokens=True).strip().strip('"').strip()
        return {"caption": text, "gen_tokens": int(out.shape[-1] - n_in)}
    return rewrite, {"model": model_id, "backend": "huggingface", "cuda": cuda,
                     "torch": torch.__version__, "transformers": transformers.__version__}


def dry_rewriter(image: str, prompt: str) -> dict:          # for testing the loop only
    return {"caption": prompt.split('"')[1], "gen_tokens": 0, "ollama_total_ms": 0}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captioner", required=True, choices=list(OLLAMA_TAG) + ["gemma3n"])
    ap.add_argument("--condition", required=True, choices=["self", "moondream", "control", "oracle"])
    ap.add_argument("--prompt", choices=["A", "B"], default="A",
                    help="feedback prompt: A = main (default), B = control prompt + 'Do not mention X' (robustness)")
    ap.add_argument("--list", default=str(V3 / "data/human_validation_100.txt"))
    ap.add_argument("--detector", default="yolov8x")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--coco_ann", help="needed for --condition oracle")
    ap.add_argument("--images_dir", action="append",
                    default=[str(V3.parent / "val2017"), str(V3 / "val2017"), "val2017"])
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    if a.condition == "oracle" and not a.coco_ann:
        sys.exit("--condition oracle needs --coco_ann")

    round0 = load_jsonl_by_name(V3 / f"results/{a.captioner}.jsonl")
    dets = load_jsonl_by_name(V3 / f"results/detections/{a.detector}.jsonl")
    gt = load_coco_gt(a.coco_ann) if a.condition == "oracle" else None

    if a.dry_run:
        rewrite, minfo = dry_rewriter, {"model": "dry_run"}
    elif a.condition == "moondream":
        rewrite, minfo = ollama_rewriter("moondream:latest")
    elif a.captioner == "gemma3n":                    # no vision gemma3n in Ollama -> HF (CPU, ~1-2 min/rewrite)
        rewrite, minfo = hf_gemma3n_rewriter()
    else:
        rewrite, minfo = ollama_rewriter(OLLAMA_TAG[a.captioner])
    info = {"captioner": a.captioner, "condition": a.condition, "rewriter": minfo,
            "detector": a.detector, "conf": a.conf, "max_rounds": a.max_rounds,
            "seed": SEED, "max_tokens": MAX_TOKENS, "temperature": 0,
            "prompt_variant": a.prompt,
            "feedback_prompt": FEEDBACK_PROMPT if a.prompt == "A" else FEEDBACK_PROMPT_B, "control_prompt": CONTROL_PROMPT,
            "script_version": "refine_v2 (feedback uses caption words)",
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("Run info:", info)

    cond_name = a.condition + ("B" if a.prompt == "B" else "")       # e.g. llava__selfB.jsonl
    out = Path(a.out or V3 / f"runs/refine/{a.captioner}__{cond_name}{'__dry' if a.dry_run else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["image"] for l in out.open()} if out.exists() else set()
    if done:
        print(f"Resuming: {len(done)} images already done")

    names = [Path(l.strip()).name for l in open(a.list) if l.strip()]
    if a.limit:
        names = names[:a.limit]

    with out.open("a") as f:
        for i, name in enumerate(names, 1):
            if name in done:
                continue
            img = find_image(name, a.images_dir)
            d = dets[name]["detections"]
            g = gt.get(name, set()) if gt is not None else None
            cap = round0[name]["caption"]
            j, fb = judge(cap, d, a.conf, g)
            rounds = [{"round": 0, "caption": cap, "feedback": fb, **j}]
            for r in range(1, a.max_rounds + 1):
                if not fb:                               # nothing flagged -> stop
                    break
                fb_words = feedback_text(cap, fb)
                prompt = (CONTROL_PROMPT.format(caption=cap) if a.condition == "control"
                          else (FEEDBACK_PROMPT if a.prompt == "A" else FEEDBACK_PROMPT_B).format(
                              caption=cap, objects=fb_words))
                rounds[-1]["feedback_words"] = fb_words           # what the model was told
                rounds[-1]["feedback_shown"] = a.condition != "control"
                t0 = time.perf_counter()
                try:
                    res = rewrite(img, prompt); err = None
                except Exception as e:                   # logged, never silently replaced
                    res, err = {"caption": ""}, f"{type(e).__name__}: {e}"
                lat = int((time.perf_counter() - t0) * 1000)
                if not res["caption"]:                   # failed rewrite: keep old caption, stop
                    rounds.append({"round": r, "ok": False, "error": err or "empty caption",
                                   "latency_ms": lat})
                    break
                cap = res["caption"]
                j, fb = judge(cap, d, a.conf, g)
                rounds.append({"round": r, "ok": True, "caption": cap, "feedback": fb, **j,
                               "latency_ms": lat, "gen_tokens": res.get("gen_tokens"),
                               "ollama_total_ms": res.get("ollama_total_ms")})
            ok_rounds = [x for x in rounds if x.get("ok", True)]
            rec = {"image": name, "rounds": rounds, "final_round": ok_rounds[-1]["round"],
                   "stopped_clean": not ok_rounds[-1]["feedback"], "run_info": info}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            fl = " -> ".join(str(len(x.get("judge_flagged", []))) if x.get("ok", True) else "ERR"
                             for x in rounds)
            print(f"[{i}/{len(names)}] {name}  flagged per round: {fl}  | {cap[:100]!r}")
            for x in rounds:
                if x.get("feedback_words"):
                    print(f"      round {x['round']} feedback: {x['feedback_words']}")
                if not x.get("ok", True):
                    print(f"      round {x['round']} ERROR: {x['error']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
