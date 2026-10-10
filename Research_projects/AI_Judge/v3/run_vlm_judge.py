"""
VLM-as-a-judge baseline on the Thor.

For every caption, the object claims are extracted with the SAME parser the
YOLO judge uses (judge_v3/parse_caption_v3.py). A large VLM then looks at the
image and answers yes/no for every claim in ONE call (structured JSON output).
This gives a claim-by-claim comparison with the YOLO rule judge:
same claims, same images -> accuracy vs COCO GT / human labels, and speed.

Usage (from the AI_Judge folder, venv active, AFTER the caption runs finish):
  python run_vlm_judge.py --judge qwen3-vl:32b --captions runs/v3/moondream.jsonl \
         --out runs/v3/judge_qwen3vl32b_moondream.jsonl --limit 10

Resumable: re-run the same command to continue.
"""
from __future__ import annotations

import argparse, json, platform, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "judge_v3"))
from parse_caption_v3 import COCO_SET, parse_caption  # noqa: E402

# Friendlier wording for a few COCO class names inside the question.
ASK_AS = {"dining table": "table", "tv": "television", "sports ball": "ball",
          "potted plant": "potted plant", "cell phone": "cell phone (mobile phone)",
          "hair drier": "hair dryer", "skis": "skis", "person": "person"}

PROMPT = (
    "Look carefully at the image. For each object in the list below, answer \"yes\" if at "
    "least one such object is clearly visible in the image, otherwise answer \"no\". "
    "Judge only from the image.\nObjects: {objects}\n"
    "Return JSON with exactly these keys: {keys}"
)


def build_call(objects):
    keys = [o.replace(" ", "_") for o in objects]
    schema = {"type": "object",
              "properties": {k: {"type": "string", "enum": ["yes", "no"]} for k in keys},
              "required": keys}
    listing = "; ".join(f"{k} = {ASK_AS.get(o, o)}" for k, o in zip(keys, objects))
    return PROMPT.format(objects=listing, keys=", ".join(keys)), schema, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3-vl:32b", help="Ollama vision model used as judge")
    ap.add_argument("--captions", required=True, help="caption file from run_captions_unified.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images", help="optional list file: judge only these images (e.g. human_validation_100.txt)")
    ap.add_argument("--coco_only", action="store_true",
                    help="judge only COCO-class claims (default: also sky/tree/wall ...)")
    a = ap.parse_args()

    import ollama
    digest = next((m.get("digest") for m in ollama.list().get("models", [])
                   if m.get("model", m.get("name")) == a.judge), None)
    info = {"judge": a.judge, "digest": digest, "host": platform.node(),
            "captions_file": a.captions, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("Judge info:", info)

    caps = [json.loads(l) for l in open(a.captions) if l.strip()]
    caps = [c for c in caps if c.get("ok") and c.get("caption")]
    if a.images:
        keep = {Path(l.strip()).name for l in open(a.images) if l.strip()}
        caps = [c for c in caps if Path(c["image"]).name in keep]
        print(f"Restricted to {len(caps)} captions from {a.images}")
    if a.limit:
        caps = caps[:a.limit]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["image"] for l in out.open()} if out.exists() else set()

    use_think_flag = True
    with out.open("a") as f:
        for i, c in enumerate(caps, 1):
            if c["image"] in done:
                continue
            claims = parse_caption(c["caption"])["objects"]
            if a.coco_only:
                claims = [x for x in claims if x in COCO_SET]
            rec = {"image": c["image"], "caption": c["caption"], "claims": claims,
                   "captioner": c.get("run_info", {}).get("model"), "judge_info": info}
            if not claims:
                rec.update(verdicts={}, latency_ms=0, ok=True)
            else:
                prompt, schema, keys = build_call(claims)
                kw = dict(model=a.judge, prompt=prompt, images=[c["image"]], format=schema,
                          stream=False, options={"temperature": 0, "seed": 42, "num_predict": 300,
                                   "num_ctx": 4096})   # default 262k context -> 92 GB, part on CPU
                t0 = time.perf_counter()
                try:
                    try:
                        r = ollama.generate(**kw, think=False) if use_think_flag else ollama.generate(**kw)
                    except Exception as e:           # model without a thinking switch
                        if "think" not in str(e).lower():
                            raise
                        use_think_flag = False
                        r = ollama.generate(**kw)
                    # With structured output, Ollama sometimes routes the JSON of
                    # thinking models (qwen3-vl) into the "thinking" field and leaves
                    # "response" empty -> take the JSON from whichever field has it.
                    resp, think = r.get("response") or "", r.get("thinking") or ""
                    rec["raw"] = resp if resp.strip() else think
                    rec["raw_source"] = "response" if resp.strip() else "thinking"
                    rec["thinking_chars"] = len(think)
                    rec["done_reason"] = r.get("done_reason")
                    m = re.search(r"\{.*\}", rec["raw"], re.S)
                    ans = json.loads(m.group(0) if m else rec["raw"])
                    rec["verdicts"] = {o: ans.get(k) for o, k in zip(claims, keys)}
                    rec["ok"] = all(v in ("yes", "no") for v in rec["verdicts"].values())
                except Exception as e:
                    rec.update(verdicts={}, ok=False, error=f"{type(e).__name__}: {e}")
                rec["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            extra = "" if rec.get("ok") else f"  raw={rec.get('raw', '')[:80]!r} thinking_chars={rec.get('thinking_chars')} done={rec.get('done_reason')}"
            print(f"[{i}/{len(caps)}] {rec['latency_ms']} ms  {rec.get('verdicts') or rec.get('error')}{extra}")


if __name__ == "__main__":
    main()
