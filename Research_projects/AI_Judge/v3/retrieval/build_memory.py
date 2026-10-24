"""
Step 4d: the Judge memory, i.e. the vector database of the project plan.

The plan asks for a vector DB (FAISS or Milvus) that stores (1) image and caption embeddings,
(2) AI Judge scores and feedback, (3) metadata such as the model and the date, and that supports
k-NN search over past examples with known judgments. This script builds that memory for the
4,000 captions of the 1,000 evaluation images (4 captioners):

  embeddings (same open_clip model as the retrieval layer, so image and text share one space)
    image    : runs/retrieval/<model>/eval_emb.npy (already computed by build_bank.py)
    caption  : mean of the sentence embeddings of the caption (L2-normalised)
    claim    : embedding of the first sentence that states the claim ("a black dog lies on an armchair")
  judgments : YOLOv8x rule judge (tau 0.25) on every COCO-class claim: max confidence, flagged yes/no,
              and the caption-level hallucination score (flagged / checked), cross-checked against
              judge_v3.rule_judge_v3
  feedback  : the judge feedback of the loop (runs/refine/<cap>__self.jsonl, round 0): flagged classes,
              the words sent to the captioner, the number of rounds, the final caption and its score.
              Only the valid self runs are used (LLaVA, Gemma 4, Gemma 3n); Moondream feedback runs are excluded.
  metadata  : captioner, backend, model name, model digest, prompt, seed, token limit, date of the run
No ground truth is stored: the memory is label-free and could be filled on a Sage node.

Storage (runs/memory/<model>/), schema in retrieval/MEMORY_SCHEMA.md:
  memory.sqlite      tables captions, claims
  caption_emb.npy    row i = captions.row_id i      claim_emb.npy  row j = claims.row_id j
  image_emb.npy      row i = image of captions.row_id i
  joint_caption.index / joint_claim.index   FAISS IndexFlatIP over [image, text] / sqrt(2)
  info.json          model, counts, date

Usage (from v3/, venv_gpu, after build_bank.py --model <model>):
  python retrieval/build_memory.py --model siglip
  python retrieval/build_memory.py --model clip
"""
from __future__ import annotations

import argparse, datetime, json, re, sqlite3, sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from judge_v3.parse_caption_v3 import COCO_SET, parse_caption   # noqa: E402
from judge_v3.rule_judge_v3 import rule_based_judge_v3          # noqa: E402

CAPTIONERS = ["gemma3n", "gemma4", "llava", "moondream"]
LOOP_SELF_VALID = {"gemma3n", "gemma4", "llava"}     # Moondream feedback runs are invalid (run_provenance.md)
MODELS = {"siglip": ("ViT-B-16-SigLIP", "webli"), "clip": ("ViT-B-32-quickgelu", "openai")}
TAU = 0.25


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def sentences(caption: str) -> list[str]:
    """Split a caption into sentences; Markdown bullets and bold markers are removed."""
    text = re.sub(r"\*\*|__|#+ ", "", caption or "")
    parts = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*([*\-•]|\d+\.)\s+", "", line).strip()
        if line:
            parts += [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    return parts or [caption or ""]


def claims_with_sentence(caption: str):
    """COCO-class claims of the whole caption (same set as the judge) and the first sentence that states each.
    Returns [(class, sentence index or -1, sentence text)] and the sentence list."""
    objs = [o for o in parse_caption(caption)["objects"] if o in COCO_SET]
    sents = sentences(caption)
    per_sent = [set(parse_caption(s)["objects"]) for s in sents]
    out = []
    for o in objs:
        k = next((i for i, ps in enumerate(per_sent) if o in ps), None)
        out.append((o, k if k is not None else -1, sents[k] if k is not None else (caption or "")[:300]))
    return out, sents


def max_conf(dets) -> dict[str, float]:
    m: dict[str, float] = {}
    for x in dets:
        m[x["class"]] = max(m.get(x["class"], 0.0), float(x["conf"]))
    return m


def load_records(detector: str):
    """Captions, claims and sentences of the memory (no embeddings). Shared by build_memory and the tests."""
    det_raw = {Path(r["image"]).name: r["detections"] for r in jl(V3 / f"results/detections/{detector}.jsonl")}
    det = {f: max_conf(d) for f, d in det_raw.items()}
    caps, cls, sent_texts = [], [], []
    n_loop_mismatch = 0
    for cap in CAPTIONERS:
        loop = {}
        lp = V3 / f"runs/refine/{cap}__self.jsonl"
        if cap in LOOP_SELF_VALID and lp.exists():
            loop = {Path(r["image"]).name: r for r in jl(lp)}
        for r in jl(V3 / f"results/{cap}.jsonl"):
            f = Path(r["image"]).name
            caption = r.get("caption") or ""
            claims, sents = claims_with_sentence(caption)
            ri = r.get("run_info", {})
            conf = det.get(f, {})
            flagged = [o for o, _, _ in claims if conf.get(o, 0.0) < TAU]
            j = rule_based_judge_v3(caption, det_raw.get(f, []), min_conf=TAU)    # cross-check
            assert j["checked_claim_count"] == len(claims) and j["unsupported_claim_count"] == len(flagged), f
            lr = loop.get(f)
            if lr is not None and lr["rounds"][0].get("caption") != caption:
                n_loop_mismatch += 1; lr = None                 # loop started from another caption: do not attach
            fin = None
            if lr is not None:
                fin = next(x for x in lr["rounds"] if x["round"] == lr["final_round"] and x.get("ok", True))
            r0 = lr["rounds"][0] if lr is not None else None
            row = len(caps)
            caps.append({"row_id": row, "image": f, "captioner": cap, "backend": ri.get("backend"),
                         "model": ri.get("model"), "digest": ri.get("digest"), "prompt": ri.get("prompt"),
                         "seed": ri.get("seed"), "max_tokens": ri.get("max_tokens"), "date": ri.get("started"),
                         "caption": caption, "n_sentences": len(sents), "n_checked": len(claims),
                         "n_flagged": len(flagged),
                         "judge_score": (len(flagged) / len(claims)) if claims else None,
                         "detector": detector, "tau": TAU,
                         "loop_run": f"{cap}__self" if lr is not None else None,
                         "feedback_classes": json.dumps(r0["feedback"]) if r0 is not None and r0["feedback"] else None,
                         "feedback_words": r0.get("feedback_words") if r0 is not None else None,
                         "loop_rounds": lr["final_round"] if lr is not None else None,
                         "final_caption": fin["caption"] if fin is not None else None,
                         "final_judge_score": fin.get("hallucination_score") if fin is not None else None,
                         "loop_date": lr["run_info"].get("started") if lr is not None else None})
            first = len(sent_texts)
            sent_texts += sents
            caps[-1]["_sent_rows"] = (first, len(sent_texts))
            for o, k, s in claims:
                c = conf.get(o, 0.0)
                cls.append({"row_id": len(cls), "caption_row": row, "image": f, "captioner": cap, "class": o,
                            "sentence_idx": k, "sentence": s, "yolo_conf": round(c, 4),
                            "flagged": int(c < TAU), "_sent_row": (first + k) if k >= 0 else None})
    return caps, cls, sent_texts, n_loop_mismatch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="siglip")
    ap.add_argument("--detector", default="yolov8x")
    ap.add_argument("--retrdir", default=str(V3 / "runs/retrieval"))
    ap.add_argument("--outdir", default=str(V3 / "runs/memory"))
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    import numpy as np, faiss, torch, open_clip

    rd = Path(a.retrdir) / a.model
    eval_emb = np.load(rd / "eval_emb.npy")
    eval_files = (rd / "eval_files.txt").read_text().split()
    row_of = {f: i for i, f in enumerate(eval_files)}
    out = Path(a.outdir) / a.model; out.mkdir(parents=True, exist_ok=True)

    caps, cls, sent_texts, n_mis = load_records(a.detector)
    missing = {c["image"] for c in caps} - set(row_of)
    assert not missing, f"{len(missing)} caption images have no image embedding"
    n_loop = sum(c["loop_run"] is not None for c in caps)
    print(f"memory: {len(caps)} captions, {len(cls)} COCO-class claims, {len(sent_texts)} sentences; "
          f"loop records attached {n_loop} (round-0 caption mismatch {n_mis})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name, pre = MODELS[a.model]
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(name)

    def embed_text(texts):
        outs = []
        for i in range(0, len(texts), a.batch):
            with torch.inference_mode():
                e = model.encode_text(tok(texts[i:i + a.batch]).to(dev))
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(outs).astype("float32")

    sent_emb = embed_text(sent_texts)
    assert sent_emb.shape[1] == eval_emb.shape[1], "text and image embeddings differ in size"
    fallback = [c for c in cls if c["_sent_row"] is None]
    fb_emb = embed_text([c["sentence"] for c in fallback]) if fallback else None
    norm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
    cap_emb = norm(np.stack([sent_emb[s:e].mean(0) for s, e in (c["_sent_rows"] for c in caps)])).astype("float32")
    claim_emb = np.zeros((len(cls), sent_emb.shape[1]), dtype="float32"); j = 0
    for c in cls:
        if c["_sent_row"] is None:
            claim_emb[c["row_id"]] = fb_emb[j]; j += 1
        else:
            claim_emb[c["row_id"]] = sent_emb[c["_sent_row"]]
    img_emb = eval_emb[[row_of[c["image"]] for c in caps]].astype("float32")
    img_claim = eval_emb[[row_of[c["image"]] for c in cls]].astype("float32")
    for c in cls:
        c["img_sent_cos"] = round(float(img_claim[c["row_id"]] @ claim_emb[c["row_id"]]), 5)
    for c in caps:
        c["img_cap_cos"] = round(float(img_emb[c["row_id"]] @ cap_emb[c["row_id"]]), 5)

    np.save(out / "caption_emb.npy", cap_emb); np.save(out / "claim_emb.npy", claim_emb); np.save(out / "image_emb.npy", img_emb)
    jc = (np.concatenate([img_emb, cap_emb], 1) / np.sqrt(2)).astype("float32")
    jk = (np.concatenate([img_claim, claim_emb], 1) / np.sqrt(2)).astype("float32")
    for arr, fn in ((jc, "joint_caption.index"), (jk, "joint_claim.index")):
        idx = faiss.IndexFlatIP(arr.shape[1]); idx.add(arr); faiss.write_index(idx, str(out / fn))

    db = out / "memory.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    ccols = [k for k in caps[0] if not k.startswith("_")]
    kcols = [k for k in cls[0] if not k.startswith("_")]
    con.execute(f"CREATE TABLE captions ({', '.join(c + (' INTEGER PRIMARY KEY' if c == 'row_id' else '') for c in ccols)})")
    con.execute(f"CREATE TABLE claims ({', '.join(c + (' INTEGER PRIMARY KEY' if c == 'row_id' else '') for c in kcols)})")
    con.executemany(f"INSERT INTO captions VALUES ({','.join('?' * len(ccols))})", [[c[k] for k in ccols] for c in caps])
    con.executemany(f"INSERT INTO claims VALUES ({','.join('?' * len(kcols))})", [[c[k] for k in kcols] for c in cls])
    con.execute("CREATE INDEX claims_class ON claims(class)"); con.execute("CREATE INDEX claims_image ON claims(image)")
    con.execute("CREATE INDEX captions_image ON captions(image)")
    con.commit(); con.close()
    info = {"model": name, "pretrained": pre, "detector": a.detector, "tau": TAU, "captions": len(caps),
            "claims": len(cls), "flagged_claims": sum(c["flagged"] for c in cls), "sentences": len(sent_texts),
            "claims_without_own_sentence": len(fallback), "loop_records": n_loop,
            "loop_round0_mismatch": n_mis, "dim_text": int(sent_emb.shape[1]), "dim_joint": int(jk.shape[1]),
            "mean_img_sent_cos": round(float(np.mean([c["img_sent_cos"] for c in cls])), 4),
            "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
