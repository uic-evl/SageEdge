"""
Step 4f: query the Judge memory for a new image and caption (end-to-end AI Judge, one call).

For one (image, caption) pair this script
  1. runs the rule judge: YOLOv8x on the image (conf >= 0.05), COCO-class claims of the caption,
     a claim is flagged when the class has no box with conf >= tau (0.25);
  2. embeds the image, every sentence and the caption in the joint open_clip space of the memory
     (SigLIP by default), and the image-sentence cosine of every claim;
  3. retrieves from the memory built by build_memory.py
       per claim  : the k most similar past claims of the same class (joint [image, sentence]),
                    with the similarity-weighted share that the judge flagged, and the top examples;
       per caption: the k most similar past captions (joint [image, caption]), the weighted mean of
                    their judge scores, and their feedback words;
  4. prints a JSON report (and writes it with --out);
  5. with --add, appends the caption, its claims, verdicts and embeddings to the memory
     (sqlite rows, .npy rows, FAISS indices, info.json), so the memory grows as a node judges new captions.
Past examples from the same image file are skipped (--keep_same_image to keep them).
No ground truth is used. Schema: retrieval/MEMORY_SCHEMA.md.

Usage (from v3/, venv_gpu):
  python retrieval/query_memory.py --image ../val2017/000000387387.jpg --caption "A yellow school bus parked on gravel next to a dog."
  python retrieval/query_memory.py --image my.jpg --caption_file cap.txt --captioner llava --model clip --out q.json
  python retrieval/query_memory.py --image my.jpg --caption "..." --captioner llava --add --memdir runs/memory_live
  --cached_detections uses results/detections/<detector>.jsonl when the image is one of the 1,000 evaluation images.
"""
from __future__ import annotations

import argparse, datetime, json, sqlite3, sys, time
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3)); sys.path.insert(0, str(V3 / "retrieval")); sys.path.insert(0, str(V3 / "refine"))
from build_memory import MODELS, TAU, claims_with_sentence, max_conf   # noqa: E402
from judge_v3.rule_judge_v3 import rule_based_judge_v3                  # noqa: E402
from run_refine_loop import feedback_text                               # noqa: E402


def detect(image: Path, detector: str, cached: bool) -> tuple[list, str]:
    if cached:
        for line in open(V3 / f"results/detections/{detector}.jsonl"):
            r = json.loads(line)
            if Path(r["image"]).name == image.name:
                return r["detections"], "cached"
    from ultralytics import YOLO
    import torch
    res = YOLO(str(V3 / f"{detector}.pt"))(str(image), conf=0.05, device=0 if torch.cuda.is_available() else "cpu",
                                           verbose=False)[0]
    return [{"class": res.names[int(b.cls)], "conf": round(float(b.conf), 4)} for b in res.boxes], "yolo"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--caption"); g.add_argument("--caption_file")
    ap.add_argument("--captioner", default="unknown", help="model that wrote the caption (metadata)")
    ap.add_argument("--model", choices=list(MODELS), default="siglip")
    ap.add_argument("--detector", default="yolov8x")
    ap.add_argument("--memdir", default=str(V3 / "runs/memory"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--show", type=int, default=3, help="examples shown per claim / caption")
    ap.add_argument("--keep_same_image", action="store_true")
    ap.add_argument("--cached_detections", action="store_true")
    ap.add_argument("--add", action="store_true", help="append this caption to the memory")
    ap.add_argument("--out")
    a = ap.parse_args()

    import numpy as np, faiss, torch, open_clip
    from PIL import Image

    if a.add and Path(a.memdir).resolve() == (V3 / "runs/memory").resolve():
        sys.exit("--add changes the memory: copy it first (cp -r runs/memory runs/memory_live) and pass --memdir runs/memory_live; "
                 "runs/memory must stay identical to the evaluated version")
    image = Path(a.image); caption = a.caption if a.caption is not None else Path(a.caption_file).read_text().strip()
    md = Path(a.memdir) / a.model
    info = json.loads((md / "info.json").read_text())
    name, pre = MODELS[a.model]
    assert info["model"] == name, f"memory {md} was built with {info['model']}, not {name}"
    t0 = time.perf_counter(); timing = {}

    # 1. rule judge
    dets, det_src = detect(image, a.detector, a.cached_detections)
    conf = max_conf(dets)
    claims, sents = claims_with_sentence(caption)
    flagged = [o for o, _, _ in claims if conf.get(o, 0.0) < TAU]
    j = rule_based_judge_v3(caption, dets, min_conf=TAU)
    assert j["checked_claim_count"] == len(claims) and j["unsupported_claim_count"] == len(flagged)
    score = len(flagged) / len(claims) if claims else None
    timing["judge_s"] = round(time.perf_counter() - t0, 3)

    # 2. embeddings
    t1 = time.perf_counter()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev)
    model.eval(); tok = open_clip.get_tokenizer(name)
    timing["model_load_s"] = round(time.perf_counter() - t1, 3); t2 = time.perf_counter()
    with torch.inference_mode():
        ie = model.encode_image(preprocess(Image.open(image).convert("RGB")).unsqueeze(0).to(dev))
        se = model.encode_text(tok(sents).to(dev))
    nz = lambda e: torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().astype("float32")
    img_e, sent_e = nz(ie)[0], nz(se)
    cap_e = sent_e.mean(0); cap_e = (cap_e / max(np.linalg.norm(cap_e), 1e-8)).astype("float32")
    claim_e = []
    for o, k, s in claims:
        if k >= 0:
            claim_e.append(sent_e[k])
        else:
            with torch.inference_mode():
                claim_e.append(nz(model.encode_text(tok([s]).to(dev)))[0])
    claim_e = np.array(claim_e, dtype="float32").reshape(len(claims), -1)
    timing["embed_s"] = round(time.perf_counter() - t2, 3)

    # 3. memory retrieval
    t3 = time.perf_counter()
    con = sqlite3.connect(md / "memory.sqlite"); con.row_factory = sqlite3.Row
    mem_claim = np.load(md / "claim_emb.npy"); mem_img = np.load(md / "image_emb.npy")
    mem_cap = np.load(md / "caption_emb.npy")
    cap_rows = [dict(r) for r in con.execute("SELECT row_id, image, captioner, judge_score, feedback_words, caption FROM captions ORDER BY row_id")]
    img_of_cap = {r["row_id"]: r for r in cap_rows}
    skip = lambda f: (not a.keep_same_image) and f == image.name

    out_claims = []
    for (o, k, s), ce in zip(claims, claim_e):
        rows = [dict(r) for r in con.execute(
            "SELECT row_id, caption_row, image, captioner, sentence, flagged, yolo_conf FROM claims WHERE class = ?", (o,))]
        rows = [r for r in rows if not skip(r["image"])]
        rep = {"class": o, "sentence": s, "yolo_conf": round(conf.get(o, 0.0), 4), "flagged": o in flagged,
               "img_sent_cos": round(float(img_e @ ce), 5), "memory_n_same_class": len(rows)}
        if rows:
            q = np.concatenate([img_e, ce]) / np.sqrt(2)
            M = np.concatenate([mem_img[[r["caption_row"] for r in rows]], mem_claim[[r["row_id"] for r in rows]]], 1) / np.sqrt(2)
            sim = M @ q; top = np.argsort(-sim)[:a.k]; w = np.clip(sim[top], 1e-6, None)
            rep["memory_flag_share"] = round(float((w * np.array([rows[i]["flagged"] for i in top])).sum() / w.sum()), 4)
            rep["neighbours"] = [{"sim": round(float(sim[i]), 4), "image": rows[i]["image"], "captioner": rows[i]["captioner"],
                                  "flagged": bool(rows[i]["flagged"]), "sentence": rows[i]["sentence"]} for i in top[:a.show]]
        else:
            rep["memory_flag_share"] = None; rep["neighbours"] = []
        out_claims.append(rep)

    index = faiss.read_index(str(md / "joint_caption.index"))
    q = (np.concatenate([img_e, cap_e]) / np.sqrt(2)).astype("float32")[None]
    sims, nn = index.search(q, min(index.ntotal, a.k + 8))
    nbr = [(float(s), int(i)) for s, i in zip(sims[0], nn[0]) if i >= 0 and not skip(img_of_cap[int(i)]["image"])][:a.k]
    scored = [(s, img_of_cap[i]["judge_score"]) for s, i in nbr if img_of_cap[i]["judge_score"] is not None]
    w = np.clip([s for s, _ in scored], 1e-6, None) if scored else None
    timing["retrieve_s"] = round(time.perf_counter() - t3, 3)

    fb = feedback_text(caption, flagged) if flagged else None
    report = {
        "image": str(image), "captioner": a.captioner, "memory": str(md), "memory_model": name,
        "detector": a.detector, "tau": TAU, "detections_from": det_src,
        "judge": {"n_checked": len(claims), "n_flagged": len(flagged), "judge_score": score,
                  "flagged_classes": flagged, "feedback_words": fb,
                  "out_of_vocabulary": [c["claim_normalized"] for c in j["claim_results"] if not c["in_yolo_vocabulary"]]},
        "img_caption_cos": round(float(img_e @ cap_e), 5),
        "claims": out_claims,
        "caption_memory": {"k": len(nbr),
                           "weighted_mean_judge_score": round(float((w * np.array([v for _, v in scored])).sum() / w.sum()), 4) if scored else None,
                           "neighbours": [{"sim": round(s, 4), "image": img_of_cap[i]["image"], "captioner": img_of_cap[i]["captioner"],
                                           "judge_score": img_of_cap[i]["judge_score"], "feedback_words": img_of_cap[i]["feedback_words"],
                                           "caption": (img_of_cap[i]["caption"] or "")[:200]} for s, i in nbr[:a.show]]},
        "timing": {**timing, "total_s": round(time.perf_counter() - t0, 3)},
        "note": "The memory priors are context only. The verdict is the rule judge (eval_memory.py: the memory adds no "
                "significant F1 over the image-bank + text judge with SigLIP).",
    }

    # 5. optional: grow the memory
    if a.add:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ccols = [r[1] for r in con.execute("PRAGMA table_info(captions)")]
        kcols = [r[1] for r in con.execute("PRAGMA table_info(claims)")]
        crow = int(con.execute("SELECT COALESCE(MAX(row_id), -1) + 1 FROM captions").fetchone()[0])
        krow0 = int(con.execute("SELECT COALESCE(MAX(row_id), -1) + 1 FROM claims").fetchone()[0])
        assert crow == len(mem_cap) and krow0 == len(mem_claim), "memory rows and embeddings out of sync"
        cvals = {"row_id": crow, "image": image.name, "captioner": a.captioner, "backend": None, "model": a.captioner,
                 "digest": None, "prompt": None, "seed": None, "max_tokens": None, "date": now, "caption": caption,
                 "n_sentences": len(sents), "n_checked": len(claims), "n_flagged": len(flagged), "judge_score": score,
                 "detector": a.detector, "tau": TAU, "loop_run": None,
                 "feedback_classes": json.dumps(flagged) if flagged else None, "feedback_words": fb,
                 "loop_rounds": None, "final_caption": None, "final_judge_score": None, "loop_date": None,
                 "img_cap_cos": round(float(img_e @ cap_e), 5)}
        con.execute(f"INSERT INTO captions ({','.join(ccols)}) VALUES ({','.join('?' * len(ccols))})", [cvals.get(c) for c in ccols])
        for n_, ((o, k, s), rep) in enumerate(zip(claims, out_claims)):
            kv = {"row_id": krow0 + n_, "caption_row": crow, "image": image.name, "captioner": a.captioner, "class": o,
                  "sentence_idx": k, "sentence": s, "yolo_conf": rep["yolo_conf"], "flagged": int(rep["flagged"]),
                  "img_sent_cos": rep["img_sent_cos"]}
            con.execute(f"INSERT INTO claims ({','.join(kcols)}) VALUES ({','.join('?' * len(kcols))})", [kv.get(c) for c in kcols])
        np.save(md / "caption_emb.npy", np.vstack([mem_cap, cap_e[None]]))
        np.save(md / "image_emb.npy", np.vstack([mem_img, img_e[None]]))
        np.save(md / "claim_emb.npy", np.vstack([mem_claim, claim_e]) if len(claims) else mem_claim)
        index.add(q); faiss.write_index(index, str(md / "joint_caption.index"))
        if len(claims):
            ki = faiss.read_index(str(md / "joint_claim.index"))
            ki.add((np.concatenate([np.repeat(img_e[None], len(claims), 0), claim_e], 1) / np.sqrt(2)).astype("float32"))
            faiss.write_index(ki, str(md / "joint_claim.index"))
        con.commit()
        info["captions"] += 1; info["claims"] += len(claims); info["flagged_claims"] += len(flagged)
        info.setdefault("added", []).append({"image": image.name, "captioner": a.captioner, "date": now})
        (md / "info.json").write_text(json.dumps(info, indent=2))
        report["added"] = {"caption_row": crow, "claim_rows": list(range(krow0, krow0 + len(claims)))}
    con.close()

    s = json.dumps(report, indent=2, ensure_ascii=False)
    print(s)
    if a.out:
        Path(a.out).write_text(s)


if __name__ == "__main__":
    main()
