"""
Step 4a: labeled image memory bank for retrieval (COCO train2017) + image embeddings.

  1. sample --n_bank COCO train2017 images (seed 0) and store their GT classes
     (the 1,000 evaluation images are val2017, so they can never be in the bank; checked)
  2. download the sampled images (resumable, only the files that are missing)
  3. embed bank + evaluation images with an open_clip model on the GPU
     (L2-normalised float32, so inner product = cosine similarity)
  4. benchmark on the Thor: embedding latency per image (batch 1), FAISS index memory
     and search latency (one query, exact inner product), peak GPU memory

Models: siglip = ViT-B-16-SigLIP (webli), clip = ViT-B-32-quickgelu (openai; the OpenAI weights were trained with QuickGELU).

Usage (from v3/, venv_gpu active, run alone for clean timing):
  python retrieval/build_bank.py --model siglip --limit 200        # quick test
  python retrieval/build_bank.py --model siglip                    # full (download ~1.6 GB once)
  python retrieval/build_bank.py --model clip                      # second model, reuses the images
  Running it again when the embeddings exist only repeats the timing benchmark
  (do that once with nothing else running on the Thor, for the paper numbers).
Outputs: runs/retrieval/bank_meta.jsonl, runs/retrieval/<model>/{bank,eval}_emb.npy, *_files.txt, benchmark.json
"""
from __future__ import annotations

import argparse, json, random, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
MODELS = {"siglip": ("ViT-B-16-SigLIP", "webli"), "clip": ("ViT-B-32-quickgelu", "openai")}
URL = "http://images.cocodataset.org/train2017/{}"


def sample_bank(ann_path: Path, n: int, seed: int, eval_names: set[str], meta_path: Path):
    if meta_path.exists():
        meta = [json.loads(l) for l in open(meta_path)]
        if len(meta) >= n:
            print(f"bank meta exists: {len(meta)} images ({meta_path})")
            return meta[:n]
    ann = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    classes: dict[int, set[str]] = {im["id"]: set() for im in ann["images"]}
    for a in ann["annotations"]:
        classes[a["image_id"]].add(cats[a["category_id"]])
    ims = sorted(ann["images"], key=lambda im: im["id"])
    random.Random(seed).shuffle(ims)
    meta = [{"id": im["id"], "file": im["file_name"], "classes": sorted(classes[im["id"]])} for im in ims[:n]]
    overlap = {m["file"] for m in meta} & eval_names
    assert not overlap, f"evaluation images in the bank: {sorted(overlap)[:5]}"
    with open(meta_path, "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"sampled {len(meta)} train2017 images (seed {seed}); "
          f"{sum(not m['classes'] for m in meta)} have no annotated object")
    return meta


def download(files: list[str], img_dir: Path, workers: int = 16):
    img_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (img_dir / f).exists()]
    print(f"download: {len(files) - len(todo)} present, {len(todo)} to fetch")

    def get(f):
        for attempt in range(3):
            try:
                tmp = img_dir / (f + ".part")
                urllib.request.urlretrieve(URL.format(f), tmp)
                tmp.rename(img_dir / f)
                return None
            except Exception as e:                  # retried, then reported
                err = f"{f}: {type(e).__name__}: {e}"
                time.sleep(1 + attempt)
        return err

    errors = []
    with ThreadPoolExecutor(workers) as ex:
        for i, e in enumerate(ex.map(get, todo), 1):
            if e:
                errors.append(e)
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}")
    if errors:
        print(f"{len(errors)} downloads failed, e.g. {errors[:3]} -> run again to retry")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="siglip")
    ap.add_argument("--n_bank", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ann_train", default=str(V3.parent / "annotations/instances_train2017.json"))
    ap.add_argument("--train_dir", default=str(V3.parent / "train2017_bank"))
    ap.add_argument("--val_dir", default=str(V3.parent / "val2017"))
    ap.add_argument("--list", default=str(V3 / "data/coco_1000.txt"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="test: only this many bank and eval images")
    ap.add_argument("--force", action="store_true", help="recompute embeddings even if they exist")
    ap.add_argument("--outdir", default=str(V3 / "runs/retrieval"))
    a = ap.parse_args()

    import numpy as np, torch, open_clip, faiss
    from PIL import Image

    out = Path(a.outdir); (out / a.model).mkdir(parents=True, exist_ok=True)
    eval_files = [Path(l.strip()).name for l in open(a.list) if l.strip()]
    meta = sample_bank(Path(a.ann_train), a.n_bank, a.seed, set(eval_files), out / "bank_meta.jsonl")
    if a.limit:
        meta, eval_files = meta[:a.limit], eval_files[:a.limit]
    errors = download([m["file"] for m in meta], Path(a.train_dir))
    if errors:
        raise SystemExit("some images are missing; run the same command again")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name, pre = MODELS[a.model]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev)
    model.eval()
    info = {"model": name, "pretrained": pre, "device": dev, "torch": torch.__version__,
            "open_clip": open_clip.__version__, "faiss": faiss.__version__}
    print("model:", info)

    def embed(paths: list[Path]) -> np.ndarray:
        outs = []
        for i in range(0, len(paths), a.batch):
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + a.batch]]).to(dev)
            with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
                e = model.encode_image(x)
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
            if (i // a.batch) % 20 == 0:
                print(f"  embedded {min(i + a.batch, len(paths))}/{len(paths)}")
        return np.concatenate(outs).astype("float32")

    suffix = f"_limit{a.limit}" if a.limit else ""
    fb, fe = out / a.model / f"bank_emb{suffix}.npy", out / a.model / f"eval_emb{suffix}.npy"
    if fb.exists() and fe.exists() and not a.force:      # embeddings done: only re-run the benchmark
        bank, ev = np.load(fb), np.load(fe)
        print(f"embeddings exist ({len(bank)} bank, {len(ev)} eval): benchmark only (use --force to recompute)")
    else:
        t0 = time.time()
        bank = embed([Path(a.train_dir) / m["file"] for m in meta])
        ev = embed([Path(a.val_dir) / f for f in eval_files])
        print(f"embedded {len(bank)} bank + {len(ev)} eval images in {time.time() - t0:.0f} s")
        np.save(fb, bank)
        np.save(fe, ev)
    (out / a.model / f"bank_files{suffix}.txt").write_text("\n".join(m["file"] for m in meta) + "\n")
    (out / a.model / f"eval_files{suffix}.txt").write_text("\n".join(eval_files) + "\n")

    # ---------------- benchmark (batch 1, the way the judge would run per image)
    bench = {**info, "n_bank": len(bank), "dim": int(bank.shape[1])}
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    lat = []
    for i, f in enumerate(eval_files[:110]):
        img = Image.open(Path(a.val_dir) / f).convert("RGB")
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        x = preprocess(img).unsqueeze(0).to(dev)
        with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
            model.encode_image(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= 10:                                    # warm-up excluded
            lat.append((time.perf_counter() - t) * 1000)
    bench["embed_ms_median"] = round(statistics.median(lat), 2)
    bench["embed_ms_p90"] = round(sorted(lat)[int(.9 * len(lat)) - 1], 2)
    bench["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev == "cuda" else None
    t = time.perf_counter(); index = faiss.IndexFlatIP(bank.shape[1]); index.add(bank)
    bench["faiss_build_ms"] = round((time.perf_counter() - t) * 1000, 2)
    bench["faiss_index_mb"] = round(bank.nbytes / 2**20, 2)
    s = []
    for q in ev[:1000]:
        t = time.perf_counter(); index.search(q[None], 50); s.append((time.perf_counter() - t) * 1000)
    bench["faiss_search_ms_median_k50"] = round(statistics.median(s), 3)
    bench["faiss_search_ms_p90_k50"] = round(sorted(s)[int(.9 * len(s)) - 1], 3)
    (out / a.model / f"benchmark{suffix}.json").write_text(json.dumps(bench, indent=2))
    print("benchmark:", json.dumps(bench, indent=2))
    print(f"saved {out / a.model}")


if __name__ == "__main__":
    main()
