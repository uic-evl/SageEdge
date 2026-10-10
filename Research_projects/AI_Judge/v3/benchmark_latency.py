"""
Clean latency benchmark on the Jetson Thor (run with NOTHING else running).

Measures, on the same N images:
  * YOLOv8 n/s/m/x detection latency on GPU (and optionally CPU)
  * rule-judge time (parser v3 + matching), given captions
  * optionally the VLM judge (Qwen3-VL-32B via Ollama) on a few captions
Reports median / p90 latency and peak GPU memory, and saves JSON.

Usage (venv with CUDA torch active, from the AI_Judge folder):
  sudo nvpmodel -q  > runs/v3/bench_powermode.txt       # record the power mode
  python benchmark_latency.py --list human_validation_100.txt --captions runs/v3/llava.jsonl \
         --vlm_n 20 --out runs/v3/benchmark.json
"""
from __future__ import annotations

import argparse, json, platform, statistics as st, time
from pathlib import Path


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summary(ms):
    return {"n": len(ms), "median_ms": round(st.median(ms), 2), "p90_ms": round(pct(ms, 0.9), 2),
            "mean_ms": round(st.mean(ms), 2)}


def bench_yolo(weights, imgs, device, warmup=5):
    import torch
    from ultralytics import YOLO
    model = YOLO(weights)
    for im in imgs[:warmup]:
        model(im, conf=0.25, device=device, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ms = []
    for im in imgs:
        t0 = time.perf_counter()
        model(im, conf=0.25, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    out = summary(ms)
    if device != "cpu":
        out["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--captions", required=True, help="caption file used to time the rule judge / VLM judge")
    ap.add_argument("--out", default="runs/v3/benchmark.json")
    ap.add_argument("--cpu_too", action="store_true", help="also time YOLO on CPU")
    ap.add_argument("--vlm", default="qwen3-vl:32b")
    ap.add_argument("--vlm_n", type=int, default=0, help="captions to time with the VLM judge (0 = skip)")
    a = ap.parse_args()

    import torch
    res = {"host": platform.node(), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        res["power_mode"] = Path("runs/v3/bench_powermode.txt").read_text().strip()
    except OSError:
        res["power_mode"] = "not recorded"
    print(res)

    imgs = [str(Path(l.strip()).resolve()) for l in open(a.list) if l.strip()]
    devices = (["cuda:0"] if torch.cuda.is_available() else []) + (["cpu"] if a.cpu_too or not torch.cuda.is_available() else [])
    res["yolo"] = {}
    for d in devices:
        for w in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"]:
            r = bench_yolo(w, imgs, d)
            res["yolo"][f"{Path(w).stem}@{d}"] = r
            print(f"{w:12s} {d:6s} {r}")

    # rule judge (parser + matching) on captions of the same images
    from judge_v3.rule_judge_v3 import rule_based_judge_v3
    keep = {Path(i).name for i in imgs}
    caps = [json.loads(l) for l in open(a.captions) if l.strip()]
    caps = [c for c in caps if Path(c["image"]).name in keep]
    fake = [{"class": "person", "conf": 0.9}, {"class": "car", "conf": 0.8}]
    ms = []
    for c in caps:
        t0 = time.perf_counter(); rule_based_judge_v3(c["caption"], fake); ms.append((time.perf_counter() - t0) * 1000)
    res["rule_judge_parse_match"] = summary(ms)
    print("rule judge (parse + match):", res["rule_judge_parse_match"])

    if a.vlm_n:
        import subprocess, sys
        tmp = Path(a.out).with_suffix(".vlm_tmp.jsonl")
        if tmp.exists():
            tmp.unlink()
        lst = Path(a.out).with_suffix(".vlm_imgs.txt")
        lst.write_text("\n".join(Path(c["image"]).name for c in caps[:a.vlm_n]))
        subprocess.run([sys.executable, "run_vlm_judge.py", "--judge", a.vlm, "--captions", a.captions,
                        "--images", str(lst), "--out", str(tmp)], check=True)
        recs = [json.loads(l) for l in open(tmp)]
        res["vlm_judge"] = {"model": a.vlm, **summary([r["latency_ms"] for r in recs[1:] if r["latency_ms"]])}
        res["vlm_judge"]["note"] = "first call (model load) excluded"
        print("VLM judge:", res["vlm_judge"])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
