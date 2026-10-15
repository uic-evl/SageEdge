"""
Energy per caption on the Jetson AGX Thor, measured with tegrastats.

Runs, one after another, while tegrastats logs the power rails:
  1. idle            (nothing running, --idle_s seconds)
  2. YOLOv8n judge   (full judge: detection + parsing + matching, GPU)
  3. YOLOv8x judge   (same)
  4. Qwen3-VL-32B    (same claims and prompt as run_vlm_judge.py, GPU via Ollama)
For every phase it reports the mean power of each rail, and the energy per caption:
  total  = mean power x time per caption
  active = (mean power - idle power) x time per caption   (energy caused by the judge)

Usage (venv_gpu active, from v3/, NOTHING else running, record the power mode first):
  sudo nvpmodel -q
  python energy_bench.py --list data/human_validation_100.txt --captions results/llava.jsonl --out runs/energy.json
If no rails are found, run tegrastats once by hand ("tegrastats --interval 200") and
send one output line; some rails need sudo ("--sudo" makes the script call sudo tegrastats).
"""
from __future__ import annotations

import argparse, json, re, subprocess, threading, time
from pathlib import Path

from judge_v3.parse_caption_v3 import parse_caption
from judge_v3.rule_judge_v3 import rule_based_judge_v3

RAIL = re.compile(r"([A-Za-z0-9_]+) (\d+)mW(?:/(\d+)mW)?")


class Tegra:
    def __init__(self, interval_ms=200, sudo=False):
        cmd = (["sudo"] if sudo else []) + ["tegrastats", "--interval", str(interval_ms)]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.samples = []      # (time, {rail: mW})
        self.first_line = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            if self.first_line is None:
                self.first_line = line.strip()
            rails = {m.group(1): int(m.group(2)) for m in RAIL.finditer(line)}
            if rails:
                self.samples.append((time.time(), rails))

    def mean(self, t0, t1):
        win = [r for t, r in self.samples if t0 <= t <= t1]
        keys = sorted({k for r in win for k in r})
        return {k: round(sum(r.get(k, 0) for r in win) / len(win), 1) for k in keys}, len(win)

    def stop(self):
        self.p.terminate()


def run_yolo(weights, imgs, caps, min_s):
    from ultralytics import YOLO
    import torch
    model = YOLO(weights)
    for im in imgs[:5]:                                   # warm-up
        model(im, conf=0.25, device="cuda:0", verbose=False)
    torch.cuda.synchronize()
    n, t0 = 0, time.time()
    while n < len(imgs) or time.time() - t0 < min_s:     # at least one pass and min_s seconds
        im = imgs[n % len(imgs)]
        r = model(im, conf=0.25, device="cuda:0", verbose=False)[0]
        torch.cuda.synchronize()
        dets = [{"class": r.names[int(b.cls)], "conf": float(b.conf)} for b in r.boxes]
        rule_based_judge_v3(caps.get(Path(im).name, ""), dets, min_conf=0.25)
        n += 1
    t1 = time.time()
    del model; torch.cuda.empty_cache()
    return t0, t1, n


def run_qwen(model, imgs, caps, k):
    import ollama
    from run_vlm_judge import build_call
    jobs = []
    for im in imgs:
        claims = parse_caption(caps.get(Path(im).name, ""))["objects"]
        if claims:
            jobs.append((im, claims))
    def call(im, claims):
        prompt, schema, _ = build_call(claims)
        ollama.generate(model=model, prompt=prompt, images=[im], format=schema, stream=False, think=False,
                        options={"temperature": 0, "seed": 42, "num_predict": 300, "num_ctx": 4096})
    call(*jobs[0])                                        # load the model (not measured)
    t0 = time.time()
    for im, claims in jobs[1:k + 1]:
        call(im, claims)
    return t0, time.time(), min(k, len(jobs) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="data/human_validation_100.txt")
    ap.add_argument("--captions", default="results/llava.jsonl")
    ap.add_argument("--images_dir", default="val2017")
    ap.add_argument("--out", default="runs/energy.json")
    ap.add_argument("--idle_s", type=int, default=30)
    ap.add_argument("--yolo_min_s", type=int, default=60)
    ap.add_argument("--qwen_k", type=int, default=10, help="captions timed with Qwen (~75 s each)")
    ap.add_argument("--sudo", action="store_true")
    a = ap.parse_args()

    imgs = [str((Path(a.images_dir) / Path(l.strip()).name).resolve()) for l in open(a.list) if l.strip()]
    caps = {Path(json.loads(l)["image"]).name: json.loads(l).get("caption") or "" for l in open(a.captions)}

    tg = Tegra(sudo=a.sudo)
    time.sleep(2)
    print("tegrastats:", tg.first_line)
    if not tg.samples:
        tg.stop(); raise SystemExit("no power rails found in tegrastats output (try --sudo)")

    phases = {}
    print(f"idle {a.idle_s} s ..."); t0 = time.time(); time.sleep(a.idle_s); phases["idle"] = (t0, time.time(), 0)
    for w in ["yolov8n.pt", "yolov8x.pt"]:
        print(w, "..."); phases[Path(w).stem] = run_yolo(w, imgs, caps, a.yolo_min_s)
        time.sleep(5)
    if a.qwen_k:
        print(f"qwen3-vl:32b on {a.qwen_k} captions (~{a.qwen_k * 75 // 60 + 1} min) ...")
        phases["qwen3-vl-32b"] = run_qwen("qwen3-vl:32b", imgs, caps, a.qwen_k)
    tg.stop()

    idle, _ = tg.mean(*phases["idle"][:2])
    res = {"tegrastats_example": tg.first_line, "idle_mW": idle, "phases": {}}
    for name, (t0, t1, n) in phases.items():
        p, ns = tg.mean(t0, t1)
        row = {"seconds": round(t1 - t0, 1), "captions": n, "samples": ns, "mean_mW": p}
        if n:
            spc = (t1 - t0) / n
            row["seconds_per_caption"] = round(spc, 4)
            row["energy_J_per_caption_total"] = {k: round(v / 1000 * spc, 4) for k, v in p.items()}
            row["energy_J_per_caption_active"] = {k: round((v - idle.get(k, 0)) / 1000 * spc, 4) for k, v in p.items()}
        res["phases"][name] = row
        print(name, json.dumps(row)[:400])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
