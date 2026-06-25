#!/usr/bin/env python3
# offline-tracking (ecr / sage version)
# - stream-first: use STREAM (file / rtsp / http) if set
# - optional camera fallback via waggle Camera()
# - yolov8 + ultralytics botsort tracker for people tracking
# - optional face blurring with a yolov8-face model (privacy)
# - threaded csv stats logger (tegrastats on jetson, psutil + nvidia-smi elsewhere)
# - counts people moving left vs right based on disappearance
# - publishes counts via waggle Plugin

import os
import cv2
import csv
import re
import time
import psutil
import threading
import queue
import datetime
import subprocess
import numpy as np
import argparse
from collections import defaultdict, deque
from ultralytics import YOLO
try:
    from waggle.plugin import Plugin
    from waggle.data.vision import Camera
except ImportError:
    Plugin = None
    Camera = None

# -------------------------------
# arg and env handling (non-interactive)
# -------------------------------

def str2bool(x):
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

parser = argparse.ArgumentParser(description="offline-tracking ecr version")

parser.add_argument("--stream",
                    default=os.getenv("STREAM", ""),
                    help="file path or rtsp/http url; preferred source")

parser.add_argument("--camera_fallback",
                    type=str2bool,
                    default=str2bool(os.getenv("CAMERA_FALLBACK", "0")),
                    help="if 1 and no stream provided, use waggle camera")

parser.add_argument("--camera",
                    default=os.getenv("CAMERA", ""),
                    help="waggle camera name (e.g., left/right) when fallback enabled")

parser.add_argument("--model",
                    default=os.getenv("MODEL_SIZE", "m"),
                    choices=list("nsmlx"),
                    help="yolov8 model size [n/s/m/l/x]")

parser.add_argument("--dir_thresh",
                    type=int,
                    default=int(os.getenv("DIR_THRESH", "100")),
                    help="direction threshold in pixels across image width")

parser.add_argument("--live_output",
                    type=str2bool,
                    default=str2bool(os.getenv("LIVE_OUTPUT", "0")),
                    help="if 1, show live display window (dev only; headless in ecr)")

parser.add_argument("--save_output",
                    type=str2bool,
                    default=str2bool(os.getenv("SAVE_OUTPUT", "0")),
                    help="if 1, save output video to OUT_PATH")

parser.add_argument("--out_path",
                    default=os.getenv("OUT_PATH", "output.mp4"),
                    help="output video path when save_output=1")

parser.add_argument("--tracker",
                    default=os.getenv("TRACKER", "botsort.yaml"),
                    help="ultralytics tracker yaml (e.g., botsort.yaml or bytetrack.yaml)")

parser.add_argument("--conf",
                    type=float,
                    default=float(os.getenv("CONF", "0.55")),
                    help="person detection confidence threshold")

parser.add_argument("--imgsz",
                    type=int,
                    default=int(os.getenv("IMGSZ", "640")),
                    help="yolo inference image size")

# face blurring options
parser.add_argument("--face_blur",
                    type=str2bool,
                    default=str2bool(os.getenv("FACE_BLUR", "1")),
                    help="if 1, enable face blurring for privacy")

parser.add_argument("--face_conf",
                    type=float,
                    default=float(os.getenv("FACE_CONF", "0.35")),
                    help="face detector confidence threshold")

parser.add_argument("--face_every",
                    type=int,
                    default=int(os.getenv("FACE_EVERY", "1")),
                    help="run face detection every N frames")

parser.add_argument("--face_weights",
                    default=os.getenv("FACE_WEIGHTS", "/app/yolov8n-face.pt"),
                    help="path to yolov8 face weights")

parser.add_argument("--face_upscale",
                    type=float,
                    default=float(os.getenv("FACE_UPSCALE", "1.8")),
                    help="upscale factor for face ROI to catch small faces")

parser.add_argument("--head_failsafe",
                    type=str2bool,
                    default=str2bool(os.getenv("HEAD_FAILSAFE", "1")),
                    help="if 1, blur head region when no face is detected")

parser.add_argument("--head_ratio",
                    type=float,
                    default=float(os.getenv("HEAD_RATIO", "0.38")),
                    help="top fraction of person box treated as head for failsafe blur")

args = parser.parse_args()

# resolve tracker path: if bare filename (no directory), look next to this script
if not os.path.dirname(args.tracker):
    args.tracker = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tracker)

live_output = 1 if args.live_output else 0
headless = not bool(live_output)
save_video = bool(args.save_output)

# -------------------------------
# choose source: stream-first, camera fallback
# -------------------------------

video_path = args.stream
use_camera = False

# after imports, before the main loop

class _NoOpPlugin:
    """drop-in replacement when waggle broker is unavailable"""
    def publish(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

USE_WAGGLE = str2bool(os.getenv("USE_WAGGLE", "1"))

def get_plugin():
    if not USE_WAGGLE:
        print("[info] waggle disabled (USE_WAGGLE=0) — publishes are no-ops.")
        return _NoOpPlugin()
    try:
        from waggle.plugin import Plugin
        return Plugin()
    except Exception as e:
        print(f"[warn] waggle plugin unavailable: {e} — running without publishes.")
        return _NoOpPlugin()

if video_path:
    is_file = os.path.isfile(video_path)
    if (not is_file and
        not (video_path.startswith("http://") or
             video_path.startswith("https://") or
             video_path.startswith("rtsp://"))):
        print(f"error: '{video_path}' is neither a local file nor a url. please check the path.")
        raise SystemExit(1)
else:
    if args.camera_fallback:
        use_camera = True
        is_file = False
    else:
        print("error: no stream provided and CAMERA_FALLBACK is not enabled.")
        print("set STREAM to a file/rtsp/http path or set CAMERA_FALLBACK=1 to use the waggle camera.")
        raise SystemExit(1)

print("this program uses yolov8 and ultralytics botsort to count the number of people "
      "moving left/right from a video file, live stream, or waggle camera.")

# -------------------------------
# model loading (.pt provided in /app/models by Dockerfile)
# -------------------------------
AI_model = args.model
print("loading yolo model... (first run may download weights)")

model_names = {
    'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large', 'x': 'xlarge'
}
engine_path = f"/app/models/yolov8{AI_model}.engine"
pt_path = f"/app/models/yolov8{AI_model}.pt"

if os.path.isfile(engine_path):
    print(f"found existing tensorrt engine ({model_names[AI_model]}) — loading.")
    person_model = YOLO(engine_path, task='detect')
else:
    print(f"engine not found. exporting {model_names[AI_model]} to engine at imgsz=640 (first run may take a few minutes)...")
    temp_model = YOLO(pt_path)
    temp_model.export(format='engine', device=0, imgsz=640, verbose=False)
    person_model = YOLO(engine_path, task='detect')

# -------------------------------
# ReID model: download ONNX to models dir, export to TensorRT engine, inject into temp yaml
# -------------------------------
reid_onnx   = "/app/models/yolo26s-reid.onnx"
reid_engine = "/app/models/yolo26s-reid.engine"

if not os.path.isfile(reid_onnx) and not os.path.isfile(reid_engine):
    print("ReID ONNX not found — downloading yolo26s-reid.onnx to models directory...")
    _prev_dir = os.getcwd()
    os.chdir("/app/models")
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset("yolo26s-reid.onnx")
    finally:
        os.chdir(_prev_dir)

if not os.path.isfile(reid_engine) and os.path.isfile(reid_onnx):
    print("Exporting ReID ONNX to TensorRT engine (one-time, may take a minute)...")
    _trtexec = next((p for p in [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ] if os.path.isfile(p)), "trtexec")  # fallback to PATH as last resort
    result = subprocess.run(
        [
            _trtexec,
            f"--onnx={reid_onnx}",
            f"--saveEngine={reid_engine}",
            "--minShapes=images:1x3x224x224",
            "--optShapes=images:4x3x224x224",
            "--maxShapes=images:16x3x224x224",
        ],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"ReID TensorRT engine saved to {reid_engine}")
    else:
        print(f"[warn] trtexec failed — falling back to ONNX ReID.\n{result.stderr[-500:]}")
        reid_engine = reid_onnx  # fallback: use onnx if trtexec unavailable

# write a runtime-only tracker yaml with the absolute ReID model path
_reid_model_path = reid_engine if os.path.isfile(reid_engine) else (
                   reid_onnx   if os.path.isfile(reid_onnx)   else "auto")
_runtime_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_botsort_runtime.yaml")
with open(args.tracker) as _f:
    _yaml_content = _f.read()
_yaml_content = re.sub(r"^model:.*$", f"model: {_reid_model_path}", _yaml_content, flags=re.MULTILINE)
with open(_runtime_yaml, "w") as _f:
    _f.write(_yaml_content)
print(f"tracker config: using ReID model at {_reid_model_path}")

direction_threshold = args.dir_thresh
print(f"direction threshold set to {direction_threshold} px (override with DIR_THRESH env var).")

# -------------------------------
# face blurring setup
# -------------------------------
FACE_BLUR = bool(args.face_blur)
FACE_CONF = float(args.face_conf)
FACE_EVERY = int(args.face_every)
FACE_UPSCALE = float(args.face_upscale)
HEAD_FAILSAFE = bool(args.head_failsafe)
HEAD_RATIO = float(args.head_ratio)

face_model = None
if FACE_BLUR:
    try:
        face_model = YOLO(args.face_weights)
        print(f"face blurring enabled (conf={FACE_CONF}, every={FACE_EVERY} frame(s)) using: {args.face_weights}")
    except Exception as e:
        print(f"warning: could not load face model: {e}. face blurring disabled.")
        FACE_BLUR = False

# -------------------------------
# paths and csv initialization
# -------------------------------
cwd = os.path.dirname(os.path.abspath(__file__))
cdt = datetime.datetime.now()
output_dir = os.path.join(cwd, "output", cdt.strftime("%Y%m%d%H"))
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

data_file = open("data.csv", "w", newline="")
csv_writer = csv.writer(data_file)
csv_writer.writerow([
    "Date", "Time", "Direction", "X_start", "X_end",
    "CPU%", "GPU%", "RAM_used_MB", "RAM_total_MB",
    "CPU_temp_C", "GPU_temp_C"
])

numLeft = 0
numRight = 0
track_history = defaultdict(lambda: deque(maxlen=120))  # track_id -> centers
bbox_history = defaultdict(lambda: deque(maxlen=5))     # track_id -> last N boxes
missing_frames = defaultdict(int)   # frames each ID has been absent
MAX_MISSING = 120                   # patience before counting + cleanup
MIN_TRACK_FRAMES = 15               # ignore tracks seen fewer than this many frames

psutil.cpu_percent(interval=None)

# -------------------------------
# threaded stats worker
# -------------------------------
data_queue = queue.Queue()

last_tegra_time = 0.0
cached_tegra_data = {
    "cpu": None,
    "ram_used": None,
    "ram_total": None,
    "cpu_temp": None,
    "gpu_temp": None,
    "gpu_util": None,
}
_tegra_supported = True  # flips False after first FileNotFoundError

def collect_system_stats():
    """fill cached_tegra_data using tegrastats on jetson, or psutil + nvidia-smi elsewhere"""
    global last_tegra_time, _tegra_supported

    current_time = time.time()
    if current_time - last_tegra_time < 1.0:
        return

    tegra_ok = False
    if _tegra_supported:
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out = proc.stdout.readline().decode("utf-8").strip()
            proc.kill()

            if out:
                ram_match = re.search(r"RAM (\d+)/(\d+)MB", out)
                if ram_match:
                    cached_tegra_data["ram_used"] = int(ram_match.group(1))
                    cached_tegra_data["ram_total"] = int(ram_match.group(2))

                gpu_match = re.search(r"GR3D_FREQ (\d+)%", out)
                if gpu_match:
                    cached_tegra_data["gpu_util"] = int(gpu_match.group(1))

                cpu_match = re.search(r"CPU \[([^\]]+)\]", out)
                if cpu_match:
                    core_usages = re.findall(r"(\d+)%", cpu_match.group(1))
                    if core_usages:
                        core_usages = list(map(int, core_usages))
                        cached_tegra_data["cpu"] = sum(core_usages) / len(core_usages)

                cpu_temp = re.search(r"cpu@([\d\.]+)C", out, re.IGNORECASE)
                gpu_temp = re.search(r"gpu@([\d\.]+)C", out, re.IGNORECASE)
                if cpu_temp:
                    cached_tegra_data["cpu_temp"] = float(cpu_temp.group(1))
                if gpu_temp:
                    cached_tegra_data["gpu_temp"] = float(gpu_temp.group(1))

                tegra_ok = True
        except FileNotFoundError:
            _tegra_supported = False
        except Exception as e:
            print(f"error collecting tegrastats: {e}")

    if not tegra_ok:
        # x86 / thor fallback
        cached_tegra_data["cpu"] = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        cached_tegra_data["ram_used"] = int(mem.used / 1024**2)
        cached_tegra_data["ram_total"] = int(mem.total / 1024**2)

        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            line = smi.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                cached_tegra_data["gpu_util"] = float(parts[0])
                cached_tegra_data["gpu_temp"] = float(parts[1])
        except Exception:
            pass

    last_tegra_time = current_time


def log_stats(direction=None, x_start=None, x_end=None, date=None, time_str=None):
    if date is None or time_str is None:
        cdt2 = datetime.datetime.now()
        date = cdt2.strftime("%Y-%m-%d")
        time_str = cdt2.strftime("%H:%M:%S")

    collect_system_stats()

    cpu_val = cached_tegra_data["cpu"]
    row_data = [
        date,
        time_str,
        direction,
        x_start,
        x_end,
        round(cpu_val, 1) if cpu_val is not None else None,
        cached_tegra_data["gpu_util"],
        cached_tegra_data["ram_used"],
        cached_tegra_data["ram_total"],
        cached_tegra_data["cpu_temp"],
        cached_tegra_data["gpu_temp"],
    ]

    try:
        csv_writer.writerow(row_data)
        data_file.flush()
    except Exception as e:
        print(f"error writing to csv: {e}")


def stats_worker():
    while True:
        item = data_queue.get()
        if item is None:
            data_queue.task_done()
            break
        direction, x_start, x_end, det_date, det_time = item
        try:
            log_stats(direction, x_start, x_end, det_date, det_time)
        except Exception as e:
            print(f"[worker error] failed to log stats: {e}")
        data_queue.task_done()


threading.Thread(target=stats_worker, daemon=True).start()


# -------------------------------
# face blurring utility
# -------------------------------
def blur_roi(frame, x1, y1, x2, y2):
    H, W = frame.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    kx = max(3, ((x2 - x1) // 7) | 1)
    ky = max(3, ((y2 - y1) // 7) | 1)
    roi[:] = cv2.GaussianBlur(roi, (kx, ky), 0)


# -------------------------------
# open video stream or camera
# -------------------------------
cap = None
cam = None

if not use_camera:
    print("connecting to video feed...")
    cap = cv2.VideoCapture(video_path)

    max_retries = 10
    retry_delay = 1.0
    attempt = 0
    ret = False
    frame0 = None
    while True:
        ret, frame0 = cap.read()
        if ret and frame0 is not None:
            break
        attempt += 1
        print(f"[warn] could not read first frame (attempt {attempt}/{max_retries}). reopening stream...")
        cap.release()
        time.sleep(retry_delay)
        cap = cv2.VideoCapture(video_path)
        if attempt >= max_retries:
            print("[warn] still no frame after retries - using default 1280x720@30 and continuing.")
            frame0 = np.zeros((720, 1280, 3), dtype=np.uint8)
            ret = True
            break

    W = 1280
    H = 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 120:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    duration = (total_frames / fps) if (is_file and fps > 0) else 0

    print(f"video resolution: {W}x{H} at {fps} fps")
    if is_file:
        print(f"total frames: {total_frames}, duration: {duration:.2f} seconds")
elif Camera is None:
    print("error: camera fallback requires waggle package (not installed).")
    raise SystemExit(1)
else:
    print("using waggle camera as source...")
    cam = Camera(args.camera) if args.camera else Camera()
    frame0 = cam.get()
    if frame0 is None:
        print("error: could not get first frame from camera.")
        raise SystemExit(1)
    H, W = frame0.shape[:2]
    fps = 30
    is_file = False

# -------------------------------
# setup display / writer
# -------------------------------
if os.name == "posix" and "DISPLAY" not in os.environ:
    headless = True
    print("no display detected. running in headless mode.")

out_writer = None
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.out_path, fourcc, fps, (W, H))

if not headless:
    cv2.namedWindow("live people tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("live people tracking", W, H)

frame_idx = 0
print("starting processing loop...")

# -------------------------------
# main loop with waggle plugin
# -------------------------------
start_time = time.time()
try:
    with get_plugin() as plugin:
        plugin.publish("movement.source.kind",
                       "camera" if use_camera else ("file" if is_file else "stream"))
        plugin.publish("movement.dir.threshold", direction_threshold)
        plugin.publish("movement.face_blur", 1 if FACE_BLUR else 0)

        max_soft_fail = 20
        soft_fail = 0
        prev_frame_time = 0.0

        is_video_file = bool(video_path) and video_path.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )

        while True:
            t_start = time.time()
            if use_camera:
                frame = cam.get()
                ret = frame is not None
            else:
                ret, frame = cap.read()
            t_read = time.time()

            if not ret or frame is None:
                if is_video_file:
                    print("[info] end of video reached.")
                    break

                if use_camera:
                    soft_fail += 1
                    if soft_fail >= max_soft_fail:
                        print("[error] too many camera read failures, stopping.")
                        break
                    time.sleep(0.05)
                    continue

                soft_fail += 1
                if soft_fail < max_soft_fail:
                    if soft_fail % 5 == 0:
                        print(f"[warn] temporary read failure x{soft_fail} - retrying...")
                    time.sleep(0.05)
                    continue

                print("[warn] read failures exceeded threshold - attempting reconnect...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(1.0)
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    print("[error] reconnect failed; stopping.")
                    break

                print("[info] reconnect successful; continuing stream.")
                soft_fail = 0
                continue

            # normalize to expected resolution for inference + writer
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))

            soft_fail = 0
            frame_idx += 1

            # yolov8 + tracker
            results = person_model.track(
                frame,
                classes=[0],
                conf=args.conf,
                tracker=_runtime_yaml,
                persist=True,
                verbose=False,
                iou=0.5,
                device=0,
                imgsz=args.imgsz,
            )

            t_ai = time.time()
            read_ms = (t_read - t_start) * 1000
            ai_ms = (t_ai - t_read) * 1000
            if frame_idx % 1000 == 0:
                print(f"read time: {read_ms:.1f}ms | ai time: {ai_ms:.1f}ms")

            person_boxes = []
            ids_tensor = results[0].boxes.id if len(results) > 0 else None

            if ids_tensor is not None and len(ids_tensor):
                current_ids = set(ids_tensor.int().cpu().numpy().tolist())
                for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(),
                                         ids_tensor.int().cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box[:4])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    track_history[track_id].append(center)
                    bbox_history[track_id].append((x1, y1, x2, y2))

                    if len(bbox_history[track_id]) > 10:
                        bbox_history[track_id].pop(0)
                    sx1, sy1, sx2, sy2 = np.mean(bbox_history[track_id], axis=0).astype(int)
                    x1, y1, x2, y2 = sx1, sy1, sx2, sy2

                    person_boxes.append((x1, y1, x2, y2))

                    missing_frames[track_id] = 0

                    if not headless or save_video:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if len(track_history[track_id]) >= 2:
                            x_diff = (track_history[track_id][-1][0]
                                      - track_history[track_id][0][0])
                            dir_txt = "Right" if x_diff > 0 else "Left" if x_diff < 0 else ""
                        else:
                            dir_txt = ""
                        cv2.putText(frame, f"ID {int(track_id)} {dir_txt}", (x1, y1 - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                current_ids = set()

            for tid in list(track_history.keys()):
                if tid not in current_ids:
                    missing_frames[tid] += 1

                    if missing_frames[tid] > MAX_MISSING:
                        if len(track_history[tid]) >= MIN_TRACK_FRAMES:
                            x_start = track_history[tid][0][0]
                            x_end = track_history[tid][-1][0]
                            x_diff = x_end - x_start

                            cdt2 = datetime.datetime.now()
                            det_date = cdt2.strftime("%Y-%m-%d")
                            det_time = cdt2.strftime("%H:%M:%S")

                            if x_diff > direction_threshold:
                                numRight += 1
                                data_queue.put(("Right", x_start, x_end, det_date, det_time))
                            elif x_diff < -direction_threshold:
                                numLeft += 1
                                data_queue.put(("Left", x_start, x_end, det_date, det_time))

                        del track_history[tid]
                        del missing_frames[tid]
                        if tid in bbox_history:
                            del bbox_history[tid]

            if not headless or save_video:
                cv2.putText(frame, f"Left: {numLeft}, Right: {numRight}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # face blurring pass (before write/show)
            if FACE_BLUR and face_model is not None and (frame_idx % FACE_EVERY == 0) and person_boxes:
                faces_found = 0
                for (px1, py1, px2, py2) in person_boxes:
                    roi = frame[py1:py2, px1:px2]
                    if roi.size == 0:
                        continue

                    roi_up = (cv2.resize(roi, None,
                                         fx=FACE_UPSCALE, fy=FACE_UPSCALE,
                                         interpolation=cv2.INTER_LINEAR)
                              if FACE_UPSCALE != 1.0 else roi)

                    dets = face_model.predict(roi_up, conf=FACE_CONF, verbose=False)
                    if dets and hasattr(dets[0], "boxes") and dets[0].boxes is not None:
                        for fx1, fy1, fx2, fy2 in dets[0].boxes.xyxy.cpu().numpy():
                            if FACE_UPSCALE != 1.0:
                                fx1, fy1, fx2, fy2 = [int(v / FACE_UPSCALE)
                                                      for v in (fx1, fy1, fx2, fy2)]
                            blur_roi(frame,
                                     px1 + int(fx1), py1 + int(fy1),
                                     px1 + int(fx2), py1 + int(fy2))
                            faces_found += 1

                if HEAD_FAILSAFE and faces_found == 0:
                    for (px1, py1, px2, py2) in person_boxes:
                        head_h = int((py2 - py1) * HEAD_RATIO)
                        blur_roi(frame, px1, py1, px2, py1 + head_h)

            # fps overlay
            new_frame_time = time.time()
            diff = new_frame_time - prev_frame_time
            fps_now = 1.0 / diff if diff > 0 else 0.0
            prev_frame_time = new_frame_time

            if not headless or save_video:
                cv2.putText(frame, f"FPS: {int(fps_now)}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # publish counts via waggle
            plugin.publish("movement.left.count", numLeft)
            plugin.publish("movement.right.count", numRight)
            plugin.publish("movement.total.count", numLeft + numRight)

            if save_video and out_writer is not None:
                out_writer.write(frame)

            if not headless:
                cv2.imshow("live people tracking", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("quit requested by user.")
                    break

            if frame_idx % 1000 == 0:
                print(f"processed frame {frame_idx} | current speed: {fps_now:.2f} fps")

except Exception as e:
    print(f"error: {str(e)}")

finally:
    print("releasing resources...")
    data_queue.put(None)
    try:
        data_queue.join()
    except Exception:
        pass

    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"total execution time: {total_seconds:.2f} seconds")

    if not use_camera and cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    if save_video and out_writer is not None:
        try:
            out_writer.release()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    data_file.close()
    # clean up runtime yaml
    try:
        if os.path.isfile(_runtime_yaml):
            os.remove(_runtime_yaml)
    except Exception:
        pass
    print(f"total left: {numLeft}, right: {numRight}")
    print("processing complete")
