#!/usr/bin/env python3
# pedestrian-tracker +  privacy layer
# - stream-first: use STREAM (file / rtsp / http) if set
# - optional camera fallback via waggle Camera()
# - yolo v8 + ultralytics bytetrack for people tracking
# - counts people moving left vs right based on disappearance
# - logs system stats to csv and publishes counts via waggle
# - optional head-band blur ("privacy strip") on detected people

import os
import cv2
import csv
import re
import psutil
import datetime
import subprocess
import numpy as np
import argparse
from collections import defaultdict, deque
from ultralytics import YOLO
from waggle.plugin import Plugin
from waggle.data.vision import Camera
from privacy_strip import apply_privacy_strip  

# -------------------------------
# arg and env handling (non-interactive)
# -------------------------------

def str2bool(x):
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

parser = argparse.ArgumentParser(description="pedestrian-tracker ecr version with privacy strip")

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
                    help="if 1, save output video to output.mp4")

parser.add_argument("--out_path",
                    default=os.getenv("OUT_PATH", "output.mp4"),
                    help="output video path when save_output=1")

# ---- NEW: privacy strip controls ----
parser.add_argument("--enable_privacy_strip",
                    type=str2bool,
                    default=str2bool(os.getenv("ENABLE_PRIVACY_STRIP", "1")),
                    help="if 1, apply head-band blur to person boxes")

parser.add_argument("--head_frac",
                    type=float,
                    default=float(os.getenv("HEAD_FRAC", "0.33")),
                    help="fraction of bbox height to blur from the top (0–1)")

parser.add_argument("--blur_strength",
                    type=int,
                    default=int(os.getenv("BLUR_STRENGTH", "25")),
                    help="blur strength (higher = stronger blur)")

args = parser.parse_args()

live_output = 1 if args.live_output else 0
headless = not bool(live_output)
save_video = bool(args.save_output)

enable_privacy = bool(args.enable_privacy_strip)
head_frac = args.head_frac
blur_strength = args.blur_strength

print("this program uses yolov8 and ultralytics bytetrack to count the number of people "
      "moving left/right from a video file, live stream, or waggle camera.")
print(f"privacy strip is {'enabled' if enable_privacy else 'disabled'} "
      f"(head_frac={head_frac}, blur_strength={blur_strength})")

# -------------------------------
# choose source: stream-first, camera fallback
# -------------------------------

video_path = args.stream
use_camera = False

if video_path:
    # original logic: file or url
    is_file = os.path.isfile(video_path)
    if (not is_file and
        not (video_path.startswith("http://") or
             video_path.startswith("https://") or
             video_path.startswith("rtsp://"))):
        print(f"error: '{video_path}' is neither a local file nor a url. please check the path.")
        raise SystemExit(1)
else:
    # no stream provided
    if args.camera_fallback:
        use_camera = True
        is_file = False
    else:
        print("error: no stream provided and CAMERA_FALLBACK is not enabled.")
        print("set STREAM to a file/rtsp/http path or set CAMERA_FALLBACK=1 to use the waggle camera.")
        raise SystemExit(1)

# -------------------------------
# model loading
# -------------------------------
AI_model = args.model
print("loading yolo model... (first run may download weights)")
# uses weights you added in Dockerfile at /app/models/...
model = YOLO(f"/app/models/yolov8{AI_model}.pt")

direction_threshold = args.dir_thresh
print(f"direction threshold set to {direction_threshold} px (override with DIR_THRESH env var).")

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
    "CPU%", "RAM_used_MB", "RAM_total_MB",
    "GPU%", "CPU_temp_C", "GPU_temp_C"
])

numLeft = 0
numRight = 0
track_history = defaultdict(lambda: deque(maxlen=20))  # track_id -> centers
bbox_history = defaultdict(lambda: deque(maxlen=5))    # track_id -> last N boxes

psutil.cpu_percent(interval=None)

# -------------------------------
# logging function (jetson + thor)
# -------------------------------
def log_stats(direction=None, x_start=None, x_end=None):
    cdt = datetime.datetime.now()
    date = cdt.strftime("%Y-%m-%d")
    time_str = cdt.strftime("%H:%M:%S")

    cpu_util = None
    ram_used = None
    ram_total = None
    gpu_util = None
    temps_parsed = {"cpu": None, "gpu": None}

    # first try jetson tegrastats
    tegra_ok = False
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out = proc.stdout.readline().decode("utf-8").strip()
        proc.kill()

        if out:
            # ram
            ram_match = re.search(r"RAM (\d+)/(\d+)MB", out)
            if ram_match:
                ram_used = int(ram_match.group(1))
                ram_total = int(ram_match.group(2))

            # gpu % (jetson)
            gpu_match = re.search(r"GR3D_FREQ (\d+)%", out)
            if gpu_match:
                gpu_util = int(gpu_match.group(1))

            # cpu %
            cpu_match = re.search(r"CPU \[([^\]]+)\]", out)
            if cpu_match:
                core_usages = re.findall(r"(\d+)%", cpu_match.group(1))
                if core_usages:
                    core_usages = list(map(int, core_usages))
                    cpu_util = sum(core_usages) / len(core_usages)

            # temps
            temps = {
                "cpu": re.search(r"cpu@([\d\.]+)C", out),
                "gpu": re.search(r"gpu@([\d\.]+)C", out),
            }
            temps_parsed = {
                k: float(v.group(1)) if v else None
                for k, v in temps.items()
            }

            tegra_ok = True

    except FileNotFoundError:
        tegra_ok = False

    # if *not* Jetson -> fall back to Thor or other x86
    if not tegra_ok:
        # cpu + ram
        cpu_util = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        ram_used = int(mem.used / 1024**2)
        ram_total = int(mem.total / 1024**2)

        # gpu stats via nvidia-smi
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
                gpu_util = float(parts[0])
                temps_parsed["gpu"] = float(parts[1])
        except:
            pass  # leave GPU fields None

        # cpu temp not available on Thor → remains None

    # write csv
    csv_writer.writerow([
        date, time_str, direction, x_start, x_end,
        round(cpu_util, 1) if cpu_util is not None else None,
        ram_used, ram_total,
        gpu_util,
        temps_parsed["cpu"], temps_parsed["gpu"]
    ])
    data_file.flush()


# -------------------------------
# open video stream or camera
# -------------------------------
if not use_camera:
    print("connecting to video feed...")
    cap = cv2.VideoCapture(video_path)

    ret, frame0 = cap.read()
    if not ret:
        print("error: could not read first frame. using default resolution 1280x720 at 30 fps.")
        W, H = 1280, 720
        fps = 30
    else:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
        duration = (total_frames / fps) if (is_file and fps > 0) else 0

    print(f"video resolution: {W}x{H} at {fps} fps")
    if is_file:
        print(f"total frames: {total_frames}, duration: {duration:.2f} seconds")
else:
    print("using waggle camera as source...")
    cam = Camera(args.camera) if args.camera else Camera()
    frame0 = cam.get()
    if frame0 is None:
        print("error: could not get first frame from camera.")
        raise SystemExit(1)
    H, W = frame0.shape[:2]
    fps = 30
    is_file = False  # camera is treated as continuous

# -------------------------------
# setup display / writer
# -------------------------------
if os.name == "posix" and "DISPLAY" not in os.environ:
    headless = True
    print("no display detected. running in headless mode.")

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_path, fourcc, fps, (W, H))

if not headless:
    cv2.namedWindow("live people tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("live people tracking", W, H)

frame_idx = 0
print("starting processing loop...")

# -------------------------------
# main loop with waggle plugin
# -------------------------------
try:
    with Plugin() as plugin:
        plugin.publish("movement.source.kind",
                       "camera" if use_camera else ("file" if is_file else "stream"))
        plugin.publish("movement.dir.threshold", direction_threshold)

        while True:
            if use_camera:
                frame = cam.get()
                ret = frame is not None
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                print("end of video reached or failed to grab frame.")
                break

            frame_idx += 1

            # yolov8 bytetrack
            results = model.track(
                frame, classes=[0], conf=0.4,
                tracker="bytetrack.yaml",
                persist=True, verbose=False
            )

            # collect boxes to pass into privacy strip
            person_boxes = []

            ids_tensor = results[0].boxes.id if len(results) > 0 else None
            if ids_tensor is not None and len(ids_tensor):
                for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(),
                                         ids_tensor.int().cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box[:4])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    track_history[track_id].append(center)
                    bbox_history[track_id].append((x1, y1, x2, y2))
                    sx1, sy1, sx2, sy2 = np.mean(bbox_history[track_id], axis=0).astype(int)
                    x1, y1, x2, y2 = sx1, sy1, sx2, sy2

                    # save smoothed box for privacy strip
                    person_boxes.append((x1, y1, x2, y2))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if len(track_history[track_id]) >= 2:
                        x_diff = track_history[track_id][-1][0] - track_history[track_id][0][0]
                        dir_txt = "Right" if x_diff > 0 else "Left" if x_diff < 0 else ""
                    else:
                        dir_txt = ""
                    cv2.putText(frame, f"ID {int(track_id)} {dir_txt}", (x1, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                current_ids = set(ids_tensor.int().cpu().numpy().tolist())
                disappeared = set(track_history.keys()) - current_ids
                for tid in disappeared:
                    if len(track_history[tid]) >= 2:
                        x_start = track_history[tid][0][0]
                        x_end = track_history[tid][-1][0]
                        x_diff = x_end - x_start
                        if x_diff > direction_threshold:
                            numRight += 1
                            log_stats("Right", x_start, x_end)
                        elif x_diff < -direction_threshold:
                            numLeft += 1
                            log_stats("Left", x_start, x_end)
                    del track_history[tid]
                    if tid in bbox_history:
                        del bbox_history[tid]

            # overlay totals
            cv2.putText(frame, f"Left: {numLeft}, Right: {numRight}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # ---- privacy strip pass ----
            if enable_privacy and person_boxes:
                apply_privacy_strip(
                    frame,
                    person_boxes,
                    head_frac=head_frac,
                    strength=blur_strength,
                )

            # publish counts every frame
            plugin.publish("movement.left.count", numLeft)
            plugin.publish("movement.right.count", numRight)
            plugin.publish("movement.total.count", numLeft + numRight)

            if not headless:
                cv2.imshow("live people tracking", frame)

            if save_video and "out" in locals():
                out.write(frame)

            if not headless and (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("quit requested by user.")
                break

            if frame_idx % 50 == 0:
                print(f"processed frame {frame_idx}")

except Exception as e:
    print(f"error: {str(e)}")

finally:
    print("releasing resources...")
    if not use_camera:
        try:
            cap.release()
        except Exception:
            pass
    if save_video and "out" in locals():
        try:
            out.release()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    data_file.close()
    print(f"total left: {numLeft}, right: {numRight}")
    print("processing complete")
