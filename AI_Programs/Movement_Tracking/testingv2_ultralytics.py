#!/usr/bin/env python3
# People-counter (beta, Ultralytics tracker): Count people moving left vs right.
# Uses YOLOv8 + Ultralytics built-in ByteTrack tracker. Restores:
# - Bounded history & disappearance-based counting
# - Short-term bbox smoothing
# - Configurable direction threshold (DIR_THRESH, default 100)
#
# Usage:
#   DIR_THRESH=100 python beta_ultralytics_tracking_patched.py

import os
import cv2
import csv
import re
import psutil
import datetime
import subprocess
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import time 

# ---- V2 head-band blur config (env-driven) ----

def _odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def apply_head_blur(frame, x1, y1, x2, y2, head_frac, strength):
    """
    Blur only the top band (approx head area) of a person bbox.
    Mutates 'frame' in place. (x1,y1,x2,y2) may be float or int.
    """
    h, w = frame.shape[:2]
    x1i, y1i = int(_clip(round(x1), 0, w - 1)), int(_clip(round(y1), 0, h - 1))
    x2i, y2i = int(_clip(round(x2), 0, w - 1)), int(_clip(round(y2), 0, h - 1))
    if x2i <= x1i or y2i <= y1i:
        return

    box_h = y2i - y1i
    head_h = max(4, int(box_h * head_frac))
    y2_head = _clip(y1i + head_h, 0, h - 1)

    roi = frame[y1i:y2_head, x1i:x2i]
    if roi.size == 0:
        return

    # scale kernel by region size for consistent blur across distances
    kx = _odd(max(3, int((x2i - x1i) / 12) + strength // 10))
    ky = _odd(max(3, int(head_h / 12) + strength // 10))
    blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
    frame[y1i:y2_head, x1i:x2i] = blurred
# -------------------------------

print("This program uses YOLOv8 and Ultralytics ByteTrack to count the number of people moving left/right from a video file or live camera feed.")
print()

# -------------------------------
# Interactive options
# -------------------------------
live_output = int(input("Would you like live display of AI detection? 1 for Yes, 0 for No: "))
print()
save = int(input("Would you like to save the output video? Enter 1 for Yes, 0 for No: "))

if live_output == 1:
    headless = False
    print("Video output enabled")
else:
    headless = True
    print("Video output disabled")

if save == 1:
    save_video = True
    print("Video saving enabled")
    print()
else:
    save_video = False
    print("Video saving disabled")
    print()

print("You can choose to either input a video file path or a live camera feed link.")
video_input_file = input("Please enter the exact path/URL to your video to be processed: ")
print()

# -------------------------------
# Paths and Initialization
# -------------------------------
video_path = video_input_file
is_file = os.path.isfile(video_path)
if not is_file and not (video_path.startswith("http://") or video_path.startswith("https://") or video_path.startswith("rtsp://")):
    print(f"Error: '{video_path}' is neither a local file nor a URL. Please check the path and try again.")
    raise SystemExit(1)

print("There are 5 different YOLO models to choose from: n (nano), s (small), m (medium), l (large), x (xlarge).")
AI_model = input("Please enter the YOLOv8 model size to use [n/s/m/l/x]: ").strip().lower()
if AI_model not in ['n', 's', 'm', 'l', 'x']:
    print("Invalid model choice, defaulting to 'm' (medium)")
    AI_model = 'm'
print()

print("Loading YOLO model... (first run may download weights)")
person_model = YOLO('yolov8' + AI_model + '.pt')

# counting sensitivity (pixels across the image width)
direction_threshold = int(os.getenv("DIR_THRESH", "100"))
print(f"Direction threshold set to {direction_threshold} px (override with DIR_THRESH env var).")

# -------------------------------
# V2 Head-band privacy blur (no extra model)
# -------------------------------
try:
    enable_blur = int(input("Enable head-band privacy blur? 1 for Yes, 0 for No: "))
except Exception:
    enable_blur = 0
print(f"Head-band blur {'enabled' if enable_blur == 1 else 'disabled'}")

# we can also make this interactive or env-driven
head_frac = float(os.getenv("HEAD_FRAC", "0.33"))

#for educational purposes let user tune blur strength interactively
if enable_blur == 1:
    try:
        blur_strength = int(input("Blur strength (e.g., 25; higher = stronger): ").strip() or "25")
    except Exception:
        blur_strength = 25
    print(f"Using HEAD_FRAC={head_frac}, BLUR_STRENGTH={blur_strength}")
else:
    blur_strength = 25
print()

# -------------------------------
# Create output directory based on current timestamp
cwd = os.path.dirname(os.path.abspath(__file__))
cdt = datetime.datetime.now()
output_dir = os.path.join(cwd, 'output', cdt.strftime("%Y%m%d%H"))
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

# Open CSV file for logging stats
data_file = open("data.csv", "w", newline="")
csv_writer = csv.writer(data_file)
csv_writer.writerow([
    "Date", "Time", "Direction", "X_start", "X_end",
    "CPU%", "RAM_used_MB", "RAM_total_MB",
    "GPU%", "CPU_temp_C", "GPU_temp_C"
])

# Initialize movement counters and histories
numLeft = 0
numRight = 0
track_history = defaultdict(lambda: deque(maxlen=20))  # track_id -> centers [(x,y), ...]
bbox_history  = defaultdict(lambda: deque(maxlen=5))   # track_id -> last N boxes (for smoothing)

# Initialize psutil
psutil.cpu_percent(interval=None)

# -------------------------------
# Logging Function
# -------------------------------
def log_stats(direction=None, x_start=None, x_end=None):
    # Timestamp
    cdt = datetime.datetime.now()
    date = cdt.strftime("%Y-%m-%d")
    time = cdt.strftime("%H:%M:%S")

    cpu_util = None
    ram_used = None
    ram_total = None
    gpu_util = None
    temps_parsed = {"cpu": None, "gpu": None}

    # Try tegrastats (Jetson); fallback to psutil on non-Jetson
    try:
        proc = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.stdout.readline().decode("utf-8").strip()
        proc.kill()

        # RAM
        ram_match = re.search(r"RAM (\d+)/(\d+)MB", out)
        ram_used = int(ram_match.group(1)) if ram_match else None
        ram_total = int(ram_match.group(2)) if ram_match else None

        # GPU util
        gpu_match = re.search(r"GR3D_FREQ (\d+)%", out)
        gpu_util = int(gpu_match.group(1)) if gpu_match else None

        # CPU average across cores
        cpu_match = re.search(r"CPU \[([^\]]+)\]", out)
        if cpu_match:
            core_usages = re.findall(r"(\d+)%", cpu_match.group(1))
            if core_usages:
                core_usages = list(map(int, core_usages))
                cpu_util = sum(core_usages) / len(core_usages)

        # Temps
        temps = {
            "cpu": re.search(r"cpu@([\d\.]+)C", out),
            "gpu": re.search(r"gpu@([\d\.]+)C", out),
        }
        temps_parsed = {k: float(v.group(1)) if v else None for k, v in temps.items()}

    except FileNotFoundError:
        # Fallback if tegrastats not available (e.g., on PC)
        cpu_util = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        ram_used = int(mem.used / 1024**2)
        ram_total = int(mem.total / 1024**2)
        temps_parsed = {"cpu": None, "gpu": None}
        gpu_util = None

    # Write CSV row
    csv_writer.writerow([
        date, time, direction, x_start, x_end,
        round(cpu_util, 1) if cpu_util is not None else None,
        ram_used, ram_total,
        gpu_util,
        temps_parsed["cpu"], temps_parsed["gpu"]
    ])
    data_file.flush()

# -------------------------------
# Open video stream
# -------------------------------
print("Connecting to video feed...")
cap = cv2.VideoCapture(video_path)


#V2.1 change to try to read first frame with retries
max_retries = 10
retry_delay = 1.0  # seconds
attempt = 0
while True:
    ret, frame0 = cap.read()
    if ret and frame0 is not None:
        break
    attempt += 1
    print(f"[WARN] Could not read first frame (attempt {attempt}/{max_retries}). Reopening stream...")
    cap.release()
    time.sleep(retry_delay)
    cap = cv2.VideoCapture(video_path)
    if attempt >= max_retries:
        print("[WARN] Still no frame after retries — using default 1280x720@30 and continuing.")
        frame0 = np.zeros((720, 1280, 3), dtype=np.uint8)
        ret = True
        break

# Infer resolution / fps safely
if ret and frame0 is not None:
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 120:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    duration = (total_frames / fps) if (is_file and fps > 0) else 0
else:
    # Fallback defaults if nothing works
    print("Error: Could not read first frame. Using default resolution 1280x720 at 30 FPS.")
    W, H, fps = 1280, 720, 30
    total_frames, duration = 0, 0
print(f"Video resolution: {W}x{H} at {fps} FPS")
if is_file:
    print(f"Total frames: {total_frames}, Duration: {duration:.2f} seconds")

# -------------------------------
# Setup display / writer
if os.name == 'posix' and "DISPLAY" not in os.environ:
    headless = True
    print("No display detected. Running in headless mode, video will be saved to output.mp4 if enabled.")

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (W, H))

if not headless:
    cv2.namedWindow("Live People Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live People Tracking", W, H)

frame_idx = 0

print("Starting processing loop...")
try:
    # --------------------------------------------
    # main processing loop with retry + reconnect
    # --------------------------------------------
    max_soft_fail = 20    # how many bad reads before reconnect
    soft_fail = 0         # counter for consecutive failed frames

    while True:
        ret, frame = cap.read()

        # if a frame fails to read, handle retries or reconnect
        if not ret or frame is None:
            soft_fail += 1

            # small transient issue — retry a few times
            if soft_fail < max_soft_fail:
                if soft_fail % 5 == 0:
                    print(f"[warn] temporary read failure x{soft_fail} — retrying...")
                time.sleep(0.05)
                continue

            # too many failures, try to reconnect
            print("[warn] read failures exceeded threshold — attempting reconnect...")
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(1.0)
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # check if reconnect worked
            if not cap.isOpened():
                print("[error] reconnect failed; stopping.")
                break

            print("[info] reconnect successful; continuing stream.")
            soft_fail = 0
            continue

        # reset fail counter on success
        soft_fail = 0
        frame_idx += 1
        # -------------------------------
        
        # Run Ultralytics tracker on this frame (ByteTrack)
        results = person_model.track(
            frame, classes=[0], conf=0.4,
            tracker="bytetrack.yaml",
            persist=True, verbose=False
        )
        
        # -- V2: Face blurring --
        person_boxes = []  # collect smoothed person boxes for face blurring
        # -- Collect person boxes for face blurring --

        # Draw detections and update histories
        ids_tensor = results[0].boxes.id
        if ids_tensor is not None and len(ids_tensor):
            for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(),
                                     ids_tensor.int().cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Append center and smooth box
                track_history[track_id].append(center)
                bbox_history[track_id].append((x1, y1, x2, y2))
                sx1, sy1, sx2, sy2 = np.mean(bbox_history[track_id], axis=0).astype(int)
                x1, y1, x2, y2 = sx1, sy1, sx2, sy2

                # -- V2: save smoothed person box for the face pass
                person_boxes.append((x1, y1, x2, y2))
                # -- end --  
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                # direction hint (based on recent centers)
                if len(track_history[track_id]) >= 2:
                    x_diff = track_history[track_id][-1][0] - track_history[track_id][0][0]
                    dir_txt = "Right" if x_diff > 0 else "Left" if x_diff < 0 else ""
                else:
                    dir_txt = ""
                cv2.putText(frame, f"ID {int(track_id)} {dir_txt}", (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Disappearance-based counting (compare known vs current IDs)
            current_ids = set(ids_tensor.int().cpu().numpy().tolist())
            disappeared = set(track_history.keys()) - current_ids
            for tid in disappeared:
                if len(track_history[tid]) >= 2:
                    x_start = track_history[tid][0][0]
                    x_end   = track_history[tid][-1][0]
                    x_diff  = x_end - x_start
                    if x_diff > direction_threshold:
                        numRight += 1
                        log_stats("Right", x_start, x_end)
                    elif x_diff < -direction_threshold:
                        numLeft += 1
                        log_stats("Left", x_start, x_end)
                # cleanup after counting
                del track_history[tid]
                if tid in bbox_history:
                    del bbox_history[tid]
        # else: skip disappearance logic this frame if no IDs

        # Overlay totals
        cv2.putText(frame, f'Left: {numLeft}, Right: {numRight}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # -- V2: Head-band blur pass --
        if enable_blur == 1 and person_boxes:
            for (px1, py1, px2, py2) in person_boxes:
                apply_head_blur(frame, px1, py1, px2, py2, head_frac, blur_strength)
        # -------------------------------
    
        if not headless:
            cv2.imshow("Live People Tracking", frame)

        if save_video:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested by user.")
            break

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")

except Exception as e:
    print(f"Error: {str(e)}")

finally:
    # -------------------------------
    # Cleanup resources
    # -------------------------------
    print("Releasing resources...")
    cap.release()
    if save_video:
        out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    data_file.close()
    print(f"Total Left: {numLeft}, Right: {numRight}")
    print("Processing complete")
