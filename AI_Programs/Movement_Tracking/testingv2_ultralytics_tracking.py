#!/usr/bin/env python3
# People-counter (beta, Ultralytics tracker): Count people moving left vs right.
# Uses YOLOv8 + Ultralytics built-in ByteTrack tracker. Restores:
# - Bounded history & disappearance-based counting
# - Short-term bbox smoothing
# - Configurable direction threshold (DIR_THRESH, default 100)
#
#
# Usage:
#   DIR_THRESH=100 python beta_ultralytics_tracking_patched.py
# /home/waggle/AI_Test/SageEdge/AI_Programs/Movement_Tracking

import os
import requests
from dotenv import load_dotenv
import cv2
import csv
import re
import time
import psutil
from jtop import jtop
import threading
import queue
import datetime
import subprocess
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import time 

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

load_dotenv()

API_URL = os.getenv("API_SECRET_CALL")
SECRET_KEY = os.getenv("API_SECRET_KEY")
HEADER_NAME = os.getenv("API_HEADER_NAME") 

url = f"{API_URL}/api/send_data"

headers = {
    HEADER_NAME: SECRET_KEY,
    "Content-Type": "application/json"
}

if not API_URL or not SECRET_KEY:
    print("Error: API_SECRET_CALL or API_SECRET_KEY not found in .env file.")
    exit()


print("There are 5 different YOLO models to choose from: n (nano), s (small), m (medium), l (large), x (xlarge).")
AI_model = input("Please enter the YOLOv8 model size to use [n/s/m/l/x]: ").strip().lower()
if AI_model not in ['n', 's', 'm', 'l', 'x']:
    print("Invalid model choice, defaulting to 'm' (medium)")
    AI_model = 'm'
print()

cwd = os.path.dirname(os.path.abspath(__file__))
person_model = 'yolov8' + AI_model + '.engine'
model_path = os.path.join(cwd, person_model)

print("Loading YOLO model... (first run may download weights)")

if os.path.isfile(model_path):
    person_model = YOLO(model_path, task='detect')
else:
    print(f"Engine not found. exporting {AI_model} to engine...")
    temp_model = YOLO('yolov8' + AI_model + '.pt')
    temp_model.export(format='engine', device=0)
    person_model = YOLO(model_path, task='detect')

#counting sensitivity (pixels across the image width)
direction_threshold = int(os.getenv("DIR_THRESH", "100"))
print(f"Direction threshold set to {direction_threshold} px (override with DIR_THRESH env var).")

#-----Sending program status to server-----
def send_status(status):
    url = f"{API_URL}/api/send_status"
    payload = {
        "status": status,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
    
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.text)
    except Exception as e:
        print(f"An error occurred: {e}")

print("Sending program status")
send_status(1)


#-------------------------------
# --- V2: Face blurring settings ---
FACE_BLUR = bool(int(os.getenv("FACE_BLUR", "1")))   # set 0 to disable
FACE_CONF = float(os.getenv("FACE_CONF", "0.35"))    # face detector confidence
FACE_EVERY = int(os.getenv("FACE_EVERY", "1"))       # run face detection every N frames (1 = every frame)

face_model = None
if FACE_BLUR:
    try:
        # Ultralytics face model; downloads on first use
        face_weights = os.getenv("FACE_WEIGHTS", "/home/waggle/SageEdge/AI_Programs/Movement_Tracking/yolov8n-face.pt")
        face_model = YOLO(face_weights)
        print(f"Face blurring enabled (conf={FACE_CONF}, every={FACE_EVERY} frame(s)) using: {face_weights}")
    except Exception as e:
        print(f"Warning: could not load face model: {e}. Face blurring disabled.")
        FACE_BLUR = False

# Helps catch small/distant faces + privacy fallback
FACE_UPSCALE  = float(os.getenv("FACE_UPSCALE", "1.8"))  # 1.6–2.0 is good
HEAD_FAILSAFE = bool(int(os.getenv("HEAD_FAILSAFE", "1")))
HEAD_RATIO    = float(os.getenv("HEAD_RATIO", "0.38"))   # top % of person box to blur if no face found

# -------------------------------
# Create output directory based on current timestamp
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

stat_queue = queue.Queue()

# -------------------------------
# Logging Function
# -------------------------------
def log_stats(direction=None, x_start=None, x_end=None):
    cdt = datetime.datetime.now()
    date = cdt.strftime("%Y-%m-%d")
    time = cdt.strftime("%H:%M:%S")

    cpu_util = None
    ram_used = None
    ram_total = None
    gpu_util = None
    temps_parsed = {"cpu": None, "gpu": None}

    try:
        with jtop() as jetson:
            if jetson.ok():
                gpu_util = jetson.stats.get('GPU', 0)
                cpu_util = jetson.stats.get('CPU', 0)
                
                ram_data = jetson.memory.get('RAM', {})
                if 'used' in ram_data:
                    ram_used = int(ram_data['used'] / 1048576)
                if 'tot' in ram_data:
                    ram_total = int(ram_data['tot'] / 1048576)

                temps = jetson.temperature
                if 'CPU' in temps:
                    temps_parsed["cpu"] = temps['CPU'].get('temp')
                if 'GPU' in temps:
                    temps_parsed["gpu"] = temps['GPU'].get('temp')
            else:
                raise Exception("jtop daemon not responding")

    except Exception as e:
        cpu_util = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        ram_used = int(mem.used / 1024**2)
        ram_total = int(mem.total / 1024**2)
        temps_parsed = {"cpu": None, "gpu": None}
        gpu_util = None

    row_data = [
        date, 
        time, 
        direction if direction is not None else "None", # Strings can be "None"
        x_start if x_start is not None else 0,          # Numbers should fallback to 0
        x_end if x_end is not None else 0,
        round(cpu_util, 1) if cpu_util is not None else 0.0,
        ram_used if ram_used is not None else 0,
        ram_total if ram_total is not None else 0,
        gpu_util if gpu_util is not None else 0.0,
        temps_parsed["cpu"] if temps_parsed["cpu"] is not None else 0.0,
        temps_parsed["gpu"] if temps_parsed["gpu"] is not None else 0.0
    ]

    csv_data = ",".join(map(str, row_data))

    csv_writer.writerow(row_data)
    data_file.flush()
    
    payload = {
        "name": "TestUser",
        "data": csv_data
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
    
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.text)
    except Exception as e:
        print(f"An API error occurred: {e}")

# --------------------------------
# V2: Face blurring function
# --------------------------------

def blur_roi(frame, x1, y1, x2, y2):
    # clamp to frame bounds
    H, W = frame.shape[:2]
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    # kernel proportional to face size; ensure odd >= 3
    kx = max(3, ( (x2 - x1) // 7 ) | 1)  # make odd
    ky = max(3, ( (y2 - y1) // 7 ) | 1)
    blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
    roi[:] = blurred

DEVICE_NAME = "Arcade Waggle"
INTERVAL = 4 * 60 # 4 minutes
# -------------------------------
# Heartbeat function
# -------------------------------
def send_heartbeat():
    payload = {
        "device": DEVICE_NAME
    }
    while True:
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=2)
            if response.status_code == 200:
                print(f"Heartbeat sent successfully at {datetime.datetime.now()}")
            else:
                print(f"Server returned error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to server. Heartbeat failed.")
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(INTERVAL)

# h = threading.Thread(target=send_heartbeat, daemon=True)
# h.start()

# -------------------------------
# Open video stream
# -------------------------------
print("Connecting to video feed...")
cap = cv2.VideoCapture(video_path)

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
    start_time = time.time()
    # --------------------------------------------
    # main processing loop with retry + reconnect
    # --------------------------------------------
    max_soft_fail = 20    # how many bad reads before reconnect
    soft_fail = 0         # counter for consecutive failed frames
    
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        t_read = time.time()
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
        frame = cv2.resize(frame, (1280, 720))
        # reset fail counter on success
        soft_fail = 0
        frame_idx += 1
        # -------------------------------
        
        # Run Ultralytics tracker on this frame (ByteTrack)
        results = person_model.track(
            frame, classes=[0], conf=0.4,
            tracker="botsort.yaml",
            persist=True, verbose=False,
            device=0
        )
        t_ai = time.time()
        read_ms = (t_read - t_start) * 1000
        ai_ms   = (t_ai - t_read) * 1000
        if frame_idx % 1000 == 0:
            print(f"Read Time: {read_ms:.1f}ms | AI Time: {ai_ms:.1f}ms")
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
                if not headless:
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
                        t = threading.Thread(target=log_stats, args=("Right", x_start, x_end))
                        t.start()
                    elif x_diff < -direction_threshold:
                        numLeft += 1
                        t = threading.Thread(target=log_stats, args=("Left", x_start, x_end))
                        t.start()
                # cleanup after counting
                del track_history[tid]
                if tid in bbox_history:
                    del bbox_history[tid]
        # else: skip disappearance logic this frame if no IDs

        if not headless:
            # Overlay totals
            cv2.putText(frame, f'Left: {numLeft}, Right: {numRight}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # -- V2: Face blurring pass (seq after people; BEFORE write/show) --
        if FACE_BLUR and (frame_idx % FACE_EVERY == 0) and person_boxes:
            faces_found = 0
            for (px1, py1, px2, py2) in person_boxes:
                roi = frame[py1:py2, px1:px2]
                if roi.size == 0:
                    continue

                # Upscale ROI to help detect small faces
                roi_up = cv2.resize(
                    roi, None, fx=FACE_UPSCALE, fy=FACE_UPSCALE, interpolation=cv2.INTER_LINEAR
                ) if FACE_UPSCALE != 1.0 else roi

                dets = face_model.predict(roi_up, conf=FACE_CONF, verbose=False)
                if dets and hasattr(dets[0], "boxes") and dets[0].boxes is not None:
                    for fx1, fy1, fx2, fy2 in dets[0].boxes.xyxy.cpu().numpy():
                        # map back from upscaled ROI to full frame
                        if FACE_UPSCALE != 1.0:
                            fx1, fy1, fx2, fy2 = [int(v / FACE_UPSCALE) for v in (fx1, fy1, fx2, fy2)]
                        blur_roi(frame, px1 + int(fx1), py1 + int(fy1), px1 + int(fx2), py1 + int(fy2))
                        faces_found += 1

            # Privacy fallback: blur the "head" slice if no faces this frame
            if HEAD_FAILSAFE and faces_found == 0:
                for (px1, py1, px2, py2) in person_boxes:
                    head_h = int((py2 - py1) * HEAD_RATIO)
                    blur_roi(frame, px1, py1, px2, py1 + head_h)
        # -- end face blurring --
        # --- CALCULATE FPS ---
        new_frame_time = time.time()
        
        # Avoid division by zero
        diff = new_frame_time - prev_frame_time
        fps = 1 / diff if diff > 0 else 0
        
        prev_frame_time = new_frame_time

        if not headless:
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Live People Tracking", frame)
    
        # Optional: Print to terminal every 100 frames so you don't spam logs
        if frame_idx % 1000 == 0:
            print(f"Current Speed: {fps:.2f} FPS")

        if save_video:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested by user.")
            break

        if frame_idx % 1000 == 0:
            print(f"Processed frame {frame_idx}")
        
        if frame_idx % 1000 == 0:
            print(f"Current Speed: {fps:.2f} FPS")


except Exception as e:
    print(f"Error: {str(e)}")

finally:
    # -------------------------------
    # Cleanup resources
    # -------------------------------
    print("Releasing resources...")
    end_time = time.time()   # Stop the clock
    total_seconds = end_time - start_time
    print(f"Total execution time: {total_seconds:.2f} seconds")
    print("sending program status")
    send_status(0)
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
