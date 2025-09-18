# People-counter: Count the number of people moving left vs right using YOLOv8 and DeepSORT

import cv2, os, datetime, psutil, subprocess, re, csv
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

print("This program uses YOLOv8 and DeepSORT to count the amount of people that have moved left/right from a video file or live camera feed. This then creates a CSV file with data.")
print()

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

print("You can choose to either to input a video file path or a live camera feed link.")
video_input_file = input("Please enter the exact path to your video file to be processed: ")

# -------------------------------
# Paths and Initialization
# -------------------------------
# video_path = "/media/waggle/New Volume/park_walking.mp4"  # Path to input video
video_path = video_input_file
# video_path = "http://77.222.181.11:8080/mjpg/video.mjpg"  # Alternative: live camera feed (enter your live link)

# -------------------------------
# Initialize YOLO model and DeepSORT tracker
# -------------------------------
print("Loading YOLO model...")
# Pretrained YOLOv8 models. Switch between yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
model = YOLO('yolov8m.pt')
tracker = DeepSort(
    max_age=15, # How long a detection ID stays after losing object
    n_init=2, # Number of consecutive detections for an object to receive ID 
    embedder='torchreid', # Model for re-identification
    half=True, 
    bgr=True, # Color chanel
    max_cosine_distance=0.5, # Appearance similarity for re-id, the lower the more similar the object has to look
    nn_budget=100 # The storage for memorizing how the object looks
)

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
# Initialize movement counters
numLeft = 0
numRight = 0
points = defaultdict(list)  # Store track_id -> list of center points

# Initialize system usage tracking (CPU, memory, temps)
psutil.cpu_percent(interval=None)

# -------------------------------
# Logging Function
# -------------------------------
def log_stats(direction=None, x_start=None, x_end=None):
    # Timestamp
    cdt = datetime.datetime.now()
    date = cdt.strftime("%Y-%m-%d")
    time = cdt.strftime("%H:%M:%S")

    # Run tegrastats (or fallback on psutil if Tegra not available)
    try:
        proc = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.stdout.readline().decode("utf-8").strip()
        proc.kill()

        # RAM
        ram_match = re.search(r"RAM (\d+)/(\d+)MB", out)
        ram_used = int(ram_match.group(1)) if ram_match else None
        ram_total = int(ram_match.group(2)) if ram_match else None

        # GPU
        gpu_match = re.search(r"GR3D_FREQ (\d+)%", out)
        gpu_util = int(gpu_match.group(1)) if gpu_match else None

        # CPU (average)
        cpu_match = re.search(r"CPU \[([^\]]+)\]", out)
        cpu_util = None
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
        gpu_util = None
        temps_parsed = {"cpu": None, "gpu": None}

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

# Read first frame to get resolution and FPS
ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame. Using default resolution 1280x720 at 30 FPS.")
    W, H = 1280, 720
    fps = 30
else:
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

print(f"Video resolution: {W}x{H} at {fps} FPS")
print(f"Total frames: {total_frames}, Duration: {duration:.2f} seconds")


# -------------------------------
# Setup display / video writer
# -------------------------------
if os.name == 'posix' and "DISPLAY" not in os.environ:
    headless = True
    print("No display detected. Running in headless mode, video will be saved to output.mp4.")

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (W, H))

if not headless:
    cv2.namedWindow("Live People Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live People Tracking", W, H)

frame_idx = 0
track_history = defaultdict(list)  # Store recent bounding boxes for smoothing

# -------------------------------
# Main Processing Loop
# -------------------------------
print("Starting processing loop...")
try:
    while True:
        # Read next frame, retry if failed
        ret, frame = cap.read()
        if not ret:
            # Check if we've reached the end of the video
            if total_frames > 0 and frame_idx >= total_frames:
                print(f"Reached end of video ({total_frames} frames processed)")
                break
            else:
                print("Frame read failed, attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(video_path)
                # Wait a bit before trying again
                cv2.waitKey(1000)
                continue
        
        frame_idx += 1
        display_frame = frame.copy()

        # -------------------------------
        # Person Detection (YOLOv8)
        # -------------------------------
        results = model(frame, classes=[0], conf=0.4, verbose=False)[0]  # Detect only class "person"

        # Parse YOLO detections into DeepSORT format
        detections = []
        for box in results.boxes:
            if int(box.cls) != 0:
                continue
            if box.xyxy is None or len(box.xyxy[0]) != 4:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            detections.append(([x1, y1, w, h], conf, 'person'))

        # -------------------------------
        # Object Tracking (DeepSORT)
        # -------------------------------
        tracks = tracker.update_tracks(detections, frame=frame)

        # Identify tracks that disappeared to count movement
        current_track_ids = [t.track_id for t in tracks if t.is_confirmed()]
        disappeared_tracks = set(points.keys()) - set(current_track_ids)

        for tid in disappeared_tracks:
            if len(points[tid]) > 1:
                x_start = points[tid][0][0]
                x_end = points[tid][-1][0]
                # Count movement to the right
                if x_start < x_end and (x_end - x_start) > 100:
                    numRight += 1
                    log_stats("Right", x_start, x_end)
                # Count movement to the left
                elif x_start > x_end and (x_start - x_end) > 100:
                    numLeft += 1
                    log_stats("Left", x_start, x_end)
                del points[tid]

        # -------------------------------
        # Draw bounding boxes and IDs
        # -------------------------------
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            points[tid].append(center)
            if len(points[tid]) > 20:
                points[tid] = points[tid][-20:]

            # Smooth bounding box positions
            track_history[tid].append((x1, y1, x2, y2))
            if len(track_history[tid]) > 5:
                track_history[tid] = track_history[tid][-5:]
            smoothed = np.mean(track_history[tid], axis=0).astype(int)
            x1, y1, x2, y2 = smoothed

            # Determine direction arrow
            if len(points[tid]) > 1:
                x_diff = points[tid][-1][0] - points[tid][0][0]
                current_direction = "Right" if x_diff > 0 else "Left" if x_diff < 0 else ""
            else:
                current_direction = ""

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(display_frame, f'ID {tid} {current_direction}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Display total counts on the frame
        cv2.putText(display_frame, f'Left: {numLeft}, Right: {numRight}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not headless:
            cv2.imshow("Live People Tracking", display_frame)

        if save_video:
            out.write(display_frame)
        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
    cv2.destroyAllWindows()
    data_file.close()
    print(f"Total Left: {numLeft}, Right: {numRight}")
    print("Processing complete")
