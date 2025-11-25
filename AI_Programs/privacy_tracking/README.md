# Pedestrian Tracking + Privacy Strip

**Thor-Optimized Version (linux/amd64 only)**

This version extends the original Left-Right-Tracker by adding an optional **privacy strip**—a lightweight head-band blur applied to the upper portion of each detected person. This preserves the simplicity of the app while adding a layer of privacy protection suitable for public-facing deployments.

> **Note:** Optimized and tested on Thor (linux/amd64).
> Jetson Orin (linux/arm64) compatibility will be added in a future update.

---

## How it works

### 1. **Input selection**

The app accepts:

* Local video files (`.mp4`, `.avi`, etc.)
* Network streams (`rtsp://`, `http://`)
* Waggle camera (when `CAMERA_FALLBACK=1`)

---

### 2. **YOLOv8 + ByteTrack tracking**

The model detects people and assigns each track a persistent ID:

```python
model.track(..., tracker="bytetrack.yaml")
```

Smoothed bounding boxes and center-point histories help stabilize tracking.

---

### 3. **Direction classification**

Movement direction is determined when a person disappears from view:

```python
x_diff = last_center_x - first_center_x
```

* `x_diff > +DIR_THRESH` → **Right**
* `x_diff < -DIR_THRESH` → **Left**

Counts and metadata can be published to Waggle when running as a plugin.

---

### 4. **Privacy Strip (Optional)**

A configurable “privacy blur” is applied to the upper section of each detected person’s bounding box:

* Does **not** require face detection
* Works with any YOLO box
* Adds privacy with minimal speed overhead

Powered by:

```python
from privacy_strip import apply_privacy_strip
```

The blur is controlled via:

* `HEAD_FRAC` (height fraction of strip)
* `BLUR_STRENGTH` (kernel size scaling)

---

### 5. **Additional features**

* Saves annotated video (optional)
* Logs CPU/GPU/memory/temperature stats per event
* Retry logic for unstable RTSP streams
* Works both as a standalone script and inside a Waggle plugin

---

## Building the container (Thor only)

```bash
sudo docker build -t left-right-thor -f Dockerfile .
```

The image includes:

* `nvcr.io/nvidia/pytorch:25.08-py3`
* Ultralytics 8.3.x (pinned)
* Python 3.12 wheels for numpy 2.x, OpenCV-headless, etc.
* `pywaggle[vision]` for plugin/camera support
* `privacy_strip.py` for the blur feature

---

## Running on Thor

### Example: Local video

```bash
sudo docker run --gpus all -it --rm \
  -e STREAM=/data/test.mp4 \
  -e MODEL_SIZE=m \
  -e DIR_THRESH=100 \
  -e ENABLE_PRIVACY=1 \
  -v /absolute/path/to/videos:/data \
  left-right-thor:v1.0
```

### Example: RTSP stream

```bash
sudo docker run --gpus all -it --rm \
  -e STREAM="rtsp://192.168.1.10:554/stream" \
  -e ENABLE_PRIVACY=1 \
  left-right-thor:v1.0
```

### Example: Waggle camera fallback

```bash
sudo docker run -it --rm \
  -e CAMERA_FALLBACK=1 \
  -e CAMERA=left \
  -e ENABLE_PRIVACY=1 \
  left-right-thor:v1.0
```

---

## Recommended Thor run flags

```bash
docker run --rm \
   --gpus all \
   --ipc=host \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -it \
   -e STREAM=/data/test.mp4 \
   -v /absolute/path/to/videos:/data \
   left-right-thor:v1.0
```

### Quick explanations

| Flag                      | Reason                                |
| ------------------------- | ------------------------------------- |
| `--gpus all`              | Enables NVIDIA acceleration           |
| `--ipc=host`              | Prevents PyTorch shared-memory errors |
| `--ulimit memlock=-1`     | Allows CUDA memory pinning            |
| `--ulimit stack=67108864` | Avoids stack overflows                |
| `-v /path:/data`          | Mounts host videos into the container |

---

## Saving Output Locally

(*for testing without Beehive*)

Create a folder:

```bash
mkdir -p /home/thorwaggle/Desktop/left_right_output
```

Run with mounted output:

```bash
sudo docker run --rm \
   --gpus all \
   --ipc=host \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -it \
   -e STREAM="/data/people_walking.mp4" \
   -e SAVE_OUTPUT=1 \
   -e ENABLE_PRIVACY=1 \
   -v /path/to/your/videos:/data \
   -v /home/thorwaggle/Desktop/left_right_output:/app/output \
   left-right-thor:v1.0
```

Results appear in:

```
/home/thorwaggle/Desktop/left_right_output/<timestamp>/
```

---

## Environment Variables

| Variable          | Description                                  | Default      |
| ----------------- | -------------------------------------------- | ------------ |
| `STREAM`          | File path or RTSP/HTTP URL                   | ""           |
| `CAMERA_FALLBACK` | Use Waggle camera if STREAM not set          | 0            |
| `CAMERA`          | Waggle camera name                           | ""           |
| `MODEL_SIZE`      | YOLOv8 model size (`n/s/m/l/x`)              | `m`          |
| `DIR_THRESH`      | Pixel threshold for direction classification | `100`        |
| `ENABLE_PRIVACY`  | Enable head-band blur                        | `0`          |
| `HEAD_FRAC`       | Fraction of bbox height to blur              | `0.33`       |
| `BLUR_STRENGTH`   | Blur strength (larger = stronger)            | `25`         |
| `LIVE_OUTPUT`     | Show GUI window                              | `0`          |
| `SAVE_OUTPUT`     | Save output video                            | `0`          |
| `OUT_PATH`        | Output video name                            | `output.mp4` |

---

## Output Files

* `output/output.mp4` — Annotated video (optional)
* `output/data.csv` — Movement + system stats
* Timestamped directory per run

---

## Repository Structure

```
left_right_tracking/
├── Dockerfile
├── sage.yaml
├── requirements.txt
├── main.py
├── privacy_strip.py
├── README.md
└── output/
```

---

## Funding

Supported by NSF Grants: **2436842**, **1935984**, **2331263**

---

## Contributors

* Micheal Papka
* Michael Cortez
* Om Patel
* Elizabeth Cardoso
* Fatima Mora Garcia
* The Sage Team (ANL, Northwestern, UIC EVL)

