# Left-Right-Tracker (YOLOv8 People Flow App)

**Thor‑Optimized Version (linux/amd64 only)**

Left-Right-Tracker estimates left-versus-right pedestrian movement using YOLOv8 with the Ultralytics ByteTrack tracker.

> **Note:** This build is currently optimized and tested for Thor (linux/amd64). Jetson Orin (linux/arm64) compatibility will be added in a future update.

## How it works

1. **Input selection**
   - Local video file (e.g. `.mp4`, `.avi`, `.mov`)
   - Network stream (`rtsp://`, `http://`)
   - Waggle camera (when `CAMERA_FALLBACK=1`)

   Priority: `STREAM` → camera fallback (optional)

2. **YOLOv8 + ByteTrack tracking**
   - A YOLOv8 model (one of `n`, `s`, `m`, `l`, `x`) is loaded and tracking is performed via ByteTrack:

```python
model.track(..., tracker="bytetrack.yaml")
```

3. **Direction classification**
   - For each tracked person, the center X coordinates are stored across frames.
   - When a track disappears, the direction is computed as:

```python
x_diff = last_center_x - first_center_x
```

   - Classification:
     - `x_diff > +DIR_THRESH` → counted as **Right**
     - `x_diff < -DIR_THRESH` → counted as **Left**

   - Counts can be published as Waggle metadata when used inside a plugin environment.

4. **Optional features**
   - Save annotated video output
   - Display live GUI window (Thor development only)
   - Log system stats (CPU, GPU, memory, temperature)
   - Retry logic for unstable RTSP sources

## Building the container (Thor only)

Thor uses an NVIDIA GPU-optimized PyTorch base image. Build with Docker on a Thor node:

```bash
sudo docker build -t left-right-thor -f Dockerfile .
```

This build includes:
- `nvcr.io/nvidia/pytorch:25.08-py3` base image
- Ultralytics 8.3.24 (pinned)
- Python 3.12-compatible wheels (numpy 2.x, opencv-headless, etc.)
- `pywaggle[vision]` for plugin support

## Running on Thor 

Example: local video of people walking

```bash
sudo docker run --gpus all -it --rm \
  -e STREAM=/data/test.mp4 \
  -e MODEL_SIZE=m \
  -e DIR_THRESH=100 \
  -v /absolute/path/to/videos:/data \
  left-right-thor
```

Example: RTSP stream

```bash
sudo docker run --gpus all -it --rm \
  -e STREAM="rtsp://192.168.1.10:554/stream" \
  -e MODEL_SIZE=s \
  -e DIR_THRESH=120 \
  left-right-thor
```

Example: Waggle camera fallback

```bash
sudo docker run -it --rm \
  -e CAMERA_FALLBACK=1 \
  -e CAMERA=left \
  -e MODEL_SIZE=n \
  -e DIR_THRESH=80 \
  left-right-thor
```

### Recommended run flags

For Thor we recommend the following (enables GPU and prevents common PyTorch/CUDA shared-memory issues):

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

What these mean:

| Flag                      | Why it matters (short)                                  |
| ------------------------- | ------------------------------------------------------- |
| `--gpus all`              | Use the NVIDIA GPU for YOLOv8 (much faster).            |
| `--ipc=host`              | Gives PyTorch enough shared memory so it doesn’t crash. |
| `--ulimit memlock=-1`     | Allows CUDA to lock memory (prevents GPU errors).       |
| `--ulimit stack=67108864` | Provides a larger stack to avoid segmentation faults.   |
| `-it`                     | Shows logs and runs interactively (useful for testing). |
| `-e STREAM=...`           | Sets your input (video file or RTSP URL).               |
| `-v /path:/data`          | Mounts your local videos into the container (optional). |


## 📦 Saving Output Locally (local testing)
If your node is not connected to Beehive, or you want to test locally, mount a host folder into the container so output files are saved outside the container and remain available after it exits.
This preserves:
- `data.csv`
- `output.mp4` (if `SAVE_OUTPUT=1`)
- the timestamped `output/` directory

Quick steps:

```bash
# create a host folder for results (change path as needed)
mkdir -p /home/thorwaggle/Desktop/left_right_output
```

Example run (replace `/path/to/your/videos` with your video folder):

```bash
sudo docker run --rm \
   --gpus all \
   --ipc=host \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -it \
   -e STREAM="/data/people_walking.mp4" \
   -e SAVE_OUTPUT=1 \
   -v /path/to/your/videos:/data \
   -v /home/thorwaggle/Desktop/left_right_output:/app/output \
   left-right-thor:v1.0
```
After the run, outputs will be under:
```
/home/thorwaggle/Desktop/left_right_output/<timestamp>/
```
## Environment variables

| Variable          | Type    | Description                                   | Default      |
| ----------------- | ------- | --------------------------------------------- | ------------ |
| `STREAM`          | string  | File path or RTSP/HTTP URL                    | `""`        |
| `CAMERA_FALLBACK` | boolean | Use Waggle camera when stream is not set      | `0`          |
| `CAMERA`          | string  | Name of Waggle camera                         | `""`        |
| `MODEL_SIZE`      | string  | YOLOv8 size (`n`/`s`/`m`/`l`/`x`)              | `m`          |
| `DIR_THRESH`      | int     | Pixel threshold for left/right classification | `100`        |
| `LIVE_OUTPUT`     | boolean | Show GUI window (Thor dev only)               | `0`          |
| `SAVE_OUTPUT`     | boolean | Save annotated video output                   | `0`          |
| `OUT_PATH`        | string  | Output video file                             | `output.mp4` |

## Output files

- `output/output.mp4` — optional annotated video
- `output/data.csv` — logs movement events and system stats (left/right counts, CPU/GPU utilization, RAM usage, temperature readings)

## Repository structure

```
left_right_tracking/
├── Dockerfile
├── sage.yaml
├── requirements.txt
├── main.py
├── README.md
└── output/
```

## Funding

Supported by NSF Grants: 2436842, 1935984, 2331263


## Contributors
Micheal Papka
Michael Cortez  
Om Patel  
Elizabeth Cardoso 
Fatima Mora Garcia  
The Sage Team (ANL, Northwestern, UIC EVL)
```
