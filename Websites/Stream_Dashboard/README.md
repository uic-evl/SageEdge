## NIU Weather Dataset Stream Server

An MJPEG Flask server that streams a historical two-year NIU camera photo dataset as video, with per-user sessions, playback controls, and a simple web UI. Runs against the real dataset when mounted, or in demo mode with generated frames.

## Project Overview

This app exposes a browser UI and API to browse/“play back” time-lapse images as a smooth MJPEG stream.
- Multiple users get isolated sessions (own speed, time, camera, loop mode).
- Top/Bottom camera switching when dataset is available; otherwise demo mode synthesizes frames.
- Smooth playback with buffering, interpolation fallback, and timestamp overlays.
   - Default frame rate is 1.0 fps for new sessions.
   - Timestamp overlay is shown in the top-left of the video by default.

## Installation Instructions

Prereqs
- Docker (recommended), or Python 3.10+ with system libs for OpenCV.

Option A — Docker (recommended)
1) Build image
```bash
docker build -t mjpeg-stream .
```
2) Run container (demo mode if dataset not mounted)
```bash
docker run -d --name mjpeg-server \
   -p 8080:8080 \
   -v "$(pwd)/data:/app/data" \
   mjpeg-stream
```
3) To mount the NIU dataset (enables real images)
```bash
docker run -d --name mjpeg-server \
   -p 8080:8080 \
   -v /home/sage/nfs/NIU:/home/sage/nfs/NIU:ro \
   -v "$(pwd)/data:/app/data" \
   mjpeg-stream
```
4) Stop/remove
```bash
docker stop mjpeg-server && docker rm mjpeg-server
Container runtime notes
- The image includes a Docker HEALTHCHECK that probes /api/health.

Option B — Local Python
1) Install system deps (Linux)
```bash
sudo apt-get update && sudo apt-get install -y \
   libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```
2) Create venv and install
```bash
python server.py

## Usage

Web UI
- Direct URL: GET /video_feed/{session_id} (no cookies; see /my-stream for your URL)

Control API (single endpoint)
- POST /control with JSON body. Actions:
   - jump_to_datetime: { action: "jump_to_datetime", datetime: "YYYY-MM-DD HH:MM:SS" }
   - toggle_playback: { action: "toggle_playback" }
   - set_speed: { action: "set_speed", frames_per_second: number }
   - toggle_timestamp: { action: "toggle_timestamp" }
   - set_loop_mode: { action: "set_loop_mode", loop_mode: "full"|"day"|"hour"|"none" }
   - set_camera: { action: "set_camera", camera: one of config.CAMERA_OPTIONS keys }

Status and session info
- GET /status — detailed session/server status for the current session (includes data_range and excluded_dates)
- GET /api/session — returns your session_id and stream URLs
- GET /api/cameras — camera list and current selection
- GET /api/health — health snapshot (base dir, DB presence, sessions)
- GET /api/analysis/dashboard — aggregate counts for weather/people/time (if analysis DB is present)
- GET /api/analysis/search?weather=&people=&visibility=&time=&limit=50 — filtered photo rows from the analysis DB

Behavior note: If the requested date/time has no images, the server automatically jumps to the nearest day with images and returns that actual_time in the /control response. The UI reflects this and updates the date/time controls accordingly.

## Functionality Breakdown

Core modules
- server.py: Flask app, MJPEG generator, per-session producer threads, /, /video_feed, /control, /status, /api/* info routes. Also creates fallback frames and manages buffering/timing.
- config.py: DEMO_MODE detection, camera paths, dates, buffer sizes, FPS options, secrets, thread settings.
- session_manager.py: Session bookkeeping structures, inactivity cleanup helpers, VideoController class used by some API code paths.
- image_processor.py: Filename timestamp parsing, demo frame generation, image range listing and loading.
- utils.py: JPEG encoding, date parsing/formatting, overlays, info frame, RateLimiter helper.

Front-end assets
- index.html, styles.css, script.js: Main UI and controls (date picker defaults to 2021-07-25; 2021-07-21 excluded).
- my-stream.html: Shows your session info and direct stream URL.

Data and storage
- data/: created for database or local assets; not committed. DB path in config.py (./data/niu_photo_analysis.db by default). Real dataset expected under /home/sage/nfs/NIU when mounted.

## Contributing

1) Fork and create a feature branch.
2) Keep changes small and modular. Prefer adding tests for public behavior.
3) Run locally (Docker or Python) and verify stream UI, /status, and /control flows.
4) Open a PR with a clear description and screenshots if UI changes.

Coding notes
- Avoid duplicating config/constants across modules; prefer importing from config.py.
- Don’t commit logs, __pycache__, or data. See .gitignore.
- If you add new API routes, wire them in server.py and document them here.

## License

MIT License. See LICENSE if present; otherwise, all rights reserved by the project owner. Update this section if you add a formal LICENSE file.

