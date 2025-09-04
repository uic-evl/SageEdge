"""
NIU Image Stream Server - Configuration Module
Contains all configuration settings for the application
"""

import os
from datetime import datetime

# Data root (can be overridden to point at a mounted remote dataset)
DATA_ROOT = os.environ.get("DATA_ROOT", "/home/sage/nfs/NIU")
TOP_SUBDIR = os.environ.get("TOP_SUBDIR", "top")
BOTTOM_SUBDIR = os.environ.get("BOTTOM_SUBDIR", "bottom")

# Demo mode configuration (true if expected top camera path not present)
DEMO_MODE = not os.path.exists(os.path.join(DATA_ROOT, TOP_SUBDIR))

# Camera options - user can select between top and bottom cameras
if DEMO_MODE:
    CAMERA_OPTIONS = {
        "demo_top": "./data",
        "demo_bottom": "./data"
    }
    BASE_DIR = "./data"
    DEFAULT_CAMERA = "demo_top"
else:
    CAMERA_OPTIONS = {
        "top": os.path.join(DATA_ROOT, TOP_SUBDIR),
        "bottom": os.path.join(DATA_ROOT, BOTTOM_SUBDIR)
    }
    BASE_DIR = CAMERA_OPTIONS["top"]
    DEFAULT_CAMERA = "top"

# Date range configuration (updated for new dataset span)
# New data starts 2021-07-23 21:00:00 and ends 2024-09-04 18:00:00
START_DATE = datetime(2021, 7, 23, 21, 0, 0)
END_DATE = datetime(2024, 9, 4, 18, 0, 0)

# Optional environment overrides (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
_env_start = os.environ.get("START_DATE")
_env_end = os.environ.get("END_DATE")
def _parse_dt(val: str):
    try:
        return datetime.fromisoformat(val)
    except Exception:
        try:
            return datetime.strptime(val, "%Y-%m-%d")
        except Exception:
            return None
if _env_start:
    _p = _parse_dt(_env_start)
    if _p:
        START_DATE = _p
if _env_end:
    _p = _parse_dt(_env_end)
    if _p:
        END_DATE = _p

# Video stream settings
TARGET_FPS = 4  # Base FPS for frame delivery
BUFFER_SIZE = 150  # Larger buffer for smoother streaming
# MIN_BUFFER removed (unused)
PRELOAD_MINUTES = 30  # Load 30 minutes worth of images for smoother playback
FRAME_INTERPOLATION = True  # Add frame interpolation for missing frames

# Database settings
DB_PATH = "./data/niu_photo_analysis.db"  # Analysis database path (relative to app)
LOCAL_DB_PATH = "./data/niu_photo_analysis.db"  # Local database path for development

# Session management
SECRET_KEY = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')
SESSION_TIMEOUT = 30 * 60  # 30 minutes

# Threading configuration
MAX_WORKERS = 4  # Number of worker threads for the thread pool executor

# Frame rate options for speed control
FRAMES_PER_SECOND_OPTIONS = {
    # Trimmed to enforced clamp (<= 8 fps)
    "1_frame_per_10s": 0.1,
    "1_frame_per_5s": 0.2,
    "1_frame_per_2s": 0.5,
    "1_frame_per_second": 1,
    "2_frames_per_second": 2,
    "4_frames_per_second": 4,
    "6_frames_per_second": 6,
    "8_frames_per_second": 8,
}

# Data exclusions removed (previous EXCLUDED_DATES feature deprecated)
