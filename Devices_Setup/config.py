"""
NIU Image Stream Server - Configuration Module
Contains all configuration settings for the application
"""

import os
from datetime import datetime

# Demo mode configuration
DEMO_MODE = not os.path.exists("/home/sage/nfs/NIU/top")

# Camera options - user can select between top and bottom cameras
if DEMO_MODE:
    CAMERA_OPTIONS = {
        "demo_top": "./data",
        "demo_bottom": "./data"
    }
    BASE_DIR = "./data"  # Use local data directory
    DEFAULT_CAMERA = "demo_top"
else:
    CAMERA_OPTIONS = {
        "top": "/home/sage/nfs/NIU/top",
        "bottom": "/home/sage/nfs/NIU/bottom"
    }
    BASE_DIR = "/home/sage/nfs/NIU/top"  # Default camera
    DEFAULT_CAMERA = "top"  # Default camera selection

# Date range configuration
START_DATE = datetime(2021, 7, 25)
END_DATE = datetime(2023, 7, 11, 3)

# Video stream settings
TARGET_FPS = 4  # Base FPS for frame delivery
BUFFER_SIZE = 150  # Larger buffer for smoother streaming
MIN_BUFFER = 30
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
    "1_frame_per_10s": 0.1,    # 1 frame every 10 seconds (very slow for AI analysis)
    "1_frame_per_5s": 0.2,     # 1 frame every 5 seconds (slow for AI analysis)
    "1_frame_per_2s": 0.5,     # 1 frame every 2 seconds (slow)
    "1_frame_per_second": 1,   # 1 frame per second (normal)
    "2_frames_per_second": 2,  # 2 frames per second (medium)
    "4_frames_per_second": 4,  # 4 frames per second (default)
    "6_frames_per_second": 6,  # 6 frames per second (fast)
    "8_frames_per_second": 8,  # 8 frames per second (faster)
    "10_frames_per_second": 10, # 10 frames per second (very fast)
    "30_frames_per_second": 30  # 30 frames per second (super fast time-lapse)
}

# Data exclusions (dates to skip entirely across cameras)
from datetime import date as _date
EXCLUDED_DATES = { _date(2021, 7, 21) }
