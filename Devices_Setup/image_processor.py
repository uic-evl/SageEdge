"""
NIU Image Stream Server - Image Processor Module
Handles image loading, processing and generation
"""

import os
import cv2
import numpy as np
import re
import logging
import random
from datetime import datetime, timedelta
from glob import glob
import threading

# Import configuration
from config import DEMO_MODE, BASE_DIR, PRELOAD_MINUTES, CAMERA_OPTIONS, EXCLUDED_DATES

# Configure logging
logger = logging.getLogger(__name__)

# Regex pattern for extracting timestamps from filenames
# Example filename: top_2021-07-24_00-00-02.jpg
TIMESTAMP_PATTERN = r'(?:top|bottom)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.jpg'

# ISO-like filenames e.g., 2022-07-07T06:00:02+0000.jpg or 2022-07-07T06:00:02.jpg
ISO_FILENAME_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:[+-]\d{4})?\.jpg$")

def extract_timestamp_from_filename(filename: str):
    """Extract timestamp from supported filename formats.

    Supports:
    - top_YYYY-MM-DD_HH-MM-SS.jpg / bottom_...
    - YYYY-MM-DDTHH:MM:SS[+zzzz].jpg
    """
    # Try camera-prefixed pattern first
    match = re.search(TIMESTAMP_PATTERN, filename)
    if match:
        try:
            date_part, time_part = match.group(1).split('_')
            time_part = time_part.replace('-', ':')
            return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.debug(f"Camera-prefix parse failed for {filename}: {e}")

    # Try ISO-like filename pattern
    m = ISO_FILENAME_PATTERN.match(filename)
    if m:
        try:
            date_str, time_str = m.group(1), m.group(2)
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.debug(f"ISO-like parse failed for {filename}: {e}")

    return None

def generate_demo_frame(timestamp, frame_number=0):
    """Generate a demo frame with date/time information when no real images are available"""
    # Create a blank image (800x600, RGB)
    height, width = 600, 800
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with a gradient background
    for y in range(height):
        color_value = int(180 * y / height) + 50
        cv2.line(frame, (0, y), (width, y), (color_value, color_value, color_value), 1)
    
    # Add some random "clouds"
    for _ in range(20):
        x = random.randint(0, width)
        y = random.randint(0, int(height/2))
        size = random.randint(20, 100)
        color = random.randint(150, 255)
        cv2.circle(frame, (x, y), size, (color, color, color), -1)
    
    # Format the timestamp
    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add the date and time
    cv2.putText(
        frame, 
        f"DEMO MODE - No actual images available", 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 160, 255), 
        2
    )
    
    cv2.putText(
        frame, 
        f"Date/Time: {time_str}", 
        (50, 100), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (255, 255, 255), 
        2
    )
    
    # Add frame number to create animation effect
    sun_x = 100 + (frame_number % 600)
    cv2.circle(frame, (sun_x, 150), 30, (0, 100, 255), -1)
    
    # Add a horizon line
    cv2.line(frame, (0, 400), (width, 400), (100, 120, 80), 3)
    
    return frame

def get_images_for_timerange(start_time, end_time, camera="top"):
    """Get a list of (timestamp, path) within the specified time range for the given camera.

    In demo mode, returns tuples of (timestamp, f"demo_{timestamp:%Y%m%d_%H%M%S}.jpg").
    """
    if DEMO_MODE:
        current = start_time
        results = []
        while current <= end_time:
            if current.date() not in EXCLUDED_DATES:
                results.append((current, f"demo_{current.strftime('%Y%m%d_%H%M%S')}.jpg"))
            current += timedelta(seconds=1)
        return results
    
    # Get the camera directory
    camera_dir = CAMERA_OPTIONS.get(camera, BASE_DIR)
    
    # Layout B: camera_dir/YYYY-MM-DD/camera_*.jpg
    date_dir = start_time.strftime("%Y-%m-%d")
    search_dir = os.path.join(camera_dir, date_dir)

    results = []
    if os.path.exists(search_dir):
        image_files = glob(os.path.join(search_dir, f"{camera}_*.jpg"))
        for image_file in image_files:
            ts = extract_timestamp_from_filename(os.path.basename(image_file))
            if ts and start_time <= ts <= end_time and ts.date() not in EXCLUDED_DATES:
                results.append((ts, image_file))

    # Sort by timestamp
    results.sort(key=lambda x: x[0])
    return results

def list_images_in_range(base_dir: str, start_time: datetime, end_time: datetime):
    """Robustly list images between start_time and end_time under base_dir.

    Supports two directory layouts:
    - Layout A: base_dir/YYYY/MM/DD/HH/*.jpg with ISO-like filenames
    - Layout B: base_dir/YYYY-MM-DD/{camera}_*.jpg
    Returns list of (timestamp, path).
    """
    results: list[tuple[datetime, str]] = []

    # Try Layout A by iterating per-hour directories
    current_hour = start_time.replace(minute=0, second=0, microsecond=0)
    end_bound = end_time
    while current_hour <= end_bound:
        hour_path = os.path.join(
            base_dir,
            current_hour.strftime("%Y"),
            current_hour.strftime("%m"),
            current_hour.strftime("%d"),
            current_hour.strftime("%H"),
        )
        if os.path.exists(hour_path):
            for img_path in glob(os.path.join(hour_path, "*.jpg")):
                ts = extract_timestamp_from_filename(os.path.basename(img_path))
                if ts and start_time <= ts <= end_time and ts.date() not in EXCLUDED_DATES:
                    results.append((ts, img_path))
        current_hour += timedelta(hours=1)

    # If nothing found, try Layout B by days
    if not results:
        current_day = start_time.date()
        last_day = end_time.date()
        while current_day <= last_day:
            day_path = os.path.join(base_dir, current_day.strftime("%Y-%m-%d"))
            if os.path.exists(day_path):
                for img_path in glob(os.path.join(day_path, "*.jpg")):
                    ts = extract_timestamp_from_filename(os.path.basename(img_path))
                    if ts and start_time <= ts <= end_time and ts.date() not in EXCLUDED_DATES:
                        results.append((ts, img_path))
            current_day += timedelta(days=1)

    results.sort(key=lambda x: x[0])
    return results

def load_and_process_image(image_path, timestamp=None):
    """Load and process an image file"""
    try:
        # Load the image
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Basic processing (resize if needed)
        # max_width = 1280
        # if frame.shape[1] > max_width:
        #     scale = max_width / frame.shape[1]
        #     frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)))
        
        return frame
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        # Return a black frame with error message if loading fails
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame, 
            f"Error loading image", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2
        )
        return frame
