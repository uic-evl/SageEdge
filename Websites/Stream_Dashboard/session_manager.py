"""
NIU Image Stream Server - Session Manager Module
Handles user sessions, queues, and thread management
"""

import threading
import time
import uuid
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from queue import Queue

# Import configuration and helpers
from config import (
    START_DATE, END_DATE, SESSION_TIMEOUT, DEMO_MODE, CAMERA_OPTIONS, BASE_DIR,
    PRELOAD_MINUTES, BUFFER_SIZE, DEFAULT_CAMERA
)
from image_processor import (
    list_images_in_range, get_images_for_timerange
)

# Configure logging
logger = logging.getLogger(__name__)

# Session storage
user_sessions = {}  # Store user playback controllers
user_frame_queues = {}  # Store frame queues for each user
user_frame_caches = {}  # Cache frames for each user session
user_last_frames = {}  # Store last frame sent to each user
user_last_activity = defaultdict(lambda: time.time())  # Track user activity
session_locks = defaultdict(threading.RLock)  # Thread locks for each session

# Cleanup thread control
stop_cleanup = False
cleanup_threads = {}

class StreamController:
    """Full-featured controller for video playback state and image listing"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_datetime = START_DATE
        self.frames_per_second = 1.0  # Default: slower by default to prevent fast playback
        self.is_playing = True
        self.show_timestamp = True
        self.current_images = []  # Loaded images with timestamps
        self.current_image_index = 0
        self.loop_mode = "full"  # "full", "day", "hour", "none"
        self.loop_start_date = START_DATE
        self.loop_end_date = END_DATE
        self.last_load_time = None
        self.frame_skip_counter = 0
        self.current_camera = DEFAULT_CAMERA
        self.base_dir = CAMERA_OPTIONS.get(self.current_camera, BASE_DIR)
        self.last_activity = time.time()
        self.stop_thread = False

    def update_activity(self):
        self.last_activity = time.time()

    def set_camera(self, camera_name: str) -> bool:
        # Map UI aliases to demo camera keys when in demo mode
        if DEMO_MODE and camera_name in {"top", "bottom"}:
            camera_name = f"demo_{camera_name}"

        if camera_name in CAMERA_OPTIONS:
            self.current_camera = camera_name
            self.base_dir = CAMERA_OPTIONS[camera_name]
            self.current_images = []
            self.current_image_index = 0

            # Clear the frame queue for this user
            user_queue = user_frame_queues.get(self.session_id)
            if user_queue:
                while not user_queue.empty():
                    try:
                        user_queue.get_nowait()
                    except Exception:
                        break

            # Reload images from new camera location
            self.load_images_around_time(self.current_datetime)
            self.update_activity()
            logger.info(f"Session {self.session_id}: Switched to {camera_name} camera")
            return True
        return False

    def set_frames_per_second(self, fps: float):
        self.frames_per_second = max(0.1, min(8.0, float(fps)))
        self.update_activity()
        logger.info(f"Session {self.session_id}: Frame rate set to {self.frames_per_second} fps")

    def set_datetime(self, target_datetime: datetime):
        try:
            # Ensure the base directory is correct for the current camera
            self.base_dir = CAMERA_OPTIONS.get(self.current_camera, self.base_dir)

            # Clamp to valid range
            if target_datetime < START_DATE:
                target_datetime = START_DATE
            elif target_datetime > END_DATE:
                target_datetime = END_DATE

            self.current_datetime = target_datetime
            self.load_images_around_time(target_datetime)

            closest_timestamp = target_datetime
            time_difference = 0

            if self.current_images:
                closest_index = 0
                min_diff = float('inf')
                for i, (img_time, _) in enumerate(self.current_images):
                    diff = abs((img_time - target_datetime).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_index = i
                        closest_timestamp = img_time
                        time_difference = (img_time - target_datetime).total_seconds()
                self.current_image_index = closest_index
            else:
                self.current_image_index = 0

            # Clear the frame queue for immediate response
            user_queue = user_frame_queues.get(self.session_id)
            if user_queue:
                while not user_queue.empty():
                    try:
                        user_queue.get_nowait()
                    except Exception:
                        break

            self.update_activity()
            logger.info(
                f"Session {self.session_id}: Jumped to {target_datetime}, closest image: {closest_timestamp}"
            )

            return {
                "requested_time": target_datetime,
                "actual_time": closest_timestamp,
                "time_difference": time_difference,
                "images_found": len(self.current_images)
            }
        except ValueError as e:
            logger.error(f"Session {self.session_id}: Error setting datetime: {e}")
            return {
                "requested_time": target_datetime,
                "actual_time": self.current_datetime,
                "time_difference": 0,
                "error": str(e)
            }

    def load_images_around_time(self, center_time: datetime, minutes_range: int = 30):
        # Ensure correct base directory for current camera
        current_base_dir = CAMERA_OPTIONS.get(self.current_camera)
        if not current_base_dir:
            logger.error(f"Session {self.session_id}: Invalid camera '{self.current_camera}' selected.")
            self.current_images = []
            return

        start_time = center_time - timedelta(minutes=minutes_range // 2)
        end_time = center_time + timedelta(minutes=minutes_range)

        # Use shared listing function for consistency
        if DEMO_MODE:
            self.current_images = get_images_for_timerange(start_time, end_time, camera=self.current_camera)
        else:
            self.current_images = list_images_in_range(current_base_dir, start_time, end_time)

        # Broaden search window intelligently if nothing found
        if not self.current_images and not DEMO_MODE:
            try:
                # 1) Try the entire day
                day_start = center_time.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
                day_imgs = list_images_in_range(current_base_dir, day_start, day_end)
                if day_imgs:
                    self.current_images = day_imgs
                    self.current_image_index = 0
                    self.current_datetime = day_imgs[0][0]
                else:
                    # 2) Walk outward day-by-day to find nearest day with images (up to 30 days)
                    found = False
                    for offset in range(1, 31):
                        for candidate in (center_time + timedelta(days=offset), center_time - timedelta(days=offset)):
                            cand_start = candidate.replace(hour=0, minute=0, second=0, microsecond=0)
                            cand_end = cand_start + timedelta(days=1) - timedelta(seconds=1)
                            cand_imgs = list_images_in_range(current_base_dir, cand_start, cand_end)
                            if cand_imgs:
                                self.current_images = cand_imgs
                                self.current_image_index = 0
                                self.current_datetime = cand_imgs[0][0]
                                logger.info(
                                    f"Session {self.session_id}: No images at {center_time.date()}, switched to nearest date with data: {cand_start.date()}"
                                )
                                found = True
                                break
                        if found:
                            break
            except Exception as e:
                logger.warning(f"Session {self.session_id}: Fallback search for images failed: {e}")

        # Find the closest image to our target time
        if self.current_images:
            closest_index = 0
            min_diff = abs((self.current_images[0][0] - center_time).total_seconds())
            for i, (img_time, _) in enumerate(self.current_images):
                diff = abs((img_time - center_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_index = i
            self.current_image_index = closest_index

        logger.info(f"Session {self.session_id}: Loaded {len(self.current_images)} images around {center_time}")

    def get_next_image_path(self):
        # Check if we need to load more images
        if not self.current_images or self.current_image_index >= len(self.current_images) - 10:
            # Load more images ahead
            if self.current_images and self.current_image_index < len(self.current_images):
                current_time = self.current_images[self.current_image_index][0]
            else:
                current_time = self.current_datetime
            self.load_images_around_time(current_time + timedelta(minutes=PRELOAD_MINUTES // 2))

        if self.current_image_index >= len(self.current_images):
            # Handle loop modes
            if self.loop_mode == "full":
                self.current_datetime = self.loop_start_date
                self.load_images_around_time(self.current_datetime)
                self.current_image_index = 0
                logger.info("Looping back to start of dataset")
            elif self.loop_mode == "day":
                day_start = self.current_datetime.replace(hour=0, minute=0, second=0)
                self.current_datetime = day_start
                self.load_images_around_time(self.current_datetime)
                self.current_image_index = 0
                logger.info("Looping back to start of day")
            elif self.loop_mode == "hour":
                hour_start = self.current_datetime.replace(minute=0, second=0)
                self.current_datetime = hour_start
                self.load_images_around_time(self.current_datetime)
                self.current_image_index = 0
                logger.info("Looping back to start of hour")
            elif self.loop_mode == "none":
                self.is_playing = False
                logger.info("Reached end of dataset, stopping playback")
                return None, None

        if self.current_images and self.current_image_index < len(self.current_images):
            img_timestamp, img_path = self.current_images[self.current_image_index]
            self.current_image_index += 1
            self.current_datetime = img_timestamp
            return img_timestamp, img_path

        return None, None

    def set_loop_mode(self, mode: str, start_date: datetime | None = None, end_date: datetime | None = None):
        self.loop_mode = mode
        if start_date:
            self.loop_start_date = start_date
        if end_date:
            self.loop_end_date = end_date
        logger.info(f"Loop mode set to: {mode}")

    def set_custom_loop_range(self, start_date: datetime, end_date: datetime):
        self.loop_start_date = start_date
        self.loop_end_date = end_date
        self.loop_mode = "full"
        logger.info(f"Custom loop range set: {start_date} to {end_date}")


def get_or_create_session_id():
    """Get existing session ID or create a new one"""
    from flask import session

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        logger.info(f"Created new session: {session['session_id']}")

    # Update last activity time
    session_id = session['session_id']
    user_last_activity[session_id] = time.time()

    return session_id


def get_user_controller(session_id=None):
    """Get or create the playback controller for a user session"""
    from flask import session

    if not session_id:
        session_id = session.get('session_id')

    if not session_id:
        return None

    with session_locks[session_id]:
        if session_id not in user_sessions:
            # Create new controller for this session
            user_sessions[session_id] = StreamController(session_id)
            user_frame_queues[session_id] = Queue(maxsize=BUFFER_SIZE)
            user_frame_caches[session_id] = {}
            user_last_frames[session_id] = None

        # Update last activity
        user_last_activity[session_id] = time.time()
        return user_sessions[session_id]


def cleanup_inactive_sessions():
    """Remove inactive sessions to free up resources"""
    global user_sessions, user_frame_queues, user_frame_caches

    while not stop_cleanup:
        try:
            current_time = time.time()
            sessions_to_remove = []

            # Find inactive sessions
            for session_id, last_active in list(user_last_activity.items()):
                if current_time - last_active > SESSION_TIMEOUT:
                    sessions_to_remove.append(session_id)

            # Remove inactive sessions
            for session_id in sessions_to_remove:
                with session_locks[session_id]:
                    if session_id in user_sessions:
                        logger.info(f"Removing inactive session: {session_id}")
                        user_sessions.pop(session_id, None)
                        user_frame_queues.pop(session_id, None)
                        user_frame_caches.pop(session_id, None)
                        user_last_frames.pop(session_id, None)
                        user_last_activity.pop(session_id, None)

            # Sleep for a while before next check
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
            time.sleep(60)  # Continue despite errors


def start_cleanup_thread():
    """Start the thread that cleans up inactive sessions"""
    global stop_cleanup

    stop_cleanup = False
    cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
    cleanup_thread.start()
    logger.info("Session cleanup thread started")
    return cleanup_thread
