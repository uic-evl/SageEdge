"""Unified status payload builder.

Provides a single function to assemble status / health information so that
server, legacy /api/status, and /api/health (deprecated) all stay consistent.
"""
from datetime import datetime
from typing import Any, Dict, Optional
import os

from config import (
    DEMO_MODE, START_DATE, END_DATE, BUFFER_SIZE, TARGET_FPS,
    CAMERA_OPTIONS, DB_PATH, BASE_DIR, SESSION_TIMEOUT
)
from session_manager import user_sessions, user_frame_queues, user_frame_caches


def build_status_payload(session_id: Optional[str], include_buffer: bool = True) -> Dict[str, Any]:
    controller = user_sessions.get(session_id) if session_id else None
    user_queue = user_frame_queues.get(session_id) if session_id else None
    user_cache = user_frame_caches.get(session_id, {}) if session_id else {}

    current_image_info = None
    if controller and controller.current_images and 0 <= controller.current_image_index < len(controller.current_images):
        ts, path = controller.current_images[controller.current_image_index]
        current_image_info = {
            "timestamp": ts.isoformat(),
            "path": path,
            "index": controller.current_image_index,
            "total_loaded": len(controller.current_images),
        }

    playback = None
    if controller:
        playback = {
            "is_playing": controller.is_playing,
            "frames_per_second": controller.frames_per_second,
            "playback_speed": controller.frames_per_second / TARGET_FPS,
            "show_timestamp": controller.show_timestamp,
            "loop_mode": controller.loop_mode,
        }

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "server_time": datetime.now().isoformat(),
        "status": "running" if controller else "no_session",
        "session_id": session_id,
        "demo_mode": DEMO_MODE,
        "base_directory": controller.base_dir if controller else BASE_DIR,
        "available_cameras": list(CAMERA_OPTIONS.keys()),
        "current_camera": controller.current_camera if controller else None,
        "current_datetime": controller.current_datetime.isoformat() if controller else None,
        "current_image": current_image_info,
        "data_range": {"start_date": START_DATE.isoformat(), "end_date": END_DATE.isoformat()},
    # excluded_dates removed
        "playback": playback,
        "session_info": {"session_timeout": SESSION_TIMEOUT},
        "analysis_database": os.path.exists(DB_PATH),
        "base_dir_accessible": os.path.exists(BASE_DIR),
        "active_sessions": len(user_sessions),
    }

    if include_buffer and controller:
        payload["buffer_info"] = {
            "buffer_size": user_queue.qsize() if user_queue else 0,
            "max_buffer_size": BUFFER_SIZE,
            "cached_frames": len(user_cache),
        }
        payload.update({
            "is_playing": playback["is_playing"],
            "frames_per_second": playback["frames_per_second"],
            "loop_mode": playback["loop_mode"],
            "show_timestamp": playback["show_timestamp"],
            "buffer_size": payload["buffer_info"]["buffer_size"],
        })

    return payload
