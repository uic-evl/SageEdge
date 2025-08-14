"""
NIU Image Stream Server - Session Manager Module
Handles user sessions, queues, and thread management
"""

import threading
import time
import uuid
import logging
from datetime import datetime
from collections import defaultdict
from queue import Queue

# Import configuration
from config import START_DATE, SESSION_TIMEOUT

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

class VideoController:
    """Controller for video playback state"""
    def __init__(self, start_datetime=START_DATE):
        self.current_datetime = start_datetime
        self.frames_per_second = 4.0
        self.is_playing = True
        self.loop_mode = "full"  # Options: "full", "hour", "none"
        self.current_camera = "top"  # Default camera
        self.last_update = time.time()
        self.show_timestamp = True
        self.show_analytics = False

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
    """Get the video controller for a user session"""
    from flask import session
    
    if not session_id:
        session_id = session.get('session_id')
    
    if not session_id:
        return None
    
    with session_locks[session_id]:
        if session_id not in user_sessions:
            # Create new controller for this session
            user_sessions[session_id] = VideoController()
            user_frame_queues[session_id] = Queue(maxsize=150)  # Buffer size
        
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
