"""
NIU Image Stream Server - Main Server Module
Handles HTTP requests and video streaming
"""

from flask import Flask, Response, request, jsonify, session
import cv2
import time
import numpy as np
from datetime import datetime
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid

# Import our modular components
from config import (
    DEMO_MODE, BASE_DIR, START_DATE, END_DATE, TARGET_FPS, BUFFER_SIZE,
    PRELOAD_MINUTES, FRAME_INTERPOLATION, DB_PATH, SECRET_KEY, MAX_WORKERS,
    FRAMES_PER_SECOND_OPTIONS, CAMERA_OPTIONS, DEFAULT_CAMERA, SESSION_TIMEOUT
)
from session_manager import (
    get_or_create_session_id, get_user_controller, start_cleanup_thread,
    user_sessions, user_frame_queues, user_frame_caches, user_last_frames
)
from image_processor import (
    generate_demo_frame
)
from utils import add_timestamp_to_frame, RateLimiter
from status_builder import build_status_payload
from api_routes import register_api_routes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='/static')
app.secret_key = SECRET_KEY  # For session management
app.config['SESSION_COOKIE_HTTPONLY'] = True
"""High-priority fixes applied:
1. SESSION_COOKIE_SECURE now dynamically set (secure when not in demo mode).
2. /api/session now returns playback and camera fields expected by my-stream.html.
3. Producer thread no longer blocks on thread pool futures (synchronous processing to avoid unnecessary overhead).
"""

# Use secure cookies outside demo mode (can be overridden by env FORCE_INSECURE_COOKIES)
app.config['SESSION_COOKIE_SECURE'] = (not DEMO_MODE) and not os.environ.get('FORCE_INSECURE_COOKIES')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS or 4)
session_create_limiter = RateLimiter(10, 60)  # 10 session creations per minute
control_limiter = RateLimiter(60, 60)  # 60 control actions per minute

## Controller class is provided by session_manager.StreamController


def preprocess_frame(frame, timestamp=None, add_overlay=True, session_controller=None):
    """Resize, add overlay, and encode frame to JPEG"""
    if frame is None:
        return None
    try:
        frame = cv2.resize(frame, (1280, 720))
        if add_overlay and timestamp and session_controller and session_controller.show_timestamp:
            frame = add_timestamp_to_frame(frame, timestamp, show=True, position="top-left")
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 90,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
            cv2.IMWRITE_JPEG_LUMA_QUALITY, 90,
            cv2.IMWRITE_JPEG_CHROMA_QUALITY, 90,
        ]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return None

def ensure_session():
    """Ensure a session exists and return session ID; rate limited for creation events."""
    session_id = get_or_create_session_id()
    if session_id not in user_sessions:
        if not session_create_limiter.allow():
            raise RuntimeError("Session creation rate limit exceeded")
        get_user_controller(session_id)
        producer_thread = threading.Thread(
            target=producer_thread_for_user,
            args=(session_id,),
            daemon=True,
        )
        producer_thread.start()
        logger.info(f"Created new user session and started producer: {session_id}")
    return session_id

## session cleanup handled by session_manager.start_cleanup_thread()

def load_and_process_image(img_data, session_id):
    """Load and process a single image for a specific user session"""
    try:
        img_timestamp, img_path = img_data
        controller = user_sessions.get(session_id)
        if not controller:
            return None
        
        # Check cache first (session-specific)
        user_cache = user_frame_caches.get(session_id, {})
        cache_key = f"{img_path}_{controller.show_timestamp}"
        if cache_key in user_cache:
            return user_cache[cache_key]
        
        # Generate demo frame if in demo mode
        if DEMO_MODE:
            frame = generate_demo_frame(img_timestamp)
        else:
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"Session {session_id}: Could not load image: {img_path}")
                return None
            
        # Use the actual timestamp from filename
        processed = preprocess_frame(frame, img_timestamp, controller.show_timestamp, controller)
        
        # Cache the processed frame (limit cache size per user)
        if len(user_cache) > 150:  # Increased cache size
            # Remove oldest entries
            oldest_keys = list(user_cache.keys())[:30]
            for key in oldest_keys:
                del user_cache[key]
                
        user_cache[cache_key] = processed
        user_frame_caches[session_id] = user_cache
        return processed
    except Exception as e:
        if DEMO_MODE:
            logger.warning(f"Session {session_id}: Demo mode error for {img_data}: {str(e)}")
        else:
            logger.error(f"Session {session_id}: Error loading image {img_data}: {str(e)}")
        return None

def producer_thread_for_user(session_id):
    """Continuously load & enqueue frames for a specific user (simplified synchronous version).

    Rationale: Previous implementation submitted work to a ThreadPoolExecutor then immediately
    called future.result(), nullifying concurrency and adding overhead/latency. We now process
    frames inline; if real parallel decoding is needed later, batching + as_completed can be added.
    """
    logger.info(f"Producer thread started for session: {session_id}")
    while session_id in user_sessions:
        try:
            controller = user_sessions.get(session_id)
            if not controller or controller.stop_thread:
                break
            if not controller.is_playing:
                time.sleep(0.1)
                continue
            user_queue = user_frame_queues.get(session_id)
            if not user_queue:
                break

            while user_queue.qsize() < BUFFER_SIZE - 20 and not controller.stop_thread:
                img_data = controller.get_next_image_path()
                if img_data[0] is not None and img_data[1] is not None:
                    try:
                        processed = load_and_process_image(img_data, session_id)
                        if processed:
                            try:
                                user_queue.put(processed, timeout=0.1)
                                user_last_frames[session_id] = processed
                                logger.debug(f"Session {session_id}: Queued: {img_data[1]}")
                            except Exception:
                                pass
                        else:
                            last_frame = user_last_frames.get(session_id)
                            if last_frame and FRAME_INTERPOLATION:
                                try:
                                    user_queue.put(last_frame, timeout=0.1)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.warning(f"Session {session_id}: Error processing {img_data}: {e}")
                        last_frame = user_last_frames.get(session_id)
                        if last_frame and FRAME_INTERPOLATION:
                            try:
                                user_queue.put(last_frame, timeout=0.1)
                            except Exception:
                                pass
                else:
                    last_frame = user_last_frames.get(session_id)
                    if last_frame and FRAME_INTERPOLATION:
                        try:
                            user_queue.put(last_frame, timeout=0.1)
                        except Exception:
                            pass
                    break

            buffer_level = user_queue.qsize()
            if buffer_level > BUFFER_SIZE * 0.8:
                time.sleep(0.1)
            else:
                target_interval = 1.0 / max(0.1, controller.frames_per_second)
                time.sleep(max(0.05, target_interval * 0.5))
        except Exception as e:
            logger.error(f"Session {session_id}: Producer thread error: {e}")
            time.sleep(0.5)
    logger.info(f"Producer thread stopped for session: {session_id}")

def generate_frames(session_id):
    """Yields frames for MJPEG stream with smooth playback for a specific user session"""
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    
    # Create a better fallback frame
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(black, "Loading...", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    controller = user_sessions.get(session_id)
    if controller and controller.show_timestamp:
        black = add_timestamp_to_frame(black, controller.current_datetime, show=True, position="top-left")
    _, fallback_buffer = cv2.imencode('.jpg', black, [cv2.IMWRITE_JPEG_QUALITY, 90])
    fallback_frame = fallback_buffer.tobytes()
    
    frame_count = 0
    last_yield_time = time.time()
    
    while session_id in user_sessions:
        try:
            controller = user_sessions.get(session_id)
            user_queue = user_frame_queues.get(session_id)
            
            if not controller or not user_queue:
                break
                
            current_time = time.time()
            
            # Dynamic frame rate based on frames per second setting
            target_interval = 1.0 / controller.frames_per_second
            time_since_last = current_time - last_yield_time
            
            # More precise timing for slower frame rates
            if time_since_last < target_interval:
                sleep_time = target_interval - time_since_last
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            if not user_queue.empty() and controller.is_playing:
                frame = user_queue.get()
                user_last_frames[session_id] = frame  # Update last valid frame
                yield boundary + frame + b'\r\n'
                frame_count += 1
                logger.debug(f"Session {session_id}: Streamed frame {frame_count}, buffer: {user_queue.qsize()}")
            else:
                # Use last valid frame instead of black screen
                last_frame = user_last_frames.get(session_id)
                if last_frame:
                    yield boundary + last_frame + b'\r\n'
                    logger.debug(f"Session {session_id}: Repeated last frame, buffer: {user_queue.qsize()}")
                else:
                    yield boundary + fallback_frame + b'\r\n'
                    logger.debug(f"Session {session_id}: Used fallback frame")
                    
            last_yield_time = time.time()
            
        except Exception as e:
            logger.error(f"Session {session_id}: Error in generate_frames: {e}")
            # On error, try to use fallback frame
            try:
                yield boundary + fallback_frame + b'\r\n'
            except:
                pass
            time.sleep(0.05)  # Shorter error recovery sleep

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/my-stream')
def my_stream():
    """Show user their personal stream information"""
    return app.send_static_file('my-stream.html')

@app.route('/styles.css')
def styles():
    return app.send_static_file('styles.css')

@app.route('/script.js')
def script():
    return app.send_static_file('script.js')

@app.route('/video_feed')
def video_feed():
    """Main video feed endpoint with improved error handling"""
    try:
        session_id = ensure_session()
        logger.info(f"Video feed accessed by session {session_id}")
        return Response(generate_frames(session_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error accessing video feed: {e}", exc_info=True)
        return "Error accessing video feed. Please try refreshing the page.", 500

@app.route('/video_feed/<session_id>')
def video_feed_for_session(session_id):
    """Unique video stream URL for each session with improved error handling"""
    try:
        if session_id not in user_sessions:
            logger.warning(f"Session not found: {session_id}")
            return "Session not found", 404
        
        logger.info(f"Direct video feed accessed with session ID: {session_id}")
        
        # Create controller if it doesn't exist
        if session_id not in user_sessions:
            get_user_controller(session_id)
            
        return Response(
            generate_frames(session_id), 
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
    except Exception as e:
        logger.error(f"Error in direct video feed: {e}", exc_info=True)
        return "Error accessing video feed. Please try refreshing the page.", 500

@app.route('/api/session')
def api_session_info():
    """Return unified session + status info (includes feed URLs)."""
    try:
        sid = ensure_session()
        payload = build_status_payload(sid)
        host = request.host
        payload.update({
            "video_feed_url": f"http://{host}/video_feed",
            "direct_video_feed_url": f"http://{host}/video_feed/{sid}",
        })
        return jsonify(payload)
    except RuntimeError as rl:
        return jsonify({"error": str(rl)}), 429
    except Exception as e:
        logger.error(f"Session info error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/control', methods=['POST'])
def control():
    """Unified control endpoint with rate limiting."""
    if not control_limiter.allow():
        return jsonify({"status": "error", "message": "Rate limit exceeded"}), 429
    try:
        session_id = ensure_session()
        controller = user_sessions.get(session_id)
        if not controller:
            return jsonify({"status": "error", "message": "No session found"}), 404
        if not request.is_json:
            return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400
        data = request.json or {}
        action = data.get('action')
        if not action:
            return jsonify({"status": "error", "message": "Missing 'action' parameter"}), 400
        if action == 'jump_to_datetime':
            dt_str = data.get('datetime')
            if not dt_str:
                return jsonify({"status": "error", "message": "Missing 'datetime' parameter"}), 400
            try:
                target_dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                result = controller.set_datetime(target_dt)
                resp = {
                    "status": "success",
                    "action": action,
                    "requested_time": target_dt.isoformat(),
                    "actual_time": result.get("actual_time").isoformat() if result.get("actual_time") else target_dt.isoformat(),
                    "time_difference": result.get("time_difference", 0),
                    "images_found": result.get("images_found", 0)
                }
                if "error" in result:
                    resp["warning"] = result["error"]
                return jsonify(resp)
            except ValueError:
                return jsonify({"status": "error", "message": "Invalid datetime format. Expected YYYY-MM-DD HH:MM:SS"}), 400
        elif action == 'toggle_playback':
            controller.is_playing = not controller.is_playing
            return jsonify({"status": "success", "action": action, "is_playing": controller.is_playing})
        elif action == 'set_speed':
            fps = float(data.get('frames_per_second', TARGET_FPS)) if 'frames_per_second' in data else TARGET_FPS * float(data.get('speed', 1.0))
            controller.set_frames_per_second(fps)
            return jsonify({"status": "success", "action": action, "frames_per_second": controller.frames_per_second})
        elif action == 'toggle_timestamp':
            controller.show_timestamp = not controller.show_timestamp
            return jsonify({"status": "success", "action": action, "show_timestamp": controller.show_timestamp})
        elif action == 'set_loop_mode':
            loop_mode = data.get('loop_mode', 'full')
            if loop_mode not in ['full', 'day', 'hour', 'none']:
                return jsonify({"status": "error", "message": "Invalid loop_mode"}), 400
            controller.set_loop_mode(loop_mode)
            return jsonify({"status": "success", "action": action, "loop_mode": loop_mode})
        elif action == 'set_camera':
            camera_name = data.get('camera', DEFAULT_CAMERA)
            if controller.set_camera(camera_name):
                return jsonify({"status": "success", "action": action, "camera": camera_name})
            return jsonify({"status": "error", "message": f"Invalid camera: {camera_name}"}), 400
        else:
            return jsonify({"status": "error", "message": "Unknown action"}), 400
    except RuntimeError as rl:
        return jsonify({"status": "error", "message": str(rl)}), 429
    except Exception as e:
        logger.error(f"Control endpoint error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal server error", "error_details": str(e)}), 500

@app.route('/status')
def status():
    """Unified status endpoint using status_builder."""
    try:
        session_id = ensure_session()
        payload = build_status_payload(session_id)
        code = 200 if payload.get("status") == "running" else 404
        return jsonify(payload), code
    except RuntimeError as rl:
        return jsonify({"status": "error", "message": str(rl)}), 429
    except Exception as e:
        logger.error(f"Status endpoint error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Error retrieving status", "error_details": str(e)}), 500


@app.route('/api/health')
def health_check():
    """Deprecated health endpoint returning unified status subset."""
    try:
        session_id = session.get('session_id')
        payload = build_status_payload(session_id, include_buffer=True)
        return jsonify({"status": "healthy", "session": payload})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
if __name__ == '__main__':
    logger.info("Server starting...")
    
    # Start session cleanup thread
    start_cleanup_thread()

    # Register additional API routes (status/camera controls) if desired
    try:
        register_api_routes(app)
        logger.info("api_routes registered")
    except Exception as e:
        logger.warning(f"api_routes not registered: {e}")
    
    # Improve error logging to catch 500 errors
    import sys
    from logging import FileHandler
    file_handler = FileHandler('server_errors.log')
    file_handler.setLevel(logging.ERROR)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.ERROR)
    
    try:
        # Running with improved settings for cross-device access
        app.run(
            host='0.0.0.0',        # Listen on all network interfaces
            port=8080,             # Default port
            threaded=True,         # Enable multi-threading for better performance
            debug=False,           # Disable debug mode in production
            use_reloader=False     # Disable auto-reloading
        )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        print(f"Critical error: {e}", file=sys.stderr)
    finally:
        executor.shutdown(wait=True)
