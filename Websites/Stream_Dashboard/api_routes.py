"""
NIU Image Stream Server - API Routes Module
Handles API endpoints for the application
"""

import os
import json
import logging
from flask import jsonify, request, session
import sqlite3
from datetime import datetime, timedelta

# Import configuration
from config import BASE_DIR, CAMERA_OPTIONS, DEFAULT_CAMERA

# Configure logging
logger = logging.getLogger(__name__)

def register_api_routes(app):
    """Register API routes with the Flask application"""
    
    @app.route('/api/status')
    def api_status():
        """Get server status information"""
        try:
            from session_manager import get_user_controller, user_sessions, user_frame_queues
            
            # Check if analysis database exists
            from config import DB_PATH
            analysis_available = os.path.exists(DB_PATH)
            
            # Check if base directory is accessible
            base_dir_accessible = os.path.exists(BASE_DIR)
            
            # Get session info if available
            session_id = session.get('session_id')
            controller = user_sessions.get(session_id) if session_id else None
            user_queue = user_frame_queues.get(session_id) if session_id else None
            
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "base_directory": BASE_DIR,
                "base_dir_accessible": base_dir_accessible,
                "analysis_database": analysis_available,
                "current_time": controller.current_datetime.isoformat() if controller else None,
                "frames_per_second": controller.frames_per_second if controller else None,
                "is_playing": controller.is_playing if controller else None,
                "buffer_size": user_queue.qsize() if user_queue else 0,
                "active_sessions": len(user_sessions)
            })
        except Exception as e:
            logger.error(f"API status error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/cameras')
    def api_cameras():
        """Get available cameras"""
        try:
            cameras = {}
            for name, path in CAMERA_OPTIONS.items():
                cameras[name] = {
                    "name": name,
                    "path": path,
                    "available": os.path.exists(path)
                }
            
            return jsonify({
                "cameras": cameras,
                "default": DEFAULT_CAMERA
            })
        except Exception as e:
            logger.error(f"API cameras error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    # --- Analysis Endpoints ---
    def _get_db_conn():
        from config import DB_PATH, LOCAL_DB_PATH
        path = DB_PATH if os.path.exists(DB_PATH) else LOCAL_DB_PATH
        if not os.path.exists(path):
            return None
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    @app.route('/api/analysis/dashboard')
    def api_analysis_dashboard():
        """Return high-level analysis aggregates for the dashboard."""
        try:
            conn = _get_db_conn()
            if not conn:
                return jsonify({"error": "analysis database not available"}), 200

            cur = conn.cursor()
            # Example schema expectations:
            # photos(filename TEXT, datetime TEXT, weather TEXT, people TEXT, visibility TEXT, time TEXT,
            #        weather_confidence REAL, people_confidence REAL, dominant_color TEXT)

            cur.execute("SELECT COUNT(*) AS c FROM photos")
            total_analyzed = cur.fetchone()[0]

            def top_counts(column, limit=5):
                cur.execute(f"SELECT {column} AS category, COUNT(*) AS count FROM photos GROUP BY {column} ORDER BY count DESC LIMIT ?", (limit,))
                return [dict(r) for r in cur.fetchall()]

            weather_stats = top_counts('weather')
            people_stats = top_counts('people')
            time_stats = top_counts('time')

            return jsonify({
                "total_analyzed": total_analyzed,
                "weather_stats": weather_stats,
                "people_stats": people_stats,
                "time_stats": time_stats
            })
        except Exception as e:
            logger.error(f"API analysis dashboard error: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            try:
                conn and conn.close()
            except Exception:
                pass

    @app.route('/api/analysis/search')
    def api_analysis_search():
        """Search photos by optional filters: weather, people, visibility, time, limit."""
        try:
            conn = _get_db_conn()
            if not conn:
                return jsonify({"error": "analysis database not available", "count": 0, "photos": []}), 200

            params = []
            where = []
            for key in ("weather", "people", "visibility", "time"):
                val = request.args.get(key)
                if val:
                    where.append(f"{key} = ?")
                    params.append(val)
            limit = int(request.args.get('limit', '50'))
            limit = max(1, min(limit, 200))

            sql = "SELECT filename, datetime, weather, weather_confidence, people, people_confidence, visibility, time, dominant_color FROM photos"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY datetime LIMIT ?"
            params.append(limit)

            cur = conn.cursor()
            cur.execute(sql, params)
            rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]

            return jsonify({
                "count": len(rows),
                "photos": rows
            })
        except Exception as e:
            logger.error(f"API analysis search error: {e}")
            return jsonify({"error": str(e), "count": 0, "photos": []}), 500
        finally:
            try:
                conn and conn.close()
            except Exception:
                pass

    @app.route('/api/set_camera', methods=['POST'])
    def api_set_camera():
        """Set the current camera"""
        try:
            from session_manager import get_user_controller, get_or_create_session_id
            
            data = request.get_json()
            camera = data.get('camera', DEFAULT_CAMERA)
            
            if camera not in CAMERA_OPTIONS:
                return jsonify({"status": "error", "message": "Invalid camera selection"}), 400
            
            # Ensure session
            _sid = get_or_create_session_id()
            controller = get_user_controller(_sid)
            if controller:
                ok = False
                try:
                    ok = controller.set_camera(camera)
                except Exception:
                    ok = False
                if ok:
                    return jsonify({"status": "success", "camera": camera})
                return jsonify({"status": "error", "message": "Failed to switch camera"}), 500
            else:
                return jsonify({"status": "error", "message": "No active session"}), 400
        except Exception as e:
            logger.error(f"API set_camera error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/set_playback_state', methods=['POST'])
    def api_set_playback_state():
        """Set playback state (play/pause)"""
        try:
            from session_manager import get_user_controller, get_or_create_session_id
            
            data = request.get_json()
            playing = data.get('playing')
            
            if playing is None:
                return jsonify({"status": "error", "message": "Missing 'playing' parameter"}), 400
            
            _sid = get_or_create_session_id()
            controller = get_user_controller(_sid)
            if controller:
                controller.is_playing = bool(playing)
                return jsonify({"status": "success", "is_playing": controller.is_playing})
            else:
                return jsonify({"status": "error", "message": "No active session"}), 400
        except Exception as e:
            logger.error(f"API set_playback_state error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/set_playback_speed', methods=['POST'])
    def api_set_playback_speed():
        """Set playback speed"""
        try:
            from session_manager import get_user_controller, get_or_create_session_id
            
            data = request.get_json()
            speed = data.get('speed')
            
            if speed is None:
                return jsonify({"status": "error", "message": "Missing 'speed' parameter"}), 400
            
            try:
                speed = float(speed)
            except ValueError:
                return jsonify({"status": "error", "message": "Speed must be a number"}), 400
            
            _sid = get_or_create_session_id()
            controller = get_user_controller(_sid)
            if controller:
                controller.frames_per_second = speed
                return jsonify({"status": "success", "frames_per_second": controller.frames_per_second})
            else:
                return jsonify({"status": "error", "message": "No active session"}), 400
        except Exception as e:
            logger.error(f"API set_playback_speed error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/set_loop_mode', methods=['POST'])
    def api_set_loop_mode():
        """Set loop mode (full, hour, none)"""
        try:
            from session_manager import get_user_controller, get_or_create_session_id
            
            data = request.get_json()
            mode = data.get('mode', 'full')
            
            if mode not in ['full', 'hour', 'none']:
                return jsonify({"status": "error", "message": "Invalid loop mode"}), 400
            
            _sid = get_or_create_session_id()
            controller = get_user_controller(_sid)
            if controller:
                controller.loop_mode = mode
                return jsonify({"status": "success", "loop_mode": controller.loop_mode})
            else:
                return jsonify({"status": "error", "message": "No active session"}), 400
        except Exception as e:
            logger.error(f"API set_loop_mode error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/set_datetime', methods=['POST'])
    def api_set_datetime():
        """Set current playback datetime"""
        try:
            from session_manager import get_user_controller, get_or_create_session_id
            
            data = request.get_json()
            datetime_str = data.get('datetime')
            
            if not datetime_str:
                return jsonify({"status": "error", "message": "Missing 'datetime' parameter"}), 400
            
            try:
                from utils import parse_iso_datetime
                new_datetime = parse_iso_datetime(datetime_str)
            except ValueError:
                return jsonify({"status": "error", "message": "Invalid datetime format"}), 400
            
            _sid = get_or_create_session_id()
            controller = get_user_controller(_sid)
            if controller:
                controller.current_datetime = new_datetime
                return jsonify({
                    "status": "success", 
                    "datetime": controller.current_datetime.isoformat()
                })
            else:
                return jsonify({"status": "error", "message": "No active session"}), 400
        except Exception as e:
            logger.error(f"API set_datetime error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/toggle_timestamp', methods=['POST'])
    def api_toggle_timestamp():
        """Toggle timestamp display"""
        try:
            from session_manager import get_user_controller, get_or_create_session_id
            
            data = request.get_json()
            show = data.get('show')
            
            if show is None:
                return jsonify({"status": "error", "message": "Missing 'show' parameter"}), 400
            
            _sid = get_or_create_session_id()
            controller = get_user_controller(_sid)
            if controller:
                controller.show_timestamp = bool(show)
                return jsonify({
                    "status": "success", 
                    "show_timestamp": controller.show_timestamp
                })
            else:
                return jsonify({"status": "error", "message": "No active session"}), 400
        except Exception as e:
            logger.error(f"API toggle_timestamp error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return app  # Return the app with routes registered
