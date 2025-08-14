"""
NIU Image Stream Server - Utilities Module
Contains helper functions used throughout the application
"""

import cv2
import time
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter to prevent excessive API calls"""
    def __init__(self, max_calls, time_period):
        self.max_calls = max_calls
        self.time_period = time_period  # in seconds
        self.calls = []
    
    def check_rate_limit(self):
        """Check if rate limit is exceeded"""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_period]
        
        # Check if we're under the limit
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False

def encode_frame_to_jpeg(frame, quality=90):
    """Encode a frame to JPEG format for streaming"""
    try:
        # Check if frame is valid
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
            
        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Error encoding frame to JPEG: {e}")
        # Return a minimal valid JPEG on error
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', empty_frame)
        return buffer.tobytes()

def format_datetime(dt):
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def parse_iso_datetime(iso_string):
    """Parse ISO format datetime string"""
    try:
        return datetime.fromisoformat(iso_string)
    except ValueError:
        # Handle ISO format with 'Z' for UTC
        if iso_string.endswith('Z'):
            return datetime.fromisoformat(iso_string[:-1])
        raise

def get_date_bounds(date_str):
    """Get start and end datetime for a specific date"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = datetime.combine(date.date(), datetime.min.time())
        end = datetime.combine(date.date(), datetime.max.time())
        return start, end
    except Exception as e:
        logger.error(f"Error parsing date bounds: {e}")
        # Return current day as fallback
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())
        return start, end

def get_hour_bounds(date_str, hour):
    """Get start and end datetime for a specific hour on a date"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = datetime.combine(date.date(), datetime.min.time().replace(hour=hour))
        end = start + timedelta(hours=1) - timedelta(seconds=1)
        return start, end
    except Exception as e:
        logger.error(f"Error parsing hour bounds: {e}")
        # Return current hour as fallback
        now = datetime.now()
        start = datetime.combine(now.date(), datetime.min.time().replace(hour=now.hour))
        end = start + timedelta(hours=1) - timedelta(seconds=1)
        return start, end

def get_month_bounds(year, month):
    """Get start and end datetime for a specific month"""
    try:
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(seconds=1)
        return start, end
    except Exception as e:
        logger.error(f"Error parsing month bounds: {e}")
        # Return current month as fallback
        now = datetime.now()
        start = datetime(now.year, now.month, 1)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(now.year, now.month + 1, 1) - timedelta(seconds=1)
        return start, end

def add_timestamp_to_frame(frame, timestamp, show=True, position="bottom"):
    """Add timestamp overlay to a frame"""
    if not show:
        return frame
    
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get text size
    font_scale = width / 1000.0
    thickness = max(1, int(font_scale * 2))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Position the text
    if position == "top":
        x = (width - text_size[0]) // 2
        y = text_size[1] + 10
    else:  # bottom
        x = (width - text_size[0]) // 2
        y = height - 10
    
    # Draw background rectangle
    cv2.rectangle(
        frame, 
        (x - 5, y - text_size[1] - 5), 
        (x + text_size[0] + 5, y + 5), 
        (0, 0, 0, 128), 
        -1
    )
    
    # Draw text
    cv2.putText(
        frame, 
        text, 
        (x, y), 
        font, 
        font_scale, 
        (255, 255, 255), 
        thickness
    )
    
    return frame

def create_info_frame(message, width=800, height=600):
    """Create an information frame with message text"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a gradient background
    for y in range(height):
        color_value = int(150 * (1 - y / height)) + 50
        cv2.line(frame, (0, y), (width, y), (color_value, color_value, color_value), 1)
    
    # Split message into lines
    lines = message.split("\n")
    
    # Draw text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_height = 30
    
    y_position = height // 3
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x_position = (width - text_size[0]) // 2
        
        cv2.putText(
            frame, 
            line, 
            (x_position, y_position), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness
        )
        
        y_position += line_height
    
    return frame
