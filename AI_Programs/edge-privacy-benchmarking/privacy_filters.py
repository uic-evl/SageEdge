import cv2

# Helper: safe bounding boxes
def clip_roi(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    
    return x1, y1, x2, y2


# -----------------------------
# Gaussian Blur
# -----------------------------
def apply_gaussian_blur(frame, x1, y1, x2, y2, ksize=25):
    roi = clip_roi(frame, x1, y1, x2, y2)
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi
        
    sub = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(sub, (ksize, ksize), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame


# -----------------------------
# Box (Average) Blur
# -----------------------------
def apply_box_blur(frame, x1, y1, x2, y2, ksize=25):
    roi = clip_roi(frame, x1, y1, x2, y2)
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi

    sub = frame[y1:y2, x1:x2]
    blurred = cv2.blur(sub, (ksize, ksize))
    frame[y1:y2, x1:x2] = blurred
    return frame


# -----------------------------
# Median Blur
# -----------------------------
def apply_median_blur(frame, x1, y1, x2, y2, ksize=25):
    roi = clip_roi(frame, x1, y1, x2, y2)
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi
    
    if ksize % 2 == 0:
        ksize += 1

    sub = frame[y1:y2, x1:x2]
    blurred = cv2.medianBlur(sub, ksize)
    frame[y1:y2, x1:x2] = blurred
    return frame


# -----------------------------
# Pixelation
# -----------------------------
def apply_pixelation(frame, x1, y1, x2, y2, pixel_size=12):
    roi = clip_roi(frame, x1, y1, x2, y2)
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi

    sub = frame[y1:y2, x1:x2]

    h, w = sub.shape[:2]

    # Downsample
    temp = cv2.resize(sub, 
                      (max(1, w // pixel_size), max(1, h // pixel_size)),
                      interpolation=cv2.INTER_LINEAR)
    
    # Upsample
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    frame[y1:y2, x1:x2] = pixelated
    return frame

# -----------------------------
# black rectangle (censorship bar)
# -----------------------------
def apply_black_rectangle(frame, x1, y1, x2, y2, color=(0, 0, 0)):
    """
    fills the roi with a solid color (default black).
    very cheap, very strong anonymization.
    """
    roi = clip_roi(frame, x1, y1, x2, y2)
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi

    frame[y1:y2, x1:x2] = color
    return frame

# -----------------------------
# universal filter dispatcher
# -----------------------------
def apply_filter(frame, x1, y1, x2, y2, mode="none"):
    """
    mode options:
      - "none"
      - "gaussian"
      - "box"
      - "median"
      - "pixel"
      - "black"
    """
    if mode == "gaussian":
        return apply_gaussian_blur(frame, x1, y1, x2, y2)
    elif mode == "box":
        return apply_box_blur(frame, x1, y1, x2, y2)
    elif mode == "median":
        return apply_median_blur(frame, x1, y1, x2, y2)
    elif mode == "pixel":
        return apply_pixelation(frame, x1, y1, x2, y2)
    elif mode == "black":
        return apply_black_rectangle(frame, x1, y1, x2, y2)
    else:               
        return frame    
