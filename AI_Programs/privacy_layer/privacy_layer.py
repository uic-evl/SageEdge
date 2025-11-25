# privacy_strip.py
# utilities for applying a short "privacy strip" blur to the upper region of a person bbox

import cv2
import numpy as np

def _odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def apply_privacy_strip(frame, boxes, head_frac=0.33, strength=25):
    """
    blur the upper band (approx head/face region) of each detected person bbox.

    frame:   np.array image (H,W,3)
    boxes:   iterable of (x1, y1, x2, y2) person boxes (ints or floats)
    head_frac: fraction of box height to blur from the top (0–1)
    strength: controls blur kernel size (higher = stronger blur)
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]

    for (x1, y1, x2, y2) in boxes:
        x1i = int(_clip(round(x1), 0, w - 1))
        y1i = int(_clip(round(y1), 0, h - 1))
        x2i = int(_clip(round(x2), 0, w - 1))
        y2i = int(_clip(round(y2), 0, h - 1))

        if x2i <= x1i or y2i <= y1i:
            continue

        box_h = y2i - y1i
        strip_h = max(4, int(box_h * head_frac))
        y2_strip = _clip(y1i + strip_h, 0, h - 1)

        roi = frame[y1i:y2_strip, x1i:x2i]
        if roi.size == 0:
            continue

        kx = _odd(max(3, int((x2i - x1i) / 12) + strength // 10))
        ky = _odd(max(3, int(strip_h / 12) + strength // 10))
        blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
        frame[y1i:y2_strip, x1i:x2i] = blurred

    return frame
