# privacy_layer.py
# face detection (YuNet) + selectable face-masking methods

import os
import cv2
import numpy as np
import urllib.request

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = os.environ.get(
    "YUNET_MODEL_PATH",
    "/app/models/face_detection_yunet_2023mar.onnx",
)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


# ---------- masking methods ----------
# each takes (roi, strength) and returns a same-shape ndarray

def _gaussian(roi, strength):
    k = _odd(max(3, int(min(roi.shape[:2]) / 6) + strength // 5))
    return cv2.GaussianBlur(roi, (k, k), 0)


def _box(roi, strength):
    k = _odd(max(3, int(min(roi.shape[:2]) / 6) + strength // 5))
    return cv2.blur(roi, (k, k))


def _median(roi, strength):
    k = _odd(max(3, min(255, strength | 1)))
    return cv2.medianBlur(roi, k)


def _pixelate(roi, strength):
    h, w = roi.shape[:2]
    blocks = max(3, int(20 - strength / 4))
    short = min(h, w)
    block = max(2, short // blocks)
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _solid(roi, strength):
    return np.zeros_like(roi)


BLUR_METHODS = {
    "box":      _box,       # default: cheap, visually close to gaussian
    "gaussian": _gaussian,  # prettier, slower (biggest per-frame cost)
    "median":   _median,    # posterized, edge-preserving
    "pixelate": _pixelate,  # mosaic / classic censor
    "solid":    _solid,     # full block-out
}


# ---------- detector ----------

def load_face_detector(model_path=YUNET_PATH, input_size=(320, 320),
                       conf_threshold=0.6, nms_threshold=0.3, top_k=5000):
    """Load YuNet. Downloads the ONNX model on first use if missing."""
    if not os.path.isfile(model_path):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        print(f"downloading YuNet model to {model_path}...")
        urllib.request.urlretrieve(YUNET_URL, model_path)
    return cv2.FaceDetectorYN.create(
        model_path, "", input_size, conf_threshold, nms_threshold, top_k,
    )


def detect_faces(detector, frame, detect_width=None):
    """Run face detection. If detect_width is set and the frame is wider,
    detect on a downscaled copy for speed and scale boxes back to original
    coordinates. Returns list of (x, y, w, h) ints in original-frame space."""
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]

    if detect_width and w > detect_width:
        scale = detect_width / w
        small = cv2.resize(frame, (detect_width, int(h * scale)))
        sh, sw = small.shape[:2]
        detector.setInputSize((sw, sh))
        _, faces = detector.detect(small)
        if faces is None:
            return []
        return [(int(f[0] / scale), int(f[1] / scale),
                 int(f[2] / scale), int(f[3] / scale)) for f in faces]

    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [tuple(map(int, f[:4])) for f in faces]


# ---------- apply ----------

def blur_faces(frame, boxes, method="box", strength=25, pad_frac=0.15):
    """Mask every face box in-place on frame using the named method."""
    if frame is None or frame.size == 0 or not boxes:
        return frame

    fn = BLUR_METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"unknown method '{method}'. options: {sorted(BLUR_METHODS)}"
        )

    h, w = frame.shape[:2]
    for (x, y, bw, bh) in boxes:
        px = int(bw * pad_frac)
        py = int(bh * pad_frac)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w, x + bw + px)
        y2 = min(h, y + bh + py)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        frame[y1:y2, x1:x2] = fn(roi, strength)

    return frame