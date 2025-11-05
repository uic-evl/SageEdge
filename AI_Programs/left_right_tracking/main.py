#!/usr/bin/env python3
# main.py — left-right-tracker (stream-first, camera fallback)
# behavior:
#   - if STREAM is set -> use cv2.VideoCapture(STREAM)
#   - else:
#        - if CAMERA_FALLBACK=1 -> use waggle Camera()
#        - else -> exit with an error message (matches "stream-first" examples)
#
# deps: pywaggle[vision], ultralytics, opencv-python, numpy

import os
import time
import argparse
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from waggle.plugin import Plugin
from waggle.data.vision import Camera

def str2bool(x):
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

# ------------------------------- #
# args / env (stream-first)
# ------------------------------- #
p = argparse.ArgumentParser(description="left-right-tracker (stream-first, camera fallback)")
p.add_argument("--stream", default=os.getenv("STREAM", ""), help="file path or rtsp/http url; preferred source")
p.add_argument("--camera", default=os.getenv("CAMERA", ""), help="waggle camera name if fallback is enabled")
p.add_argument("--camera_fallback", type=str2bool, default=str2bool(os.getenv("CAMERA_FALLBACK", "0")),
               help="if 1 and no stream is provided, use waggle camera")
p.add_argument("--model", default=os.getenv("MODEL_SIZE", "m"), choices=list("nsmlx"),
               help="yolov8 model size")
p.add_argument("--dir_thresh", type=int, default=int(os.getenv("DIR_THRESH", "100")),
               help="pixels of x-displacement to count a direction")
p.add_argument("--live_output", type=str2bool, default=str2bool(os.getenv("LIVE_OUTPUT", "0")),
               help="show dev window if 1 (not for headless)")
p.add_argument("--save_output", type=str2bool, default=str2bool(os.getenv("SAVE_OUTPUT", "0")),
               help="save mp4 if 1")
p.add_argument("--out_path", default=os.getenv("OUT_PATH", "output.mp4"),
               help="output video path when save_output=1")
args = p.parse_args()

headless = not args.live_output

# ------------------------------- #
# pick source (prefer stream)
# ------------------------------- #
src_kind = None

if args.stream:
    cap = cv2.VideoCapture(args.stream)
    if not cap.isOpened():
        raise SystemExit(f"cannot open stream: {args.stream}")
    def get_frame():
        ok, f = cap.read()
        return f if ok else None
    def cleanup():
        cap.release()
    src_kind = f"cv2:{args.stream}"
else:
    if not args.camera_fallback:
        raise SystemExit("no STREAM provided. set STREAM (file/rtsp/http). "
                         "or set CAMERA_FALLBACK=1 to use the waggle camera.")
    cam = Camera(args.camera) if args.camera else Camera()
    def get_frame():
        return cam.get()
    def cleanup():
        pass
    src_kind = f"waggle:{args.camera or 'default'}"

first = get_frame()
if first is None:
    cleanup()
    raise SystemExit(f"no frames from source ({src_kind}).")

h, w = first.shape[:2]

# ------------------------------- #
# yolov8 model (integrated bytetrack)
# ------------------------------- #
model = YOLO(f"yolov8{args.model}.pt")

# optional writer
writer = None
if args.save_output:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, 20.0, (w, h))
    if not writer.isOpened():
        print("[warn] videowriter failed; disabling save_output")
        writer = None

# ------------------------------- #
# tracking state
# ------------------------------- #
xhist = defaultdict(lambda: deque(maxlen=24))  # track id -> recent x values
counted = set()
left_count = 0
right_count = 0

def draw_hud(frame, l, r):
    txt = f"L: {l}   R: {r}   T: {l + r}"
    cv2.rectangle(frame, (8, 8), (8 + 290, 40), (0, 0, 0), -1)
    cv2.putText(frame, txt, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ------------------------------- #
# main loop
# ------------------------------- #
with Plugin() as plugin:
    plugin.publish("movement.source.kind", src_kind)
    plugin.publish("movement.dir.threshold", args.dir_thresh)

    while True:
        frame = get_frame()
        if frame is None:
            break

        results = model.track(frame, persist=True, verbose=False)

        if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
            boxes = results[0].boxes
            ids = getattr(boxes, "id", None)
            if ids is not None and hasattr(boxes, "xyxy"):
                ids_np = ids.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                for tid, bb in zip(ids_np, xyxy):
                    x1, y1, x2, y2 = bb
                    cx = 0.5 * (x1 + x2)
                    hist = xhist[tid]
                    hist.append(float(cx))

                    if len(hist) >= 2:
                        delta = hist[-1] - hist[0]
                        if abs(delta) >= args.dir_thresh:
                            if tid not in counted:
                                if delta > 0:
                                    right_count += 1
                                else:
                                    left_count += 1
                                counted.add(tid)
                                xhist[tid] = deque([hist[-1]], maxlen=24)
                        else:
                            if tid in counted and abs(delta) < (0.3 * args.dir_thresh):
                                counted.discard(tid)

        ts = time.time()
        plugin.publish("movement.left.count", left_count, timestamp=ts)
        plugin.publish("movement.right.count", right_count, timestamp=ts)
        plugin.publish("movement.total.count", left_count + right_count, timestamp=ts)

        if not headless:
            draw_hud(frame, left_count, right_count)
            cv2.imshow("left-right-tracker", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
        if writer:
            writer.write(frame)

if writer:
    writer.release()
cleanup()
cv2.destroyAllWindows()
