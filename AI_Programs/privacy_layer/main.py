#!/usr/bin/env python3
# face-blur edge app for Sage / Waggle
# - source: STREAM (file / rtsp / http) or named/default waggle Camera
# - detects faces with YuNet (optionally on a downscaled copy)
# - masks each face with a selectable method (box / gaussian / median / pixelate / solid)
# - output modes: none | file | upload | both

import os
import cv2
import argparse
import datetime

from waggle.plugin import Plugin
from waggle.data.vision import Camera

from privacy_layer import (
    load_face_detector,
    detect_faces,
    blur_faces,
    BLUR_METHODS,
)


OUTPUT_MODES = ("none", "file", "upload", "both")


def str2bool(x):
    return str(x).lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    p = argparse.ArgumentParser(description="face-blur edge app")

    # source
    p.add_argument("--stream", default=os.getenv("STREAM", ""),
                   help="file path or rtsp/http url")
    p.add_argument("--camera", default=os.getenv("CAMERA", ""),
                   help="waggle camera name (e.g., left/right)")
    p.add_argument("--snapshot_only", type=str2bool,
                   default=str2bool(os.getenv("SNAPSHOT_ONLY", "0")),
                   help="process one frame and exit")

    # detection
    p.add_argument("--conf", type=float,
                   default=float(os.getenv("CONF", "0.6")),
                   help="face detection confidence threshold")
    p.add_argument("--detect_width", type=int,
                   default=int(os.getenv("DETECT_WIDTH", "640")),
                   help="downscale frame to this width before detection "
                        "(0 = detect at full resolution)")

    # masking
    p.add_argument("--method",
                   default=os.getenv("METHOD", "box"),
                   choices=sorted(BLUR_METHODS.keys()),
                   help="face masking method")
    p.add_argument("--blur_strength", type=int,
                   default=int(os.getenv("BLUR_STRENGTH", "25")))
    p.add_argument("--pad_frac", type=float,
                   default=float(os.getenv("PAD_FRAC", "0.15")),
                   help="padding fraction around each face box")

    # output
    p.add_argument("--output",
                   default=os.getenv("OUTPUT", "none"),
                   choices=OUTPUT_MODES,
                   help="what to do with blurred frames: "
                        "none=discard, file=write mp4, "
                        "upload=periodic snapshot to Beehive, both=file+upload")
    p.add_argument("--out_path", default=os.getenv("OUT_PATH", "blurred.mp4"),
                   help="output mp4 filename when --output includes 'file'")
    p.add_argument("--upload_every", type=int,
                   default=int(os.getenv("UPLOAD_EVERY", "150")),
                   help="upload one blurred snapshot every N frames "
                        "(only used when --output includes 'upload')")
    p.add_argument("--publish_count", type=str2bool,
                   default=str2bool(os.getenv("PUBLISH_COUNT", "0")),
                   help="if true, publish per-frame face count to Beehive")
    return p.parse_args()


def make_writer(frame, out_path, fps=30.0):
    """Create a VideoWriter sized to the first frame; return (writer, full_path)."""
    H, W = frame.shape[:2]
    out_dir = os.path.join(
        "output", datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, os.path.basename(out_path))
    writer = cv2.VideoWriter(
        full_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    print(f"writing video to {full_path}")
    return writer, full_path


def process_frame(frame, detector, args):
    """Detect and blur every face in `frame` in-place. Returns face count."""
    boxes = detect_faces(detector, frame,
                         detect_width=args.detect_width or None)
    blur_faces(frame, boxes,
               method=args.method,
               strength=args.blur_strength,
               pad_frac=args.pad_frac)
    return len(boxes)


def maybe_upload(plugin, frame, frame_idx, sample_ts):
    """Write blurred frame to /tmp and upload it to Beehive."""
    tmp = f"/tmp/blurred_{frame_idx}.jpg"
    cv2.imwrite(tmp, frame)
    plugin.upload_file(tmp, timestamp=sample_ts)


def main():
    args = parse_args()
    print(f"face-blur: method={args.method} conf={args.conf} "
          f"strength={args.blur_strength} pad={args.pad_frac} "
          f"detect_width={args.detect_width or 'full'} "
          f"output={args.output}")

    do_file = args.output in ("file", "both")
    do_upload = args.output in ("upload", "both")

    detector = load_face_detector(conf_threshold=args.conf)
    source = args.stream or args.camera or None
    print(f"source: {source if source else 'default-camera'}")

    # ---------- snapshot mode ----------
    if args.snapshot_only:
        with Plugin() as plugin:
            sample = Camera(source).snapshot()
            frame = sample.data
            n = process_frame(frame, detector, args)
            if args.publish_count:
                plugin.publish("privacy.faces.count", n,
                               timestamp=sample.timestamp)
            if do_file:
                _, path = make_writer(frame, args.out_path)
                cv2.imwrite(path.replace(".mp4", ".jpg"), frame)
            if do_upload:
                maybe_upload(plugin, frame, 0, sample.timestamp)
        print("done. processed 1 frame")
        return

    # ---------- streaming mode ----------
    writer = None
    frame_idx = 0
    try:
        with Plugin() as plugin, Camera(source) as camera:
            if args.publish_count:
                plugin.publish("privacy.method", args.method)

            for sample in camera.stream():
                frame = sample.data
                if frame is None:
                    continue
                frame_idx += 1

                n = process_frame(frame, detector, args)
                if args.publish_count:
                    plugin.publish("privacy.faces.count", n,
                                   timestamp=sample.timestamp)

                if do_file:
                    if writer is None:
                        writer, _ = make_writer(frame, args.out_path)
                    writer.write(frame)

                if do_upload and frame_idx % args.upload_every == 0:
                    maybe_upload(plugin, frame, frame_idx, sample.timestamp)

                if frame_idx % 50 == 0:
                    print(f"processed frame {frame_idx}")
    finally:
        if writer is not None:
            writer.release()
        print(f"done. processed {frame_idx} frames")


if __name__ == "__main__":
    main()