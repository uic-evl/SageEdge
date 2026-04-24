import cv2
import time
import csv
import psutil
import argparse
from datetime import datetime
import subprocess

from ultralytics import YOLO

from privacy_filters import (
    apply_black_rectangle,
    apply_gaussian_blur,
    apply_box_blur,
    apply_median_blur,
    apply_pixelation
)

# ---------------------------------------------------------
# filter name -> function mapping
# ---------------------------------------------------------
FILTERS = {
    "gaussian": apply_gaussian_blur,
    "box": apply_box_blur,
    "median": apply_median_blur,
    "pixelate": apply_pixelation,
    "black": apply_black_rectangle
}


# ---------------------------------------------------------
# gpu utilization helpers
# for thor / jetson devices we first try tegrastats
# if that is not available, we try nvidia-smi
# otherwise we return None
# ---------------------------------------------------------
def get_gpu_utilization_from_tegrastats():
    """
    parse a single tegrastats sample and return gpu utilization percentage,
    based on the gr3d_freq field (e.g., 'GR3D_FREQ 37%@110').
    returns float or None.
    """
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "200", "--count", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.7,
        )
        output = result.stdout.strip()
        if not output:
            return None

        line = output.splitlines()[0]
        tokens = line.replace(",", " ").split()
        for i, token in enumerate(tokens):
            if "GR3D_FREQ" in token.upper():
                if i + 1 < len(tokens):
                    val = tokens[i + 1]
                else:
                    continue
                digits = "".join(ch for ch in val if ch.isdigit())
                if digits:
                    return float(digits)
        return None
    except Exception:
        return None


def get_gpu_utilization_from_nvidia_smi():
    """
    use nvidia-smi to query gpu utilization.
    returns float or None.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.5,
        )
        output = result.stdout.strip()
        if output:
            line = output.splitlines()[0]
            return float(line)
        return None
    except Exception:
        return None


def get_gpu_utilization():
    """
    wrapper that tries tegrastats first (thor / jetson),
    then falls back to nvidia-smi if available.
    returns float percent or None.
    """
    gpu = get_gpu_utilization_from_tegrastats()
    if gpu is not None:
        return gpu

    gpu = get_gpu_utilization_from_nvidia_smi()
    if gpu is not None:
        return gpu

    return None


# ---------------------------------------------------------
# argument parsing
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="privacy filter performance lab on sage nodes")

    parser.add_argument(
        "--filter",
        type=str,
        default="pixelate",
        choices=FILTERS.keys(),
        help="which privacy filter to apply"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="camera source (0 for webcam or path to video file)"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="metrics_log.csv",
        help="path to output csv file for metrics"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="run without opening an opencv window (recommended on thor)"
    )

    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="path to save processed video (e.g., output.mp4)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # select the filter function
    filter_fn = FILTERS[args.filter]

    # load yolo model
    model = YOLO("yolov8n.pt")

    # open video source
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        print("failed to open video source.")
        return

    # set up video writer if saving output
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # get original frame size
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # attempt to match input fps
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if input_fps == 0 or input_fps is None:
            input_fps = 30

        video_writer = cv2.VideoWriter(
            args.save_video,
            fourcc,
            input_fps,
            (width, height)
        )

    print("running with filter:", args.filter)
    print("logging metrics to:", args.csv)

    # create/open csv and write header
    csv_file = open(args.csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp",
        "filter",
        "frame_index",
        "fps",
        "cpu_percent",
        "memory_percent",
        "gpu_percent"
    ])

    prev_time = time.time()
    frame_index = 0

    # initialize cpu tracking
    psutil.cpu_percent(interval=None)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            current_time = time.time()
            dt = current_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = current_time

            # run tracking
            results = model.track(frame, persist=True, verbose=False, classes=[0])

            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    if cls_id != 0:
                        continue  # skip non-person detections

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # clamp coordinates
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # apply selected filter
                    frame = filter_fn(frame, x1, y1, x2, y2)


            # write frame to output video if enabled
            if video_writer is not None:
                video_writer.write(frame)

            # collect system metrics
            cpu_percent = psutil.cpu_percent(interval=0)
            memory_percent = psutil.virtual_memory().percent
            gpu_percent = get_gpu_utilization()

            # log csv row
            timestamp = datetime.now(timezone.utc).isoformat()
            csv_writer.writerow([
                timestamp,
                args.filter,
                frame_index,
                round(fps, 2),
                round(cpu_percent, 2),
                round(memory_percent, 2),
                round(gpu_percent, 2) if gpu_percent is not None else ""
            ])

            # status update every 30 frames
            if frame_index % 30 == 0:
                print(
                    f"frame {frame_index:6d} | "
                    f"fps: {fps:5.1f} | "
                    f"cpu: {cpu_percent:5.1f}% | "
                    f"mem: {memory_percent:5.1f}% | "
                    f"gpu: {gpu_percent if gpu_percent is not None else 'n/a'}"
                )

            if not args.no_display:
                cv2.imshow("privacy filter output", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # still allow quit on q even without visible window
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print("finished. metrics saved to:", args.csv)


if __name__ == "__main__":
    main()
