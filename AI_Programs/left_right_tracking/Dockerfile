# thor (amd64, gpu) dockerfile for pedestrian-direction-tracker
# tested on environment using nvcr.io/nvidia/pytorch:25.08-py3
# adds pywaggle so it can run as a sage plugin

FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# FORCE PyTorch to use NVIDIA's custom UCX library instead of the broken system defaults
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# system packages 
# (Removed libopenmpi-dev to stop it from installing conflicting libucc/libucx system libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake pkg-config build-essential gfortran \
    libgeos-dev sqlite3 \
    libjpeg-dev libpng-dev libtiff-dev \
    libopenblas0-pthread liblapack-dev libhdf5-dev libomp-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir onnxslim onnxruntime
RUN python -m pip install --no-cache-dir --no-deps ultralytics

# ------------------------------------
# Add YOLOv8 weights for offline use
# ------------------------------------
RUN mkdir -p /app/models

ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt /app/models/yolov8n.pt
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt /app/models/yolov8s.pt
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt /app/models/yolov8m.pt
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt /app/models/yolov8l.pt
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt /app/models/yolov8x.pt

# yolov8-face weights for the Offline_Tracking.py privacy blur pass
ADD https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov8n-person.pt /app/models/yolov8n-face.pt

COPY . .

# default entrypoint: main.py (left/right counter).
ENTRYPOINT ["python", "main.py"]
