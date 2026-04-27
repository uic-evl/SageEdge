# CPU-only Sage edge plugin for face blurring.
# Works on amd64 and arm64 (Thor, Jetson, Pi, x86 dev nodes).
# YuNet runs through OpenCV's DNN module — no GPU, no PyTorch needed.

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Runtime libs OpenCV needs (FFMPEG for RTSP/mp4, plus the usual image libs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libopenblas0 \
    libpng16-16 \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

# Pre-download YuNet ONNX so the container runs offline on edge nodes.
RUN mkdir -p /app/models && \
    python -c "import urllib.request; urllib.request.urlretrieve(\
'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', \
'/app/models/face_detection_yunet_2023mar.onnx')"

COPY . .

ENTRYPOINT ["python", "main.py"]