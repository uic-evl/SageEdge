# thor / local test base image with cuda + pytorch
FROM nvcr.io/nvidia/pytorch:25.08-py3

# avoid interactive tz dialogs, etc.
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# system dependencies for opencv, video, and monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    procps \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install python deps
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# copy application code
COPY main.py privacy_filters.py ./

# unbuffered python output (good for logs on SAGE)
ENV PYTHONUNBUFFERED=1

# default command:
# - pixelation filter
# - no display (container is headless)
# - metrics logged to metrics_log.csv
CMD ["python", "main.py", "--filter", "pixelate", "--no-display", "--csv", "metrics_log.csv"]
