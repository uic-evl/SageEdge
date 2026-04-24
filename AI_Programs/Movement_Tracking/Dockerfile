# JetPack 6 / L4T r36.4.x with PyTorch preinstalled
FROM dustynv/l4t-pytorch:r36.4.0

# system deps (OpenCV, GL, GStreamer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv libgl1 \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

# python basics + your libs
RUN python3 -m pip install --upgrade pip setuptools wheel
ENV OPENBLAS_CORETYPE=ARMV8
RUN pip install "numpy==1.26.4" ultralytics==8.3.0 shapely sympy recordtype pascal_voc_writer \
    matplotlib pandas tqdm psutil

WORKDIR /app
COPY . /app
CMD ["python3", "testing_ultralytics.py"]
