#!/bin/bash


# Sets up the environment variables
export OPENBLAS_CORETYPE=ARMV8
echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
echo "export PYTHONPATH=\"${PYTHONPATH}:/usr/lib/python3.10/dist-packages\"" >> ~/.bashrc  # Updated to python3.10
export PYTHONPATH="${PYTHONPATH}:/usr/lib/python3.10/dist-packages"

echo
echo "This setup script will install all the necessary components for the Movement_Tracking project on the NVIDIA Jetson."
echo "This will take some time. Please ensure you have a stable internet connection."
echo

# Update system packages first
sudo apt -y update && sudo apt -y upgrade

# Install base dependencies
sudo apt -y install \
    git cmake libpython3-dev python3-numpy \
    libgeos-dev sqlite3 python3-pip \
    libjpeg-dev libpng-dev libtiff-dev \
    libopenblas-base libopenmpi-dev libomp-dev \
    libhdf5-dev pkg-config \
    libgtk2.0-dev libgl1-mesa-glx

# Clean up any existing virtualenv
rm -rf tracking || true

# Create and activate virtual environment
sudo apt install python3-pip python3-dev
sudo -H pip3 install virtualenv
virtualenv ./tracking
source tracking/bin/activate
pip install --upgrade pip setuptools wheel

# Install numpy first with specific version
pip install "numpy==1.26.4"

# Install OpenCV with GUI support
pip install "opencv-python==4.9.0.80"

# Install PyTorch for Jetson
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl -O torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Install torchvision
git clone --depth 1 --branch v0.20.0 https://github.com/pytorch/vision.git
cd vision
pip install .
cd ..
rm -rf vision

# Install deep-person-reid
git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
pip install -e . --no-build-isolation
cd ..

# Install YOLOv8 and tracking dependencies
pip install \
    ultralytics \
    deep-sort-realtime \
    shapely \
    sympy \
    recordtype \
    pascal_voc_writer \
    matplotlib \
    pandas \
    tqdm \
    psutil

# Fix any potential numpy conflicts
pip install --force-reinstall "numpy==1.26.4"

# Cleanup
deactivate
echo
echo "Setup completed successfully!"
echo "To activate the virtual environment: source tracking/bin/activate"
