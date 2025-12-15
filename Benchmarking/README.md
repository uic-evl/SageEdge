## System Assumptions

This project was developed and tested on a SAGE Thor node with:

- Ubuntu 22.04 (Jetson-based)
- NVIDIA CUDA preinstalled (CUDA 13)
- NVIDIA drivers compatible with JetPack
- Python 3.10+
- Internet access for model downloads

## Python Environment Setup

We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```
## Core dependencies
These are the things **without which the project does not run**.
- torch: model execution on CPU/GPU
- transformers: Hugging Face vision-language models (Gemma, Moondream)
- accelerate: device mapping on constrained hardware
- pillow: image loading and preprocessing
- psutil: system-level performance metrics
- ultralytics: YOLO object detection

## Thor Setup Notes (Reproducible)

### Goal

Run Gemma-3n (HF), moondream2, YOLO benchmarks on Jetson Thor reliably.

### Repo context

* Gemma model card: `google/gemma-3n-E4B-it`
* Related baseline repo: `SageImageCaption` (for reference)

---

## 1) Create and activate environment

```bash
cd ~/Desktop/SageEdge/Benchmarking
python3 -m venv thor-env
source thor-env/bin/activate
pip install --upgrade pip
```

---

## 2) Disable TorchVision inside Transformers (Thor-safe)

On Thor, TorchVision can cause import/runtime issues. We disable it for Transformers:

**one-time**

```bash
echo 'export TRANSFORMERS_NO_TORCHVISION=1' >> ~/.bashrc
source ~/.bashrc
```

**verify**

```bash
echo $TRANSFORMERS_NO_TORCHVISION
# expected: 1
```

**run scripts like**

```bash
TRANSFORMERS_NO_TORCHVISION=1 python3 scripts/test_gemma.py
```

---

## 3) Install PyTorch (Thor-compatible) + dependencies

### What went wrong initially (important)

Installing PyTorch from the standard CUDA wheel index (`download.pytorch.org/whl/cu126`) produced:

* `NVIDIA Thor with CUDA capability sm_110 is not compatible...`
* `no kernel image is available for execution on the device`

This indicates the wheel was not built with **sm_110** support (Thor GPU arch).

### Fix: install Thor-compatible wheels

**remove incorrect installs**

```bash
pip uninstall -y torch torchvision torchaudio
```

**install Thor-compatible PyTorch**

```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
```

> This index contains wheels compiled for Thor (sm_110) with CUDA 13.


##  Install model + benchmarking dependencies

```bash
pip install "transformers>=4.53.0" timm pillow sentencepiece accelerate protobuf psutil
pip install huggingface_hub
pip install ultralytics
```

---

## 6) Verify GPU + Torch works

```bash
python3 - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
EOF
```

Expected:

* `cuda available: True`
* device shows `NVIDIA Thor`
* no warnings about `sm_110 not compatible`

---

## 7) Sanity test: Gemma captioning

```bash
export TRANSFORMERS_NO_TORCHVISION=1
python3 scripts/test_gemma.py
```

During environment setup on Jetson Thor, standard PyTorch CUDA wheels were incompatible with the Thor GPU architecture (sm_110), causing kernel execution errors. We resolved this by installing Thor-specific PyTorch wheels (CUDA 13 build) and disabling TorchVision usage within Transformers via `TRANSFORMERS_NO_TORCHVISION=1` to avoid unsupported torchvision operators.”
