# SAGE Edge: Compact VLM Benchmarking Suite

Deployment-oriented benchmarking of compact vision-language models on edge computing platforms for the NSF SAGE testbed.

## Overview

This repository contains benchmarking scripts and results for evaluating six compact vision-language models (VLMs) across three edge computing platforms. Rather than focusing on accuracy leaderboards, we measure **deployment-critical metrics**: latency, throughput, memory consumption, power usage, and execution reliability under real-world constraints.

**Key Findings:**
- **Moondream2** achieves lowest latency (895ms on Thor, 2.3s on Orin) with minimal memory footprint (12GB)
- **SmolVLM2** exhibits extreme token generation (3M tokens/run), limiting practical deployment despite competitive latency
- **InternVL2-2B** offers balanced performance across platforms
- Memory pressure becomes the primary constraint on embedded devices (Jetson AGX Orin)

## Models Evaluated

| Model | Parameters | Key Characteristics |
|-------|-----------|---------------------|
| **Moondream2** | 1.8B | Efficient vision encoder, compact language model |
| **SmolVLM2** | 2B | Dense vision encoding, high token generation |
| **LLaVA-Mini** | 9B | Smaller LLaVA variant, higher memory requirements |
| **Gemma 3n (E4B-IT)** | 8B/4B eff. | Instruction-tuned with selective parameter activation |
| **InternVL2-2B** | 2B | Balanced vision-language architecture |
| **Phi-3.5-Vision** | 4.2B | Microsoft's compact multimodal model |

## Hardware Platforms

### Dell Pro Max (GB10)
- **GPU:** NVIDIA RTX A2000 (12GB VRAM)
- **CPU:** Intel Xeon
- **RAM:** 64GB
- **Role:** Development/near-edge environment

### NVIDIA Jetson Thor
- **Memory:** 32GB unified memory
- **Role:** Higher-performance edge inference
- **Status:** JetPack 7 (experimental)

### NVIDIA Jetson AGX Orin
- **Memory:** 32GB unified memory
- **Power Limit:** 50W
- **Role:** Power-constrained embedded deployment

## Evaluation Methodology

### Tasks
Five vision-language tasks reflecting realistic edge use cases, each with 128-token generation limit:

1. **caption_brief** - Concise 1-2 sentence factual captioning
2. **objects_and_counts** - Object detection with approximate counts
3. **spatial_relationships** - Relative spatial grounding (left/right, near/far)
4. **scene_context** - Scene type classification (urban, indoor, landscape)
5. **attributes** - Visual attributes (color, lighting, materials)

### Dataset
- **500 images** from COCO validation split
- **2,500 total inferences** (5 tasks × 500 images)
- Both **fp16** and **bf16** precision tested where supported

### Metrics
- **Latency:** Mean and tail (P90, P99) per-image inference time
- **Throughput:** Images processed per second
- **Memory:** Peak RAM usage
- **Power:** GPU-specific (NVML on Dell, VDD_GPU rail on Jetson)
- **Token metrics:** Total tokens, tokens/sec, energy per token

## Repository Structure

```
Benchmarking/
├── Dell/
│   ├── scripts/          # Benchmarking scripts for Dell Pro Max
│   └── outputs/          # Results and logs
├── Thor/
│   ├── scripts/          # Benchmarking scripts for Jetson Thor
│   └── outputs/          # Results and logs
|── Orin/
|   ├── scripts/          # Benchmarking scripts for Jetson AGX Orin
|   └── outputs/          # Results and logs
|── datat/testsets/       # manifest files for COCO images
```

## Setup & Requirements

### Software Environment
- **OS:** Ubuntu (20.04+ recommended)
- **CUDA:** 12.x
- **Python:** 3.10+
- **PyTorch:** 2.5.1
- **Transformers:** 4.48.1


### Dataset Preparation

Download COCO validation images:
```bash
# Download and extract COCO val2017
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

## Usage

Navigate to the appropriate platform directory and run benchmarking scripts:

```bash
cd Benchmarking/Dell/scripts

# Example: Run Moondream2 benchmark
python benchmark_moondream2.py \
    --images_dir /path/to/val2017 \
    --output_dir ../outputs \
    --num_images 500 \
    --dtype bf16
```

Each script includes:
- Standardized 5-task evaluation suite
- Power monitoring (platform-specific)
- GPU utilization tracking
- JSONL logging with resume capability
- Comprehensive warmup procedures

## Key Findings

| Model | Platform | Latency (ms) | RAM (GB) | Power (W) | Total Tokens |
|-------|----------|--------------|----------|-----------|--------------|
| Moondream2 | Thor | 895 | 24.9 | 27.2 | 140,867 |
| Moondream2 | Orin | 2,276 | 12.2 | 14.2 | 140,863 |
| SmolVLM2 | Thor | 2,079 | 26.6 | 29.5 | 2,956,750 |
| InternVL2-2B | Thor | 2,658 | 24.4 | 31.6 | 185,571 |
| InternVL2-2B | Orin | 5,071 | 18.2 | 21.9 | 192,957 |


## Deployment Insights

### Best for Resource-Constrained Edge
**Moondream2** - Lowest latency, smallest memory footprint, stable tail behavior

### Best for Balanced Performance
**InternVL2-2B** - Good latency with moderate memory, works well on Thor and Dell

### Avoid for Production Edge
**SmolVLM2** - Despite fast inference, 30× token generation creates downstream bottlenecks

## Implementation Notes

- **Attention Backend:** Eager mode for consistency across devices
- **Precision Control:** Explicit dtype management (fp16/bf16)
- **Power Telemetry:** Device-specific (NVML vs. tegrastats)
- **No Ollama:** Direct Transformers implementation for precise control

## Limitations & Future Work

Current limitations:
- No task correctness/accuracy evaluation at VQA benchmark level
- Power measurement inconsistencies on JetPack 7 (Thor)
- GPU utilization monitoring challenges on Thor: jtop is incompatible with JetPack 7, and tegrastats GPU metrics extraction requires further investigation

Planned extensions:
1. Quantitative accuracy evaluation against VQA benchmarks

## Acknowledgments

This work was supported by NSF grant OAC-2331263 (SAGE Testbed), with additional support from:
- Electronic Visualization Laboratory at UIC
- Argonne National Laboratory

**Project Status:** Active Development  
**Last Updated:** February 2026
