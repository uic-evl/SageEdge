# SAGE Edge: Compact VLM Benchmarking Suite

Deployment-oriented benchmarking of compact vision-language models (VLMs) on edge computing platforms for the NSF SAGE testbed.

## Overview

This repository contains benchmarking scripts and results used to evaluate **six compact vision-language models (VLMs)** across **three representative edge computing platforms** used in the SAGE testbed.

Rather than focusing on traditional accuracy leaderboards, this project evaluates **deployment-critical metrics** including:

* Latency (including tail latency)
* Throughput
* Memory footprint
* Platform-specific power signals
* Execution reliability under constrained hardware conditions

The goal is to better understand **model-device trade-offs for edge deployment**, where memory, power, and latency constraints often determine feasibility.

The evaluation uses a **standardized five-task workload across 500 COCO validation images**, enabling consistent comparison across model and hardware configurations.

## Models Evaluated

Six compact VLMs were selected to balance **model size, accessibility, and architectural diversity**.

| Model | Parameters | Notes |
|-------|-----------|-------|
| **Moondream2** | 1.8B | Compact vision encoder + lightweight language model |
| **SmolVLM2** | 2B | Dense visual tokenization, high token output |
| **LLaVA-Mini** | 9B | Smaller LLaVA architecture |
| **Gemma 3n (E4B-IT)** | ~8B / 4B effective | Selective parameter activation |
| **InternVL2-2B** | 2B | Balanced vision-language architecture |
| **Phi-3.5-Vision** | 4.2B | Microsoft multimodal model |

These models represent a range of compact multimodal architectures commonly considered for edge deployment.

## Hardware Platforms

The experiments were conducted across three representative SAGE environments.

### Dell Pro Max (GB10)

Development / near-edge workstation used for comparison experiments.

* Workstation-class GPU
* High-memory host system
* Used to compare edge results with a development environment

### NVIDIA Jetson Thor

Next-generation embedded AI platform designed for robotics and edge AI.

* Unified memory architecture
* Used for higher-performance edge inference
* JetPack 7 experimental environment

### NVIDIA Jetson AGX Orin

Embedded platform representing **power-constrained edge deployments**.

* 32 GB unified memory
* Configured with a 50W power limit
* Common platform for edge robotics and sensing systems

## Evaluation Methodology

### Dataset

Experiments use a subset of the **COCO (Common Objects in Context) validation dataset**.

* **500 images**
* Reused across all experiments
* Enables consistent comparison across models and hardware

### Task Suite

Each image is evaluated with a standardized **five-task vision-language workload** designed to reflect typical edge inference scenarios.

1. **caption_brief** - Short factual caption describing the scene
2. **objects_and_counts** - Identification of visible objects and approximate counts
3. **spatial_relationships** - Description of spatial relations (left/right, near/far, etc.)
4. **scene_context** - Classification of the overall scene type
5. **attributes** - Description of visual attributes such as color, lighting, or materials

All tasks use a **128-token generation limit** to maintain consistency across models.

## Metrics Collected

The benchmark focuses on **deployment-relevant system metrics** rather than task accuracy.

### Latency
* Mean latency
* Tail latency (P90 / P99)

### Throughput
* Images processed per second

### Memory Usage
* Peak RAM usage
* GPU memory footprint

### Power Signals
Platform-specific telemetry:
* **NVML** on workstation GPUs
* **tegrastats / VDD_GPU rail** on Jetson platforms

### Token Statistics
* Total tokens generated
* Tokens per second

These metrics help capture **runtime behavior and resource pressure** during inference.

## Repository Structure

```
Benchmarking/
├── Dell/
│   ├── scripts/          # Benchmarking scripts for Dell Pro Max
│   └── outputs/          # Results and logs
├── Thor/
│   ├── scripts/          # Benchmarking scripts for Jetson Thor
│   └── outputs/          # Results and logs
├── Orin/
│   ├── scripts/          # Benchmarking scripts for Jetson AGX Orin
│   └── outputs/          # Results and logs
└── data/testsets/        # COCO manifest files
```

## Setup & Requirements

### Software Environment
* **OS:** Ubuntu 20.04+
* **CUDA:** 12.x
* **Python:** 3.10+
* **PyTorch:** 2.5.1
* **Transformers:** 4.48.1

### Dataset Preparation

Download COCO validation images:
```bash
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
- Platform-specific power monitoring
- GPU utilization tracking
- JSONL logging with resume capability
- Warmup procedures

## Observations from Initial Experiments

Several deployment-relevant patterns emerged during exploratory evaluation:

### Low-Latency Model
**Moondream2**
* Lowest observed latency across platforms
* Small memory footprint
* Stable runtime behavior under repeated inference

### High Token Generation
**SmolVLM2**
* Very high token output relative to other models
* Increases downstream buffering and scheduling pressure

### Balanced Deployment Candidate
**InternVL2-2B**
* Moderate latency
* Moderate memory requirements
* Consistent behavior across devices

## Limitations

This benchmark focuses on **deployment feasibility rather than task correctness**.

Current limitations include:
* No quantitative evaluation of caption accuracy or VQA performance
* Power measurement inconsistencies on early JetPack 7 builds
* GPU telemetry limitations on Jetson Thor

## Future Work

Planned extensions include:
1. Accuracy evaluation using VQA benchmarks
2. Additional deployment scenarios
3. Improved GPU telemetry on Jetson platforms
4. Expanded datasets and workloads

## Acknowledgments

This work is supported by:

**NSF SAGE Testbed — Grant OAC-2331263**

With additional support from:
* Electronic Visualization Laboratory (UIC)
* Argonne National Laboratory

**Project Status:** Active research project  
**Last Updated:** March 2026
