# SAGE Edge: Compact VLM Deployment Benchmarking Suite

Deployment-oriented evaluation of compact vision-language models (VLMs) on edge computing platforms for the NSF SAGE testbed.

## Overview

This repository contains benchmarking scripts, semantic evaluation scripts, and results used to evaluate **six compact vision-language models (VLMs)** across **three representative edge computing platforms** used in the SAGE testbed.

Rather than focusing on traditional accuracy leaderboards, this project evaluates **deployment-critical metrics** including:

* Latency (including tail latency)
* Throughput
* Memory footprint
* Platform-specific power and energy signals
* Output quality (CLIPScore, entity recall)

The goal is to better understand **model-device trade-offs for edge deployment**, where memory, power, and latency constraints often determine feasibility.

The deployment benchmark uses a **standardized five-task workload across 500 COCO validation images**, enabling consistent comparison across model and hardware configurations. This work was published at **PAISE '26 (IPDPS 2026 Workshop)**: *Evaluating Deployment Trade-offs of Compact Vision-Language Models on Edge Systems*.

## Models Evaluated

Six compact VLMs were selected to balance **model size, accessibility, and architectural diversity**. All models run with fp16 and bf16 precision via HuggingFace Transformers.

| Model | Parameters | Notes |
|-------|-----------|-------|
| **Moondream2** | 1.8B | Compact vision encoder; lowest latency; Pareto-optimal for edge |
| **SmolVLM2** | 2B | Dense visual tokenization; ~3M tokens/run |
| **LLaVA-Mini** | 9B | Smaller LLaVA architecture; high memory demand |
| **Gemma 3n (E4B-IT)** | ~4B effective | Selective parameter activation; highest CLIPScore |
| **InternVL2-2B** | 2B | Balanced vision-language architecture; strong mid-ground |
| **Phi-3.5-Vision** | 4.2B | Microsoft multimodal model; high latency on Orin |

These models represent a range of compact multimodal architectures commonly considered for edge deployment.

## Hardware Platforms

The experiments were conducted across three representative SAGE environments.

### DGX Spark (Dell Pro Max GB10)

Development / near-edge workstation used for comparison experiments.

* NVIDIA RTX A2000 (12 GB GPU), Intel Xeon, 64 GB RAM
* Used to compare edge results with a development environment

> DGX Spark refers to the Dell Pro Max GB10 OEM implementation.

### NVIDIA Jetson AGX Thor

Next-generation embedded AI platform designed for robotics and edge AI.

* Up to 128 GB unified memory
* Used for higher-performance edge inference
* JetPack 7 experimental environment

### NVIDIA Jetson AGX Orin

Embedded platform representing **power-constrained edge deployments**.

* 32 GB unified memory
* Configured with a 50W power limit
* Common platform for edge robotics and sensing systems

### Platform-Specific Telemetry

* **Dell DGX**: GPU power via NVML (`pynvml`)
* **Thor**: Power via `tegrastats` (VIN rail); reports GPU frequency (MHz), not utilization — `pynvml` is not supported
* **Orin**: Power via `tegrastats` (VDD_GPU rail)

Power signals are platform-specific and not cross-device normalized.

## Evaluation Methodology

### Dataset

Experiments use a subset of the **COCO (Common Objects in Context) validation dataset**.

* **500 images**, reused across all experiments
* Enables consistent comparison across models and hardware
* Manifest files in `data/testsets/`

### Task Suite

Each image is evaluated with a standardized **five-task vision-language workload** designed to reflect typical edge inference scenarios.

1. **caption_brief** — Short factual caption describing the scene
2. **objects_and_counts** — Identification of visible objects and approximate counts
3. **spatial_relationships** — Description of spatial relations (left/right, near/far, etc.)
4. **scene_context** — Classification of the overall scene type
5. **attributes** — Description of visual attributes such as color, lighting, or materials

All tasks use a **128-token generation limit** to maintain consistency across models. No prompt engineering was applied — prompts are held constant across all models.

## Metrics Collected

The benchmark focuses on **deployment-relevant system metrics** rather than task accuracy.

### Latency
* Mean per-image inference time (ms)
* Tail latency (P90 / P99)
* Tail ratio (P99 / P50)

### Throughput
* Images processed per second
* Tokens per second
* TPS / Watt

### Memory Usage
* Peak RAM usage (GB)
* GPU memory footprint

### Power and Energy
Platform-specific telemetry:
* **NVML** on Dell DGX
* **tegrastats** on Jetson platforms

Derived: J/token, total energy per run.

### Token Statistics
* Total tokens generated
* Tokens per second

## Semantic Evaluation

A separate evaluation pass measuring caption output quality, run independently of the deployment benchmark to avoid contaminating hardware measurements. Covers **4 of 6 models** (Moondream2, SmolVLM2, InternVL2-2B, Gemma 3n) across all three devices.

> Phi-3.5-Vision and LLaVA-Mini are not included in current semantic results. Gemma 3n on Orin is pending.

The pipeline runs in three stages:

1. **`bench_semantic_hf.py`** — Runs each VLM on 500 COCO images with 5 repeated generations per (image, task). Captures raw text outputs and hardware telemetry. Writes `runs.jsonl`. No embeddings or quality scoring happen here.

2. **`compute_quality_metrics.py`** — Reads `runs.jsonl`. Loads sentence encoders (mpnet) and CLIP. Computes CLIPScore, entity recall, output stability, and cross-device agreement. Writes `quality_scores.jsonl`.

3. **`analyze_semantic.py`** — Reads both files. Computes summary statistics and produces figures.

### Semantic Metrics
* **CLIPScore** — Cosine similarity between CLIP-ViT-L/14 image and text embeddings
* **Entity recall** — Fraction of COCO ground-truth entities present in generated captions
* **Output stability** — Mean pairwise cosine similarity across 5 repeated runs (within-device)
* **Cross-device agreement** — Cosine similarity between mean embeddings across devices

## Repository Structure

```
Benchmarking/
├── Dell/
│   ├── scripts/               # Deployment benchmark scripts (one per model)
│   └── outputs/
│       ├── benchmark/         # Deployment run outputs per model (timestamped)
│       └── semantic/          # Semantic evaluation outputs per model (timestamped)
├── Thor/
│   ├── scripts/
│   └── outputs/
│       ├── benchmark/
│       └── semantic/
├── Orin/
│   ├── scripts/
│   └── outputs/
│       ├── benchmark/
│       └── semantic/
├── semantic/                  # Shared semantic pipeline scripts (Steps 1–3)
│   ├── bench_semantic_hf.py
│   ├── compute_quality_metrics.py
│   └── analyze_semantic.py
├── ollama_reproducibility_experiment
│   ├── ollama_run_experiment.py      # Ollama-based inference + nomic embeddings (Q4_K_M)
├── figures/                   # Paper and presentation figures
├── data/
│   └── testsets/              # COCO manifest files
└── README.md
```

Output folders follow the naming convention:
* `benchmark/`: `{model}/{model}_{dtype}_{n}img_{timestamp}/`
* `semantic/`: `{model}/{model}_{device}_{timestamp}/` containing `runs.jsonl`, `quality_scores.jsonl`, `run_meta.json`, `summary.json`

## Setup & Requirements

### Software Environment
* **OS:** Ubuntu 20.04+
* **CUDA:** 12.x
* **Python:** 3.10+
* **PyTorch:** 2.5.1
* **Transformers:** 4.48.1

Each model runs in its own virtual environment due to dependency conflicts between models.

### Dataset Preparation

Download COCO validation images and ground truth annotations:

```bash
# Validation images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Ground truth annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

COCO images are excluded from the repository:

```
Benchmarking/data/coco/
```

### Git LFS

Large output files are stored with Git LFS. One-time setup required before pulling:

```bash
git lfs install
```

## Usage

### Deployment Benchmark

Navigate to the appropriate platform directory and run the per-model benchmark script:

```bash
cd SageEdge/Benchmarking/semantic_experiment/bench_semantic_hf.py
# Example: Run Moondream2 benchmark
python benchmark_moondream2.py \
    --images_dir /path/to/val2017 \
    --output_dir ../outputs/benchmark \
    --num_images 500 \
    --dtype bf16
```

Each script includes:
* Standardized 5-task evaluation suite
* Platform-specific power monitoring
* JSONL logging with resume capability
* Warmup procedures

### Semantic Evaluation

```bash
# Step 1 — run on device (compute-intensive)
source envs/moondream-env/bin/activate
python SageEdge/Benchmarking/semantic_experiment/bench_semantic_hf.py \
    --model moondream \
    --manifest data/testsets/coco_val2017_500.txt \
    --output_dir Dell/outputs/semantic \
    --run_group "moondream_dell_$(date +%Y%m%d_%H%M)" \
    --repeats 5 \
    --max_new_tokens 200 \
    --dtype bf16

# Step 2 — compute quality metrics (lightweight, re-runnable)
python SageEdge/Benchmarking/semantic_experiment/compute_quality_metrics.py \
    --input Dell/outputs/semantic/moondream/{run_group}/runs.jsonl

# Step 3 — analyze and plot
python SageEdge/Benchmarking/semantic_experiment/analyze_semantic.py \
    --input Dell/outputs/semantic/moondream/{run_group}/
```

## Key Findings

**Moondream2** is the Pareto-optimal choice for edge deployment — lowest latency across all three platforms (895 ms on Thor, 954 ms on Dell, 2,276 ms on Orin) with a 12 GB memory footprint and near-best CLIPScore.

**SmolVLM2** maintains competitive latency but generates ~3M tokens per run — approximately 30× more than other models — creating disproportionate downstream costs in buffering, scheduling, and power management.

**InternVL2-2B** is the strongest mid-ground option: moderate latency, moderate memory, consistent behavior across devices.

**Hardware changes how fast a model runs, not what it produces.** CLIPScore and entity recall are consistent across devices for the same model. Output stability is near-perfect (cosine similarity > 0.99 in most cases) confirming that fp16/bf16 precision does not introduce meaningful non-determinism.

## Limitations

* Semantic evaluation covers 4 of 6 models; Phi-3.5-Vision and LLaVA-Mini are not included.
* Gemma 3n results are partial across both benchmark and semantic evaluation.
* Power signals are platform-specific and not cross-device normalized.
* Benchmarks were run as a single process; live SAGE nodes are multi-tenant and may exhibit higher tail latency under contention.
* No prompt engineering applied; all tasks use fixed prompts across all models.

## Acknowledgments

This work is supported by:

**NSF Awards No. 2331263 and 2436842** — *A National-Scale Testbed Supporting Artificial Intelligence Research Spanning the Computing Continuum*

With additional support from:
* Electronic Visualization Laboratory (UIC)
* Argonne National Laboratory

**Project Status:** Active research  
**Last Updated:** June 2026