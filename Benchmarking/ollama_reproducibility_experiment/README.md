# Ollama Reproducibility Experiment

An exploratory cross-device reproducibility study of compact vision-language models (VLMs) using Ollama. This experiment was conducted independently from the primary HuggingFace deployment benchmark and served as the basis for the **GCASR 2026 poster**: *A Joint Evaluation of Deployment Efficiency and Semantic Accuracy in Vision-Language Models on Edge Systems*.

## Motivation

Before committing to full semantic evaluation across all models and devices, we wanted to answer a foundational question: **do VLMs produce consistent outputs across different hardware platforms when all other conditions are held constant?**

If a model generates meaningfully different captions on Dell vs. Thor for the same image, then cross-device quality comparisons are confounded by hardware. This experiment tests that assumption using a controlled setup with Ollama for reproducible inference conditions.

## Experiment Design

| Setting | Value |
|---|---|
| Runtime | Ollama (Q4_K_M quantization) |
| Devices | Dell Pro Max GB10, NVIDIA Jetson AGX Thor |
| Images | 100 COCO val2017 images |
| Prompt | Single prompt: "Describe what you see in this image." |
| Runs per image | 5 |
| Generation | Deterministic (temperature 0, fixed seed) |
| Embeddings | `nomic-embed-text` via Ollama |

### Models Tested

| Model | Parameters |
|---|---|
| moondream | 1.8B |
| llava-phi3 | 3.8B |
| gemma3 | 4B |
| minicpm-v | 8B |
| qwen2.5vl | 3B |
| qwen3-vl | 2B |

> These models differ from the six models in the primary HuggingFace benchmark. This experiment used Ollama-native model variants available at the time of the study.

## Methodology

Each model was run on both devices independently. For each image, the model generated 5 responses. Cross-device reproducibility was measured as the cosine similarity between mean output embeddings from Dell and Thor. Within-device stability was measured as the mean pairwise cosine similarity across the 5 repeated runs on each device.

Embeddings were produced using `nomic-embed-text` via the Ollama API to ensure consistent embedding behavior across devices.

## Key Findings

* **Five of six models cluster between 0.96 and 0.98 cross-device cosine similarity**, confirming that outputs reproduce consistently across platforms.
* **Within-device stability is 0.99+ on both platforms** across every task in the 5-task workload.
* `qwen3-vl` is an outlier at 0.912, below the 0.96 reproducibility threshold — flagged for further investigation.
* Tail latency ratio (P99/P50) transfers consistently across devices, confirming that runtime behavior as well as output content reproduces.

These findings supported the decision to treat cross-device output quality comparisons as valid in the primary semantic evaluation.

## Usage

```bash
# Run on each device independently
python ollama_run_experiment.py --device dell_gb10 --model moondream
python ollama_run_experiment.py --device thor --model moondream
```

Requirements:
```bash
pip install ollama
ollama pull nomic-embed-text
ollama pull <model-name>
```

## Relation to Primary Benchmark

This experiment is **separate** from the primary HuggingFace deployment benchmark. It uses:

* A different runtime (Ollama vs. HuggingFace Transformers)
* A different model set
* A different quantization method (Q4_K_M vs. fp16/bf16)
* A single prompt vs. the five-task suite
* 100 images vs. 500 images

Results from this experiment are not directly comparable to the primary benchmark and should be interpreted independently.

## Acknowledgments

This work is supported by NSF Awards No. 2331263 and 2436842.

With additional support from the Electronic Visualization Laboratory (UIC) and Argonne National Laboratory.