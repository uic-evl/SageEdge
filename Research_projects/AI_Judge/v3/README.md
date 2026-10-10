# AI Judge: lightweight visual-grounding judge for VLM captions (v3)

A claim-level hallucination judge that runs on an edge device (NVIDIA Jetson AGX Thor):
object claims are extracted from a VLM caption and verified against a YOLO detector.
A large VLM (Qwen3-VL 32B) is used as a reference judge, and everything is scored
against COCO ground truth and human labels.

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# COCO val2017 images in ./val2017/ and annotations/instances_val2017.json
# Ollama models: ollama pull moondream llava gemma4:e2b qwen3-vl:32b
python -m pytest -q tests/          # 14 unit tests for parser + judge
```

## Pipeline
| Step | Command | Output |
|---|---|---|
| 1. Captions (identical prompt "Describe this image.", T=0, seed 42, 160 tokens) | `python run_captions_unified.py --model <m> --list coco_1000.txt --out runs/v3/<m>.jsonl` | caption, latency, model digest |
| 2. Detections (YOLOv8 n/s/m/x, all boxes >= 0.05) | `python run_detector_ablation.py --list coco_1000.txt --out runs/v3/detections` | one JSONL per detector |
| 3. Rule judge (parser v3 + YOLO) | `python score_rule_judge.py --captions runs/v3/<m>.jsonl --detections runs/v3/detections/yolov8n.jsonl --out runs/v3/judged/<m>_yolov8n.jsonl` | claim verdicts |
| 4. VLM reference judge | `python run_vlm_judge.py --judge qwen3-vl:32b --captions runs/v3/<m>.jsonl --out runs/v3/judge_qwen_<m>.jsonl` | yes/no per claim |
| 5. Evaluation vs COCO GT (+ human labels) | `python evaluate.py --coco_ann annotations/instances_val2017.json --files runs/v3/judged/*.jsonl runs/v3/judge_qwen_*.jsonl --human human_validation_100_labeled_clean.csv` | CHAIR-i/-s, P/R/F1, kappa, 95% bootstrap CIs |

Captioners: `moondream:latest`, `llava:latest`, `gemma4:e2b` (Ollama, `think=False`), `gemma3n-hf` (`google/gemma-3n-E2B-it`, HuggingFace).
All scripts are resumable: re-running the same command continues where it stopped.

## Code
- `judge_v3/parse_caption_v3.py`: claim extraction with negation handling ("no people"), blocked compounds ("table lamp", "bus stop", "train car"), COCO synonym map, and out-of-vocabulary words kept separately.
- `judge_v3/rule_judge_v3.py`: a COCO-class claim is *supported* if YOLO detected that class (confidence >= `min_conf`). Captions with no checkable claim are *unverifiable* (`None`), not scored as 0.
- `evaluate.py`: judge accuracy (flag = hallucination) vs COCO GT and human labels; CHAIR metrics for the captioners.

## Notes
- The earlier pipeline (`judge/`, v1/v2 with FAISS retrieval) is kept only to reproduce preliminary results; retrieval added no signal (Spearman 0.06–0.22 with GT) and is not part of v3.
- Timing numbers from the batch runs are not used for reporting; speed is measured in a separate benchmark with a fixed power mode.
