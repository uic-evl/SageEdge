# AI Judge: lightweight visual-grounding judge for VLM captions (v3)

A claim-level hallucination judge that runs on an edge device (NVIDIA Jetson AGX Thor):
object claims are extracted from a VLM caption and verified against a YOLO detector.
A large VLM (Qwen3-VL 32B) is used as a reference judge, and everything is scored
against COCO ground truth and blinded human labels.

## How the judge works
```
caption --> parse_caption_v3 --> object claims (bus, man->person, dog, sky; "no cars" = negated)
image   --> YOLOv8x (conf >= 0.25) --> detected classes (bus, person)
claims vs detections --> bus OK, person OK, dog UNSUPPORTED (likely hallucination), sky ? (not a COCO class)
hallucination score = unsupported / checked = 1/3
```
On the Jetson AGX Thor GPU this takes about 28 ms per caption with YOLOv8x
(27.4 ms detection + 0.4 ms parsing and matching), vs. 72.9 s for a Qwen3-VL-32B judge.

## Use the judge
```bash
pip install -r requirements.txt
# one image + caption
python judge.py --image val2017/000000387387.jpg --caption "A yellow bus parked next to a man with a dog."
# a whole caption file -> JSONL with a verdict per claim (same format evaluate.py reads)
python judge.py --captions results/llava.jsonl --out runs/llava_judged.jsonl
```
Options: `--detector yolov8n.pt|yolov8s.pt|yolov8m.pt|yolov8x.pt` (default x), `--conf 0.25`, `--device cuda:0`.
In Python: `from judge import Judge; Judge()("img.jpg", "caption")`.

## Folder layout
| Path | Content |
|---|---|
| `data/coco_1000.txt` | the 1,000 COCO val2017 images used in all experiments |
| `data/human_validation_100.txt` | the 100-image subset judged by Qwen and labeled by a human |
| `data/human_v2_pairs_100.csv` | the 909 (image, object) pairs shown to the annotator |
| `data/human_labels_100.csv` | the human labels (present / absent / unsure) |
| `results/<captioner>.jsonl` | captions of the 4 VLMs (gemma3n, gemma4, llava, moondream) |
| `results/detections/yolov8{n,s,m,x}.jsonl` | all YOLO boxes with confidence >= 0.05 |
| `results/judge_qwen_<captioner>.jsonl` | Qwen3-VL-32B yes/no verdicts (100 images) |
| `results/benchmark.json` | latency and memory on the Thor (GPU, 120 W mode) |

## Reproduce the paper numbers (no GPU needed)
Only the COCO annotation file is needed; all model outputs are in `results/`.
```bash
python3 -m venv venv && source venv/bin/activate
pip install pytest
python -m pytest -q tests/        # 14 unit tests for parser + judge
# download annotations_trainval2017.zip from cocodataset.org -> annotations/instances_val2017.json

# 1. Captioners (CHAIR) and rule judge with each detector vs COCO GT (1,000 images)
for m in gemma3n gemma4 llava moondream; do for d in n s m x; do
  python score_rule_judge.py --captions results/$m.jsonl \
      --detections results/detections/yolov8$d.jsonl --out runs/judged/${m}__yolov8$d.jsonl
done; done
python evaluate.py --coco_ann annotations/instances_val2017.json --files runs/judged/*.jsonl --out runs/eval_summary.csv

# 2. YOLO judge vs Qwen3-VL-32B judge on the same claims (100 images x 4 captioners)
python compare_judges.py --coco_ann annotations/instances_val2017.json

# 3. Blinded human study (909 labels)
python analyze_human.py --coco_ann annotations/instances_val2017.json
```
`evaluate.py` and `compare_judges.py`/`analyze_human.py` use 1,000 and 1,000/2,000
bootstrap samples with fixed seeds, so the printed numbers are identical on every run.

## Re-run the models (Jetson Thor or any CUDA machine)
Needs `pip install -r requirements.txt`, COCO val2017 images in `val2017/`, and Ollama
with `ollama pull moondream llava gemma4:e2b qwen3-vl:32b`.

| Step | Command |
|---|---|
| Captions (prompt "Describe this image.", T=0, seed 42, 160 tokens) | `python run_captions_unified.py --model llava:latest --list data/coco_1000.txt --out runs/llava.jsonl` (also `moondream:latest`, `gemma4:e2b`, `gemma3n-hf`) |
| Detections (YOLOv8 n/s/m/x, boxes >= 0.05) | `python run_detector_ablation.py --list data/coco_1000.txt --out runs/detections` |
| Qwen3-VL-32B judge | `python run_vlm_judge.py --judge qwen3-vl:32b --captions runs/llava.jsonl --images data/human_validation_100.txt --out runs/judge_qwen_llava.jsonl` |
| Latency benchmark (run alone, record `sudo nvpmodel -q`) | `python benchmark_latency.py --list data/human_validation_100.txt --captions runs/llava.jsonl --vlm_n 20 --out runs/benchmark.json` |
| Human labeling page (offline HTML) | `python make_labeling_page.py --pairs data/human_v2_pairs_100.csv --images val2017 --out label_tool.html` |

All model scripts are resumable: re-running the same command continues where it stopped.
Ollama models ignore `seed` on some backends, so new captions can differ slightly from `results/`.

## Code
- `judge_v3/parse_caption_v3.py`: claim extraction with negation handling ("no people"), blocked compounds ("table lamp", "bus stop", "train car"), COCO synonym map, and out-of-vocabulary words kept separately.
- `judge_v3/rule_judge_v3.py`: a COCO-class claim is *supported* if YOLO detected that class (confidence >= `min_conf`, 0.25 in the paper). Captions with no checkable claim are *unverifiable* (`None`), not scored as 0.
- `judge.py`: the judge itself (YOLO + parser + rule), for new images and captions.
- `evaluate.py`: CHAIR-i/-s for captioners; precision, recall, F1 and Cohen's kappa for judges (flag = hallucination) vs COCO GT, with bootstrap CIs.
- `compare_judges.py`, `analyze_human.py`: the judge comparison and the human study.

## Notes
- The earlier pipeline (`../judge/`, v1/v2 with FAISS retrieval) is kept only to reproduce preliminary results; retrieval added no signal (Spearman 0.06-0.22 with GT) and is not part of v3.
- Timing numbers inside the batch result files are not used; speed comes only from `benchmark_latency.py` with a fixed power mode.
- Image paths inside `results/*.jsonl` are absolute paths from the Thor; the scripts use only the file name.
