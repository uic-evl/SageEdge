# AI Judge: lightweight visual-grounding judge for VLM captions (v3)

A claim-level hallucination judge that runs on an edge device (NVIDIA Jetson AGX Thor):
object claims are extracted from a VLM caption and verified against a YOLO detector.
A large VLM (Qwen3-VL 32B) is used as a reference judge, and everything is scored
against COCO ground truth and blinded human labels.

Beyond detection, the judge is used to (1) measure grounding as precision + coverage, (2) give feedback
to the captioner in a rewrite loop, (3) run an edge-cloud cascade with the VLM judge, (4) add a retrieval
layer (image memory bank, image-text agreement, memory of past judge verdicts) and (5) check color
attributes. Each part has its own folder below.

## Main results (1,000 COCO val2017 images, 4 captioners, Jetson AGX Thor)
| Experiment | Result |
|---|---|
| Detector size (rule judge, tau 0.25) | mean F1 0.680 (YOLOv8n) -> 0.812 (YOLOv8x); YOLOv8x estimates CHAIR-i within 1.6 pp without labels |
| YOLOv8x judge vs Qwen3-VL-32B (889 claims) | F1 0.770 vs 0.755 (tied); 27.8 ms vs 72.9 s; 1.8 J vs 4.1 kJ per caption |
| Blinded human study (909 labels) | F1 vs human: YOLOv8x 0.58, Qwen 0.60, COCO GT 0.62; COCO flags 15.4% of claims, the human 8.0% |
| Synthetic insertions (YOLOv8x) | caught 99.5% random, 96.4% popular, 92.0% adversarial; parser 100%, negation leak 0% |
| Grounding (precision + coverage) | Gemma 4 covers 44% of the annotated classes (others 63-69%); judge coverage error <= 2.4 pp |
| Feedback loop (self vs control vs oracle) | CHAIR-i LLaVA 18.8 -> 8.3, Gemma 4 15.5 -> 6.7, Gemma 3n 15.1 -> 9.8; 78-80% of the oracle gain; Moondream cannot use feedback |
| Edge-cloud cascade (CV, 30% budget) | F1 0.811 (better than YOLOv8x 0.770 and Qwen 0.755), 26.7% of captions sent to the VLM |
| Retrieval (SigLIP) | image retrieval improves AUC, not F1; image-text agreement + judge-verdict memory: F1 +0.016 (YOLOv8x), +0.035 (YOLOv8n) |
| Judge memory (captions, claims, verdicts, feedback) | caption-side features add no F1 over that judge with SigLIP (+0.003 YOLOv8x, -0.002 YOLOv8n, n.s.); a judge-labeled memory works as well as a GT-labeled one (AUC 0.823 vs 0.822) |
| Color attributes | 5.2% of 194 human-labeled color claims wrong; zero-shot color check on YOLO boxes AUC 0.89 |
| Count claims ("two dogs") | 19.2% of 895 count claims say more objects than COCO annotates; YOLOv8x box count finds them with F1 0.866 |

## How the judge works
```
caption --> parse_caption_v3 --> object claims (bus, man->person, dog, sky; "no cars" = negated)
image   --> YOLOv8x (conf >= 0.25) --> detected classes (bus, person)
claims vs detections --> bus OK, person OK, dog UNSUPPORTED (likely hallucination), sky ? (not a COCO class)
hallucination score = unsupported / checked = 1/3
```
On the Jetson AGX Thor GPU this takes about 28 ms per caption with YOLOv8x
(27.4 ms detection + 0.4 ms parsing and matching), vs. 72.9 s for a Qwen3-VL-32B judge.

| Judge (Jetson AGX Thor, 120 W) | Time / caption (GPU) | Memory | Energy / caption | F1 vs COCO GT |
|---|---|---|---|---|
| YOLOv8n + parser | 8.3 ms | 36 MB | 0.36 J | 0.56 |
| **YOLOv8x + parser** | **27.8 ms** | **0.5 GB** | **1.8 J** | **0.77** |
| Qwen3-VL-32B (Q4_K_M, Ollama) | 72.9 s | 20 GB | 4,143 J | 0.76 |

F1 on the same 889 claims (100 images x 4 captioners). Energy is the whole module (tegrastats, VIN rail),
see `results/energy.json`.

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
| `data/coco_val2017_1000_ids.txt` | COCO image IDs of the 1,000 images |
| `results/benchmark.json` | latency and memory on the Thor (GPU, 120 W mode) |
| `results/energy.json` | energy per caption on the Thor (tegrastats) |
| `synthetic/` | synthetic hallucination stress test (POPE-style insertions + negation check) |
| `grounding/` | grounding = precision + coverage, label-free estimate by the judge, captioner pairs |
| `refine/` | judge-feedback rewrite loop (self / control / oracle / Moondream refiner), evaluation, Moondream prompt test |
| `cascade/` | edge-cloud cascade (YOLOv8x on the edge, Qwen3-VL-32B for uncertain claims), CV thresholds |
| `retrieval/` | COCO train2017 memory bank, SigLIP/CLIP embeddings, FAISS k-NN, image-text agreement, judge-verdict memory |
| `attributes/` | color + object claims: blinded labeling page, zero-shot color judge on YOLO boxes, evaluation; count claims vs COCO instance counts |
| `data/color_pairs.csv`, `data/color_labels.csv` | the 202 color claims shown to the annotator and the human labels |
| `data/color_claims_all.csv` | all 1,357 color claims of the 4 captioners (1,000 images) |

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

# 1b. Paired CHAIR-i differences between captioners
python captioner_pairs.py --coco_ann annotations/instances_val2017.json

# 2. YOLO judge vs Qwen3-VL-32B judge on the same claims (100 images x 4 captioners)
python compare_judges.py --coco_ann annotations/instances_val2017.json

# 3. Blinded human study (909 labels)
python analyze_human.py --coco_ann annotations/instances_val2017.json
```
`evaluate.py` and `compare_judges.py`/`analyze_human.py` use 1,000 and 1,000/2,000
bootstrap samples with fixed seeds, so the printed numbers are identical on every run.

### Extensions (same COCO file; `ANN=annotations/instances_val2017.json`)
```bash
# 4. Synthetic hallucinations (also needs instances_train2017.json for popular/adversarial objects)
python synthetic/synthetic_test.py --coco_ann $ANN --coco_train annotations/instances_train2017.json

# 5. Grounding: precision + coverage (COCO GT and label-free judge estimate), captioner pairs
python grounding/grounding_metrics.py --coco_ann $ANN
python grounding/captioner_pairs_grounding.py

# 6. Edge-cloud cascade (reuses the Qwen verdicts and the clean latency/energy numbers)
python cascade/cascade_eval.py --coco_ann $ANN

# 7. Feedback loop: evaluate the rewrite runs (runs/refine/<captioner>__<condition>.jsonl)
python refine/eval_refine.py --coco_ann $ANN
python refine/eval_refine.py --coco_ann $ANN --subset data/human_validation_100.txt
```

### Extensions that need the GPU / Ollama (Jetson Thor)
```bash
# Feedback loop (resumable; conditions self, control, oracle, moondream; --prompt B = control prompt + "Do not mention X.")
python refine/run_refine_loop.py --captioner llava --condition self
python refine/run_refine_loop.py --captioner llava --condition oracle --coco_ann $ANN
python refine/test_moondream_prompts.py --n 10

# Retrieval: memory bank of 10,000 COCO train2017 images (downloads them once), embeddings + benchmark
python retrieval/build_bank.py --model siglip        # also --model clip (ViT-B-32-quickgelu)
python retrieval/eval_retrieval.py    --coco_ann $ANN --model siglip   # image retrieval prior
python retrieval/eval_retrieval_v2.py --coco_ann $ANN --model siglip   # + image-text agreement, judge-verdict bank
# Judge memory: captions, claims, judge verdicts, loop feedback, metadata + joint FAISS indices (schema: retrieval/MEMORY_SCHEMA.md)
python retrieval/build_memory.py --model siglip     # GPU, after build_bank.py; also --model clip
python retrieval/eval_memory.py  --coco_ann $ANN --model siglip   # CPU; image-sentence agreement, k-NN prior over past claims
python retrieval/eval_memory_grounding.py --coco_ann $ANN --model siglip   # CPU; k-NN estimate of precision and coverage
python retrieval/query_memory.py --image ../val2017/000000387387.jpg --caption "A yellow school bus parked on gravel next to a dog."

# Color attributes: labeling page -> human labels -> color judge -> evaluation
python attributes/make_color_labeling.py --images val2017
python attributes/color_judge.py --model siglip       # also --model clip
python attributes/eval_color.py --labels data/color_labels.csv --coco_ann $ANN --model siglip
# Count claims ("two dogs") vs COCO instance counts; judge = YOLO box count (CPU, stored detections)
python attributes/count_eval.py --coco_ann $ANN
```
The feedback loop records the model digest of every rewrite (`run_info`), so runs made with a different model
version can be detected. Pin the Ollama models while experiments run.

## Re-run the models (Jetson Thor or any CUDA machine)
Needs `pip install -r requirements.txt`, COCO val2017 images in `val2017/`, and Ollama
with `ollama pull moondream llava gemma4:e2b qwen3-vl:32b`.

| Step | Command |
|---|---|
| Captions (prompt "Describe this image.", T=0, seed 42, 160 tokens) | `python run_captions_unified.py --model llava:latest --list data/coco_1000.txt --out runs/llava.jsonl` (also `moondream:latest`, `gemma4:e2b`, `gemma3n-hf`) |
| Detections (YOLOv8 n/s/m/x, boxes >= 0.05) | `python run_detector_ablation.py --list data/coco_1000.txt --out runs/detections` |
| Qwen3-VL-32B judge | `python run_vlm_judge.py --judge qwen3-vl:32b --captions runs/llava.jsonl --images data/human_validation_100.txt --out runs/judge_qwen_llava.jsonl` |
| Latency benchmark (run alone, record `sudo nvpmodel -q`) | `python benchmark_latency.py --list data/human_validation_100.txt --captions runs/llava.jsonl --vlm_n 20 --out runs/benchmark.json` |
| Energy per caption (run alone; tegrastats) | `python energy_bench.py --list data/human_validation_100.txt --captions results/llava.jsonl --out runs/energy.json` |
| Human labeling page (offline HTML) | `python make_labeling_page.py --pairs data/human_v2_pairs_100.csv --images val2017 --out label_tool.html` |

All model scripts are resumable: re-running the same command continues where it stopped.
Ollama models ignore `seed` on some backends, so new captions can differ slightly from `results/`.

## Code
- `judge_v3/parse_caption_v3.py`: claim extraction with negation handling ("no people"), blocked compounds ("table lamp", "bus stop", "train car"), COCO synonym map, and out-of-vocabulary words kept separately.
- `judge_v3/rule_judge_v3.py`: a COCO-class claim is *supported* if YOLO detected that class (confidence >= `min_conf`, 0.25 in the paper). Captions with no checkable claim are *unverifiable* (`None`), not scored as 0.
- `judge.py`: the judge itself (YOLO + parser + rule), for new images and captions.
- `evaluate.py`: CHAIR-i/-s for captioners; precision, recall, F1 and Cohen's kappa for judges (flag = hallucination) vs COCO GT, with bootstrap CIs.
- `captioner_pairs.py`, `compare_judges.py`, `analyze_human.py`: captioner comparison, judge comparison and human study.
- `synthetic/synthetic_test.py`: appends one absent COCO object ("The image also shows a <word>.") chosen at random, by frequency or by co-occurrence; also tests negated sentences.
- `grounding/grounding_metrics.py`: precision (= 1 - CHAIR-i), coverage (share of present classes the caption names), salient coverage, grounded F1; COCO GT, label-free judge estimate and human labels.
- `refine/run_refine_loop.py`: rewrite loop driven by judge flags (feedback names the caption's own words, max 3 rounds); `refine/eval_refine.py`: per-round CHAIR, coverage, human-labeled rate and paired condition comparisons.
- `cascade/cascade_eval.py`: thresholds (lo, hi) on the YOLO confidence; claims in between go to the VLM judge; 5-fold CV grouped by image under a budget.
- `retrieval/build_bank.py`, `eval_retrieval.py`, `eval_retrieval_v2.py`: FAISS (exact inner product) over SigLIP/CLIP image embeddings; retrieval prior from a COCO-labeled or a judge-labeled bank; image-text agreement with "a photo of a <class>."; logistic regression with 5-fold CV grouped by image.
- `retrieval/build_memory.py`, `eval_memory.py`, `query_memory.py`: the Judge memory (sqlite + .npy + FAISS joint [image, text] indices) with judge verdicts, feedback and metadata; k-NN over past claims of the same class and past captions; `query_memory.py` judges one new image + caption end to end and can add it to a copy of the memory (`--add --memdir`). Schema in `retrieval/MEMORY_SCHEMA.md`.
- `attributes/`: color claims (color words before a COCO object), blinded labeling page, SigLIP/CLIP zero-shot color check on YOLOv8x box crops; `count_eval.py`: number words (two..ten) before a COCO object, checked against COCO instance counts and the number of YOLO boxes.

## Notes
- The earlier pipeline (`../judge/`, v1/v2 with caption-text FAISS retrieval) is kept only to reproduce preliminary results; that retrieval added no signal (Spearman 0.06-0.22 with GT). The new retrieval layer in `retrieval/` uses image embeddings instead.
- Moondream's feedback rewrites were broken (84% empty or < 10 words), so its feedback conditions are reported as a negative result; `gemma4__selfB` was excluded because the Gemma 4 model tag changed during the run.
- Timing numbers inside the batch result files are not used; speed comes only from `benchmark_latency.py` with a fixed power mode.
- Image paths inside `results/*.jsonl` are absolute paths from the Thor; the scripts use only the file name.
