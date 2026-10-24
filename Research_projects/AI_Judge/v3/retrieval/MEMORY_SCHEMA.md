# Judge memory: schema

The Judge memory is the vector database of the project plan. It stores image and caption embeddings,
the AI Judge verdicts and feedback, and metadata (model, digest, date), and supports k-NN search over past
captions and claims with known judgments. It holds no ground truth, so a Sage node can build it from its own data.

- Built by `retrieval/build_memory.py --model {siglip,clip}`.
- Evaluated by `retrieval/eval_memory.py`.
- Queried, and optionally extended, by `retrieval/query_memory.py`.

## Files (`runs/memory/<model>/`)

| File | Content |
|---|---|
| `memory.sqlite` | tables `captions` and `claims` (below) |
| `image_emb.npy` | float32 [n_captions, d]; row i = image of `captions.row_id` i |
| `caption_emb.npy` | float32 [n_captions, d]; mean of the caption's sentence embeddings, L2-normalised |
| `claim_emb.npy` | float32 [n_claims, d]; embedding of the sentence that states the claim (`claims.row_id` j) |
| `joint_caption.index` | FAISS `IndexFlatIP` over [image, caption] / √2 (cosine on the joint vector) |
| `joint_claim.index` | FAISS `IndexFlatIP` over [image, claim sentence] / √2 |
| `info.json` | model, counts, dimensions, build date, rows added later (`added`) |

Embedding models:

- SigLIP: `ViT-B-16-SigLIP` / webli, d = 768.
- CLIP: `ViT-B-32-quickgelu` / openai, d = 512.

The image embeddings are the same as in `runs/retrieval/<model>/eval_emb.npy`.

## Table `captions` (one row per caption)

| Column | Meaning |
|---|---|
| `row_id` | primary key; row in `image_emb.npy` / `caption_emb.npy` |
| `image` | image file name |
| `captioner`, `backend`, `model`, `digest` | captioner and its exact model version |
| `prompt`, `seed`, `max_tokens`, `date` | generation settings and run start |
| `caption`, `n_sentences` | caption text and number of sentences |
| `n_checked`, `n_flagged`, `judge_score` | COCO-class claims checked, flagged, and flagged / checked (`NULL` when nothing is checked) |
| `detector`, `tau` | judge settings (YOLOv8x, 0.25) |
| `loop_run` | feedback-loop run attached (`<captioner>__self`), or `NULL` |
| `feedback_classes`, `feedback_words` | round-0 judge feedback: JSON list of flagged classes, and the caption words sent to the captioner |
| `loop_rounds`, `final_caption`, `final_judge_score`, `loop_date` | result of the feedback loop |
| `img_cap_cos` | cosine (image, caption) |

Feedback comes only from the valid self runs (LLaVA, Gemma 4, Gemma 3n). The Moondream feedback runs are excluded.

## Table `claims` (one row per COCO-class claim)

| Column | Meaning |
|---|---|
| `row_id` | primary key; row in `claim_emb.npy` |
| `caption_row` | `captions.row_id` |
| `image`, `captioner`, `class` | claim |
| `sentence_idx`, `sentence` | first sentence that states the claim (−1 = no single sentence; the start of the caption is used) |
| `yolo_conf` | highest YOLOv8x confidence for the class (0 = no box ≥ 0.05) |
| `flagged` | 1 if `yolo_conf` < `tau` (judge verdict: hallucinated) |
| `img_sent_cos` | cosine (image, sentence) |

The claim set and its order are the same as in `eval_retrieval_v2.py` (8,566 claims). `eval_memory.py` checks this.

## Queries

- **Per claim:** the k past claims of the same class with the highest joint [image, sentence] similarity. The similarity-weighted share of these that the judge flagged is the claim's memory prior.
- **Per caption:** the k nearest past captions in `joint_caption.index`. The weighted mean of their `judge_score` is the caption's memory prior; their `feedback_words` are shown as context.
- Past examples from the same image are skipped.

`query_memory.py --add` appends a new caption to the memory: sqlite rows, `.npy` rows, both FAISS indices and `info.json`. It refuses to change `runs/memory/`. Copy the memory first (`cp -r runs/memory runs/memory_live`) and pass `--memdir runs/memory_live`, so the evaluated version stays unchanged.

## Example

```
python retrieval/query_memory.py --image ../val2017/000000387387.jpg \
    --caption "A yellow school bus parked on gravel next to a dog." --captioner llava --cached_detections
```

The output is a JSON report with the following fields:

- `judge`: verdicts, score, flagged classes, feedback words, out-of-vocabulary claims.
- `claims`: per-claim memory prior and nearest past claims.
- `caption_memory`: nearest past captions and the weighted mean of their judge scores.
- `timing`.
