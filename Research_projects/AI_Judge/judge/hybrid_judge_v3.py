from __future__ import annotations
from typing import Dict, Any, List
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from judge.rule_judge_v2 import rule_based_judge_v2

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index/caption_index.faiss"
METADATA_PATH = "embeddings/caption_metadata.json"

RULE_WEIGHT = 0.7
RETRIEVAL_WEIGHT = 0.3
K = 5

_model = None
_index = None
_metadata = None


def _load_retrieval_resources():
    global _model, _index, _metadata

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    if _index is None:
        _index = faiss.read_index(INDEX_PATH)

    if _metadata is None:
        with open(METADATA_PATH, "r") as f:
            _metadata = json.load(f)


def retrieval_scores(caption: str, k: int = K) -> Dict[str, Any]:
    _load_retrieval_resources()

    query_embedding = _model.encode([caption]).astype("float32")
    distances, indices = _index.search(query_embedding, k)

    neighbors = []
    grounding_scores = []
    hallucination_scores = []

    for rank, idx in enumerate(indices[0]):
        item = _metadata[idx]

        grounding = item.get("grounding")
        hallucination = item.get("hallucination")

        if grounding is not None:
            grounding_scores.append(float(grounding))
        if hallucination is not None:
            hallucination_scores.append(float(hallucination))

        neighbors.append({
            "rank": rank + 1,
            "caption": item.get("caption"),
            "grounding": grounding,
            "hallucination": hallucination,
            "distance": float(distances[0][rank]),
        })

    return {
        "retrieval_grounding_score": float(np.mean(grounding_scores)) if grounding_scores else 0.0,
        "retrieval_hallucination_score": float(np.mean(hallucination_scores)) if hallucination_scores else 1.0,
        "retrieval_neighbors": neighbors,
    }


def hybrid_judge_v3(caption: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    rule_result = rule_based_judge_v2(caption, detections)
    retrieval_result = retrieval_scores(caption)

    rule_grounding = rule_result["grounding_score"]
    rule_hallucination = rule_result["hallucination_score"]

    retrieval_grounding = retrieval_result["retrieval_grounding_score"]
    retrieval_hallucination = retrieval_result["retrieval_hallucination_score"]

    hybrid_grounding = (RULE_WEIGHT * rule_grounding) + (RETRIEVAL_WEIGHT * retrieval_grounding)
    hybrid_hallucination = (RULE_WEIGHT * rule_hallucination) + (RETRIEVAL_WEIGHT * retrieval_hallucination)

    return {
        **rule_result,
        **retrieval_result,
        "rule_grounding_score": float(rule_grounding),
        "rule_hallucination_score": float(rule_hallucination),
        "hybrid_grounding_score": float(hybrid_grounding),
        "hybrid_hallucination_score": float(hybrid_hallucination),
        "hybrid_weights": {
            "rule_weight": RULE_WEIGHT,
            "retrieval_weight": RETRIEVAL_WEIGHT,
            "k": K,
        },
    }
