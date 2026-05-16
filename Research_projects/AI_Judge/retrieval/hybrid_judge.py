import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index/caption_index.faiss"
METADATA_PATH = "embeddings/caption_metadata.json"

RULE_WEIGHT = 0.7
RETRIEVAL_WEIGHT = 0.3
K = 5

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


def retrieval_scores(caption, k=K):
    query_embedding = model.encode([caption]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    neighbors = []
    grounding_scores = []
    hallucination_scores = []

    for rank, idx in enumerate(indices[0]):
        item = metadata[idx]

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
            "distance": float(distances[0][rank])
        })

    retrieval_grounding = float(np.mean(grounding_scores)) if grounding_scores else 0.0
    retrieval_hallucination = float(np.mean(hallucination_scores)) if hallucination_scores else 1.0

    return retrieval_grounding, retrieval_hallucination, neighbors


def hybrid_scores(rule_grounding, rule_hallucination, retrieval_grounding, retrieval_hallucination):
    final_grounding = (RULE_WEIGHT * rule_grounding) + (RETRIEVAL_WEIGHT * retrieval_grounding)
    final_hallucination = (RULE_WEIGHT * rule_hallucination) + (RETRIEVAL_WEIGHT * retrieval_hallucination)

    return final_grounding, final_hallucination


if __name__ == "__main__":
    caption = input("Enter caption: ")

    rule_grounding = float(input("Enter rule grounding score: "))
    rule_hallucination = float(input("Enter rule hallucination score: "))

    retrieval_grounding, retrieval_hallucination, neighbors = retrieval_scores(caption)

    final_grounding, final_hallucination = hybrid_scores(
        rule_grounding,
        rule_hallucination,
        retrieval_grounding,
        retrieval_hallucination
    )

    print("\nRETRIEVAL SCORES")
    print("Retrieval Grounding:", round(retrieval_grounding, 3))
    print("Retrieval Hallucination:", round(retrieval_hallucination, 3))

    print("\nHYBRID SCORES")
    print("Final Hybrid Grounding:", round(final_grounding, 3))
    print("Final Hybrid Hallucination:", round(final_hallucination, 3))

    print("\nTOP RETRIEVED EXAMPLES")
    for n in neighbors:
        print("-" * 50)
        print("Rank:", n["rank"])
        print("Caption:", n["caption"])
        print("Grounding:", n["grounding"])
        print("Hallucination:", n["hallucination"])
        print("Distance:", round(n["distance"], 3))
