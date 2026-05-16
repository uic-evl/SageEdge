import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("faiss_index/caption_index.faiss")

# Load metadata
with open("embeddings/caption_metadata.json", "r") as f:
    metadata = json.load(f)

# Example query caption
query_caption = input("Enter caption: ")

# Convert caption to embedding
query_embedding = model.encode([query_caption]).astype("float32")

# Search top 5 nearest neighbors
k = 5
distances, indices = index.search(query_embedding, k)

print("\nTOP RETRIEVAL RESULTS:\n")

for rank, idx in enumerate(indices[0]):
    item = metadata[idx]

    print(f"Rank {rank+1}")
    print("Caption:", item["caption"])
    print("Grounding:", item["grounding"])
    print("Hallucination:", item["hallucination"])
    print("Distance:", distances[0][rank])
    print("-" * 50)
