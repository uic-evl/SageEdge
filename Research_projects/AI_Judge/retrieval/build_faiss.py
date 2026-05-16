import faiss
import numpy as np
import json

# Load embeddings
embeddings = np.load("embeddings/caption_embeddings.npy")

# Convert to float32 (required by FAISS)
embeddings = embeddings.astype("float32")

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index/caption_index.faiss")

print("DONE")
print("Number of embeddings:", index.ntotal)
print("Embedding dimension:", dimension)
print("Saved index to: faiss_index/caption_index.faiss")
