import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input JSONL
INPUT_FILE = "runs/week12_final.jsonl"

# Output files
EMBEDDINGS_OUT = "embeddings/caption_embeddings.npy"
METADATA_OUT = "embeddings/caption_metadata.json"

embeddings = []
metadata = []

# Read JSONL
with open(INPUT_FILE, "r") as f:
    for line in f:
        item = json.loads(line)

        caption = item.get("caption", "")

        if caption.strip() == "":
            continue

        # Generate embedding
        emb = model.encode(caption)

        embeddings.append(emb)

        metadata.append({
            "image": item.get("image"),
            "caption": caption,
            "hallucination": item.get("hallucination"),
            "grounding": item.get("grounding")
        })

# Convert to numpy
embeddings = np.array(embeddings)

# Save embeddings
np.save(EMBEDDINGS_OUT, embeddings)

# Save metadata
with open(METADATA_OUT, "w") as f:
    json.dump(metadata, f, indent=2)

print("DONE")
print("Embeddings shape:", embeddings.shape)
print("Saved embeddings to:", EMBEDDINGS_OUT)
