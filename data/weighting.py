import json
import faiss
import numpy as np
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
INPUT_FILE = "clustered_dataset.json"
FAISS_INDEX_FILE = "equinet_faiss.index"
METADATA_FILE = "equinet_metadata.json"

# ---------------------------
# LOAD DATA
# ---------------------------
print("[INFO] Loading clustered dataset...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = np.array([entry["embedding"] for entry in data], dtype='float32')
print(f"[INFO] {len(embeddings)} embeddings loaded.")

# ---------------------------
# OPTIONAL: PRUNE LOW-FAIRNESS CLUSTERS
# ---------------------------
fairness_threshold = 0.5  # tune for your domain
print(f"[INFO] Pruning embeddings with fairness score < {fairness_threshold}...")
pruned_data = []
pruned_embeddings = []

for i, entry in enumerate(data):
    if entry["fairness_score"] >= fairness_threshold:
        pruned_data.append(entry)
        pruned_embeddings.append(embeddings[i])

embeddings = np.array(pruned_embeddings, dtype='float32')
data = pruned_data
print(f"[INFO] {len(embeddings)} embeddings remain after pruning.")

# ---------------------------
# BUILD FAISS INDEX
# ---------------------------
print("[INFO] Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 similarity; can switch to IndexFlatIP for cosine
index.add(embeddings)

print(f"[INFO] FAISS index built with {index.ntotal} vectors.")

# ---------------------------
# SAVE INDEX
# ---------------------------
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"[INFO] FAISS index saved to {FAISS_INDEX_FILE}.")

# ---------------------------
# SAVE METADATA
# ---------------------------
metadata = [
    {
        "id": entry["id"],
        "source": entry.get("source", "unknown"),
        "domain": entry.get("domain", "unknown"),
        "language": entry.get("language", "unknown"),
        "cluster": entry.get("cluster", -1),
        "fairness_score": entry.get("fairness_score", 1.0)
    }
    for entry in data
]

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"[INFO] Metadata saved to {METADATA_FILE}")
print("[INFO] ✅ Fairness-weighted EquiNet vector database is ready.")
