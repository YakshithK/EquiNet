import json
import numpy as np
import umap
import faiss

# CONFIG
FAISS_INDEX_FILE = "faiss_index/equinet_faiss.index"
METADATA_FILE = "faiss_index/equinet_metadata.json"
OUTPUT_FILE = "sphere_data.json"

# LOAD METADATA
with open(METADATA_FILE, "r") as f:
    metadata_list = json.load(f)

# LOAD FAISS INDEX
index = faiss.read_index(FAISS_INDEX_FILE)

num_vectors = index.ntotal
dim = index.d

print(f"FAISS index loaded: {num_vectors} vectors, dimension={dim}")

# EXTRACT EMBEDDINGS
embeddings = np.zeros((num_vectors, dim), dtype=np.float32)
for i in range(num_vectors):
    embeddings[i] = index.reconstruct(i)

print("✅ Reconstructed embeddings from FAISS index.")

# DIMENSIONALITY REDUCTION (to 3D for sphere)
reducer = umap.UMAP(n_components=3, random_state=42)
coords = reducer.fit_transform(embeddings)

print("✅ Reduced embeddings to 3D coordinates.")

# CREATE JSON DATA
sphere_data = []
for idx, meta in enumerate(metadata_list):
    sphere_data.append({
        "id": f"point_{idx}",
        "coords": coords[idx].tolist(),
        "cluster": meta.get("cluster", 0),
        "fairness_score": meta.get("fairness_score", 0.5),
        "source": meta.get("source", "Unknown"),
        "domain": meta.get("domain", "Unknown"),
        "language": meta.get("language", "en"),
        "text": meta.get("text", "")[:500],
        "metadata": {
            "timestamp": meta.get("timestamp", ""),
            **meta.get("extra", {})
        }
    })

# SAVE TO JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(sphere_data, f, indent=2)

print(f"✅ Generated {OUTPUT_FILE} with {len(sphere_data)} points.")
