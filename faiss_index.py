import faiss
import numpy as np
import json

with open("embedded_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dim = len(data[0]["embedding"])
index = faiss.IndexFlatL2(dim)  # L2 distance index

embeddings = np.array([entry["embedding"] for entry in data], dtype="float32")
index.add(embeddings)

faiss.write_index(index, "equinet_faiss.index")

with open("embedded_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[INFO] FAISS index stored with {index.ntotal} entries.")
