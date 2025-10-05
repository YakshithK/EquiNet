from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
index = faiss.read_index("equinet_faiss.index")

with open("embedded_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def query_equiNet(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    results = []
    for idx in I[0]:
        results.append({
            "text": data[idx]["text"][:500] + "...",
            "source": data[idx]["source"],
            "metadata": data[idx].get("metadata", {})
        })
    return results

# Example query
results = query_equiNet("indigenous climate adaptation")
for r in results:
    print(r["text"], "\nSource:", r["source"], "\n---")
