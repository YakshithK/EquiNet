from sentence_transformers import SentenceTransformer
import json
import numpy as np

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

with open("processed_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [entry["text"] for entry in data]
embeddings = model.encode(texts, show_progress_bar=True)

for i, entry in enumerate(data):
    entry["embedding"] = embeddings[i].tolist()

with open("embedded_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("[INFO] Embeddings generated and saved.")