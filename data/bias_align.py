# bias_align.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("🔧 Loading model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("📂 Loading dataset...")
with open("embedded_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")

# --- STEP 1: Generate embeddings ---
texts = [d["text"] for d in data]
embeddings = model.encode(texts, show_progress_bar=True)

# --- STEP 2: Tag groups (basic heuristic) ---
for i, entry in enumerate(data):
    entry["embedding"] = embeddings[i].tolist()
    src = entry["source"].lower()
    if any(x in src for x in ["indigenous", "globalvoices", "community", "grassroots", "local"]):
        entry["group"] = "underrepresented"
    else:
        entry["group"] = "mainstream"

# --- STEP 3: Normalize embeddings ---
for d in data:
    emb = np.array(d["embedding"])
    d["embedding"] = (emb / np.linalg.norm(emb)).tolist()

# --- STEP 4: Compute centroids per group ---
groups = {"underrepresented": [], "mainstream": []}
for d in data:
    groups[d["group"]].append(np.array(d["embedding"]))

under_mean = np.mean(groups["underrepresented"], axis=0)
main_mean = np.mean(groups["mainstream"], axis=0)

# --- STEP 5: Align embeddings (equalize representation) ---
for d in data:
    emb = np.array(d["embedding"])
    if d["group"] == "underrepresented":
        aligned = emb - under_mean + main_mean
    else:
        aligned = emb
    d["embedding"] = aligned.tolist()

# --- STEP 6: Save aligned embeddings ---
with open("embedded_dataset_aligned.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Saved aligned dataset → embedded_dataset_aligned.json")

# --- STEP 7: Optional validation ---
from sklearn.metrics.pairwise import cosine_similarity
under = np.array(groups["underrepresented"])
main = np.array(groups["mainstream"])
cross_sim = np.mean(cosine_similarity(under, main))
print(f"🔍 Cross-group similarity (post-alignment): {cross_sim:.3f}")
