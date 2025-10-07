import json
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
INPUT_FILE = "embedded_dataset.json"
OUTPUT_FILE = "clustered_dataset.json"
NUM_CLUSTERS = 10  # Tune this for your dataset size

# ---------------------------
# LOAD DATA
# ---------------------------
print("[INFO] Loading embedded dataset...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = np.array([entry["embedding"] for entry in data])
print(f"[INFO] {len(embeddings)} embeddings loaded.")

# ---------------------------
# CLUSTERING
# ---------------------------
print(f"[INFO] Running KMeans clustering with {NUM_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(embeddings)

for i, entry in enumerate(data):
    entry["cluster"] = int(labels[i])

print("[INFO] Clustering complete.")

# ---------------------------
# REPRESENTATION ANALYSIS
# ---------------------------
print("[INFO] Calculating representation per cluster...")
cluster_counts = Counter([entry["cluster"] for entry in data])
total_snippets = len(data)

print("Cluster representation (% of dataset):")
for cluster_id, count in cluster_counts.items():
    representation = count / total_snippets
    print(f"  Cluster {cluster_id}: {representation*100:.2f}%")

# ---------------------------
# FAIRNESS SCORING
# ---------------------------
print("[INFO] Assigning fairness scores...")
max_representation = max(cluster_counts.values()) / total_snippets

for cluster_id, count in cluster_counts.items():
    representation = count / total_snippets
    fairness_score = max_representation / representation
    for entry in data:
        if entry["cluster"] == cluster_id:
            entry["fairness_score"] = round(fairness_score, 3)

print("[INFO] Fairness scoring complete.")

# ---------------------------
# SAVE OUTPUT
# ---------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[INFO] Clustered dataset saved to {OUTPUT_FILE}")
