from sentence_transformers import SentenceTransformer
import json
import numpy as np
import re
import langdetect
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INPUT_FILE = "processed_dataset.json"   # Dataset generated from scraping/transcription
OUTPUT_FILE = "embedded_dataset.json"

model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# TEXT PREPROCESSING FUNCTION
# ---------------------------
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters except punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)
    return text.strip()

# ---------------------------
# CREDIBILITY SCORE FUNCTION
# ---------------------------
def calculate_credibility(entry):
    score = 0.0

    # 1️⃣ Source credibility (example heuristic)
    credible_sources = ["wikipedia", "gov", "edu", "research", "org"]
    source = entry.get("source", "").lower()
    if any(keyword in source for keyword in credible_sources):
        score += 0.5
    elif source != "unknown":
        score += 0.3
    else:
        score += 0.1

    # 2️⃣ Text length — longer content usually more reliable
    text_length = len(entry.get("processed_text", ""))
    if text_length > 500:
        score += 0.4
    elif text_length > 200:
        score += 0.2
    else:
        score += 0.1

    # 3️⃣ Language detection confidence
    language = entry.get("language", "unknown")
    if language != "unknown":
        score += 0.1

    # Normalize to 0–1
    return min(score, 1.0)


# ---------------------------
# LANGUAGE DETECTION FUNCTION
# ---------------------------
def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

# ---------------------------
# MAIN EMBEDDING GENERATION
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Processing {len(data)} entries...")
    texts = []
    for i, entry in enumerate(data):
        entry["id"] = entry.get("id", f"snippet_{i:04d}")
        text = entry.get("text", "")
        processed = preprocess_text(text)
        entry["processed_text"] = processed

        # Detect language
        entry["language"] = detect_language(processed)

        # Ensure metadata
        entry.setdefault("source", "unknown")
        entry.setdefault("domain", "unknown")
        entry.setdefault("credibility_score", calculate_credibility(entry))  # default credibility

        texts.append(processed)

    print("[INFO] Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    for i, entry in enumerate(data):
        entry["embedding"] = embeddings[i].tolist()

    print(f"[INFO] Saving embeddings to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("[INFO] ✅ Embeddings generated and saved.")
