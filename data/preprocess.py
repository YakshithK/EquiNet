import json
import re

with open("combined_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

processed_data = []
for entry in data:
    if "text" in entry and len(entry["text"].split()) > 50:
        text = re.sub(r"\s+", " ", entry["text"]).strip()
        processed_data.append({
            "text": text,
            "source": entry.get("source"),
            "language": entry.get("language"),
            "metadata": {k: entry.get(k) for k in ["title", "authors", "date", "keywords"] if entry.get(k)}
        })

with open("processed_dataset.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

print(f"[INFO] Preprocessed dataset: {len(processed_data)} entries")
