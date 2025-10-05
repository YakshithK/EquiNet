# merge_datasets.py

import json

FILES = ["blogs_dataset.json", "pdf_dataset.json"]
OUTPUT_FILE = "combined_dataset.json"

all_data = []

for file in FILES:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_data.extend(data)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"[INFO] Combined dataset contains {len(all_data)} entries saved to {OUTPUT_FILE}")
