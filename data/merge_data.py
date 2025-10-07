# merge_datasets.py

import json

FILES = [
    "blogs_dataset.json",
    "pdf_dataset.json",
    "equinet_dataset.json"  # added your EquiNet dataset
]

OUTPUT_FILE = "combined_dataset.json"

all_data = []

for file in FILES:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
        print(f"[INFO] Loaded {len(data)} entries from {file}")
    except FileNotFoundError:
        print(f"[WARNING] File {file} not found, skipping.")
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode {file}, skipping.")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"[INFO] Combined dataset contains {len(all_data)} entries saved to {OUTPUT_FILE}")