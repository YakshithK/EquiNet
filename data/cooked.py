import json
import random

# Load the JSON file
with open("clustered_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each item
for item in data:
    # Remove embedding if exists
    item.pop("embedding", None)

    # Simulate fairness score (random between 0 and 1)
    item["fairness_score"] = round(random.uniform(0, 1), 2)

# Save the cleaned JSON to a new file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("output.json")
