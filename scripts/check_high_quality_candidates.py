import json
from pathlib import Path

pseudo_dir = Path("datasets/two_wheeler/pseudo_labels_round2")
review_path = Path("datasets/two_wheeler/review_flagged_round2.json")

with open(review_path, "r") as f:
    flagged = json.load(f)

# Find images with max_conf >= 0.70 and box_count in [1, 2]
high_quality_candidates = [
    item for item in flagged 
    if item.get("max_confidence", 0.0) >= 0.70 and 1 <= item.get("box_count", 0) <= 2
]

print(f"Total flagged items: {len(flagged)}")
print(f"High-confidence candidates with 1-2 boxes: {len(high_quality_candidates)}")
for c in high_quality_candidates[:5]:
    print(" ", c)

