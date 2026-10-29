import json
from pathlib import Path

with open("datasets/two_wheeler/zero_detection_153_investigation.json", "r") as f:
    data = json.load(f)

diag = data["detailed_diagnostics"]
high_conf_recovered = 0
mod_conf_recovered = 0
for fname, props in diag.items():
    max_c = props.get("max_conf", 0.0)
    if max_c >= 0.60:
        high_conf_recovered += 1
    elif max_c >= 0.40:
        mod_conf_recovered += 1

print(f"Zero detection images (153 total):")
print(f"  Recovered with max_conf >= 0.60: {high_conf_recovered} ({(high_conf_recovered/153)*100:.1f}%)")
print(f"  Recovered with 0.40 <= max_conf < 0.60: {mod_conf_recovered} ({(mod_conf_recovered/153)*100:.1f}%)")
print(f"  Total with confident detection (>= 0.40): {high_conf_recovered + mod_conf_recovered} ({((high_conf_recovered + mod_conf_recovered)/153)*100:.1f}%)")

