import json

with open("datasets/two_wheeler/zero_detection_153_investigation.json", "r") as f:
    data = json.load(f)

diag = data["detailed_diagnostics"]
low_conf_cases = [(fname, props.get("max_conf", 0.0), props.get("dimensions"), props.get("sharpness_laplacian")) 
                  for fname, props in diag.items() if props.get("max_conf", 0.0) < 0.40]

print("Images with max_conf < 0.40:", low_conf_cases)

