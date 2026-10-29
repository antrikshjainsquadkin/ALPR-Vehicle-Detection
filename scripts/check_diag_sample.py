import json

with open("datasets/two_wheeler/zero_detection_153_investigation.json", "r") as f:
    data = json.load(f)

diag = data["detailed_diagnostics"]
for k in ["2.jpg", "3.jpg", "4.jpg", "10.jpg", "11.jpg"]:
    print(f"=== {k} ===")
    print(json.dumps(diag.get(k, {}), indent=2))

