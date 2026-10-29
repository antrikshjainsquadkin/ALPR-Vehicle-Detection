from pathlib import Path
from ultralytics import YOLO

model = YOLO("models/two_wheeler_detector_baseline.pt")
uploads_dir = Path(r"C:\Users\A\.gemini\antigravity\brain\dae2382e-fd39-4f64-9bf4-7f3f1f8cf1be\.user_uploaded")

print("=== TESTING TWO_WHEELER BASELINE ON USER UPLOADED IMAGES ===")
for img_path in sorted(uploads_dir.glob("*.jpg")):
    results = model.predict(str(img_path), conf=0.25, verbose=False)[0]
    boxes = results.boxes
    confs = [round(float(c), 3) for c in boxes.conf]
    classes = [model.names[int(c)] for c in boxes.cls]
    print(f"  {img_path.name}: detected {list(zip(classes, confs))}")

