from pathlib import Path
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")
raw_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect\datasets\two_wheeler\raw_images")

test_files = ["2.jpg", "3.jpg", "4.jpg", "7.jpg", "10.jpg", "11.jpg"]
print("=== INSPECTING UNLABELLED NO_DETECTION IMAGES ===")
for fname in test_files:
    fpath = raw_dir / fname
    if not fpath.exists():
        continue
    with Image.open(fpath) as img:
        size = img.size
    res = model.predict(str(fpath), conf=0.10, verbose=False)[0]
    classes = [model.names[int(c)] for c in res.boxes.cls]
    confs = [round(float(c), 3) for c in res.boxes.conf]
    print(f"{fname} (size {size}): detected {list(zip(classes, confs))}")

