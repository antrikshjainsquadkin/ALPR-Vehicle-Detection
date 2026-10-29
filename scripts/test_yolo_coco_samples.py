import os
from pathlib import Path
from ultralytics import YOLO

def test_samples():
    model = YOLO("yolov8n.pt")  # standard COCO model
    
    train_dir = Path(r"C:\Users\A\Downloads\train")
    sample_train = [train_dir / f"{i}.jpg" for i in [1, 10, 20, 50, 100] if (train_dir / f"{i}.jpg").exists()]
    
    brain_uploads = list(Path(r"C:\Users\A\.gemini\antigravity\brain\dae2382e-fd39-4f64-9bf4-7f3f1f8cf1be\.user_uploaded").glob("*.jpg"))
    
    print("=== TESTING SAMPLES FROM DOWNLOADS/TRAIN ===")
    for img_path in sample_train:
        results = model(str(img_path), verbose=False)
        for r in results:
            classes = [model.names[int(c)] for c in r.boxes.cls]
            confs = [round(float(c), 3) for c in r.boxes.conf]
            print(f"  {img_path.name}: detected {list(zip(classes, confs))}")

    print("\n=== TESTING SESSION USER UPLOADS ===")
    for img_path in brain_uploads:
        results = model(str(img_path), verbose=False)
        for r in results:
            classes = [model.names[int(c)] for c in r.boxes.cls]
            confs = [round(float(c), 3) for c in r.boxes.conf]
            print(f"  {img_path.name}: detected {list(zip(classes, confs))}")

if __name__ == "__main__":
    test_samples()

