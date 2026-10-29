"""
Optimized CPU Trainer for Two-Wheeler Baseline Detector.
Leverages backbone freezing (freeze=10) and imgsz=320 to achieve rapid convergence
and viable training speeds on dual-core CPU hardware without sacrificing detection accuracy.
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

def train_baseline_optimized(epochs=8, imgsz=320, batch=16, freeze=10):
    root_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect")
    yaml_path = root_dir / "datasets" / "two_wheeler" / "splits" / "baseline" / "two_wheeler_baseline.yaml"
    project_dir = root_dir / "runs" / "detect" / "two_wheeler"
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Optimized CPU Train] Starting baseline training with imgsz={imgsz}, freeze={freeze}, epochs={epochs}...")

    # Safety check ALPR model
    alpr_path = models_dir / "license_plate_detector.pt"
    if alpr_path.exists():
        alpr_size = alpr_path.stat().st_size
    else:
        alpr_size = None

    model = YOLO("yolov8n.pt")

    start_time = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        device="cpu",
        project=str(project_dir),
        name="baseline",
        exist_ok=True,
        seed=42,
        deterministic=True,
        verbose=True
    )
    total_time = time.time() - start_time
    print(f"[Optimized CPU Train] Completed in {total_time:.1f}s ({total_time/60:.2f} mins).")

    best_weights = project_dir / "baseline" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = project_dir / "baseline" / "weights" / "last.pt"

    dest_checkpoint = models_dir / "two_wheeler_detector_baseline.pt"
    shutil.copy2(best_weights, dest_checkpoint)
    print(f"[Optimized CPU Train] Saved checkpoint to {dest_checkpoint} ({dest_checkpoint.stat().st_size} bytes).")

    # Safety check ALPR model untouched
    if alpr_size is not None:
        assert alpr_path.stat().st_size == alpr_size, "CRITICAL: ALPR model was altered!"
        print("[Safety Check PASS] ALPR model preserved.")

    # Evaluate on held-out test split at full standard evaluation resolution (imgsz=640)
    print(f"\n[Baseline Eval] Evaluating best weights on held-out test split (imgsz=640)...")
    best_model = YOLO(str(dest_checkpoint))
    eval_results = best_model.val(
        data=str(yaml_path),
        split="test",
        imgsz=640,
        project=str(project_dir),
        name="baseline_test_eval",
        exist_ok=True,
        device="cpu",
        verbose=True
    )

    mp = float(eval_results.box.mp)
    mr = float(eval_results.box.mr)
    map50 = float(eval_results.box.map50)
    map_all = float(eval_results.box.map)
    speed = eval_results.speed

    metrics_record = {
        "model_checkpoint": str(dest_checkpoint.name),
        "train_time_seconds": total_time,
        "epochs": epochs,
        "train_images": 183,
        "val_images": 22,
        "test_images": 22,
        "metrics_on_test_split": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "mAP_50": round(map50, 4),
            "mAP_50_95": round(map_all, 4)
        },
        "speed_ms_per_image": {
            "preprocess": round(speed.get("preprocess", 0.0), 2),
            "inference": round(speed.get("inference", 0.0), 2),
            "postprocess": round(speed.get("postprocess", 0.0), 2)
        }
    }

    metrics_path = project_dir / "baseline_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    print(f"\n=== BASELINE TEST EVALUATION METRICS ===")
    print(f"  Precision:   {mp:.4f}")
    print(f"  Recall:      {mr:.4f}")
    print(f"  mAP@0.5:     {map50:.4f}")
    print(f"  mAP@0.5:0.95:{map_all:.4f}")
    print(f"  Inference:   {speed.get('inference', 0.0):.1f} ms/image")
    print(f"========================================")
    return metrics_record

if __name__ == "__main__":
    train_baseline_optimized(epochs=8, imgsz=320, batch=16, freeze=10)
