"""
Training script for the Updated Two-Wheeler Detector (v2) on Expanded Dataset.
Uses identical architecture and training settings (epochs=8, imgsz=320, freeze=10, seed=42)
as baseline, evaluates both on the identical untouched held-out test split,
and generates a rigorous head-to-head metric comparison.
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

def train_and_compare_updated():
    root_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect")
    yaml_path = root_dir / "datasets" / "two_wheeler" / "splits" / "expanded_retrain" / "two_wheeler_retrain.yaml"
    project_dir = root_dir / "runs" / "detect" / "two_wheeler"
    models_dir = root_dir / "models"

    print("=== STEP 1: SAFETY CHECKS & MODEL INTEGRITY ===")
    alpr_path = models_dir / "license_plate_detector.pt"
    baseline_chk_path = models_dir / "two_wheeler_detector_baseline.pt"

    assert alpr_path.exists(), "ALPR model missing!"
    assert baseline_chk_path.exists(), "Baseline model missing!"

    alpr_size = alpr_path.stat().st_size
    baseline_size = baseline_chk_path.stat().st_size

    print(f"Preserving ALPR detector:      {alpr_size} bytes")
    print(f"Preserving Baseline detector:  {baseline_size} bytes")

    print("\n=== STEP 2: TRAIN UPDATED MODEL (v2) ===")
    model = YOLO("yolov8n.pt")

    start_time = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=8,
        imgsz=320,
        batch=16,
        freeze=10,
        device="cpu",
        project=str(project_dir),
        name="updated_retrain",
        exist_ok=True,
        seed=42,
        deterministic=True,
        verbose=True
    )
    train_time = time.time() - start_time
    print(f"[Updated Train] Completed in {train_time:.1f}s ({train_time/60:.2f} mins).")

    best_weights = project_dir / "updated_retrain" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = project_dir / "updated_retrain" / "weights" / "last.pt"

    v2_checkpoint = models_dir / "two_wheeler_detector_v2.pt"
    shutil.copy2(best_weights, v2_checkpoint)
    print(f"[Updated Train] Saved checkpoint to {v2_checkpoint} ({v2_checkpoint.stat().st_size} bytes).")

    # Re-verify safety check
    assert alpr_path.stat().st_size == alpr_size, "CRITICAL: ALPR model was altered!"
    assert baseline_chk_path.stat().st_size == baseline_size, "CRITICAL: Baseline model was altered!"
    print("[Safety Check PASS] Existing models 100% untouched.")

    print("\n=== STEP 3: HEAD-TO-HEAD EVALUATION ON UNTOUCHED TEST SET (imgsz=640) ===")
    
    # 1. Evaluate baseline on test split
    print("\n--> Evaluating Baseline Model on Test Set...")
    baseline_model = YOLO(str(baseline_chk_path))
    eval_base = baseline_model.val(
        data=str(yaml_path),
        split="test",
        imgsz=640,
        project=str(project_dir),
        name="eval_baseline_test_official",
        exist_ok=True,
        device="cpu",
        verbose=False
    )

    base_p = float(eval_base.box.mp)
    base_r = float(eval_base.box.mr)
    base_map50 = float(eval_base.box.map50)
    base_map = float(eval_base.box.map)
    base_speed = eval_base.speed

    # 2. Evaluate updated v2 model on test split
    print("--> Evaluating Updated Model (v2) on Test Set...")
    v2_model = YOLO(str(v2_checkpoint))
    eval_v2 = v2_model.val(
        data=str(yaml_path),
        split="test",
        imgsz=640,
        project=str(project_dir),
        name="eval_v2_test_official",
        exist_ok=True,
        device="cpu",
        verbose=False
    )

    v2_p = float(eval_v2.box.mp)
    v2_r = float(eval_v2.box.mr)
    v2_map50 = float(eval_v2.box.map50)
    v2_map = float(eval_v2.box.map)
    v2_speed = eval_v2.speed

    # Compute absolute deltas
    delta_p = v2_p - base_p
    delta_r = v2_r - base_r
    delta_map50 = v2_map50 - base_map50
    delta_map = v2_map - base_map

    comparison = {
        "dataset_counts": {
            "train_baseline": 183,
            "train_updated": 389,
            "val_frozen": 22,
            "test_frozen": 22
        },
        "baseline_model": {
            "checkpoint": str(baseline_chk_path.name),
            "precision": round(base_p, 4),
            "recall": round(base_r, 4),
            "mAP_50": round(base_map50, 4),
            "mAP_50_95": round(base_map, 4),
            "inference_ms": round(base_speed.get("inference", 0.0), 2)
        },
        "updated_model_v2": {
            "checkpoint": str(v2_checkpoint.name),
            "precision": round(v2_p, 4),
            "recall": round(v2_r, 4),
            "mAP_50": round(v2_map50, 4),
            "mAP_50_95": round(v2_map, 4),
            "inference_ms": round(v2_speed.get("inference", 0.0), 2)
        },
        "absolute_changes": {
            "delta_precision": round(delta_p, 4),
            "delta_recall": round(delta_r, 4),
            "delta_mAP_50": round(delta_map50, 4),
            "delta_mAP_50_95": round(delta_map, 4)
        }
    }

    comp_path = project_dir / "baseline_vs_updated_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n================ HEAD-TO-HEAD TEST RESULTS ================")
    print(f"Metric          | Baseline (183 imgs) | Updated v2 (389 imgs) | Absolute Change")
    print(f"----------------|---------------------|-----------------------|----------------")
    print(f"Precision       | {base_p:19.4f} | {v2_p:21.4f} | {delta_p:+15.4f}")
    print(f"Recall          | {base_r:19.4f} | {v2_r:21.4f} | {delta_r:+15.4f}")
    print(f"mAP@0.5         | {base_map50:19.4f} | {v2_map50:21.4f} | {delta_map50:+15.4f}")
    print(f"mAP@0.5:0.95    | {base_map:19.4f} | {v2_map:21.4f} | {delta_map:+15.4f}")
    print(f"Inference Lat.  | {base_speed.get('inference',0):16.1f} ms | {v2_speed.get('inference',0):18.1f} ms | {v2_speed.get('inference',0)-base_speed.get('inference',0):+12.1f} ms")
    print("===========================================================")
    print(f"Comparison report saved to: {comp_path}")

    return comparison

if __name__ == "__main__":
    train_and_compare_updated()

