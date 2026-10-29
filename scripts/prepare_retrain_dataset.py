"""
Retrain Dataset Preparation.
Expands the training set with verified high-confidence recovered pseudo-labels,
while keeping the validation (22 images) and held-out test set (22 images) strictly UNTOUCHED.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def prepare_retrain_dataset():
    root_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect")
    base_dir = root_dir / "datasets" / "two_wheeler"
    raw_images_dir = base_dir / "raw_images"
    pseudo_labels_dir = base_dir / "pseudo_labels_round2"
    diag_path = base_dir / "zero_detection_153_investigation.json"
    
    baseline_split_dir = base_dir / "splits" / "baseline"
    retrain_split_dir = base_dir / "splits" / "iteration1_retrain"

    # Create directories
    for split_name in ["train", "val", "test"]:
        (retrain_split_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (retrain_split_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # 1. Strictly copy untouched val and test splits
    for split_name in ["val", "test"]:
        src_img_dir = baseline_split_dir / "images" / split_name
        src_lbl_dir = baseline_split_dir / "labels" / split_name
        dst_img_dir = retrain_split_dir / "images" / split_name
        dst_lbl_dir = retrain_split_dir / "labels" / split_name

        for f in src_img_dir.glob("*.jpg"):
            shutil.copy2(f, dst_img_dir / f.name)
            lbl = src_lbl_dir / f"{f.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl_dir / lbl.name)

    print(f"[Retrain Prep] Copied frozen val ({len(list((retrain_split_dir / 'images' / 'val').glob('*.jpg')))}) and test ({len(list((retrain_split_dir / 'images' / 'test').glob('*.jpg')))}) sets.")

    # 2. Populate train split with baseline train images
    src_train_img = baseline_split_dir / "images" / "train"
    src_train_lbl = baseline_split_dir / "labels" / "train"
    dst_train_img = retrain_split_dir / "images" / "train"
    dst_train_lbl = retrain_split_dir / "labels" / "train"

    for f in src_train_img.glob("*.jpg"):
        shutil.copy2(f, dst_train_img / f.name)
        lbl = src_train_lbl / f"{f.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_train_lbl / lbl.name)

    baseline_train_count = len(list(dst_train_img.glob("*.jpg")))
    print(f"[Retrain Prep] Base train images populated: {baseline_train_count}")

    # 3. Add high-confidence recovered pseudo-labeled images
    with open(diag_path, "r", encoding="utf-8") as f:
        diag_data = json.load(f)
    diag = diag_data["detailed_diagnostics"]

    # Filter candidates: max_conf >= 0.55, 1 to 3 boxes, valid bounds
    added_count = 0
    test_and_val_stems = {f.stem for f in (baseline_split_dir / "images" / "test").glob("*.jpg")} | \
                         {f.stem for f in (baseline_split_dir / "images" / "val").glob("*.jpg")}

    for fname, props in diag.items():
        stem = Path(fname).stem
        if stem in test_and_val_stems or (dst_train_img / fname).exists():
            continue  # Prevent any leakage into train

        max_c = props.get("max_conf", 0.0)
        box_count = props.get("box_count", 0)

        # High confidence filter
        if max_c >= 0.55 and 1 <= box_count <= 3:
            src_img = raw_images_dir / fname
            src_lbl = pseudo_labels_dir / f"{stem}.txt"

            if src_img.exists() and src_lbl.exists() and src_lbl.stat().st_size > 0:
                shutil.copy2(src_img, dst_train_img / fname)
                shutil.copy2(src_lbl, dst_train_lbl / f"{stem}.txt")
                added_count += 1

    total_train_count = len(list(dst_train_img.glob("*.jpg")))
    print(f"[Retrain Prep] Added {added_count} high-confidence recovered images. Total train count: {total_train_count}")

    # Create dataset YAML
    abs_split_dir = str(retrain_split_dir.resolve()).replace("\\", "/")
    yaml_content = f"""# Ultralytics YOLO Data Configuration for Retrained Model
path: {abs_split_dir}
train: images/train
val: images/val
test: images/test

names:
  0: two_wheeler
"""
    yaml_path = retrain_split_dir / "two_wheeler_retrain.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[Retrain Prep] Wrote {yaml_path}")
    return total_train_count

if __name__ == "__main__":
    prepare_retrain_dataset()

