"""
Dataset Splitter for Baseline Two-Wheeler Detector.
Takes the 227 confirmed high-confidence images, prevents temporal/burst leakage,
and creates an 80/10/10 train/val/test split with standard YOLO directory structure.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path

def split_baseline_dataset(
    base_dataset_dir: Path,
    summary_json_path: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
):
    base_dataset_dir = Path(base_dataset_dir)
    raw_images_dir = base_dataset_dir / "raw_images"
    raw_labels_dir = base_dataset_dir / "raw_labels"
    
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # Load list of confirmed high confidence filenames
    pseudo_summary_path = base_dataset_dir / "pseudo_labeling_summary.json"
    flagged_path = base_dataset_dir / "flagged_for_human_review.json"
    
    with open(flagged_path, "r", encoding="utf-8") as f:
        flagged_list = json.load(f)
    flagged_set = {item["filename"] for item in flagged_list}

    all_images = sorted([f.name for f in raw_images_dir.glob("*.jpg")])
    confirmed_images = [fname for fname in all_images if fname not in flagged_set]
    
    print(f"[Splitter] Total raw images: {len(all_images)}")
    print(f"[Splitter] Confirmed high-confidence images: {len(confirmed_images)}")

    # Sort numerically/lexicographically to cluster bursts
    def sort_key(fname):
        stem = Path(fname).stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    
    sorted_confirmed = sorted(confirmed_images, key=sort_key)

    # Group into sequence chunks to prevent frame-burst leakage across splits
    # Images with adjacent IDs (e.g. 101, 102) likely came from same sequence/burst
    chunks = []
    current_chunk = []
    last_id = None

    for fname in sorted_confirmed:
        stem = Path(fname).stem
        if stem.isdigit():
            curr_id = int(stem)
            if last_id is not None and (curr_id == last_id + 1):
                current_chunk.append(fname)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [fname]
            last_id = curr_id
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            chunks.append([fname])
            last_id = None
    if current_chunk:
        chunks.append(current_chunk)

    print(f"[Splitter] Grouped {len(confirmed_images)} images into {len(chunks)} sequence clusters.")

    # Shuffle chunks reproducibly
    rng = random.Random(seed)
    rng.shuffle(chunks)

    # Distribute chunk by chunk to maintain 80/10/10 target counts
    total_imgs = len(confirmed_images)
    target_val = int(total_imgs * val_ratio)
    target_test = int(total_imgs * test_ratio)

    val_imgs = []
    test_imgs = []
    train_imgs = []

    for chunk in chunks:
        if len(test_imgs) < target_test:
            test_imgs.extend(chunk)
        elif len(val_imgs) < target_val:
            val_imgs.extend(chunk)
        else:
            train_imgs.extend(chunk)

    print(f"[Splitter] Allocated: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    # Create destination directories
    split_dir = base_dataset_dir / "splits" / "baseline"
    for split_name in ["train", "val", "test"]:
        (split_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (split_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    def populate_split(file_list, split_name):
        for fname in file_list:
            src_img = raw_images_dir / fname
            src_lbl = raw_labels_dir / f"{Path(fname).stem}.txt"
            
            dst_img = split_dir / "images" / split_name / fname
            dst_lbl = split_dir / "labels" / split_name / f"{Path(fname).stem}.txt"

            shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

    populate_split(train_imgs, "train")
    populate_split(val_imgs, "val")
    populate_split(test_imgs, "test")

    # Generate YOLO data config
    data_yaml_path = split_dir / "two_wheeler_baseline.yaml"
    # Use forward slashes for Windows path in yaml
    abs_split_dir = str(split_dir.resolve()).replace("\\", "/")
    yaml_content = f"""# Ultralytics YOLO Data Configuration for Two-Wheeler Baseline
path: {abs_split_dir}
train: images/train
val: images/val
test: images/test

names:
  0: two_wheeler
"""
    data_yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[Splitter] Wrote dataset YAML configuration to {data_yaml_path}")

    # Save split manifest
    manifest = {
        "total_confirmed_images": total_imgs,
        "seed": seed,
        "counts": {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        },
        "train_images": train_imgs,
        "val_images": val_imgs,
        "test_images": test_imgs
    }
    manifest_path = split_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Splitter] Saved split manifest to {manifest_path}")
    return manifest

if __name__ == "__main__":
    base = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect\datasets\two_wheeler")
    summary = base / "duplicates_and_dataset_audit_report.json"
    split_baseline_dataset(base, summary)

