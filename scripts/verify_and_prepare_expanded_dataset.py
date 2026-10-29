"""
Verification and Leakage Prevention Script for Expanded Two-Wheeler Training.
Validates bounding-box coordinates, class IDs, image-label pairing, and performs
strict cross-split leakage checks (hashes, duplicate groups, and sequence proximity)
against the frozen validation and test sets.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from PIL import Image

def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_and_prepare_dataset():
    root_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect")
    base_dir = root_dir / "datasets" / "two_wheeler"
    raw_images_dir = base_dir / "raw_images"
    curated_labels_dir = base_dir / "clean_curated_pseudo_labels"
    curation_manifest_path = base_dir / "clean_curation_manifest.json"
    audit_report_path = base_dir / "duplicates_and_dataset_audit_report.json"
    
    baseline_split_dir = base_dir / "splits" / "baseline"
    retrain_split_dir = base_dir / "splits" / "expanded_retrain"

    print("=== STEP 1: LOAD FROZEN VAL & TEST SETS ===")
    val_images = sorted([f.name for f in (baseline_split_dir / "images" / "val").glob("*.jpg")])
    test_images = sorted([f.name for f in (baseline_split_dir / "images" / "test").glob("*.jpg")])
    base_train_images = sorted([f.name for f in (baseline_split_dir / "images" / "train").glob("*.jpg")])

    print(f"Frozen Val Images:  {len(val_images)}")
    print(f"Frozen Test Images: {len(test_images)}")
    print(f"Base Train Images:  {len(base_train_images)}")

    # Compute hashes of val and test images
    val_hashes = {compute_md5(baseline_split_dir / "images" / "val" / f): f for f in val_images}
    test_hashes = {compute_md5(baseline_split_dir / "images" / "test" / f): f for f in test_images}

    # Load duplicate groups from audit
    with open(audit_report_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    dup_groups = audit_data.get("duplicate_groups", {})

    # Build blacklisted hash/name sets from val and test
    eval_hashes = set(val_hashes.keys()) | set(test_hashes.keys())
    eval_names = set(val_images) | set(test_images)

    # Sequence clusters from val & test
    eval_numeric_ids = set()
    for name in eval_names:
        stem = Path(name).stem
        if stem.isdigit():
            eval_numeric_ids.add(int(stem))

    print(f"Blacklisted eval names: {len(eval_names)}")

    print("\n=== STEP 2: VERIFY AND FILTER CANDIDATE RECOVERED IMAGES ===")
    with open(curation_manifest_path, "r", encoding="utf-8") as f:
        curation_manifest = json.load(f)

    # Filter only CLEAN_HIGH_CONF candidates
    clean_candidates = [item for item in curation_manifest if item["status"] == "CLEAN_HIGH_CONF"]
    print(f"Total CLEAN_HIGH_CONF candidates in manifest: {len(clean_candidates)}")

    accepted_images = []
    excluded_images = []

    for item in clean_candidates:
        fname = item["filename"]
        img_path = raw_images_dir / fname
        lbl_path = curated_labels_dir / f"{Path(fname).stem}.txt"

        # Check 1: Existence
        if not img_path.exists():
            excluded_images.append({"filename": fname, "reason": "Image file does not exist"})
            continue
        if not lbl_path.exists():
            excluded_images.append({"filename": fname, "reason": "Label file does not exist"})
            continue

        # Check 2: Already in base train
        if fname in base_train_images:
            excluded_images.append({"filename": fname, "reason": "Already in baseline training set"})
            continue

        # Check 3: Data leakage check against val/test
        if fname in eval_names:
            excluded_images.append({"filename": fname, "reason": "LEAKAGE: Filename matches frozen val/test set"})
            continue

        img_hash = compute_md5(img_path)
        if img_hash in eval_hashes:
            excluded_images.append({"filename": fname, "reason": "LEAKAGE: Exact hash matches frozen val/test set"})
            continue

        # Check 4: Check if image is an identified duplicate of any val/test image
        is_dup_of_eval = False
        for h, grp in dup_groups.items():
            grp_members = set([grp["canonical"]] + grp.get("duplicate_names", []))
            if fname in grp_members and any(en in grp_members for en in eval_names):
                is_dup_of_eval = True
                excluded_images.append({"filename": fname, "reason": f"LEAKAGE: Duplicate group member of eval image in group {grp['canonical']}"})
                break
        if is_dup_of_eval:
            continue

        # Check 5: Sequence proximity check (avoid adjacent frame burst within +-1 of test set)
        stem = Path(fname).stem
        if stem.isdigit():
            num_id = int(stem)
            if any(abs(num_id - tid) <= 1 for tid in eval_numeric_ids):
                excluded_images.append({"filename": fname, "reason": f"LEAKAGE_RISK: Frame burst proximity (+-1) to eval image"})
                continue

        # Check 6: Image readability and dimensions
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                if w < 10 or h < 10:
                    excluded_images.append({"filename": fname, "reason": "Degenerate image resolution"})
                    continue
        except Exception as e:
            excluded_images.append({"filename": fname, "reason": f"Corrupt image file: {e}"})
            continue

        # Check 7: Label contents & bounding-box normalization validity
        content = lbl_path.read_text(encoding="utf-8").strip()
        if not content:
            excluded_images.append({"filename": fname, "reason": "Empty label file"})
            continue

        lines = content.split("\n")
        valid_boxes = []
        box_valid = True

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                box_valid = False
                excluded_images.append({"filename": fname, "reason": f"Invalid YOLO format (expected 5 tokens, got {len(parts)})"})
                break
            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                box_valid = False
                excluded_images.append({"filename": fname, "reason": "Non-numeric values in label"})
                break

            if cls_id != 0:
                box_valid = False
                excluded_images.append({"filename": fname, "reason": f"Incorrect class ID {cls_id} (expected 0)"})
                break

            # Geometry checks
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                box_valid = False
                excluded_images.append({"filename": fname, "reason": f"Box coordinates out of bounds: {parts}"})
                break

            if bw < 0.005 or bh < 0.005 or (bw * bh) < 0.005:
                box_valid = False
                excluded_images.append({"filename": fname, "reason": f"Degenerate box size: w={bw}, h={bh}"})
                break

            valid_boxes.append((cls_id, xc, yc, bw, bh))

        if not box_valid or not valid_boxes:
            continue

        accepted_images.append({
            "filename": fname,
            "hash": img_hash,
            "dimensions": [w, h],
            "box_count": len(valid_boxes),
            "max_conf": item.get("max_conf", 0.0),
            "boxes": valid_boxes
        })

    print(f"\n[Verification Summary]")
    print(f"Accepted clean recovered images: {len(accepted_images)}")
    print(f"Excluded images:                {len(excluded_images)}")

    # Save exclusion report
    exclusion_report_path = base_dir / "excluded_from_expanded_training.json"
    with open(exclusion_report_path, "w", encoding="utf-8") as f:
        json.dump(excluded_images, f, indent=2)
    print(f"Exclusion report saved to {exclusion_report_path}")

    print("\n=== STEP 3: STRUCTURE EXPANDED RETRAIN DATASET ===")
    for split_name in ["train", "val", "test"]:
        (retrain_split_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (retrain_split_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # Copy frozen val and test sets
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

    # Copy base train
    dst_train_img = retrain_split_dir / "images" / "train"
    dst_train_lbl = retrain_split_dir / "labels" / "train"
    src_train_img = baseline_split_dir / "images" / "train"
    src_train_lbl = baseline_split_dir / "labels" / "train"

    for f in src_train_img.glob("*.jpg"):
        shutil.copy2(f, dst_train_img / f.name)
        lbl = src_train_lbl / f"{f.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_train_lbl / lbl.name)

    # Add verified accepted recovered images
    for item in accepted_images:
        fname = item["filename"]
        shutil.copy2(raw_images_dir / fname, dst_train_img / fname)
        shutil.copy2(curated_labels_dir / f"{Path(fname).stem}.txt", dst_train_lbl / f"{Path(fname).stem}.txt")

    final_train_count = len(list(dst_train_img.glob("*.jpg")))
    final_val_count = len(list((retrain_split_dir / "images" / "val").glob("*.jpg")))
    final_test_count = len(list((retrain_split_dir / "images" / "test").glob("*.jpg")))

    print(f"\nFinal Dataset Counts in Expanded Retrain Split:")
    print(f"  Train: {final_train_count} images (183 baseline + {len(accepted_images)} verified additions)")
    print(f"  Val:   {final_val_count} images (frozen)")
    print(f"  Test:  {final_test_count} images (frozen)")

    # Assert test set is 100% identical to baseline test set
    base_test_files = sorted([f.name for f in (baseline_split_dir / "images" / "test").glob("*.jpg")])
    retrain_test_files = sorted([f.name for f in (retrain_split_dir / "images" / "test").glob("*.jpg")])
    assert base_test_files == retrain_test_files, "CRITICAL ERROR: Test set files do not match baseline!"
    for tf in base_test_files:
        assert compute_md5(baseline_split_dir / "images" / "test" / tf) == compute_md5(retrain_split_dir / "images" / "test" / tf), f"Test image {tf} altered!"
        assert (baseline_split_dir / "labels" / "test" / f"{Path(tf).stem}.txt").read_text() == (retrain_split_dir / "labels" / "test" / f"{Path(tf).stem}.txt").read_text(), f"Test label for {tf} altered!"
    print("[Verification PASS] Held-out test set is 100% identical byte-for-byte to baseline.")

    # Write data YAML
    abs_split_dir = str(retrain_split_dir.resolve()).replace("\\", "/")
    yaml_content = f"""# Ultralytics YOLO Data Configuration for Retrained Two-Wheeler Model
path: {abs_split_dir}
train: images/train
val: images/val
test: images/test

names:
  0: two_wheeler
"""
    yaml_path = retrain_split_dir / "two_wheeler_retrain.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Wrote retrain YAML to {yaml_path}")

    return {
        "final_train_count": final_train_count,
        "final_val_count": final_val_count,
        "final_test_count": final_test_count,
        "accepted_additions": len(accepted_images),
        "excluded_count": len(excluded_images)
    }

if __name__ == "__main__":
    verify_and_prepare_dataset()

