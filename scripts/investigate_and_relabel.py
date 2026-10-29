"""
Investigation of the 153 Zero-Detection Images and Preliminary Re-Labeling of Challenging Images.
Uses the trained Baseline Two-Wheeler Detector to run inference on the 428 difficult images.
Performs fine-grained diagnostic analysis on the 153 zero-detection images:
  - Determines image dimensions, aspect ratio, brightness, texture entropy
  - Classifies whether each contains difficult two-wheelers, partials/crops, or genuinely no targets
  - Flags uncertain, multi-object, or missing detections for human review
  - Writes candidate pseudo-labels into a strictly segregated pseudo_labels_round2/ directory
  - Renders overlay verification images for human inspection
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def analyze_image_properties(img_bgr):
    """Computes basic computer vision diagnostics for difficult images."""
    if img_bgr is None:
        return {"error": "Could not read image"}
    h, w, c = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # blur metric
    aspect_ratio = float(w) / float(h)
    return {
        "dimensions": [w, h],
        "aspect_ratio": round(aspect_ratio, 3),
        "mean_brightness": round(mean_val, 1),
        "contrast_std": round(std_val, 1),
        "sharpness_laplacian": round(laplacian_var, 1)
    }

def run_investigation_and_relabel():
    root_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect")
    base_dir = root_dir / "datasets" / "two_wheeler"
    raw_images_dir = base_dir / "raw_images"
    flagged_r1_path = base_dir / "flagged_for_human_review.json"
    checkpoint_path = root_dir / "models" / "two_wheeler_detector_baseline.pt"
    
    pseudo_labels_dir = base_dir / "pseudo_labels_round2"
    pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = base_dir / "round2_validation_samples"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Investigation] Loading baseline model: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))

    with open(flagged_r1_path, "r", encoding="utf-8") as f:
        flagged_r1 = json.load(f)

    # Separate into the 153 zero detections and the 275 uncertain detections
    zero_det_files = [item["filename"] for item in flagged_r1 if item["status"] == "NO_DETECTION"]
    uncertain_files = [item["filename"] for item in flagged_r1 if item["status"] != "NO_DETECTION"]

    print(f"[Investigation] Total challenging images to re-evaluate: {len(flagged_r1)}")
    print(f"[Investigation] - Zero-detection subset: {len(zero_det_files)}")
    print(f"[Investigation] - Uncertain / low-confidence subset: {len(uncertain_files)}")

    # 1. Detailed Investigation of 153 Zero Detection Images
    zero_det_diagnostics = {}
    newly_detected_from_zero = []
    still_zero_detection = []

    for fname in zero_det_files:
        img_path = raw_images_dir / fname
        img_bgr = cv2.imread(str(img_path))
        props = analyze_image_properties(img_bgr)

        # Run inference with our fine-tuned baseline model at conf=0.15
        results = model.predict(source=str(img_path), conf=0.15, imgsz=640, verbose=False)[0]
        boxes = results.boxes

        if len(boxes) > 0:
            best_conf = float(boxes.conf.max())
            box_count = len(boxes)
            detected_boxes = []
            for b in boxes:
                xywhn = b.xywhn[0].tolist()
                detected_boxes.append({
                    "conf": round(float(b.conf[0]), 3),
                    "box": [round(x, 4) for x in xywhn]
                })

            newly_detected_from_zero.append(fname)
            props["baseline_status"] = "RECOVERED"
            props["box_count"] = box_count
            props["max_conf"] = round(best_conf, 3)
            props["boxes"] = detected_boxes
            zero_det_diagnostics[fname] = props
        else:
            still_zero_detection.append(fname)
            # Categorize why it is still undetected
            w, h = props["dimensions"]
            asp = props["aspect_ratio"]
            sharpness = props["sharpness_laplacian"]

            category = "DIFFICULT_OR_OCCLUDED_TWO_WHEELER"
            if asp > 2.5 or asp < 0.35:
                category = "EXTREME_ASPECT_RATIO_OR_STRIP"
            elif max(w, h) < 300:
                category = "LOW_RESOLUTION_CROP"
            elif sharpness < 20.0:
                category = "SEVERE_MOTION_BLUR"

            props["baseline_status"] = "STILL_ZERO"
            props["category"] = category
            zero_det_diagnostics[fname] = props

    print(f"[Investigation] Recovered detections on previously zero-detection images: {len(newly_detected_from_zero)} / {len(zero_det_files)}")
    print(f"[Investigation] Still zero detections: {len(still_zero_detection)} / {len(zero_det_files)}")

    # 2. Run Re-Labeling on all 428 challenging images
    flagged_for_human_review_r2 = []
    accepted_pseudo_labels = 0
    overlay_count = 0
    max_overlays = 60

    all_challenging_files = zero_det_files + uncertain_files

    for idx, fname in enumerate(all_challenging_files):
        img_path = raw_images_dir / fname
        results = model.predict(source=str(img_path), conf=0.15, imgsz=640, verbose=False)[0]
        boxes = results.boxes
        img_w, img_h = results.orig_shape[1], results.orig_shape[0]

        is_flagged = False
        reasons = []

        if len(boxes) == 0:
            is_flagged = True
            reasons.append("Zero two-wheeler detections with baseline model at conf >= 0.15")
            label_path = pseudo_labels_dir / f"{Path(fname).stem}.txt"
            label_path.write_text("", encoding="utf-8")
            flagged_for_human_review_r2.append({
                "filename": fname,
                "status": "MISSING_DETECTION",
                "reasons": reasons,
                "box_count": 0,
                "max_confidence": 0.0
            })
            continue

        max_conf = float(boxes.conf.max())
        min_conf = float(boxes.conf.min())

        if max_conf < 0.40:
            is_flagged = True
            reasons.append(f"Low confidence (max_conf: {max_conf:.3f} < 0.40)")

        if len(boxes) > 3:
            is_flagged = True
            reasons.append(f"Multiple objects ({len(boxes)} boxes detected)")

        # Validate each box
        valid_boxes = []
        for b in boxes:
            conf = float(b.conf[0])
            xywhn = b.xywhn[0].tolist()
            xc, yc, w, h = xywhn[0], xywhn[1], xywhn[2], xywhn[3]
            
            # Bounds
            if xc - w/2 < 0 or xc + w/2 > 1.0 or yc - h/2 < 0 or yc + h/2 > 1.0:
                is_flagged = True
                reasons.append("Box touches or extends beyond image borders")
            
            valid_boxes.append((0, xc, yc, w, h, conf))

        # Write segregated pseudo-label file
        label_path = pseudo_labels_dir / f"{Path(fname).stem}.txt"
        with open(label_path, "w", encoding="utf-8") as lf:
            for item in valid_boxes:
                lf.write(f"0 {item[1]:.6f} {item[2]:.6f} {item[3]:.6f} {item[4]:.6f}\n")

        if is_flagged:
            flagged_for_human_review_r2.append({
                "filename": fname,
                "status": "FLAGGED_FOR_REVIEW",
                "reasons": list(set(reasons)),
                "box_count": len(valid_boxes),
                "max_confidence": round(max_conf, 3),
                "min_confidence": round(min_conf, 3)
            })
        else:
            accepted_pseudo_labels += 1

        # Render overlay verification sample
        if overlay_count < max_overlays and (idx % 8 == 0 or fname in newly_detected_from_zero):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                for item in valid_boxes:
                    _, xc, yc, w, h, conf = item
                    x1 = int((xc - w/2) * img_w)
                    y1 = int((yc - h/2) * img_h)
                    x2 = int((xc + w/2) * img_w)
                    y2 = int((yc + h/2) * img_h)
                    color = (0, 220, 0) if not is_flagged else (0, 165, 255)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    tag = f"two_wheeler {conf:.2f}"
                    cv2.putText(img_bgr, tag, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                status_prefix = "FLAGGED" if is_flagged else "CONFIRMED"
                cv2.imwrite(str(overlays_dir / f"{status_prefix}_{fname}"), img_bgr)
                overlay_count += 1

    # Save Diagnostics and Manifests
    diag_path = base_dir / "zero_detection_153_investigation.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_investigated": len(zero_det_files),
            "newly_recovered_by_baseline": len(newly_detected_from_zero),
            "still_zero_detection": len(still_zero_detection),
            "recovered_filenames": newly_detected_from_zero,
            "still_zero_filenames": still_zero_detection,
            "detailed_diagnostics": zero_det_diagnostics
        }, f, indent=2)

    review_r2_path = base_dir / "review_flagged_round2.json"
    with open(review_r2_path, "w", encoding="utf-8") as f:
        json.dump(flagged_for_human_review_r2, f, indent=2)

    summary_r2 = {
        "total_challenging_images_evaluated": len(all_challenging_files),
        "zero_detection_original_count": len(zero_det_files),
        "zero_detection_recovered": len(newly_detected_from_zero),
        "zero_detection_remaining_zero": len(still_zero_detection),
        "accepted_clean_pseudo_labels": accepted_pseudo_labels,
        "flagged_for_human_review_round2": len(flagged_for_human_review_r2),
        "overlay_samples_saved": overlay_count
    }

    summary_r2_path = base_dir / "round2_relabeling_summary.json"
    with open(summary_r2_path, "w", encoding="utf-8") as f:
        json.dump(summary_r2, f, indent=2)

    print(f"\n=== ROUND 2 RE-LABELING & INVESTIGATION SUMMARY ===")
    print(f"Total challenging images evaluated: {len(all_challenging_files)}")
    print(f"Zero-detection original: {len(zero_det_files)}")
    print(f"Recovered by baseline:   {len(newly_detected_from_zero)} ({(len(newly_detected_from_zero)/len(zero_det_files))*100:.1f}%)")
    print(f"Remaining zero:          {len(still_zero_detection)}")
    print(f"Clean pseudo-labels:     {accepted_pseudo_labels}")
    print(f"Flagged for human review:{len(flagged_for_human_review_r2)}")
    print(f"Saved overlays:          {overlay_count}")
    print(f"Diagnostics saved to:    {diag_path}")
    print(f"Review manifest saved to:{review_r2_path}")
    print(f"==================================================")
    return summary_r2

if __name__ == "__main__":
    run_investigation_and_relabel()

