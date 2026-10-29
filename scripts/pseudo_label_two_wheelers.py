"""
Pseudo-Labeling and Bounding-Box Validation for Two-Wheeler Dataset.
Uses pretrained YOLOv8 model to detect two-wheelers, remaps to class 0 (two_wheeler),
validates geometry and normalization, flags low-confidence or degenerate annotations,
and generates human-review manifests and visual overlay samples.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, val))

def validate_box(x_center, y_center, width, height):
    """
    Validates normalized YOLO bounding box.
    Returns (is_valid, reason, clamped_box)
    """
    # Check for NaN or Inf
    if not (np.isfinite(x_center) and np.isfinite(y_center) and np.isfinite(width) and np.isfinite(height)):
        return False, "Non-finite box coordinates", None

    # Calculate edges
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    # Degenerate dimension checks
    if width <= 0.005 or height <= 0.005:
        return False, f"Degenerate dimension (w={width:.4f}, h={height:.4f})", None

    area = width * height
    if area < 0.005:
        return False, f"Area too small ({area:.5f})", None

    # Clamp edges within [0, 1]
    cx1 = clamp(x1, 0.0, 1.0)
    cy1 = clamp(y1, 0.0, 1.0)
    cx2 = clamp(x2, 0.0, 1.0)
    cy2 = clamp(y2, 0.0, 1.0)

    c_width = cx2 - cx1
    c_height = cy2 - cy1
    c_xc = cx1 + c_width / 2.0
    c_yc = cy1 + c_height / 2.0

    if c_width <= 0.005 or c_height <= 0.005:
        return False, "Box outside image boundaries", None

    # Aspect ratio check
    aspect_ratio = c_width / c_height
    if aspect_ratio < 0.15 or aspect_ratio > 6.0:
        return True, f"Unusual aspect ratio ({aspect_ratio:.2f})", (c_xc, c_yc, c_width, c_height)

    return True, "Valid", (c_xc, c_yc, c_width, c_height)

def run_pseudo_labeling(
    raw_images_dir: Path,
    output_labels_dir: Path,
    summary_output_dir: Path,
    conf_threshold: float = 0.15,
    high_conf_threshold: float = 0.50
):
    raw_images_dir = Path(raw_images_dir)
    output_labels_dir = Path(output_labels_dir)
    summary_output_dir = Path(summary_output_dir)
    
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = summary_output_dir / "label_validation_samples"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pseudo-Labeling] Loading YOLOv8 base model...")
    model = YOLO("yolov8n.pt")

    # Map COCO classes to two_wheeler (COCO 3: motorcycle, 1: bicycle)
    target_classes = {3: "motorcycle", 1: "bicycle"}

    image_files = sorted(
        [f for f in raw_images_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda x: (0, int(x.stem)) if x.stem.isdigit() else (1, str(x.stem))
    )
    print(f"[Pseudo-Labeling] Processing {len(image_files)} raw images...")

    categorized_images = {
        "CONFIRMED_HIGH_CONF": [],
        "UNCERTAIN_LOW_CONF": [],
        "NO_DETECTION": [],
        "DEGENERATE_BOX": []
    }

    all_detections_record = {}
    flagged_for_human_review = []
    
    saved_overlays = 0
    max_overlays = 50

    for idx, img_path in enumerate(image_files):
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=0.45,
            classes=[1, 3],  # bicycle and motorcycle
            verbose=False
        )

        r = results[0]
        boxes = r.boxes
        img_w, img_h = r.orig_shape[1], r.orig_shape[0]

        valid_boxes_for_label = []
        image_flagged = False
        flag_reasons = []

        if len(boxes) == 0:
            categorized_images["NO_DETECTION"].append(img_path.name)
            flagged_for_human_review.append({
                "filename": img_path.name,
                "status": "NO_DETECTION",
                "reasons": ["Pretrained model detected no two-wheeler at conf >= 0.15"],
                "box_count": 0,
                "max_confidence": 0.0
            })
            # Save empty label file
            label_file = output_labels_dir / f"{img_path.stem}.txt"
            label_file.write_text("", encoding="utf-8")
            continue

        max_conf = 0.0
        min_conf = 1.0

        for b in boxes:
            cls_id = int(b.cls[0])
            cls_name = target_classes.get(cls_id, model.names.get(cls_id, "unknown"))
            conf = float(b.conf[0])
            max_conf = max(max_conf, conf)
            min_conf = min(min_conf, conf)

            # Get normalized xywh
            xywhn = b.xywhn[0].tolist()
            xc, yc, w, h = xywhn[0], xywhn[1], xywhn[2], xywhn[3]

            is_valid, reason, clamped_box = validate_box(xc, yc, w, h)
            if not is_valid:
                image_flagged = True
                flag_reasons.append(f"Box rejected: {reason}")
                categorized_images["DEGENERATE_BOX"].append(img_path.name)
                continue

            c_xc, c_yc, c_w, c_h = clamped_box
            if conf < high_conf_threshold:
                image_flagged = True
                flag_reasons.append(f"Low confidence detection ({cls_name}: {conf:.3f})")

            # We map class to 0 (two_wheeler)
            valid_boxes_for_label.append({
                "class_id": 0,
                "class_name": "two_wheeler",
                "coco_class": cls_name,
                "confidence": conf,
                "box": [c_xc, c_yc, c_w, c_h],
                "box_pixel": [
                    int((c_xc - c_w/2)*img_w),
                    int((c_yc - c_h/2)*img_h),
                    int((c_xc + c_w/2)*img_w),
                    int((c_yc + c_h/2)*img_h)
                ]
            })

        # Write YOLO label file
        label_file = output_labels_dir / f"{img_path.stem}.txt"
        with open(label_file, "w", encoding="utf-8") as lf:
            for item in valid_boxes_for_label:
                b = item["box"]
                lf.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

        all_detections_record[img_path.name] = {
            "image_shape": [img_h, img_w],
            "detections_count": len(valid_boxes_for_label),
            "max_confidence": max_conf,
            "min_confidence": min_conf if len(valid_boxes_for_label) > 0 else 0.0,
            "detections": valid_boxes_for_label
        }

        if image_flagged or min_conf < high_conf_threshold:
            categorized_images["UNCERTAIN_LOW_CONF"].append(img_path.name)
            flagged_for_human_review.append({
                "filename": img_path.name,
                "status": "UNCERTAIN_LOW_CONF",
                "reasons": flag_reasons,
                "box_count": len(valid_boxes_for_label),
                "max_confidence": max_conf
            })
        else:
            categorized_images["CONFIRMED_HIGH_CONF"].append(img_path.name)

        # Generate overlay sample for visual review (select both confident and uncertain)
        if saved_overlays < max_overlays and (idx % 13 == 0 or image_flagged):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                for item in valid_boxes_for_label:
                    x1, y1, x2, y2 = item["box_pixel"]
                    conf = item["confidence"]
                    color = (0, 220, 0) if conf >= high_conf_threshold else (0, 215, 255)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    tag = f"two_wheeler {conf:.2f}"
                    cv2.putText(img_bgr, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                status_str = "REVIEW" if image_flagged else "PASS"
                out_overlay_path = overlays_dir / f"{status_str}_{img_path.name}"
                cv2.imwrite(str(out_overlay_path), img_bgr)
                saved_overlays += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
            print(f"[Pseudo-Labeling] Processed {idx + 1}/{len(image_files)} images...")

    # Save summary and review files
    review_path = summary_output_dir / "flagged_for_human_review.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(flagged_for_human_review, f, indent=2)

    summary = {
        "total_images_processed": len(image_files),
        "total_labels_generated": len(list(output_labels_dir.glob("*.txt"))),
        "confirmed_high_confidence_count": len(categorized_images["CONFIRMED_HIGH_CONF"]),
        "uncertain_low_confidence_count": len(categorized_images["UNCERTAIN_LOW_CONF"]),
        "no_detection_count": len(categorized_images["NO_DETECTION"]),
        "degenerate_box_count": len(categorized_images["DEGENERATE_BOX"]),
        "flagged_for_review_total": len(flagged_for_human_review),
        "overlay_samples_saved": saved_overlays,
        "high_confidence_rate": f"{(len(categorized_images['CONFIRMED_HIGH_CONF']) / len(image_files)) * 100:.1f}%"
    }

    summary_path = summary_output_dir / "pseudo_labeling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown Report
    md_path = summary_output_dir / "LABELING_AND_VALIDATION_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Two-Wheeler Preliminary Pseudo-Labeling & Bounding-Box Validation Report\n\n")
        f.write(f"- **Total Images Evaluated:** {summary['total_images_processed']}\n")
        f.write(f"- **Confident High-Quality Annotations:** {summary['confirmed_high_confidence_count']} ({summary['high_confidence_rate']})\n")
        f.write(f"- **Uncertain / Low Confidence Labels (conf < 0.50):** {summary['uncertain_low_confidence_count']}\n")
        f.write(f"- **Zero Detections (Need Manual Review or Specialized Anchor):** {summary['no_detection_count']}\n")
        f.write(f"- **Degenerate Boxes Prevented/Rejected:** {summary['degenerate_box_count']}\n")
        f.write(f"- **Total Flagged for Human Review:** {summary['flagged_for_review_total']}\n")
        f.write(f"- **Visual Verification Overlays Saved:** {summary['overlay_samples_saved']} images in `label_validation_samples/`\n\n")
        f.write("## Validation Rules Enforced\n")
        f.write("1. **Coordinate Normalization:** All boxes guaranteed $0.0 \\le x, y, w, h \\le 1.0$.\n")
        f.write("2. **Degenerate Filter:** Discarded boxes with area $< 0.005$ or dimension $< 0.005$.\n")
        f.write("3. **Class Normalization:** Pretrained classes `motorcycle` (3) and `bicycle` (1) mapped to uniform `class_id 0` (`two_wheeler`).\n")
        f.write("4. **Isolation:** Flagged uncertain images kept in `flagged_for_human_review.json` so they can be isolated or human-verified prior to training.\n")

    print(f"[Pseudo-Labeling] Finished. Summary saved to {summary_path}")
    print(f"[Pseudo-Labeling] Review list saved to {review_path}")
    print(f"[Pseudo-Labeling] Report saved to {md_path}")
    return summary

if __name__ == "__main__":
    raw_imgs = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect\datasets\two_wheeler\raw_images")
    out_lbls = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect\datasets\two_wheeler\raw_labels")
    base_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect\datasets\two_wheeler")
    run_pseudo_labeling(raw_imgs, out_lbls, base_dir)
