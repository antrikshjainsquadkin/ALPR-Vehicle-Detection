"""
Detailed Regression Check for Baseline vs Updated Two-Wheeler Detector.
Compares predictions on the held-out test set:
  - Counts true positives, false positives, false negatives
  - Analyzes confidence distribution and box tightness
  - Generates side-by-side visual comparison images (Baseline vs Updated)
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

def run_regression_analysis():
    root_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect")
    test_img_dir = root_dir / "datasets" / "two_wheeler" / "splits" / "baseline" / "images" / "test"
    test_lbl_dir = root_dir / "datasets" / "two_wheeler" / "splits" / "baseline" / "labels" / "test"
    
    baseline_chk = root_dir / "models" / "two_wheeler_detector_baseline.pt"
    v2_chk = root_dir / "models" / "two_wheeler_detector_v2.pt"

    out_vis_dir = root_dir / "runs" / "detect" / "two_wheeler" / "side_by_side_visual_comparison"
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    print("[Regression Check] Loading models...")
    model_base = YOLO(str(baseline_chk))
    model_v2 = YOLO(str(v2_chk))

    test_files = sorted(list(test_img_dir.glob("*.jpg")))
    print(f"[Regression Check] Evaluating {len(test_files)} test images...")

    results_per_image = []

    base_total_fp = 0
    base_total_fn = 0
    base_total_tp = 0

    v2_total_fp = 0
    v2_total_fn = 0
    v2_total_tp = 0

    for f in test_files:
        img_bgr = cv2.imread(str(f))
        h, w = img_bgr.shape[:2]

        # Load ground truth
        gt_boxes = []
        lbl_f = test_lbl_dir / f"{f.stem}.txt"
        if lbl_f.exists() and lbl_f.stat().st_size > 0:
            for line in lbl_f.read_text().strip().split("\n"):
                parts = line.split()
                if len(parts) == 5:
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    gt_boxes.append([x1, y1, x2, y2])

        # Run predictions
        pred_base = model_base.predict(str(f), conf=0.35, imgsz=640, verbose=False)[0].boxes
        pred_v2 = model_v2.predict(str(f), conf=0.35, imgsz=640, verbose=False)[0].boxes

        def eval_boxes(pred_boxes, gt_boxes):
            tp = 0
            fp = 0
            matched_gt = set()
            p_boxes_xyxy = []
            for b in pred_boxes:
                xyxy = [int(x) for x in b.xyxy[0].tolist()]
                conf = float(b.conf[0])
                p_boxes_xyxy.append((xyxy, conf))
                
                # Check match against GT
                matched = False
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx not in matched_gt and box_iou(xyxy, g_box) >= 0.5:
                        matched = True
                        matched_gt.add(g_idx)
                        tp += 1
                        break
                if not matched:
                    fp += 1
            fn = len(gt_boxes) - len(matched_gt)
            return tp, fp, fn, p_boxes_xyxy

        tp_b, fp_b, fn_b, boxes_b = eval_boxes(pred_base, gt_boxes)
        tp_v, fp_v, fn_v, boxes_v = eval_boxes(pred_v2, gt_boxes)

        base_total_tp += tp_b
        base_total_fp += fp_b
        base_total_fn += fn_b

        v2_total_tp += tp_v
        v2_total_fp += fp_v
        v2_total_fn += fn_v

        results_per_image.append({
            "filename": f.name,
            "gt_instances": len(gt_boxes),
            "baseline": {"tp": tp_b, "fp": fp_b, "fn": fn_b, "detections": len(boxes_b)},
            "updated_v2": {"tp": tp_v, "fp": fp_v, "fn": fn_v, "detections": len(boxes_v)}
        })

        # Draw visual side-by-side comparison
        # Panel 1: Baseline
        img_base_draw = img_bgr.copy()
        for xyxy in gt_boxes:
            cv2.rectangle(img_base_draw, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 255), 1)  # white for GT
        for xyxy, conf in boxes_b:
            cv2.rectangle(img_base_draw, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 165, 255), 2)  # orange for baseline
            cv2.putText(img_base_draw, f"B:{conf:.2f}", (xyxy[0], max(18, xyxy[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.putText(img_base_draw, f"Baseline (TP:{tp_b} FP:{fp_b} FN:{fn_b})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Panel 2: Updated v2
        img_v2_draw = img_bgr.copy()
        for xyxy in gt_boxes:
            cv2.rectangle(img_v2_draw, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 255), 1)
        for xyxy, conf in boxes_v:
            cv2.rectangle(img_v2_draw, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 220, 0), 2)  # green for v2
            cv2.putText(img_v2_draw, f"V2:{conf:.2f}", (xyxy[0], max(18, xyxy[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
        cv2.putText(img_v2_draw, f"Updated V2 (TP:{tp_v} FP:{fp_v} FN:{fn_v})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

        # Stack horizontally (resize if large)
        disp_w, disp_h = 480, int(480 * (h / w))
        p1 = cv2.resize(img_base_draw, (disp_w, disp_h))
        p2 = cv2.resize(img_v2_draw, (disp_w, disp_h))
        combined = np.hstack([p1, p2])
        cv2.imwrite(str(out_vis_dir / f"compare_{f.name}"), combined)

    total_gt = sum(r["gt_instances"] for r in results_per_image)

    analysis = {
        "total_test_images": len(test_files),
        "total_ground_truth_instances": total_gt,
        "baseline_detection_counts": {
            "true_positives": base_total_tp,
            "false_positives": base_total_fp,
            "false_negatives": base_total_fn,
            "precision_at_035": round(base_total_tp / max(1, base_total_tp + base_total_fp), 4),
            "recall_at_035": round(base_total_tp / max(1, base_total_tp + base_total_fn), 4)
        },
        "updated_v2_detection_counts": {
            "true_positives": v2_total_tp,
            "false_positives": v2_total_fp,
            "false_negatives": v2_total_fn,
            "precision_at_035": round(v2_total_tp / max(1, v2_total_tp + v2_total_fp), 4),
            "recall_at_035": round(v2_total_tp / max(1, v2_total_tp + v2_total_fn), 4)
        },
        "comparison_visualizations_dir": str(out_vis_dir),
        "per_image_results": results_per_image
    }

    report_path = root_dir / "runs" / "detect" / "two_wheeler" / "detailed_regression_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print("\n=== REGRESSION ANALYSIS SUMMARY (Test Set at conf=0.35) ===")
    print(f"Ground Truth Instances: {total_gt}")
    print(f"Baseline:   TP={base_total_tp}, FP={base_total_fp}, FN={base_total_fn} (P={analysis['baseline_detection_counts']['precision_at_035']:.4f}, R={analysis['baseline_detection_counts']['recall_at_035']:.4f})")
    print(f"Updated V2: TP={v2_total_tp}, FP={v2_total_fp}, FN={v2_total_fn} (P={analysis['updated_v2_detection_counts']['precision_at_035']:.4f}, R={analysis['updated_v2_detection_counts']['recall_at_035']:.4f})")
    print(f"Detailed report saved to {report_path}")
    print(f"Side-by-side images saved to {out_vis_dir}")
    return analysis

if __name__ == "__main__":
    run_regression_analysis()

