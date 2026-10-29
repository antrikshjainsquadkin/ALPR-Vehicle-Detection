"""
Curates clean, high-precision pseudo-labels using the baseline model.
Applies a robust confidence threshold (conf=0.45) and strict NMS (iou=0.45)
to extract clean, dominant two-wheeler bounding boxes without noisy low-confidence fragments.
"""

from pathlib import Path
from ultralytics import YOLO
import cv2

def curate_clean_labels():
    model = YOLO("models/two_wheeler_detector_baseline.pt")
    base_dir = Path(r"c:\Users\A\OneDrive\Desktop\vehicle_detect\datasets\two_wheeler")
    raw_images_dir = base_dir / "raw_images"
    clean_pseudo_dir = base_dir / "clean_curated_pseudo_labels"
    clean_pseudo_dir.mkdir(parents=True, exist_ok=True)
    
    # Load list of challenging images
    flagged_path = base_dir / "flagged_for_human_review.json"
    import json
    with open(flagged_path, "r") as f:
        challenging_files = [item["filename"] for item in json.load(f)]

    print(f"[Curate] Running clean inference on {len(challenging_files)} challenging images with conf=0.45, iou=0.45...")
    
    high_quality_count = 0
    multi_box_count = 0
    zero_box_count = 0

    curation_manifest = []

    for idx, fname in enumerate(challenging_files):
        img_p = raw_images_dir / fname
        results = model.predict(str(img_p), conf=0.45, iou=0.45, imgsz=640, verbose=False)[0]
        boxes = results.boxes

        lbl_path = clean_pseudo_dir / f"{Path(fname).stem}.txt"
        
        if len(boxes) == 0:
            zero_box_count += 1
            lbl_path.write_text("", encoding="utf-8")
            curation_manifest.append({"filename": fname, "status": "ZERO_BOX", "boxes": 0})
            continue

        box_count = len(boxes)
        if box_count > 3:
            multi_box_count += 1
            status = "MULTIPLE_OBJECTS"
        else:
            high_quality_count += 1
            status = "CLEAN_HIGH_CONF"

        with open(lbl_path, "w", encoding="utf-8") as lf:
            for b in boxes:
                xywhn = b.xywhn[0].tolist()
                lf.write(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

        curation_manifest.append({
            "filename": fname,
            "status": status,
            "boxes": box_count,
            "max_conf": round(float(boxes.conf.max()), 3)
        })

    print(f"[Curate] Results:")
    print(f"  Clean, confident (1-3 boxes): {high_quality_count} ({(high_quality_count/len(challenging_files))*100:.1f}%)")
    print(f"  Multiple objects (>3 boxes):  {multi_box_count}")
    print(f"  Zero boxes at conf>=0.45:    {zero_box_count}")

    manifest_path = base_dir / "clean_curation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(curation_manifest, f, indent=2)

    return high_quality_count

if __name__ == "__main__":
    curate_clean_labels()

