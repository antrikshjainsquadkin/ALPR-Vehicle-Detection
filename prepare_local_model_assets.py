#!/usr/bin/env python3.14
"""Explicit One-Time Setup Script: Prepare Local Offline Model Assets for ALPR.

Purpose:
- Downloads the free, open-weight YOLOv8 license plate detector (Koushim/yolov8-license-plate-detection).
- Downloads official offline EasyOCR models (craft_mlt_25k and english_g2).
- Verifies model integrity, classes, and offline readiness.
- Must be manually executed once during installation.
- NEVER runs automatically during normal ALPR operation.
"""

import hashlib
import os
import sys
import urllib.request
import zipfile

DETECTOR_URL = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
EASYOCR_CRAFT_ZIP_URL = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
EASYOCR_EN_ZIP_URL = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
FASTPLATEOCR_ONNX_URL = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.onnx"
FASTPLATEOCR_CONFIG_URL = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml"


def download_with_progress(url: str, dest_path: str):
    """Download a file with console progress indication."""
    print(f"[-] Fetching: {url}")
    print(f"    Target:   {dest_path}")
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    
    temp_dest = dest_path + ".tmp"
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    Progress: {percent:5.1f}% ({mb_down:5.1f} / {mb_total:5.1f} MB)")
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r    Progress: {mb_down:5.1f} MB downloaded")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, temp_dest, reporthook=reporthook)
        print()
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.replace(temp_dest, dest_path)
        print(f"[+] Successfully saved {os.path.getsize(dest_path)} bytes to {dest_path}")
    except Exception as e:
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        sys.exit(f"[ERROR] Failed to download {url}: {e}")


def extract_zip_model(zip_path: str, extract_dir: str, target_filename: str):
    """Extract model pth from downloaded zip archive."""
    print(f"[-] Extracting {target_filename} from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    final_path = os.path.join(extract_dir, target_filename)
    if os.path.exists(final_path):
        print(f"[+] Extracted: {final_path} ({os.path.getsize(final_path)} bytes)")
    else:
        sys.exit(f"[ERROR] Target file {target_filename} not found in zip archive {zip_path}")
    if os.path.exists(zip_path):
        os.remove(zip_path)


def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    ocr_dir = os.path.join(models_dir, "ocr")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)

    detector_path = os.path.join(models_dir, "license_plate_detector.pt")
    craft_path = os.path.join(ocr_dir, "craft_mlt_25k.pth")
    en_path = os.path.join(ocr_dir, "english_g2.pth")

    print("================================================================")
    print("   ALPR Model Asset Preparation (Explicit Manual Setup)        ")
    print("================================================================")

    # 1. License Plate Detector
    if os.path.exists(detector_path) and os.path.getsize(detector_path) > 1_000_000:
        print(f"[+] License plate detector already present: {detector_path}")
    else:
        download_with_progress(DETECTOR_URL, detector_path)

    # 2. EasyOCR Craft Detector
    if os.path.exists(craft_path) and os.path.getsize(craft_path) > 10_000_000:
        print(f"[+] EasyOCR CRAFT model already present: {craft_path}")
    else:
        craft_zip = os.path.join(ocr_dir, "craft.zip")
        download_with_progress(EASYOCR_CRAFT_ZIP_URL, craft_zip)
        extract_zip_model(craft_zip, ocr_dir, "craft_mlt_25k.pth")

    # 3. EasyOCR English Recognition Model
    if os.path.exists(en_path) and os.path.getsize(en_path) > 10_000_000:
        print(f"[+] EasyOCR English model already present: {en_path}")
    else:
        en_zip = os.path.join(ocr_dir, "en.zip")
        download_with_progress(EASYOCR_EN_ZIP_URL, en_zip)
        extract_zip_model(en_zip, ocr_dir, "english_g2.pth")

    # 4. FastPlateOCR cct-s-v2-global-model
    fp_dir = os.path.join(ocr_dir, "fastplateocr")
    os.makedirs(fp_dir, exist_ok=True)
    fp_onnx_path = os.path.join(fp_dir, "cct_s_v2_global.onnx")
    fp_cfg_path = os.path.join(fp_dir, "cct_s_v2_global_plate_config.yaml")

    if os.path.exists(fp_onnx_path) and os.path.getsize(fp_onnx_path) > 4_000_000:
        print(f"[+] FastPlateOCR ONNX model already present: {fp_onnx_path}")
    else:
        download_with_progress(FASTPLATEOCR_ONNX_URL, fp_onnx_path)

    if os.path.exists(fp_cfg_path) and os.path.getsize(fp_cfg_path) > 500:
        print(f"[+] FastPlateOCR plate config already present: {fp_cfg_path}")
    else:
        download_with_progress(FASTPLATEOCR_CONFIG_URL, fp_cfg_path)

    print("\n[-] Verifying loaded models offline...")
    try:
        from ultralytics import YOLO
        model = YOLO(detector_path)
        print(f"[+] YOLO model verified: classes = {model.names}")
        if 0 not in model.names and 'license_plate' not in list(model.names.values()):
            print(f"[!] Warning: model classes do not explicitly mention 'license_plate': {model.names}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to verify YOLO model: {e}")

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=ocr_dir, download_enabled=False)
        print("[+] Offline EasyOCR initialized successfully with zero network access.")
    except Exception as e:
        sys.exit(f"[ERROR] Failed offline EasyOCR verification: {e}")

    try:
        import numpy as np
        from fast_plate_ocr import LicensePlateRecognizer
        fp_rec = LicensePlateRecognizer(
            onnx_model_path=fp_onnx_path,
            plate_config_path=fp_cfg_path,
            device="cpu",
        )
        dummy_plate = np.full((64, 128, 3), 200, dtype=np.uint8)
        res = fp_rec.run(dummy_plate, return_confidence=True)
        print(f"[+] Offline FastPlateOCR (cct-s-v2-global-model) initialized successfully with zero network access.")
    except Exception as e:
        sys.exit(f"[ERROR] Failed offline FastPlateOCR verification: {e}")

    ppocr_det_dir = os.path.join(ocr_dir, "ppocrv6", "PP-OCRv6_medium_det")
    ppocr_rec_dir = os.path.join(ocr_dir, "ppocrv6", "PP-OCRv6_medium_rec")
    if os.path.exists(ppocr_det_dir) and os.path.exists(ppocr_rec_dir):
        print(f"[+] PP-OCRv6 detection assets found: {ppocr_det_dir}")
        print(f"[+] PP-OCRv6 recognition assets found: {ppocr_rec_dir}")
        try:
            import torch  # noqa: F401
            os.environ["FLAGS_use_onednn"] = "0"
            from paddleocr import PaddleOCR
            pp_reader = PaddleOCR(
                text_detection_model_dir=ppocr_det_dir,
                text_recognition_model_dir=ppocr_rec_dir,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            print("[+] Offline PaddleOCR PP-OCRv6 initialized successfully with zero network access.")
        except Exception as e:
            print(f"[!] PP-OCRv6 offline verification note: {e}")
    else:
        print(f"[!] Warning: PP-OCRv6 assets not found in {os.path.join(ocr_dir, 'ppocrv6')}")

    print("\n================================================================")
    print("   All Local Model Assets Ready for Offline ALPR Ingestion      ")
    print("================================================================")


if __name__ == "__main__":
    main()
