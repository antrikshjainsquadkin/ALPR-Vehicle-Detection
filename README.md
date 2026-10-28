# Automatic Vehicle License Plate Recognition (ALPR) System

A local-only, offline vehicle detection and license plate recognition system utilizing a verified local YOLOv8 detector and high-performance **FastPlateOCR** (`cct-s-v2-global-model`) recognition engine with character-level confidence scores and confidence-aware multi-frame consensus (with offline PP-OCRv6 and EasyOCR rollback).

## Key Capabilities

- **High-Performance FastPlateOCR Recognition**:
  - Employs `cct-s-v2-global-model` on ONNX Runtime CPU (or CUDA).
  - **Character-Level Confidence**: Every recognized character provides authentic model confidence probabilities (`char_probs`).
  - **Confidence-Aware Multi-Frame Consensus**: Aggregates repeated observations across video frames using position-aligned weighted voting. Unstable characters are placed in `review_required` rather than silently modified.
  - **Speed & Efficiency**: Over 30x faster startup and 7x faster inference than legacy EasyOCR.
- **Multi-Color Plate Recognition**: Reliably detects and classifies plates across color profiles:
  - **Commercial Yellow Plates** (e.g., Indian HSRP commercial plates such as `KA53MH3895`).
  - **Electric Vehicle Green Plates** (e.g., standard EV plates such as `GAE 5379`, `KBC 6233`).
  - **Standard White & Metallic Plates** (e.g., UK post-2001 plates `YG21DBM`, US/NA alphanumeric formats).
- **False Positive Rejection**: Rejects large non-plate signage (storefront signs, banners, roadside ads) through geometric aspect ratio bounding ($1.05 \le \text{AR} \le 6.5$), maximum physical dimensions ($h \le 20\%$ frame height, $w \le 60\%$ frame width), and structured alphanumeric validation.
- **Passage Consolidation & Deduplication**: Employs spatial IoU tracking and cost-capped character disambiguation ($\le 2$ substitutions) to consolidate continuous video detections into single passage records, preventing artificial duplicates.
- **100% Offline & Local**: Zero cloud APIs, telemetry, or network egress.
- **Atomic Persistence**: Thread-safe atomic JSON ledger with precision ISO 8601 UTC timestamps, per-character confidence scores, and review flags.

---

## Launching the Application

### 1. Interactive Terminal Menu (No Arguments)
Running the application directly without arguments automatically launches the interactive terminal menu:

```powershell
py -3.14 vehicle_license_plate_recognition.py
# or:
.\.venv\Scripts\python.exe main.py
```

#### Main Menu Features:
- **[1] Process Video File (Browse Dialog...)**: Opens a standard Windows file-picker dialog filtered for video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`).
- **[2] Process Video File (Enter Path Manually)**: Paste an absolute or relative path (quotes and spaces are automatically normalized).
- **[3] Process Live Camera / Webcam**: Enter a numeric index (e.g., `0` for default USB/built-in camera).
- **[4] Process RTSP / Network Stream**: Enter a live stream URL (`rtsp://`, `rtmp://`, `http://`, etc.).
- **[5] Advanced Options & Configuration**:
  - Live Preview toggle (`--preview`).
  - Frame sampling stride (`--frame-stride`, default `3`).
  - OCR Engine switch (`fastplateocr` default, `ppocrv6`, or `easyocr` rollback).
  - Deduplication cooldown window (`--dedup-seconds`, default `30.0s`).
  - Detector confidence threshold (`--det-conf`, default `0.20`).
  - OCR confidence threshold (`--ocr-conf`, default `0.35`).
  - Inference device (`--device cpu` / `cuda`).
  - Detector input image size (`--imgsz`, default `auto`).
- **[6] View Recognized Records Ledger**: Inspect total vehicles detected and recent recognition records in the terminal.
- **[Q] Exit**: Closes the application cleanly.

---

### 2. Direct Command Line Ingestion

Process a video file directly from PowerShell:

```powershell
.\.venv\Scripts\python.exe -u main.py --source "C:\path\to\video.mp4" --preview
```

Custom ledger path and frame stride:
```powershell
.\.venv\Scripts\python.exe -u main.py `
    --source "C:\path\to\video.mp4" `
    --ledger "data/recognized_license_plate_records.json" `
    --frame-stride 5 `
    --preview
```

---

## Configuration

Settings are maintained in `config/license_plate_recognition_config.json`:

```json
{
  "schema_version": 1,
  "inference_thresholds": {
    "detector_confidence_threshold": 0.20,
    "ocr_confidence_threshold": 0.20,
    "ppocrv6_confidence_threshold": 0.35,
    "combined_confidence_threshold": 0.20,
    "box_padding_ratio": 0.06
  },
  "plate_validation": {
    "validation_mode": "accept_if_no_match",
    "country_patterns": [
      {
        "country": "IN_STANDARD",
        "description": "Standard Indian HSRP format",
        "regex": "^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{4}$"
      },
      {
        "country": "GB_STANDARD",
        "description": "UK Standard Post-2001 format",
        "regex": "^[A-Z]{2}[0-9]{2}[A-Z]{3}$"
      },
      {
        "country": "INTL_STANDARD_3A4D",
        "description": "International 3-letter 4-digit format (EV green plates)",
        "regex": "^[A-Z]{3}[0-9]{4}$"
      },
      {
        "country": "US_GENERIC",
        "description": "General North American 6-8 alphanumeric format",
        "regex": "^[A-Z0-9]{6,8}$"
      }
    ]
  }
}
```

---

## Running Verification Suites

To verify regression safety and multi-color detection accuracy:

```powershell
# 1. Comprehensive ALPR Regression Suite (7 Checks)
.\.venv\Scripts\python.exe scratch\test_alpr_regression_suite.py

# 2. End-to-End Multi-Color & False Positive Verification
.\.venv\Scripts\python.exe -u scratch\verify_phase_d_e2e.py
```
