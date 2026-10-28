# Automatic Vehicle License Plate Recognition (ALPR) System

A local-only, CPU-optimized computer vision system that detects vehicle license plates and recognizes alphanumeric registration numbers from video footage, live webcams, and RTSP surveillance streams.

The system requires no external cloud APIs or runtime internet access. It combines a local YOLOv8 plate detector with **FastPlateOCR** (`cct-s-v2-global-model` ONNX runtime), automated multi-frame consensus aggregation, Laplacian sharpness weighting, stacked (two-line) plate handling, and structured JSON record persistence.

---

## Table of Contents

- [Key Features](#key-features)
- [How It Works (Pipeline Architecture)](#how-it-works-pipeline-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites & System Requirements](#prerequisites--system-requirements)
- [Installation](#installation)
- [Offline Model Assets Setup](#offline-model-assets-setup)
- [Usage Guide](#usage-guide)
  - [1. Interactive Terminal Menu](#1-interactive-terminal-menu)
  - [2. Command-Line Interface (CLI)](#2-command-line-interface-cli)
  - [CLI Reference](#cli-reference)
- [JSON Ledger Output & Schema](#json-ledger-output--schema)
- [Configuration Reference](#configuration-reference)
- [Advanced Engine Capabilities](#advanced-engine-capabilities)
- [Troubleshooting & Best Practices](#troubleshooting--best-practices)
- [License](#license)

---

## Key Features

- **Adaptive Resolution Plate Detection:**
  - Employs a local YOLOv8 license plate detector (`models/license_plate_detector.pt`).
  - Automatically selects inference resolution (`1920` for 1080p surveillance video, `1280` or `640` for lower resolutions) to retain high-frequency stroke details on small, distant plates.
- **Stacked / Two-Line Plate Recognition:**
  - Automatically identifies square aspect-ratio plates ($1.05 \le \text{AR} \le 2.4$, typical on motorcycles, scooters, auto-rickshaws, and certain commercial carriers).
  - Performs an intelligent horizontal split with central overlap, evaluates both lines separately with FastPlateOCR, and combines them into unified plate numbers (e.g., `GJ01X` + `H8398` $\rightarrow$ `GJ01XH8398`).
- **DVR OSD Watermark Suppression:**
  - Rejects false positive detections triggered by on-screen surveillance text overlays (such as bottom-left camera names and timestamps like *"D RAMP ENTRY MAIN ROAD"*).
- **Safe Crop Margins & Boundary Protection:**
  - Applies dynamic horizontal (8% / min 4px) and vertical (6% / min 3px) boundary padding to eliminate character stroke clipping at plate edges.
- **Pluggable OCR Architecture:**
  - **FastPlateOCR (Default):** High-speed ONNX Runtime inference using `cct-s-v2-global-model`. Delivers authentic per-character confidence scores and operates ~30x faster to load and ~7x faster to infer than legacy engines.
  - **PP-OCRv6 (Alternative):** Offline PaddleOCR detection + recognition pipeline.
  - **EasyOCR (Rollback):** PyTorch CRAFT text detector + English recognition.
- **Sharpness-Weighted Multi-Frame Consensus:**
  - Computes Laplacian variance for every plate crop to assess optical clarity.
  - Aligns multiple observations across frames, weighting character votes by sharpness and authentic model confidence.
  - Uses national pattern syntax rules (Indian HSRP, UK post-2001, International 3A4D) to resolve ambiguous characters (e.g., `0` vs `O`, `1` vs `I`).
  - Protects validated national plates from being overwritten by unpatterned low-confidence noise from subsequent blurry frames.
  - Flags uncertain characters in `review_required` rather than silently guessing.
- **Plate Color Classification:**
  - Performs HSV color distribution analysis on plate crops to identify plate types: **White** (private vehicles), **Yellow** (commercial carriers / taxis), **Green** (electric vehicles), or **Other**.
- **Dual User Interfaces:**
  - **Interactive Terminal Menu:** Easy navigation with automated Windows file-picker dialogs, live camera ingestion, and ledger viewing.
  - **CLI / Headless Mode:** Fully automated ingestion pipeline supporting video files, streams, and batch scripts.
- **Thread-Safe Atomic Persistence:**
  - Writes structured records to a local JSON ledger atomically using temporary staging files and replace semantics to eliminate data corruption.

---

## How It Works (Pipeline Architecture)

```
[ Video File / Camera / RTSP Stream ]
                  │
                  ▼
         [ Frame Decoding ]
                  │ (Adaptive resolution: imgsz=1920 for 1080p)
                  ▼
     [ YOLOv8 Plate Detection ] ──> Reject DVR Watermarks / Bad AR
                  │
                  ▼
      [ Safe Boundary Padding ]
                  │
         [ Aspect Ratio Check ]
         ├── If 1.05 <= AR <= 2.4 ──> [ Two-Line Split & Dual OCR ]
         └── If AR > 2.4          ──> [ Standard Single-Line OCR ]
                  │
                  ▼
      [ FastPlateOCR Inference ] ──> Extract Raw Characters & Per-Char Confidence
                  │
                  ▼
      [ HSV Color Classification ] ──> (WHITE / YELLOW / GREEN)
                  │
                  ▼
 [ Laplacian Variance Sharpness ] ──> Image Clarity Score
                  │
                  ▼
[ Multi-Frame Trajectory Matching ] ──> Track Plate Across Frames (IoU / Distance)
                  │
                  ▼
   [ Weighted Consensus Engine ]
   ├── Position-wise character voting (weighted by sharpness & conf)
   ├── National pattern matching (IN_STANDARD, GB_STANDARD, INTL)
   ├── Pattern-aware disambiguation (0/O, 1/I)
   └── Low-confidence tagging (review_required)
                  │
                  ▼
    [ Atomic JSON Ledger Update ] ──> `data/recognized_license_plate_records.json`
```

---

## Technology Stack

- **Core Language:** Python 3.12 / 3.14 compatible
- **Object Detection:** Ultralytics YOLOv8 (`ultralytics`)
- **Primary OCR:** FastPlateOCR (`cct-s-v2-global-model` via `onnxruntime`)
- **Fallback OCR Engines:** PP-OCRv6 (`paddleocr`), EasyOCR (`easyocr` + PyTorch)
- **Computer Vision:** OpenCV (`opencv-python`) for decoding, color space transformations, and Laplacian filtering
- **Data Persistence:** Local JSON ledger with ISO 8601 UTC timestamps
- **Testing & Verification:** `pytest` (45 automated test cases covering edge cases, pipeline flows, and consensus logic)

---

## Project Structure

```text
vehicle_detect/
├── vehicle_license_plate_recognition.py   # Primary named application entrypoint
├── main.py                                # Core ALPR pipeline, engines, consensus & CLI/menu
├── prepare_local_model_assets.py          # One-time setup script to download offline models
├── requirements.txt                       # Pinned Python package dependencies
├── config/
│   └── license_plate_recognition_config.json # Runtime thresholds, regex patterns, deduplication settings
├── data/
│   └── recognized_license_plate_records.json # Primary JSON ledger storing recognized vehicle passages
├── models/
│   ├── license_plate_detector.pt          # Local YOLOv8 license plate detector weights
│   └── ocr/
│       ├── fastplateocr/
│       │   ├── cct_s_v2_global.onnx       # FastPlateOCR ONNX model weights
│       │   └── cct_s_v2_global_plate_config.yaml # Plate character dictionary & architecture config
│       ├── craft_mlt_25k.pth              # EasyOCR CRAFT detector (offline rollback)
│       └── english_g2.pth                 # EasyOCR English recognizer (offline rollback)
├── tests/
│   ├── test_fastplateocr_and_consensus.py # Unit tests for FastPlateOCR and consensus scoring
│   ├── test_pipeline_edge_cases.py        # End-to-end integration and mock pipeline tests
│   └── test_unit_edge_cases.py            # Geometry, Levenshtein, color, and parsing tests
├── .gitignore
└── README.md
```

---

## Prerequisites & System Requirements

- **Operating System:** Windows 10/11, Linux, or macOS.
- **Python Version:** Python 3.12 or 3.14.
- **Hardware:**
  - **CPU:** Standard multi-core x86_64 or ARM processor (CPU-first execution is supported out of the box).
  - **RAM:** Minimum 4 GB (8 GB recommended for processing 1080p/4K streams).
  - **GPU (Optional):** NVIDIA CUDA-compatible GPU if `--device cuda` is specified.

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/antrikshjainsquadkin/Vehicle-Detection.git
cd Vehicle-Detection
```

### 2. Create and activate a virtual environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## Offline Model Assets Setup

The system operates strictly offline. All model weights reside in the `models/` directory.

If you are cloning into a fresh environment without pre-bundled weights, run the included setup helper once:

```powershell
python prepare_local_model_assets.py
```

This downloads:

1. `models/license_plate_detector.pt` (~6 MB)
2. `models/ocr/fastplateocr/cct_s_v2_global.onnx` (~15 MB) and its YAML config
3. `models/ocr/craft_mlt_25k.pth` and `english_g2.pth` (for EasyOCR rollback)

Once these files are present, the setup script is no longer needed during normal operation.

---

## Usage Guide

### 1. Interactive Terminal Menu

Launch the application without command-line arguments to access the interactive terminal menu:

```powershell
python vehicle_license_plate_recognition.py
```

```text
========================================================================
      Automatic Vehicle License Plate Recognition (ALPR) System         
========================================================================
 Active OCR Engine:  fastplateocr
 Live Preview:       ENABLED
 Inference Device:   cpu
------------------------------------------------------------------------
 [1] Process Video File (Browse Dialog...)
 [2] Process Video File (Enter Path Manually)
 [3] Process Live Camera / Webcam (e.g., Index 0, 1)
 [4] Process RTSP / Network Stream (e.g., rtsp://...)
 [5] Advanced Options & Configuration
 [6] View Recognized Records Ledger
------------------------------------------------------------------------
 [Q] Exit
========================================================================
```

- **[1] Browse Dialog:** Opens a standard file dialog to select a video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`).
- **[2] Manual Path:** Type or paste any absolute or relative video file path.
- **[3] Camera:** Ingest from a connected webcam (e.g., index `0` or `1`).
- **[4] RTSP Stream:** Enter a live network camera URI (e.g., `rtsp://admin:pass@192.168.1.50:554/live`).
- **[5] Advanced Options:** Adjust frame stride, OCR engine, confidence thresholds, detector image size, preview toggles, and inference devices on the fly.
- **[6] View Ledger:** Display recent vehicle passage records directly in your terminal.

---

### 2. Command-Line Interface (CLI)

For headless deployments, batch automation, or scripting, invoke the application directly via arguments:

#### Process a video file with live preview

```powershell
python vehicle_license_plate_recognition.py --source "C:\path\to\video.mp4" --preview
```

#### Process in headless mode with adaptive 1080p resolution and frame sampling stride

```powershell
python vehicle_license_plate_recognition.py --source "C:\path\to\video.mp4" --frame-stride 3 --imgsz 1920
```

#### Process a live RTSP stream with custom deduplication cooldown

```powershell
python vehicle_license_plate_recognition.py --source "rtsp://camera.local:554/h264" --dedup-seconds 45.0
```

#### Run with alternative or rollback OCR engines

```powershell
# Using PP-OCRv6:
python vehicle_license_plate_recognition.py --source "video.mp4" --ocr-engine ppocrv6

# Using EasyOCR (rollback):
python vehicle_license_plate_recognition.py --source "video.mp4" --ocr-engine easyocr
```

---

### CLI Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | *Required* | Path to video file, integer webcam index (`0`), or RTSP/stream URL. |
| `--preview` | `flag` | `False` | Display live OpenCV preview window with tracking bounding boxes and metadata. |
| `--frame-stride` | `int` | `3` | Process every $N$-th frame (improves CPU throughput on high-fps feeds). |
| `--imgsz` | `int` | `auto` | YOLO detector input resolution (`640`, `1280`, `1920`). `auto` selects 1920 for 1080p feeds. |
| `--ocr-engine` | `str` | `fastplateocr` | Active OCR engine: `fastplateocr` (default), `ppocrv6`, or `easyocr`. |
| `--det-conf` | `float` | `0.20` | Detection confidence threshold $[0.0 - 1.0]$. |
| `--ocr-conf` | `float` | `0.35` | OCR acceptance confidence threshold $[0.0 - 1.0]$. |
| `--dedup-seconds` | `float` | `30.0` | Cooldown period in seconds to suppress duplicate records for the same vehicle. |
| `--ledger` | `str` | `data/...` | Custom output path for recognized JSON records. |
| `--device` | `str` | `cpu` | Hardware device for inference: `cpu` or `cuda`. |
| `--max-frames` | `int` | `None` | Max frames to process before stopping cleanly (useful for benchmarks and testing). |
| `--config` | `str` | `config/...` | Path to custom JSON configuration file. |

---

## JSON Ledger Output & Schema

Recognized license plates are stored in `data/recognized_license_plate_records.json`.

Each vehicle passage record captures comprehensive metadata, including per-character confidence scores, format classifications, bounding coordinates, and review requirements:

```json
{
  "schema_version": 1,
  "updated_at": "2026-10-28T09:33:47.538Z",
  "records": [
    {
      "event_id": "aa169bb9-351c-4f58-83d9-bed5a6dd9a60",
      "timestamp": "2026-10-28T09:33:22.495Z",
      "detected_plate_text": "GJ011R8398",
      "normalized_plate_text": "GJ01IH8398",
      "confidence_score": 0.9072,
      "detector_confidence": 0.3457,
      "ocr_confidence": 0.9072,
      "ocr_engine": "fastplateocr",
      "character_confidences": [
        {"char": "G", "confidence": 0.7524},
        {"char": "J", "confidence": 0.9979},
        {"char": "0", "confidence": 0.8101},
        {"char": "1", "confidence": 0.9969},
        {"char": "I", "confidence": 0.5791},
        {"char": "H", "confidence": 0.9991},
        {"char": "8", "confidence": 0.9408},
        {"char": "3", "confidence": 0.9997},
        {"char": "9", "confidence": 0.9977},
        {"char": "8", "confidence": 0.9987}
      ],
      "min_character_confidence": 0.5791,
      "review_required": ["I"],
      "matched_country_pattern": "IN_STANDARD",
      "disambiguation_applied": true,
      "plate_color": "WHITE",
      "source_id": "IMG_8426.MP4",
      "frame_index": 3,
      "last_frame_index": 72,
      "observations_count": 13,
      "bounding_box": [947, 706, 1001, 751]
    }
  ]
}
```

### Key Field Descriptions

- **`normalized_plate_text`**: Final consensus registration number after character-level voting and pattern-aware disambiguation.
- **`confidence_score`**: True mean confidence across recognized characters bounded strictly within $[0.0, 1.0]$.
- **`character_confidences`**: Exact probability distribution for each individual character position.
- **`review_required`**: Array of characters whose confidence fell below the review threshold (`< 0.60`), allowing human auditing.
- **`matched_country_pattern`**: Matched national syntax standard (`IN_STANDARD`, `GB_STANDARD`, `INTL_STANDARD_3A4D`, or `US_GENERIC`).
- **`plate_color`**: Detected background profile (`WHITE`, `YELLOW`, `GREEN`, `OTHER`).
- **`observations_count`**: Number of distinct video frames that contributed to the consensus reading.

---

## Configuration Reference

System behavior is configured via [`config/license_plate_recognition_config.json`](file:///c:/Users/A/OneDrive/Desktop/vehicle_detect/config/license_plate_recognition_config.json):

```json
{
  "inference_thresholds": {
    "detector_confidence_threshold": 0.20,
    "ocr_confidence_threshold": 0.20,
    "fastplateocr_confidence_threshold": 0.35,
    "char_low_confidence_threshold": 0.60,
    "box_padding_ratio": 0.06
  },
  "plate_validation": {
    "country_patterns": [
      {
        "country": "IN_STANDARD",
        "description": "Standard Indian High Security Registration Plate",
        "regex": "^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{4}$"
      },
      {
        "country": "GB_STANDARD",
        "description": "UK Standard Post-2001 Format",
        "regex": "^[A-Z]{2}[0-9]{2}[A-Z]{3}$"
      },
      {
        "country": "INTL_STANDARD_3A4D",
        "description": "Standard 3-letter 4-digit format",
        "regex": "^[A-Z]{3}[0-9]{4}$"
      }
    ]
  },
  "deduplication": {
    "cooldown_seconds": 30.0
  },
  "video_ingestion": {
    "default_frame_stride": 3
  }
}
```

---

## Advanced Engine Capabilities

### 1. Two-Line / Stacked Plate Recognition

Two-wheelers and commercial carriers often use squarish two-line plates (aspect ratio $1.05 \le \text{AR} \le 2.4$). Standard single-line OCR models fail when fed both lines simultaneously. The engine automatically detects this geometry:

1. Splits the crop into top (54% height) and bottom (56% height) slices with central overlap.
2. Inferences both segments independently.
3. Merges the results into a single coherent registration string (e.g., `GJ01X` + `H8398` $\rightarrow$ `GJ01XH8398`).

### 2. Multi-Frame Trajectory Matching & Sharpness Weighting

Instead of relying on a single snapshot:
- Vehicles are tracked across frames using spatial IoU, center distance, and prefix matching.
- Every frame calculates Laplacian variance ($s = \text{Var}(\nabla^2 I)$). Sharp, close-up frames naturally outvote distant, motion-blurred frames.
- National pattern matches receive structural bonuses to eliminate noise from random background textures.

---

## Troubleshooting & Best Practices

- **Small or Distant Plates:**
  If plates appear small in 1080p or 4K footage, ensure `--imgsz 1920` (or `auto`) is active so the detector does not downsample the frame.
- **High CPU Utilization:**
  Increase `--frame-stride` to `4` or `5`. In surveillance footage, vehicles remain in frame across dozens of frames; skipping intermediate frames saves CPU cycles without degrading consensus accuracy.
- **Camera Overlays / Watermarks:**
  The system automatically suppresses watermarks in the bottom-left quadrant. If your camera has text elsewhere, adjust the detection region or crop bounds in `main.py`.
- **Running Automated Tests:**
  Verify your local environment at any time using `pytest`:

  ```powershell
  python -m pytest
  ```

  *(All 45 unit and pipeline integration tests should pass cleanly).*

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for the full license terms.

Third-party libraries, detection models, OCR engines, and model weights may have separate license terms. Refer to their respective licenses for details.
