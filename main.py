#!/usr/bin/env python3
"""Vehicle License Plate Recognition (ALPR) - Local-Only Continuous Ingestion & Recognition.

Core Architectural Principles:
1. Local-Only Execution: No cloud inference APIs, no analytics or telemetry egress.
2. Verified Local Model: YOLOv8 license plate detector loaded from local weights.
3. Offline Local OCR: PaddleOCR PP-OCRv6 as primary recognition engine, EasyOCR as offline rollback.
4. Timezone-Aware ISO 8601 Timestamps: Precision UTC format (YYYY-MM-DDTHH:MM:SS.sssZ).
5. Normalized Deduplication: Cooldown window suppressing repeated events across restarts.
6. Atomic JSON Persistence: Safe atomic replace using same-filesystem temporary files + fsync.
7. Resilient Stream Ingestion: Reconnection backoff for RTSP/network drops and clean resource release.
"""

import argparse
import atexit
import ctypes
import json
import os
import platform
import re
import signal
import sys
import tempfile
import time
import uuid

# Pre-import torch and set onednn flag on Windows to prevent DLL loading conflicts with PaddlePaddle
try:
    import torch  # noqa: F401
except ImportError:
    pass
os.environ.setdefault("FLAGS_use_onednn", "0")
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

_GLOBAL_LOCK_FILE: Optional[str] = None


def get_utc_now_iso() -> str:
    """Return current timestamp in ISO 8601 UTC representation."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso_utc(ts_str: str) -> datetime:
    """Parse ISO 8601 UTC timestamp string into timezone-aware datetime."""
    clean_ts = ts_str.rstrip("Z")
    if "+" in clean_ts:
        clean_ts = clean_ts.split("+")[0]
    dt = datetime.fromisoformat(clean_ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_process_running(pid: int) -> bool:
    """Check if process with given PID is currently active on the host OS."""
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def acquire_instance_lock(lock_dir: str, source_id: str) -> str:
    """Acquire single-instance lock to ensure single-writer policy."""
    global _GLOBAL_LOCK_FILE
    lock_path = os.path.join(lock_dir, ".license_plate_recognition.lock")

    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
            prev_pid = lock_data.get("pid")
            if prev_pid is not None and isinstance(prev_pid, int):
                if is_process_running(prev_pid):
                    sys.exit(
                        f"[ALPR ERROR] Active instance with PID {prev_pid} holds lock at '{lock_path}'. "
                        "Simultaneous updates to the detection ledger are forbidden to prevent corruption."
                    )
                else:
                    print(f"[ALPR] Cleaning stale lock from inactive PID {prev_pid}.", file=sys.stderr, flush=True)
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass
        except Exception:
            pass

    lock_payload = {
        "pid": os.getpid(),
        "hostname": platform.node(),
        "source_id": source_id,
        "acquired_at": get_utc_now_iso(),
    }

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(lock_payload, f, indent=2)
            f.flush()
    except FileExistsError:
        sys.exit(f"[ALPR ERROR] Lock conflict: Another process created '{lock_path}' concurrently.")
    except Exception as e:
        sys.exit(f"[ALPR ERROR] Failed to acquire instance lock at '{lock_path}': {e}")

    _GLOBAL_LOCK_FILE = lock_path
    atexit.register(release_instance_lock)
    return lock_path


def release_instance_lock():
    """Release active single-instance lock."""
    global _GLOBAL_LOCK_FILE
    if _GLOBAL_LOCK_FILE and os.path.exists(_GLOBAL_LOCK_FILE):
        try:
            os.remove(_GLOBAL_LOCK_FILE)
            _GLOBAL_LOCK_FILE = None
        except Exception:
            pass


def normalize_plate_text(raw_text: str) -> str:
    """Strip spaces, punctuation, special symbols, and convert to uppercase alphanumeric."""
    if not raw_text:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()


LETTER_TO_DIGIT_MAP: Dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6", "B": "8", "T": "7",
}

DIGIT_TO_LETTER_MAP: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}


def disambiguate_char(ch: str, target_type: str) -> Tuple[Optional[str], int]:
    """Disambiguate a single character to target_type ('numeric', 'alpha', or 'alphanumeric').
    Returns (disambiguated_char, cost) where cost is 0 if no change, 1 if disambiguated, or (None, 999) if invalid.
    """
    c = ch.upper()
    if target_type == "numeric":
        if c.isdigit():
            return c, 0
        if c in LETTER_TO_DIGIT_MAP:
            return LETTER_TO_DIGIT_MAP[c], 1
        return None, 999
    elif target_type == "alpha":
        if c.isalpha():
            return c, 0
        if c in DIGIT_TO_LETTER_MAP:
            return DIGIT_TO_LETTER_MAP[c], 1
        return None, 999
    elif target_type == "alphanumeric":
        if c.isalnum():
            return c, 0
        return None, 999
    return c, 0


def partition_match(text: str, segments: List[Dict[str, Any]], seg_idx: int = 0, char_idx: int = 0) -> List[Tuple[str, int]]:
    """Recursively partition and disambiguate text across format segment constraints."""
    if seg_idx == len(segments):
        if char_idx == len(text):
            return [("", 0)]
        return []

    seg = segments[seg_idx]
    results: List[Tuple[str, int]] = []
    min_l = seg.get("min_len", 1)
    max_l = min(seg.get("max_len", len(text)), len(text) - char_idx)

    for l in range(min_l, max_l + 1):
        chunk = text[char_idx : char_idx + l]
        valid = True
        cand_chunk: List[str] = []
        chunk_cost = 0
        for ch in chunk:
            d_ch, cost = disambiguate_char(ch, seg.get("type", "alphanumeric"))
            if d_ch is None:
                valid = False
                break
            cand_chunk.append(d_ch)
            chunk_cost += cost
        if valid:
            rest = partition_match(text, segments, seg_idx + 1, char_idx + l)
            for r_text, r_cost in rest:
                results.append(("".join(cand_chunk) + r_text, chunk_cost + r_cost))
    return results


def match_and_disambiguate_pattern(
    raw_normalized: str, country_patterns: List[Dict[str, Any]]
) -> Tuple[str, Optional[str], bool]:
    """Test normalized plate text against regional schemas.
    Applies format-aware character disambiguation ONLY according to configured segment types.
    Never globally replaces letters with digits.
    Returns (disambiguated_text, matched_country_name, disambiguation_applied_bool).
    """
    if not raw_normalized or not country_patterns:
        return raw_normalized, None, False

    test_variants: List[Tuple[str, bool]] = [(raw_normalized, False)]
    if raw_normalized.startswith("IND") and len(raw_normalized) >= 7:
        test_variants.append((raw_normalized[3:], True))
    if raw_normalized.endswith("IND") and len(raw_normalized) >= 7:
        test_variants.append((raw_normalized[:-3], True))
    elif raw_normalized.endswith("ND") and len(raw_normalized) >= 7:
        test_variants.append((raw_normalized[:-2], True))

    for text_variant, prefix_stripped in test_variants:
        # 1. Direct match with zero cost against structured patterns
        for item in country_patterns:
            pattern = item.get("regex")
            segments = item.get("segments")
            if pattern and segments:
                if re.match(pattern, text_variant):
                    return text_variant, item.get("country", "MATCHED"), prefix_stripped

        # 2. Format-aware disambiguation for structured patterns
        best_candidate: Optional[str] = None
        best_country: Optional[str] = None
        best_cost = 999

        for item in country_patterns:
            pattern = item.get("regex")
            segments = item.get("segments")
            if not pattern or not segments:
                continue

            candidates = partition_match(text_variant, segments)
            for cand_text, cost in candidates:
                if re.match(pattern, cand_text):
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = cand_text
                        best_country = item.get("country", "MATCHED")

        # Allow at most 2 character disambiguations to prevent distorting unrelated plates
        if best_candidate and best_cost <= 2:
            is_changed = (best_candidate != raw_normalized)
            return best_candidate, best_country, is_changed

        # 3. Fallback for generic unstructured patterns (e.g. US_GENERIC)
        for item in country_patterns:
            pattern = item.get("regex")
            segments = item.get("segments")
            if pattern and not segments:
                if re.match(pattern, text_variant):
                    return text_variant, item.get("country", "MATCHED"), prefix_stripped

    return raw_normalized, None, False


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute minimum character edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]


def compute_box_iou(boxA: List[int], boxB: List[int]) -> float:
    """Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0.0


def compute_center_distance(boxA: List[int], boxB: List[int]) -> float:
    """Euclidean distance between bounding box centers."""
    cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
    return ((cA[0] - cB[0]) ** 2 + (cA[1] - cB[1]) ** 2) ** 0.5


def classify_plate_color(crop: Any) -> str:
    """Classify license plate background color using HSV chromatic analysis.
    Returns: 'YELLOW', 'GREEN', 'WHITE', or 'OTHER'.
    """
    import cv2
    import numpy as np

    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim < 3:
        return "OTHER"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # In a cropped license plate, the plate background constitutes the majority of pixels
    # Yellow: Hue in [15, 38], Saturation >= 35, Value >= 45
    yellow_mask = (h_chan >= 15) & (h_chan <= 38) & (s_chan >= 35) & (v_chan >= 45)
    # Green: Hue in [38, 88], Saturation >= 28, Value >= 30
    green_mask = (h_chan >= 38) & (h_chan <= 88) & (s_chan >= 28) & (v_chan >= 30)
    # White: Low saturation, medium-high value
    white_mask = (s_chan <= 35) & (v_chan >= 110)

    total_pixels = float(crop.shape[0] * crop.shape[1])
    yellow_ratio = np.count_nonzero(yellow_mask) / total_pixels
    green_ratio = np.count_nonzero(green_mask) / total_pixels
    white_ratio = np.count_nonzero(white_mask) / total_pixels

    if yellow_ratio >= 0.22:
        return "YELLOW"
    elif green_ratio >= 0.18:
        return "GREEN"
    elif white_ratio >= 0.20:
        return "WHITE"
    return "OTHER"


def enhance_plate_crop(crop: Any, color_hint: Optional[str] = None) -> Any:
    """Enhance license plate crop for OCR adapted to background color.
    Upscales to optimal recognition height, applies bilateral denoising to remove compression artifacts,
    uses unsharp masking to accentuate thin diagonal strokes, and normalizes local contrast via CLAHE.
    """
    import cv2
    if crop is None or crop.size == 0:
        return crop

    ch, cw = crop.shape[:2]
    scale = max(2.0, min(3.5, 90.0 / max(1, ch)))
    resized = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)

    if color_hint is None:
        color_hint = classify_plate_color(crop)

    if color_hint == "GREEN":
        # Green plates (white/reflective text on dark green background):
        # Bilateral filter and CLAHE preserving white text strokes on darker green
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(denoised)
    elif color_hint == "YELLOW":
        # Yellow plates (dark text on bright yellow background):
        # Mild sharpening to prevent stroke blowout/erosion on bright yellow
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.5)
        sharpened = cv2.addWeighted(denoised, 1.4, gaussian, -0.4, 0)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        return clahe.apply(sharpened)
    else:
        # Standard white / default plates
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.8, gaussian, -0.8, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(sharpened)



def atomic_replace_json(target_path: str, data: Dict[str, Any]):
    """Crash-resilient JSON persistence using same-directory temporary file + fsync + os.replace."""
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=target_dir, delete=False, encoding="utf-8", suffix=".tmp") as tf:
        temp_name = tf.name
        json.dump(data, tf, indent=2, ensure_ascii=False)
        tf.write("\n")
        tf.flush()
        os.fsync(tf.fileno())

    try:
        os.replace(temp_name, target_path)
    except Exception as e:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except OSError:
                pass
        raise e


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "model_settings": {
        "detector_weights_path": "models/license_plate_detector.pt",
        "supported_plate_classes": [
            "license_plate",
            "license plate",
            "number_plate",
            "number plate",
            "plate",
        ],
        "ocr_engine": "ppocrv6",
        "ocr_languages": ["en"],
        "ocr_model_storage_directory": "models/ocr",
        "ppocrv6_det_directory": "models/ocr/ppocrv6/PP-OCRv6_medium_det",
        "ppocrv6_rec_directory": "models/ocr/ppocrv6/PP-OCRv6_medium_rec",
        "device": "cpu",
        "allow_cpu_fallback": True,
    },
    "inference_thresholds": {
        "detector_confidence_threshold": 0.20,
        "ocr_confidence_threshold": 0.20,
        "ppocrv6_confidence_threshold": 0.35,
        "combined_confidence_threshold": 0.20,
        "box_padding_ratio": 0.06,
    },
    "plate_validation": {
        "ocr_allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "min_plate_length": 4,
        "max_plate_length": 12,
        "validation_mode": "accept_if_no_match",
        "country_patterns": [
            {
                "country": "IN_STANDARD",
                "description": "Standard Indian High Security Registration Plate (e.g., DL01AB1234, MH12DE1433, KA53MH3895)",
                "regex": "^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{4}$",
                "segments": [
                    {"type": "alpha", "min_len": 2, "max_len": 2},
                    {"type": "numeric", "min_len": 2, "max_len": 2},
                    {"type": "alpha", "min_len": 0, "max_len": 2},
                    {"type": "numeric", "min_len": 4, "max_len": 4},
                ],
            },
            {
                "country": "GB_STANDARD",
                "description": "UK Standard Post-2001 Format (e.g., BD51SMR, YG21DBM)",
                "regex": "^[A-Z]{2}[0-9]{2}[A-Z]{3}$",
                "segments": [
                    {"type": "alpha", "min_len": 2, "max_len": 2},
                    {"type": "numeric", "min_len": 2, "max_len": 2},
                    {"type": "alpha", "min_len": 3, "max_len": 3},
                ],
            },
            {
                "country": "INTL_STANDARD_3A4D",
                "description": "Standard 3-letter 4-digit format (e.g., GAE5379, KBC6233)",
                "regex": "^[A-Z]{3}[0-9]{4}$",
                "segments": [
                    {"type": "alpha", "min_len": 3, "max_len": 3},
                    {"type": "numeric", "min_len": 4, "max_len": 4},
                ],
            },
            {
                "country": "US_GENERIC",
                "description": "General North American 6-8 alphanumeric sequence",
                "regex": "^[A-Z0-9]{6,8}$",
            },
        ],
    },
    "deduplication": {
        "cooldown_seconds": 30.0,
    },
    "video_ingestion": {
        "default_frame_stride": 3,
        "max_reconnect_attempts": 10,
        "reconnect_backoff_max_seconds": 30.0,
        "stream_buffer_size": 1,
    },
    "persistence": {
        "ledger_path": "data/recognized_license_plate_records.json",
    },
    "logging": {
        "log_level": "INFO",
        "verbose": False,
    },
}


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Read and validate the application configuration schema. Self-heals if empty or missing."""
    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        atomic_replace_json(config_path, DEFAULT_CONFIG)
        print(f"[ALPR INFO] Created default configuration at '{config_path}'.", flush=True)
        return DEFAULT_CONFIG

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                atomic_replace_json(config_path, DEFAULT_CONFIG)
                return DEFAULT_CONFIG
            config = json.loads(content)
    except Exception as e:
        sys.exit(f"[ALPR ERROR] Failed to parse config JSON at '{config_path}': {e}")

    if config.get("schema_version") != 1:
        sys.exit(f"[ALPR ERROR] Unsupported config schema_version: {config.get('schema_version')}. Expected 1.")

    model_settings = config.get("model_settings", {})
    detector_path = model_settings.get("detector_weights_path")
    if not detector_path:
        sys.exit("[ALPR ERROR] Missing 'detector_weights_path' in configuration.")

    return config


def load_and_validate_ledger(ledger_path: str) -> Tuple[Dict[str, Any], Dict[str, datetime]]:
    """Load canonical detection records ledger and build recent deduplication cache.
    Safely initializes empty or missing ledger files without crashing.
    """
    dedup_cache: Dict[str, datetime] = {}

    # If missing or 0 bytes (empty file), safely initialize
    if not os.path.exists(ledger_path) or os.path.getsize(ledger_path) == 0:
        initial_ledger = {
            "schema_version": 1,
            "updated_at": get_utc_now_iso(),
            "records": [],
        }
        atomic_replace_json(ledger_path, initial_ledger)
        return initial_ledger, dedup_cache

    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                initial_ledger = {
                    "schema_version": 1,
                    "updated_at": get_utc_now_iso(),
                    "records": [],
                }
                atomic_replace_json(ledger_path, initial_ledger)
                return initial_ledger, dedup_cache
            ledger_data = json.loads(content)
    except Exception as e:
        sys.exit(
            f"[ALPR ERROR] Detection ledger at '{ledger_path}' is corrupted or unreadable: {e}. "
            "Halting immediately to prevent overwriting existing historical data."
        )

    if not isinstance(ledger_data, dict):
        sys.exit(f"[ALPR ERROR] Ledger root at '{ledger_path}' must be a JSON object.")

    if ledger_data.get("schema_version") != 1:
        sys.exit(f"[ALPR ERROR] Unsupported ledger schema_version: {ledger_data.get('schema_version')}. Expected 1.")

    records = ledger_data.get("records")
    if not isinstance(records, list):
        sys.exit(f"[ALPR ERROR] Ledger 'records' field must be a list in '{ledger_path}'.")

    # Validate existing records and populate in-memory deduplication state
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            sys.exit(f"[ALPR ERROR] Record #{idx} in '{ledger_path}' is not an object.")

        required_keys = ("timestamp", "detected_plate_text", "normalized_plate_text", "confidence_score")
        for k in required_keys:
            if k not in rec:
                sys.exit(f"[ALPR ERROR] Record #{idx} missing required key '{k}'.")

        try:
            ts = parse_iso_utc(rec["timestamp"])
            norm_text = rec["normalized_plate_text"]
            if norm_text:
                if norm_text not in dedup_cache or ts > dedup_cache[norm_text]:
                    dedup_cache[norm_text] = ts
        except Exception as e:
            sys.exit(f"[ALPR ERROR] Record #{idx} has invalid timestamp '{rec.get('timestamp')}': {e}")

    # Deduplicate existing exact passage duplicates safely
    unique_records = []
    seen_passages = set()
    for rec in records:
        passage_sig = (rec.get("source_id"), rec.get("frame_index"), rec.get("normalized_plate_text"))
        if passage_sig not in seen_passages:
            seen_passages.add(passage_sig)
            unique_records.append(rec)

    if len(unique_records) < len(records):
        ledger_data["records"] = unique_records
        try:
            atomic_replace_json(ledger_path, ledger_data)
        except Exception:
            pass

    return ledger_data, dedup_cache


def verify_and_load_detector(weights_path: str, supported_classes: List[str], device: str) -> Tuple[Any, Set[int]]:
    """Load local YOLO model and verify presence of supported license plate class."""
    if not os.path.exists(weights_path):
        sys.exit(
            f"[ALPR ERROR] Local detector weights not found at '{weights_path}'. "
            "Please run 'prepare_local_model_assets.py' once to prepare offline weights."
        )

    try:
        from ultralytics import YOLO
        try:
            from ultralytics import settings
            settings.update({"sync": False})
        except Exception:
            pass
    except ImportError:
        sys.exit("[ALPR ERROR] 'ultralytics' library is not installed in the active Python environment.")

    try:
        model = YOLO(weights_path)
    except Exception as e:
        sys.exit(f"[ALPR ERROR] Failed to load YOLO weights from '{weights_path}': {e}")

    model_names = model.names
    if not model_names or not isinstance(model_names, dict):
        sys.exit(f"[ALPR ERROR] Incompatible detector: Model names dictionary is empty or malformed.")

    class_name_to_id = {name.lower().strip(): idx for idx, name in model_names.items()}
    plate_class_ids = set()

    for cls_name in supported_classes:
        key = cls_name.lower().strip()
        if key in class_name_to_id:
            plate_class_ids.add(class_name_to_id[key])

    if not plate_class_ids:
        if len(model_names) == 1:
            plate_class_ids.add(list(model_names.keys())[0])
            print(f"[ALPR INFO] Single-class detector mapped class 0 ({list(model_names.values())[0]}) as license plate.", flush=True)
        else:
            sys.exit(
                f"[ALPR ERROR] Model classes {list(model_names.values())} do not match any configured plate classes {supported_classes}."
            )

    return model, plate_class_ids


class OCRItem(tuple):
    """3-tuple (box, text, confidence) with attached character-level confidence metadata."""
    def __new__(cls, box: Any, text: str, confidence: float, char_confidences: Optional[List[Dict[str, Any]]] = None):
        inst = super().__new__(cls, (box, text, confidence))
        inst.box = box
        inst.text = text
        inst.confidence = confidence
        inst.char_confidences = char_confidences if char_confidences is not None else []
        return inst


class BaseOCREngine:
    """Abstract base adapter for local offline OCR engines."""
    name: str = "base"

    def readtext(self, image: Any, allowlist: Optional[str] = None, **kwargs) -> List[Any]:
        raise NotImplementedError


class FastPlateOCREngine(BaseOCREngine):
    """FastPlateOCR local offline recognition engine adapter using cct-s-v2-global-model."""
    name = "fastplateocr"

    def __init__(self, recognizer: Any, device: str = "cpu"):
        self.recognizer = recognizer
        self.device = device

    def readtext(self, image: Any, allowlist: Optional[str] = None, **kwargs) -> List[OCRItem]:
        import cv2
        import numpy as np

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        # FastPlateOCR cct-s-v2-global-model expects 3-channel RGB uint8 image
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image

        try:
            predictions = self.recognizer.run(img_rgb, return_confidence=True, remove_pad_char=True)
        except Exception as e:
            print(f"[ALPR WARN] FastPlateOCR inference exception: {e}", file=sys.stderr, flush=True)
            return []

        results: List[OCRItem] = []
        for pred in predictions:
            raw_plate = pred.plate.strip() if pred.plate else ""
            if not raw_plate:
                continue

            raw_probs = pred.char_probs
            # Strictly align char_probs with plate text
            if raw_probs is not None:
                aligned_probs = [float(p) for p in raw_probs[:len(raw_plate)]]
            else:
                aligned_probs = [1.0] * len(raw_plate)

            if allowlist is not None:
                filtered_chars = []
                filtered_probs = []
                for char, prob in zip(raw_plate, aligned_probs):
                    if char in allowlist:
                        filtered_chars.append(char)
                        filtered_probs.append(prob)
                clean_plate = "".join(filtered_chars)
                aligned_probs = filtered_probs
            else:
                clean_plate = raw_plate

            if not clean_plate:
                continue

            char_confs = [
                {"char": c, "confidence": round(float(p), 4)}
                for c, p in zip(clean_plate, aligned_probs)
            ]
            overall_conf = float(np.mean(aligned_probs)) if aligned_probs else 0.0
            results.append(OCRItem(None, clean_plate, overall_conf, char_confs))

        return results


class PPOCRv6Engine(BaseOCREngine):
    """PaddleOCR PP-OCRv6 local offline recognition engine adapter."""
    name = "ppocrv6"

    def __init__(self, reader: Any, device: str = "cpu"):
        self.reader = reader
        self.device = device

    def readtext(self, image: Any, allowlist: Optional[str] = None, **kwargs) -> List[OCRItem]:
        import cv2
        import numpy as np

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        # PaddleOCR detection/recognition expects a 3-channel BGR uint8 image
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = image

        try:
            pages = self.reader.predict(img_bgr)
        except Exception as e:
            print(f"[ALPR WARN] PP-OCRv6 inference exception: {e}", file=sys.stderr, flush=True)
            return []

        results: List[OCRItem] = []
        for page in pages:
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            boxes_raw = page.get("rec_boxes")
            if boxes_raw is None or len(boxes_raw) == 0:
                boxes_raw = page.get("dt_polys")
            boxes = boxes_raw if (boxes_raw is not None and len(boxes_raw) > 0) else []

            for i, (txt, score) in enumerate(zip(rec_texts, rec_scores)):
                clean_txt = str(txt).strip()
                if allowlist is not None:
                    clean_txt = "".join(c for c in clean_txt if c in allowlist)
                if clean_txt:
                    box = boxes[i] if i < len(boxes) else None
                    results.append(OCRItem(box, clean_txt, float(score), []))

        return results


class EasyOCREngine(BaseOCREngine):
    """EasyOCR local offline recognition engine adapter (rollback engine)."""
    name = "easyocr"

    def __init__(self, reader: Any, device: str = "cpu"):
        self.reader = reader
        self.device = device

    def readtext(self, image: Any, allowlist: Optional[str] = None, **kwargs) -> List[OCRItem]:
        import numpy as np

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        try:
            raw_res = self.reader.readtext(
                image,
                allowlist=allowlist,
                detail=1,
                paragraph=False,
            )
        except Exception as e:
            print(f"[ALPR WARN] EasyOCR inference exception: {e}", file=sys.stderr, flush=True)
            return []

        results: List[OCRItem] = []
        for item in raw_res:
            box = item[0]
            txt = str(item[1]).strip()
            conf = float(item[2])
            if txt:
                results.append(OCRItem(box, txt, conf, []))
        return results


def verify_and_load_fastplateocr(model_path: str, config_path: str, device: str = "cpu") -> FastPlateOCREngine:
    """Initialize local offline FastPlateOCR engine strictly with local model assets."""
    if not os.path.exists(model_path):
        sys.exit(
            f"[ALPR ERROR] FastPlateOCR ONNX model file not found at '{model_path}'. "
            "Please run 'prepare_local_model_assets.py' to download offline weights."
        )
    if not os.path.exists(config_path):
        sys.exit(
            f"[ALPR ERROR] FastPlateOCR config YAML file not found at '{config_path}'. "
            "Please run 'prepare_local_model_assets.py' to download offline weights."
        )

    try:
        from fast_plate_ocr import LicensePlateRecognizer
    except ImportError as e:
        sys.exit(
            f"[ALPR ERROR] 'fast-plate-ocr' or 'onnxruntime' is not installed in the active Python environment: {e}.\n"
            "Please install via: pip install fast-plate-ocr onnxruntime"
        )

    try:
        device_lower = str(device).lower().strip()
        provider_device = "cuda" if "cuda" in device_lower else "cpu"
        recognizer = LicensePlateRecognizer(
            onnx_model_path=model_path,
            plate_config_path=config_path,
            device=provider_device,
        )
        return FastPlateOCREngine(recognizer, device=device)
    except Exception as e:
        sys.exit(
            f"[ALPR ERROR] Failed to initialize offline FastPlateOCR recognizer: {e}. "
            "Verify local ONNX weights and ONNX Runtime CPU compatibility."
        )


def verify_and_load_ppocrv6(det_dir: str, rec_dir: str, device: str) -> PPOCRv6Engine:
    """Initialize local offline PaddleOCR PP-OCRv6 engine strictly with local model assets."""
    if not os.path.exists(det_dir):
        sys.exit(
            f"[ALPR ERROR] PP-OCRv6 detection model directory not found at '{det_dir}'. "
            "Please ensure offline detection weights exist in models/ocr/ppocrv6/."
        )
    if not os.path.exists(rec_dir):
        sys.exit(
            f"[ALPR ERROR] PP-OCRv6 recognition model directory not found at '{rec_dir}'. "
            "Please ensure offline recognition weights exist in models/ocr/ppocrv6/."
        )

    for d_name, d_path in [("detection", det_dir), ("recognition", rec_dir)]:
        param_file = os.path.join(d_path, "inference.pdiparams")
        json_file = os.path.join(d_path, "inference.json")
        if not (os.path.exists(param_file) and os.path.exists(json_file)):
            sys.exit(
                f"[ALPR ERROR] Required PP-OCRv6 {d_name} files ('inference.pdiparams' and 'inference.json') "
                f"were not found in '{d_path}'. Local offline operation cannot proceed."
            )

    try:
        os.environ.setdefault("FLAGS_use_onednn", "0")
        from paddleocr import PaddleOCR
    except ImportError as e:
        sys.exit(
            f"[ALPR ERROR] 'paddleocr' or 'paddlepaddle' is not installed in the active Python environment: {e}.\n"
            "Note: On Windows, PaddlePaddle requires Python <= 3.12. Use the project virtual environment:\n"
            r"      .\.venv\Scripts\python.exe vehicle_license_plate_recognition.py ..."
        )

    try:
        reader = PaddleOCR(
            text_detection_model_dir=det_dir,
            text_recognition_model_dir=rec_dir,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        return PPOCRv6Engine(reader, device=device)
    except Exception as e:
        sys.exit(
            f"[ALPR ERROR] Failed to initialize offline PaddleOCR PP-OCRv6: {e}. "
            "Verify local model assets and PaddlePaddle CPU compatibility."
        )


def verify_and_load_ocr(ocr_storage_dir: str, languages: List[str], device: str) -> EasyOCREngine:
    """Initialize local offline EasyOCR reader with network downloads strictly disabled."""
    if not os.path.exists(ocr_storage_dir):
        sys.exit(
            f"[ALPR ERROR] OCR model directory not found at '{ocr_storage_dir}'. "
            "Please run 'prepare_local_model_assets.py' to download offline weights."
        )

    craft_file = os.path.join(ocr_storage_dir, "craft_mlt_25k.pth")
    if not os.path.exists(craft_file):
        sys.exit(f"[ALPR ERROR] Required CRAFT detector model not found at '{craft_file}'.")

    try:
        import easyocr
    except ImportError:
        sys.exit("[ALPR ERROR] 'easyocr' library is not installed in the active Python environment.")

    use_gpu = ("cuda" in device.lower()) if device else False

    try:
        reader = easyocr.Reader(
            lang_list=languages,
            gpu=use_gpu,
            model_storage_directory=ocr_storage_dir,
            download_enabled=False,
            verbose=False,
        )
        return EasyOCREngine(reader, device=device)
    except Exception as e:
        sys.exit(
            f"[ALPR ERROR] Failed to initialize offline EasyOCR reader: {e}. "
            "Ensure required language weights exist in the local storage directory."
        )


def get_default_ocr_engine(preferred: str = "fastplateocr") -> str:
    """Resolve default OCR engine: fastplateocr (primary default), ppocrv6, or easyocr."""
    if preferred == "fastplateocr":
        try:
            import fast_plate_ocr  # noqa: F401
            import onnxruntime  # noqa: F401
            return "fastplateocr"
        except ImportError:
            try:
                import paddleocr  # noqa: F401
                return "ppocrv6"
            except ImportError:
                return "easyocr"
    elif preferred == "ppocrv6":
        try:
            import paddleocr  # noqa: F401
            return "ppocrv6"
        except ImportError:
            return "easyocr"
    return preferred


def compute_multiframe_consensus(
    candidates: List[Dict[str, Any]],
    country_patterns: Optional[List[Dict[str, Any]]] = None,
    char_low_conf_thresh: float = 0.60,
) -> Tuple[str, float, List[Dict[str, Any]], List[str], Optional[str]]:
    """Compute confidence-aware positional consensus across multiple observations of a plate.

    Aligns character positions and aggregates evidence weighted by character-level confidences,
    frame sharpness, and structural national pattern adherence. Preserves disputed characters
    and flags them in review_required rather than silently guessing.
    """
    if not candidates:
        return "", 0.0, [], [], None

    patterns = country_patterns or []

    if len(candidates) == 1:
        c = candidates[0]
        plate = c.get("plate", "")
        conf = float(c.get("conf", 0.0))
        char_confs = c.get("character_confidences", [])
        if not char_confs and plate:
            char_confs = [{"char": ch, "confidence": conf} for ch in plate]
        review_flags = [ch["char"] for ch in char_confs if ch.get("confidence", 1.0) < char_low_conf_thresh]
        matched_country = None
        for pat in patterns:
            regex_pat = pat.get("regex")
            if regex_pat and re.match(regex_pat, plate):
                matched_country = pat.get("country")
                break
        return plate, conf, char_confs, review_flags, matched_country

    import numpy as np

    # 1. Determine dominant plate length by summing confidence and pattern adherence
    length_weights: Dict[int, float] = {}
    for c in candidates:
        plate_str = c.get("plate", "")
        if plate_str:
            L = len(plate_str)
            w = float(c.get("conf", 0.5))
            # Award structural bonus to candidates matching standard national formats
            for pat in patterns:
                reg = pat.get("regex")
                if reg and pat.get("country") != "US_GENERIC" and re.match(reg, plate_str):
                    w += 2.0
                    break
            length_weights[L] = length_weights.get(L, 0.0) + w

    if not length_weights:
        return "", 0.0, [], [], None

    target_length = max(length_weights.keys(), key=lambda l: length_weights[l])
    matching_candidates = [c for c in candidates if len(c.get("plate", "")) == target_length]
    if not matching_candidates:
        matching_candidates = candidates
        target_length = len(candidates[-1].get("plate", ""))

    consensus_chars: List[str] = []
    consensus_confs: List[float] = []
    review_flags: List[str] = []

    for i in range(target_length):
        char_votes: Dict[str, List[Tuple[float, float]]] = {}
        for c in matching_candidates:
            p = c.get("plate", "")
            if i >= len(p):
                continue
            char_at_i = p[i]
            c_confs = c.get("character_confidences", [])
            if c_confs and i < len(c_confs):
                c_conf = float(c_confs[i].get("confidence", c.get("conf", 0.5)))
            else:
                c_conf = float(c.get("conf", 0.5))
            sharpness = float(c.get("sharpness", 0.0))
            sharp_factor = 1.0 + min(1.0, sharpness / 1000.0) if sharpness > 0 else 1.0
            vote_weight = c_conf * sharp_factor
            char_votes.setdefault(char_at_i, []).append((vote_weight, c_conf))

        if not char_votes:
            continue

        aggregated = {
            ch: (
                sum(w for w, _ in entries),
                float(np.mean([conf for _, conf in entries])),
                len(entries),
            )
            for ch, entries in char_votes.items()
        }

        winner_char = max(aggregated.keys(), key=lambda ch: aggregated[ch][0])
        winner_sum, winner_mean, winner_count = aggregated[winner_char]

        consensus_chars.append(winner_char)
        consensus_confs.append(round(min(1.0, max(0.0, winner_mean)), 4))

        # Check for ambiguity: competing candidate has close score and repeated observations
        is_ambiguous = False
        if len(aggregated) > 1:
            competing = [ch for ch in aggregated if ch != winner_char]
            second_char = max(competing, key=lambda ch: aggregated[ch][0])
            second_sum, second_mean, second_count = aggregated[second_char]
            if (winner_sum - second_sum < 0.30) and second_count >= 2:
                is_ambiguous = True

        if winner_mean < char_low_conf_thresh or is_ambiguous:
            if winner_char not in review_flags:
                review_flags.append(winner_char)

    consensus_plate = "".join(consensus_chars)
    consensus_overall_conf = round(float(np.mean(consensus_confs)), 4) if consensus_confs else 0.0
    consensus_char_confs = [
        {"char": c, "confidence": conf} for c, conf in zip(consensus_chars, consensus_confs)
    ]

    matched_country = None
    for pat in patterns:
        regex_pat = pat.get("regex")
        if regex_pat and re.match(regex_pat, consensus_plate):
            matched_country = pat.get("country")
            break

    # Format-aware pattern disambiguation (e.g. for Indian HSRP or UK formats)
    if patterns:
        disambiguated_plate, disambiguated_country, _ = match_and_disambiguate_pattern(consensus_plate, patterns)
        if disambiguated_country and (matched_country is None or matched_country == "US_GENERIC"):
            matched_country = disambiguated_country
            if disambiguated_plate != consensus_plate and len(disambiguated_plate) == len(consensus_plate):
                new_chars = []
                for orig_cd, new_ch in zip(consensus_char_confs, disambiguated_plate):
                    new_chars.append({"char": new_ch, "confidence": orig_cd["confidence"]})
                consensus_plate = disambiguated_plate
                consensus_char_confs = new_chars

    return consensus_plate, consensus_overall_conf, consensus_char_confs, review_flags, matched_country



def open_capture_stream(source: str):
    """Open OpenCV VideoCapture supporting local video files, webcams, and network streams."""
    import cv2

    source_str = str(source).strip()
    if source_str.isdigit():
        cap = cv2.VideoCapture(int(source_str))
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
        return cap, "camera"

    is_stream = any(source_str.lower().startswith(p) for p in ("rtsp://", "rtmp://", "http://", "https://", "tcp://"))
    if is_stream:
        cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
        return cap, "stream"

    if not os.path.exists(source_str):
        sys.exit(f"[ALPR ERROR] Local video source file not found: '{source_str}'")

    cap = cv2.VideoCapture(source_str)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    return cap, "file"


def run_pipeline(args):
    """Main execution loop for license plate detection, recognition, and persistence."""
    import cv2
    import numpy as np

    # 1. Config loading
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.abspath(args.config if args.config else os.path.join(base_dir, "config", "license_plate_recognition_config.json"))
    config = load_and_validate_config(config_path)

    # 2. Paths and ledger resolution
    data_dir = os.path.abspath(os.path.join(base_dir, "data"))
    os.makedirs(data_dir, exist_ok=True)

    ledger_path = os.path.abspath(args.ledger if args.ledger else os.path.join(data_dir, "recognized_license_plate_records.json"))
    ledger_data, dedup_cache = load_and_validate_ledger(ledger_path)

    # 3. Model assets resolution
    model_settings = config.get("model_settings", {})
    weights_arg = getattr(args, "weights", None)
    weights_path = os.path.abspath(weights_arg if weights_arg else os.path.join(base_dir, model_settings.get("detector_weights_path", "models/license_plate_detector.pt")))
    ocr_engine_arg = getattr(args, "ocr_engine", None)
    if ocr_engine_arg:
        ocr_engine_type = ocr_engine_arg.lower().strip()
    else:
        cfg_engine = model_settings.get("ocr_engine", "fastplateocr").lower().strip()
        ocr_engine_type = get_default_ocr_engine(cfg_engine)
    if ocr_engine_type not in ("fastplateocr", "ppocrv6", "easyocr"):
        sys.exit(f"[ALPR ERROR] Unsupported OCR engine '{ocr_engine_type}'. Supported engines are 'fastplateocr', 'ppocrv6', and 'easyocr'.")

    device = getattr(args, "device", None) or model_settings.get("device", "cpu")
    supported_classes = model_settings.get("supported_plate_classes", ["license_plate", "plate"])
    ocr_langs = model_settings.get("ocr_languages", ["en"])

    # 4. Acquire single-writer lock
    clean_source_id = os.path.basename(str(getattr(args, "source", "unknown")))
    acquire_instance_lock(data_dir, clean_source_id)

    # 5. Initialize detector and OCR
    print(f"[ALPR] Loading local detector from: {weights_path}", flush=True)
    model, plate_class_ids = verify_and_load_detector(weights_path, supported_classes, device)

    thresh_cfg = config.get("inference_thresholds", {})
    char_low_conf_thresh = thresh_cfg.get("char_low_confidence_threshold", 0.60)
    if ocr_engine_type == "fastplateocr":
        fp_model_arg = getattr(args, "fastplateocr_model", None)
        fp_cfg_arg = getattr(args, "fastplateocr_config", None)
        fp_model = os.path.abspath(fp_model_arg if fp_model_arg else os.path.join(base_dir, model_settings.get("fastplateocr_model_path", "models/ocr/fastplateocr/cct_s_v2_global.onnx")))
        fp_cfg = os.path.abspath(fp_cfg_arg if fp_cfg_arg else os.path.join(base_dir, model_settings.get("fastplateocr_config_path", "models/ocr/fastplateocr/cct_s_v2_global_plate_config.yaml")))
        print(f"[ALPR] Initializing offline FastPlateOCR cct-s-v2-global-model from: {fp_model}", flush=True)
        ocr_reader = verify_and_load_fastplateocr(fp_model, fp_cfg, device)
        default_ocr_conf = thresh_cfg.get("fastplateocr_confidence_threshold", 0.35)
    elif ocr_engine_type == "ppocrv6":
        det_dir_arg = getattr(args, "ppocr_det_dir", None)
        rec_dir_arg = getattr(args, "ppocr_rec_dir", None)
        det_dir = os.path.abspath(det_dir_arg if det_dir_arg else os.path.join(base_dir, model_settings.get("ppocrv6_det_directory", "models/ocr/ppocrv6/PP-OCRv6_medium_det")))
        rec_dir = os.path.abspath(rec_dir_arg if rec_dir_arg else os.path.join(base_dir, model_settings.get("ppocrv6_rec_directory", "models/ocr/ppocrv6/PP-OCRv6_medium_rec")))
        print(f"[ALPR] Initializing offline PaddleOCR PP-OCRv6 reader (Det: {det_dir}, Rec: {rec_dir})", flush=True)
        ocr_reader = verify_and_load_ppocrv6(det_dir, rec_dir, device)
        default_ocr_conf = thresh_cfg.get("ppocrv6_confidence_threshold", 0.40)
    else:
        ocr_dir_arg = getattr(args, "ocr_dir", None)
        ocr_dir = os.path.abspath(ocr_dir_arg if ocr_dir_arg else os.path.join(base_dir, model_settings.get("ocr_model_storage_directory", "models/ocr")))
        print(f"[ALPR] Initializing offline EasyOCR rollback reader from: {ocr_dir}", flush=True)
        ocr_reader = verify_and_load_ocr(ocr_dir, ocr_langs, device)
        default_ocr_conf = thresh_cfg.get("ocr_confidence_threshold", 0.20)

    # 6. Open Video Source
    cap, source_type = open_capture_stream(args.source)
    if not cap.isOpened():
        sys.exit(f"[ALPR ERROR] Failed to open video source '{args.source}' ({source_type}).")

    # Ingestion parameters
    ingestion_cfg = config.get("video_ingestion", {})
    frame_stride = max(1, args.frame_stride if args.frame_stride is not None else ingestion_cfg.get("default_frame_stride", 3))
    max_reconnects = ingestion_cfg.get("max_reconnect_attempts", 10)
    reconnect_backoff_max = ingestion_cfg.get("reconnect_backoff_max_seconds", 30.0)

    # Thresholds
    det_conf_thresh = args.det_conf if args.det_conf is not None else thresh_cfg.get("detector_confidence_threshold", 0.25)
    ocr_conf_thresh = args.ocr_conf if args.ocr_conf is not None else default_ocr_conf
    combined_conf_thresh = thresh_cfg.get("combined_confidence_threshold", 0.20)
    box_padding_ratio = thresh_cfg.get("box_padding_ratio", 0.05)

    # Validation rules
    val_cfg = config.get("plate_validation", {})
    ocr_allowlist = val_cfg.get("ocr_allowlist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    min_length = val_cfg.get("min_plate_length", 4)
    max_length = val_cfg.get("max_plate_length", 12)
    val_mode = val_cfg.get("validation_mode", "accept_if_no_match")
    country_patterns = val_cfg.get("country_patterns", [])

    # Deduplication
    cooldown_seconds = args.dedup_seconds if args.dedup_seconds is not None else config.get("deduplication", {}).get("cooldown_seconds", 30.0)

    print(f"[ALPR] System ready. Source: '{args.source}' | Stride: {frame_stride} | Cooldown: {cooldown_seconds}s", flush=True)
    print(f"[ALPR] Persisting records to: {ledger_path}", flush=True)
    if args.preview:
        print("[ALPR] Live preview enabled. Press 'q' or 'ESC' on the preview window to exit.", flush=True)

    meta_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(cv2, "CAP_PROP_FRAME_WIDTH") else 0
    meta_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(cv2, "CAP_PROP_FRAME_HEIGHT") else 0
    meta_fps = float(cap.get(cv2.CAP_PROP_FPS)) if hasattr(cv2, "CAP_PROP_FPS") else 0.0
    orient_auto = float(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)) if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") else 0.0
    orient_meta = float(cap.get(cv2.CAP_PROP_ORIENTATION_META)) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0.0

    configured_imgsz = getattr(args, "imgsz", None) or model_settings.get("detector_input_size")
    user_imgsz = int(configured_imgsz) if configured_imgsz and str(configured_imgsz).isdigit() else None

    user_max_frames = getattr(args, "max_frames", None)
    frame_index = 0
    reconnect_count = 0
    shutdown_requested = False

    def handle_signal(sig, frame):
        nonlocal shutdown_requested
        print("\n[ALPR] Graceful shutdown requested by user.", flush=True)
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not shutdown_requested:
            if user_max_frames is not None and frame_index >= user_max_frames:
                print(f"[ALPR] Processed max-frames limit ({user_max_frames}). Stopping cleanly.", flush=True)
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                if source_type in ("camera", "stream"):
                    reconnect_count += 1
                    if reconnect_count > max_reconnects:
                        print(f"[ALPR ERROR] Exceeded maximum reconnect attempts ({max_reconnects}). Halting.", flush=True)
                        break
                    backoff = min(reconnect_backoff_max, 1.5 ** reconnect_count)
                    print(f"[ALPR WARN] Stream interrupted. Reconnect attempt {reconnect_count}/{max_reconnects} in {backoff:.1f}s...", flush=True)
                    time.sleep(backoff)
                    cap.release()
                    cap, _ = open_capture_stream(args.source)
                    continue
                else:
                    print("[ALPR] Reached end of video file.", flush=True)
                    break

            reconnect_count = 0
            frame_index += 1

            # Respect video rotation/orientation metadata, applying rotation exactly once
            if orient_auto == 0.0 and orient_meta in (90.0, 180.0, 270.0):
                if orient_meta == 90.0:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif orient_meta == 180.0:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif orient_meta == 270.0:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            h, w = frame.shape[:2]

            # Determine detector input resolution (imgsz): adaptively sized for high-res/4K to preserve small plates
            if user_imgsz is not None:
                det_imgsz = user_imgsz
            else:
                max_dim = max(h, w)
                if max_dim >= 1920:
                    det_imgsz = 1920
                elif max_dim >= 1280:
                    det_imgsz = 1280
                else:
                    det_imgsz = 640

            if frame_index == 1:
                disp_info = "aspect-fit <= 1280x720" if args.preview else "disabled"
                print(
                    f"[ALPR] Video geometry: Source metadata: {meta_w}x{meta_h} ({meta_fps:.1f} fps) | "
                    f"Decoded frame: {w}x{h} | Detector imgsz: {det_imgsz} | Preview: {disp_info}",
                    flush=True,
                )

            if frame_index % frame_stride != 0:
                continue

            results = model.predict(frame, imgsz=det_imgsz, conf=det_conf_thresh, device=device, verbose=False)

            detected_boxes = []
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if cls_id in plate_class_ids and conf >= det_conf_thresh:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            # Clip bounding box
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            bw = x2 - x1
                            bh = y2 - y1
                            aspect_ratio = bw / max(1, bh)

                            # Geometric sanity validation:
                            # 1. License plates have standard aspect ratios between 1.05 and 6.5
                            # 2. Reject boxes that exceed 20% of frame height (e.g. storefront signs, banners)
                            # 3. Reject boxes that are impossibly narrow or short
                            # 4. Reject camera DVR OSD watermark banners (e.g. "D RAMP ENTRY MAIN ROAD" at bottom margin)
                            if y2 >= (h * 0.88) and x1 <= (w * 0.25):
                                continue
                            if not (1.05 <= aspect_ratio <= 6.5):
                                continue
                            if bh > (h * 0.20) or bw > (w * 0.60):
                                continue
                            if bw >= 15 and bh >= 10:
                                detected_boxes.append((conf, x1, y1, x2, y2))

            preview_labels = []

            for det_conf, x1, y1, x2, y2 in detected_boxes:
                # Add safe padding around plate crop to prevent character edge truncation
                pad_w = max(4, int((x2 - x1) * max(0.08, box_padding_ratio)))
                pad_h = max(3, int((y2 - y1) * max(0.06, box_padding_ratio)))
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(w, x2 + pad_w)
                cy2 = min(h, y2 + pad_h)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                detected_color = classify_plate_color(crop)
                proc_crop = enhance_plate_crop(crop, color_hint=detected_color)
                ch, cw = crop.shape[:2]
                crop_ar = cw / max(1, ch)
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

                # Candidate image evaluation: FastPlateOCR is trained on natural plate crops,
                # so raw crop is passed directly (per FastPlateOCR specs). EasyOCR uses enhanced/inverted passes.
                ocr_candidates = []
                candidate_imgs = [crop] if ocr_engine_type == "fastplateocr" else [proc_crop, crop]
                for candidate_img in candidate_imgs:
                    if candidate_img is None:
                        continue
                    try:
                        res = ocr_reader.readtext(
                            candidate_img,
                            allowlist=ocr_allowlist,
                            detail=1,
                            paragraph=False,
                        )
                        if res:
                            txt = " ".join([item[1].strip() for item in res if item[1].strip()])
                            confs = [item[2] for item in res if item[1].strip()]
                            char_confs = []
                            for item in res:
                                char_confs.extend(getattr(item, "char_confidences", []))
                            if txt and confs:
                                ocr_candidates.append((res, float(np.mean(confs)), txt, char_confs))
                    except Exception:
                        pass

                # Two-line / stacked plate recognition (e.g. motorcycles, scooters, square plates)
                if ocr_engine_type == "fastplateocr" and 1.05 <= crop_ar <= 2.4:
                    top_crop = crop[:int(ch * 0.54), :]
                    bot_crop = crop[int(ch * 0.44):, :]
                    try:
                        res_t = ocr_reader.readtext(top_crop, allowlist=ocr_allowlist, detail=1, paragraph=False)
                        res_b = ocr_reader.readtext(bot_crop, allowlist=ocr_allowlist, detail=1, paragraph=False)
                        if res_t and res_b:
                            txt_t = "".join([item[1].strip() for item in res_t if item[1].strip()])
                            txt_b = "".join([item[1].strip() for item in res_b if item[1].strip()])
                            confs_t = [item[2] for item in res_t if item[1].strip()]
                            confs_b = [item[2] for item in res_b if item[1].strip()]
                            char_confs_comb = []
                            for item in res_t:
                                char_confs_comb.extend(getattr(item, "char_confidences", []))
                            for item in res_b:
                                char_confs_comb.extend(getattr(item, "char_confidences", []))
                            if txt_t and txt_b and confs_t and confs_b:
                                txt_comb = txt_t + txt_b
                                mean_comb = float(np.mean(confs_t + confs_b))
                                ocr_candidates.append((res_t + res_b, mean_comb, txt_comb, char_confs_comb))
                    except Exception:
                        pass

                # If fastplateocr found nothing on raw crop, fall back to enhanced crop
                if not ocr_candidates and ocr_engine_type == "fastplateocr" and proc_crop is not None:
                    try:
                        res = ocr_reader.readtext(
                            proc_crop,
                            allowlist=ocr_allowlist,
                            detail=1,
                            paragraph=False,
                        )
                        if res:
                            txt = " ".join([item[1].strip() for item in res if item[1].strip()])
                            confs = [item[2] for item in res if item[1].strip()]
                            char_confs = []
                            for item in res:
                                char_confs.extend(getattr(item, "char_confidences", []))
                            if txt and confs:
                                ocr_candidates.append((res, float(np.mean(confs)), txt, char_confs))
                    except Exception:
                        pass

                # Inversion pass only for EasyOCR on green plates (CRAFT requires dark-on-light text)
                if ocr_engine_type != "fastplateocr" and detected_color == "GREEN" and proc_crop is not None:
                    try:
                        inv = cv2.bitwise_not(proc_crop if proc_crop.ndim == 2 else cv2.cvtColor(proc_crop, cv2.COLOR_BGR2GRAY))
                        res_inv = ocr_reader.readtext(
                            inv,
                            allowlist=ocr_allowlist,
                            detail=1,
                            paragraph=False,
                        )
                        if res_inv:
                            txt_inv = " ".join([item[1].strip() for item in res_inv if item[1].strip()])
                            confs_inv = [item[2] for item in res_inv if item[1].strip()]
                            char_confs_inv = []
                            for item in res_inv:
                                char_confs_inv.extend(getattr(item, "char_confidences", []))
                            if txt_inv and confs_inv:
                                ocr_candidates.append((res_inv, float(np.mean(confs_inv)), txt_inv, char_confs_inv))
                    except Exception:
                        pass

                if not ocr_candidates:
                    preview_labels.append((x1, y1, x2, y2, f"Plate ({det_conf:.2f}) [{detected_color}]", (0, 165, 255)))
                    continue

                # Select best candidate favoring national pattern matching, valid plate length, and high confidence
                def cand_rank_key(c_tuple):
                    c_txt = normalize_plate_text(c_tuple[2])
                    c_conf = c_tuple[1]
                    p_match, _, _ = match_and_disambiguate_pattern(c_txt, country_patterns)
                    is_national = 1 if p_match is not None and p_match != "US_GENERIC" else 0
                    has_valid_len = 1 if min_length <= len(c_txt) <= max_length else 0
                    return (is_national, has_valid_len, c_conf)

                ocr_candidates.sort(key=cand_rank_key, reverse=True)
                ocr_results, ocr_conf, raw_plate_text, selected_char_confs = ocr_candidates[0]

                if ocr_conf < ocr_conf_thresh:
                    preview_labels.append((x1, y1, x2, y2, f"{raw_plate_text} (low conf)", (0, 165, 255)))
                    continue

                raw_normalized_plate = normalize_plate_text(raw_plate_text)

                # Length constraints
                if not (min_length <= len(raw_normalized_plate) <= max_length):
                    preview_labels.append((x1, y1, x2, y2, f"{raw_normalized_plate} (len)", (0, 165, 255)))
                    continue

                # Align or fallback character confidences to normalized plate length
                if not selected_char_confs:
                    selected_char_confs = [
                        {"char": c, "confidence": round(ocr_conf, 4)}
                        for c in raw_normalized_plate
                    ]
                else:
                    norm_confs = []
                    for cd in selected_char_confs:
                        nc = normalize_plate_text(cd["char"])
                        if nc:
                            norm_confs.append({"char": nc, "confidence": cd["confidence"]})
                    if len(norm_confs) == len(raw_normalized_plate):
                        selected_char_confs = norm_confs

                # Identify low-confidence characters
                review_flags = [c["char"] for c in selected_char_confs if c.get("confidence", 1.0) < char_low_conf_thresh]
                min_char_conf = min([c.get("confidence", 1.0) for c in selected_char_confs]) if selected_char_confs else ocr_conf

                # Regional pattern validation and format-aware disambiguation
                normalized_plate, matched_country, disambiguation_applied = match_and_disambiguate_pattern(
                    raw_normalized_plate, country_patterns
                )
                if val_mode == "require_match" and matched_country is None:
                    preview_labels.append((x1, y1, x2, y2, f"{normalized_plate} (no match)", (0, 165, 255)))
                    continue

                # Filter unpatterned false positives: if no country pattern matches or unstructured US_GENERIC, require at least 1 digit and 1 letter, len >= 5, and sufficient character confidence
                if matched_country is None or matched_country == "US_GENERIC":
                    has_digit = any(c.isdigit() for c in normalized_plate)
                    has_alpha = any(c.isalpha() for c in normalized_plate)
                    min_conf = 0.45 if matched_country == "US_GENERIC" else 0.50
                    too_many_uncertain_chars = len(review_flags) > (len(normalized_plate) // 2)
                    if not (has_digit and has_alpha and len(normalized_plate) >= 5 and ocr_conf >= min_conf and min_char_conf >= 0.30 and not too_many_uncertain_chars):
                        preview_labels.append((x1, y1, x2, y2, f"{normalized_plate} (noise)", (0, 165, 255)))
                        continue

                # Combined confidence
                combined_score = min(det_conf, ocr_conf)
                if combined_score < combined_conf_thresh:
                    preview_labels.append((x1, y1, x2, y2, f"{normalized_plate} ({combined_score:.2f})", (0, 165, 255)))
                    continue

                now_utc = datetime.now(timezone.utc)
                current_box = [x1, y1, x2, y2]

                # Check if this observation belongs to an existing active vehicle passage
                matched_record = None
                for rec in reversed(ledger_data["records"]):
                    if rec.get("source_id") != clean_source_id:
                        continue

                    existing_plate = rec["normalized_plate_text"]
                    rec_first_frame = rec.get("frame_index", 0)
                    rec_last_frame = rec.get("last_frame_index", rec_first_frame)

                    if source_type == "file":
                        # For video files: passage identity is governed by video frame timeline and plate similarity
                        frame_gap = min(abs(frame_index - rec_first_frame), abs(frame_index - rec_last_frame))

                        # 1. Exact plate match from same file within 750 frames (30s at 25fps)
                        if existing_plate == normalized_plate and frame_gap <= 750:
                            matched_record = rec
                            break

                        # 2. Re-run idempotency: exact same frame range from same file
                        if abs(frame_index - rec_first_frame) <= 30:
                            if levenshtein_distance(existing_plate, normalized_plate) <= 3 or \
                               existing_plate in normalized_plate or normalized_plate in existing_plate:
                                matched_record = rec
                                break

                        # 3. Trajectory continuity during video playback
                        dist = levenshtein_distance(existing_plate, normalized_plate)
                        iou = compute_box_iou(rec["bounding_box"], current_box)
                        cdist = compute_center_distance(rec["bounding_box"], current_box)
                        prefix_match = (len(existing_plate) >= 3 and len(normalized_plate) >= 3 and existing_plate[:3] == normalized_plate[:3])
                        if frame_gap <= 100:
                            if frame_gap <= 30 and (iou > 0.15 or cdist < 150):
                                matched_record = rec
                                break
                            if (dist <= 3 or prefix_match) and (iou > 0.05 or cdist < 220):
                                matched_record = rec
                                break
                            if (existing_plate in normalized_plate or normalized_plate in existing_plate) and cdist < 220:
                                matched_record = rec
                                break
                    else:
                        # Live camera/stream: governed by wall-clock cooldown window
                        try:
                            rec_ts = parse_iso_utc(rec.get("last_observed_timestamp", rec["timestamp"]))
                            elapsed = (now_utc - rec_ts).total_seconds()
                        except Exception:
                            elapsed = 9999.0

                        if elapsed > cooldown_seconds:
                            continue

                        frame_gap = frame_index - rec_last_frame
                        if frame_gap < 0 or frame_gap > 100:
                            continue

                        if existing_plate == normalized_plate:
                            matched_record = rec
                            break

                        dist = levenshtein_distance(existing_plate, normalized_plate)
                        iou = compute_box_iou(rec["bounding_box"], current_box)
                        cdist = compute_center_distance(rec["bounding_box"], current_box)
                        prefix_match = (len(existing_plate) >= 3 and len(normalized_plate) >= 3 and existing_plate[:3] == normalized_plate[:3])

                        if frame_gap <= 30 and (iou > 0.15 or cdist < 150):
                            matched_record = rec
                            break

                        if (dist <= 3 or prefix_match) and (iou > 0.05 or cdist < 220):
                            matched_record = rec
                            break

                        if (existing_plate in normalized_plate or normalized_plate in existing_plate) and cdist < 220:
                            matched_record = rec
                            break

                if matched_record:
                    # Consolidate observation into existing passage record without creating duplicates
                    matched_record["observations_count"] = matched_record.get("observations_count", 1) + 1
                    matched_record["last_frame_index"] = frame_index
                    matched_record["last_observed_timestamp"] = now_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                    matched_record["bounding_box"] = current_box
                    if detected_color and (detected_color != "OTHER" or not matched_record.get("plate_color")):
                        matched_record["plate_color"] = detected_color

                    candidates = matched_record.setdefault("candidate_readings", [])
                    candidates.append({
                        "raw": raw_plate_text,
                        "plate": normalized_plate,
                        "conf": round(ocr_conf, 4),
                        "frame": frame_index,
                        "color": detected_color,
                        "sharpness": round(crop_sharpness, 2),
                        "char_probs": [c["confidence"] for c in selected_char_confs],
                        "character_confidences": selected_char_confs,
                        "review_required": review_flags,
                    })
                    # Bounded memory: keep at most 20 candidate observations per passage
                    if len(candidates) > 20:
                        candidates.pop(0)

                    # Confidence-aware multi-frame positional consensus
                    consensus_plate, consensus_conf, consensus_chars, consensus_review, consensus_country = compute_multiframe_consensus(
                        candidates, country_patterns, char_low_conf_thresh=char_low_conf_thresh
                    )

                    prev_plate = matched_record["normalized_plate_text"]
                    prev_ocr_conf = matched_record.get("ocr_confidence", 0.0)
                    prev_pattern = matched_record.get("matched_country_pattern")

                    prev_is_national = prev_pattern in ("IN_STANDARD", "GB_STANDARD", "INTL_STANDARD_3A4D")
                    new_is_national = consensus_country in ("IN_STANDARD", "GB_STANDARD", "INTL_STANDARD_3A4D")

                    # Apply consensus update protecting verified national patterns from being overwritten by unpatterned noise
                    should_update = False
                    if prev_is_national and not new_is_national:
                        should_update = False
                    elif not prev_is_national and new_is_national:
                        should_update = True
                    elif consensus_plate:
                        if consensus_plate == prev_plate:
                            should_update = (consensus_conf > prev_ocr_conf)
                        else:
                            should_update = (consensus_conf >= prev_ocr_conf or prev_ocr_conf < 0.60)

                    if should_update:
                        if consensus_plate != prev_plate:
                            char_scores_brief = " ".join([f"{c['char']}:{int(round(c['confidence'] * 100))}%" for c in consensus_chars])
                            review_note = f" (Review required: {consensus_review})" if consensus_review else ""
                            print(
                                f"[PASSAGE CONSENSUS UPDATE] Frame {frame_index}: Updating '{prev_plate}' -> '{consensus_plate}' "
                                f"({consensus_country or 'GENERAL'}, Conf: {consensus_conf:.2f} >= {prev_ocr_conf:.2f}) [{char_scores_brief}]{review_note}",
                                flush=True,
                            )
                        matched_record["normalized_plate_text"] = consensus_plate
                        matched_record["detected_plate_text"] = raw_plate_text
                        matched_record["confidence_score"] = round(consensus_conf, 4)
                        matched_record["detector_confidence"] = round(det_conf, 4)
                        matched_record["ocr_confidence"] = round(consensus_conf, 4)
                        matched_record["ocr_engine"] = ocr_reader.name
                        matched_record["character_confidences"] = consensus_chars
                        matched_record["min_character_confidence"] = round(min(c["confidence"] for c in consensus_chars), 4) if consensus_chars else round(consensus_conf, 4)
                        matched_record["review_required"] = consensus_review
                        if consensus_country:
                            matched_record["matched_country_pattern"] = consensus_country

                    dedup_cache[matched_record["normalized_plate_text"]] = now_utc
                    dedup_cache[normalized_plate] = now_utc

                    ledger_data["updated_at"] = matched_record["last_observed_timestamp"]
                    try:
                        atomic_replace_json(ledger_path, ledger_data)
                    except Exception as e:
                        sys.exit(f"[ALPR FATAL] Ledger persistence failed: {e}. Stopping to preserve data integrity.")

                    if matched_record["observations_count"] == 2:
                        c_info = f"({matched_record.get('matched_country_pattern')})" if matched_record.get('matched_country_pattern') else ""
                        color_str = f" [{matched_record.get('plate_color')}]" if matched_record.get('plate_color') else ""
                        char_scores_repr = " ".join([f"{c['char']}:{int(round(c['confidence'] * 100))}%" for c in matched_record.get("character_confidences", [])])
                        review_repr = f" | Review required: {', '.join(matched_record['review_required'])}" if matched_record.get("review_required") else ""
                        print(
                            f"[CONFIRMED RECORD] Vehicle passage confirmed: '{matched_record['normalized_plate_text']}' {c_info}{color_str} | "
                            f"Observations: {matched_record['observations_count']} | Scores: {char_scores_repr}{review_repr} | Frame: {frame_index}",
                            flush=True,
                        )

                    preview_labels.append((x1, y1, x2, y2, f"{matched_record['normalized_plate_text']} [{matched_record.get('plate_color', 'PLATE')}] [PASSAGE #{matched_record['observations_count']}]", (0, 255, 0)))
                    continue

                # New vehicle passage
                new_record = {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": now_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                    "detected_plate_text": raw_plate_text,
                    "normalized_plate_text": normalized_plate,
                    "confidence_score": round(ocr_conf, 4),
                    "detector_confidence": round(det_conf, 4),
                    "ocr_confidence": round(ocr_conf, 4),
                    "ocr_engine": ocr_reader.name,
                    "character_confidences": selected_char_confs,
                    "min_character_confidence": round(min_char_conf, 4),
                    "review_required": review_flags,
                    "matched_country_pattern": matched_country,
                    "disambiguation_applied": disambiguation_applied,
                    "plate_color": detected_color,
                    "source_id": clean_source_id,
                    "frame_index": frame_index,
                    "bounding_box": current_box,
                    "observations_count": 1,
                    "last_frame_index": frame_index,
                    "last_observed_timestamp": now_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                    "candidate_readings": [{
                        "raw": raw_plate_text,
                        "plate": normalized_plate,
                        "conf": round(ocr_conf, 4),
                        "frame": frame_index,
                        "color": detected_color,
                        "sharpness": round(crop_sharpness, 2),
                        "char_probs": [c["confidence"] for c in selected_char_confs],
                        "character_confidences": selected_char_confs,
                        "review_required": review_flags,
                    }],
                }

                # Atomic persistence
                ledger_data["records"].append(new_record)
                ledger_data["updated_at"] = new_record["timestamp"]
                try:
                    atomic_replace_json(ledger_path, ledger_data)
                except Exception as e:
                    ledger_data["records"].pop()
                    sys.exit(f"[ALPR FATAL] Ledger persistence failed: {e}. Stopping to preserve data integrity.")

                dedup_cache[normalized_plate] = now_utc

                country_info = f"({matched_country})" if matched_country else ""
                disambig_info = " [DISAMBIGUATED]" if disambiguation_applied else ""
                color_info = f" [{detected_color}]" if detected_color else ""
                char_scores_str = ""
                if selected_char_confs:
                    char_scores_str = " | Chars: " + " ".join([f"{c['char']}:{int(round(c['confidence'] * 100))}%" for c in selected_char_confs])
                if review_flags:
                    char_scores_str += f" | Review required: {', '.join(review_flags)}"
                print(
                    f"[PASSAGE OBSERVATION] Plate candidate: '{normalized_plate}' {country_info}{disambig_info}{color_info} | "
                    f"OCR: {ocr_conf:.2f} (Det: {det_conf:.2f}){char_scores_str} | Frame: {frame_index} | Time: {new_record['timestamp']}",
                    flush=True,
                )
                preview_labels.append((x1, y1, x2, y2, f"Plate: {normalized_plate} [{detected_color}] ({combined_score:.2f})", (0, 255, 0)))

            # Optional OpenCV Preview
            if args.preview:
                preview_img = frame.copy()
                for x1, y1, x2, y2, label_text, color in preview_labels:
                    cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(preview_img, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Fit into viewport preserving exact aspect ratio without cropping or stretching
                h_prev, w_prev = preview_img.shape[:2]
                max_w, max_h = 1280, 720
                scale = min(max_w / w_prev, max_h / h_prev, 1.0)
                if scale < 1.0:
                    disp_w, disp_h = int(w_prev * scale), int(h_prev * scale)
                    display_frame = cv2.resize(preview_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                else:
                    display_frame = preview_img

                window_name = "Vehicle License Plate Recognition Preview"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, display_frame.shape[1], display_frame.shape[0])
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("[ALPR] Preview window closed by user. Halting.", flush=True)
                    break

    finally:
        cap.release()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        release_instance_lock()
        print(f"[ALPR] Run complete. Total unique vehicle passages stored: {len(ledger_data.get('records', []))}", flush=True)


def build_arg_parser():
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vehicle_license_plate_recognition",
        description="Local-Only Vehicle License Plate Recognition (ALPR) System."
    )
    parser.add_argument("--source", required=True, help="Input source: video file path, webcam index, or RTSP/stream URL.")
    parser.add_argument("--config", default=None, help="Path to custom JSON configuration file.")
    parser.add_argument("--ledger", default=None, help="Path to custom recognized records JSON ledger.")
    parser.add_argument("--weights", default=None, help="Path to custom YOLO detector weights.")
    parser.add_argument("--ocr-engine", choices=["fastplateocr", "ppocrv6", "easyocr"], default=None, help="OCR engine to use: 'fastplateocr' (default primary), 'ppocrv6', or 'easyocr' (rollback).")
    parser.add_argument("--fastplateocr-model", default=None, help="Path to offline FastPlateOCR ONNX model file (.onnx).")
    parser.add_argument("--fastplateocr-config", default=None, help="Path to offline FastPlateOCR plate configuration YAML file.")
    parser.add_argument("--ppocr-det-dir", default=None, help="Directory containing offline PP-OCRv6 detection model files.")
    parser.add_argument("--ppocr-rec-dir", default=None, help="Directory containing offline PP-OCRv6 recognition model files.")
    parser.add_argument("--ocr-dir", default=None, help="Directory containing offline EasyOCR model files (for rollback).")
    parser.add_argument("--frame-stride", type=int, default=None, help="Frame sampling stride (process every Nth frame).")
    parser.add_argument("--det-conf", type=float, default=None, help="Detector confidence threshold [0.0 - 1.0].")
    parser.add_argument("--ocr-conf", type=float, default=None, help="OCR confidence threshold [0.0 - 1.0].")
    parser.add_argument("--dedup-seconds", type=float, default=None, help="Deduplication cooldown window in seconds.")
    parser.add_argument("--device", default=None, help="Inference device: 'cpu', 'cuda', etc.")
    parser.add_argument("--preview", action="store_true", help="Display OpenCV live visual preview window.")
    parser.add_argument("--imgsz", type=int, default=None, help="Detector input image size (e.g., 640, 1024, 1280). Default: auto-adaptive based on frame resolution.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process before stopping.")
    return parser


def check_python_runtime(ocr_engine: Optional[str] = None):
    """Verify runtime compatibility and provide guidance if running under Python 3.14 on Windows."""
    if sys.version_info >= (3, 14):
        # FastPlateOCR runs on native ONNX Runtime without issues on Python 3.14.
        # Warn only if PaddleOCR PP-OCRv6 is requested where binary wheels are not yet on PyPI for 3.14.
        if ocr_engine == "ppocrv6":
            try:
                import paddle  # noqa: F401
            except ImportError:
                print(
                    "[ALPR WARNING] Running under Python 3.14 where PaddlePaddle binary wheels are not yet available on Windows.\n"
                    "PaddleOCR PP-OCRv6 requires the project's Python 3.12 virtual environment:\n"
                    r"    .\.venv\Scripts\python.exe vehicle_license_plate_recognition.py ...",
                    file=sys.stderr,
                    flush=True,
                )


def select_video_file_dialog() -> Optional[str]:
    """Open standard Windows/desktop file dialog to pick a video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Video File for License Plate Recognition",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        if file_path:
            return os.path.normpath(file_path)
        return None
    except Exception as e:
        print(f"[NOTE] Graphical file dialog unavailable ({e}). Falling back to manual entry.")
        return None


def normalize_input_path(raw_path: str, base_dir: str) -> Optional[str]:
    """Strip quotes/whitespace and resolve relative path against base directory."""
    clean = raw_path.strip().strip('"').strip("'").strip()
    if not clean:
        return None
    if not os.path.isabs(clean):
        clean = os.path.join(base_dir, clean)
    return os.path.normpath(clean)


def pause_menu(prompt: str = "\nPress [Enter] to return to menu...") -> None:
    """Safely prompt user to press Enter to return in interactive terminal menus."""
    try:
        input(prompt)
    except (EOFError, KeyboardInterrupt):
        pass


def get_default_menu_settings() -> Dict[str, Any]:
    """Return default settings dictionary for interactive terminal menu."""
    return {
        "preview": True,
        "frame_stride": None,
        "ocr_engine": get_default_ocr_engine("fastplateocr"),
        "dedup_seconds": None,
        "det_conf": None,
        "ocr_conf": None,
        "device": "cpu",
        "imgsz": None,
    }


def show_ledger_summary(base_dir: str):
    """Display summary of stored license plate records from JSON ledger."""
    ledger_path = os.path.join(base_dir, "data", "recognized_license_plate_records.json")
    print("\n" + "=" * 72)
    print("                 Recognized License Plate Records Ledger                ")
    print("=" * 72)
    if not os.path.exists(ledger_path):
        print(f"Ledger file not found at: '{ledger_path}'")
        pause_menu()
        return

    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        print(f"Ledger Path:   {ledger_path}")
        print(f"Last Updated:  {data.get('updated_at', 'unknown')}")
        print(f"Total Records: {len(records)}\n")

        if records:
            print(f"{'#':<4} | {'Plate':<12} | {'Country':<12} | {'Conf':<6} | {'Obs':<6} | {'Source':<18} | Timestamp")
            print("-" * 72)
            recent_records = records[-15:]
            for idx, r in enumerate(recent_records, start=1):
                country = r.get("matched_country_pattern") or "N/A"
                conf = r.get("confidence_score", 0.0)
                obs = r.get("observations_count", 0)
                src = os.path.basename(r.get("source_id", "unknown"))
                ts = r.get("timestamp", "unknown")
                plate = r.get("normalized_plate_text", "")
                print(f"{idx:<4} | {plate:<12} | {country:<12} | {conf:<6.2f} | {obs:<6} | {src:<18} | {ts}")
        else:
            print("No vehicle passage records found in ledger (empty).")
    except Exception as e:
        print(f"[ERROR] Could not parse ledger JSON: {e}")

    pause_menu()


def show_advanced_menu(settings: Dict[str, Any]):
    """Advanced processing settings submenu."""
    while True:
        print("\n" + "=" * 72)
        print("          ALPR Configuration & Advanced Processing Settings             ")
        print("=" * 72)
        preview_status = "[ENABLED]" if settings["preview"] else "[DISABLED]"
        stride_status = f"[{settings['frame_stride']}]" if settings["frame_stride"] is not None else "[Default: 3]"
        cooldown_status = f"[{settings['dedup_seconds']}s]" if settings["dedup_seconds"] is not None else "[Default: 30.0s]"
        det_conf_status = f"[{settings['det_conf']}]" if settings["det_conf"] is not None else "[Default: 0.25]"
        ocr_conf_status = f"[{settings['ocr_conf']}]" if settings["ocr_conf"] is not None else "[Default: auto]"
        imgsz_status = f"[{settings['imgsz']}]" if settings["imgsz"] is not None else "[Default: auto]"

        print(f" [1] Live Preview Window:       {preview_status}")
        print(f" [2] Frame Sampling Stride:     {stride_status}")
        print(f" [3] OCR Recognition Engine:    [{settings['ocr_engine']}] (fastplateocr / ppocrv6 / easyocr)")
        print(f" [4] Deduplication Cooldown:    {cooldown_status}")
        print(f" [5] Detector Conf Threshold:   {det_conf_status}")
        print(f" [6] OCR Conf Threshold:        {ocr_conf_status}")
        print(f" [7] Inference Device:          [{settings['device']}] (cpu / cuda)")
        print(f" [8] Detector Input Size:       {imgsz_status}")
        print(" [9] Reset All to Default Settings")
        print("-" * 72)
        print(" [B] Back to Main Menu")
        print("=" * 72)

        try:
            choice = input("Select setting to configure [1-9, B]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            settings["preview"] = not settings["preview"]
        elif choice == "2":
            try:
                val = input("Enter frame stride (e.g., 1 for every frame, 3 for every 3rd, or Enter for default): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not val:
                settings["frame_stride"] = None
            elif val.isdigit() and int(val) >= 1:
                settings["frame_stride"] = int(val)
            else:
                print("[!] Invalid integer stride. Must be >= 1.")
        elif choice == "3":
            try:
                val = input("Select OCR Engine: [1] FastPlateOCR (Default) | [2] PP-OCRv6 | [3] EasyOCR (Rollback): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                continue
            if val in ("1", "fastplateocr"):
                settings["ocr_engine"] = "fastplateocr"
            elif val in ("2", "ppocrv6"):
                settings["ocr_engine"] = "ppocrv6"
            elif val in ("3", "easyocr"):
                settings["ocr_engine"] = "easyocr"
            else:
                print("[!] Unrecognized engine selection.")
        elif choice == "4":
            try:
                val = input("Enter cooldown window in seconds (e.g., 30.0, or Enter for default): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not val:
                settings["dedup_seconds"] = None
            else:
                try:
                    sec = float(val)
                    if sec >= 0:
                        settings["dedup_seconds"] = sec
                    else:
                        print("[!] Cooldown duration must be non-negative.")
                except ValueError:
                    print("[!] Invalid number format for cooldown.")
        elif choice == "5":
            try:
                val = input("Enter detector confidence [0.05 - 0.95] (or Enter for default 0.25): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not val:
                settings["det_conf"] = None
            else:
                try:
                    c = float(val)
                    if 0.0 <= c <= 1.0:
                        settings["det_conf"] = c
                    else:
                        print("[!] Confidence must be between 0.0 and 1.0.")
                except ValueError:
                    print("[!] Invalid number format.")
        elif choice == "6":
            try:
                val = input("Enter OCR confidence [0.05 - 0.95] (or Enter for default): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not val:
                settings["ocr_conf"] = None
            else:
                try:
                    c = float(val)
                    if 0.0 <= c <= 1.0:
                        settings["ocr_conf"] = c
                    else:
                        print("[!] Confidence must be between 0.0 and 1.0.")
                except ValueError:
                    print("[!] Invalid number format.")
        elif choice == "7":
            try:
                val = input("Select device: [1] cpu | [2] cuda: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                continue
            if val in ("2", "cuda"):
                settings["device"] = "cuda"
            else:
                settings["device"] = "cpu"
        elif choice == "8":
            try:
                val = input("Enter detector image size (e.g., 640, 1024, 1280, or Enter for auto): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not val:
                settings["imgsz"] = None
            elif val.isdigit() and int(val) >= 320:
                settings["imgsz"] = int(val)
            else:
                print("[!] Invalid image size. Must be an integer >= 320.")
        elif choice == "9":
            settings.update(get_default_menu_settings())
            print("[+] All settings reset to defaults.")
        elif choice == "B":
            break
        else:
            print("[!] Unknown option.")


def launch_interactive_menu():
    """Interactive terminal menu entrypoint when launched without CLI arguments."""
    check_python_runtime()
    base_dir = os.path.abspath(os.path.dirname(__file__))
    settings = get_default_menu_settings()

    while True:
        print("\n" + "=" * 72)
        print("      Automatic Vehicle License Plate Recognition (ALPR) System         ")
        print("=" * 72)
        print(f" Active OCR Engine:  {settings['ocr_engine']}")
        print(f" Live Preview:       {'ENABLED' if settings['preview'] else 'DISABLED'}")
        print(f" Inference Device:   {settings['device']}")
        print("-" * 72)
        print(" [1] Process Video File (Browse Dialog...)")
        print(" [2] Process Video File (Enter Path Manually)")
        print(" [3] Process Live Camera / Webcam (e.g., Index 0, 1)")
        print(" [4] Process RTSP / Network Stream (e.g., rtsp://...)")
        print(" [5] Advanced Options & Configuration")
        print(" [6] View Recognized Records Ledger")
        print("-" * 72)
        print(" [Q] Exit")
        print("=" * 72)

        try:
            choice = input("Select an option [1-6, Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting ALPR. Goodbye!")
            break

        if choice == "1":
            print("\n[-] Opening file selection dialog...")
            selected = select_video_file_dialog()
            if selected:
                if os.path.exists(selected):
                    _execute_menu_run(selected, settings)
                else:
                    print(f"[ERROR] Selected file does not exist: '{selected}'")
                    pause_menu()
            else:
                print("[INFO] File selection was cancelled. Returning to menu.")

        elif choice == "2":
            print("\nEnter video file path (or 'C' to cancel):")
            try:
                raw_inp = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if raw_inp.upper() == "C" or not raw_inp:
                continue
            clean_path = normalize_input_path(raw_inp, base_dir)
            if clean_path and os.path.exists(clean_path):
                _execute_menu_run(clean_path, settings)
            else:
                print(f"[ERROR] File not found or invalid path: '{raw_inp}'")
                pause_menu()

        elif choice == "3":
            print("\nEnter Camera Index (0 for default built-in/USB webcam, or 'C' to cancel):")
            try:
                cam_inp = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if cam_inp.upper() == "C" or not cam_inp:
                continue
            if cam_inp.isdigit():
                _execute_menu_run(cam_inp, settings)
            else:
                print("[ERROR] Camera index must be a numeric integer (e.g., 0, 1).")
                pause_menu()

        elif choice == "4":
            print("\nEnter Stream URL (e.g., rtsp://user:pass@ip:port/stream, or 'C' to cancel):")
            try:
                stream_inp = input("> ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                continue
            if stream_inp.upper() == "C" or not stream_inp:
                continue
            if re.match(r"^(rtsp|rtmp|http|https|tcp)://", stream_inp, re.IGNORECASE):
                _execute_menu_run(stream_inp, settings)
            else:
                print("[ERROR] Invalid stream URL. Must begin with rtsp://, rtmp://, http://, https://, or tcp://")
                pause_menu()

        elif choice == "5":
            show_advanced_menu(settings)

        elif choice == "6":
            show_ledger_summary(base_dir)

        elif choice == "Q":
            print("\nExiting ALPR. Goodbye!\n")
            break

        else:
            print(f"[!] Unknown selection '{choice}'. Please enter 1-6 or Q.")


def _execute_menu_run(source: str, settings: Dict[str, Any]):
    """Execute pipeline with menu settings and return to menu cleanly afterwards."""
    args = argparse.Namespace(
        source=source,
        config=None,
        ledger=None,
        weights=None,
        ocr_engine=settings["ocr_engine"],
        fastplateocr_model=None,
        fastplateocr_config=None,
        ppocr_det_dir=None,
        ppocr_rec_dir=None,
        ocr_dir=None,
        frame_stride=settings["frame_stride"],
        det_conf=settings["det_conf"],
        ocr_conf=settings["ocr_conf"],
        dedup_seconds=settings["dedup_seconds"],
        device=settings["device"],
        preview=settings["preview"],
        imgsz=settings["imgsz"],
        max_frames=None,
    )

    start_time = time.time()
    print("\n" + "=" * 72)
    print(f"[ALPR LAUNCH] Starting processing for source: '{source}'")
    print(f" Preview: {'ENABLED' if settings['preview'] else 'DISABLED'} | Engine: {settings['ocr_engine']} | Device: {settings['device']}")
    print("=" * 72 + "\n", flush=True)

    exit_code = 0
    try:
        run_pipeline(args)
    except SystemExit as se:
        if se.code not in (0, None):
            print(f"\n[ALPR NOTICE] Pipeline exited: {se}", flush=True)
            exit_code = se.code if isinstance(se.code, int) else 1
    except KeyboardInterrupt:
        print("\n[ALPR] Processing halted by user.", flush=True)
    except Exception as e:
        print(f"\n[ALPR ERROR] Unexpected exception during processing: {e}", file=sys.stderr, flush=True)
        exit_code = -1

    elapsed = round(time.time() - start_time, 1)
    print("\n" + "-" * 72)
    if exit_code == 0:
        print(f"[SUCCESS] Processing session finished in {elapsed}s (Exit code: 0).", flush=True)
    else:
        print(f"[WARNING] Session concluded with code {exit_code} in {elapsed}s.", flush=True)
    print("-" * 72)

    pause_menu("\nPress [Enter] to return to the main menu...")


def main():
    if len(sys.argv) == 1:
        launch_interactive_menu()
    else:
        parser = build_arg_parser()
        args = parser.parse_args()
        check_python_runtime(getattr(args, "ocr_engine", None))
        run_pipeline(args)


if __name__ == "__main__":
    main()
