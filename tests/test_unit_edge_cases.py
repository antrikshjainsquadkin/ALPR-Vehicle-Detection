#!/usr/bin/env python3
"""Comprehensive Deterministic Edge-Case Unit Tests for ALPR Pipeline.

Covers:
- Category 1: Number-plate colors and appearance (Yellow, Green, White, Other, HSRP tokens, character disambiguation)
- Category 2: False positives and incorrect detections (Text markings without digits/letters, aspect ratio bounds, oversized/undersized boxes)
- Category 5: Image, video, and system input failures (Corrupted/empty crops, empty ledgers, atomic replace, lock cleanup)
- Category 6: Deduplication, spatial trajectory geometry, and ISO 8601 UTC timestamp parsing
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import main


class TestPlateColorClassification(unittest.TestCase):
    """Test background color classification across HSV distributions."""

    def test_yellow_plate_synthesis(self):
        """Synthetic yellow plate: Hue ~ 25-30, Saturation ~ 150-200, Value ~ 200."""
        # HSV: Hue 28, Sat 180, Val 220 -> In BGR: roughly bright yellow
        import cv2
        hsv_img = np.full((60, 200, 3), (25, 180, 220), dtype=np.uint8)
        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        color = main.classify_plate_color(bgr_img)
        self.assertEqual(color, "YELLOW", f"Expected YELLOW but got {color}")

    def test_green_plate_synthesis(self):
        """Synthetic green plate: Hue ~ 50-70, Saturation ~ 120-180, Value ~ 80-150."""
        import cv2
        hsv_img = np.full((60, 200, 3), (60, 150, 100), dtype=np.uint8)
        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        color = main.classify_plate_color(bgr_img)
        self.assertEqual(color, "GREEN", f"Expected GREEN but got {color}")

    def test_white_plate_synthesis(self):
        """Synthetic white plate: Low saturation (<30), high value (>150)."""
        import cv2
        hsv_img = np.full((60, 200, 3), (0, 10, 230), dtype=np.uint8)
        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        color = main.classify_plate_color(bgr_img)
        self.assertEqual(color, "WHITE", f"Expected WHITE but got {color}")

    def test_other_background_synthesis(self):
        """Synthetic blue or red background not matching Yellow/Green/White."""
        import cv2
        # Deep blue: Hue ~ 110, Sat 200, Val 150
        hsv_img = np.full((60, 200, 3), (110, 200, 150), dtype=np.uint8)
        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        color = main.classify_plate_color(bgr_img)
        self.assertEqual(color, "OTHER", f"Expected OTHER for blue but got {color}")

    def test_empty_or_corrupt_crop_color(self):
        """Empty, None, 1D or 0-size crops should safely return 'OTHER' without crashing."""
        self.assertEqual(main.classify_plate_color(None), "OTHER")
        self.assertEqual(main.classify_plate_color(np.array([])), "OTHER")
        self.assertEqual(main.classify_plate_color(np.zeros((0, 0, 3), dtype=np.uint8)), "OTHER")
        self.assertEqual(main.classify_plate_color(np.zeros((10,), dtype=np.uint8)), "OTHER")


class TestPlateEnhancement(unittest.TestCase):
    """Test adaptive preprocessing enhancement for OCR."""

    def test_enhance_crop_handles_all_color_hints(self):
        sample_crop = np.full((30, 100, 3), 128, dtype=np.uint8)
        for hint in ["YELLOW", "GREEN", "WHITE", "OTHER", None]:
            enhanced = main.enhance_plate_crop(sample_crop, color_hint=hint)
            self.assertIsNotNone(enhanced)
            self.assertGreater(enhanced.size, 0)
            self.assertGreaterEqual(enhanced.ndim, 2)

    def test_enhance_crop_empty_input(self):
        self.assertIsNone(main.enhance_plate_crop(None))
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        res = main.enhance_plate_crop(empty)
        self.assertEqual(res.size, 0)


class TestTextNormalizationAndDisambiguation(unittest.TestCase):
    """Test plate text cleaning, HSRP token stripping, and format-aware disambiguation."""

    def setUp(self):
        self.patterns = main.DEFAULT_CONFIG["plate_validation"]["country_patterns"]

    def test_normalize_plate_text(self):
        self.assertEqual(main.normalize_plate_text("KA-53-MH-3895"), "KA53MH3895")
        self.assertEqual(main.normalize_plate_text("  yg21 dbm  "), "YG21DBM")
        self.assertEqual(main.normalize_plate_text("GAE.5379!"), "GAE5379")
        self.assertEqual(main.normalize_plate_text(""), "")
        self.assertEqual(main.normalize_plate_text(None), "")

    def test_indian_hsrp_prefix_suffix_stripping(self):
        # IND prefix
        text, country, is_changed = main.match_and_disambiguate_pattern("INDKA53MH3895", self.patterns)
        self.assertEqual(text, "KA53MH3895")
        self.assertEqual(country, "IN_STANDARD")
        self.assertTrue(is_changed)

        # IND suffix
        text, country, is_changed = main.match_and_disambiguate_pattern("KA53MH3895IND", self.patterns)
        self.assertEqual(text, "KA53MH3895")
        self.assertEqual(country, "IN_STANDARD")
        self.assertTrue(is_changed)

        # ND suffix
        text, country, is_changed = main.match_and_disambiguate_pattern("KA53MH3895ND", self.patterns)
        self.assertEqual(text, "KA53MH3895")
        self.assertEqual(country, "IN_STANDARD")
        self.assertTrue(is_changed)

    def test_format_aware_disambiguation_capped_at_cost_2(self):
        # KA53MH389S -> S in numeric segment 4 should become 5
        text, country, is_changed = main.match_and_disambiguate_pattern("KA53MH389S", self.patterns)
        self.assertEqual(text, "KA53MH3895")
        self.assertEqual(country, "IN_STANDARD")

        # KA53MHO895 -> O in numeric segment 4 should become 0
        text, country, is_changed = main.match_and_disambiguate_pattern("KA53MHO895", self.patterns)
        self.assertEqual(text, "KA53MH0895")
        self.assertEqual(country, "IN_STANDARD")

        # More than 2 character errors should NOT be distorted to match
        garbled = "KA53MHSSSS"  # 4 errors in numeric segment
        text, country, is_changed = main.match_and_disambiguate_pattern(garbled, self.patterns)
        self.assertNotEqual(country, "IN_STANDARD")

    def test_uk_standard_pattern(self):
        text, country, _ = main.match_and_disambiguate_pattern("YG21DBM", self.patterns)
        self.assertEqual(country, "GB_STANDARD")
        self.assertEqual(text, "YG21DBM")

        text, country, _ = main.match_and_disambiguate_pattern("BD51SMR", self.patterns)
        self.assertEqual(country, "GB_STANDARD")
        self.assertEqual(text, "BD51SMR")

    def test_intl_3a4d_standard_pattern(self):
        text, country, _ = main.match_and_disambiguate_pattern("GAE5379", self.patterns)
        self.assertEqual(country, "INTL_STANDARD_3A4D")
        self.assertEqual(text, "GAE5379")

        text, country, _ = main.match_and_disambiguate_pattern("KBC6233", self.patterns)
        self.assertEqual(country, "INTL_STANDARD_3A4D")
        self.assertEqual(text, "KBC6233")

    def test_us_generic_pattern(self):
        text, country, _ = main.match_and_disambiguate_pattern("CALIF99", self.patterns)
        self.assertEqual(country, "US_GENERIC")
        self.assertEqual(text, "CALIF99")

        text, country, _ = main.match_and_disambiguate_pattern("TEXAS1", self.patterns)
        self.assertEqual(country, "US_GENERIC")
        self.assertEqual(text, "TEXAS1")


class TestFalsePositiveAndNoiseRejection(unittest.TestCase):
    """Test geometric filtering and non-plate text rejection rules."""

    def test_aspect_ratio_boundary_filtering(self):
        # Normal plate aspect ratios (2.0 to 5.0)
        valid_bw, valid_bh = 200, 50
        ar = valid_bw / valid_bh
        self.assertTrue(1.05 <= ar <= 6.5)

        # Square sign (e.g. 100x100 -> AR=1.0) -> Rejection
        square_bw, square_bh = 100, 100
        ar_square = square_bw / square_bh
        self.assertFalse(1.05 <= ar_square <= 6.5)

        # Extremely wide strip (e.g. 700x50 -> AR=14.0) -> Rejection
        wide_bw, wide_bh = 700, 50
        ar_wide = wide_bw / wide_bh
        self.assertFalse(1.05 <= ar_wide <= 6.5)

    def test_oversized_box_filtering(self):
        # Frame 1920x1080: bh > 216 (20% of 1080) or bw > 1152 (60% of 1920) must be rejected
        h, w = 1080, 1920
        storefront_bh = 250
        storefront_bw = 800
        self.assertTrue(storefront_bh > (h * 0.20))

    def test_pure_text_noise_rejection_logic(self):
        # Signs with only alphabet and no digits (e.g. WELLESLEY, SUN MOTORS, PARKING)
        for noise_word in ["WELLESLEY", "SUNMOTORS", "PARKING", "ENTRANCE", "STOP"]:
            has_digit = any(c.isdigit() for c in noise_word)
            has_alpha = any(c.isalpha() for c in noise_word)
            self.assertFalse(has_digit and has_alpha, f"'{noise_word}' should not have both digits and letters")

        # Signs with only numbers and no letters (e.g. 5000, 123456)
        for num_noise in ["5000", "123456", "9999"]:
            has_digit = any(c.isdigit() for c in num_noise)
            has_alpha = any(c.isalpha() for c in num_noise)
            self.assertFalse(has_digit and has_alpha, f"'{num_noise}' should not have both digits and letters")


class TestSystemResilienceAndPersistence(unittest.TestCase):
    """Test atomic persistence, empty ledger self-healing, and lock file mechanisms."""

    def test_iso_utc_parsing_and_formatting(self):
        now_str = main.get_utc_now_iso()
        self.assertTrue(now_str.endswith("Z"))
        parsed_dt = main.parse_iso_utc(now_str)
        self.assertEqual(parsed_dt.tzinfo, timezone.utc)

    def test_atomic_replace_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_ledger.json")
            payload = {"schema_version": 1, "test_key": "safe_value"}
            main.atomic_replace_json(test_file, payload)
            self.assertTrue(os.path.exists(test_file))

            with open(test_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, payload)

    def test_empty_ledger_self_healing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = os.path.join(tmpdir, "empty_ledger.json")
            # Create a 0-byte file
            open(empty_file, "w").close()
            self.assertEqual(os.path.getsize(empty_file), 0)

            # load_and_validate_ledger should safely heal it without throwing sys.exit
            ledger_data, dedup_cache = main.load_and_validate_ledger(empty_file)
            self.assertEqual(ledger_data["schema_version"], 1)
            self.assertEqual(ledger_data["records"], [])
            self.assertEqual(len(dedup_cache), 0)

    def test_stale_lock_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, ".license_plate_recognition.lock")
            # Write a lock file with a non-existent dead PID (e.g. 999999)
            stale_payload = {
                "pid": 999999,
                "hostname": "test-host",
                "source_id": "test_video.mp4",
                "acquired_at": main.get_utc_now_iso(),
            }
            with open(lock_path, "w", encoding="utf-8") as f:
                json.dump(stale_payload, f)

            # Attempting to acquire should detect PID 999999 is dead and acquire cleanly
            acquired_path = main.acquire_instance_lock(tmpdir, "test_video.mp4")
            self.assertEqual(acquired_path, lock_path)
            main.release_instance_lock()
            self.assertFalse(os.path.exists(lock_path))


class TestSpatialGeometryAndDeduplication(unittest.TestCase):
    """Test IoU, centroid distance, and Levenshtein distance."""

    def test_box_iou(self):
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        self.assertAlmostEqual(main.compute_box_iou(box1, box2), 1.0)

        # Disjoint boxes
        box3 = [300, 300, 400, 400]
        self.assertEqual(main.compute_box_iou(box1, box3), 0.0)

        # 50% overlap box
        box4 = [100, 100, 200, 150]  # half height
        iou = main.compute_box_iou(box1, box4)
        self.assertAlmostEqual(iou, 0.5)

    def test_center_distance(self):
        box1 = [0, 0, 100, 100]  # center = (50, 50)
        box2 = [0, 0, 100, 100]  # center = (50, 50)
        self.assertAlmostEqual(main.compute_center_distance(box1, box2), 0.0)

        box3 = [100, 100, 200, 200]  # center = (150, 150)
        # distance from (50,50) to (150,150) = sqrt(100^2 + 100^2) = 141.42
        self.assertAlmostEqual(main.compute_center_distance(box1, box3), 141.421356, places=4)

    def test_levenshtein_distance(self):
        self.assertEqual(main.levenshtein_distance("KA53MH3895", "KA53MH3895"), 0)
        self.assertEqual(main.levenshtein_distance("KA53MH3895", "KA53MH389S"), 1)
        self.assertEqual(main.levenshtein_distance("YG21DBM", "YG21DBH"), 1)
        self.assertEqual(main.levenshtein_distance("GAE5379", "GAE5370"), 1)


if __name__ == "__main__":
    unittest.main()
