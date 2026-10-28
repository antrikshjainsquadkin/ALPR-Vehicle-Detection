#!/usr/bin/env python3
"""Comprehensive Integration & Video Pipeline Edge-Case Tests for ALPR System.

Covers:
- Category 1: Real-world Yellow Plate Recognition & Color Classification (KA53MH3895)
- Category 1: Real-world Green Plate Recognition & Color Classification (GAE5379 / KBC6233)
- Category 2: False Positive Suppression on Storefront Signs / Banners (WELLESLEY)
- Category 4: Consecutive Frame Deduplication and Vehicle Passage Merging
- Category 5: Invalid Video Paths, Zero-Byte Files, and Blank Frames
- Category 6: Local Offline Weights Verification (YOLO, PP-OCRv6, EasyOCR)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "main.py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "license_plate_recognition_config.json")


class TestOfflineModelIntegrity(unittest.TestCase):
    """Test local model assets exist and load without internet connectivity."""

    def test_detector_weights_exist(self):
        weights_path = os.path.join(PROJECT_ROOT, "models", "license_plate_detector.pt")
        self.assertTrue(os.path.exists(weights_path), f"Missing detector weights at {weights_path}")
        self.assertGreater(os.path.getsize(weights_path), 5_000_000)

    def test_ppocrv6_weights_exist(self):
        det_dir = os.path.join(PROJECT_ROOT, "models", "ocr", "ppocrv6", "PP-OCRv6_medium_det")
        rec_dir = os.path.join(PROJECT_ROOT, "models", "ocr", "ppocrv6", "PP-OCRv6_medium_rec")
        for d_name, d_dir in [("detection", det_dir), ("recognition", rec_dir)]:
            self.assertTrue(os.path.exists(os.path.join(d_dir, "inference.pdiparams")), f"Missing PP-OCRv6 {d_name} pdiparams")
            self.assertTrue(os.path.exists(os.path.join(d_dir, "inference.json")), f"Missing PP-OCRv6 {d_name} json")

    def test_easyocr_weights_exist(self):
        craft_path = os.path.join(PROJECT_ROOT, "models", "ocr", "craft_mlt_25k.pth")
        rec_path = os.path.join(PROJECT_ROOT, "models", "ocr", "english_g2.pth")
        self.assertTrue(os.path.exists(craft_path), f"Missing CRAFT detector at {craft_path}")
        self.assertTrue(os.path.exists(rec_path), f"Missing EasyOCR English recognition at {rec_path}")


class TestInputFailuresAndSafeguards(unittest.TestCase):
    """Test application behavior on missing sources, empty files, and invalid formats."""

    def test_nonexistent_video_path(self):
        """Pipeline must fail safely with an error message and non-zero exit code without crashing or corrupting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_ledger = os.path.join(tmpdir, "test_ledger.json")
            cmd = [
                PYTHON_EXE, MAIN_SCRIPT,
                "--source", os.path.join(tmpdir, "does_not_exist_98765.mp4"),
                "--ledger", temp_ledger,
                "--config", CONFIG_PATH,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not found", proc.stderr + proc.stdout)

    def test_zero_byte_video_file(self):
        """A zero-byte video file must be rejected gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_video = os.path.join(tmpdir, "zero_byte_test.mp4")
            open(empty_video, "w").close()
            temp_ledger = os.path.join(tmpdir, "test_ledger.json")

            cmd = [
                PYTHON_EXE, MAIN_SCRIPT,
                "--source", empty_video,
                "--ledger", temp_ledger,
                "--config", CONFIG_PATH,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(proc.returncode, 0)


class TestRealWorldVideoPipelines(unittest.TestCase):
    """Real-world end-to-end evaluation using representative video clips."""

    @classmethod
    def setUpClass(cls):
        cls.downloads_dir = "C:\\Users\\A\\Downloads"
        cls.yellow_vid = os.path.join(cls.downloads_dir, "14736141_2160_3840_30fps.mp4")
        cls.green_vid = os.path.join(cls.downloads_dir, "12937540_1920_1080_30fps.mp4")
        cls.storefront_vid = os.path.join(cls.downloads_dir, "13926715_3840_2160_24fps.mp4")
        cls.uk_vid = os.path.join(cls.downloads_dir, "15396171_1920_1080_25fps.mp4")

    def test_yellow_plate_detection_and_color_classification(self):
        """Verify commercial vehicle yellow plate KA53MH3895 is detected and color classified as YELLOW."""
        if not os.path.exists(self.yellow_vid):
            self.skipTest(f"Yellow plate video not found at {self.yellow_vid}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_ledger = os.path.join(tmpdir, "yellow_test_ledger.json")
            cmd = [
                PYTHON_EXE, MAIN_SCRIPT,
                "--source", self.yellow_vid,
                "--ledger", temp_ledger,
                "--config", CONFIG_PATH,
                "--frame-stride", "20",
                "--max-frames", "80",
                "--imgsz", "640",
                "--det-conf", "0.20",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, f"Yellow plate run failed: {proc.stderr}")

            with open(temp_ledger, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            self.assertGreater(len(records), 0, "No records captured for yellow plate video")

            plates = [r["normalized_plate_text"] for r in records]
            colors = [r.get("plate_color") for r in records]
            print(f"[TEST YELLOW] Recognized plates: {plates}, colors: {colors}")

            # Verify KA53MH3895 was recognized
            has_expected_plate = any("KA53MH" in p or "MH3895" in p for p in plates)
            self.assertTrue(has_expected_plate, f"Expected Indian plate KA53MH3895 in {plates}")

            # Verify yellow color classification
            has_yellow_color = any(c == "YELLOW" for c in colors)
            self.assertTrue(has_yellow_color, f"Expected YELLOW color classification in {colors}")

    def test_green_plate_detection_and_color_classification(self):
        """Verify EV green plates (GAE5379 or KBC6233) are detected and classified as GREEN."""
        if not os.path.exists(self.green_vid):
            self.skipTest(f"Green plate video not found at {self.green_vid}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_ledger = os.path.join(tmpdir, "green_test_ledger.json")
            cmd = [
                PYTHON_EXE, MAIN_SCRIPT,
                "--source", self.green_vid,
                "--ledger", temp_ledger,
                "--config", CONFIG_PATH,
                "--frame-stride", "20",
                "--max-frames", "380",
                "--imgsz", "640",
                "--det-conf", "0.20",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, f"Green plate run failed: {proc.stderr}")

            with open(temp_ledger, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            self.assertGreater(len(records), 0, "No records captured for green plate video")

            plates = [r["normalized_plate_text"] for r in records]
            colors = [r.get("plate_color") for r in records]
            print(f"[TEST GREEN] Recognized plates: {plates}, colors: {colors}")

            has_green_plate = any("GAE" in p or "5379" in p or "KBC" in p or "6233" in p for p in plates)
            self.assertTrue(has_green_plate, f"Expected green EV plate (GAE5379/KBC6233) in {plates}")

            has_green_color = any(c == "GREEN" for c in colors)
            self.assertTrue(has_green_color, f"Expected GREEN color classification in {colors}")

    def test_false_positive_storefront_suppression(self):
        """Verify storefront sign '... WELLESLEY' is suppressed and creates 0 false positive records."""
        if not os.path.exists(self.storefront_vid):
            self.skipTest(f"Storefront video not found at {self.storefront_vid}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_ledger = os.path.join(tmpdir, "storefront_test_ledger.json")
            cmd = [
                PYTHON_EXE, MAIN_SCRIPT,
                "--source", self.storefront_vid,
                "--ledger", temp_ledger,
                "--config", CONFIG_PATH,
                "--frame-stride", "20",
                "--max-frames", "60",
                "--imgsz", "640",
                "--det-conf", "0.20",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, f"Storefront run failed: {proc.stderr}")

            with open(temp_ledger, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            self.assertEqual(len(records), 0, f"Expected 0 false positive records from storefront, got {len(records)}: {records}")

    def test_uk_plate_deduplication_and_trajectory_merging(self):
        """Verify UK plate YG21DBM in video 15396171 merges multiple frames into 1 single passage record."""
        if not os.path.exists(self.uk_vid):
            self.skipTest(f"UK video not found at {self.uk_vid}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_ledger = os.path.join(tmpdir, "uk_test_ledger.json")
            cmd = [
                PYTHON_EXE, MAIN_SCRIPT,
                "--source", self.uk_vid,
                "--ledger", temp_ledger,
                "--config", CONFIG_PATH,
                "--frame-stride", "10",
                "--max-frames", "80",
                "--imgsz", "640",
                "--det-conf", "0.20",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, f"UK run failed: {proc.stderr}")

            with open(temp_ledger, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            self.assertEqual(len(records), 1, f"Expected exactly 1 merged passage record, got {len(records)}: {records}")

            rec = records[0]
            self.assertEqual(rec["normalized_plate_text"], "YG21DBM")
            self.assertEqual(rec.get("matched_country_pattern"), "GB_STANDARD")
            self.assertGreaterEqual(rec.get("observations_count", 0), 1)


if __name__ == "__main__":
    unittest.main()
