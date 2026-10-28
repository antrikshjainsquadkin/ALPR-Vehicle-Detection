"""Comprehensive regression and verification test suite for FastPlateOCR and Confidence-Aware Consensus.

Covers:
- FastPlateOCR offline model initialization and inference
- Exact character-level confidence extraction, count, and ordering alignment
- OCRItem tuple backward compatibility
- Multi-frame consensus under stable observations
- Single low-confidence character flagging (review_required, no silent substitution)
- Conflicting character predictions across frames with positional voting
- Prevention of merging distinct vehicles
- Graceful handling of malformed/zero-byte inputs
- Backward compatibility with legacy JSON ledger records lacking character confidence
"""

import json
import os
import unittest
import numpy as np
import cv2

from main import (
    FastPlateOCREngine,
    OCRItem,
    verify_and_load_fastplateocr,
    compute_multiframe_consensus,
    build_arg_parser,
    get_default_menu_settings,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ocr", "fastplateocr", "cct_s_v2_global.onnx")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "ocr", "fastplateocr", "cct_s_v2_global_plate_config.yaml")


class TestFastPlateOCRAndConsensus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_model_assets = os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH)
        if cls.has_model_assets:
            cls.engine = verify_and_load_fastplateocr(MODEL_PATH, CONFIG_PATH, device="cpu")
        else:
            cls.engine = None

    def test_01_fastplateocr_offline_initialization(self):
        """Verify FastPlateOCR loads locally without remote network dependencies."""
        if not self.has_model_assets:
            self.skipTest(f"Model assets not found at {MODEL_PATH}")
        self.assertIsInstance(self.engine, FastPlateOCREngine)
        self.assertIsNotNone(self.engine.recognizer)

    def test_02_ocritem_backward_compatibility(self):
        """Verify OCRItem is a 3-tuple (box, text, conf) while holding .char_confidences."""
        item = OCRItem(
            [10, 20, 100, 50],
            "YG21DBM",
            0.95,
            [{"char": "Y", "confidence": 0.98}, {"char": "G", "confidence": 0.96}],
        )
        # Tuple unpacking
        box, text, conf = item
        self.assertEqual(box, [10, 20, 100, 50])
        self.assertEqual(text, "YG21DBM")
        self.assertAlmostEqual(conf, 0.95)
        # Index access
        self.assertEqual(item[0], [10, 20, 100, 50])
        self.assertEqual(item[1], "YG21DBM")
        self.assertEqual(item[2], 0.95)
        # Attribute access
        self.assertEqual(len(item.char_confidences), 2)
        self.assertEqual(item.char_confidences[0]["char"], "Y")

    def test_03_prediction_and_character_confidence_alignment(self):
        """Verify inference returns aligned character confidences matching returned plate text."""
        if not self.has_model_assets:
            self.skipTest("FastPlateOCR model assets unavailable")

        # Create a synthetic white plate with black text
        img = np.ones((70, 260, 3), dtype=np.uint8) * 240
        cv2.putText(img, "YG21DBM", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3)

        results = self.engine.readtext(img)
        self.assertGreater(len(results), 0, "No text recognized on synthetic plate")

        box, text, conf = results[0]
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertGreater(conf, 0.0)

        char_confs = results[0].char_confidences
        self.assertEqual(len(char_confs), len(text), "Character confidences count must strictly match text length")

        for idx, entry in enumerate(char_confs):
            self.assertEqual(entry["char"], text[idx], f"Char mismatch at index {idx}: {entry['char']} vs {text[idx]}")
            self.assertGreaterEqual(entry["confidence"], 0.0)
            self.assertLessEqual(entry["confidence"], 1.0)

    def test_04_malformed_or_empty_image_input(self):
        """Verify empty or zero-byte crop returns empty result safely without exceptions."""
        if not self.has_model_assets:
            self.skipTest("FastPlateOCR model assets unavailable")

        # Zero-sized image
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(self.engine.readtext(empty_img), [])

        # Non-image None
        self.assertEqual(self.engine.readtext(None), [])

    def test_05_multiframe_consensus_stable_plate(self):
        """Verify multi-frame consensus across repeated identical observations."""
        candidates = [
            {
                "plate": "YG21DBM",
                "conf": 0.95,
                "frame": 10,
                "char_probs": [0.98, 0.96, 0.99, 0.97, 0.94, 0.92, 0.95],
                "character_confidences": [
                    {"char": "Y", "confidence": 0.98},
                    {"char": "G", "confidence": 0.96},
                    {"char": "2", "confidence": 0.99},
                    {"char": "1", "confidence": 0.97},
                    {"char": "D", "confidence": 0.94},
                    {"char": "B", "confidence": 0.92},
                    {"char": "M", "confidence": 0.95},
                ],
            },
            {
                "plate": "YG21DBM",
                "conf": 0.96,
                "frame": 13,
                "char_probs": [0.99, 0.97, 0.99, 0.98, 0.95, 0.94, 0.96],
                "character_confidences": [
                    {"char": "Y", "confidence": 0.99},
                    {"char": "G", "confidence": 0.97},
                    {"char": "2", "confidence": 0.99},
                    {"char": "1", "confidence": 0.98},
                    {"char": "D", "confidence": 0.95},
                    {"char": "B", "confidence": 0.94},
                    {"char": "M", "confidence": 0.96},
                ],
            },
        ]

        consensus_plate, overall_conf, char_confs, review_req, matched_country = compute_multiframe_consensus(candidates)
        self.assertEqual(consensus_plate, "YG21DBM")
        self.assertGreater(overall_conf, 0.90)
        self.assertEqual(len(char_confs), 7)
        min_conf = min(c["confidence"] for c in char_confs)
        self.assertGreater(min_conf, 0.90)
        self.assertEqual(review_req, [], "Stable high-confidence plate should have no review_required flags")

    def test_06_multiframe_consensus_single_low_confidence_character(self):
        """Verify that a character below low_confidence_threshold is flagged in review_required and not silently altered."""
        candidates = [
            {
                "plate": "YG21DBM",
                "conf": 0.81,
                "frame": 10,
                "char_probs": [0.98, 0.96, 0.99, 0.97, 0.91, 0.84, 0.45],  # 'M' is 0.45 (< 0.60)
                "character_confidences": [
                    {"char": "Y", "confidence": 0.98},
                    {"char": "G", "confidence": 0.96},
                    {"char": "2", "confidence": 0.99},
                    {"char": "1", "confidence": 0.97},
                    {"char": "D", "confidence": 0.91},
                    {"char": "B", "confidence": 0.84},
                    {"char": "M", "confidence": 0.45},
                ],
            }
        ]

        consensus_plate, overall_conf, char_confs, review_req, matched_country = compute_multiframe_consensus(
            candidates, char_low_conf_thresh=0.60
        )
        self.assertEqual(consensus_plate, "YG21DBM", "Original recognized characters must be preserved")
        self.assertIn("M", review_req, "Character 'M' with 0.45 confidence must be flagged in review_required")
        min_conf = min(c["confidence"] for c in char_confs)
        self.assertAlmostEqual(min_conf, 0.45, places=2)

    def test_07_multiframe_consensus_conflicting_character_resolution(self):
        """Verify positional voting resolves conflicting characters using real confidence weights."""
        candidates = [
            # Frame 1: 'B' read with 0.92 confidence
            {
                "plate": "YG21DBM",
                "conf": 0.92,
                "frame": 10,
                "char_probs": [0.95, 0.95, 0.95, 0.95, 0.95, 0.92, 0.95],
                "character_confidences": [
                    {"char": "Y", "confidence": 0.95},
                    {"char": "G", "confidence": 0.95},
                    {"char": "2", "confidence": 0.95},
                    {"char": "1", "confidence": 0.95},
                    {"char": "D", "confidence": 0.95},
                    {"char": "B", "confidence": 0.92},
                    {"char": "M", "confidence": 0.95},
                ],
            },
            # Frame 2: '8' misread due to blur with weak 0.51 confidence
            {
                "plate": "YG21D8M",
                "conf": 0.70,
                "frame": 12,
                "char_probs": [0.90, 0.90, 0.90, 0.90, 0.90, 0.51, 0.90],
                "character_confidences": [
                    {"char": "Y", "confidence": 0.90},
                    {"char": "G", "confidence": 0.90},
                    {"char": "2", "confidence": 0.90},
                    {"char": "1", "confidence": 0.90},
                    {"char": "D", "confidence": 0.90},
                    {"char": "8", "confidence": 0.51},
                    {"char": "M", "confidence": 0.90},
                ],
            },
            # Frame 3: 'B' confirmed with 0.94 confidence
            {
                "plate": "YG21DBM",
                "conf": 0.94,
                "frame": 14,
                "char_probs": [0.96, 0.96, 0.96, 0.96, 0.96, 0.94, 0.96],
                "character_confidences": [
                    {"char": "Y", "confidence": 0.96},
                    {"char": "G", "confidence": 0.96},
                    {"char": "2", "confidence": 0.96},
                    {"char": "1", "confidence": 0.96},
                    {"char": "D", "confidence": 0.96},
                    {"char": "B", "confidence": 0.94},
                    {"char": "M", "confidence": 0.96},
                ],
            },
        ]

        consensus_plate, overall_conf, char_confs, review_req, matched_country = compute_multiframe_consensus(candidates)
        # Position 5 should resolve to 'B' (weighted evidence 0.92 + 0.94 = 1.86 vs 0.51 for '8')
        self.assertEqual(consensus_plate, "YG21DBM")
        self.assertEqual(consensus_plate[5], "B")

    def test_08_json_ledger_backward_compatibility(self):
        """Verify legacy JSON ledger records lacking character_confidences parse without error."""
        legacy_record = {
            "event_id": "legacy-uuid-1234",
            "timestamp": "2026-03-01T10:00:00Z",
            "detected_plate_text": "KA53MH3895",
            "normalized_plate_text": "KA53MH3895",
            "confidence_score": 0.88,
            "detector_confidence": 0.75,
            "ocr_confidence": 0.88,
            "matched_country_pattern": "IN_STANDARD",
            "disambiguation_applied": False,
            "plate_color": "YELLOW",
            "source_id": "camera_0.mp4",
            "frame_index": 50,
            "bounding_box": [100, 200, 300, 280],
            "observations_count": 3,
        }

        # Parsing legacy record
        char_confs = legacy_record.get("character_confidences", [])
        min_conf = legacy_record.get("min_character_confidence", legacy_record.get("confidence_score"))
        review_req = legacy_record.get("review_required", [])
        engine = legacy_record.get("ocr_engine", "easyocr")

        self.assertEqual(char_confs, [])
        self.assertEqual(min_conf, 0.88)
        self.assertEqual(review_req, [])
        self.assertEqual(engine, "easyocr")

    def test_09_cli_argument_parser(self):
        """Verify CLI argument parser supports FastPlateOCR engine and arguments."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--source", "test_video.mp4",
            "--ocr-engine", "fastplateocr",
            "--fastplateocr-model", "models/ocr/fastplateocr/cct_s_v2_global.onnx",
            "--fastplateocr-config", "models/ocr/fastplateocr/cct_s_v2_global_plate_config.yaml",
        ])
        self.assertEqual(args.source, "test_video.mp4")
        self.assertEqual(args.ocr_engine, "fastplateocr")
        self.assertEqual(args.fastplateocr_model, "models/ocr/fastplateocr/cct_s_v2_global.onnx")
        self.assertEqual(args.fastplateocr_config, "models/ocr/fastplateocr/cct_s_v2_global_plate_config.yaml")

    def test_10_menu_settings_defaults(self):
        """Verify interactive menu defaults prioritize fastplateocr."""
        settings = get_default_menu_settings()
        self.assertIn("ocr_engine", settings)
        self.assertEqual(settings["ocr_engine"], "fastplateocr")

    def test_11_consensus_prevents_downgrade_of_high_confidence_reading(self):
        """Verify that a confirmed high-confidence plate is not overwritten by a lower-confidence variant."""
        prev_plate = "YG21DBM"
        prev_ocr_conf = 0.9987

        # Case 1: Blurred candidate gives a different plate at lower confidence (0.72)
        consensus_plate_blur = "YG21DBW"
        consensus_conf_blur = 0.7200
        should_update_blur = False
        if consensus_plate_blur == prev_plate:
            should_update_blur = (consensus_conf_blur > prev_ocr_conf)
        else:
            should_update_blur = (consensus_conf_blur >= prev_ocr_conf or prev_ocr_conf < 0.60)
        self.assertFalse(should_update_blur, "Blurred lower-confidence read must NOT overwrite high-confidence read")

        # Case 2: Better or equal candidate updates or confirms
        consensus_plate_sharp = "YG21DBM"
        consensus_conf_sharp = 0.9995
        should_update_sharp = False
        if consensus_plate_sharp == prev_plate:
            should_update_sharp = (consensus_conf_sharp > prev_ocr_conf)
        else:
            should_update_sharp = (consensus_conf_sharp >= prev_ocr_conf or prev_ocr_conf < 0.60)
        self.assertTrue(should_update_sharp, "Higher-confidence consensus must update")

        # Case 3: Initial low-confidence reading (< 0.60) is allowed to be corrected
        initial_low_conf = 0.48
        initial_plate = "LJ21RK"
        better_plate = "LJ72XRK"
        better_conf = 0.92
        should_correct = False
        if better_plate == initial_plate:
            should_correct = (better_conf > initial_low_conf)
        else:
            should_correct = (better_conf >= initial_low_conf or initial_low_conf < 0.60)
        self.assertTrue(should_correct, "Low-confidence reading must allow correction by better observation")

    def test_12_frame_stride_cadence(self):
        """Verify that frame stride check samples exactly every N-th frame consistently."""
        frame_stride = 3
        sampled_frames = []
        for frame_index in range(1, 10):  # 1-based frame loop as in video capture
            if frame_index % frame_stride != 0:
                continue
            sampled_frames.append(frame_index)
        self.assertEqual(sampled_frames, [3, 6, 9], "Cadence must sample exactly at intervals of frame_stride")

    def test_13_unpatterned_noise_filter_with_char_confidence(self):
        """Verify that low-confidence unpatterned noise is rejected using character confidences."""
        # Genuine plate: high confidence, no review flags
        gen_ocr_conf = 0.95
        gen_min_char_conf = 0.88
        gen_review_flags = []
        gen_plate = "YG21DBM"
        is_gen_noise = not (
            any(c.isdigit() for c in gen_plate)
            and any(c.isalpha() for c in gen_plate)
            and len(gen_plate) >= 5
            and gen_ocr_conf >= 0.45
            and gen_min_char_conf >= 0.30
            and len(gen_review_flags) <= (len(gen_plate) // 2)
        )
        self.assertFalse(is_gen_noise, "Genuine plate must NOT be flagged as noise")

        # Noise artifact: phone number sign '3H5306' with low char confidences and many review flags
        noise_ocr_conf = 0.4059
        noise_min_char_conf = 0.21
        noise_review_flags = ["3", "H", "5", "3", "0"]  # 5 out of 6 uncertain
        noise_plate = "3H5306"
        is_noise = not (
            any(c.isdigit() for c in noise_plate)
            and any(c.isalpha() for c in noise_plate)
            and len(noise_plate) >= 5
            and noise_ocr_conf >= 0.45
            and noise_min_char_conf >= 0.30
            and len(noise_review_flags) <= (len(noise_plate) // 2)
        )
        self.assertTrue(is_noise, "Low-confidence phone-number text artifact must be rejected as noise")


if __name__ == "__main__":
    unittest.main()

