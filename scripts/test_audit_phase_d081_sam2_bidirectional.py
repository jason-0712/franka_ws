#!/usr/bin/env python3
"""CPU unit tests for Phase D-0.8.1 bidirectional SAM2 audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if not (SCRIPT_DIR / "audit_phase_d081_sam2_bidirectional.py").is_file():
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPT_DIR / "audit_phase_d081_sam2_bidirectional.py"
SPEC = importlib.util.spec_from_file_location("phase_d081", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def square(y0=20, x0=30, size=10):
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


class PhaseD081BidirectionalTest(unittest.TestCase):
    def test_nested_anchor_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "anchors.json"
            path.write_text(json.dumps({"82": {"approach": [1, 2, 10, 20], "release": [5, 6, 15, 16]}}))
            self.assertEqual(
                MODULE.load_anchor_overrides(path, 518)[82]["approach"],
                [1.0, 2.0, 10.0, 20.0],
            )

    def test_padding_is_clipped(self):
        self.assertEqual(MODULE.pad_box([2, 3, 516, 517], 4, 518), [0.0, 0.0, 517.0, 517.0])

    def test_frame_aware_visible_anchor_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "visible.json"
            path.write_text(
                json.dumps(
                    {
                        "10": {
                            "release": {
                                "dataset_frame_index": 205,
                                "box_xyxy": [120, 220, 150, 250],
                            }
                        }
                    }
                )
            )
            loaded = MODULE.load_visible_anchor_overrides(path, 518)
            self.assertEqual(loaded[10]["release"]["dataset_frame_index"], 205)
            self.assertEqual(
                loaded[10]["release"]["box_xyxy"], [120.0, 220.0, 150.0, 250.0]
            )

    def test_visible_anchor_direction_is_enforced(self):
        MODULE.validate_visible_anchor_direction("approach", 30, 40)
        MODULE.validate_visible_anchor_direction("release", 210, 200)
        with self.assertRaises(ValueError):
            MODULE.validate_visible_anchor_direction("approach", 41, 40)
        with self.assertRaises(ValueError):
            MODULE.validate_visible_anchor_direction("release", 199, 200)

    def test_agreeing_masks_select_phase_preferred_direction(self):
        forward = square()
        backward = square(x0=31)
        fused, metrics = MODULE.fuse_directional_masks(
            forward, backward, phase="pre_grasp", minimum_agreement_iou=0.15
        )
        self.assertTrue(np.array_equal(fused, forward))
        self.assertEqual(metrics["source"], "agreement_forward")

    def test_disagreement_abstains_away_from_anchor(self):
        fused, metrics = MODULE.fuse_directional_masks(
            square(x0=10), square(x0=70), phase="grasp", minimum_agreement_iou=0.15
        )
        self.assertIsNone(fused)
        self.assertEqual(metrics["source"], "abstain_direction_disagreement")

    def test_anchor_uses_its_direction_during_disagreement(self):
        forward = square(x0=10)
        backward = square(x0=70)
        fused, metrics = MODULE.fuse_directional_masks(
            forward, backward, phase="release", minimum_agreement_iou=0.15
        )
        self.assertTrue(np.array_equal(fused, backward))
        self.assertEqual(metrics["source"], "backward_anchor")

    def test_no_mask_abstains(self):
        fused, metrics = MODULE.fuse_directional_masks(
            None, np.zeros((100, 100), dtype=bool), phase="transport", minimum_agreement_iou=0.15
        )
        self.assertIsNone(fused)
        self.assertTrue(metrics["abstained"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
