#!/usr/bin/env python3
"""CPU tests for Phase D-0.8.2 oracle-anchor export helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if not (SCRIPT_DIR / "export_phase_d082_oracle_anchor_review.py").is_file():
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPT_DIR / "export_phase_d082_oracle_anchor_review.py"
SPEC = importlib.util.spec_from_file_location("phase_d082", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PhaseD082OracleExportTest(unittest.TestCase):
    def test_default_anchor_specs(self):
        self.assertEqual(
            MODULE.parse_anchor_specs(MODULE.DEFAULT_ANCHORS),
            [(0, "release"), (10, "release"), (31, "release"), (41, "release"), (82, "approach")],
        )

    def test_duplicate_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.parse_anchor_specs("0:release,0:release")

    def test_non_anchor_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.parse_anchor_specs("0:grasp")

    def test_native_box_scales_to_square_model_input(self):
        scaled = MODULE.scale_box(
            [320, 240, 640, 480],
            source_width=640,
            source_height=480,
            target_size=518,
        )
        self.assertEqual(scaled, [259.0, 259.0, 518.0, 518.0])

    def test_grid_keeps_image_dimensions(self):
        image = Image.new("RGB", (518, 518))
        self.assertEqual(MODULE.draw_coordinate_grid(image).size, (518, 518))


if __name__ == "__main__":
    unittest.main(verbosity=2)
