#!/usr/bin/env python3
"""CPU-only unit tests for the Phase D-0.8 SAM2 prompt pilot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if not (SCRIPT_DIR / "audit_phase_d08_sam2_prompt_pilot.py").is_file():
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPT_DIR / "audit_phase_d08_sam2_prompt_pilot.py"
SPEC = importlib.util.spec_from_file_location("phase_d08_sam2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PhaseD08Sam2PromptPilotTest(unittest.TestCase):
    def test_prompt_overrides_are_validated_and_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overrides.json"
            path.write_text(json.dumps({"7": [10, 20, 30, 40]}))
            self.assertEqual(MODULE.load_prompt_overrides(path, 518), {7: [10.0, 20.0, 30.0, 40.0]})

    def test_invalid_prompt_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overrides.json"
            path.write_text(json.dumps({"7": [30, 20, 10, 40]}))
            with self.assertRaises(ValueError):
                MODULE.load_prompt_overrides(path, 518)

    def test_stable_nonempty_masks_pass_automatic_gate(self):
        masks = {}
        for index, phase in enumerate(MODULE.d06.PHASES):
            mask = np.zeros((100, 100), dtype=bool)
            mask[20 + index : 30 + index, 40:50] = True
            masks[phase] = mask
        metrics = MODULE.mask_metrics(masks)
        self.assertTrue(metrics["automatic_mask_gate"])
        self.assertAlmostEqual(metrics["max_min_area_ratio"], 1.0)

    def test_missing_phase_mask_fails_without_json_infinity(self):
        masks = {
            phase: np.ones((100, 100), dtype=bool)
            for phase in MODULE.d06.PHASES
        }
        masks[MODULE.d06.PHASES[-1]] = np.zeros((100, 100), dtype=bool)
        metrics = MODULE.mask_metrics(masks)
        self.assertFalse(metrics["automatic_mask_gate"])
        self.assertIsNone(metrics["max_min_area_ratio"])
        json.dumps(metrics, allow_nan=False)

    def test_overlay_preserves_expected_dimensions(self):
        image = Image.new("RGB", (518, 518), (25, 30, 35))
        mask = np.zeros((518, 518), dtype=bool)
        mask[200:240, 250:290] = True
        panel = MODULE.draw_mask_panel(
            image,
            mask,
            episode=3,
            phase="approach",
            prompt_box=[245, 195, 295, 245],
            prompt_source="automatic",
        )
        self.assertEqual(panel.size, (250, 250))
        self.assertEqual(panel.mode, "RGB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
