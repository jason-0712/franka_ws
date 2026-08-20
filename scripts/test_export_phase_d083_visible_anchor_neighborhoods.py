#!/usr/bin/env python3
"""CPU tests for Phase D-0.8.3 visible-anchor neighborhood export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if not (SCRIPT_DIR / "export_phase_d083_visible_anchor_neighborhoods.py").is_file():
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPT_DIR / "export_phase_d083_visible_anchor_neighborhoods.py"
SPEC = importlib.util.spec_from_file_location("phase_d083", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PhaseD083VisibleAnchorExportTest(unittest.TestCase):
    def test_approach_samples_only_at_or_before_phase(self):
        self.assertEqual(
            MODULE.neighborhood_indices(
                phase="approach",
                phase_frame=39,
                episode_length=200,
                approach_before=20,
                release_after=40,
                stride=5,
            ),
            [19, 24, 29, 34, 39],
        )

    def test_release_samples_only_at_or_after_phase_and_clip(self):
        self.assertEqual(
            MODULE.neighborhood_indices(
                phase="release",
                phase_frame=184,
                episode_length=200,
                approach_before=40,
                release_after=45,
                stride=5,
            ),
            [184, 189, 194, 199],
        )

    def test_phase_frame_is_always_included(self):
        for phase in ("approach", "release"):
            indices = MODULE.neighborhood_indices(
                phase=phase,
                phase_frame=12,
                episode_length=20,
                approach_before=7,
                release_after=7,
                stride=5,
            )
            self.assertIn(12, indices)

    def test_invalid_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.neighborhood_indices(
                phase="grasp",
                phase_frame=10,
                episode_length=20,
                approach_before=5,
                release_after=5,
                stride=1,
            )

    def test_thumbnail_dimensions_are_stable(self):
        image = Image.new("RGB", (518, 518), (10, 20, 30))
        panel = MODULE.draw_thumbnail(
            image,
            [],
            label="ep0000 release f=10",
            size=230,
            show_candidates=False,
        )
        self.assertEqual(panel.size, (230, 230))

    def test_sheet_supports_variable_row_lengths(self):
        panel = Image.new("RGB", (100, 100))
        sheet = MODULE.build_sheet([[panel], [panel, panel]], cell=100)
        self.assertEqual(sheet.size, (200, 200))


if __name__ == "__main__":
    unittest.main(verbosity=2)
