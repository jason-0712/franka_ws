#!/usr/bin/env python3
"""CPU tests for the Phase D-0.6 cube-candidate audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).with_name("audit_phase_d06_cube_candidates.py")
if not MODULE_PATH.is_file():
    MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_phase_d06_cube_candidates.py"
SPEC = importlib.util.spec_from_file_location("phase_d06_cube_candidates", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PhaseD06CubeCandidatesTest(unittest.TestCase):
    def test_primary_prefers_small_square_in_table_workspace(self):
        array = np.zeros((200, 200, 3), dtype=np.uint8)
        blue = np.array([15, 95, 170], dtype=np.uint8)
        array[142:154, 82:95] = blue
        # Robot-like blue parts: outside table ROI and elongated inside it.
        array[50:80, 130:160] = blue
        array[125:130, 115:165] = blue
        candidates = MODULE.rank_cube_candidates(Image.fromarray(array), "primary", 3)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["bbox_xyxy"], [82, 142, 94, 153])

    def test_wrist_returns_compact_candidate_without_fixed_roi(self):
        array = np.zeros((100, 120, 3), dtype=np.uint8)
        array[12:17, 8:14] = np.array([20, 90, 165], dtype=np.uint8)
        candidates = MODULE.rank_cube_candidates(Image.fromarray(array), "wrist", 2)
        self.assertEqual(candidates[0]["bbox_xyxy"], [8, 12, 13, 16])
        self.assertGreater(candidates[0]["score"], 0.5)

    def test_phase_order_is_monotonic(self):
        values = [1.0] * 10 + [0.0] * 20 + [1.0] * 10
        phases = MODULE.infer_phases(values, fps=10, threshold=0.5, window=3)
        self.assertEqual(sorted(phases.values()), list(phases.values()))
        self.assertEqual(phases["release"], 30)

    def test_explicit_episode_selection_rejects_unknown(self):
        self.assertEqual(MODULE.choose_episode_indices([0, 1, 2], 2, "2,0"), [2, 0])
        with self.assertRaisesRegex(ValueError, "Unknown episodes"):
            MODULE.choose_episode_indices([0, 1], 2, "3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
