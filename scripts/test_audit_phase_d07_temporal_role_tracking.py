#!/usr/bin/env python3
"""CPU tests for Phase D-0.7 temporal-role tracking."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if not (SCRIPT_DIR / "audit_phase_d07_temporal_role_tracking.py").is_file():
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "audit_phase_d07_temporal_role_tracking.py"
SPEC = importlib.util.spec_from_file_location("phase_d07_temporal_role", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_track(centers):
    offsets = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    return torch.stack([torch.tensor(center) + offsets for center in centers])


class PhaseD07TemporalRoleTest(unittest.TestCase):
    def test_cube_role_beats_static_background_and_early_moving_gripper(self):
        cube = MODULE.temporal_role_metrics(
            make_track([(100, 100), (101, 100), (101, 101), (130, 105), (145, 110)]),
            width=518,
            height=518,
            candidate_score=0.7,
        )
        background = MODULE.temporal_role_metrics(
            make_track([(100, 100)] * 5), width=518, height=518, candidate_score=0.7
        )
        gripper = MODULE.temporal_role_metrics(
            make_track([(100, 100), (125, 100), (145, 100), (160, 105), (170, 110)]),
            width=518,
            height=518,
            candidate_score=0.7,
        )
        self.assertGreater(cube["role_score"], background["role_score"])
        self.assertGreater(cube["role_score"], gripper["role_score"])
        self.assertTrue(
            MODULE.passes_role_gate(cube, max_early=20, min_late=8, min_in_bounds=0.8)
        )
        self.assertFalse(
            MODULE.passes_role_gate(background, max_early=20, min_late=8, min_in_bounds=0.8)
        )
        self.assertFalse(
            MODULE.passes_role_gate(gripper, max_early=20, min_late=8, min_in_bounds=0.8)
        )

    def test_out_of_bounds_track_fails_gate(self):
        metrics = MODULE.temporal_role_metrics(
            make_track([(100, 100), (100, 100), (100, 100), (600, 600), (650, 650)]),
            width=518,
            height=518,
            candidate_score=0.9,
        )
        self.assertFalse(
            MODULE.passes_role_gate(metrics, max_early=20, min_late=8, min_in_bounds=0.8)
        )

    def test_candidate_points_stay_in_component_box(self):
        mask = np.zeros((50, 60), dtype=bool)
        mask[20:30, 10:22] = True
        candidate = {"bbox_xyxy": [10, 20, 21, 29]}
        points = MODULE.points_for_candidate(mask, candidate, 5)
        self.assertEqual(points.shape, (5, 2))
        self.assertTrue(np.all((10 <= points[:, 0]) & (points[:, 0] <= 21)))
        self.assertTrue(np.all((20 <= points[:, 1]) & (points[:, 1] <= 29)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
