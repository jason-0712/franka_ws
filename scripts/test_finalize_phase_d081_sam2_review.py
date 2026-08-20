#!/usr/bin/env python3
"""CPU unit tests for the Phase D-0.8.1 semantic gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if not (SCRIPT_DIR / "finalize_phase_d081_sam2_review.py").is_file():
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPT_DIR / "finalize_phase_d081_sam2_review.py"
SPEC = importlib.util.spec_from_file_location("phase_d081_finalize", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def review_with(default="correct"):
    return {
        "episodes": {
            str(episode): {phase: default for phase in MODULE.PHASES}
            for episode in range(10)
        }
    }


class PhaseD081ReviewTest(unittest.TestCase):
    def setUp(self):
        self.pilot = {"episode_indices": list(range(10))}

    def test_all_correct_passes(self):
        self.assertTrue(MODULE.score_review(self.pilot, review_with())["manual_semantic_gate"])

    def test_occluded_abstention_does_not_count_as_visible_error(self):
        review = review_with()
        review["episodes"]["0"]["grasp"] = "occluded_abstain"
        score = MODULE.score_review(self.pilot, review)
        self.assertTrue(score["manual_semantic_gate"])
        self.assertEqual(score["visible_phase_accuracy"], 1.0)

    def test_any_wrong_object_fails(self):
        review = review_with()
        review["episodes"]["0"]["transport"] = "wrong"
        self.assertFalse(MODULE.score_review(self.pilot, review)["manual_semantic_gate"])

    def test_visible_missing_reduces_accuracy(self):
        review = review_with()
        for episode in range(6):
            review["episodes"][str(episode)]["grasp"] = "visible_missing"
        score = MODULE.score_review(self.pilot, review)
        self.assertLess(score["visible_phase_accuracy"], 0.9)
        self.assertFalse(score["manual_semantic_gate"])

    def test_bad_label_is_rejected(self):
        review = review_with()
        review["episodes"]["0"]["grasp"] = "maybe"
        with self.assertRaises(ValueError):
            MODULE.score_review(self.pilot, review)


if __name__ == "__main__":
    unittest.main(verbosity=2)
