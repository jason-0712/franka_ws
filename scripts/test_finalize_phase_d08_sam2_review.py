#!/usr/bin/env python3
"""Integration tests for the Phase D-0.8 manual review gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if not (SCRIPT_DIR / "finalize_phase_d08_sam2_review.py").is_file():
    SCRIPT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPT_DIR / "finalize_phase_d08_sam2_review.py"


class PhaseD08ManualReviewTest(unittest.TestCase):
    def run_case(self, decisions):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = root / "pilot.json"
            review = root / "review.json"
            output = root / "result.json"
            pilot.write_text(json.dumps({"episode_indices": list(range(10))}))
            review.write_text(json.dumps({str(i): value for i, value in enumerate(decisions)}))
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--pilot-result", str(pilot), "--decisions", str(review), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            return completed, json.loads(output.read_text()) if output.exists() else None

    def test_nine_of_ten_passes(self):
        completed, result = self.run_case([True] * 9 + [False])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(result["manual_cube_mask_gate"])

    def test_eight_of_ten_fails_gate(self):
        completed, result = self.run_case([True] * 8 + [False] * 2)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(result["manual_cube_mask_gate"])

    def test_null_decision_is_rejected(self):
        completed, result = self.run_case([True] * 9 + [None])
        self.assertNotEqual(completed.returncode, 0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
