#!/usr/bin/env python3
"""Self-contained test for compare_starvla_gate_ablation.py."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("compare_starvla_gate_ablation.py")
FIELDS = (
    "dataset",
    "frame_index",
    "gt_action_index",
    "inference_seed",
    "first_xyz_l2",
    "chunk_xyz_l2_mean",
    "first_l2",
    "chunk_l2_mean",
    "pred_gripper",
    "gt_gripper",
)


class GateAblationComparisonTest(unittest.TestCase):
    def test_matched_improving_sweep_passes_evidence_gates(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths = {}
            for name, scale in (
                ("gate0", 1.00),
                ("gate05", 0.99),
                ("gate1", 0.98),
                ("gate2", 0.96),
            ):
                path = directory / f"{name}.csv"
                paths[name] = path
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    for frame in (0, 5):
                        writer.writerow(
                            {
                                "dataset": "episode",
                                "frame_index": frame,
                                "gt_action_index": frame,
                                "inference_seed": 42,
                                "first_xyz_l2": scale,
                                "chunk_xyz_l2_mean": 2.0 * scale,
                                "first_l2": scale,
                                "chunk_l2_mean": 2.0 * scale,
                                "pred_gripper": 1.0,
                                "gt_gripper": 1.0,
                            }
                        )
            command = [sys.executable, str(SCRIPT)]
            for name, _ in (("gate0", 0), ("gate05", 0), ("gate1", 0), ("gate2", 0)):
                command.extend((f"--{name}", str(paths[name])))
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            self.assertIn("matched_queries=2", result.stdout)
            self.assertIn("GATE_BRANCH_MEASURABLE=PASS", result.stdout)
            self.assertIn("GATE_SWEEP_MONOTONIC_SPATIAL_BENEFIT=PASS", result.stdout)
            self.assertIn("NOMINAL_GATE_SPATIAL_BENEFIT=PASS", result.stdout)
            self.assertIn("ROBOT_COMMANDS_SENT=0", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
