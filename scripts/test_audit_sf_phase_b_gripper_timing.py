#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("audit_sf_phase_b_gripper_timing.py")
SPEC = importlib.util.spec_from_file_location("gripper_timing", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GripperTimingAuditTest(unittest.TestCase):
    def _queries(self, prediction):
        gt = [1.0, 1.0, 0.0, 0.0, 1.0]
        result = []
        for index, (pred, target) in enumerate(zip(prediction, gt)):
            result.append(
                MODULE.Query(
                    dataset="episode",
                    frame_index=index * 5,
                    gt_action_index=index * 5,
                    inference_seed=100 + index,
                    pred_gripper=pred,
                    gt_gripper=target,
                    state_z=0.20 - index * 0.01,
                    pred_chunk_close_fraction=1.0 if pred < 0.5 else 0.0,
                )
            )
        return result

    def test_exact_treatment_improves_early_control(self):
        control = MODULE.analyze_episode(
            "control", 1, "episode", self._queries([1, 0, 0, 0, 1]), 0.5, 0.75
        )
        treatment = MODULE.analyze_episode(
            "treatment", 1, "episode", self._queries([1, 1, 0, 0, 1]), 0.5, 0.75
        )
        self.assertEqual(control.close_onset_error_frames, -5)
        self.assertEqual(treatment.close_onset_error_frames, 0)
        self.assertEqual(control.pregrasp_false_close_count, 1)
        self.assertEqual(treatment.pregrasp_false_close_count, 0)
        self.assertAlmostEqual(control.close_state_z_error_m, 0.01)
        self.assertAlmostEqual(treatment.close_state_z_error_m, 0.0)

    def test_release_and_switches(self):
        metrics = MODULE.analyze_episode(
            "treatment", 1, "episode", self._queries([1, 1, 0, 0, 1]), 0.5, 0.75
        )
        self.assertEqual(metrics.gt_release_frame, 20)
        self.assertEqual(metrics.pred_release_frame, 20)
        self.assertEqual(metrics.release_onset_error_frames, 0)
        self.assertEqual(metrics.pred_switches, 2)

    def test_aggregate_uses_pooled_phase_rates(self):
        first = MODULE.analyze_episode(
            "control", 1, "episode_a", self._queries([1, 0, 0, 0, 1]), 0.5, 0.75
        )
        second = MODULE.analyze_episode(
            "control", 1, "episode_b", self._queries([1, 1, 0, 0, 1]), 0.5, 0.75
        )
        summary = MODULE.aggregate([first, second])
        self.assertAlmostEqual(summary["pregrasp_false_close_rate"], 0.25)
        self.assertAlmostEqual(summary["close_onset_mae_frames"], 2.5)

    def test_read_rows_rejects_missing_seed(self):
        fields = [
            "dataset",
            "frame_index",
            "gt_action_index",
            "inference_seed",
            "pred_gripper",
            "gt_gripper",
            "state_z",
            "pred_chunk_close_fraction",
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "episode",
                        "frame_index": 0,
                        "gt_action_index": 0,
                        "inference_seed": "",
                        "pred_gripper": 1,
                        "gt_gripper": 1,
                        "state_z": 0.2,
                        "pred_chunk_close_fraction": 0,
                    }
                )
            with self.assertRaisesRegex(RuntimeError, "empty inference_seed"):
                MODULE.read_rows(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
