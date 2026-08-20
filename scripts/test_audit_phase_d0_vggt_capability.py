#!/usr/bin/env python3
"""CPU unit tests for the Phase D-0 VGGT capability audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image


MODULE_CANDIDATES = (
    Path(__file__).with_name("audit_phase_d0_vggt_capability.py"),
    Path(__file__).resolve().parents[1] / "audit_phase_d0_vggt_capability.py",
)
MODULE_PATH = next((path for path in MODULE_CANDIDATES if path.is_file()), None)
if MODULE_PATH is None:
    checked = ", ".join(str(path) for path in MODULE_CANDIDATES)
    raise FileNotFoundError(f"Phase D-0 audit module not found; checked: {checked}")
SPEC = importlib.util.spec_from_file_location("phase_d0_vggt_capability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PhaseD0VGGTCapabilityTest(unittest.TestCase):
    def test_phase_inference_uses_sustained_close_and_release(self):
        values = [1.0] * 10 + [0.0] * 20 + [1.0] * 10
        phases, metadata = MODULE.infer_phases(
            values,
            fps=10.0,
            threshold=0.5,
            window=3,
        )
        self.assertEqual(metadata["close_index"], 10)
        self.assertEqual(metadata["release_index"], 30)
        self.assertFalse(metadata["close_fallback"])
        self.assertFalse(metadata["release_fallback"])
        self.assertEqual(list(phases), list(MODULE.PHASES))
        self.assertEqual(sorted(phases.values()), list(phases.values()))

    def test_phase_inference_does_not_call_initial_open_a_release(self):
        phases, metadata = MODULE.infer_phases(
            [1.0] * 30,
            fps=15.0,
            threshold=0.5,
            window=3,
        )
        self.assertTrue(metadata["close_fallback"])
        self.assertTrue(metadata["release_fallback"])
        self.assertGreater(phases["release"], phases["grasp"])

    def test_blue_component_and_queries_use_xy_coordinates(self):
        array = np.zeros((100, 120, 3), dtype=np.uint8)
        array[60:70, 40:55] = np.array([20, 100, 160], dtype=np.uint8)
        # A huge blue distractor must be rejected by the compact-area limit.
        array[45:95, 70:115] = np.array([20, 100, 160], dtype=np.uint8)
        mask = MODULE.largest_compact_blue_component(Image.fromarray(array), "primary")
        self.assertEqual(int(mask.sum()), 150)
        points = MODULE.sample_query_points(mask, 8)
        self.assertEqual(points.shape, (8, 2))
        self.assertTrue(np.all(points[:, 0] >= 40))
        self.assertTrue(np.all(points[:, 0] < 55))
        self.assertTrue(np.all(points[:, 1] >= 60))
        self.assertTrue(np.all(points[:, 1] < 70))

    def test_state_dict_unwraps_module_prefix(self):
        tensor = torch.ones(1)
        result = MODULE.unwrap_state_dict(
            {"state_dict": {"module.aggregator.weight": tensor}}
        )
        self.assertEqual(list(result), ["aggregator.weight"])
        self.assertIs(result["aggregator.weight"], tensor)

    def test_tensor_summary_uses_exact_integer_finite_count(self):
        value = torch.ones(1_400_000)
        summary = MODULE.tensor_summary(value)
        self.assertTrue(summary["all_finite"])
        self.assertEqual(summary["invalid_count"], 0)
        self.assertEqual(summary["finite_fraction"], 1.0)

        value[-1] = float("nan")
        summary = MODULE.tensor_summary(value)
        self.assertFalse(summary["all_finite"])
        self.assertEqual(summary["invalid_count"], 1)

    def test_query_geometry_reports_exact_first_frame_reprojection(self):
        query = np.array([[2.0, 1.0], [3.0, 2.0]], dtype=np.float32)
        track = torch.tensor(
            [[[[2.0, 1.0], [3.0, 2.0]], [[3.0, 1.0], [4.0, 2.0]]]]
        )
        depth = torch.ones(1, 2, 6, 6, 1)
        yy, xx = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        world = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).float()
        world = world.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1)
        result = MODULE.summarize_query_geometry(
            {"track": track, "depth": depth, "world_points": world},
            ["approach", "grasp"],
            query,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["first_frame_reprojection_error_px_mean"], 0.0)
        self.assertEqual(result["in_bounds_fraction"], 1.0)
        self.assertEqual(result["positive_depth_fraction"], 1.0)
        self.assertEqual(result["median_temporal_pixel_step_px"], 1.0)

    def test_tracking_overlay_writes_evidence_and_coordinates(self):
        images = [Image.new("RGB", (32, 32), color=(20, 20, 20)) for _ in range(2)]
        query = np.array([[10.0, 12.0], [14.0, 15.0]], dtype=np.float32)
        track = torch.tensor(
            [[[[10.0, 12.0], [14.0, 15.0]], [[11.0, 12.0], [15.0, 15.0]]]]
        )
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:18, 8:17] = True
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "overlay.jpg"
            evidence = MODULE.save_tracking_overlay(
                images=images,
                phase_order=["approach", "grasp"],
                query_points=query,
                track=track,
                query_mask=mask,
                output_path=output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(evidence["query_mask_box_xyxy"], [8, 10, 16, 17])
            self.assertEqual(len(evidence["track_points_xy_by_phase"]["grasp"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
