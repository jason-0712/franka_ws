"""Dependency-light unit tests for crop and phase selection logic."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("starvla_crop_coverage_audit.py")
SPEC = importlib.util.spec_from_file_location("starvla_crop_coverage_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CropCoverageAuditTest(unittest.TestCase):
    def test_normalized_box_to_pixels(self):
        self.assertEqual(
            MODULE.normalized_box_to_pixels((0.2, 0.25, 0.8, 1.0), 100, 80),
            (20, 20, 80, 80),
        )

    def test_phase_indices_follow_close_then_release(self):
        values = [1.0] * 20 + [0.0] * 30 + [1.0] * 10
        phases, metadata = MODULE.infer_phase_indices(values, fps=10, confirmation_window=3)
        self.assertEqual(metadata["close_index"], 20)
        self.assertEqual(metadata["release_index"], 50)
        self.assertEqual(phases["pre_grasp"], 15)
        self.assertEqual(phases["grasp"], 22)
        self.assertEqual(phases["transport"], 36)
        self.assertEqual(phases["release"], 50)
        self.assertEqual(list(phases), list(MODULE.PHASES))

    def test_phase_fallback_is_ordered(self):
        phases, metadata = MODULE.infer_phase_indices([1.0] * 20, fps=5)
        self.assertTrue(metadata["close_fallback"])
        self.assertTrue(metadata["release_fallback"])
        indices = [phases[name] for name in MODULE.PHASES]
        self.assertEqual(indices, sorted(indices))
        self.assertLess(indices[-1], 20)


if __name__ == "__main__":
    unittest.main()
