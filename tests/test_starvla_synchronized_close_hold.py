#!/usr/bin/env python3
"""Unit tests for generic measured-pose synchronized close holding."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CLIENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "starvla_franka_delta_pose_client.py"
)
SPEC = importlib.util.spec_from_file_location("starvla_franka_delta_pose_client", CLIENT_PATH)
CLIENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CLIENT
SPEC.loader.exec_module(CLIENT)


class SynchronizedCloseHoldTransitionTest(unittest.TestCase):
    def transition(self, **overrides):
        values = {
            "enabled": True,
            "hold_active": False,
            "physical_gripper_enabled": True,
            "last_published_gripper": 1.0,
            "grasp_validation_active": False,
            "close_candidate": False,
        }
        values.update(overrides)
        return CLIENT.synchronized_close_hold_transition(**values)

    def test_close_candidate_activates_hold_only_while_open(self):
        self.assertEqual(self.transition(close_candidate=True), "activate")
        self.assertEqual(
            self.transition(
                close_candidate=True,
                last_published_gripper=0.0,
            ),
            "inactive",
        )

    def test_missing_candidate_cancels_only_before_physical_close(self):
        self.assertEqual(
            self.transition(hold_active=True, close_candidate=False),
            "cancel",
        )
        self.assertEqual(
            self.transition(
                hold_active=True,
                close_candidate=False,
                last_published_gripper=0.0,
            ),
            "keep",
        )
        self.assertEqual(
            self.transition(
                hold_active=True,
                close_candidate=False,
                grasp_validation_active=True,
            ),
            "keep",
        )

    def test_disabled_or_dry_run_never_activates(self):
        self.assertEqual(
            self.transition(enabled=False, close_candidate=True),
            "inactive",
        )
        self.assertEqual(
            self.transition(
                physical_gripper_enabled=False,
                close_candidate=True,
            ),
            "inactive",
        )

    def test_nonfinite_gripper_command_is_rejected(self):
        with self.assertRaises(ValueError):
            self.transition(last_published_gripper=float("nan"))

    def test_flickering_candidate_cancels_then_rearms(self):
        hold_active = False
        observed = []
        for close_candidate in (True, True, False, True, True, True):
            transition = self.transition(
                hold_active=hold_active,
                close_candidate=close_candidate,
            )
            observed.append(transition)
            if transition == "activate":
                hold_active = True
            elif transition == "cancel":
                hold_active = False
        self.assertEqual(
            observed,
            ["activate", "keep", "cancel", "activate", "keep", "keep"],
        )


class SynchronizedCloseHoldCliTest(unittest.TestCase):
    def test_enabled_by_default_and_raw_ablation_is_available(self):
        with patch.object(sys, "argv", [str(CLIENT_PATH)]):
            self.assertTrue(CLIENT.parse_args().synchronized_close_hold)
        with patch.object(
            sys,
            "argv",
            [str(CLIENT_PATH), "--no-synchronized-close-hold"],
        ):
            self.assertFalse(CLIENT.parse_args().synchronized_close_hold)
        with patch.object(
            sys,
            "argv",
            [str(CLIENT_PATH), "--synchronized-close-hold"],
        ):
            self.assertTrue(CLIENT.parse_args().synchronized_close_hold)


if __name__ == "__main__":
    unittest.main(verbosity=2)
