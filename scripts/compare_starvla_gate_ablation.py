#!/usr/bin/env python3
"""Compare a matched Phase-C inference-only action-gate multiplier sweep."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


SPATIAL_METRICS = ("first_xyz_l2", "chunk_xyz_l2_mean")
CONTINUOUS_METRICS = SPATIAL_METRICS + ("first_l2", "chunk_l2_mean")
ARMS = (
    ("gate0", 0.0),
    ("gate05", 0.5),
    ("gate1", 1.0),
    ("gate2", 2.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, _ in ARMS:
        parser.add_argument(f"--{name}", nargs="+", required=True)
    parser.add_argument(
        "--minimum-spatial-improvement-percent",
        type=float,
        default=1.0,
        help="Required improvement of both XYZ metrics at the nominal gate=1.",
    )
    parser.add_argument(
        "--minimum-measurable-effect-percent",
        type=float,
        default=0.5,
        help="Absolute gate=2 versus gate=0 change needed on either XYZ metric.",
    )
    return parser.parse_args()


def load(paths: list[str]) -> dict[tuple[str, int, int, int], dict[str, str]]:
    rows: dict[tuple[str, int, int, int], dict[str, str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("inference_seed"):
                    raise RuntimeError(f"{path} has no inference_seed")
                key = (
                    row["dataset"],
                    int(row["frame_index"]),
                    int(row["gt_action_index"]),
                    int(row["inference_seed"]),
                )
                if key in rows:
                    raise RuntimeError(f"Duplicate matched key in {path}: {key}")
                rows[key] = row
    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    result = {
        metric: mean([float(row[metric]) for row in rows])
        for metric in CONTINUOUS_METRICS
    }
    pred_close = [float(row["pred_gripper"]) < 0.5 for row in rows]
    gt_close = [float(row["gt_gripper"]) < 0.5 for row in rows]
    result["gripper_accuracy"] = mean(
        [float(pred == target) for pred, target in zip(pred_close, gt_close)]
    )
    result["false_close_rate"] = mean(
        [float(pred) for pred, target in zip(pred_close, gt_close) if not target]
    )
    result["missed_close_rate"] = mean(
        [float(not pred) for pred, target in zip(pred_close, gt_close) if target]
    )
    return result


def percent_change(reference: float, value: float) -> float:
    return 100.0 * (value - reference) / reference if reference else math.nan


def main() -> None:
    args = parse_args()
    tables = {name: load(getattr(args, name)) for name, _ in ARMS}
    reference_keys = set(tables["gate0"])
    for name, _ in ARMS[1:]:
        keys = set(tables[name])
        if keys != reference_keys:
            raise RuntimeError(
                f"Matched-key mismatch for {name}: missing={len(reference_keys - keys)} "
                f"extra={len(keys - reference_keys)}"
            )
    ordered_keys = sorted(reference_keys)
    summaries = {
        name: summarize([tables[name][key] for key in ordered_keys])
        for name, _ in ARMS
    }
    reference = summaries["gate0"]

    print("===== PHASE-C CAUSAL GATE SWEEP =====")
    print(f"matched_queries={len(ordered_keys)}")
    for name, multiplier in ARMS:
        print(f"\n{name} multiplier={multiplier:g}")
        for metric, value in summaries[name].items():
            delta = value - reference[metric]
            if metric in CONTINUOUS_METRICS:
                change = percent_change(reference[metric], value)
                print(
                    f"  {metric}: mean={value:.9f} delta_vs_gate0={delta:+.9f} "
                    f"change_percent={change:+.6f}"
                )
            else:
                print(
                    f"  {metric}: mean={value:.9f} "
                    f"delta_vs_gate0={delta:+.9f}"
                )

    nominal_improvements = {
        metric: -percent_change(reference[metric], summaries["gate1"][metric])
        for metric in SPATIAL_METRICS
    }
    nominal_pass = all(
        improvement >= args.minimum_spatial_improvement_percent
        for improvement in nominal_improvements.values()
    )
    gate2_changes = {
        metric: percent_change(reference[metric], summaries["gate2"][metric])
        for metric in SPATIAL_METRICS
    }
    measurable = any(
        abs(change) >= args.minimum_measurable_effect_percent
        for change in gate2_changes.values()
    )
    monotonic_benefit = all(
        summaries["gate0"][metric]
        >= summaries["gate05"][metric]
        >= summaries["gate1"][metric]
        >= summaries["gate2"][metric]
        for metric in SPATIAL_METRICS
    )

    print("\n===== CAUSAL EVIDENCE GATES =====")
    for metric in SPATIAL_METRICS:
        print(
            f"nominal_{metric}_improvement_percent="
            f"{nominal_improvements[metric]:+.6f}"
        )
        print(
            f"gate2_{metric}_change_percent={gate2_changes[metric]:+.6f}"
        )
    print(f"GATE_BRANCH_MEASURABLE={'PASS' if measurable else 'FAIL'}")
    print(f"GATE_SWEEP_MONOTONIC_SPATIAL_BENEFIT={'PASS' if monotonic_benefit else 'FAIL'}")
    print(f"NOMINAL_GATE_SPATIAL_BENEFIT={'PASS' if nominal_pass else 'FAIL'}")
    print("ROBOT_COMMANDS_SENT=0")


if __name__ == "__main__":
    main()
