#!/usr/bin/env python3
"""Compare matched seeded StarVLA open-loop CSVs without third-party packages."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


CONTINUOUS_METRICS = (
    "first_xyz_l2",
    "chunk_xyz_l2_mean",
    "first_l2",
    "chunk_l2_mean",
    "y_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", required=True)
    parser.add_argument("--treatment", nargs="+", required=True)
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Also report matched-seed metrics separately for every dataset.",
    )
    parser.add_argument(
        "--minimum-spatial-improvement-percent",
        type=float,
        default=None,
        help=(
            "Apply an evidence gate to first_xyz_l2 and chunk_xyz_l2_mean. "
            "PASS requires both aggregate improvements to reach this percentage "
            "and at least two datasets to reach it on both metrics."
        ),
    )
    return parser.parse_args()


def read_rows(path: str) -> dict[tuple[str, int, int, int], dict[str, str]]:
    result = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("inference_seed"):
                raise RuntimeError(
                    f"{path} has no inference_seed values; it was not produced by seeded evaluation"
                )
            key = (
                row["dataset"],
                int(row["frame_index"]),
                int(row["gt_action_index"]),
                int(row["inference_seed"]),
            )
            if key in result:
                raise RuntimeError(f"Duplicate key in {path}: {key}")
            result[key] = row
    return result


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    summary = {
        "first_xyz_l2": mean([float(row["first_xyz_l2"]) for row in rows]),
        "chunk_xyz_l2_mean": mean([float(row["chunk_xyz_l2_mean"]) for row in rows]),
        "first_l2": mean([float(row["first_l2"]) for row in rows]),
        "chunk_l2_mean": mean([float(row["chunk_l2_mean"]) for row in rows]),
        "y_mae": mean([abs(float(row["pred_dy"]) - float(row["gt_dy"])) for row in rows]),
    }
    pred_close = [float(row["pred_gripper"]) < 0.5 for row in rows]
    gt_close = [float(row["gt_gripper"]) < 0.5 for row in rows]
    summary["gripper_accuracy"] = mean(
        [float(prediction == target) for prediction, target in zip(pred_close, gt_close)]
    )
    false_close = [
        float(prediction) for prediction, target in zip(pred_close, gt_close) if not target
    ]
    missed_close = [
        float(not prediction) for prediction, target in zip(pred_close, gt_close) if target
    ]
    summary["false_close_rate"] = mean(false_close)
    summary["missed_close_rate"] = mean(missed_close)
    return summary


def report_across_runs(
    title: str,
    run_summaries: list[tuple[dict[str, float], dict[str, float]]],
) -> dict[str, tuple[float, float, float]]:
    print(f"\n{title}")
    aggregate = {}
    for metric in (*CONTINUOUS_METRICS, "gripper_accuracy", "false_close_rate", "missed_close_rate"):
        controls = [item[0][metric] for item in run_summaries]
        treatments = [item[1][metric] for item in run_summaries]
        deltas = [t - c for c, t in zip(controls, treatments)]
        delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        control_mean = mean(controls)
        treatment_mean = mean(treatments)
        delta_mean = mean(deltas)
        aggregate[metric] = (control_mean, treatment_mean, delta_mean)
        print(
            f"  {metric}: control_mean={control_mean:.9f} "
            f"treatment_mean={treatment_mean:.9f} "
            f"paired_delta_mean={delta_mean:+.9f} paired_delta_sd={delta_sd:.9f}"
        )
    return aggregate


def relative_improvement_percent(control: float, treatment: float) -> float:
    if control == 0.0:
        return math.nan
    return 100.0 * (control - treatment) / control


def main() -> None:
    args = parse_args()
    if len(args.control) != len(args.treatment):
        raise RuntimeError("--control and --treatment must contain the same number of CSVs")

    run_summaries = []
    control_runs = []
    treatment_runs = []
    for index, (control_path, treatment_path) in enumerate(
        zip(args.control, args.treatment), start=1
    ):
        control = read_rows(control_path)
        treatment = read_rows(treatment_path)
        if control.keys() != treatment.keys():
            only_control = list(control.keys() - treatment.keys())[:3]
            only_treatment = list(treatment.keys() - control.keys())[:3]
            raise RuntimeError(
                f"Pair {index} is not matched: control_only={only_control}, "
                f"treatment_only={only_treatment}"
            )
        keys = sorted(control)
        control_runs.append(control)
        treatment_runs.append(treatment)
        control_summary = summarize([control[key] for key in keys])
        treatment_summary = summarize([treatment[key] for key in keys])
        run_summaries.append((control_summary, treatment_summary))
        print(f"\nPAIR {index}: queries={len(keys)}")
        print(f"  control={control_path}")
        print(f"  treatment={treatment_path}")
        for metric in (*CONTINUOUS_METRICS, "gripper_accuracy", "false_close_rate", "missed_close_rate"):
            c_value = control_summary[metric]
            t_value = treatment_summary[metric]
            print(
                f"  {metric}: control={c_value:.9f} treatment={t_value:.9f} "
                f"delta={t_value - c_value:+.9f}"
            )

    aggregate = report_across_runs(
        f"ACROSS {len(run_summaries)} MATCHED SEEDS", run_summaries
    )

    dataset_aggregates = {}
    if args.by_dataset:
        dataset_names = sorted({key[0] for rows in control_runs for key in rows})
        for dataset_name in dataset_names:
            dataset_runs = []
            for control, treatment in zip(control_runs, treatment_runs):
                keys = sorted(key for key in control if key[0] == dataset_name)
                dataset_runs.append(
                    (
                        summarize([control[key] for key in keys]),
                        summarize([treatment[key] for key in keys]),
                    )
                )
            dataset_aggregates[dataset_name] = report_across_runs(
                f"DATASET {dataset_name} ACROSS {len(dataset_runs)} MATCHED SEEDS",
                dataset_runs,
            )

    if args.minimum_spatial_improvement_percent is not None:
        threshold = args.minimum_spatial_improvement_percent
        aggregate_first = relative_improvement_percent(
            aggregate["first_xyz_l2"][0], aggregate["first_xyz_l2"][1]
        )
        aggregate_chunk = relative_improvement_percent(
            aggregate["chunk_xyz_l2_mean"][0],
            aggregate["chunk_xyz_l2_mean"][1],
        )
        passing_datasets = []
        for dataset_name, values in dataset_aggregates.items():
            first = relative_improvement_percent(
                values["first_xyz_l2"][0], values["first_xyz_l2"][1]
            )
            chunk = relative_improvement_percent(
                values["chunk_xyz_l2_mean"][0],
                values["chunk_xyz_l2_mean"][1],
            )
            if first >= threshold and chunk >= threshold:
                passing_datasets.append(dataset_name)
        gate_pass = (
            aggregate_first >= threshold
            and aggregate_chunk >= threshold
            and len(passing_datasets) >= 2
        )
        print("\nSPATIAL EVIDENCE GATE")
        print(f"  threshold_percent={threshold:.6f}")
        print(f"  aggregate_first_xyz_improvement_percent={aggregate_first:.6f}")
        print(f"  aggregate_chunk_xyz_improvement_percent={aggregate_chunk:.6f}")
        print(f"  datasets_passing_both={passing_datasets}")
        print(f"SPATIAL_GATE={'PASS' if gate_pass else 'FAIL'}")
    print("\nSEEDED_MATCHED_COMPARISON=PASS")


if __name__ == "__main__":
    main()
