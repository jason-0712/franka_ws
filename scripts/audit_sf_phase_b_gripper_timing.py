#!/usr/bin/env python3
"""Matched temporal gripper audit for StarVLA open-loop CSV files.

The audit treats every (evaluation run, dataset) pair as one episode.  It
compares first-action gripper decisions and the action-chunk consensus signal
without invoking a policy server or sending robot commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", required=True)
    parser.add_argument("--treatment", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--chunk-consensus", type=float, default=0.75)
    parser.add_argument(
        "--minimum-pregrasp-false-close-reduction",
        type=float,
        default=0.02,
        help="Required absolute reduction, where 0.02 means two percentage points.",
    )
    parser.add_argument(
        "--maximum-closed-phase-missed-close-increase",
        type=float,
        default=0.01,
        help="Maximum tolerated absolute increase during the GT-closed phase.",
    )
    parser.add_argument(
        "--maximum-onset-mae-increase-frames",
        type=float,
        default=0.0,
        help="Maximum tolerated treatment minus control close-onset MAE.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Query:
    dataset: str
    frame_index: int
    gt_action_index: int
    inference_seed: int
    pred_gripper: float
    gt_gripper: float
    state_z: float
    pred_chunk_close_fraction: float


@dataclass
class EpisodeMetrics:
    arm: str
    run_index: int
    dataset: str
    query_count: int
    gt_close_frame: Optional[int]
    pred_close_frame: Optional[int]
    close_onset_error_frames: Optional[int]
    close_onset_abs_error_frames: Optional[int]
    gt_close_state_z: Optional[float]
    pred_close_state_z: Optional[float]
    close_state_z_error_m: Optional[float]
    close_state_z_abs_error_m: Optional[float]
    chunk_close_frame: Optional[int]
    chunk_close_onset_error_frames: Optional[int]
    chunk_close_onset_abs_error_frames: Optional[int]
    gt_release_frame: Optional[int]
    pred_release_frame: Optional[int]
    release_onset_error_frames: Optional[int]
    pred_switches: int
    chunk_candidate_switches: int
    gt_switches: int
    pregrasp_false_close_count: int
    pregrasp_query_count: int
    chunk_pregrasp_false_close_count: int
    closed_phase_missed_close_count: int
    closed_phase_query_count: int
    postrelease_false_close_count: int
    postrelease_query_count: int


def _require_fields(row: dict[str, str], path: Path) -> None:
    required = {
        "dataset",
        "frame_index",
        "gt_action_index",
        "inference_seed",
        "pred_gripper",
        "gt_gripper",
        "state_z",
        "pred_chunk_close_fraction",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise RuntimeError(f"{path} is missing CSV columns: {missing}")


def read_rows(path: Path) -> dict[tuple[str, int, int, int], Query]:
    rows: dict[tuple[str, int, int, int], Query] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            _require_fields(raw, path)
            if not raw["inference_seed"]:
                raise RuntimeError(f"{path} has empty inference_seed values")
            query = Query(
                dataset=raw["dataset"],
                frame_index=int(raw["frame_index"]),
                gt_action_index=int(raw["gt_action_index"]),
                inference_seed=int(raw["inference_seed"]),
                pred_gripper=float(raw["pred_gripper"]),
                gt_gripper=float(raw["gt_gripper"]),
                state_z=float(raw["state_z"]),
                pred_chunk_close_fraction=float(raw["pred_chunk_close_fraction"]),
            )
            key = (
                query.dataset,
                query.frame_index,
                query.gt_action_index,
                query.inference_seed,
            )
            if key in rows:
                raise RuntimeError(f"Duplicate matched key in {path}: {key}")
            rows[key] = query
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def _first_true(values: list[bool]) -> Optional[int]:
    return next((index for index, value in enumerate(values) if value), None)


def _first_false_after(values: list[bool], start: Optional[int]) -> Optional[int]:
    if start is None:
        return None
    return next(
        (index for index in range(start + 1, len(values)) if not values[index]),
        None,
    )


def _predicted_release(pred_close: list[bool], gt_close_index: Optional[int]) -> Optional[int]:
    if gt_close_index is None:
        return None
    observed_predicted_close = False
    for index in range(gt_close_index, len(pred_close)):
        if pred_close[index]:
            observed_predicted_close = True
        elif observed_predicted_close:
            return index
    return None


def _switch_count(values: list[bool]) -> int:
    return sum(left != right for left, right in zip(values, values[1:]))


def _difference(
    predicted_index: Optional[int],
    target_index: Optional[int],
    frames: list[int],
) -> Optional[int]:
    if predicted_index is None or target_index is None:
        return None
    return frames[predicted_index] - frames[target_index]


def analyze_episode(
    arm: str,
    run_index: int,
    dataset: str,
    queries: list[Query],
    close_threshold: float,
    chunk_consensus: float,
) -> EpisodeMetrics:
    queries = sorted(queries, key=lambda item: item.frame_index)
    frames = [item.frame_index for item in queries]
    gt_close = [item.gt_gripper < close_threshold for item in queries]
    pred_close = [item.pred_gripper < close_threshold for item in queries]
    chunk_close = [
        item.pred_chunk_close_fraction >= chunk_consensus for item in queries
    ]

    gt_close_index = _first_true(gt_close)
    pred_close_index = _first_true(pred_close)
    chunk_close_index = _first_true(chunk_close)
    gt_release_index = _first_false_after(gt_close, gt_close_index)
    pred_release_index = _predicted_release(pred_close, gt_close_index)

    pregrasp_indices = (
        list(range(gt_close_index)) if gt_close_index is not None else list(range(len(queries)))
    )
    closed_indices = [index for index, value in enumerate(gt_close) if value]
    postrelease_indices = (
        list(range(gt_release_index, len(queries)))
        if gt_release_index is not None
        else []
    )

    onset_error = _difference(pred_close_index, gt_close_index, frames)
    chunk_onset_error = _difference(chunk_close_index, gt_close_index, frames)
    release_error = _difference(pred_release_index, gt_release_index, frames)

    gt_close_z = queries[gt_close_index].state_z if gt_close_index is not None else None
    pred_close_z = queries[pred_close_index].state_z if pred_close_index is not None else None
    z_error = (
        pred_close_z - gt_close_z
        if pred_close_z is not None and gt_close_z is not None
        else None
    )

    return EpisodeMetrics(
        arm=arm,
        run_index=run_index,
        dataset=dataset,
        query_count=len(queries),
        gt_close_frame=(frames[gt_close_index] if gt_close_index is not None else None),
        pred_close_frame=(frames[pred_close_index] if pred_close_index is not None else None),
        close_onset_error_frames=onset_error,
        close_onset_abs_error_frames=(abs(onset_error) if onset_error is not None else None),
        gt_close_state_z=gt_close_z,
        pred_close_state_z=pred_close_z,
        close_state_z_error_m=z_error,
        close_state_z_abs_error_m=(abs(z_error) if z_error is not None else None),
        chunk_close_frame=(frames[chunk_close_index] if chunk_close_index is not None else None),
        chunk_close_onset_error_frames=chunk_onset_error,
        chunk_close_onset_abs_error_frames=(
            abs(chunk_onset_error) if chunk_onset_error is not None else None
        ),
        gt_release_frame=(frames[gt_release_index] if gt_release_index is not None else None),
        pred_release_frame=(
            frames[pred_release_index] if pred_release_index is not None else None
        ),
        release_onset_error_frames=release_error,
        pred_switches=_switch_count(pred_close),
        chunk_candidate_switches=_switch_count(chunk_close),
        gt_switches=_switch_count(gt_close),
        pregrasp_false_close_count=sum(pred_close[index] for index in pregrasp_indices),
        pregrasp_query_count=len(pregrasp_indices),
        chunk_pregrasp_false_close_count=sum(
            chunk_close[index] for index in pregrasp_indices
        ),
        closed_phase_missed_close_count=sum(
            not pred_close[index] for index in closed_indices
        ),
        closed_phase_query_count=len(closed_indices),
        postrelease_false_close_count=sum(
            pred_close[index] for index in postrelease_indices
        ),
        postrelease_query_count=len(postrelease_indices),
    )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else math.nan


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.median(materialized) if materialized else math.nan


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def aggregate(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    onset_errors = [
        item.close_onset_error_frames
        for item in episodes
        if item.close_onset_error_frames is not None
    ]
    onset_abs = [abs(value) for value in onset_errors]
    chunk_errors = [
        item.chunk_close_onset_error_frames
        for item in episodes
        if item.chunk_close_onset_error_frames is not None
    ]
    z_errors = [
        item.close_state_z_error_m
        for item in episodes
        if item.close_state_z_error_m is not None
    ]
    release_errors = [
        item.release_onset_error_frames
        for item in episodes
        if item.release_onset_error_frames is not None
    ]
    pre_false = sum(item.pregrasp_false_close_count for item in episodes)
    pre_count = sum(item.pregrasp_query_count for item in episodes)
    chunk_pre_false = sum(item.chunk_pregrasp_false_close_count for item in episodes)
    closed_missed = sum(item.closed_phase_missed_close_count for item in episodes)
    closed_count = sum(item.closed_phase_query_count for item in episodes)
    post_false = sum(item.postrelease_false_close_count for item in episodes)
    post_count = sum(item.postrelease_query_count for item in episodes)

    return {
        "episode_units": float(len(episodes)),
        "valid_close_onsets": float(len(onset_errors)),
        "missing_predicted_close_rate": _rate(
            sum(item.pred_close_frame is None for item in episodes), len(episodes)
        ),
        "close_onset_error_frames_mean": _mean(onset_errors),
        "close_onset_error_frames_median": _median(onset_errors),
        "close_onset_mae_frames": _mean(onset_abs),
        "early_close_episode_rate": _rate(
            sum(value < 0 for value in onset_errors), len(onset_errors)
        ),
        "exact_close_episode_rate": _rate(
            sum(value == 0 for value in onset_errors), len(onset_errors)
        ),
        "late_close_episode_rate": _rate(
            sum(value > 0 for value in onset_errors), len(onset_errors)
        ),
        "close_state_z_error_m_mean": _mean(z_errors),
        "close_state_z_mae_m": _mean(abs(value) for value in z_errors),
        "chunk_close_onset_error_frames_mean": _mean(chunk_errors),
        "chunk_close_onset_mae_frames": _mean(abs(value) for value in chunk_errors),
        "release_onset_error_frames_mean": _mean(release_errors),
        "release_onset_mae_frames": _mean(abs(value) for value in release_errors),
        "pred_switches_mean": _mean(item.pred_switches for item in episodes),
        "chunk_candidate_switches_mean": _mean(
            item.chunk_candidate_switches for item in episodes
        ),
        "gt_switches_mean": _mean(item.gt_switches for item in episodes),
        "pregrasp_false_close_rate": _rate(pre_false, pre_count),
        "chunk_pregrasp_false_close_rate": _rate(chunk_pre_false, pre_count),
        "closed_phase_missed_close_rate": _rate(closed_missed, closed_count),
        "postrelease_false_close_rate": _rate(post_false, post_count),
    }


def _json_number(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def print_comparison(control: dict[str, float], treatment: dict[str, float]) -> None:
    keys = (
        "close_onset_error_frames_mean",
        "close_onset_mae_frames",
        "early_close_episode_rate",
        "exact_close_episode_rate",
        "close_state_z_error_m_mean",
        "close_state_z_mae_m",
        "chunk_close_onset_error_frames_mean",
        "chunk_close_onset_mae_frames",
        "pregrasp_false_close_rate",
        "chunk_pregrasp_false_close_rate",
        "closed_phase_missed_close_rate",
        "postrelease_false_close_rate",
        "pred_switches_mean",
        "chunk_candidate_switches_mean",
    )
    print("\n===== MATCHED GRIPPER TEMPORAL SUMMARY =====")
    for key in keys:
        control_value = control[key]
        treatment_value = treatment[key]
        delta = treatment_value - control_value
        print(
            f"{key}: control={control_value:.9f} treatment={treatment_value:.9f} "
            f"delta={delta:+.9f}"
        )


def write_episode_csv(path: Path, episodes: list[EpisodeMetrics]) -> None:
    rows = [asdict(item) for item in episodes]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(args.control) != len(args.treatment):
        raise RuntimeError("--control and --treatment must have equal lengths")
    if not 0.0 <= args.chunk_consensus <= 1.0:
        raise ValueError("--chunk-consensus must be in [0, 1]")

    episodes: list[EpisodeMetrics] = []
    for run_index, (control_raw, treatment_raw) in enumerate(
        zip(args.control, args.treatment), start=1
    ):
        control_path = Path(control_raw)
        treatment_path = Path(treatment_raw)
        control_rows = read_rows(control_path)
        treatment_rows = read_rows(treatment_path)
        if control_rows.keys() != treatment_rows.keys():
            raise RuntimeError(
                f"Run pair {run_index} is not query-matched: "
                f"control_only={list(control_rows.keys() - treatment_rows.keys())[:3]}, "
                f"treatment_only={list(treatment_rows.keys() - control_rows.keys())[:3]}"
            )
        datasets = sorted({key[0] for key in control_rows})
        for dataset in datasets:
            keys = sorted(key for key in control_rows if key[0] == dataset)
            episodes.append(
                analyze_episode(
                    "control",
                    run_index,
                    dataset,
                    [control_rows[key] for key in keys],
                    args.close_threshold,
                    args.chunk_consensus,
                )
            )
            episodes.append(
                analyze_episode(
                    "treatment",
                    run_index,
                    dataset,
                    [treatment_rows[key] for key in keys],
                    args.close_threshold,
                    args.chunk_consensus,
                )
            )

    control_episodes = [item for item in episodes if item.arm == "control"]
    treatment_episodes = [item for item in episodes if item.arm == "treatment"]
    control_summary = aggregate(control_episodes)
    treatment_summary = aggregate(treatment_episodes)
    print_comparison(control_summary, treatment_summary)

    pregrasp_reduction = (
        control_summary["pregrasp_false_close_rate"]
        - treatment_summary["pregrasp_false_close_rate"]
    )
    missed_increase = (
        treatment_summary["closed_phase_missed_close_rate"]
        - control_summary["closed_phase_missed_close_rate"]
    )
    onset_mae_increase = (
        treatment_summary["close_onset_mae_frames"]
        - control_summary["close_onset_mae_frames"]
    )
    finite_gate_values = all(
        math.isfinite(value)
        for value in (pregrasp_reduction, missed_increase, onset_mae_increase)
    )
    evidence_gate = (
        finite_gate_values
        and pregrasp_reduction >= args.minimum_pregrasp_false_close_reduction
        and missed_increase
        <= args.maximum_closed_phase_missed_close_increase
        and onset_mae_increase <= args.maximum_onset_mae_increase_frames
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = args.output_dir / "gripper_episode_metrics.csv"
    summary_json = args.output_dir / "gripper_timing_summary.json"
    write_episode_csv(episode_csv, episodes)
    payload = {
        "format": "starvla_sf_phase_b_gripper_timing_v1",
        "control": {key: _json_number(value) for key, value in control_summary.items()},
        "treatment": {
            key: _json_number(value) for key, value in treatment_summary.items()
        },
        "comparison": {
            "pregrasp_false_close_reduction": _json_number(pregrasp_reduction),
            "closed_phase_missed_close_increase": _json_number(missed_increase),
            "close_onset_mae_increase_frames": _json_number(onset_mae_increase),
        },
        "thresholds": {
            "minimum_pregrasp_false_close_reduction": args.minimum_pregrasp_false_close_reduction,
            "maximum_closed_phase_missed_close_increase": args.maximum_closed_phase_missed_close_increase,
            "maximum_onset_mae_increase_frames": args.maximum_onset_mae_increase_frames,
        },
        "evidence_gate": "PASS" if evidence_gate else "FAIL",
        "robot_commands_sent": 0,
    }
    summary_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print("\n===== GRIPPER TEMPORAL EVIDENCE GATE =====")
    print(f"pregrasp_false_close_reduction={pregrasp_reduction:.9f}")
    print(f"closed_phase_missed_close_increase={missed_increase:+.9f}")
    print(f"close_onset_mae_increase_frames={onset_mae_increase:+.9f}")
    print(f"GRIPPER_TIMING_EVIDENCE_GATE={'PASS' if evidence_gate else 'FAIL'}")
    print("GRIPPER_TEMPORAL_AUDIT=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"EPISODE_CSV={episode_csv}")
    print(f"SUMMARY_JSON={summary_json}")


if __name__ == "__main__":
    main()
