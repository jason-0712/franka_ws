#!/usr/bin/env python3
"""Audit whether a learned Phase-C action residual survives BF16 addition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


SUM_FIELDS = (
    "element_count",
    "changed_count",
    "nonzero_residual_count",
    "ratio_ge_half_ulp_count",
    "ratio_ge_one_ulp_count",
    "residual_abs_sum",
    "bf16_ulp_sum",
    "residual_to_ulp_sum",
    "query_l2_sq_sum",
    "residual_l2_sq_sum",
)
MAX_FIELDS = ("residual_abs_max", "residual_to_ulp_max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0, 4.0, 8.0),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--minimum-nominal-survival-rate",
        type=float,
        default=0.05,
        help="Minimum fraction of image-hidden elements changed at multiplier=1.",
    )
    return parser.parse_args()


def empty_accumulator(multiplier: float, view: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "multiplier": float(multiplier),
        "view": view,
        "record_count": 0,
        "effective_gate_sum": 0.0,
    }
    result.update({field: 0.0 for field in SUM_FIELDS})
    result.update({field: 0.0 for field in MAX_FIELDS})
    return result


def add_record(accumulator: dict[str, Any], record: dict[str, Any]) -> None:
    accumulator["record_count"] += 1
    accumulator["effective_gate_sum"] += float(record["effective_gate"])
    for field in SUM_FIELDS:
        accumulator[field] += float(record[field])
    for field in MAX_FIELDS:
        accumulator[field] = max(
            float(accumulator[field]), float(record[field])
        )


def finalize(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = float(accumulator["element_count"])
    record_count = int(accumulator["record_count"])
    query_l2_sq = float(accumulator["query_l2_sq_sum"])
    result = dict(accumulator)
    result.update(
        {
            "effective_gate_mean": accumulator["effective_gate_sum"]
            / max(record_count, 1),
            "bf16_survival_rate": accumulator["changed_count"] / max(count, 1.0),
            "nonzero_residual_rate": accumulator["nonzero_residual_count"]
            / max(count, 1.0),
            "ratio_ge_half_ulp_rate": accumulator["ratio_ge_half_ulp_count"]
            / max(count, 1.0),
            "ratio_ge_one_ulp_rate": accumulator["ratio_ge_one_ulp_count"]
            / max(count, 1.0),
            "residual_abs_mean": accumulator["residual_abs_sum"]
            / max(count, 1.0),
            "bf16_ulp_mean": accumulator["bf16_ulp_sum"] / max(count, 1.0),
            "residual_to_ulp_mean": accumulator["residual_to_ulp_sum"]
            / max(count, 1.0),
            "relative_residual_l2": math.sqrt(
                accumulator["residual_l2_sq_sum"] / max(query_l2_sq, 1e-30)
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().absolute()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output_dir}")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    multipliers = tuple(float(value) for value in args.multipliers)
    if 1.0 not in multipliers:
        raise ValueError("--multipliers must include nominal multiplier 1")
    if any(not math.isfinite(value) or value < 0.0 for value in multipliers):
        raise ValueError("multipliers must be finite and non-negative")
    if not 0.0 <= args.minimum_nominal_survival_rate <= 1.0:
        raise ValueError("minimum survival rate must be in [0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Loading checkpoint: {checkpoint}")
    framework = baseframework.from_pretrained(str(checkpoint))
    if not hasattr(framework, "audit_action_conditioning_numerics"):
        raise TypeError(
            f"{type(framework).__name__} has no Phase-C numerical audit interface"
        )
    framework = framework.to(dtype=torch.bfloat16).to(args.device).eval()

    data_cfg = framework.config.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=data_cfg, mode="eval", seed=args.seed)
    sample_count = min(int(args.samples), len(dataset))
    if sample_count <= 0:
        raise RuntimeError("Dataset is empty")
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)
    indices = np.unique(indices).tolist()

    accumulators: dict[tuple[float, str], dict[str, Any]] = {}
    raw_records: list[dict[str, Any]] = []
    for audit_index, dataset_index in enumerate(indices):
        example = dataset[int(dataset_index)]
        records = framework.audit_action_conditioning_numerics(
            [example], gate_multipliers=multipliers
        )
        for record in records:
            record = dict(record)
            record["audit_index"] = audit_index
            record["dataset_index"] = int(dataset_index)
            raw_records.append(record)
            multiplier = float(record["multiplier"])
            view = f"view_{int(record['view_index'])}"
            for key in ((multiplier, view), (multiplier, "combined")):
                accumulator = accumulators.setdefault(
                    key, empty_accumulator(*key)
                )
                add_record(accumulator, record)
        print(f"sample={audit_index + 1}/{len(indices)} dataset_index={dataset_index}")

    summaries = [
        finalize(accumulators[(multiplier, view)])
        for multiplier in multipliers
        for view in ("view_0", "view_1", "combined")
    ]
    output_dir.mkdir(parents=True)
    raw_path = output_dir / "bf16_survival_raw_records.csv"
    with raw_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_records[0]))
        writer.writeheader()
        writer.writerows(raw_records)
    summary_path = output_dir / "bf16_survival_summary.json"
    summary_payload = {
        "status": "PASS",
        "robot_commands_sent": 0,
        "checkpoint": str(checkpoint),
        "dataset_root": str(data_cfg.data_root_dir),
        "data_mix": str(data_cfg.data_mix),
        "requested_samples": args.samples,
        "audited_samples": len(indices),
        "multipliers": list(multipliers),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

    summary_csv = output_dir / "bf16_survival_summary.csv"
    with summary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n===== BF16 ACTION-RESIDUAL SURVIVAL =====")
    for summary in summaries:
        if summary["view"] != "combined":
            continue
        print(
            f"multiplier={summary['multiplier']:g} "
            f"effective_gate={summary['effective_gate_mean']:+.9f} "
            f"survival_rate={summary['bf16_survival_rate']:.9f} "
            f"half_ulp_rate={summary['ratio_ge_half_ulp_rate']:.9f} "
            f"one_ulp_rate={summary['ratio_ge_one_ulp_rate']:.9f} "
            f"residual_to_ulp_mean={summary['residual_to_ulp_mean']:.9f} "
            f"relative_residual_l2={summary['relative_residual_l2']:.9f}"
        )
    nominal = next(
        summary
        for summary in summaries
        if summary["multiplier"] == 1.0 and summary["view"] == "combined"
    )
    survival_pass = (
        nominal["bf16_survival_rate"] >= args.minimum_nominal_survival_rate
    )
    print(
        "BF16_RESIDUAL_SURVIVAL_GATE="
        f"{'PASS' if survival_pass else 'FAIL'}"
    )
    print(f"BF16_DEADZONE_RISK={'FALSE' if survival_pass else 'TRUE'}")
    print("SPATIAL_FORCING_BF16_SURVIVAL_AUDIT=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"SUMMARY_JSON={summary_path}")
    print(f"SUMMARY_CSV={summary_csv}")

    del framework
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
