#!/usr/bin/env python3
"""Enforce the matched Phase-C zero-gated action-conditioning signal gate."""

import argparse
import math
from pathlib import Path
import re


NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def values(text: str, key: str) -> list[float]:
    compact = "".join(ANSI_ESCAPE.sub("", text).split())
    pattern = re.compile(re.escape(repr(key)) + r":" + NUMBER)
    return [float(value) for value in pattern.findall(compact)]


def load(path: Path) -> tuple[str, dict[str, list[float]]]:
    text = path.read_text(errors="replace")
    keys = (
        "action_dit_loss",
        "total_loss",
        "relational_alignment_loss",
        "weighted_relational_alignment_loss",
        "parameter_update_norm/spatial_forcing_lora_B",
        "parameter_update_norm/alignment_head",
        "parameter_update_norm/action_spatial_gate",
        "parameter_update_norm/action_spatial_out_proj",
        "parameter_norm/action_spatial_gate",
    )
    metrics = {key: values(text, key) for key in keys}
    if not metrics["action_dit_loss"]:
        total = metrics["total_loss"]
        weighted = metrics["weighted_relational_alignment_loss"]
        if total and len(total) == len(weighted):
            metrics["action_dit_loss"] = [
                total_value - weighted_value
                for total_value, weighted_value in zip(total, weighted)
            ]
            print(
                f"ACTION_DIT_LOSS_RECOVERED={path} count={len(total)} "
                "formula=total_loss-weighted_relational_alignment_loss"
            )
    return text, metrics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--treatment-log", type=Path, required=True)
    args = parser.parse_args()
    control_text, control = load(args.control_log)
    treatment_text, treatment = load(args.treatment_log)

    for name, text, metrics in (
        ("control", control_text, control),
        ("treatment", treatment_text, treatment),
    ):
        require("SF_PHASE_C_SMOKE=PASS" in text, f"{name} PASS marker missing")
        require("ROBOT_COMMANDS_SENT=0" in text, f"{name} safety marker missing")
        for key, sequence in metrics.items():
            require(sequence, f"{name} metric missing: {key}")
            require(
                all(math.isfinite(value) for value in sequence),
                f"{name} non-finite {key}",
            )
        print(f"\n===== {name.upper()} =====")
        for key, sequence in metrics.items():
            print(
                f"{key}: count={len(sequence)} first={sequence[0]:.9g} "
                f"last={sequence[-1]:.9g}"
            )

    c_action = control["action_dit_loss"]
    t_action = treatment["action_dit_loss"]
    c_patch = control["relational_alignment_loss"]
    t_patch = treatment["relational_alignment_loss"]
    c_weighted = control["weighted_relational_alignment_loss"]
    t_weighted = treatment["weighted_relational_alignment_loss"]

    require(c_weighted[0] > 0.0 and t_weighted[0] > 0.0, "both arms need patch supervision")
    require(
        abs(c_weighted[0] - t_weighted[0]) <= max(1e-5, 1e-3 * abs(c_weighted[0])),
        "first-step weighted patch losses differ",
    )
    require(
        abs(c_patch[0] - t_patch[0]) <= max(1e-4, 1e-3 * abs(c_patch[0])),
        "first-step raw patch losses differ",
    )
    require(
        abs(c_action[0] - t_action[0]) <= max(1e-4, 1e-3 * abs(c_action[0])),
        "zero gate is not an action-loss identity at step one",
    )
    require(
        abs(control["parameter_update_norm/alignment_head"][-1]) <= 1e-12
        and abs(treatment["parameter_update_norm/alignment_head"][-1]) <= 1e-12,
        "unused projection head updated",
    )
    require(
        abs(control["parameter_update_norm/action_spatial_gate"][-1]) <= 1e-12,
        "control action gate unexpectedly updated",
    )
    require(
        abs(control["parameter_update_norm/action_spatial_out_proj"][-1]) <= 1e-12,
        "control action conditioner unexpectedly updated",
    )
    require(
        treatment["parameter_update_norm/action_spatial_gate"][-1] > 0.0,
        "treatment action gate did not update",
    )
    require(
        treatment["parameter_update_norm/action_spatial_out_proj"][-1] > 0.0,
        "treatment action conditioner projection did not update after the gate opened",
    )
    require(
        treatment["parameter_norm/action_spatial_gate"][-1] > 0.0,
        "treatment action gate remained exactly zero",
    )
    require(
        control["parameter_update_norm/spatial_forcing_lora_B"][-1] > 0.0
        and treatment["parameter_update_norm/spatial_forcing_lora_B"][-1] > 0.0,
        "one arm has no LoRA-B update",
    )

    print("\nSF_PHASE_C_MATCHED_SIGNAL_GATE=PASS")
    print("ZERO_GATE_STEP1_IDENTITY=PASS")
    print("CONTROL_ACTION_CONDITIONER_UPDATE=0")
    print("TREATMENT_ACTION_CONDITIONER_UPDATE=POSITIVE")
    print("PROJECTION_HEAD_UPDATE=0")
    print("ROBOT_COMMANDS_SENT=0")


if __name__ == "__main__":
    main()
