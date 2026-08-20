#!/usr/bin/env python3
"""Compare Phase-B control/treatment logs and enforce the matched signal gate."""

import argparse
import math
from pathlib import Path
import re


NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def values(text: str, key: str) -> list[float]:
    # Rich/tqdm may inject ANSI cursor sequences while tee is writing a
    # progress update.  Remove those before joining wrapped metric keys.
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
    )
    metrics = {key: values(text, key) for key in keys}
    # train_starvla defines total_loss = action_loss + weighted spatial loss.
    # Recover action_dit_loss only when Rich/tqdm corrupted that one printed
    # key and both source sequences are complete; never invent an independent
    # metric or mask a missing spatial signal.
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
        require("SF_PHASE_B_SMOKE=PASS" in text, f"{name} PASS marker missing")
        require("ROBOT_COMMANDS_SENT=0" in text, f"{name} robot safety marker missing")
        for key, sequence in metrics.items():
            require(sequence, f"{name} metric missing: {key}")
            require(all(math.isfinite(value) for value in sequence), f"{name} non-finite {key}")
        print(f"\n===== {name.upper()} =====")
        for key, sequence in metrics.items():
            print(f"{key}: count={len(sequence)} first={sequence[0]:.9g} last={sequence[-1]:.9g}")

    c_raw = control["relational_alignment_loss"]
    t_raw = treatment["relational_alignment_loss"]
    c_weighted = control["weighted_relational_alignment_loss"]
    t_weighted = treatment["weighted_relational_alignment_loss"]
    require(abs(c_weighted[-1]) <= 1e-12, "control weighted patch loss is not zero")
    require(t_weighted[-1] > 0.0, "treatment weighted patch loss is not positive")
    require(c_raw[-1] > 0.0 and t_raw[-1] > 0.0, "raw patch loss must be positive")
    require(
        abs(c_raw[0] - t_raw[0]) <= max(1e-4, 1e-3 * abs(c_raw[0])),
        "first-step raw patch losses differ; matched input/initialization contract failed",
    )
    require(
        abs(control["action_dit_loss"][0] - treatment["action_dit_loss"][0])
        <= max(1e-4, 1e-3 * abs(control["action_dit_loss"][0])),
        "first-step action losses differ; matched seed/data contract failed",
    )
    require(
        abs(control["parameter_update_norm/alignment_head"][-1]) <= 1e-12,
        "control alignment head unexpectedly updated",
    )
    require(
        abs(treatment["parameter_update_norm/alignment_head"][-1]) <= 1e-12,
        "treatment alignment head unexpectedly updated; projection leakage returned",
    )
    require(
        control["parameter_update_norm/spatial_forcing_lora_B"][-1] > 0.0,
        "control LoRA-B did not update",
    )
    require(
        treatment["parameter_update_norm/spatial_forcing_lora_B"][-1] > 0.0,
        "treatment LoRA-B did not update",
    )
    print("\nSF_PHASE_B_MATCHED_SIGNAL_GATE=PASS")
    print("PROJECTION_HEAD_UPDATE=0")
    print("ROBOT_COMMANDS_SENT=0")


if __name__ == "__main__":
    main()
