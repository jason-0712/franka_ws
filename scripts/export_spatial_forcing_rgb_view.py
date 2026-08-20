#!/usr/bin/env python3
"""Create an RGB-only inference view of a Spatial-Forcing checkpoint.

The exported run directory contains a modified config, copied normalization
statistics, and a symbolic link to the original policy checkpoint.  It does
not duplicate the multi-gigabyte model and never modifies the training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument(
        "--action-conditioning-gate-multiplier",
        type=float,
        default=None,
        help=(
            "Optional inference-only multiplier for a Phase-C learned action gate. "
            "Use 0, 0.5, 1, and 2 for a causal gate sweep."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output_run_dir = args.output_run_dir.expanduser().absolute()
    if checkpoint.suffix not in {".pt", ".safetensors"}:
        raise ValueError(f"Unsupported checkpoint suffix: {checkpoint.suffix}")
    if output_run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output_run_dir}")

    source_run_dir = checkpoint.parents[1]
    source_config = source_run_dir / "config.yaml"
    source_statistics = source_run_dir / "dataset_statistics.json"
    for required in (source_config, source_statistics):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"Required source metadata is missing: {required}")

    config = yaml.safe_load(source_config.read_text())
    try:
        spatial = config["framework"]["spatial_forcing"]
    except Exception as exc:
        raise ValueError(
            f"Checkpoint is not configured as Spatial-Forcing: {source_config}"
        ) from exc
    spatial["teacher_enabled"] = False
    spatial["image_augmentation_enabled"] = False
    # Phase-C keeps the learned gate tensor in the checkpoint, but inference
    # never updates it.  Freezing the conditioner avoids unnecessary gradient
    # bookkeeping without changing any saved weight or forward value.
    if "action_conditioning_gate_trainable" in spatial:
        spatial["action_conditioning_gate_trainable"] = False
    if args.action_conditioning_gate_multiplier is not None:
        multiplier = float(args.action_conditioning_gate_multiplier)
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(
                "--action-conditioning-gate-multiplier must be finite and non-negative"
            )
        if "action_conditioning_gate_trainable" not in spatial:
            raise ValueError(
                "Gate multiplier requested for a checkpoint without Phase-C "
                "action conditioning"
            )
        spatial["action_conditioning_gate_multiplier"] = multiplier
    # Keep the deterministic spatial counterpart of the train-time random
    # crop.  Older checkpoints without this field preserve their historical
    # inference behavior; new corrected runs write it explicitly.
    inference_center_crop_enabled = bool(
        spatial.get("inference_center_crop_enabled", False)
    )

    temporary_dir = output_run_dir.with_name(
        f".{output_run_dir.name}.tmp-{os.getpid()}"
    )
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary export path already exists: {temporary_dir}")
    try:
        final_model_dir = temporary_dir / "final_model"
        final_model_dir.mkdir(parents=True)
        (temporary_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        shutil.copy2(source_statistics, temporary_dir / "dataset_statistics.json")
        exported_checkpoint = final_model_dir / checkpoint.name
        exported_checkpoint.symlink_to(checkpoint)
        manifest = {
            "format": "starvla_spatial_forcing_rgb_inference_view_v1",
            "source_checkpoint": str(checkpoint),
            "checkpoint_is_symlink": True,
            "teacher_enabled": False,
            "image_augmentation_enabled": False,
            "action_conditioning_gate_trainable": spatial.get(
                "action_conditioning_gate_trainable"
            ),
            "action_conditioning_gate_multiplier": spatial.get(
                "action_conditioning_gate_multiplier"
            ),
            "inference_center_crop_enabled": inference_center_crop_enabled,
        }
        (temporary_dir / "rgb_inference_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temporary_dir.rename(output_run_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    exported_checkpoint = output_run_dir / "final_model" / checkpoint.name
    if not exported_checkpoint.is_file():
        raise RuntimeError(f"Exported checkpoint link is invalid: {exported_checkpoint}")
    print("SPATIAL_FORCING_RGB_EXPORT=PASS")
    print(f"SOURCE_CHECKPOINT={checkpoint}")
    print(f"OUTPUT_RUN_DIR={output_run_dir}")
    print(f"EXPORTED_CHECKPOINT={exported_checkpoint}")
    print("CHECKPOINT_BYTES_COPIED=0")


if __name__ == "__main__":
    main()
