#!/usr/bin/env python3
"""Install StarVLA modality.json for Quest3 Franka delta end-pose datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODALITY = {
    "state": {
        "eef_position": {
            "start": 0,
            "end": 3,
            "original_key": "observation.state.cartesian",
        },
        "eef_rotation": {
            "start": 3,
            "end": 6,
            "original_key": "observation.state.cartesian",
            "rotation_type": "euler_angles_rpy",
        },
        "gripper": {
            "start": 6,
            "end": 7,
            "original_key": "observation.state",
        },
    },
    "action": {
        "delta_eef_position": {
            "start": 0,
            "end": 3,
            "original_key": "action",
            "absolute": False,
        },
        "delta_eef_rotation": {
            "start": 3,
            "end": 6,
            "original_key": "action",
            "absolute": False,
            "rotation_type": "euler_angles_rpy",
        },
        "gripper": {
            "start": 6,
            "end": 7,
            "original_key": "action",
            "absolute": True,
        },
    },
    "video": {
        "primary_image": {
            "original_key": "observation.images.primary",
        },
    },
    "annotation": {
        "human.action.task_description": {
            "original_key": "task_index",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dase-hw101/franka_ws/dataset/snkdjn"),
        help="Directory containing quest3_franka_tele_XXXX dataset folders.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Numeric dataset ids to update. Defaults to all "
            "quest3_franka_tele_XXXX folders under --root."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def has_parquet(dataset_dir: Path, info: dict) -> bool:
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    parquet = dataset_dir / data_path.format(episode_chunk=0, episode_index=0)
    return parquet.exists()


def main() -> None:
    args = parse_args()
    dataset_ids = args.ids
    if dataset_ids is None:
        dataset_ids = sorted(
            int(path.name.rsplit("_", 1)[1])
            for path in args.root.glob("quest3_franka_tele_*")
            if path.name.rsplit("_", 1)[-1].isdigit()
        )

    updated = []
    skipped = []

    for dataset_id in dataset_ids:
        name = f"quest3_franka_tele_{dataset_id:04d}"
        dataset_dir = args.root / name
        info_path = dataset_dir / "meta" / "info.json"
        modality_path = dataset_dir / "meta" / "modality.json"

        if not info_path.exists():
            skipped.append((name, "missing info.json"))
            continue

        info = json.loads(info_path.read_text())
        action_feature = info.get("features", {}).get("action", {})
        if action_feature.get("shape") != [7]:
            skipped.append((name, f"unexpected action shape {action_feature.get('shape')}"))
            continue
        if not has_parquet(dataset_dir, info):
            skipped.append((name, "missing parquet"))
            continue
        if modality_path.exists() and not args.overwrite:
            skipped.append((name, "modality.json exists"))
            continue

        updated.append(name)
        if not args.dry_run:
            modality_path.parent.mkdir(parents=True, exist_ok=True)
            modality_path.write_text(json.dumps(MODALITY, indent=2) + "\n")

    print(f"root: {args.root}")
    print(f"updated: {len(updated)}")
    for name in updated:
        print(f"  {name}")
    print(f"skipped: {len(skipped)}")
    for name, reason in skipped:
        print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
