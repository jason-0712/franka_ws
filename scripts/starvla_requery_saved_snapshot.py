#!/usr/bin/env python3
"""Query one StarVLA server with a previously saved live snapshot.

The source directory must have been produced by
``starvla_live_snapshot_probe.py``.  This tool reuses the exact primary/wrist
RGB arrays and raw policy state, making model comparisons matched at the byte
level.  It does not initialize ROS and cannot publish robot commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import starvla_franka_delta_pose_client as deployment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot-dir", type=Path, required=True)
    parser.add_argument("--policy-host", default="192.168.1.113")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(array)


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.policy_port <= 0 or args.policy_port > 65535:
        raise ValueError("--policy-port must be in [1,65535]")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")

    source_dir = args.source_snapshot_dir.expanduser().resolve(strict=True)
    source_probe_path = source_dir / "probe.json"
    primary_path = source_dir / "primary_original.png"
    wrist_path = source_dir / "wrist_original.png"
    for required in (source_probe_path, primary_path, wrist_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing source snapshot file: {required}")

    output_dir = args.output_dir.expanduser().absolute()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_probe = json.loads(source_probe_path.read_text())
    raw_state = np.asarray(source_probe["raw_policy_state"], dtype=np.float64)
    if raw_state.shape != (7,) or not np.isfinite(raw_state).all():
        raise ValueError(f"Source raw_policy_state must be finite shape (7,), got {raw_state}")
    images = [load_rgb(primary_path), load_rgb(wrist_path)]
    hashes = {
        "primary": sha256_array(images[0]),
        "wrist": sha256_array(images[1]),
    }
    expected_hashes = source_probe.get("image_sha256", {})
    for camera, digest in hashes.items():
        if camera in expected_hashes and expected_hashes[camera] != digest:
            raise RuntimeError(
                f"Source {camera} image hash changed: expected={expected_hashes[camera]} "
                f"actual={digest}"
            )

    task = args.task or source_probe.get("task")
    if not task:
        raise ValueError("Task is absent from source probe; pass --task")
    query_args = SimpleNamespace(
        task=task,
        unnorm_key=args.unnorm_key,
        image_size=args.image_size,
    )
    client = deployment.MinimalWebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    try:
        metadata = client.get_server_metadata()
        actions, timing = deployment.request_action_chunk(
            client,
            images,
            raw_state,
            query_args,
            request_id=0,
        )
    finally:
        client.close()

    close_fraction = float(np.mean(actions[:, 6] < 0.5))
    record = {
        "label": args.label or source_probe.get("label"),
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "task": task,
        "metadata": metadata,
        "source_snapshot_dir": str(source_dir),
        "source_probe": str(source_probe_path),
        "source_snapshot_image_sha256": hashes,
        "raw_policy_state": raw_state.tolist(),
        "actions": actions.tolist(),
        "first_action": actions[0].tolist(),
        "translation_mean": actions[:, :3].mean(axis=0).tolist(),
        "translation_min": actions[:, :3].min(axis=0).tolist(),
        "translation_max": actions[:, :3].max(axis=0).tolist(),
        "close_fraction": close_fraction,
        "open_fraction": 1.0 - close_fraction,
        "timing": timing,
        "ros_initialized": False,
        "ros_command_publishers_created": 0,
    }
    result_path = output_dir / "probe.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    print("SAVED_SNAPSHOT_REQUERY=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print("SOURCE_IMAGE_HASH_MATCH=PASS")
    print("label=", record["label"])
    print("ckpt_path=", metadata.get("ckpt_path"))
    print("first_action=", actions[0])
    print("translation_mean=", actions[:, :3].mean(axis=0))
    print("close_fraction=", close_fraction)
    print("result=", result_path)


if __name__ == "__main__":
    main()
