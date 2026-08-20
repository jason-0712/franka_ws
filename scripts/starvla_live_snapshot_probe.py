#!/usr/bin/env python3
"""Capture one live Franka observation and query StarVLA without commanding ROS.

This diagnostic intentionally creates subscriptions only.  It stores the exact
primary/wrist RGB images, raw 7D policy state, server metadata, and returned
action chunk so multiple object placements can be compared while the robot
remains fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image as PILImage
import rclpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import starvla_franka_delta_pose_client as deployment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="192.168.1.113")
    parser.add_argument("--policy-port", type=int, default=10097)
    parser.add_argument(
        "--task",
        default="pick up the cube and place it on the box",
    )
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-observation-age", type=float, default=1.0)
    parser.add_argument(
        "--primary-image-topic",
        default="/right/right_third_person_camera/color/image_raw/compressed",
    )
    parser.add_argument(
        "--wrist-image-topic",
        default="/right/right_wrist_camera/color/image_raw/compressed",
    )
    parser.add_argument("--current-pose-topic", default="/current_pose")
    parser.add_argument(
        "--gripper-state-topic",
        default="/franka_gripper/joint_states",
    )
    parser.add_argument("--target-pose-topic", default="/target_pose")
    parser.add_argument("--gripper-max-width", type=float, default=0.08)
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_rgb(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Expected RGB HxWx3 image, got {image.shape}")
    PILImage.fromarray(image, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    if args.max_observation_age <= 0.0:
        raise ValueError("--max-observation-age must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not args.label.strip():
        raise ValueError("--label must not be empty")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty snapshot directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    node_args = SimpleNamespace(
        primary_image_topic=args.primary_image_topic,
        wrist_image_topic=args.wrist_image_topic,
        compressed_image=True,
        current_pose_topic=args.current_pose_topic,
        target_pose_topic=args.target_pose_topic,
        target_frame_id="base",
        gripper_command_topic="/franka_gripper/commands",
        gripper_state_topic=args.gripper_state_topic,
        gripper_max_width=args.gripper_max_width,
        disable_gripper=True,
        publish_rate=40.0,
    )
    query_args = SimpleNamespace(
        task=args.task,
        unnorm_key=args.unnorm_key,
        image_size=args.image_size,
    )

    rclpy.init()
    node = deployment.FrankaCartesianObservationNode(node_args)
    executor = deployment.BackgroundROSExecutor(node)
    executor.start()
    client = None
    try:
        images, position, rotation = node.wait_for_observation(timeout=args.timeout)
        executor.raise_if_failed()

        ages = node.refresh_stale_observations(args.max_observation_age)
        stale = {key: age for key, age in ages.items() if age > args.max_observation_age}
        if stale:
            raise RuntimeError(
                f"Stale observation(s), max age={args.max_observation_age:.3f}s: {stale}"
            )

        existing_publishers = node.target_pose_publishers()
        if existing_publishers != 0:
            raise RuntimeError(
                f"Refusing snapshot while {args.target_pose_topic} has "
                f"{existing_publishers} publisher(s)"
            )

        # Re-copy after the freshness check so the saved frames are the exact
        # arrays passed to request_action_chunk below.
        images = node.copy_images()
        position, rotation = node.copy_current_pose()
        gripper_width, gripper_closed = node.copy_gripper_state()
        raw_state = np.concatenate(
            [
                position,
                rotation.as_euler("xyz"),
                np.array([gripper_closed], dtype=np.float64),
            ]
        )

        client = deployment.MinimalWebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
        )
        metadata = client.get_server_metadata()
        actions, timing = deployment.request_action_chunk(
            client,
            images,
            raw_state,
            query_args,
            request_id=0,
        )

        resized_images = [
            deployment.resize_rgb(image, args.image_size) for image in images
        ]
        for camera_name, original, resized in zip(
            ("primary", "wrist"), images, resized_images
        ):
            save_rgb(output_dir / f"{camera_name}_original.png", original)
            save_rgb(output_dir / f"{camera_name}_{args.image_size}.png", resized)

        close_fraction = float(np.mean(actions[:, 6] < 0.5))
        record = {
            "label": args.label,
            "policy_host": args.policy_host,
            "policy_port": args.policy_port,
            "task": args.task,
            "metadata": metadata,
            "image_topics": {
                "primary": args.primary_image_topic,
                "wrist": args.wrist_image_topic,
            },
            "image_shapes": {
                "primary": list(images[0].shape),
                "wrist": list(images[1].shape),
            },
            "image_sha256": {
                "primary": sha256_array(images[0]),
                "wrist": sha256_array(images[1]),
            },
            "observation_ages_s": ages,
            "raw_policy_state": raw_state.tolist(),
            "measured_gripper_width_m": gripper_width,
            "actions": actions.tolist(),
            "first_action": actions[0].tolist(),
            "translation_mean": actions[:, :3].mean(axis=0).tolist(),
            "translation_min": actions[:, :3].min(axis=0).tolist(),
            "translation_max": actions[:, :3].max(axis=0).tolist(),
            "close_fraction": close_fraction,
            "open_fraction": 1.0 - close_fraction,
            "timing": timing,
            "target_pose_publishers_before": existing_publishers,
            "ros_command_publishers_created": 0,
        }
        result_path = output_dir / "probe.json"
        with result_path.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")

        print("SNAPSHOT_PROBE=PASS")
        print("ROBOT_COMMANDS_SENT=0")
        print("label=", args.label)
        print("ckpt_path=", metadata.get("ckpt_path"))
        print("raw_policy_state=", raw_state)
        print("action_chunk_shape=", actions.shape)
        print("first_action=", actions[0])
        print("translation_mean=", actions[:, :3].mean(axis=0))
        print("close_fraction=", close_fraction)
        print("client_policy_roundtrip_s=", timing.get("client_policy_roundtrip_s"))
        print("result=", result_path)
    finally:
        if client is not None:
            client.close()
        executor.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
