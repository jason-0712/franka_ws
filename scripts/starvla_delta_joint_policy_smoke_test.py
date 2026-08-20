#!/usr/bin/env python3
"""Smoke-test a StarVLA 8D delta-joint policy server without moving the robot.

Run this from the starVLA repo root on the training/server machine:

    PYTHONPATH=$(pwd):$PYTHONPATH python /path/to/starvla_delta_joint_policy_smoke_test.py \
      --video /data/hanyu/franka_real_delta/snkdjn_delta/franka_test_161/videos/chunk-000/observation.images.primary/episode_000000.mp4 \
      --host 127.0.0.1 \
      --port 10093 \
      --task "pick up the cube and place it on the bowl" \
      --unnorm-key franka_test_161

The script only prints server outputs. It does not publish ROS commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--video", type=Path, help="Path to an episode_*.mp4 file.")
    parser.add_argument("--image", type=Path, help="Optional path to a single RGB image.")
    parser.add_argument("--task", default="pick up the cube and place it on the bowl")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def read_video_first_frame(path: Path) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to read mp4 here. Install/use the starVLA env that has `av`, "
            "or pass --image /path/to/frame.png instead."
        ) from exc

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            return frame.to_rgb().to_ndarray()
    raise RuntimeError(f"No video frame found in {path}")


def read_image(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        image = Image.open(args.image).convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
    elif args.video:
        arr = read_video_first_frame(args.video)
    else:
        raise ValueError("Pass either --video or --image")

    image = Image.fromarray(arr).resize((args.image_size, args.image_size))
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    image = read_image(args)

    try:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import StarVLA websocket client. Run from the starVLA repo root with "
            "`export PYTHONPATH=$(pwd):$PYTHONPATH`."
        ) from exc

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    print("=== server metadata ===")
    print(metadata)

    payload = {
        "examples": [
            {
                "image": [image],
                "lang": args.task,
            }
        ],
    }
    if args.unnorm_key:
        payload["unnorm_key"] = args.unnorm_key
    elif metadata.get("default_unnorm_key") is not None:
        payload["unnorm_key"] = metadata["default_unnorm_key"]
    elif len(metadata.get("available_unnorm_keys", [])) == 1:
        payload["unnorm_key"] = metadata["available_unnorm_keys"][0]
    else:
        print("Multiple unnorm keys available; pass --unnorm-key one of:")
        print(metadata.get("available_unnorm_keys"))
        client.close()
        sys.exit(2)

    request = {
        "type": "predict_action",
        "request_id": "delta-joint-smoke-test",
        "payload": payload,
    }
    response = client.predict_action(request)
    client.close()

    if not response.get("ok", False):
        print("=== server error ===")
        print(response)
        sys.exit(1)

    actions = np.asarray(response["data"]["actions"][0], dtype=np.float32)
    print("=== action chunk ===")
    print("shape:", actions.shape)
    print(actions)
    print("joint_delta_min:", actions[:, :7].min(axis=0))
    print("joint_delta_max:", actions[:, :7].max(axis=0))
    print("joint_delta_abs_max:", float(np.max(np.abs(actions[:, :7]))))
    print("gripper_values:", actions[:, 7])


if __name__ == "__main__":
    main()
