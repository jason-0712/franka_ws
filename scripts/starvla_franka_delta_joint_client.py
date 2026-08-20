#!/usr/bin/env python3
"""Safe StarVLA delta-joint client for Franka FR3.

This script runs on the ROS/Franka machine. It connects to a StarVLA websocket
policy server, builds an observation from the latest camera frame, receives an
8D action chunk:

    [delta_joint_0, ..., delta_joint_6, gripper]

and, only when --execute is passed, publishes conservative joint targets to
/target_joint.

Default mode is dry-run and does not move the robot.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray


FR3_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

# Conservative FR3/Panda-style limits. These are still a last line of defense;
# the primary safety limits are max_delta, max_steps, low rate, and E-stop.
FR3_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=10093)
    parser.add_argument("--task", default="pick up the cube and place it on the bowl")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--starvla-root", default=None)
    parser.add_argument("--image-topic", default="/right/right_third_person_camera/color/image_raw")
    parser.add_argument("--compressed-image", action="store_true")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--target-joint-topic", default="/target_joint")
    parser.add_argument(
        "--gripper-command-topic",
        default="/gripper/gripper_position_controller/commands",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rate", type=float, default=2.0, help="Execution rate in Hz.")
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=20.0,
        help="Rate in Hz for repeatedly publishing the held joint target during execution.",
    )
    parser.add_argument("--max-steps", type=int, default=4, help="Total chunk steps to execute.")
    parser.add_argument("--max-delta", type=float, default=0.02, help="Rad clamp per joint per step.")
    parser.add_argument("--max-abs-action", type=float, default=0.3, help="Abort if raw action exceeds this.")
    parser.add_argument("--execute", action="store_true", help="Actually publish /target_joint.")
    parser.add_argument(
        "--allow-existing-publisher",
        action="store_true",
        help="Do not abort if /target_joint already has another publisher.",
    )
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument(
        "--log-timing",
        action="store_true",
        help="Print request interval, policy round-trip, and action pacing timings.",
    )
    return parser.parse_args()


def add_starvla_to_path(starvla_root: Optional[str]) -> bool:
    candidates = []
    if starvla_root:
        candidates.append(Path(starvla_root))

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "third_party" / "starVLA")

    candidates.extend(
        [
            Path("/home/hanyu/starVLA"),
            Path("/home/dase-hw101/franka_ws/third_party/starVLA"),
            Path("/home/ros/ros2_ws/third_party/starVLA"),
        ]
    )

    for path in candidates:
        if (path / "deployment" / "model_server" / "tools").exists():
            sys.path.insert(0, str(path))
            return True

    return False


def import_policy_client():
    try:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        return WebsocketClientPolicy
    except ImportError:
        return MinimalWebsocketClientPolicy


def _pack_array(obj):
    import msgpack

    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class MinimalWebsocketClientPolicy:
    """Small fallback compatible with StarVLA's websocket policy server."""

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None):
        import msgpack
        import websockets.sync.client

        self._msgpack = msgpack
        self._packer = msgpack.Packer(default=_pack_array)
        self._uri = f"ws://{host}" if port is None else f"ws://{host}:{port}"
        self._api_key = api_key
        self._ws = None
        self._server_metadata = None

        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=150,
            ping_interval=None,
            ping_timeout=60,
        )
        self._server_metadata = self._unpack(self._ws.recv())

    def _unpack(self, data):
        return self._msgpack.unpackb(data, object_hook=_unpack_array)

    def get_server_metadata(self):
        return self._server_metadata

    def predict_action(self, query_info):
        self._ws.send(self._packer.pack(query_info))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpack(response)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()


def raw_image_to_rgb(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        channels = 3
        image = data.reshape(msg.height, msg.width, channels)
        if msg.encoding == "bgr8":
            image = image[..., ::-1]
        return image.copy()
    if msg.encoding in ("rgba8", "bgra8"):
        channels = 4
        image = data.reshape(msg.height, msg.width, channels)[..., :3]
        if msg.encoding == "bgra8":
            image = image[..., ::-1]
        return image.copy()
    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


def compressed_image_to_rgb(msg: CompressedImage) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for --compressed-image") from exc

    arr = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode compressed image")
    return bgr[..., ::-1].copy()


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    pil = PILImage.fromarray(image.astype(np.uint8), mode="RGB")
    pil = pil.resize((size, size))
    return np.asarray(pil, dtype=np.uint8)


class FrankaObservationNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("starvla_franka_delta_joint_client")
        self.args = args
        self.latest_image: Optional[np.ndarray] = None
        self.latest_joint: Optional[np.ndarray] = None
        self.latest_gripper: float = 1.0
        self._last_image_time = 0.0
        self._last_joint_time = 0.0

        if args.compressed_image:
            self.create_subscription(
                CompressedImage,
                args.image_topic,
                self._compressed_image_callback,
                5,
            )
        else:
            self.create_subscription(Image, args.image_topic, self._image_callback, 5)

        self.create_subscription(JointState, args.joint_topic, self._joint_callback, 20)
        self.target_joint_pub = None
        self.gripper_pub = None

    def start_publishers(self) -> None:
        self.target_joint_pub = self.create_publisher(JointState, self.args.target_joint_topic, 1)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            self.args.gripper_command_topic,
            10,
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            self.latest_image = raw_image_to_rgb(msg)
            self._last_image_time = time.time()
        except Exception as exc:
            self.get_logger().warn(f"Skipping image: {exc}")

    def _compressed_image_callback(self, msg: CompressedImage) -> None:
        try:
            self.latest_image = compressed_image_to_rgb(msg)
            self._last_image_time = time.time()
        except Exception as exc:
            self.get_logger().warn(f"Skipping compressed image: {exc}")

    def _joint_callback(self, msg: JointState) -> None:
        values = np.full(7, np.nan, dtype=np.float64)
        for name, pos in zip(msg.name, msg.position):
            stripped = name.lstrip("/")
            if stripped in FR3_JOINT_NAMES:
                values[FR3_JOINT_NAMES.index(stripped)] = pos
        if np.isfinite(values).all():
            self.latest_joint = values
            self._last_joint_time = time.time()

    def wait_for_observation(self, timeout: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        start = time.time()
        while time.time() - start < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_image is not None and self.latest_joint is not None:
                return self.latest_image.copy(), self.latest_joint.copy()
        raise TimeoutError("Timed out waiting for image and joint observations")

    def target_joint_publishers(self) -> int:
        return self.count_publishers(self.args.target_joint_topic)

    def wait_for_target_joint_subscribers(self, timeout: float = 3.0) -> int:
        start = time.time()
        while time.time() - start < timeout and rclpy.ok():
            subscribers = self.count_subscribers(self.args.target_joint_topic)
            if subscribers > 0:
                return subscribers
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.count_subscribers(self.args.target_joint_topic)

    def publish_joint_target(self, q: np.ndarray) -> None:
        if self.target_joint_pub is None:
            raise RuntimeError("Publishers were not started")
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = FR3_JOINT_NAMES
        msg.position = q.astype(float).tolist()
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        self.target_joint_pub.publish(msg)

    def publish_gripper(self, value: float) -> None:
        if self.gripper_pub is None:
            return
        msg = Float64MultiArray()
        msg.data = [float(value)]
        self.gripper_pub.publish(msg)


def pick_unnorm_key(metadata: dict, requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    if metadata.get("default_unnorm_key") is not None:
        return metadata["default_unnorm_key"]
    keys = metadata.get("available_unnorm_keys", [])
    if len(keys) == 1:
        return keys[0]
    return None


def request_action_chunk(
    client,
    image: np.ndarray,
    args: argparse.Namespace,
    request_id: int,
) -> tuple[np.ndarray, dict]:
    metadata = client.get_server_metadata()
    unnorm_key = pick_unnorm_key(metadata, args.unnorm_key)
    if unnorm_key is None:
        raise RuntimeError(f"Could not choose unnorm_key from metadata: {metadata}")

    payload = {
        "examples": [
            {
                "image": [resize_rgb(image, args.image_size)],
                "lang": args.task,
            }
        ],
        "unnorm_key": unnorm_key,
    }
    request = {
        "type": "predict_action",
        "request_id": f"franka-delta-client-{request_id}",
        "payload": payload,
    }
    request_start = time.perf_counter()
    response = client.predict_action(request)
    client_roundtrip_s = time.perf_counter() - request_start
    if not response.get("ok", False):
        raise RuntimeError(f"Policy server returned error: {response}")
    actions = np.asarray(response["data"]["actions"][0], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise RuntimeError(f"Expected action chunk shape (T, 8+), got {actions.shape}")
    timing = dict(response.get("timing", {}))
    timing["client_policy_roundtrip_s"] = client_roundtrip_s
    return actions[:, :8], timing


def main() -> None:
    args = parse_args()
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.publish_rate <= 0:
        raise ValueError("--publish-rate must be positive")
    if args.max_delta <= 0:
        raise ValueError("--max-delta must be positive")

    add_starvla_to_path(args.starvla_root)
    WebsocketClientPolicy = import_policy_client()

    rclpy.init()
    node = FrankaObservationNode(args)

    try:
        image, current_joint = node.wait_for_observation(timeout=10.0)
        node.get_logger().info(f"Initial joints: {np.array2string(current_joint, precision=4)}")

        if args.execute:
            existing = node.target_joint_publishers()
            if existing > 0 and not args.allow_existing_publisher:
                raise RuntimeError(
                    f"{args.target_joint_topic} already has {existing} publisher(s). "
                    "Stop teleop/recording first, or pass --allow-existing-publisher."
                )
            node.start_publishers()
            target_subscribers = node.wait_for_target_joint_subscribers(timeout=3.0)
            if target_subscribers <= 0:
                raise RuntimeError(
                    f"{args.target_joint_topic} has no subscribers. "
                    "The joint controller is not ready to receive commands."
                )
            node.get_logger().info(
                f"{args.target_joint_topic} has {target_subscribers} subscriber(s)."
            )
            node.get_logger().warn(
                "EXECUTE mode enabled. Keep hand on E-stop. "
                f"max_delta={args.max_delta}, rate={args.rate}, max_steps={args.max_steps}"
            )
        else:
            node.get_logger().warn("Dry-run mode. No /target_joint commands will be published.")

        client = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
        node.get_logger().info(f"Policy metadata: {client.get_server_metadata()}")

        target_joint = current_joint.copy()
        steps_done = 0
        request_id = 0
        last_request_start = None
        period = 1.0 / args.rate

        while rclpy.ok() and steps_done < args.max_steps:
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.latest_image is None:
                raise RuntimeError("No image available")

            request_start = time.perf_counter()
            request_interval_s = (
                None
                if last_request_start is None
                else request_start - last_request_start
            )
            last_request_start = request_start
            actions, timing = request_action_chunk(
                client,
                node.latest_image,
                args,
                request_id,
            )
            request_id += 1
            raw_abs_max = float(np.max(np.abs(actions[:, :7])))
            print("action_chunk_shape:", actions.shape)
            if args.log_timing:
                print(
                    "timing_request:",
                    {
                        "request_id": request_id - 1,
                        "request_interval_s": request_interval_s,
                        **timing,
                    },
                )
            print("joint_delta_min:", actions[:, :7].min(axis=0))
            print("joint_delta_max:", actions[:, :7].max(axis=0))
            print("joint_delta_abs_max:", raw_abs_max)
            print("gripper_values:", actions[:, 7])
            if raw_abs_max > args.max_abs_action:
                raise RuntimeError(
                    f"Abort: policy raw joint_delta_abs_max={raw_abs_max:.4f} "
                    f"> --max-abs-action={args.max_abs_action:.4f}"
                )

            for action in actions:
                if steps_done >= args.max_steps:
                    break
                step_start = time.perf_counter()
                delta = np.clip(action[:7], -args.max_delta, args.max_delta)
                target_joint = np.clip(target_joint + delta, FR3_LOWER, FR3_UPPER)
                gripper = 1.0 if float(action[7]) >= 0.5 else 0.0

                print(
                    f"step={steps_done:03d} "
                    f"delta={np.array2string(delta, precision=4)} "
                    f"target={np.array2string(target_joint, precision=4)} "
                    f"gripper={gripper:.1f} "
                    f"execute={args.execute}"
                )

                if args.execute:
                    node.publish_joint_target(target_joint)
                    node.publish_gripper(gripper)

                steps_done += 1
                end_time = time.time() + period
                next_publish_time = time.time() + (1.0 / args.publish_rate)
                held_publish_count = 0
                while rclpy.ok() and time.time() < end_time:
                    if args.execute and time.time() >= next_publish_time:
                        node.publish_joint_target(target_joint)
                        node.publish_gripper(gripper)
                        held_publish_count += 1
                        next_publish_time += 1.0 / args.publish_rate
                    rclpy.spin_once(node, timeout_sec=0.01)
                if args.log_timing:
                    print(
                        "timing_step:",
                        {
                            "step": steps_done - 1,
                            "action_period_target_s": period,
                            "action_period_actual_s": time.perf_counter() - step_start,
                            "held_target_publish_count": held_publish_count,
                            "held_target_publish_rate_hz": args.publish_rate
                            if args.execute
                            else 0.0,
                        },
                    )

        client.close()
        node.get_logger().info("Finished StarVLA Franka delta-joint client.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
