#!/usr/bin/env python3
"""Safe StarVLA delta end-pose client for Franka FR3.

This script runs on the ROS/Franka machine. It connects to a StarVLA websocket
policy server, builds an observation from the latest primary and wrist camera
frames, receives a
7D Cartesian action chunk:

    [dx, dy, dz, droll, dpitch, dyaw, gripper]

and, only when --execute is passed, integrates those deltas into absolute
PoseStamped commands on /target_pose for the cartesian_impedance_controller.

Default mode is dry-run and does not move the robot.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray


class BackgroundROSExecutor:
    """Continuously service ROS callbacks while policy inference blocks.

    The websocket policy request is synchronous and can take longer than one
    action period.  Running the ROS executor on dedicated threads prevents pose,
    gripper, camera, and held-command timer callbacks from starving during that
    request.
    """

    def __init__(self, node: Node, num_threads: int = 4) -> None:
        self._executor = MultiThreadedExecutor(num_threads=num_threads)
        self._executor.add_node(node)
        self._thread = threading.Thread(
            target=self._spin,
            name="starvla-ros-executor",
            daemon=True,
        )
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except BaseException as exc:  # Surface executor death to the safety loop.
            with self._error_lock:
                self._error = exc

    def start(self) -> None:
        self._thread.start()

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Background ROS executor stopped unexpectedly") from error
        if not self._thread.is_alive() and rclpy.ok():
            raise RuntimeError("Background ROS executor is not running")

    def stop(self) -> None:
        self._executor.shutdown(timeout_sec=2.0)
        self._thread.join(timeout=2.0)


@dataclass(frozen=True)
class TemporalEnsembleDiagnostics:
    """Summary of how many overlapping predictions contributed per action."""

    candidate_counts: tuple[int, ...]
    newest_weights: tuple[float, ...]
    oldest_weights: tuple[float, ...]


class TemporalActionEnsembler:
    """Align and combine overlapping action chunks by absolute execution step.

    A chunk predicted when ``steps_done == s`` describes absolute execution
    steps ``s, s + 1, ...``.  After the client executes part of that chunk and
    replans, an older future action and a newer first action can therefore refer
    to the same execution step.  This class averages only those aligned actions.

    Cartesian and rotation deltas use an exponentially recency-weighted mean.
    Gripper values use a weighted binary vote because their 0/1 values are
    commands, not physical widths.
    """

    def __init__(
        self,
        window: int,
        decay: float,
        gripper_threshold: float = 0.5,
    ) -> None:
        if window <= 0:
            raise ValueError("temporal ensemble window must be positive")
        if not np.isfinite(decay) or not 0.0 < decay <= 1.0:
            raise ValueError("temporal ensemble decay must be in (0, 1]")
        if not np.isfinite(gripper_threshold) or not 0.0 < gripper_threshold < 1.0:
            raise ValueError("temporal ensemble gripper threshold must be in (0, 1)")
        self.window = int(window)
        self.decay = float(decay)
        self.gripper_threshold = float(gripper_threshold)
        self._chunks: list[tuple[int, np.ndarray]] = []

    def add_and_ensemble(
        self,
        start_step: int,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, TemporalEnsembleDiagnostics]:
        """Store ``actions`` and return its temporally ensembled counterpart."""
        chunk = np.asarray(actions, dtype=np.float64)
        if start_step < 0:
            raise ValueError("temporal ensemble start_step must be non-negative")
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != 7:
            raise ValueError(
                "temporal ensemble expects a non-empty action chunk with shape (N, 7), "
                f"got {chunk.shape}"
            )
        if not np.all(np.isfinite(chunk)):
            raise ValueError("temporal ensemble received non-finite actions")

        self._chunks.append((int(start_step), chunk.copy()))
        self._chunks = self._chunks[-self.window :]

        output = chunk.copy()
        candidate_counts: list[int] = []
        newest_weights: list[float] = []
        oldest_weights: list[float] = []

        # Reversed order makes age=0 the newest prediction.
        newest_first = list(reversed(self._chunks))
        for offset in range(len(chunk)):
            absolute_step = start_step + offset
            candidates: list[np.ndarray] = []
            weights: list[float] = []
            for age, (candidate_start, candidate_chunk) in enumerate(newest_first):
                candidate_offset = absolute_step - candidate_start
                if 0 <= candidate_offset < len(candidate_chunk):
                    candidates.append(candidate_chunk[candidate_offset])
                    weights.append(self.decay**age)

            candidate_array = np.stack(candidates, axis=0)
            weight_array = np.asarray(weights, dtype=np.float64)
            output[offset, :6] = np.average(
                candidate_array[:, :6],
                axis=0,
                weights=weight_array,
            )
            open_votes = (candidate_array[:, 6] >= 0.5).astype(np.float64)
            open_score = float(np.average(open_votes, weights=weight_array))
            output[offset, 6] = 1.0 if open_score >= self.gripper_threshold else 0.0

            candidate_counts.append(len(candidates))
            newest_weights.append(float(weight_array[0]))
            oldest_weights.append(float(weight_array[-1]))

        return output, TemporalEnsembleDiagnostics(
            candidate_counts=tuple(candidate_counts),
            newest_weights=tuple(newest_weights),
            oldest_weights=tuple(oldest_weights),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=10093)
    parser.add_argument("--task", default="pick up the cube and place it on the box")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--starvla-root", default=None)
    parser.add_argument(
        "--primary-image-topic",
        "--image-topic",
        dest="primary_image_topic",
        default="/right/right_third_person_camera/color/image_raw",
    )
    parser.add_argument(
        "--wrist-image-topic",
        default="/right/right_wrist_camera/color/image_raw",
    )
    parser.add_argument("--compressed-image", action="store_true")
    parser.add_argument("--current-pose-topic", default="/current_pose")
    parser.add_argument("--target-pose-topic", default="/target_pose")
    parser.add_argument("--target-frame-id", default="base")
    parser.add_argument(
        "--gripper-command-topic",
        default="/gripper/gripper_position_controller/commands",
    )
    parser.add_argument(
        "--gripper-state-topic",
        default="/franka_gripper/joint_states",
        help="Franka finger JointState topic used for policy proprioception.",
    )
    parser.add_argument(
        "--gripper-max-width",
        type=float,
        default=0.08,
        help=(
            "Total two-finger width in meters when fully open. The policy state is "
            "computed as clip(1 - measured_width / max_width, 0, 1)."
        ),
    )
    parser.add_argument(
        "--empty-grasp-width",
        type=float,
        default=0.015,
        help=(
            "Abort after closing if the measured total finger width is below this "
            "value in meters. Successful training grasps were about 0.028-0.030m."
        ),
    )
    parser.add_argument(
        "--empty-grasp-check-delay",
        type=float,
        default=1.2,
        help=(
            "Minimum seconds to continue bounded policy motion while the fingers "
            "close before measured-width validation can begin."
        ),
    )
    parser.add_argument(
        "--grasp-close-width-timeout",
        type=float,
        default=3.0,
        help=(
            "Maximum seconds after a physical close command to wait for the "
            "measured finger width to enter the cube-compatible range. Once it "
            "enters, the stable-width duration is allowed to finish."
        ),
    )
    parser.add_argument(
        "--grasp-width-stable-duration",
        type=float,
        default=0.25,
        help=(
            "Seconds the measured finger width must remain in the accepted range "
            "before the grasp is allowed to enter the lift phase."
        ),
    )
    parser.add_argument(
        "--grasp-close-max-descent",
        type=float,
        default=0.014,
        help=(
            "Maximum accumulated target-pose descent while a new grasp is "
            "closing but not yet width-validated, in meters."
        ),
    )
    parser.add_argument(
        "--grasp-close-max-lift",
        type=float,
        default=0.018,
        help=(
            "Maximum accumulated target-pose lift while a new grasp is closing "
            "but not yet width-validated, in meters."
        ),
    )
    parser.add_argument(
        "--grasp-close-max-xy-drift",
        type=float,
        default=0.012,
        help=(
            "Maximum accumulated XY target drift from the first close command "
            "until measured-width validation, in meters."
        ),
    )
    parser.add_argument(
        "--grasp-width-min",
        type=float,
        default=0.022,
        help="Minimum accepted total finger width for a cube grasp, in meters.",
    )
    parser.add_argument(
        "--grasp-width-max",
        type=float,
        default=0.036,
        help="Maximum accepted total finger width for a cube grasp, in meters.",
    )
    parser.add_argument(
        "--grasp-min-lift",
        type=float,
        default=0.030,
        help="Required measured Z lift after closing before a grasp is confirmed.",
    )
    parser.add_argument(
        "--grasp-lift-timeout",
        type=float,
        default=5.0,
        help=(
            "Abort if the required grasp lift is not reached within this many "
            "seconds after measured-width validation succeeds."
        ),
    )
    parser.add_argument(
        "--release-open-width",
        type=float,
        default=0.060,
        help=(
            "Minimum measured total finger width required before a model-requested "
            "release is considered physically complete."
        ),
    )
    parser.add_argument(
        "--release-open-stable-duration",
        type=float,
        default=0.25,
        help=(
            "Seconds the measured width must remain above --release-open-width "
            "before reporting episode completion."
        ),
    )
    parser.add_argument(
        "--release-open-timeout",
        type=float,
        default=3.0,
        help=(
            "Maximum seconds to hold the model-requested open command while "
            "waiting for physical gripper feedback."
        ),
    )
    parser.add_argument(
        "--max-grasp-attempts",
        type=int,
        default=1,
        help="Maximum physical open-to-close attempts allowed in one deployment.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Action execution rate in Hz (fast real-Franka profile by default).",
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=40.0,
        help="Rate in Hz for repeatedly publishing the held target pose during execution.",
    )
    parser.add_argument("--max-steps", type=int, default=4, help="Total chunk steps to execute.")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=1,
        help=(
            "How many actions to execute from each predicted chunk before requesting "
            "a fresh image-conditioned chunk. Use 1 for closed-loop deployment."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-window",
        type=int,
        default=1,
        help=(
            "Number of overlapping policy chunks to combine after aligning them by "
            "absolute execution step. Set 1 to disable; use 3 for the first test."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.8,
        help=(
            "Recency decay for aligned chunks: newest weight is 1 and an age-k "
            "prediction has weight decay**k."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-gripper-threshold",
        type=float,
        default=0.5,
        help="Weighted open-vote threshold used by temporal ensembling.",
    )
    parser.add_argument(
        "--max-trans-delta",
        type=float,
        default=0.009,
        help="Norm clamp for translation delta in meters per action step.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.25,
        help="Multiply policy translation deltas before applying --max-trans-delta.",
    )
    parser.add_argument(
        "--max-rot-delta",
        type=float,
        default=0.003,
        help="Norm clamp for Euler RPY delta in radians per action step.",
    )
    parser.add_argument(
        "--max-abs-trans-action",
        type=float,
        default=0.15,
        help="Abort if raw policy translation delta norm exceeds this value.",
    )
    parser.add_argument(
        "--max-abs-rot-action",
        type=float,
        default=0.25,
        help="Abort if raw policy rotation delta norm exceeds this value.",
    )
    parser.add_argument("--min-x", type=float, default=0.28)
    parser.add_argument("--max-x", type=float, default=0.57)
    parser.add_argument("--min-y", type=float, default=-0.23)
    parser.add_argument("--max-y", type=float, default=0.10)
    parser.add_argument("--min-z", type=float, default=0.03)
    parser.add_argument("--max-z", type=float, default=0.70)
    parser.add_argument("--execute", action="store_true", help="Actually publish /target_pose.")
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument(
        "--disable-gripper-close-latch",
        action="store_true",
        help=(
            "Disable the physical close latch. By default, once close is accepted, "
            "open requests are suppressed until grasp width and lift validation finish."
        ),
    )
    parser.add_argument(
        "--synchronized-close-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When a reliable close candidate first appears, rebase the Cartesian "
            "target to the latest measured pose and hold that pose while close "
            "confirmations and physical finger-width validation complete. This "
            "generic latency compensation is enabled by default; use "
            "--no-synchronized-close-hold only for a raw-policy ablation."
        ),
    )
    parser.add_argument(
        "--gripper-switch-confirmations",
        type=int,
        default=3,
        help=(
            "Number of consecutive policy requests that must agree before changing "
            "the gripper state. The safety-filtered default is 3. Set to 1 only "
            "for an explicitly unfiltered comparison run."
        ),
    )
    parser.add_argument(
        "--gripper-chunk-consensus",
        type=float,
        default=0.75,
        help=(
            "Fraction of a predicted action chunk that must agree on open/close when "
            "gripper debouncing is enabled."
        ),
    )
    parser.add_argument(
        "--initial-gripper-state",
        choices=("open", "closed"),
        default="open",
        help="Initial debounced gripper state; use closed when resuming while holding an object.",
    )
    parser.add_argument(
        "--allow-existing-publisher",
        action="store_true",
        help="Do not abort if /target_pose already has another publisher.",
    )
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument(
        "--max-observation-age",
        type=float,
        default=1.0,
        help="Abort if either camera frame or the current pose is older than this many seconds.",
    )
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
        image = data.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            image = image[..., ::-1]
        return image.copy()
    if msg.encoding in ("rgba8", "bgra8"):
        image = data.reshape(msg.height, msg.width, 4)[..., :3]
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
    """Match the square center-crop used by CRISP while recording.

    CRISP's Camera._resize_with_aspect_ratio preserves the source aspect ratio
    and center-crops the excess dimension before saving each 256x256 training
    frame.  Stretching a live 16:9 or 4:3 frame directly to a square changes
    object geometry and exposes image regions that were absent during
    training.  A square center-crop followed by the model resize has the same
    field of view as the recorder pipeline.
    """
    image = image.astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Expected an RGB HxWx3 image, got {image.shape}")

    height, width = image.shape[:2]
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    image = image[top : top + crop_size, left : left + crop_size]

    pil = PILImage.fromarray(image, mode="RGB")
    # Pillow < 9.1 exposes BILINEAR directly on Image rather than through
    # Image.Resampling (the Franka ROS container currently uses that API).
    resampling = getattr(PILImage, "Resampling", PILImage)
    pil = pil.resize((size, size), resample=resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def pose_to_position_rotation(msg: PoseStamped) -> tuple[np.ndarray, Rotation]:
    position = np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        dtype=np.float64,
    )
    quat = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    if np.linalg.norm(quat) < 1e-6:
        raise RuntimeError("Current pose has an invalid near-zero quaternion")
    return position, Rotation.from_quat(quat)


def clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return np.zeros_like(vector)
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)


def compute_action_hold_duration(
    action_period: float,
    inference_elapsed: float,
    replan_after_action: bool,
) -> float:
    """Return how long to hold an action while keeping inference inside its period.

    Actions within one predicted chunk are held for the full configured period.
    Before the next synchronous policy request, the most recently measured
    inference time is deducted from the hold. With the normal
    ``execution_horizon=1`` setting this makes request-start and action-update
    intervals approximately ``1 / rate`` instead of ``inference + 1 / rate``.
    """
    if not np.isfinite(action_period) or action_period <= 0.0:
        raise ValueError(f"action period must be finite and positive: {action_period}")
    if not np.isfinite(inference_elapsed) or inference_elapsed < 0.0:
        raise ValueError(
            f"inference elapsed time must be finite and non-negative: {inference_elapsed}"
        )
    if not replan_after_action:
        return float(action_period)
    return float(max(0.0, action_period - inference_elapsed))


def measured_gripper_state(msg: JointState, max_width: float) -> tuple[float, float]:
    """Return total finger width and training-compatible closed amount.

    CRISP recorded ``observation.state.gripper`` as ``1 - gripper.value``:
    fully open is 0, while increasingly closed values approach 1. Franka's
    native JointState reports one position per finger, so the measured total
    opening is the sum of the two finger joint positions.
    """
    if max_width <= 0.0 or not np.isfinite(max_width):
        raise ValueError(f"gripper max width must be finite and positive, got {max_width}")
    if len(msg.position) < 2:
        raise RuntimeError(
            f"Expected two Franka finger positions, got {len(msg.position)}"
        )

    finger_indices = [
        index for index, name in enumerate(msg.name) if "finger_joint" in name
    ]
    if len(finger_indices) >= 2:
        positions = np.asarray(
            [msg.position[finger_indices[0]], msg.position[finger_indices[1]]],
            dtype=np.float64,
        )
    else:
        positions = np.asarray(msg.position[:2], dtype=np.float64)
    if not np.all(np.isfinite(positions)):
        raise RuntimeError(f"Non-finite Franka finger positions: {positions}")

    total_width = float(np.sum(positions))
    closed_amount = float(np.clip(1.0 - total_width / max_width, 0.0, 1.0))
    return total_width, closed_amount


def is_empty_grasp(total_width: float, empty_width_threshold: float) -> bool:
    """Classify a post-close finger width using the configured strict threshold."""
    if not np.isfinite(total_width) or total_width < 0.0:
        raise ValueError(f"Measured gripper width must be finite and non-negative: {total_width}")
    if not np.isfinite(empty_width_threshold) or empty_width_threshold <= 0.0:
        raise ValueError(
            "Empty-grasp width threshold must be finite and positive: "
            f"{empty_width_threshold}"
        )
    return bool(total_width < empty_width_threshold)


def classify_grasp_width(
    total_width: float,
    empty_width_threshold: float,
    valid_width_min: float,
    valid_width_max: float,
) -> str:
    """Classify measured total finger width after a physical close command."""
    values = np.asarray(
        [total_width, empty_width_threshold, valid_width_min, valid_width_max],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Grasp width values must be finite: {values}")
    if total_width < 0.0:
        raise ValueError(f"Measured gripper width must be non-negative: {total_width}")
    if not 0.0 < empty_width_threshold < valid_width_min < valid_width_max:
        raise ValueError(
            "Expected 0 < empty threshold < valid minimum < valid maximum, got "
            f"{empty_width_threshold}, {valid_width_min}, {valid_width_max}"
        )
    if total_width < empty_width_threshold:
        return "empty"
    if total_width < valid_width_min:
        return "too_narrow"
    if total_width > valid_width_max:
        return "too_wide"
    return "valid"


def apply_gripper_close_latch(
    requested_gripper: float,
    latch_active: bool,
    latch_enabled: bool,
) -> tuple[float, bool]:
    """Return the effective binary command and whether an early open was held.

    The latch never decides when to grasp.  It becomes relevant only after the
    filtered policy has already caused a physical close transition.  While
    width/lift validation is active it makes that actuator transaction atomic:
    transient model open requests are recorded but the physical command stays
    closed.  Invalid width, stale feedback, or timeout still aborts the episode.
    """
    if not np.isfinite(requested_gripper):
        raise ValueError(f"Non-finite requested gripper command: {requested_gripper}")
    binary_request = 1.0 if requested_gripper >= 0.5 else 0.0
    if latch_enabled and latch_active and binary_request >= 0.5:
        return 0.0, True
    return binary_request, False


def consensus_gripper_request(
    actions: np.ndarray,
    required_fraction: float,
) -> Optional[float]:
    """Return binary chunk intent, or ``None`` when neither side has consensus."""
    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] < 7:
        raise ValueError(f"Expected a non-empty (N, 7+) action chunk, got {chunk.shape}")
    if not np.all(np.isfinite(chunk[:, 6])):
        raise ValueError("Non-finite gripper values in action chunk")
    if not np.isfinite(required_fraction) or not 0.5 < required_fraction <= 1.0:
        raise ValueError("required gripper consensus must be in (0.5, 1]")
    close_fraction = float(np.mean(chunk[:, 6] < 0.5))
    if close_fraction >= required_fraction:
        return 0.0
    if 1.0 - close_fraction >= required_fraction:
        return 1.0
    return None


def synchronized_close_hold_transition(
    enabled: bool,
    hold_active: bool,
    physical_gripper_enabled: bool,
    last_published_gripper: float,
    grasp_validation_active: bool,
    close_candidate: bool,
) -> str:
    """Return the deterministic close-hold state-machine transition.

    The hold is armed only by a policy close candidate; it never uses an object
    pose or a task-specific workspace region.  Once the physical close begins,
    the hold remains active until measured-width validation explicitly releases
    it in the main control loop.
    """
    if not enabled or not physical_gripper_enabled:
        return "inactive"
    if not np.isfinite(last_published_gripper):
        raise ValueError(
            "last published gripper command must be finite for synchronized hold"
        )
    physical_close_started = bool(
        last_published_gripper < 0.5 or grasp_validation_active
    )
    if hold_active:
        if physical_close_started or close_candidate:
            return "keep"
        return "cancel"
    if not physical_close_started and close_candidate:
        return "activate"
    return "inactive"


def enforce_physical_grasp_transition(
    previous_gripper: float,
    next_gripper: float,
    grasp_attempt_count: int,
    max_grasp_attempts: int,
    grasp_in_progress: bool,
    required_lift: float,
) -> tuple[bool, bool]:
    """Validate one effective physical transition and return (close, open).

    With the close latch enabled, an in-progress open should already have been
    converted to close by :func:`apply_gripper_close_latch`; this remaining
    check is an invariant.  With the latch explicitly disabled it preserves the
    previous strict-failure behavior for controlled ablations.
    """
    close_transition = bool(previous_gripper >= 0.5 and next_gripper < 0.5)
    open_transition = bool(previous_gripper < 0.5 and next_gripper >= 0.5)
    if close_transition and grasp_attempt_count >= max_grasp_attempts:
        raise RuntimeError(
            "Abort: policy requested another grasp after the allowed "
            f"{max_grasp_attempts} physical attempt(s)"
        )
    if open_transition and grasp_in_progress:
        raise RuntimeError(
            "EPISODE FAILURE: effective gripper command requested opening before "
            "the grasp was width-validated and completed the required "
            f"{required_lift:.3f}m lift; enable the close latch unless this is an "
            "intentional raw-policy ablation"
        )
    return close_transition, open_transition


def constrain_closing_target(
    candidate_position: np.ndarray,
    close_target_position: np.ndarray,
    max_xy_drift: float,
    max_descent: float,
    max_lift: float,
) -> tuple[np.ndarray, bool]:
    """Bound target motion while the fingers are closing around the object.

    The demonstrations continue moving briefly after the first close command.
    This preserves that behavior without allowing an unverified grasp to travel
    arbitrarily far from the grasp site.
    """
    candidate = np.asarray(candidate_position, dtype=np.float64)
    origin = np.asarray(close_target_position, dtype=np.float64)
    if candidate.shape != (3,) or origin.shape != (3,):
        raise ValueError("candidate_position and close_target_position must have shape (3,)")
    values = np.concatenate(
        [candidate, origin, [max_xy_drift, max_descent, max_lift]]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite closing-motion constraint input: {values}")
    if max_xy_drift <= 0.0 or max_descent <= 0.0 or max_lift <= 0.0:
        raise ValueError("Closing-motion bounds must all be positive")

    constrained = candidate.copy()
    xy_offset = constrained[:2] - origin[:2]
    xy_norm = float(np.linalg.norm(xy_offset))
    if xy_norm > max_xy_drift:
        constrained[:2] = origin[:2] + xy_offset * (max_xy_drift / xy_norm)
    constrained[2] = float(
        np.clip(constrained[2], origin[2] - max_descent, origin[2] + max_lift)
    )
    was_constrained = not np.allclose(constrained, candidate, rtol=0.0, atol=1e-12)
    return constrained, was_constrained


class FrankaCartesianObservationNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("starvla_franka_delta_pose_client")
        self.args = args
        self._observation_lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._publish_lock = threading.Lock()
        self._image_callback_groups = {
            "primary": MutuallyExclusiveCallbackGroup(),
            "wrist": MutuallyExclusiveCallbackGroup(),
        }
        self._feedback_callback_group = MutuallyExclusiveCallbackGroup()
        self._command_callback_group = MutuallyExclusiveCallbackGroup()
        self.latest_images: dict[str, Optional[np.ndarray]] = {
            "primary": None,
            "wrist": None,
        }
        self.latest_position: Optional[np.ndarray] = None
        self.latest_rotation: Optional[Rotation] = None
        self.latest_gripper_width: Optional[float] = None
        self.latest_gripper_closed_amount: Optional[float] = None
        self.latest_pose_frame_id = args.target_frame_id
        self._last_image_times = {"primary": 0.0, "wrist": 0.0}
        self._last_pose_time = 0.0
        self._last_gripper_time = 0.0
        self._held_position: Optional[np.ndarray] = None
        self._held_quaternion: Optional[np.ndarray] = None
        self._held_gripper: Optional[float] = None
        self._held_command_active = False
        self._held_command_generation = 0
        self._command_publish_count = 0
        self._last_command_publish_time = 0.0
        self._command_timer = None

        camera_topics = {
            "primary": args.primary_image_topic,
            "wrist": args.wrist_image_topic,
        }
        latest_image_qos = QoSProfile(depth=1)
        latest_image_qos.reliability = qos_profile_sensor_data.reliability
        latest_image_qos.durability = qos_profile_sensor_data.durability
        for camera_name, topic in camera_topics.items():
            if args.compressed_image:
                self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda msg, name=camera_name: self._compressed_image_callback(msg, name),
                    latest_image_qos,
                    callback_group=self._image_callback_groups[camera_name],
                )
            else:
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, name=camera_name: self._image_callback(msg, name),
                    latest_image_qos,
                    callback_group=self._image_callback_groups[camera_name],
                )

        # Match the feedback publishers' KEEP_LAST(1), RELIABLE QoS so a slow
        # policy request can never leave a queue of historical robot states for
        # the client to consume later.
        latest_feedback_qos = QoSProfile(depth=1)
        self.create_subscription(
            PoseStamped,
            args.current_pose_topic,
            self._current_pose_callback,
            latest_feedback_qos,
            callback_group=self._feedback_callback_group,
        )
        self.create_subscription(
            JointState,
            args.gripper_state_topic,
            self._gripper_state_callback,
            latest_feedback_qos,
            callback_group=self._feedback_callback_group,
        )
        self.target_pose_pub = None
        self.gripper_pub = None

    def start_publishers(self) -> None:
        self.target_pose_pub = self.create_publisher(PoseStamped, self.args.target_pose_topic, 1)
        if not self.args.disable_gripper:
            self.gripper_pub = self.create_publisher(
                Float64MultiArray,
                self.args.gripper_command_topic,
                10,
            )
        self._command_timer = self.create_timer(
            1.0 / self.args.publish_rate,
            self._publish_held_command,
            callback_group=self._command_callback_group,
        )

    def _image_callback(self, msg: Image, camera_name: str) -> None:
        try:
            image = raw_image_to_rgb(msg)
            with self._observation_lock:
                self.latest_images[camera_name] = image
                self._last_image_times[camera_name] = time.monotonic()
        except Exception as exc:
            self.get_logger().warn(f"Skipping {camera_name} image: {exc}")

    def _compressed_image_callback(self, msg: CompressedImage, camera_name: str) -> None:
        try:
            image = compressed_image_to_rgb(msg)
            with self._observation_lock:
                self.latest_images[camera_name] = image
                self._last_image_times[camera_name] = time.monotonic()
        except Exception as exc:
            self.get_logger().warn(f"Skipping compressed {camera_name} image: {exc}")

    def _current_pose_callback(self, msg: PoseStamped) -> None:
        try:
            position, rotation = pose_to_position_rotation(msg)
        except Exception as exc:
            self.get_logger().warn(f"Skipping current pose: {exc}")
            return
        with self._observation_lock:
            self.latest_position = position
            self.latest_rotation = rotation
            if msg.header.frame_id:
                self.latest_pose_frame_id = msg.header.frame_id
            self._last_pose_time = time.monotonic()

    def _gripper_state_callback(self, msg: JointState) -> None:
        try:
            total_width, closed_amount = measured_gripper_state(
                msg, self.args.gripper_max_width
            )
        except Exception as exc:
            self.get_logger().warn(f"Skipping Franka gripper state: {exc}")
            return
        with self._observation_lock:
            self.latest_gripper_width = total_width
            self.latest_gripper_closed_amount = closed_amount
            self._last_gripper_time = time.monotonic()

    def wait_for_observation(
        self, timeout: float = 10.0
    ) -> tuple[list[np.ndarray], np.ndarray, Rotation]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            with self._observation_lock:
                ready = (
                    all(image is not None for image in self.latest_images.values())
                    and self.latest_position is not None
                    and self.latest_rotation is not None
                    and self.latest_gripper_width is not None
                    and self.latest_gripper_closed_amount is not None
                )
            if ready:
                position, rotation = self.copy_current_pose()
                return self.copy_images(), position, rotation
            time.sleep(0.01)
        with self._observation_lock:
            missing = [
                name for name, image in self.latest_images.items() if image is None
            ]
            if self.latest_position is None or self.latest_rotation is None:
                missing.append("current_pose")
            if (
                self.latest_gripper_width is None
                or self.latest_gripper_closed_amount is None
            ):
                missing.append("gripper_state")
        raise TimeoutError(f"Timed out waiting for observations: {missing}")

    def copy_images(self) -> list[np.ndarray]:
        """Return images in the same [primary, wrist] order used for training."""
        with self._observation_lock:
            images = [self.latest_images["primary"], self.latest_images["wrist"]]
            if any(image is None for image in images):
                raise RuntimeError("Both primary and wrist camera images are required")
            return [image.copy() for image in images]

    def copy_current_pose(self) -> tuple[np.ndarray, Rotation]:
        with self._observation_lock:
            if self.latest_position is None or self.latest_rotation is None:
                raise RuntimeError("Current pose is unavailable")
            return self.latest_position.copy(), Rotation.from_quat(
                self.latest_rotation.as_quat().copy()
            )

    def copy_gripper_state(self) -> tuple[float, float]:
        with self._observation_lock:
            if (
                self.latest_gripper_width is None
                or self.latest_gripper_closed_amount is None
            ):
                raise RuntimeError("Franka gripper state is unavailable")
            return self.latest_gripper_width, self.latest_gripper_closed_amount

    def observation_ages(self) -> dict[str, float]:
        now = time.monotonic()
        with self._observation_lock:
            return {
                "primary_image": now - self._last_image_times["primary"],
                "wrist_image": now - self._last_image_times["wrist"],
                "current_pose": now - self._last_pose_time,
                "gripper_state": now - self._last_gripper_time,
            }

    def refresh_stale_observations(
        self,
        max_age: float,
        timeout: float = 0.5,
    ) -> dict[str, float]:
        """Wait briefly for the background executor to refresh stale streams."""
        deadline = time.monotonic() + timeout
        ages = self.observation_ages()
        while rclpy.ok() and any(age > max_age for age in ages.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.01, remaining))
            ages = self.observation_ages()
        return ages

    def target_pose_publishers(self) -> int:
        return self.count_publishers(self.args.target_pose_topic)

    def wait_for_target_pose_subscribers(self, timeout: float = 3.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            subscribers = self.count_subscribers(self.args.target_pose_topic)
            if subscribers > 0:
                return subscribers
            time.sleep(0.05)
        return self.count_subscribers(self.args.target_pose_topic)

    def set_held_command(
        self,
        position: np.ndarray,
        rotation: Rotation,
        gripper: float,
        publish_immediately: bool = True,
    ) -> None:
        """Atomically replace the command continuously published by the timer."""
        if self.target_pose_pub is None or self._command_timer is None:
            raise RuntimeError("Publishers were not started")
        position_array = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(rotation.as_quat(), dtype=np.float64)
        if position_array.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Held pose must contain a 3D position and quaternion")
        if not np.all(np.isfinite(position_array)) or not np.all(
            np.isfinite(quaternion)
        ):
            raise ValueError("Held pose must be finite")
        if not np.isfinite(gripper):
            raise ValueError("Held gripper command must be finite")
        with self._command_lock:
            self._held_position = position_array.copy()
            self._held_quaternion = quaternion.copy()
            self._held_gripper = float(gripper)
            self._held_command_active = True
            self._held_command_generation += 1
        if publish_immediately:
            self._publish_held_command()

    def stop_held_command(self) -> None:
        """Stop the heartbeat without publishing any replacement robot command."""
        with self._command_lock:
            self._held_command_active = False
            self._held_command_generation += 1

    def command_publish_count(self) -> int:
        with self._command_lock:
            return self._command_publish_count

    def command_publish_age(self) -> float:
        with self._command_lock:
            last_publish = self._last_command_publish_time
        if last_publish <= 0.0:
            return float("inf")
        return time.monotonic() - last_publish

    def _publish_held_command(self) -> None:
        """Timer callback that keeps the controller alive during inference."""
        with self._command_lock:
            if not self._held_command_active:
                return
            if (
                self._held_position is None
                or self._held_quaternion is None
                or self._held_gripper is None
            ):
                return
            position = self._held_position.copy()
            quaternion = self._held_quaternion.copy()
            gripper = self._held_gripper
            generation = self._held_command_generation
        with self._observation_lock:
            frame_id = self.args.target_frame_id or self.latest_pose_frame_id

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [float(gripper)]

        # A timer callback and an immediate action update may overlap in the
        # multi-threaded executor.  Serialize each pose/gripper pair so the
        # physical controller never receives an interleaved command pair.
        with self._publish_lock:
            with self._command_lock:
                if (
                    not self._held_command_active
                    or generation != self._held_command_generation
                    or self.target_pose_pub is None
                ):
                    return
                self.target_pose_pub.publish(pose_msg)
                if self.gripper_pub is not None:
                    self.gripper_pub.publish(gripper_msg)
                self._command_publish_count += 1
                self._last_command_publish_time = time.monotonic()


def wait_for_physical_release(
    node: FrankaCartesianObservationNode,
    ros_executor: BackgroundROSExecutor,
    open_width: float,
    stable_duration: float,
    timeout: float,
    max_feedback_age: float,
) -> float:
    """Keep the model-requested open command active until feedback confirms it.

    The held-command timer is already publishing the frozen Cartesian target and
    gripper-open value.  This function only observes physical finger width; it
    never initiates a release on its own.
    """
    start = time.monotonic()
    deadline = start + timeout
    open_width_since: Optional[float] = None
    next_progress_log = start

    while rclpy.ok():
        ros_executor.raise_if_failed()
        now = time.monotonic()
        feedback_age = node.observation_ages()["gripper_state"]
        if feedback_age > max_feedback_age:
            raise RuntimeError(
                "Abort: gripper feedback became stale while confirming physical "
                f"release: age={feedback_age:.3f}s"
            )

        measured_width, measured_closed_amount = node.copy_gripper_state()
        if measured_width >= open_width:
            if open_width_since is None:
                open_width_since = now
            width_stable_for = now - open_width_since
            if width_stable_for >= stable_duration:
                return measured_width
        else:
            open_width_since = None
            width_stable_for = 0.0

        if now >= next_progress_log:
            print(
                "release_open_progress:",
                {
                    "elapsed_s": now - start,
                    "total_width_m": measured_width,
                    "state_gripper_closed": measured_closed_amount,
                    "required_open_width_m": open_width,
                    "width_stable_s": width_stable_for,
                    "required_stable_s": stable_duration,
                    "feedback_age_s": feedback_age,
                    "command_publish_count": node.command_publish_count(),
                },
            )
            next_progress_log = now + 0.25

        if now >= deadline:
            raise RuntimeError(
                "Abort: model requested release but the physical gripper did not "
                "open before timeout: "
                f"measured={measured_width:.5f}m < {open_width:.5f}m, "
                f"timeout={timeout:.2f}s"
            )
        time.sleep(min(0.01, deadline - now))

    raise RuntimeError("Abort: ROS stopped while confirming physical gripper release")


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
    images: list[np.ndarray],
    raw_state: np.ndarray,
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
                "image": [resize_rgb(image, args.image_size) for image in images],
                "lang": args.task,
                "state": np.asarray(raw_state, dtype=np.float32).reshape(1, -1),
            }
        ],
        "unnorm_key": unnorm_key,
        "normalize_state": True,
    }
    request = {
        "type": "predict_action",
        "request_id": f"franka-delta-pose-client-{request_id}",
        "payload": payload,
    }
    request_start = time.perf_counter()
    response = client.predict_action(request)
    client_roundtrip_s = time.perf_counter() - request_start
    if not response.get("ok", False):
        raise RuntimeError(f"Policy server returned error: {response}")

    data = response.get("data", {})
    if "actions" not in data:
        raise RuntimeError(
            "Policy response does not contain server-side unnormalized `actions`. "
            f"Available keys: {list(data.keys())}"
        )
    actions = np.asarray(data["actions"], dtype=np.float64)
    if actions.ndim == 3:
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"Expected action chunk shape (T, 7+), got {actions.shape}")
    timing = dict(response.get("timing", {}))
    timing["client_policy_roundtrip_s"] = client_roundtrip_s
    return actions[:, :7], timing


def validate_metadata(metadata: dict) -> None:
    action_keys = metadata.get("action_keys")
    if not action_keys:
        print("WARNING: server metadata does not include action_keys; verify checkpoint manually.")
        return
    joined = ",".join(action_keys)
    if "delta_joints" in joined or "target_joints" in joined:
        raise RuntimeError(
            f"Server action_keys look like a joint-space model, not Cartesian delta pose: {action_keys}"
        )
    if "delta_eef_position" not in joined and "action.x" not in joined:
        print(f"WARNING: unexpected action_keys for delta-pose deployment: {action_keys}")


def main() -> None:
    args = parse_args()
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.publish_rate <= 0:
        raise ValueError("--publish-rate must be positive")
    if args.execution_horizon <= 0:
        raise ValueError("--execution-horizon must be positive")
    if args.temporal_ensemble_window <= 0:
        raise ValueError("--temporal-ensemble-window must be positive")
    if (
        not np.isfinite(args.temporal_ensemble_decay)
        or not 0.0 < args.temporal_ensemble_decay <= 1.0
    ):
        raise ValueError("--temporal-ensemble-decay must be in (0, 1]")
    if (
        not np.isfinite(args.temporal_ensemble_gripper_threshold)
        or not 0.0 < args.temporal_ensemble_gripper_threshold < 1.0
    ):
        raise ValueError(
            "--temporal-ensemble-gripper-threshold must be in (0, 1)"
        )
    if args.max_trans_delta < 0 or args.max_rot_delta < 0:
        raise ValueError("--max-trans-delta and --max-rot-delta must be non-negative")
    if args.translation_scale <= 0:
        raise ValueError("--translation-scale must be positive")
    if args.gripper_max_width <= 0 or not np.isfinite(args.gripper_max_width):
        raise ValueError("--gripper-max-width must be finite and positive")
    if not (
        np.isfinite(args.empty_grasp_width)
        and np.isfinite(args.grasp_width_min)
        and np.isfinite(args.grasp_width_max)
        and 0.0
        < args.empty_grasp_width
        < args.grasp_width_min
        < args.grasp_width_max
        < args.gripper_max_width
    ):
        raise ValueError(
            "Expected 0 < --empty-grasp-width < --grasp-width-min < "
            "--grasp-width-max < --gripper-max-width"
        )
    if (
        not np.isfinite(args.empty_grasp_check_delay)
        or args.empty_grasp_check_delay <= 0.0
    ):
        raise ValueError("--empty-grasp-check-delay must be finite and positive")
    if (
        not np.isfinite(args.grasp_close_width_timeout)
        or args.grasp_close_width_timeout <= args.empty_grasp_check_delay
    ):
        raise ValueError(
            "--grasp-close-width-timeout must be finite and greater than "
            "--empty-grasp-check-delay"
        )
    if (
        not np.isfinite(args.grasp_width_stable_duration)
        or args.grasp_width_stable_duration <= 0.0
    ):
        raise ValueError(
            "--grasp-width-stable-duration must be finite and positive"
        )
    closing_motion_limits = {
        "--grasp-close-max-descent": args.grasp_close_max_descent,
        "--grasp-close-max-lift": args.grasp_close_max_lift,
        "--grasp-close-max-xy-drift": args.grasp_close_max_xy_drift,
    }
    for name, value in closing_motion_limits.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.grasp_min_lift) or args.grasp_min_lift <= 0.0:
        raise ValueError("--grasp-min-lift must be finite and positive")
    if (
        not np.isfinite(args.grasp_lift_timeout)
        or args.grasp_lift_timeout <= 0.0
    ):
        raise ValueError("--grasp-lift-timeout must be finite and positive")
    if not (
        np.isfinite(args.release_open_width)
        and args.grasp_width_max
        < args.release_open_width
        <= args.gripper_max_width
    ):
        raise ValueError(
            "Expected --grasp-width-max < --release-open-width <= "
            "--gripper-max-width"
        )
    if (
        not np.isfinite(args.release_open_stable_duration)
        or args.release_open_stable_duration <= 0.0
    ):
        raise ValueError(
            "--release-open-stable-duration must be finite and positive"
        )
    if (
        not np.isfinite(args.release_open_timeout)
        or args.release_open_timeout <= args.release_open_stable_duration
    ):
        raise ValueError(
            "--release-open-timeout must be finite and greater than "
            "--release-open-stable-duration"
        )
    if args.max_grasp_attempts <= 0:
        raise ValueError("--max-grasp-attempts must be positive")
    if args.max_observation_age <= 0:
        raise ValueError("--max-observation-age must be positive")
    if args.gripper_switch_confirmations <= 0:
        raise ValueError("--gripper-switch-confirmations must be positive")
    if not 0.5 < args.gripper_chunk_consensus <= 1.0:
        raise ValueError("--gripper-chunk-consensus must be in (0.5, 1.0]")
    if not (args.min_x < args.max_x and args.min_y < args.max_y and args.min_z < args.max_z):
        raise ValueError("Each workspace minimum must be smaller than its maximum")

    add_starvla_to_path(args.starvla_root)
    WebsocketClientPolicy = import_policy_client()

    rclpy.init()
    node = FrankaCartesianObservationNode(args)
    ros_executor = BackgroundROSExecutor(node)
    ros_executor.start()
    client = None

    try:
        images, current_position, current_rotation = node.wait_for_observation(timeout=10.0)
        node.get_logger().info(
            "Camera shapes [primary, wrist]: "
            + str([tuple(image.shape) for image in images])
        )
        node.get_logger().info(f"Initial position: {np.array2string(current_position, precision=4)}")
        node.get_logger().info(
            f"Initial rpy xyz: {np.array2string(current_rotation.as_euler('xyz'), precision=4)}"
        )
        initial_gripper_width, initial_gripper_closed_amount = node.copy_gripper_state()
        node.get_logger().info(
            "Initial measured gripper: "
            f"total_width={initial_gripper_width:.5f}m, "
            f"state.gripper={initial_gripper_closed_amount:.5f}"
        )

        if args.execute:
            existing = node.target_pose_publishers()
            if existing > 0 and not args.allow_existing_publisher:
                raise RuntimeError(
                    f"{args.target_pose_topic} already has {existing} publisher(s). "
                    "Stop teleop/recording first, or pass --allow-existing-publisher."
                )
            node.start_publishers()
            target_subscribers = node.wait_for_target_pose_subscribers(timeout=3.0)
            if target_subscribers <= 0:
                raise RuntimeError(
                    f"{args.target_pose_topic} has no subscribers. "
                    "The Cartesian controller is not ready to receive commands."
                )
            node.get_logger().info(
                f"{args.target_pose_topic} has {target_subscribers} subscriber(s)."
            )
            node.get_logger().warn(
                "EXECUTE mode enabled. Keep hand on E-stop. "
                f"translation_scale={args.translation_scale}, "
                f"max_trans_delta={args.max_trans_delta}, max_rot_delta={args.max_rot_delta}, "
                f"rate={args.rate}, max_steps={args.max_steps}"
            )
        else:
            node.get_logger().warn("Dry-run mode. No /target_pose commands will be published.")

        client = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
        metadata = client.get_server_metadata()
        node.get_logger().info(f"Policy metadata: {metadata}")
        validate_metadata(metadata)
        if metadata.get("state_keys") and not metadata.get("supports_raw_state_normalization", False):
            raise RuntimeError(
                "Policy requires proprioceptive state, but the server does not advertise "
                "raw-state normalization support. Restart the patched policy server."
            )

        target_position = current_position.copy()
        target_rotation = current_rotation
        steps_done = 0
        request_id = 0
        last_request_start = None
        period = 1.0 / args.rate
        gripper_state = 1.0 if args.initial_gripper_state == "open" else 0.0
        last_published_gripper = gripper_state
        pending_gripper_state = None
        pending_gripper_count = 0
        grasp_attempt_count = 0
        grasp_closing_pending = False
        grasp_lift_pending = False
        grasp_confirmed = False
        grasp_close_time = None
        grasp_close_target_position = None
        grasp_close_z = None
        grasp_lift_deadline = None
        grasp_valid_width_since = None
        synchronized_close_hold_active = False
        synchronized_close_hold_position = None
        synchronized_close_hold_rotation = None
        synchronized_close_hold_activations = 0
        synchronized_close_hold_cancellations = 0
        synchronized_close_hold_releases = 0
        episode_completed = False
        physical_gripper_enabled = args.execute and not args.disable_gripper
        close_latch_enabled = not args.disable_gripper_close_latch
        temporal_ensembler = TemporalActionEnsembler(
            args.temporal_ensemble_window,
            args.temporal_ensemble_decay,
            args.temporal_ensemble_gripper_threshold,
        )
        raw_first_gripper_previous = None
        ensembled_first_gripper_previous = None
        raw_first_gripper_switches = 0
        ensembled_first_gripper_switches = 0
        temporal_chunk_intent_changes = 0
        latch_suppressed_open_requests = 0

        if args.execute:
            node.set_held_command(
                target_position,
                target_rotation,
                gripper_state,
                publish_immediately=True,
            )
            node.get_logger().info(
                "Continuous command heartbeat enabled: "
                f"{args.publish_rate:.1f} Hz background publishing remains active "
                "during synchronous policy inference."
            )

        if args.gripper_switch_confirmations > 1:
            node.get_logger().warn(
                "Gripper temporal safety filter ENABLED: "
                f"chunk consensus >= {args.gripper_chunk_consensus:.2f}, "
                f"consecutive policy requests = {args.gripper_switch_confirmations}. "
                "No object-specific XYZ close gate is used."
            )
        else:
            node.get_logger().warn(
                "Gripper temporal safety filter DISABLED: every per-step policy "
                "gripper action will be forwarded."
            )

        if args.synchronized_close_hold:
            node.get_logger().warn(
                "Measured-pose synchronized close hold ENABLED: the first reliable "
                "policy close candidate rebases and freezes the Cartesian target "
                "through debounce and physical width validation. No object pose or "
                "task-specific close region is used."
            )
        else:
            node.get_logger().warn(
                "Measured-pose synchronized close hold DISABLED (raw-policy ablation)."
            )

        if args.temporal_ensemble_window > 1:
            node.get_logger().warn(
                "Aligned temporal action ensembling ENABLED: "
                f"window={args.temporal_ensemble_window}, "
                f"recency_decay={args.temporal_ensemble_decay:.3f}, "
                "Cartesian/RPY=weighted mean, gripper=weighted binary vote."
            )
        else:
            node.get_logger().info(
                "Aligned temporal action ensembling disabled (window=1)."
            )

        if close_latch_enabled:
            node.get_logger().warn(
                "Physical gripper close latch ENABLED: after a policy-confirmed "
                "close, transient open requests are held closed until width and "
                "lift validation complete. A later fresh policy request must still "
                "request release."
            )
        else:
            node.get_logger().warn(
                "Physical gripper close latch DISABLED for raw-policy ablation; "
                "an early open during grasp validation remains an episode failure."
            )

        node.get_logger().info(
            "Gripper feedback validation enabled (no object-specific XYZ close gate): "
            f"valid grasp width=[{args.grasp_width_min:.3f}, "
            f"{args.grasp_width_max:.3f}]m; "
            f"close validation starts at {args.empty_grasp_check_delay:.2f}s, "
            f"must enter range by {args.grasp_close_width_timeout:.2f}s, "
            f"requires {args.grasp_width_stable_duration:.2f}s stable width; "
            f"max descent={args.grasp_close_max_descent:.3f}m, "
            f"max lift={args.grasp_close_max_lift:.3f}m, "
            f"max XY drift={args.grasp_close_max_xy_drift:.3f}m; "
            f"required lift={args.grasp_min_lift:.3f}m; "
            f"physical release width>={args.release_open_width:.3f}m for "
            f"{args.release_open_stable_duration:.2f}s "
            f"(timeout={args.release_open_timeout:.2f}s); "
            f"max attempts={args.max_grasp_attempts}"
        )

        def update_active_grasp(context: str) -> None:
            """Advance nonblocking close, width validation, and lift confirmation."""
            nonlocal grasp_closing_pending, grasp_lift_pending, grasp_confirmed
            nonlocal grasp_close_z, grasp_lift_deadline, grasp_valid_width_since
            nonlocal target_position, target_rotation
            nonlocal synchronized_close_hold_active
            nonlocal synchronized_close_hold_position
            nonlocal synchronized_close_hold_rotation
            nonlocal synchronized_close_hold_releases
            if not physical_gripper_enabled:
                return
            if last_published_gripper >= 0.5:
                return
            if not (grasp_closing_pending or grasp_lift_pending or grasp_confirmed):
                return

            # Synchronous policy inference may have blocked callback processing
            # while the physical fingers continued moving.
            refreshed_ages = node.refresh_stale_observations(
                args.max_observation_age
            )
            measured_active_width, measured_closed_amount = node.copy_gripper_state()
            feedback_age = refreshed_ages["gripper_state"]
            if feedback_age > args.max_observation_age:
                raise RuntimeError(
                    "Abort: gripper feedback became stale while validating grasp: "
                    f"age={feedback_age:.3f}s"
                )
            active_position, active_rotation = node.copy_current_pose()

            if grasp_closing_pending:
                if grasp_close_time is None or grasp_close_z is None:
                    raise RuntimeError("Internal error: nonblocking close state is incomplete")
                grasp_close_z = min(grasp_close_z, float(active_position[2]))
                validation_now = time.monotonic()
                close_elapsed = validation_now - grasp_close_time
                if context == "after_action":
                    print(
                        "grasp_closing_progress:",
                        {
                            "elapsed_s": close_elapsed,
                            "validation_start_s": args.empty_grasp_check_delay,
                            "validation_timeout_s": args.grasp_close_width_timeout,
                            "total_width_m": measured_active_width,
                            "state_gripper_closed": measured_closed_amount,
                            "lowest_measured_z_m": grasp_close_z,
                        },
                    )
                if close_elapsed < args.empty_grasp_check_delay:
                    return

                width_status = classify_grasp_width(
                    measured_active_width,
                    args.empty_grasp_width,
                    args.grasp_width_min,
                    args.grasp_width_max,
                )
                print(
                    "grasp_width_check:",
                    {
                        "context": context,
                        "elapsed_s": close_elapsed,
                        "total_width_m": measured_active_width,
                        "state_gripper_closed": measured_closed_amount,
                        "empty_width_threshold_m": args.empty_grasp_width,
                        "valid_width_min_m": args.grasp_width_min,
                        "valid_width_max_m": args.grasp_width_max,
                        "feedback_age_s": feedback_age,
                        "width_status": width_status,
                    },
                )
                if width_status == "too_wide":
                    grasp_valid_width_since = None
                    if close_elapsed < args.grasp_close_width_timeout:
                        print(
                            "grasp_width_pending:",
                            {
                                "reason": "fingers_still_too_wide",
                                "elapsed_s": close_elapsed,
                                "timeout_s": args.grasp_close_width_timeout,
                                "total_width_m": measured_active_width,
                            },
                        )
                        return
                    raise RuntimeError(
                        "Abort: gripper did not reach the cube-compatible width "
                        "before the close timeout: "
                        f"measured={measured_active_width:.5f}m, "
                        f"timeout={args.grasp_close_width_timeout:.2f}s, "
                        f"valid=[{args.grasp_width_min:.5f}, "
                        f"{args.grasp_width_max:.5f}]m"
                    )
                if width_status == "empty":
                    grasp_valid_width_since = None
                    raise RuntimeError(
                        "Abort: empty grasp detected after nonblocking close: "
                        f"measured total finger width={measured_active_width:.5f}m "
                        f"< threshold={args.empty_grasp_width:.5f}m"
                    )
                if width_status == "too_narrow":
                    grasp_valid_width_since = None
                    raise RuntimeError(
                        "Abort: grasp contact width is outside the cube-compatible range: "
                        f"status={width_status}, measured={measured_active_width:.5f}m, "
                        f"valid=[{args.grasp_width_min:.5f}, "
                        f"{args.grasp_width_max:.5f}]m"
                    )
                if grasp_valid_width_since is None:
                    grasp_valid_width_since = validation_now
                stable_duration = validation_now - grasp_valid_width_since
                if stable_duration < args.grasp_width_stable_duration:
                    print(
                        "grasp_width_pending:",
                        {
                            "reason": "valid_width_not_yet_stable",
                            "stable_s": stable_duration,
                            "required_stable_s": args.grasp_width_stable_duration,
                            "range_entry_timeout_already_satisfied": True,
                            "total_width_m": measured_active_width,
                        },
                    )
                    return
                grasp_closing_pending = False
                grasp_lift_pending = True
                grasp_lift_deadline = validation_now + args.grasp_lift_timeout
                if synchronized_close_hold_active:
                    previous_anchor = synchronized_close_hold_position.copy()
                    target_position = active_position.copy()
                    target_rotation = active_rotation
                    synchronized_close_hold_active = False
                    synchronized_close_hold_position = None
                    synchronized_close_hold_rotation = None
                    synchronized_close_hold_releases += 1
                    if args.execute:
                        node.set_held_command(
                            target_position,
                            target_rotation,
                            last_published_gripper,
                            publish_immediately=True,
                        )
                    print(
                        "synchronized_close_hold:",
                        {
                            "event": "released_after_width_validation",
                            "previous_anchor": previous_anchor,
                            "measured_rebase": target_position.copy(),
                            "anchor_to_measured_xyz": target_position - previous_anchor,
                        },
                    )
                node.get_logger().info(
                    "Gripper phase CLOSING_VALIDATION -> LIFT_VALIDATION. "
                    "Stable grasp width check passed; awaiting lift confirmation "
                    f"from lowest z={grasp_close_z:.5f}m, "
                    f"width={measured_active_width:.5f}m, "
                    f"stable={stable_duration:.3f}s"
                )

            width_status = classify_grasp_width(
                measured_active_width,
                args.empty_grasp_width,
                args.grasp_width_min,
                args.grasp_width_max,
            )
            if width_status != "valid":
                raise RuntimeError(
                    "Abort: grasp width left the cube-compatible range while closed: "
                    f"context={context}, status={width_status}, "
                    f"measured={measured_active_width:.5f}m, "
                    f"valid=[{args.grasp_width_min:.5f}, "
                    f"{args.grasp_width_max:.5f}]m"
                )

            if not grasp_lift_pending:
                return
            if grasp_close_z is None or grasp_lift_deadline is None:
                raise RuntimeError("Internal error: grasp lift state is incomplete")

            measured_lift = float(active_position[2] - grasp_close_z)
            print(
                "grasp_lift_check:",
                {
                    "context": context,
                    "measured_lift_m": measured_lift,
                    "required_lift_m": args.grasp_min_lift,
                    "total_width_m": measured_active_width,
                },
            )
            if measured_lift >= args.grasp_min_lift:
                grasp_lift_pending = False
                grasp_confirmed = True
                node.get_logger().info(
                    "Gripper phase LIFT_VALIDATION -> HOLDING_OBJECT. "
                    "Grasp confirmed after lift: "
                    f"lift={measured_lift:.5f}m, width={measured_active_width:.5f}m"
                )
                return
            if time.monotonic() > grasp_lift_deadline:
                raise RuntimeError(
                    "Abort: grasp did not reach the required lift before timeout: "
                    f"lift={measured_lift:.5f}m < {args.grasp_min_lift:.5f}m"
                )

        while rclpy.ok() and steps_done < args.max_steps:
            ros_executor.raise_if_failed()
            # Advance the physical grasp state before taking a new observation.
            # If validation finishes here, only this fresh inference may begin
            # voting to release; open votes from older requests are never queued.
            update_active_grasp("before_request")
            close_latch_active_at_request = bool(
                physical_gripper_enabled
                and close_latch_enabled
                and (grasp_closing_pending or grasp_lift_pending)
            )
            # A policy request may take longer than --max-observation-age.
            # The background executor remains active during that request; wait
            # briefly only if a source has genuinely stopped updating.
            observation_ages = node.refresh_stale_observations(
                args.max_observation_age
            )
            stale = {
                key: age
                for key, age in observation_ages.items()
                if age > args.max_observation_age
            }
            if stale:
                raise RuntimeError(
                    f"Stale observation(s), max age={args.max_observation_age:.3f}s: {stale}"
                )
            images = node.copy_images()

            request_start = time.perf_counter()
            request_interval_s = None if last_request_start is None else request_start - last_request_start
            last_request_start = request_start
            publish_count_before_inference = node.command_publish_count()
            state_position, state_rotation = node.copy_current_pose()
            # CRISP recorded state.gripper as measured closed amount:
            #   1 - (finger_joint1 + finger_joint2) / 0.08
            # This preserves the partial closure seen while holding the cube
            # (about 0.63-0.65 in the successful training episodes) instead of
            # falsely reporting a binary commanded state.
            measured_width, state_gripper_closed = node.copy_gripper_state()
            raw_state = np.concatenate(
                [
                    state_position,
                    state_rotation.as_euler("xyz"),
                    np.array([state_gripper_closed], dtype=np.float64),
                ]
            )
            raw_actions, timing = request_action_chunk(
                client,
                images,
                raw_state,
                args,
                request_id,
            )
            cycle_inference_elapsed_s = time.perf_counter() - request_start
            publishes_during_inference = (
                node.command_publish_count() - publish_count_before_inference
            )
            request_id += 1

            trans_norms = np.linalg.norm(raw_actions[:, :3], axis=1)
            rot_norms = np.linalg.norm(raw_actions[:, 3:6], axis=1)
            raw_trans_abs_max = float(np.max(trans_norms))
            raw_rot_abs_max = float(np.max(rot_norms))

            actions, ensemble_diagnostics = temporal_ensembler.add_and_ensemble(
                steps_done,
                raw_actions,
            )

            actions_to_execute = actions[: min(args.execution_horizon, len(actions))]

            print("action_chunk_shape:", raw_actions.shape)
            print("execution_horizon:", len(actions_to_execute))
            print(
                "temporal_ensemble:",
                {
                    "enabled": args.temporal_ensemble_window > 1,
                    "start_step": steps_done,
                    "window": args.temporal_ensemble_window,
                    "decay": args.temporal_ensemble_decay,
                    "candidate_counts": ensemble_diagnostics.candidate_counts,
                    "oldest_weights": ensemble_diagnostics.oldest_weights,
                },
            )
            if args.log_timing:
                print(
                    "timing_request:",
                    {
                        "request_id": request_id - 1,
                        "request_interval_s": request_interval_s,
                        "client_cycle_inference_s": cycle_inference_elapsed_s,
                        "background_publishes_during_inference": publishes_during_inference,
                        "command_publish_age_s": node.command_publish_age(),
                        **timing,
                    },
                )
            print("raw_translation_min:", raw_actions[:, :3].min(axis=0))
            print("raw_translation_max:", raw_actions[:, :3].max(axis=0))
            print("translation_norm_max:", raw_trans_abs_max)
            print("raw_rpy_min:", raw_actions[:, 3:6].min(axis=0))
            print("raw_rpy_max:", raw_actions[:, 3:6].max(axis=0))
            print("rpy_norm_max:", raw_rot_abs_max)
            print("raw_gripper_values:", raw_actions[:, 6])
            print("ensembled_translation_min:", actions[:, :3].min(axis=0))
            print("ensembled_translation_max:", actions[:, :3].max(axis=0))
            print("ensembled_rpy_min:", actions[:, 3:6].min(axis=0))
            print("ensembled_rpy_max:", actions[:, 3:6].max(axis=0))
            print("ensembled_gripper_values:", actions[:, 6])
            print("raw_policy_state:", raw_state)
            print(
                "measured_gripper:",
                {
                    "total_width_m": measured_width,
                    "state_gripper_closed": state_gripper_closed,
                },
            )

            raw_first_gripper = float(raw_actions[0, 6] >= 0.5)
            ensembled_first_gripper = float(actions[0, 6] >= 0.5)
            if (
                raw_first_gripper_previous is not None
                and raw_first_gripper != raw_first_gripper_previous
            ):
                raw_first_gripper_switches += 1
            if (
                ensembled_first_gripper_previous is not None
                and ensembled_first_gripper != ensembled_first_gripper_previous
            ):
                ensembled_first_gripper_switches += 1
            raw_first_gripper_previous = raw_first_gripper
            ensembled_first_gripper_previous = ensembled_first_gripper

            raw_chunk_gripper_request = consensus_gripper_request(
                raw_actions,
                args.gripper_chunk_consensus,
            )
            chunk_gripper_request = consensus_gripper_request(
                actions,
                args.gripper_chunk_consensus,
            )
            if raw_chunk_gripper_request != chunk_gripper_request:
                temporal_chunk_intent_changes += 1
            latch_suppressed_chunk_open = False
            if args.gripper_switch_confirmations > 1:
                close_fraction = float(np.mean(actions[:, 6] < 0.5))
                open_fraction = 1.0 - close_fraction

                if close_latch_active_at_request:
                    # Do not let an early open become a queued release.  Once
                    # width/lift validation completes, a later fresh request
                    # must build confirmations from zero.
                    latch_suppressed_chunk_open = bool(
                        chunk_gripper_request is not None
                        and chunk_gripper_request >= 0.5
                    )
                    if latch_suppressed_chunk_open:
                        latch_suppressed_open_requests += 1
                    gripper_state = 0.0
                    pending_gripper_state = None
                    pending_gripper_count = 0
                elif chunk_gripper_request is None or chunk_gripper_request == gripper_state:
                    pending_gripper_state = None
                    pending_gripper_count = 0
                else:
                    if pending_gripper_state == chunk_gripper_request:
                        pending_gripper_count += 1
                    else:
                        pending_gripper_state = chunk_gripper_request
                        pending_gripper_count = 1
                    if pending_gripper_count >= args.gripper_switch_confirmations:
                        gripper_state = chunk_gripper_request
                        pending_gripper_state = None
                        pending_gripper_count = 0

                print(
                    "gripper_filter:",
                    {
                        "close_fraction": close_fraction,
                        "open_fraction": open_fraction,
                        "required_consensus": args.gripper_chunk_consensus,
                        "required_confirmations": args.gripper_switch_confirmations,
                        "raw_chunk_request": raw_chunk_gripper_request,
                        "chunk_request": chunk_gripper_request,
                        "pending_state": pending_gripper_state,
                        "pending_count": pending_gripper_count,
                        "output_state": gripper_state,
                        "close_latch_active_at_request": close_latch_active_at_request,
                        "latch_suppressed_chunk_open": latch_suppressed_chunk_open,
                    },
                )

            if args.gripper_switch_confirmations > 1:
                synchronized_close_candidate = bool(
                    chunk_gripper_request is not None
                    and chunk_gripper_request < 0.5
                )
            else:
                synchronized_close_candidate = bool(
                    len(actions_to_execute) > 0
                    and float(actions_to_execute[0, 6]) < 0.5
                )
            close_hold_transition = synchronized_close_hold_transition(
                enabled=args.synchronized_close_hold,
                hold_active=synchronized_close_hold_active,
                physical_gripper_enabled=physical_gripper_enabled,
                last_published_gripper=last_published_gripper,
                grasp_validation_active=bool(
                    grasp_closing_pending or grasp_lift_pending
                ),
                close_candidate=synchronized_close_candidate,
            )
            if close_hold_transition == "activate":
                measured_hold_position, measured_hold_rotation = (
                    node.copy_current_pose()
                )
                previous_target_position = target_position.copy()
                synchronized_close_hold_position = measured_hold_position.copy()
                synchronized_close_hold_rotation = measured_hold_rotation
                synchronized_close_hold_active = True
                synchronized_close_hold_activations += 1
                target_position = measured_hold_position.copy()
                target_rotation = measured_hold_rotation
                if args.execute:
                    node.set_held_command(
                        target_position,
                        target_rotation,
                        last_published_gripper,
                        publish_immediately=True,
                    )
                print(
                    "synchronized_close_hold:",
                    {
                        "event": "activated_on_first_close_candidate",
                        "measured_anchor": target_position.copy(),
                        "previous_target": previous_target_position,
                        "previous_target_lead_xyz": (
                            previous_target_position - target_position
                        ),
                        "previous_target_lead_xy_m": float(
                            np.linalg.norm(
                                previous_target_position[:2]
                                - target_position[:2]
                            )
                        ),
                    },
                )
            elif close_hold_transition == "cancel":
                previous_anchor = synchronized_close_hold_position.copy()
                measured_resume_position, measured_resume_rotation = (
                    node.copy_current_pose()
                )
                target_position = measured_resume_position.copy()
                target_rotation = measured_resume_rotation
                synchronized_close_hold_active = False
                synchronized_close_hold_position = None
                synchronized_close_hold_rotation = None
                synchronized_close_hold_cancellations += 1
                if args.execute:
                    node.set_held_command(
                        target_position,
                        target_rotation,
                        last_published_gripper,
                        publish_immediately=True,
                    )
                print(
                    "synchronized_close_hold:",
                    {
                        "event": "cancelled_before_physical_close",
                        "previous_anchor": previous_anchor,
                        "measured_rebase": target_position.copy(),
                        "anchor_to_measured_xyz": (
                            target_position - previous_anchor
                        ),
                    },
                )

            if raw_trans_abs_max > args.max_abs_trans_action:
                raise RuntimeError(
                    f"Abort: policy raw translation_norm_max={raw_trans_abs_max:.4f} "
                    f"> --max-abs-trans-action={args.max_abs_trans_action:.4f}"
                )
            if raw_rot_abs_max > args.max_abs_rot_action:
                raise RuntimeError(
                    f"Abort: policy raw rpy_norm_max={raw_rot_abs_max:.4f} "
                    f"> --max-abs-rot-action={args.max_abs_rot_action:.4f}"
                )

            close_latch_active_for_chunk = close_latch_active_at_request
            for action_index, action in enumerate(actions_to_execute):
                if steps_done >= args.max_steps:
                    break
                step_start = time.perf_counter()

                translation = clip_vector_norm(
                    action[:3] * args.translation_scale,
                    args.max_trans_delta,
                )
                delta_rpy = clip_vector_norm(action[3:6], args.max_rot_delta)
                if args.gripper_switch_confirmations > 1:
                    requested_gripper = gripper_state
                else:
                    requested_gripper = 1.0 if float(action[6]) >= 0.5 else 0.0

                update_active_grasp("before_action")

                # Forward the model's gripper decision regardless of Cartesian
                # position. Global workspace limits and feedback-based grasp
                # validation remain active below.
                measured_position, _ = node.copy_current_pose()
                close_latch_active_for_chunk = bool(
                    close_latch_active_for_chunk
                    or (
                        physical_gripper_enabled
                        and close_latch_enabled
                        and (grasp_closing_pending or grasp_lift_pending)
                    )
                )
                gripper, latch_suppressed_open = apply_gripper_close_latch(
                    requested_gripper,
                    close_latch_active_for_chunk,
                    physical_gripper_enabled and close_latch_enabled,
                )
                if latch_suppressed_open:
                    print(
                        "gripper_close_latch:",
                        {
                            "phase": (
                                "CLOSING_VALIDATION"
                                if grasp_closing_pending
                                else "LIFT_VALIDATION"
                                if grasp_lift_pending
                                else "VALIDATION_COMPLETED_DURING_THIS_CHUNK"
                            ),
                            "policy_requested": float(requested_gripper),
                            "effective_command": gripper,
                            "pending_open_votes_cleared": True,
                        },
                    )
                if args.gripper_switch_confirmations == 1:
                    gripper_state = gripper

                open_transition = bool(
                    last_published_gripper < 0.5 and gripper >= 0.5
                )
                physical_close_transition = False
                if physical_gripper_enabled:
                    physical_close_transition, open_transition = (
                        enforce_physical_grasp_transition(
                            last_published_gripper,
                            gripper,
                            grasp_attempt_count,
                            args.max_grasp_attempts,
                            grasp_closing_pending or grasp_lift_pending,
                            args.grasp_min_lift,
                        )
                    )

                if synchronized_close_hold_active:
                    if (
                        synchronized_close_hold_position is None
                        or synchronized_close_hold_rotation is None
                    ):
                        raise RuntimeError(
                            "Internal error: synchronized close hold has no anchor"
                        )
                    candidate_position = synchronized_close_hold_position.copy()
                    candidate_rotation = synchronized_close_hold_rotation
                else:
                    candidate_position = target_position + translation
                    candidate_rotation = (
                        Rotation.from_euler("xyz", delta_rpy) * target_rotation
                    )
                if not (
                    args.min_x <= candidate_position[0] <= args.max_x
                    and args.min_y <= candidate_position[1] <= args.max_y
                ):
                    raise RuntimeError(
                        "Abort: policy target left the XY workspace: "
                        f"target={np.array2string(candidate_position, precision=4)}, "
                        f"x=[{args.min_x}, {args.max_x}], y=[{args.min_y}, {args.max_y}]"
                    )
                candidate_position[2] = float(
                    np.clip(candidate_position[2], args.min_z, args.max_z)
                )
                if (
                    physical_gripper_enabled
                    and grasp_closing_pending
                    and not synchronized_close_hold_active
                ):
                    if grasp_close_target_position is None:
                        raise RuntimeError(
                            "Internal error: closing target origin is unavailable"
                        )
                    unconstrained_position = candidate_position.copy()
                    candidate_position, closing_motion_limited = constrain_closing_target(
                        candidate_position,
                        grasp_close_target_position,
                        args.grasp_close_max_xy_drift,
                        args.grasp_close_max_descent,
                        args.grasp_close_max_lift,
                    )
                    if closing_motion_limited:
                        print(
                            "grasp_closing_motion_limited:",
                            {
                                "requested_target": unconstrained_position,
                                "limited_target": candidate_position,
                                "close_origin": grasp_close_target_position,
                            },
                        )
                target_position = candidate_position
                target_rotation = candidate_rotation

                if physical_close_transition:
                    grasp_attempt_count += 1
                    grasp_closing_pending = True
                    grasp_lift_pending = False
                    grasp_confirmed = False
                    grasp_close_time = time.monotonic()
                    grasp_close_target_position = target_position.copy()
                    grasp_close_z = float(measured_position[2])
                    grasp_lift_deadline = None
                    grasp_valid_width_since = None
                    close_latch_active_for_chunk = bool(close_latch_enabled)
                    pending_gripper_state = None
                    pending_gripper_count = 0
                    node.get_logger().warn(
                        "Gripper phase OPEN -> CLOSING_VALIDATION. "
                        f"Physical grasp attempt {grasp_attempt_count}/"
                        f"{args.max_grasp_attempts}; continuing bounded policy motion "
                        f"for at least {args.empty_grasp_check_delay:.2f}s before width "
                        f"validation (timeout={args.grasp_close_width_timeout:.2f}s); "
                        f"measured_position={np.array2string(measured_position, precision=5)}, "
                        f"target_origin={np.array2string(grasp_close_target_position, precision=5)}"
                    )
                successful_release = bool(
                    open_transition and physical_gripper_enabled and grasp_confirmed
                )
                if successful_release:
                    node.get_logger().info(
                        "Gripper phase HOLDING_OBJECT -> OPENING_VALIDATION. "
                        "Validated first release requested; publishing the model's "
                        "open command, freezing the Cartesian target, and awaiting "
                        "physical finger-width confirmation."
                    )
                last_published_gripper = gripper

                print(
                    f"step={steps_done:03d} "
                    f"dpos={np.array2string(translation, precision=5)} "
                    f"drpy={np.array2string(delta_rpy, precision=5)} "
                    f"target_pos={np.array2string(target_position, precision=4)} "
                    f"gripper={gripper:.1f} "
                    f"synchronized_close_hold={synchronized_close_hold_active} "
                    f"execute={args.execute}"
                )

                if args.execute:
                    node.set_held_command(
                        target_position,
                        target_rotation,
                        gripper,
                        publish_immediately=True,
                    )

                steps_done += 1
                replan_after_action = bool(
                    action_index == len(actions_to_execute) - 1
                    and steps_done < args.max_steps
                )
                action_hold_target_s = compute_action_hold_duration(
                    period,
                    cycle_inference_elapsed_s,
                    replan_after_action,
                )
                hold_start = time.perf_counter()
                hold_publish_count_start = node.command_publish_count()
                end_time = time.monotonic() + action_hold_target_s
                while rclpy.ok() and time.monotonic() < end_time:
                    ros_executor.raise_if_failed()
                    remaining = end_time - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(min(0.01, remaining))
                action_hold_actual_s = time.perf_counter() - hold_start
                held_publish_count = (
                    node.command_publish_count() - hold_publish_count_start
                )
                if args.log_timing:
                    print(
                        "timing_step:",
                        {
                            "step": steps_done - 1,
                            "action_period_target_s": period,
                            "action_period_actual_s": time.perf_counter() - step_start,
                            "action_hold_target_s": action_hold_target_s,
                            "action_hold_actual_s": action_hold_actual_s,
                            "inference_included_in_period": replan_after_action,
                            "held_target_publish_count": held_publish_count,
                            "held_target_publish_rate_hz": args.publish_rate if args.execute else 0.0,
                        },
                    )
                update_active_grasp("after_action")

                if successful_release:
                    physical_open_width = wait_for_physical_release(
                        node,
                        ros_executor,
                        args.release_open_width,
                        args.release_open_stable_duration,
                        args.release_open_timeout,
                        args.max_observation_age,
                    )
                    episode_completed = True
                    node.get_logger().info(
                        "Gripper phase OPENING_VALIDATION -> RELEASED. "
                        "EPISODE COMPLETE: valid grasp, required lift, and first "
                        "policy-requested release were physically confirmed; "
                        f"open width={physical_open_width:.5f}m. Visually verify "
                        "that the object reached the task target."
                    )
                    break

            if episode_completed:
                break

        print(
            "gripper_stability_summary:",
            {
                "policy_requests": request_id,
                "raw_first_action_switches": raw_first_gripper_switches,
                "ensembled_first_action_switches": ensembled_first_gripper_switches,
                "temporal_chunk_intent_changes": temporal_chunk_intent_changes,
                "latch_suppressed_open_requests": latch_suppressed_open_requests,
                "synchronized_close_hold_enabled": args.synchronized_close_hold,
                "synchronized_close_hold_active": synchronized_close_hold_active,
                "synchronized_close_hold_activations": synchronized_close_hold_activations,
                "synchronized_close_hold_cancellations": synchronized_close_hold_cancellations,
                "synchronized_close_hold_releases": synchronized_close_hold_releases,
                "physical_grasp_attempts": grasp_attempt_count,
                "grasp_confirmed": grasp_confirmed,
                "episode_completed": episode_completed,
            },
        )

        if physical_gripper_enabled and (grasp_closing_pending or grasp_lift_pending):
            raise RuntimeError(
                "Deployment reached --max-steps while grasp validation was still in "
                "progress; increase --max-steps and return to the standard pose before "
                "the next run"
            )
        if episode_completed:
            node.get_logger().info(
                "Finished StarVLA Franka delta-pose client after validated release."
            )
        else:
            node.get_logger().info(
                "Finished StarVLA Franka delta-pose client without a validated release."
            )
    finally:
        node.stop_held_command()
        try:
            if client is not None:
                client.close()
        finally:
            ros_executor.stop()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
