#!/usr/bin/env python3
"""Open-loop L2 evaluation for a StarVLA websocket policy server.

The script reads LeRobot v2.1 datasets, sends recorded primary and wrist image
observations to a running StarVLA policy server, and compares predicted 7D
delta EEF actions with the ground-truth dataset actions.

It does not use ROS and never moves the robot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


TASK = "pick up the cube and place it on the box"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=10093)
    parser.add_argument("--dataset-root", default="dataset/snkdjn")
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="Episode ids such as 0150, or dataset names such as quest3_franka_tele_0150.",
    )
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--action-offset",
        type=int,
        default=0,
        help="Compare prediction at obs frame t against GT action[t + offset]. Try 0 and 1 if unsure.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Evaluate one observation every N dataset frames. 15 means about 1 Hz for 15 FPS data.",
    )
    parser.add_argument("--max-queries-per-episode", type=int, default=32)
    parser.add_argument(
        "--compare",
        choices=["first", "chunk", "both"],
        default="both",
        help="Report first-step L2, whole predicted chunk L2, or both.",
    )
    parser.add_argument(
        "--y-sign-eps",
        type=float,
        default=5e-4,
        help="Ignore GT y actions with abs(y) below this when computing y sign accuracy.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--keep-frames-dir", default=None)
    parser.add_argument(
        "--inference-seed-base",
        type=int,
        default=None,
        help=(
            "Enable deterministic diffusion sampling. Each dataset/frame gets "
            "a stable seed derived from this base; use the same base for matched "
            "control and treatment evaluations."
        ),
    )
    return parser.parse_args()


def _pack_array(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
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
    import numpy as np

    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class WebsocketPolicyClient:
    def __init__(self, host: str, port: int):
        import msgpack
        import websockets.sync.client

        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        self._msgpack = msgpack
        self._packer = self._msgpack.Packer(default=_pack_array)
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=150,
            ping_interval=None,
            ping_timeout=60,
        )
        self._metadata = self._unpack(self._ws.recv())

    def _unpack(self, data):
        return self._msgpack.unpackb(data, object_hook=_unpack_array)

    def get_server_metadata(self) -> dict:
        return self._metadata

    def predict_action(self, request: dict) -> dict:
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server returned text error:\n{response}")
        return self._unpack(response)

    def close(self) -> None:
        self._ws.close()


def resolve_dataset(root: Path, raw_id: str) -> Path:
    raw_path = Path(raw_id).expanduser()
    candidates = []
    if raw_path.is_absolute() or "/" in raw_id:
        candidates.append(raw_path)
    else:
        # First accept any dataset directory exactly as named under the root
        # (for example quest3_9_grids_055). Keep historical shorthand rules as
        # fallbacks for bare numeric episode ids.
        candidates.append(root / raw_id)
    if not (raw_path.is_absolute() or "/" in raw_id) and not raw_id.startswith(
        "quest3_franka_"
    ):
        # The vetted dual-camera sources used to build the 74-episode dataset
        # are named quest3_franka_dualcam_test_<ID>.  Keep the older tele name
        # as a fallback for historical single-camera recordings.
        candidates.extend(
            [
                root / f"quest3_franka_dualcam_test_{raw_id}",
                root / f"quest3_franka_tele_{raw_id}",
            ]
        )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Dataset not found; checked: " + ", ".join(str(path) for path in candidates)
    )


def read_actions(parquet_path: Path) -> np.ndarray:
    import numpy as np

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read LeRobot parquet files") from exc

    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns:
        raise RuntimeError(f"{parquet_path} has no `action` column")
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"Expected action shape (T, 7+), got {actions.shape}")
    return actions[:, :7]


def read_policy_states(parquet_path: Path) -> np.ndarray:
    """Read the raw 7D state used by the real Franka deployment client."""
    import numpy as np

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read LeRobot parquet files") from exc

    df = pd.read_parquet(parquet_path)
    cartesian_key = "observation.state.cartesian"
    gripper_key = "observation.state.gripper"
    if cartesian_key not in df.columns or gripper_key not in df.columns:
        raise RuntimeError(
            f"{parquet_path} must contain {cartesian_key!r} and {gripper_key!r}"
        )
    cartesian = np.stack(df[cartesian_key].to_numpy()).astype(np.float32)
    gripper = np.stack(df[gripper_key].to_numpy()).astype(np.float32)
    if cartesian.ndim != 2 or cartesian.shape[1] < 6:
        raise RuntimeError(f"Expected Cartesian state shape (T, 6+), got {cartesian.shape}")
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    if gripper.ndim != 2 or gripper.shape[1] < 1:
        raise RuntimeError(f"Expected gripper state shape (T, 1+), got {gripper.shape}")
    if len(cartesian) != len(gripper):
        raise RuntimeError(
            f"State length mismatch: Cartesian={len(cartesian)}, gripper={len(gripper)}"
        )
    return np.concatenate([cartesian[:, :6], gripper[:, :1]], axis=1)


def video_path(dataset_dir: Path, video_key: str) -> Path:
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        rel = info.get("video_path")
        if rel:
            candidate = dataset_dir / rel.format(
                episode_chunk=0,
                video_key=video_key,
                episode_index=0,
            )
            if candidate.exists():
                return candidate
    candidate = dataset_dir / f"videos/chunk-000/{video_key}/episode_000000.mp4"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find {video_key} video under {dataset_dir}")


def parquet_path(dataset_dir: Path) -> Path:
    candidate = dataset_dir / "data/chunk-000/episode_000000.parquet"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find parquet under {dataset_dir}")


def selected_indices(num_frames: int, start_frame: int, stride: int, max_queries: int) -> list[int]:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    indices = list(range(start_frame, num_frames, stride))
    if max_queries > 0:
        indices = indices[:max_queries]
    return indices


def extract_selected_frames(video: Path, indices: list[int], out_dir: Path) -> list[Path]:
    if not indices:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer PyAV so the evaluator uses the codecs bundled with the active
    # Python environment.  The Franka container also contains an old standalone
    # ffmpeg whose shared-library dependencies no longer match the environment.
    try:
        import av

        wanted = {frame_index: output_index for output_index, frame_index in enumerate(indices, 1)}
        extracted: dict[int, Path] = {}
        with av.open(str(video)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                output_index = wanted.get(frame_index)
                if output_index is None:
                    continue
                output_path = out_dir / f"frame_{output_index:06d}.png"
                frame.to_image().convert("RGB").save(output_path)
                extracted[frame_index] = output_path
                if len(extracted) == len(indices):
                    break
        missing = [index for index in indices if index not in extracted]
        if missing:
            raise RuntimeError(
                f"Could not decode requested frame indices {missing} from {video}"
            )
        return [extracted[index] for index in indices]
    except ImportError:
        pass

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("PyAV or ffmpeg is required to extract video frames")

    start = indices[0]
    stride = indices[1] - indices[0] if len(indices) > 1 else 1
    max_frames = len(indices)
    # Select n=start,start+stride,... . The frame count cap keeps extraction fast.
    select_expr = f"gte(n\\,{start})*not(mod(n-{start}\\,{stride}))"
    out_pattern = str(out_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "0",
        "-frames:v",
        str(max_frames),
        out_pattern,
    ]
    subprocess.run(cmd, check=True)
    paths = sorted(out_dir.glob("frame_*.png"))
    if len(paths) != len(indices):
        raise RuntimeError(f"Expected {len(indices)} extracted frames, got {len(paths)} from {video}")
    return paths


def resize_rgb(path: Path, size: int) -> np.ndarray:
    import numpy as np
    from PIL import Image as PILImage

    image = PILImage.open(path).convert("RGB")
    image = image.resize((size, size))
    return np.asarray(image, dtype=np.uint8)


def choose_unnorm_key(metadata: dict, requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    if metadata.get("default_unnorm_key") is not None:
        return metadata["default_unnorm_key"]
    keys = metadata.get("available_unnorm_keys") or []
    if len(keys) == 1:
        return keys[0]
    return None


def request_action_chunk(
    client: WebsocketPolicyClient,
    images: list[np.ndarray],
    raw_state: np.ndarray,
    task: str,
    unnorm_key: str,
    request_id: str,
    inference_seed: Optional[int] = None,
) -> np.ndarray:
    import numpy as np

    request = {
        "type": "predict_action",
        "request_id": request_id,
        "payload": {
            "examples": [
                {
                    "image": images,
                    "lang": task,
                    "state": np.asarray(raw_state, dtype=np.float32).reshape(1, -1),
                }
            ],
            "unnorm_key": unnorm_key,
            "normalize_state": True,
        },
    }
    if inference_seed is not None:
        request["payload"]["inference_seed"] = int(inference_seed)
    response = client.predict_action(request)
    if not response.get("ok", False):
        raise RuntimeError(f"Policy server returned error: {response}")
    data = response.get("data", {})
    if "actions" not in data:
        raise RuntimeError(f"Policy response has no `actions`. Keys: {list(data.keys())}")
    actions = np.asarray(data["actions"], dtype=np.float64)
    if actions.ndim == 3:
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"Expected predicted action chunk shape (T, 7+), got {actions.shape}")
    return actions[:, :7]


def stable_inference_seed(
    seed_base: int,
    dataset_name: str,
    frame_index: int,
    action_offset: int,
) -> int:
    """Derive a stable signed-64-bit-compatible seed for paired A/B inference."""
    material = f"{seed_base}:{dataset_name}:{frame_index}:{action_offset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


@dataclass
class Row:
    dataset: str
    frame_index: int
    gt_action_index: int
    inference_seed: Optional[int]
    chunk_len: int
    first_l2: float
    first_xyz_l2: float
    first_rpy_l2: float
    first_gripper_abs: float
    chunk_l2_mean: float
    chunk_xyz_l2_mean: float
    chunk_rpy_l2_mean: float
    chunk_gripper_abs_mean: float
    pred_dx: float
    pred_dy: float
    pred_dz: float
    pred_gripper: float
    gt_dx: float
    gt_dy: float
    gt_dz: float
    gt_gripper: float
    state_x: float
    state_y: float
    state_z: float
    state_gripper: float
    pred_chunk_close_fraction: float
    gt_chunk_close_fraction: float


def compute_row(
    dataset: str,
    frame_index: int,
    gt_action_index: int,
    inference_seed: Optional[int],
    pred: np.ndarray,
    gt_actions: np.ndarray,
    raw_state: np.ndarray,
) -> Row:
    import numpy as np

    gt = gt_actions[gt_action_index : gt_action_index + len(pred)]
    chunk_len = min(len(pred), len(gt))
    pred = pred[:chunk_len]
    gt = gt[:chunk_len]
    diff = pred - gt
    first = diff[0]
    l2 = np.linalg.norm(diff, axis=1)
    xyz_l2 = np.linalg.norm(diff[:, :3], axis=1)
    rpy_l2 = np.linalg.norm(diff[:, 3:6], axis=1)
    gripper_abs = np.abs(diff[:, 6])
    return Row(
        dataset=dataset,
        frame_index=frame_index,
        gt_action_index=gt_action_index,
        inference_seed=inference_seed,
        chunk_len=chunk_len,
        first_l2=float(np.linalg.norm(first)),
        first_xyz_l2=float(np.linalg.norm(first[:3])),
        first_rpy_l2=float(np.linalg.norm(first[3:6])),
        first_gripper_abs=float(abs(first[6])),
        chunk_l2_mean=float(l2.mean()),
        chunk_xyz_l2_mean=float(xyz_l2.mean()),
        chunk_rpy_l2_mean=float(rpy_l2.mean()),
        chunk_gripper_abs_mean=float(gripper_abs.mean()),
        pred_dx=float(pred[0, 0]),
        pred_dy=float(pred[0, 1]),
        pred_dz=float(pred[0, 2]),
        pred_gripper=float(pred[0, 6]),
        gt_dx=float(gt[0, 0]),
        gt_dy=float(gt[0, 1]),
        gt_dz=float(gt[0, 2]),
        gt_gripper=float(gt[0, 6]),
        state_x=float(raw_state[0]),
        state_y=float(raw_state[1]),
        state_z=float(raw_state[2]),
        state_gripper=float(raw_state[6]),
        pred_chunk_close_fraction=float(np.mean(pred[:, 6] < 0.5)),
        gt_chunk_close_fraction=float(np.mean(gt[:, 6] < 0.5)),
    )


def summarize(rows: list[Row], y_sign_eps: float) -> dict:
    import numpy as np

    if not rows:
        return {}
    first_l2 = np.array([r.first_l2 for r in rows])
    chunk_l2 = np.array([r.chunk_l2_mean for r in rows])
    first_xyz = np.array([r.first_xyz_l2 for r in rows])
    first_gripper = np.array([r.first_gripper_abs for r in rows])
    pred_y = np.array([r.pred_dy for r in rows])
    gt_y = np.array([r.gt_dy for r in rows])
    pred_close = np.array([r.pred_gripper < 0.5 for r in rows], dtype=bool)
    gt_close = np.array([r.gt_gripper < 0.5 for r in rows], dtype=bool)
    gt_open = ~gt_close
    y_mask = np.abs(gt_y) >= y_sign_eps
    y_sign_acc = np.nan
    if np.any(y_mask):
        y_sign_acc = float((np.sign(pred_y[y_mask]) == np.sign(gt_y[y_mask])).mean())
    first_pred_close_frame = next(
        (r.frame_index for r in rows if r.pred_gripper < 0.5), None
    )
    first_gt_close_frame = next(
        (r.frame_index for r in rows if r.gt_gripper < 0.5), None
    )
    return {
        "queries": len(rows),
        "first_l2_mean": float(first_l2.mean()),
        "first_l2_median": float(np.median(first_l2)),
        "first_l2_p90": float(np.percentile(first_l2, 90)),
        "chunk_l2_mean": float(chunk_l2.mean()),
        "first_xyz_l2_mean": float(first_xyz.mean()),
        "first_gripper_abs_mean": float(first_gripper.mean()),
        "y_mae": float(np.abs(pred_y - gt_y).mean()),
        "y_sign_acc": y_sign_acc,
        "pred_y_mean": float(pred_y.mean()),
        "gt_y_mean": float(gt_y.mean()),
        "pred_y_abs_mean": float(np.abs(pred_y).mean()),
        "gt_y_abs_mean": float(np.abs(gt_y).mean()),
        "gripper_binary_accuracy": float((pred_close == gt_close).mean()),
        "false_close_rate_when_gt_open": (
            float(pred_close[gt_open].mean()) if np.any(gt_open) else np.nan
        ),
        "missed_close_rate_when_gt_closed": (
            float((~pred_close[gt_close]).mean()) if np.any(gt_close) else np.nan
        ),
        "first_pred_close_frame": first_pred_close_frame,
        "first_gt_close_frame": first_gt_close_frame,
        "first_close_frame_error": (
            first_pred_close_frame - first_gt_close_frame
            if first_pred_close_frame is not None and first_gt_close_frame is not None
            else None
        ),
    }


def print_summary(title: str, rows: list[Row], y_sign_eps: float) -> None:
    stats = summarize(rows, y_sign_eps)
    if not stats:
        print(f"{title}: no rows")
        return
    print(f"\n{title}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    client = WebsocketPolicyClient(args.policy_host, args.policy_port)
    try:
        metadata = client.get_server_metadata()
        print("Policy metadata:", metadata)
        if args.inference_seed_base is not None and not metadata.get(
            "supports_inference_seed", False
        ):
            raise RuntimeError(
                "The policy server does not advertise supports_inference_seed=True. "
                "Install the seeded-inference server patch and restart this server."
            )
        unnorm_key = choose_unnorm_key(metadata, args.unnorm_key)
        if unnorm_key is None:
            raise RuntimeError(f"Could not choose unnorm_key from metadata: {metadata}")
        print("Using unnorm_key:", unnorm_key)

        all_rows: list[Row] = []
        for raw_id in args.ids:
            ds_dir = resolve_dataset(root, raw_id)
            name = ds_dir.name
            parquet = parquet_path(ds_dir)
            gt_actions = read_actions(parquet)
            policy_states = read_policy_states(parquet)
            if len(policy_states) != len(gt_actions):
                raise RuntimeError(
                    f"State/action length mismatch for {name}: "
                    f"states={len(policy_states)}, actions={len(gt_actions)}"
                )
            indices = selected_indices(
                len(gt_actions),
                args.start_frame,
                args.stride,
                args.max_queries_per_episode,
            )
            if not indices:
                print(f"\n{name}: no selected frames")
                continue

            if args.keep_frames_dir:
                frame_dir = Path(args.keep_frames_dir).expanduser().resolve() / name
                frame_dir.mkdir(parents=True, exist_ok=True)
                cleanup = False
            else:
                frame_dir = Path(tempfile.mkdtemp(prefix=f"starvla_open_loop_{name}_"))
                cleanup = True

            try:
                primary_frames = extract_selected_frames(
                    video_path(ds_dir, "observation.images.primary"),
                    indices,
                    frame_dir / "primary",
                )
                wrist_frames = extract_selected_frames(
                    video_path(ds_dir, "observation.images.wrist"),
                    indices,
                    frame_dir / "wrist",
                )
                ds_rows: list[Row] = []
                for index, primary_path, wrist_path in zip(
                    indices, primary_frames, wrist_frames
                ):
                    gt_action_index = index + args.action_offset
                    if gt_action_index < 0 or gt_action_index >= len(gt_actions):
                        continue
                    images = [
                        resize_rgb(primary_path, args.image_size),
                        resize_rgb(wrist_path, args.image_size),
                    ]
                    inference_seed = (
                        stable_inference_seed(
                            args.inference_seed_base,
                            name,
                            index,
                            args.action_offset,
                        )
                        if args.inference_seed_base is not None
                        else None
                    )
                    request_id = (
                        f"open-loop-{name}-{index}-seed-{inference_seed}"
                        if inference_seed is not None
                        else f"open-loop-{name}-{index}-{time.time_ns()}"
                    )
                    raw_state = policy_states[index]
                    pred = request_action_chunk(
                        client,
                        images,
                        raw_state,
                        args.task,
                        unnorm_key,
                        request_id,
                        inference_seed,
                    )
                    row = compute_row(
                        name,
                        index,
                        gt_action_index,
                        inference_seed,
                        pred,
                        gt_actions,
                        raw_state,
                    )
                    ds_rows.append(row)
                    print(
                        f"{name} frame={index:04d} "
                        f"gt_action={gt_action_index:04d} "
                        f"first_l2={row.first_l2:.5f} "
                        f"chunk_l2={row.chunk_l2_mean:.5f} "
                        f"pred_xyz=[{row.pred_dx:.4f},{row.pred_dy:.4f},{row.pred_dz:.4f}] "
                        f"gt_xyz=[{row.gt_dx:.4f},{row.gt_dy:.4f},{row.gt_dz:.4f}] "
                        f"z={row.state_z:.4f} "
                        f"pred_g={row.pred_gripper:.3f} gt_g={row.gt_gripper:.3f} "
                        f"pred_close_frac={row.pred_chunk_close_fraction:.2f} "
                        f"gt_close_frac={row.gt_chunk_close_fraction:.2f}"
                    )
                print_summary(f"{name} summary", ds_rows, args.y_sign_eps)
                all_rows.extend(ds_rows)
            finally:
                if cleanup:
                    shutil.rmtree(frame_dir, ignore_errors=True)

        print_summary("OVERALL summary", all_rows, args.y_sign_eps)
        if args.output_csv:
            write_csv(Path(args.output_csv).expanduser().resolve(), all_rows)
            print(f"\nSaved CSV: {Path(args.output_csv).expanduser().resolve()}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
