#!/usr/bin/env python3
"""Probe full VGGT depth/point/track heads on one offline Replay94 episode."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw


PHASES = ("approach", "pre_grasp", "grasp", "transport", "release")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--query-points", type=int, default=8)
    parser.add_argument("--expected-weight-sha256", default=None)
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--confirmation-window", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def resolve_video_keys(info: dict[str, Any]) -> dict[str, str]:
    keys = [
        key
        for key, value in info.get("features", {}).items()
        if value.get("dtype") == "video"
    ]
    primary = [key for key in keys if "primary" in key.lower()]
    wrist = [key for key in keys if "wrist" in key.lower()]
    if len(primary) != 1 or len(wrist) != 1:
        raise RuntimeError(
            f"Expected primary/wrist video keys, got primary={primary}, wrist={wrist}"
        )
    return {"primary": primary[0], "wrist": wrist[0]}


def episode_path(
    root: Path,
    template: str,
    episode_index: int,
    chunks_size: int,
    *,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return root / template.format(**values)


def read_gripper_values(path: Path) -> list[float]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if "action.gripper" in table.column_names:
        values = table["action.gripper"].to_pylist()
        return [float(value[0] if isinstance(value, list) else value) for value in values]
    if "action" not in table.column_names:
        raise KeyError(f"No gripper/action column in {path}")
    result = []
    for action in table["action"].to_pylist():
        if not isinstance(action, (list, tuple)) or not action:
            raise ValueError(f"Malformed action row: {action!r}")
        result.append(float(action[-1]))
    return result


def first_sustained(values, *, start: int, window: int, predicate) -> int | None:
    for index in range(max(start, 0), max(0, len(values) - window + 1)):
        if all(predicate(float(value)) for value in values[index : index + window]):
            return index
    return None


def infer_phases(
    values: list[float],
    *,
    fps: float,
    threshold: float,
    window: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    if not values or fps <= 0 or window <= 0:
        raise ValueError("Invalid phase inputs")
    close = first_sustained(
        values, start=1, window=window, predicate=lambda value: value <= threshold
    )
    close_fallback = close is None
    if close is None:
        close = max(1, int(round(0.42 * (len(values) - 1))))
    release = None
    if not close_fallback:
        release = first_sustained(
            values,
            start=min(len(values) - 1, close + window),
            window=window,
            predicate=lambda value: value > threshold,
        )
    release_fallback = release is None
    if release is None:
        release = min(
            len(values) - 1,
            max(close + 1, int(round(0.82 * (len(values) - 1)))),
        )
    half_second = max(1, int(round(0.5 * fps)))
    grasp = min(len(values) - 1, close + window - 1)
    phases = {
        "approach": max(0, int(round(0.4 * close))),
        "pre_grasp": max(0, close - half_second),
        "grasp": grasp,
        "transport": min(len(values) - 1, max(grasp, int(round((grasp + release) / 2)))),
        "release": release,
    }
    previous = 0
    for phase in PHASES:
        phases[phase] = max(previous, min(len(values) - 1, phases[phase]))
        previous = phases[phase]
    return phases, {
        "close_index": close,
        "release_index": release,
        "close_fallback": close_fallback,
        "release_fallback": release_fallback,
    }


def extract_frames(video: Path, indices: Iterable[int], output: Path) -> dict[int, Path]:
    indices = sorted(set(int(index) for index in indices))
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    output.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    pattern = output / "frame_%03d.png"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select={expression}",
        "-fps_mode",
        "vfr",
        str(pattern),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    paths = sorted(output.glob("frame_*.png"))
    if len(paths) != len(indices):
        raise RuntimeError(f"Expected {len(indices)} frames, found {len(paths)}")
    return dict(zip(indices, paths, strict=True))


def largest_compact_blue_component(image: Image.Image, view: str) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (
        (blue >= 75)
        & (green >= 65)
        & ((blue - red) >= 22)
        & ((green - red) >= 15)
    )
    height, width = mask.shape
    envelope_top = int((0.44 if view == "primary" else 0.12) * height)
    mask[:envelope_top] = False
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    maximum_area = int(0.05 * height * width)
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        queue = deque([(y, x)])
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if 12 <= len(component) <= maximum_area and len(component) > len(best):
            best = component
    result = np.zeros_like(mask, dtype=bool)
    if best:
        yy, xx = zip(*best)
        result[np.asarray(yy), np.asarray(xx)] = True
    return result


def sample_query_points(mask: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    coordinates = np.argwhere(mask)
    if len(coordinates) < 1:
        return np.empty((0, 2), dtype=np.float32)
    centroid = coordinates.mean(axis=0, keepdims=True)
    distances = np.square(coordinates - centroid).sum(axis=1)
    order = np.argsort(distances)
    selected = coordinates[order[np.linspace(0, len(order) - 1, count, dtype=int)]]
    points_yx = np.concatenate((centroid, selected), axis=0)[:count]
    return points_yx[:, ::-1].astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_state_dict(value: Any) -> dict[str, torch.Tensor]:
    """Accept the flat official VGGT file and common nested checkpoint layouts."""

    if not isinstance(value, dict):
        raise TypeError(f"VGGT checkpoint must be a dict, got {type(value).__name__}")
    for key in ("state_dict", "model", "module"):
        nested = value.get(key)
        if isinstance(nested, dict) and nested:
            value = nested
            break
    if not value or not all(isinstance(key, str) for key in value):
        raise ValueError("VGGT checkpoint contains no string-keyed state dict")
    if all(key.startswith("module.") for key in value):
        value = {key.removeprefix("module."): tensor for key, tensor in value.items()}
    return value


def tensor_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    finite = torch.isfinite(value)
    element_count = int(value.numel())
    finite_count = int(finite.sum(dtype=torch.int64).detach().cpu().item())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        # Do not use a float32 mean over million-element boolean maps: its
        # reduction can round an all-True tensor to 0.99999994.
        "element_count": element_count,
        "finite_count": finite_count,
        "invalid_count": element_count - finite_count,
        "all_finite": finite_count == element_count,
        "finite_fraction": finite_count / element_count if element_count else 0.0,
        "abs_mean": float(value.float().abs().mean().detach().cpu().item()),
        "abs_max": float(value.float().abs().max().detach().cpu().item()),
    }


def summarize_query_geometry(
    predictions: dict[str, Any],
    phase_order: list[str],
    query_points: np.ndarray,
) -> dict[str, Any]:
    """Summarize calibration-free depth/point evidence at tracked cube pixels."""

    required = ("track", "depth", "world_points")
    if not len(query_points) or any(not isinstance(predictions.get(key), torch.Tensor) for key in required):
        return {"available": False}
    track = predictions["track"].detach().float().cpu()
    depth = predictions["depth"].detach().float().cpu()
    points = predictions["world_points"].detach().float().cpu()
    if track.ndim != 4 or depth.ndim != 5 or points.ndim != 5:
        return {
            "available": False,
            "reason": f"unexpected shapes: track={tuple(track.shape)}, depth={tuple(depth.shape)}, points={tuple(points.shape)}",
        }
    if track.shape[0] != 1 or depth.shape[0] != 1 or points.shape[0] != 1:
        return {"available": False, "reason": "capability audit expects batch size one"}
    sequence = min(track.shape[1], depth.shape[1], points.shape[1], len(phase_order))
    if sequence < 1:
        return {"available": False, "reason": "empty temporal output"}
    height, width = int(depth.shape[2]), int(depth.shape[3])
    if tuple(points.shape[2:4]) != (height, width):
        return {"available": False, "reason": "depth/point spatial shapes differ"}

    query = torch.from_numpy(query_points).float()
    first_reprojection_error = torch.linalg.vector_norm(track[0, 0] - query, dim=-1)
    per_phase: dict[str, Any] = {}
    all_depth = []
    all_in_bounds = []
    previous_xy = None
    previous_xyz = None
    pixel_step_motion = []
    point_step_motion = []
    for index in range(sequence):
        xy = track[0, index]
        finite_xy = torch.isfinite(xy).all(dim=-1)
        in_bounds = (
            finite_xy
            & (xy[:, 0] >= 0)
            & (xy[:, 0] <= width - 1)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] <= height - 1)
        )
        x = xy[:, 0].round().clamp(0, width - 1).long()
        y = xy[:, 1].round().clamp(0, height - 1).long()
        sampled_depth = depth[0, index, y, x, 0]
        sampled_xyz = points[0, index, y, x]
        valid_depth = in_bounds & torch.isfinite(sampled_depth) & (sampled_depth > 0)
        valid_xyz = in_bounds & torch.isfinite(sampled_xyz).all(dim=-1)
        all_depth.append(valid_depth)
        all_in_bounds.append(in_bounds)
        if previous_xy is not None:
            pixel_step_motion.append(torch.linalg.vector_norm(xy - previous_xy, dim=-1))
        if previous_xyz is not None:
            point_step_motion.append(torch.linalg.vector_norm(sampled_xyz - previous_xyz, dim=-1))
        previous_xy = xy
        previous_xyz = sampled_xyz
        per_phase[phase_order[index]] = {
            "in_bounds_fraction": float(in_bounds.float().mean().item()),
            "positive_depth_fraction": float(valid_depth.float().mean().item()),
            "median_depth": float(sampled_depth[valid_depth].median().item()) if valid_depth.any() else None,
            "median_track_xy": [float(value) for value in xy[finite_xy].median(dim=0).values.tolist()]
            if finite_xy.any()
            else None,
            "median_world_point": [float(value) for value in sampled_xyz[valid_xyz].median(dim=0).values.tolist()]
            if valid_xyz.any()
            else None,
        }
    return {
        "available": True,
        "first_frame_reprojection_error_px_mean": float(first_reprojection_error.mean().item()),
        "in_bounds_fraction": float(torch.cat(all_in_bounds).float().mean().item()),
        "positive_depth_fraction": float(torch.cat(all_depth).float().mean().item()),
        "median_temporal_pixel_step_px": float(torch.cat(pixel_step_motion).median().item())
        if pixel_step_motion
        else 0.0,
        "median_temporal_world_point_step": float(torch.cat(point_step_motion).median().item())
        if point_step_motion
        else 0.0,
        "per_phase": per_phase,
    }


def save_tracking_overlay(
    *,
    images: list[Image.Image],
    phase_order: list[str],
    query_points: np.ndarray,
    track: torch.Tensor,
    query_mask: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    """Save a contact sheet showing the cube query and VGGT track evidence."""

    coordinates = track.detach().float().cpu()
    if coordinates.ndim != 4 or coordinates.shape[0] != 1:
        raise ValueError(f"Expected track [1,S,N,2], got {tuple(coordinates.shape)}")
    sequence = min(len(images), len(phase_order), coordinates.shape[1])
    if sequence < 1:
        raise ValueError("No tracking frames to visualize")
    cell_size = 360
    panels = []
    track_by_phase: dict[str, list[list[float]]] = {}
    mask_yx = np.argwhere(query_mask)
    mask_box = None
    if len(mask_yx):
        y0, x0 = mask_yx.min(axis=0)
        y1, x1 = mask_yx.max(axis=0)
        mask_box = [int(x0), int(y0), int(x1), int(y1)]
    for index in range(sequence):
        source = images[index].convert("RGB")
        width, height = source.size
        panel = source.resize((cell_size, cell_size), Image.Resampling.BICUBIC)
        scale_x, scale_y = cell_size / width, cell_size / height
        draw = ImageDraw.Draw(panel)
        points = coordinates[0, index].numpy()
        track_by_phase[phase_order[index]] = points.tolist()
        if index == 0 and mask_box is not None:
            x0, y0, x1, y1 = mask_box
            draw.rectangle(
                (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                outline=(0, 255, 255),
                width=3,
            )
        if index == 0:
            for x, y in query_points:
                x, y = float(x) * scale_x, float(y) * scale_y
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 255, 0), width=3)
        for point_index, (x, y) in enumerate(points):
            if not np.isfinite((x, y)).all():
                continue
            x, y = float(x) * scale_x, float(y) * scale_y
            radius = 5
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 50, 50),
                outline=(255, 255, 255),
                width=1,
            )
            draw.text((x + 6, y - 7), str(point_index), fill=(255, 255, 0))
        draw.rectangle((0, 0, cell_size, 27), fill=(0, 0, 0))
        draw.text((8, 7), phase_order[index], fill=(255, 255, 255))
        panels.append(panel)
    sheet = Image.new("RGB", (cell_size * len(panels), cell_size), color=(30, 30, 30))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * cell_size, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return {
        "overlay": str(output_path),
        "query_points_xy": query_points.tolist(),
        "query_mask_box_xyxy": mask_box,
        "track_points_xy_by_phase": track_by_phase,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    weight = args.weight.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().absolute()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if args.input_size <= 0 or args.query_points <= 0:
        raise ValueError("input size and query point count must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Phase D-0 full-head audit requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    weight_sha256 = sha256_file(weight)
    if args.expected_weight_sha256 is not None:
        expected = args.expected_weight_sha256.lower().strip()
        if weight_sha256 != expected:
            raise RuntimeError(
                f"VGGT weight SHA256 mismatch: actual={weight_sha256}, expected={expected}"
            )

    info = read_json(dataset_root / "meta/info.json")
    episodes = read_jsonl(dataset_root / "meta/episodes.jsonl")
    episode_indices = {int(row["episode_index"]) for row in episodes}
    if args.episode_index not in episode_indices:
        raise ValueError(f"Episode {args.episode_index} is not in dataset")
    video_keys = resolve_video_keys(info)
    chunks_size = int(info.get("chunks_size", 1000))
    parquet = episode_path(
        dataset_root,
        str(info["data_path"]),
        args.episode_index,
        chunks_size,
    )
    gripper = read_gripper_values(parquet)
    phases, phase_metadata = infer_phases(
        gripper,
        fps=float(info["fps"]),
        threshold=args.close_threshold,
        window=args.confirmation_window,
    )

    from vggt.models.vggt import VGGT

    signature = str(inspect.signature(VGGT))
    model = VGGT(
        img_size=args.input_size,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        feature_only=False,
    )
    state_dict = unwrap_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    output_dir.mkdir(parents=True)
    view_results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="phase_d0_vggt_") as temporary:
        temporary_root = Path(temporary)
        for view, video_key in video_keys.items():
            video = episode_path(
                dataset_root,
                str(info["video_path"]),
                args.episode_index,
                chunks_size,
                video_key=video_key,
            )
            frames = extract_frames(
                video,
                phases.values(),
                temporary_root / view,
            )
            images = []
            resized_pil = []
            for phase in PHASES:
                image = Image.open(frames[phases[phase]]).convert("RGB").resize(
                    (args.input_size, args.input_size),
                    Image.Resampling.BICUBIC,
                )
                resized_pil.append(image)
                array = np.asarray(image, dtype=np.float32) / 255.0
                images.append(torch.from_numpy(array).permute(2, 0, 1))
            cube_masks = [largest_compact_blue_component(image, view) for image in resized_pil]
            query_phase_index = next(
                (index for index, mask in enumerate(cube_masks[:3]) if mask.any()),
                None,
            )
            query_points = (
                sample_query_points(cube_masks[query_phase_index], args.query_points)
                if query_phase_index is not None
                else np.empty((0, 2), dtype=np.float32)
            )
            # TrackHead queries are anchored to the first sequence image.  Start
            # at the first detectable phase and preserve chronological order;
            # never wrap later phases back to an earlier approach frame.
            order = list(range(len(PHASES)))
            if query_phase_index is not None:
                order = order[query_phase_index:]
            image_tensor = torch.stack([images[index] for index in order], dim=0)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            query_tensor = (
                torch.from_numpy(query_points).unsqueeze(0).to(device)
                if len(query_points)
                else None
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), autocast:
                predictions = model(image_tensor, query_points=query_tensor)
            summaries = {key: tensor_summary(value) for key, value in predictions.items()}
            model_phase_order = [PHASES[index] for index in order]
            query_geometry = summarize_query_geometry(
                predictions,
                model_phase_order,
                query_points,
            )
            visualization = None
            if len(query_points) and isinstance(predictions.get("track"), torch.Tensor):
                visualization = save_tracking_overlay(
                    images=[resized_pil[index] for index in order],
                    phase_order=model_phase_order,
                    query_points=query_points,
                    track=predictions["track"],
                    query_mask=cube_masks[order[0]],
                    output_path=output_dir / f"{view}_tracking_overlay.jpg",
                )
            view_results[view] = {
                "original_phase_order": list(PHASES),
                "model_phase_order": model_phase_order,
                "query_phase": (
                    PHASES[query_phase_index] if query_phase_index is not None else None
                ),
                "query_point_count": int(len(query_points)),
                "cube_detectable_by_phase": {
                    phase: bool(mask.any()) for phase, mask in zip(PHASES, cube_masks)
                },
                "output_keys": sorted(predictions),
                "outputs": summaries,
                "query_geometry": query_geometry,
                "visualization": visualization,
            }
            del predictions, image_tensor, query_tensor
            torch.cuda.empty_cache()

    required_geometry = {"depth", "depth_conf", "world_points", "world_points_conf"}
    geometry_pass = all(
        required_geometry.issubset(result["output_keys"])
        and all(
            result["outputs"][key].get("all_finite") is True
            for key in required_geometry
        )
        for result in view_results.values()
    )
    tracking_pass = all(
        result["query_point_count"] > 0
        and {"track", "vis", "conf"}.issubset(result["output_keys"])
        and all(
            result["outputs"][key].get("all_finite") is True
            for key in ("track", "vis", "conf")
        )
        for result in view_results.values()
    )
    task_relative_pass = all(
        result["query_geometry"].get("available") is True
        and result["query_geometry"].get("in_bounds_fraction", 0.0) >= 0.5
        and result["query_geometry"].get("positive_depth_fraction", 0.0) >= 0.5
        and result["query_geometry"].get(
            "first_frame_reprojection_error_px_mean", float("inf")
        )
        <= 15.0
        for result in view_results.values()
    )
    weight_pass = not incompatible.missing_keys and not incompatible.unexpected_keys
    capability_ready = weight_pass and geometry_pass and tracking_pass and task_relative_pass
    payload = {
        "status": "PASS" if capability_ready else "FAIL",
        "audit_execution_status": "PASS",
        "capability_ready": capability_ready,
        "robot_commands_sent": 0,
        "dataset_root": str(dataset_root),
        "episode_index": args.episode_index,
        "phases": phases,
        "phase_metadata": phase_metadata,
        "vggt_signature": signature,
        "weight": str(weight),
        "weight_sha256": weight_sha256,
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "missing_keys_preview": list(incompatible.missing_keys[:20]),
        "unexpected_keys_preview": list(incompatible.unexpected_keys[:20]),
        "view_results": view_results,
        "weight_coverage_pass": weight_pass,
        "geometry_heads_pass": geometry_pass,
        "tracking_head_pass": tracking_pass,
        "task_relative_signal_pass": task_relative_pass,
    }
    result_path = output_dir / "phase_d0_vggt_capability.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")

    print("===== PHASE D-0 VGGT CAPABILITY =====")
    print(f"missing_key_count={len(incompatible.missing_keys)}")
    print(f"unexpected_key_count={len(incompatible.unexpected_keys)}")
    for view, result in view_results.items():
        print(
            f"view={view} query_phase={result['query_phase']} "
            f"query_points={result['query_point_count']} "
            f"output_keys={result['output_keys']}"
        )
        if result["visualization"] is not None:
            print(f"  tracking_overlay={result['visualization']['overlay']}")
        for key in ("depth", "world_points", "track", "vis", "conf"):
            if key in result["outputs"]:
                summary = result["outputs"][key]
                print(
                    f"  {key}: shape={summary.get('shape')} "
                    f"finite_fraction={summary.get('finite_fraction')}"
                )
    print(f"VGGT_WEIGHT_COVERAGE_GATE={'PASS' if weight_pass else 'FAIL'}")
    print(f"VGGT_GEOMETRY_HEAD_GATE={'PASS' if geometry_pass else 'FAIL'}")
    print(f"VGGT_TRACKING_HEAD_GATE={'PASS' if tracking_pass else 'FAIL'}")
    print(f"VGGT_TASK_RELATIVE_SIGNAL_GATE={'PASS' if task_relative_pass else 'FAIL'}")
    print(f"PHASE_D0_CAPABILITY_READY={'PASS' if capability_ready else 'FAIL'}")
    print("PHASE_D0_VGGT_CAPABILITY_AUDIT=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"RESULT={result_path}")


if __name__ == "__main__":
    main()
