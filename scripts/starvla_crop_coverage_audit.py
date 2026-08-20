#!/usr/bin/env python3
"""Audit fixed image-space crops over every episode of a LeRobot dataset.

The audit is read-only and action-free.  It infers five manipulation phases
from the recorded gripper command, extracts the corresponding primary/wrist
video frames with ffmpeg, overlays proposed normalized crops, and writes
contact sheets plus a CSV/JSON report.  No ROS modules are imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterable, Sequence


PHASES = ("approach", "pre_grasp", "grasp", "transport", "release")
DEFAULT_DATASET = Path(
    "/data/hanyu/quest3_franka_real/snkdjn/"
    "quest3_franka_dualcam_replay_94eps_v1"
)


def parse_normalized_box(text: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid crop box: {text}") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Crop must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise argparse.ArgumentTypeError(
            f"Crop coordinates must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1: {values}"
        )
    return values


def normalized_box_to_pixels(
    box: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("Image dimensions must be positive and box must have four values")
    x0, y0, x1, y1 = box
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"Invalid normalized crop: {tuple(box)}")
    left = max(0, min(width - 1, int(math.floor(x0 * width))))
    top = max(0, min(height - 1, int(math.floor(y0 * height))))
    right = max(left + 1, min(width, int(math.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height))))
    return left, top, right, bottom


def first_sustained(
    values: Sequence[float],
    *,
    start: int,
    window: int,
    predicate,
) -> int | None:
    if window <= 0:
        raise ValueError("window must be positive")
    for index in range(max(0, start), max(0, len(values) - window + 1)):
        if all(predicate(float(value)) for value in values[index : index + window]):
            return index
    return None


def infer_phase_indices(
    gripper_values: Sequence[float],
    *,
    fps: float,
    close_threshold: float = 0.5,
    confirmation_window: int = 3,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Infer five representative frames from open=1 / closed=0 commands."""

    values = [float(value) for value in gripper_values]
    if not values:
        raise ValueError("gripper_values must not be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    close_index = first_sustained(
        values,
        start=1,
        window=confirmation_window,
        predicate=lambda value: value <= close_threshold,
    )
    close_fallback = close_index is None
    if close_index is None:
        close_index = max(1, int(round(0.42 * (len(values) - 1))))
    # A release is meaningful only after an observed, sustained close.  If the
    # demonstration never closes, do not reinterpret its initial open run as a
    # release merely because ``close_index`` was filled with a fallback value.
    release_index = None
    if not close_fallback:
        release_index = first_sustained(
            values,
            start=min(len(values) - 1, close_index + confirmation_window),
            window=confirmation_window,
            predicate=lambda value: value > close_threshold,
        )
    release_fallback = release_index is None
    if release_index is None:
        release_index = max(close_index + 1, int(round(0.82 * (len(values) - 1))))
        release_index = min(len(values) - 1, release_index)

    half_second = max(1, int(round(0.5 * fps)))
    approach = max(0, int(round(0.40 * close_index)))
    pre_grasp = max(0, close_index - half_second)
    grasp = min(len(values) - 1, close_index + confirmation_window - 1)
    transport = min(
        len(values) - 1,
        max(grasp, int(round((grasp + release_index) / 2.0))),
    )
    phases = {
        "approach": approach,
        "pre_grasp": pre_grasp,
        "grasp": grasp,
        "transport": transport,
        "release": release_index,
    }
    # Preserve phase order even for very short or malformed demonstrations.
    previous = 0
    for name in PHASES:
        phases[name] = max(previous, min(len(values) - 1, phases[name]))
        previous = phases[name]
    metadata = {
        "close_index": close_index,
        "release_index": release_index,
        "close_fallback": close_fallback,
        "release_fallback": release_fallback,
        "confirmation_window": confirmation_window,
    }
    return phases, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--primary-crop",
        type=parse_normalized_box,
        default=parse_normalized_box("0.20,0.48,0.85,1.00"),
    )
    parser.add_argument(
        "--wrist-crop",
        type=parse_normalized_box,
        default=parse_normalized_box("0.00,0.18,1.00,1.00"),
    )
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--confirmation-window", type=int, default=3)
    parser.add_argument("--episodes-per-sheet", type=int, default=10)
    parser.add_argument("--cell-width", type=int, default=180)
    parser.add_argument("--cell-height", type=int, default=170)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_video_keys(info: dict[str, Any]) -> dict[str, str]:
    video_keys = [
        key
        for key, value in info.get("features", {}).items()
        if value.get("dtype") == "video"
    ]
    primary = [key for key in video_keys if "primary" in key.lower()]
    wrist = [key for key in video_keys if "wrist" in key.lower()]
    if len(primary) != 1 or len(wrist) != 1:
        raise RuntimeError(
            "Expected exactly one primary and one wrist video feature, got "
            f"primary={primary}, wrist={wrist}, all={video_keys}"
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
    values = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return root / template.format(**values)


def read_gripper_values(parquet_path: Path) -> list[float]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read LeRobot parquet files") from exc

    table = pq.read_table(parquet_path)
    names = set(table.column_names)
    if "action.gripper" in names:
        raw = table["action.gripper"].to_pylist()
        return [float(value[0] if isinstance(value, list) else value) for value in raw]
    if "action" not in names:
        raise KeyError(
            f"Neither action.gripper nor action exists in {parquet_path}; columns={sorted(names)}"
        )
    actions = table["action"].to_pylist()
    values = []
    for action in actions:
        if not isinstance(action, (list, tuple)) or not action:
            raise ValueError(f"Malformed action in {parquet_path}: {action!r}")
        values.append(float(action[-1]))
    return values


def extract_video_frames(
    video_path: Path,
    frame_indices: Iterable[int],
    output_dir: Path,
) -> dict[int, Path]:
    indices = sorted(set(int(value) for value in frame_indices))
    if not indices:
        raise ValueError("At least one frame index is required")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the crop audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    pattern = output_dir / "selected_%03d.png"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select={expression}",
        "-fps_mode",
        "vfr",
        str(pattern),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}: {completed.stderr.strip()}"
        )
    outputs = sorted(output_dir.glob("selected_*.png"))
    if len(outputs) != len(indices):
        raise RuntimeError(
            f"Expected {len(indices)} frames from {video_path}, got {len(outputs)}"
        )
    return dict(zip(indices, outputs, strict=True))


def blue_cube_retention(image, crop_box: Sequence[float], view_name: str) -> dict[str, Any]:
    """Heuristic diagnostic only; it is not used as a training mask."""

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for blue-cube coverage diagnostics") from exc
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (
        (blue >= 75)
        & (green >= 65)
        & ((blue - red) >= 22)
        & ((green - red) >= 15)
    )
    height, width = mask.shape
    # Exclude the blue laboratory floor in the distant upper part of primary.
    envelope_top = int((0.44 if view_name == "primary" else 0.12) * height)
    envelope = np.zeros_like(mask, dtype=bool)
    envelope[envelope_top:, :] = True
    candidate = mask & envelope
    total = int(candidate.sum())
    x0, y0, x1, y1 = normalized_box_to_pixels(crop_box, width, height)
    retained = int(candidate[y0:y1, x0:x1].sum())
    detectable = total >= max(12, int(0.00003 * width * height))
    return {
        "blue_like_pixels": total,
        "retained_blue_like_pixels": retained,
        "retention": (retained / total if total else None),
        "detectable": detectable,
        "diagnostic_only": True,
    }


def make_cell(image, crop_box, title: str, *, width: int, height: int):
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_height = 18
    full_height = max(50, int((height - title_height) * 0.62))
    crop_height = height - title_height - full_height
    full_preview = ImageOps.contain(image.convert("RGB"), (width, full_height))
    full_x = (width - full_preview.width) // 2
    full_y = title_height + (full_height - full_preview.height) // 2
    canvas.paste(full_preview, (full_x, full_y))
    x0, y0, x1, y1 = normalized_box_to_pixels(crop_box, image.width, image.height)
    scale_x = full_preview.width / image.width
    scale_y = full_preview.height / image.height
    draw.rectangle(
        (
            full_x + x0 * scale_x,
            full_y + y0 * scale_y,
            full_x + x1 * scale_x,
            full_y + y1 * scale_y,
        ),
        outline=(255, 30, 30),
        width=2,
    )
    cropped = image.crop((x0, y0, x1, y1))
    crop_preview = ImageOps.fit(cropped.convert("RGB"), (width, crop_height))
    canvas.paste(crop_preview, (0, title_height + full_height))
    draw.text((3, 2), title, fill=(0, 0, 0))
    return canvas


def save_sheet(
    cells: list[list[Any]],
    episode_indices: list[int],
    path: Path,
    *,
    cell_width: int,
    cell_height: int,
    jpeg_quality: int,
) -> None:
    from PIL import Image, ImageDraw

    if not cells:
        return
    columns = len(PHASES) * len(("primary", "wrist"))
    header_height = 28
    label_width = 72
    canvas = Image.new(
        "RGB",
        (label_width + columns * cell_width, header_height + len(cells) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for phase_index, phase in enumerate(PHASES):
        for view_index, view in enumerate(("primary", "wrist")):
            column = phase_index * 2 + view_index
            draw.text(
                (label_width + column * cell_width + 3, 7),
                f"{phase}/{view[0].upper()}",
                fill=(0, 0, 0),
            )
    for row_index, (episode_index, row_cells) in enumerate(zip(episode_indices, cells)):
        y = header_height + row_index * cell_height
        draw.text((4, y + 8), f"ep {episode_index:03d}", fill=(0, 0, 0))
        for column, cell in enumerate(row_cells):
            canvas.paste(cell, (label_width + column * cell_width, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=jpeg_quality, optimize=True)


def validate_dataset(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset root: {root}")
    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise RuntimeError(
            f"Episode metadata mismatch: info={info.get('total_episodes')} jsonl={len(episodes)}"
        )
    return info, episodes, resolve_video_keys(info)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    info, episodes, video_keys = validate_dataset(dataset_root)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("/data/hanyu/starVLA_runs") / time.strftime(
            "replay94_crop_coverage_audit_%Y%m%d_%H%M%S"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_size = int(info.get("chunks_size", 1000))
    fps = float(info["fps"])
    data_template = str(info["data_path"])
    video_template = str(info["video_path"])
    crop_by_view = {"primary": args.primary_crop, "wrist": args.wrist_crop}
    rows: list[dict[str, Any]] = []
    fallback_counts = {"close": 0, "release": 0}
    sheet_cells: list[list[Any]] = []
    sheet_episodes: list[int] = []
    sheet_index = 0

    with tempfile.TemporaryDirectory(prefix="starvla_crop_audit_") as temporary:
        temporary_root = Path(temporary)
        for episode_meta in episodes:
            episode_index = int(episode_meta["episode_index"])
            parquet = episode_path(
                dataset_root, data_template, episode_index, chunks_size
            )
            if not parquet.is_file():
                raise FileNotFoundError(parquet)
            gripper = read_gripper_values(parquet)
            phases, phase_meta = infer_phase_indices(
                gripper,
                fps=fps,
                close_threshold=args.close_threshold,
                confirmation_window=args.confirmation_window,
            )
            fallback_counts["close"] += int(phase_meta["close_fallback"])
            fallback_counts["release"] += int(phase_meta["release_fallback"])
            frames_by_view = {}
            for view_name, video_key in video_keys.items():
                video = episode_path(
                    dataset_root,
                    video_template,
                    episode_index,
                    chunks_size,
                    video_key=video_key,
                )
                if not video.is_file() or video.stat().st_size == 0:
                    raise FileNotFoundError(video)
                frames_by_view[view_name] = extract_video_frames(
                    video,
                    phases.values(),
                    temporary_root / f"episode_{episode_index:06d}" / view_name,
                )

            row_cells = []
            from PIL import Image

            for phase_name in PHASES:
                frame_index = phases[phase_name]
                for view_name in ("primary", "wrist"):
                    frame_path = frames_by_view[view_name][frame_index]
                    with Image.open(frame_path) as opened:
                        image = opened.convert("RGB")
                    diagnostic = blue_cube_retention(
                        image, crop_by_view[view_name], view_name
                    )
                    rows.append(
                        {
                            "episode_index": episode_index,
                            "phase": phase_name,
                            "frame_index": frame_index,
                            "view": view_name,
                            "image_width": image.width,
                            "image_height": image.height,
                            "crop": ",".join(str(value) for value in crop_by_view[view_name]),
                            **diagnostic,
                            "close_fallback": phase_meta["close_fallback"],
                            "release_fallback": phase_meta["release_fallback"],
                        }
                    )
                    cell = make_cell(
                        image,
                        crop_by_view[view_name],
                        f"f{frame_index:04d}",
                        width=args.cell_width,
                        height=args.cell_height,
                    )
                    row_cells.append(cell)
                    if args.keep_frames:
                        destination = (
                            output_dir
                            / "selected_frames"
                            / f"episode_{episode_index:06d}"
                            / f"{phase_name}_{view_name}.png"
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        image.save(destination)
            sheet_cells.append(row_cells)
            sheet_episodes.append(episode_index)
            if len(sheet_cells) >= args.episodes_per_sheet:
                sheet_path = output_dir / "contact_sheets" / f"sheet_{sheet_index:03d}.jpg"
                save_sheet(
                    sheet_cells,
                    sheet_episodes,
                    sheet_path,
                    cell_width=args.cell_width,
                    cell_height=args.cell_height,
                    jpeg_quality=args.jpeg_quality,
                )
                print(
                    f"CROP_AUDIT_SHEET={sheet_path} episodes={sheet_episodes[0]}-{sheet_episodes[-1]}",
                    flush=True,
                )
                sheet_cells, sheet_episodes = [], []
                sheet_index += 1
        if sheet_cells:
            sheet_path = output_dir / "contact_sheets" / f"sheet_{sheet_index:03d}.jpg"
            save_sheet(
                sheet_cells,
                sheet_episodes,
                sheet_path,
                cell_width=args.cell_width,
                cell_height=args.cell_height,
                jpeg_quality=args.jpeg_quality,
            )
            print(
                f"CROP_AUDIT_SHEET={sheet_path} episodes={sheet_episodes[0]}-{sheet_episodes[-1]}",
                flush=True,
            )

    csv_path = output_dir / "crop_coverage.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    retention_by_view = {}
    for view_name in ("primary", "wrist"):
        detectable = [
            float(row["retention"])
            for row in rows
            if row["view"] == view_name
            and row["detectable"]
            and row["retention"] is not None
        ]
        retention_by_view[view_name] = {
            "detectable_samples": len(detectable),
            "minimum": min(detectable) if detectable else None,
            "mean": sum(detectable) / len(detectable) if detectable else None,
            "below_0_98": sum(value < 0.98 for value in detectable),
        }
    summary = {
        "status": "PASS",
        "robot_commands_sent": 0,
        "dataset_root": str(dataset_root),
        "episodes": len(episodes),
        "sampled_frames_per_episode": len(PHASES) * 2,
        "total_samples": len(rows),
        "fps": fps,
        "video_keys": video_keys,
        "crops": {key: list(value) for key, value in crop_by_view.items()},
        "crop_area_fraction": {
            key: (value[2] - value[0]) * (value[3] - value[1])
            for key, value in crop_by_view.items()
        },
        "phase_fallback_counts": fallback_counts,
        "blue_cube_retention_diagnostic": retention_by_view,
        "manual_review_required": [
            "box remains visible through release",
            "gripper remains visible through grasp and transport",
            "crop does not remove valid front/back cube placements",
        ],
        "automatic_retention_is_not_a_training_mask": True,
    }
    json_path = output_dir / "crop_coverage_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("STARVLA_CROP_COVERAGE_AUDIT=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"SUMMARY={json_path}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
