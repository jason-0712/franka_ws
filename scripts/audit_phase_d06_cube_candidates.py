#!/usr/bin/env python3
"""Audit blue-cube localization candidates before task-relative VGGT training.

This is an offline, CPU-only and robot-command-free audit.  It deliberately
keeps the top candidates instead of treating a color heuristic as ground truth.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


PHASES = ("approach", "pre_grasp", "grasp", "transport", "release")
COLORS = ((0, 255, 0), (255, 200, 0), (255, 0, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--top-k", type=int, default=3)
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
    result = {}
    for view in ("primary", "wrist"):
        matches = [key for key in keys if view in key.lower()]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {view} video key, got {matches}")
        result[view] = matches[0]
    return result


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
        raw = table["action.gripper"].to_pylist()
        return [float(value[0] if isinstance(value, list) else value) for value in raw]
    if "action" not in table.column_names:
        raise KeyError(f"No action/gripper column in {path}")
    values = []
    for action in table["action"].to_pylist():
        if not isinstance(action, (list, tuple)) or not action:
            raise ValueError(f"Malformed action row: {action!r}")
        values.append(float(action[-1]))
    return values


def first_sustained(values, *, start: int, window: int, predicate) -> int | None:
    for index in range(max(0, start), max(0, len(values) - window + 1)):
        if all(predicate(float(value)) for value in values[index : index + window]):
            return index
    return None


def infer_phases(
    values: list[float], *, fps: float, threshold: float, window: int
) -> dict[str, int]:
    if not values or fps <= 0 or window <= 0:
        raise ValueError("Invalid phase inputs")
    close = first_sustained(
        values, start=1, window=window, predicate=lambda value: value <= threshold
    )
    observed_close = close is not None
    if close is None:
        close = max(1, int(round(0.42 * (len(values) - 1))))
    release = None
    if observed_close:
        release = first_sustained(
            values,
            start=min(len(values) - 1, close + window),
            window=window,
            predicate=lambda value: value > threshold,
        )
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
    return phases


def extract_frames(video: Path, indices: Iterable[int], output: Path) -> dict[int, Path]:
    indices = sorted(set(int(index) for index in indices))
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    output.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    completed = subprocess.run(
        [
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
            str(output / "frame_%03d.png"),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    paths = sorted(output.glob("frame_*.png"))
    if len(paths) != len(indices):
        raise RuntimeError(f"Expected {len(indices)} frames, found {len(paths)}")
    return dict(zip(indices, paths, strict=True))


def blue_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (
        (blue >= 72)
        & (green >= 55)
        & ((blue - red) >= 24)
        & ((green - red) >= 12)
        & ((blue - green) >= -8)
    )


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    height, width = mask.shape
    for raw_y, raw_x in np.argwhere(mask):
        y, x = int(raw_y), int(raw_x)
        if visited[y, x]:
            continue
        queue = deque(((y, x),))
        visited[y, x] = True
        coordinates = []
        while queue:
            cy, cx = queue.popleft()
            coordinates.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
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
        components.append(np.asarray(coordinates, dtype=np.int32))
    return components


def rank_cube_candidates(image: Image.Image, view: str, top_k: int) -> list[dict[str, Any]]:
    if view not in ("primary", "wrist") or top_k <= 0:
        raise ValueError("Invalid view/top_k")
    mask = blue_mask(image)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = mask.shape
    targets = {"primary": 0.0006, "wrist": 0.0025}
    candidates = []
    for coordinates in connected_components(mask):
        area = len(coordinates)
        fraction = area / (height * width)
        if area < 12 or fraction > 0.012:
            continue
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0)
        box_width, box_height = x1 - x0 + 1, y1 - y0 + 1
        aspect = min(box_width, box_height) / max(box_width, box_height)
        fill = area / (box_width * box_height)
        center_x = float((x0 + x1) / (2 * width))
        center_y = float((y0 + y1) / (2 * height))
        if aspect < 0.18 or fill < 0.12:
            continue
        in_workspace = True
        if view == "primary":
            in_workspace = 0.18 <= center_x <= 0.86 and 0.56 <= center_y <= 0.96
        if not in_workspace:
            continue
        pixels = rgb[coordinates[:, 0], coordinates[:, 1]]
        blue_strength = float(
            np.clip(((pixels[:, 2] - pixels[:, 0]) / 180.0).mean(), 0.0, 1.0)
        )
        size_score = math.exp(-abs(math.log(max(fraction, 1e-9) / targets[view])))
        score = 0.32 * aspect + 0.22 * min(fill, 1.0) + 0.28 * size_score + 0.18 * blue_strength
        candidates.append(
            {
                "score": float(score),
                "area_pixels": int(area),
                "area_fraction": float(fraction),
                "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "center_xy": [center_x * width, center_y * height],
                "center_normalized_xy": [center_x, center_y],
                "aspect_score": float(aspect),
                "fill_fraction": float(fill),
                "size_score": float(size_score),
                "blue_strength": blue_strength,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def draw_candidates(
    image: Image.Image,
    *,
    view: str,
    episode: int,
    phase: str,
    candidates: list[dict[str, Any]],
    cell_size: int = 250,
) -> Image.Image:
    source = image.convert("RGB")
    width, height = source.size
    panel = source.resize((cell_size, cell_size), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(panel)
    scale_x, scale_y = cell_size / width, cell_size / height
    for rank, candidate in enumerate(candidates):
        color = COLORS[min(rank, len(COLORS) - 1)]
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        box = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        draw.rectangle(box, outline=color, width=4 if rank == 0 else 2)
        draw.text((box[0] + 3, max(28, box[1] - 13)), f"{rank + 1}:{candidate['score']:.2f}", fill=color)
    draw.rectangle((0, 0, cell_size, 25), fill=(0, 0, 0))
    draw.text((5, 6), f"ep{episode:04d} {phase} {view}", fill=(255, 255, 255))
    if not candidates:
        draw.text((8, 35), "NO CANDIDATE", fill=(255, 50, 50))
    return panel


def choose_episode_indices(available: list[int], count: int, explicit: str | None) -> list[int]:
    if explicit:
        selected = [int(value.strip()) for value in explicit.split(",") if value.strip()]
        missing = sorted(set(selected).difference(available))
        if missing:
            raise ValueError(f"Unknown episodes: {missing}")
        return selected
    if count <= 0:
        raise ValueError("episodes must be positive")
    positions = np.linspace(0, len(available) - 1, min(count, len(available)), dtype=int)
    return [available[index] for index in dict.fromkeys(positions)]


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.input_size <= 0 or args.top_k <= 0:
        raise ValueError("input-size/top-k must be positive")
    info = read_json(root / "meta/info.json")
    episode_rows = read_jsonl(root / "meta/episodes.jsonl")
    available = sorted(int(row["episode_index"]) for row in episode_rows)
    selected = choose_episode_indices(available, args.episodes, args.episode_indices)
    keys = resolve_video_keys(info)
    chunks_size = int(info.get("chunks_size", 1000))
    output.mkdir(parents=True)
    records = []
    panels: dict[str, list[list[Image.Image]]] = {"primary": [], "wrist": []}

    with tempfile.TemporaryDirectory(prefix="phase_d06_cube_") as raw_temporary:
        temporary = Path(raw_temporary)
        for episode in selected:
            parquet = episode_path(root, str(info["data_path"]), episode, chunks_size)
            phases = infer_phases(
                read_gripper_values(parquet),
                fps=float(info["fps"]),
                threshold=args.close_threshold,
                window=args.confirmation_window,
            )
            for view, video_key in keys.items():
                video = episode_path(
                    root,
                    str(info["video_path"]),
                    episode,
                    chunks_size,
                    video_key=video_key,
                )
                frames = extract_frames(video, phases.values(), temporary / f"ep{episode}_{view}")
                row_panels = []
                for phase in PHASES:
                    image = Image.open(frames[phases[phase]]).convert("RGB").resize(
                        (args.input_size, args.input_size), Image.Resampling.BICUBIC
                    )
                    candidates = rank_cube_candidates(image, view, args.top_k)
                    row_panels.append(
                        draw_candidates(
                            image,
                            view=view,
                            episode=episode,
                            phase=phase,
                            candidates=candidates,
                        )
                    )
                    if not candidates:
                        records.append(
                            {
                                "episode": episode,
                                "view": view,
                                "phase": phase,
                                "rank": None,
                                "manual_is_cube": "",
                            }
                        )
                    for rank, candidate in enumerate(candidates, start=1):
                        records.append(
                            {
                                "episode": episode,
                                "view": view,
                                "phase": phase,
                                "rank": rank,
                                **candidate,
                                "manual_is_cube": "",
                            }
                        )
                panels[view].append(row_panels)

    for view, rows in panels.items():
        cell = rows[0][0].size[0]
        sheet = Image.new("RGB", (cell * len(PHASES), cell * len(rows)), (25, 25, 25))
        for row_index, row in enumerate(rows):
            for column_index, panel in enumerate(row):
                sheet.paste(panel, (column_index * cell, row_index * cell))
        sheet.save(output / f"{view}_cube_candidate_contact_sheet.jpg", quality=91)

    csv_path = output / "cube_candidates_manual_review.csv"
    fields = [
        "episode",
        "view",
        "phase",
        "rank",
        "score",
        "area_pixels",
        "area_fraction",
        "bbox_xyxy",
        "center_xy",
        "center_normalized_xy",
        "aspect_score",
        "fill_fraction",
        "size_score",
        "blue_strength",
        "manual_is_cube",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    top1_count = sum(record.get("rank") == 1 for record in records)
    expected_samples = len(selected) * len(keys) * len(PHASES)
    payload = {
        "status": "PASS",
        "robot_commands_sent": 0,
        "dataset_root": str(root),
        "episode_indices": selected,
        "views": keys,
        "phases": list(PHASES),
        "expected_samples": expected_samples,
        "samples_with_top1_candidate": top1_count,
        "automatic_candidate_coverage": top1_count / expected_samples,
        "manual_review_required": True,
        "manual_acceptance_target": {
            "primary_top1_precision": 0.90,
            "wrist_is_single_frame_only": True,
            "cross_time_wrist_tracking_is_forbidden": True,
        },
    }
    result = output / "phase_d06_cube_candidate_audit.json"
    result.write_text(json.dumps(payload, indent=2) + "\n")
    print("PHASE_D06_CUBE_CANDIDATE_AUDIT=PASS")
    print(f"EPISODES={selected}")
    print(f"AUTOMATIC_CANDIDATE_COVERAGE={payload['automatic_candidate_coverage']:.6f}")
    print("MANUAL_REVIEW_REQUIRED=TRUE")
    print("CROSS_TIME_WRIST_TRACKING=FORBIDDEN")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"OUTPUT_DIR={output}")
    print(f"PRIMARY_SHEET={output / 'primary_cube_candidate_contact_sheet.jpg'}")
    print(f"WRIST_SHEET={output / 'wrist_cube_candidate_contact_sheet.jpg'}")
    print(f"CSV={csv_path}")
    print(f"RESULT={result}")


if __name__ == "__main__":
    main()
