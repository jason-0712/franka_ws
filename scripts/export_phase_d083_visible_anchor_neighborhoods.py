#!/usr/bin/env python3
"""Export nearby frames for selecting fully visible, frame-aware SAM2 anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image, ImageDraw

import audit_phase_d06_cube_candidates as d06
import export_phase_d082_oracle_anchor_review as d082


DEFAULT_ANCHORS = "0:release,10:release,31:release,41:release,82:approach"
COLORS = ((255, 220, 0), (0, 255, 255), (255, 0, 255))
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--approach-before-frames", type=int, default=45)
    parser.add_argument("--release-after-frames", type=int, default=45)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--thumbnail-size", type=int, default=230)
    return parser.parse_args()


def neighborhood_indices(
    *,
    phase: str,
    phase_frame: int,
    episode_length: int,
    approach_before: int,
    release_after: int,
    stride: int,
) -> list[int]:
    if phase not in ("approach", "release"):
        raise ValueError(f"Unsupported anchor phase {phase}")
    if episode_length <= 0 or not 0 <= phase_frame < episode_length:
        raise ValueError("Invalid episode/phase frame")
    if approach_before < 0 or release_after < 0 or stride <= 0:
        raise ValueError("Invalid neighborhood configuration")
    if phase == "approach":
        frames = range(phase_frame, max(-1, phase_frame - approach_before - 1), -stride)
    else:
        frames = range(
            phase_frame,
            min(episode_length, phase_frame + release_after + 1),
            stride,
        )
    return sorted(set(int(frame) for frame in frames))


def draw_thumbnail(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    label: str,
    size: int,
    show_candidates: bool,
) -> Image.Image:
    if size < 80:
        raise ValueError("Thumbnail size is too small")
    source = image.convert("RGB")
    panel = source.resize((size, size), BICUBIC)
    draw = ImageDraw.Draw(panel)
    if show_candidates:
        scale_x = size / source.width
        scale_y = size / source.height
        for index, candidate in enumerate(candidates):
            x0, y0, x1, y1 = candidate["bbox_xyxy"]
            color = COLORS[min(index, len(COLORS) - 1)]
            draw.rectangle(
                (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                outline=color,
                width=3 if index == 0 else 2,
            )
            draw.text(
                (x0 * scale_x + 2, max(32, y0 * scale_y - 13)),
                f"C{index + 1}",
                fill=color,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
    draw.rectangle((0, 0, size, 29), fill=(0, 0, 0))
    draw.text((4, 7), label, fill=(255, 255, 255))
    return panel


def build_sheet(rows: list[list[Image.Image]], *, cell: int) -> Image.Image:
    if not rows or not any(rows):
        raise ValueError("No review panels")
    columns = max(len(row) for row in rows)
    sheet = Image.new("RGB", (columns * cell, len(rows) * cell), (20, 20, 20))
    for row_index, row in enumerate(rows):
        for column_index, panel in enumerate(row):
            sheet.paste(panel, (column_index * cell, row_index * cell))
    return sheet


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.input_size <= 1 or args.top_k <= 0 or args.thumbnail_size < 80:
        raise ValueError("Invalid image/export configuration")
    specs = d082.parse_anchor_specs(args.anchors)

    info = d06.read_json(root / "meta/info.json")
    episode_rows = d06.read_jsonl(root / "meta/episodes.jsonl")
    available = {int(row["episode_index"]) for row in episode_rows}
    missing = sorted({episode for episode, _ in specs} - available)
    if missing:
        raise ValueError(f"Episodes absent from dataset: {missing}")
    primary_key = d06.resolve_video_keys(info)["primary"]
    chunks_size = int(info.get("chunks_size", 1000))
    output.mkdir(parents=True)

    records = []
    raw_sheet_rows: list[list[Image.Image]] = []
    candidate_sheet_rows: list[list[Image.Image]] = []
    override_template: dict[str, dict[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="phase_d083_visible_anchor_") as raw_temporary:
        temporary = Path(raw_temporary)
        for episode, phase in specs:
            parquet = d06.episode_path(root, str(info["data_path"]), episode, chunks_size)
            gripper_values = d06.read_gripper_values(parquet)
            phases = d06.infer_phases(
                gripper_values,
                fps=float(info["fps"]),
                threshold=0.5,
                window=3,
            )
            phase_frame = phases[phase]
            indices = neighborhood_indices(
                phase=phase,
                phase_frame=phase_frame,
                episode_length=len(gripper_values),
                approach_before=args.approach_before_frames,
                release_after=args.release_after_frames,
                stride=args.sample_stride,
            )
            video = d06.episode_path(
                root,
                str(info["video_path"]),
                episode,
                chunks_size,
                video_key=primary_key,
            )
            extracted = d06.extract_frames(
                video, indices, temporary / f"ep{episode:04d}_{phase}"
            )
            frame_records = []
            raw_panels = []
            candidate_panels = []
            for frame_index in indices:
                native = Image.open(extracted[frame_index]).convert("RGB")
                model_input = native.resize(
                    (args.input_size, args.input_size), BICUBIC
                )
                candidates = d06.rank_cube_candidates(model_input, "primary", args.top_k)
                stem = f"ep{episode:04d}_{phase}_frame{frame_index:04d}"
                image_path = output / f"{stem}_model_input_{args.input_size}.png"
                model_input.save(image_path)
                offset = frame_index - phase_frame
                label = f"ep{episode:04d} {phase} f={frame_index} ({offset:+d})"
                raw_panels.append(
                    draw_thumbnail(
                        model_input,
                        candidates,
                        label=label,
                        size=args.thumbnail_size,
                        show_candidates=False,
                    )
                )
                candidate_panels.append(
                    draw_thumbnail(
                        model_input,
                        candidates,
                        label=label,
                        size=args.thumbnail_size,
                        show_candidates=True,
                    )
                )
                frame_records.append(
                    {
                        "dataset_frame_index": frame_index,
                        "offset_from_phase_frame": offset,
                        "model_input_image": str(image_path),
                        "model_input_filename": image_path.name,
                        "automatic_candidates": candidates,
                        "manual_visibility": None,
                        "manual_box_xyxy": None,
                    }
                )
            raw_sheet_rows.append(raw_panels)
            candidate_sheet_rows.append(candidate_panels)
            records.append(
                {
                    "episode": episode,
                    "phase": phase,
                    "phase_frame_index": phase_frame,
                    "direction_constraint": (
                        f"dataset_frame_index <= {phase_frame}"
                        if phase == "approach"
                        else f"dataset_frame_index >= {phase_frame}"
                    ),
                    "sampled_frames": frame_records,
                }
            )
            override_template.setdefault(str(episode), {})[phase] = {
                "dataset_frame_index": None,
                "box_xyxy": None,
            }

    raw_sheet_path = output / "phase_d083_visible_anchor_raw_contact_sheet.jpg"
    candidate_sheet_path = output / "phase_d083_visible_anchor_candidates_contact_sheet.jpg"
    build_sheet(raw_sheet_rows, cell=args.thumbnail_size).save(raw_sheet_path, quality=94)
    build_sheet(candidate_sheet_rows, cell=args.thumbnail_size).save(
        candidate_sheet_path, quality=94
    )
    manifest_path = output / "phase_d083_visible_anchor_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "robot_commands_sent": 0,
                "dataset_root": str(root),
                "input_size": args.input_size,
                "anchor_specs": [
                    {"episode": episode, "phase": phase} for episode, phase in specs
                ],
                "selection_rule": (
                    "Choose a frame where the complete physical blue cube is visible, then "
                    "draw a tight model-input xyxy box around only the cube. Do not include "
                    "cyan gripper fingers, box edges, or inferred occluded pixels."
                ),
                "direction_rule": (
                    "Approach anchor must be at/before the approach evaluation frame; "
                    "release anchor must be at/after the release evaluation frame."
                ),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    template_path = output / "visible_anchor_overrides_template.json"
    template_path.write_text(json.dumps(override_template, indent=2) + "\n")

    print("PHASE_D083_VISIBLE_ANCHOR_EXPORT=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"OUTPUT_DIR={output}")
    print(f"RAW_CONTACT_SHEET={raw_sheet_path}")
    print(f"CANDIDATE_CONTACT_SHEET={candidate_sheet_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"VISIBLE_ANCHOR_TEMPLATE={template_path}")


if __name__ == "__main__":
    main()
