#!/usr/bin/env python3
"""Export failed SAM2 anchor frames for a blinded oracle-box review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image, ImageDraw

import audit_phase_d06_cube_candidates as d06


DEFAULT_ANCHORS = "0:release,10:release,31:release,41:release,82:approach"
COLORS = ((255, 220, 0), (0, 255, 255), (255, 0, 255), (255, 120, 0), (80, 255, 80))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def parse_anchor_specs(raw: str) -> list[tuple[int, str]]:
    result = []
    seen = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            episode_text, phase = item.split(":", 1)
            episode = int(episode_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid anchor specification {item!r}") from exc
        if episode < 0 or phase not in ("approach", "release"):
            raise ValueError(f"Invalid anchor specification {item!r}")
        key = (episode, phase)
        if key in seen:
            raise ValueError(f"Duplicate anchor specification {item!r}")
        seen.add(key)
        result.append(key)
    if not result:
        raise ValueError("At least one anchor is required")
    return result


def scale_box(
    box: list[int | float],
    *,
    source_width: int,
    source_height: int,
    target_size: int,
) -> list[float]:
    if source_width <= 0 or source_height <= 0 or target_size <= 1 or len(box) != 4:
        raise ValueError("Invalid box scaling inputs")
    x0, y0, x1, y1 = (float(value) for value in box)
    return [
        x0 * target_size / source_width,
        y0 * target_size / source_height,
        x1 * target_size / source_width,
        y1 * target_size / source_height,
    ]


def draw_coordinate_grid(image: Image.Image, spacing: int = 50) -> Image.Image:
    panel = image.convert("RGB").copy()
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    for x in range(0, width, spacing):
        draw.line((x, 0, x, height - 1), fill=(255, 255, 255), width=1)
        draw.text((x + 2, 3), str(x), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    for y in range(0, height, spacing):
        draw.line((0, y, width - 1, y), fill=(255, 255, 255), width=1)
        draw.text((3, y + 2), str(y), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return panel


def draw_candidates_on_grid(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    title: str,
) -> Image.Image:
    panel = draw_coordinate_grid(image)
    draw = ImageDraw.Draw(panel)
    for index, candidate in enumerate(candidates):
        color = COLORS[min(index, len(COLORS) - 1)]
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4 if index == 0 else 2)
        draw.text(
            (x0 + 3, max(28, y0 - 16)),
            f"C{index + 1} {candidate['score']:.2f}",
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    draw.rectangle((0, 0, panel.width, 27), fill=(0, 0, 0))
    draw.text((5, 6), title, fill=(255, 255, 255))
    return panel


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    specs = parse_anchor_specs(args.anchors)
    if args.input_size <= 1 or args.top_k <= 0:
        raise ValueError("Invalid input size/top-k")

    info = d06.read_json(root / "meta/info.json")
    episode_rows = d06.read_jsonl(root / "meta/episodes.jsonl")
    available = {int(row["episode_index"]) for row in episode_rows}
    missing = sorted({episode for episode, _ in specs} - available)
    if missing:
        raise ValueError(f"Episodes absent from dataset: {missing}")
    primary_key = d06.resolve_video_keys(info)["primary"]
    chunks_size = int(info.get("chunks_size", 1000))
    output.mkdir(parents=True)

    manifest = []
    review_panels = []
    override_template: dict[str, dict[str, None]] = {}
    with tempfile.TemporaryDirectory(prefix="phase_d082_anchor_") as raw_temporary:
        temporary = Path(raw_temporary)
        for episode, phase in specs:
            parquet = d06.episode_path(root, str(info["data_path"]), episode, chunks_size)
            phases = d06.infer_phases(
                d06.read_gripper_values(parquet),
                fps=float(info["fps"]),
                threshold=0.5,
                window=3,
            )
            frame_index = phases[phase]
            video = d06.episode_path(
                root,
                str(info["video_path"]),
                episode,
                chunks_size,
                video_key=primary_key,
            )
            extracted = d06.extract_frames(
                video,
                [frame_index],
                temporary / f"ep{episode:04d}_{phase}",
            )[frame_index]
            native = Image.open(extracted).convert("RGB")
            model_input = native.resize(
                (args.input_size, args.input_size), Image.Resampling.BICUBIC
            )
            native_candidates = d06.rank_cube_candidates(native, "primary", args.top_k)
            model_candidates = d06.rank_cube_candidates(model_input, "primary", args.top_k)

            stem = f"ep{episode:04d}_{phase}"
            native_path = output / f"{stem}_native.png"
            model_path = output / f"{stem}_model_input_{args.input_size}.png"
            native.save(native_path)
            model_input.save(model_path)
            native_review = draw_candidates_on_grid(
                native,
                native_candidates,
                title=f"{stem} native {native.width}x{native.height}",
            )
            model_review = draw_candidates_on_grid(
                model_input,
                model_candidates,
                title=f"{stem} SAM2 input {args.input_size}x{args.input_size}",
            )
            review_panels.append((native_review, model_review))
            override_template.setdefault(str(episode), {})[phase] = None
            manifest.append(
                {
                    "episode": episode,
                    "phase": phase,
                    "dataset_frame_index": frame_index,
                    "native_size_wh": [native.width, native.height],
                    "model_input_size_wh": [args.input_size, args.input_size],
                    "native_image": str(native_path),
                    "model_input_image": str(model_path),
                    "native_candidates": native_candidates,
                    "model_input_candidates": model_candidates,
                    "native_candidate_boxes_scaled_to_model_input": [
                        scale_box(
                            candidate["bbox_xyxy"],
                            source_width=native.width,
                            source_height=native.height,
                            target_size=args.input_size,
                        )
                        for candidate in native_candidates
                    ],
                    "manual_oracle_box_model_input_xyxy": None,
                }
            )

    cell = args.input_size
    sheet = Image.new("RGB", (cell * 2, cell * len(review_panels)), (20, 20, 20))
    for row, (native_panel, model_panel) in enumerate(review_panels):
        sheet.paste(
            native_panel.resize((cell, cell), Image.Resampling.BICUBIC),
            (0, row * cell),
        )
        sheet.paste(model_panel, (cell, row * cell))
    sheet_path = output / "phase_d082_oracle_anchor_contact_sheet.jpg"
    sheet.save(sheet_path, quality=94)
    manifest_path = output / "phase_d082_oracle_anchor_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "robot_commands_sent": 0,
                "dataset_root": str(root),
                "input_size": args.input_size,
                "anchor_specs": [{"episode": ep, "phase": phase} for ep, phase in specs],
                "coordinate_convention": "model-input pixel xyxy; 0 <= x,y < 518",
                "manual_instruction": "Draw a tight box around only the physical blue cube; do not include gripper or box.",
                "records": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    override_path = output / "oracle_anchor_overrides_template.json"
    override_path.write_text(json.dumps(override_template, indent=2) + "\n")
    print("PHASE_D082_ORACLE_ANCHOR_EXPORT=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"OUTPUT_DIR={output}")
    print(f"CONTACT_SHEET={sheet_path}")
    print(f"MANIFEST={manifest_path}")
    print(f"OVERRIDE_TEMPLATE={override_path}")


if __name__ == "__main__":
    main()
