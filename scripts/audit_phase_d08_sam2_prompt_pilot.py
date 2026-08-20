#!/usr/bin/env python3
"""Evaluate SAM2.1 box-prompt cube masks on ten primary-camera episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import audit_phase_d06_cube_candidates as d06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--prompt-overrides", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_overrides(path: Path | None, image_size: int) -> dict[int, list[float]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text())
    if not isinstance(payload, dict):
        raise ValueError("prompt override JSON must be an episode-to-box object")
    result = {}
    for raw_episode, raw_box in payload.items():
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError(f"Episode {raw_episode} box must contain x0,y0,x1,y1")
        box = [float(value) for value in raw_box]
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 < image_size and 0 <= y0 < y1 < image_size):
            raise ValueError(f"Episode {raw_episode} has invalid box {box}")
        result[int(raw_episode)] = box
    return result


def extract_continuous_segment(
    video: Path,
    *,
    start: int,
    end: int,
    size: int,
    output: Path,
) -> list[Path]:
    if start < 0 or end < start or size <= 0:
        raise ValueError("Invalid video segment")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    output.mkdir(parents=True, exist_ok=True)
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
            f"select=between(n\\,{start}\\,{end}),scale={size}:{size}",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(output / "%05d.jpg"),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    frames = sorted(output.glob("*.jpg"), key=lambda path: int(path.stem))
    expected = end - start + 1
    if len(frames) != expected:
        raise RuntimeError(f"Expected {expected} frames, extracted {len(frames)}")
    return frames


def mask_metrics(mask_by_phase: dict[str, np.ndarray]) -> dict[str, Any]:
    areas = {}
    centroids = {}
    for phase in d06.PHASES:
        mask = np.asarray(mask_by_phase.get(phase, np.zeros((1, 1), dtype=bool)), dtype=bool)
        area = float(mask.mean())
        areas[phase] = area
        coordinates = np.argwhere(mask)
        centroids[phase] = (
            [float(value) for value in coordinates.mean(axis=0)[::-1]]
            if len(coordinates)
            else None
        )
    positive = [value for value in areas.values() if value > 0]
    ratio = max(positive) / min(positive) if len(positive) == len(areas) else None
    nonempty_fraction = len(positive) / len(areas)
    plausible_fraction = sum(1e-5 <= value <= 0.12 for value in areas.values()) / len(areas)
    return {
        "area_fraction_by_phase": areas,
        "centroid_xy_by_phase": centroids,
        "nonempty_phase_fraction": nonempty_fraction,
        "plausible_area_phase_fraction": plausible_fraction,
        "max_min_area_ratio": ratio,
        "automatic_mask_gate": (
            nonempty_fraction == 1.0
            and plausible_fraction == 1.0
            and ratio is not None
            and ratio <= 15.0
        ),
    }


def draw_mask_panel(
    image: Image.Image,
    mask: np.ndarray,
    *,
    episode: int,
    phase: str,
    prompt_box: list[float] | None,
    prompt_source: str,
    size: int = 250,
) -> Image.Image:
    source = image.convert("RGB")
    width, height = source.size
    panel = source.resize((size, size), Image.Resampling.BICUBIC).convert("RGBA")
    resized_mask = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255)).resize(
        (size, size), Image.Resampling.NEAREST
    )
    tint = Image.new("RGBA", (size, size), (0, 255, 0, 95))
    panel.alpha_composite(Image.composite(tint, Image.new("RGBA", (size, size)), resized_mask))
    mask_array = np.asarray(resized_mask) > 0
    boundary = mask_array & ~(
        np.roll(mask_array, 1, 0)
        & np.roll(mask_array, -1, 0)
        & np.roll(mask_array, 1, 1)
        & np.roll(mask_array, -1, 1)
    )
    boundary_layer = np.zeros((size, size, 4), dtype=np.uint8)
    boundary_layer[boundary] = (0, 255, 0, 255)
    panel.alpha_composite(Image.fromarray(boundary_layer, mode="RGBA"))
    draw = ImageDraw.Draw(panel)
    if prompt_box is not None:
        x0, y0, x1, y1 = prompt_box
        draw.rectangle(
            (x0 * size / width, y0 * size / height, x1 * size / width, y1 * size / height),
            outline=(255, 220, 0, 255),
            width=3,
        )
    draw.rectangle((0, 0, size, 27), fill=(0, 0, 0, 255))
    draw.text((5, 7), f"ep{episode:04d} {phase} {prompt_source}", fill=(255, 255, 255, 255))
    return panel.convert("RGB")


def main() -> None:
    args = parse_args()
    import torch

    root = args.dataset_root.expanduser().resolve(strict=True)
    sam2_repo = args.sam2_repo.expanduser().resolve(strict=True)
    checkpoint = args.sam2_checkpoint.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    overrides = load_prompt_overrides(args.prompt_overrides, args.input_size)

    info = d06.read_json(root / "meta/info.json")
    rows = d06.read_jsonl(root / "meta/episodes.jsonl")
    available = sorted(int(row["episode_index"]) for row in rows)
    selected = d06.choose_episode_indices(available, args.episodes, args.episode_indices)
    primary_key = d06.resolve_video_keys(info)["primary"]
    chunks_size = int(info.get("chunks_size", 1000))

    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        args.model_config,
        str(checkpoint),
        device=args.device,
        apply_postprocessing=False,
    )
    output.mkdir(parents=True)
    episode_results = []
    sheet_rows = []
    csv_rows = []

    with tempfile.TemporaryDirectory(prefix="phase_d08_sam2_") as raw_temporary:
        temporary = Path(raw_temporary)
        for episode in selected:
            parquet = d06.episode_path(root, str(info["data_path"]), episode, chunks_size)
            phases = d06.infer_phases(
                d06.read_gripper_values(parquet),
                fps=float(info["fps"]),
                threshold=0.5,
                window=3,
            )
            start, end = phases["approach"], phases["release"]
            video = d06.episode_path(
                root,
                str(info["video_path"]),
                episode,
                chunks_size,
                video_key=primary_key,
            )
            frames = extract_continuous_segment(
                video,
                start=start,
                end=end,
                size=args.input_size,
                output=temporary / f"ep{episode:04d}",
            )
            first_image = Image.open(frames[0]).convert("RGB")
            prompt_source = "override" if episode in overrides else "automatic"
            if episode in overrides:
                prompt_box = overrides[episode]
            else:
                candidates = d06.rank_cube_candidates(first_image, "primary", 1)
                if not candidates:
                    episode_results.append(
                        {
                            "episode": episode,
                            "prompt_source": prompt_source,
                            "prompt_available": False,
                            "automatic_mask_gate": False,
                            "manual_mask_is_cube": None,
                        }
                    )
                    sheet_rows.append(
                        [
                            d06.draw_candidates(
                                Image.open(frames[phases[phase] - start]).convert("RGB"),
                                view="primary",
                                episode=episode,
                                phase=phase,
                                candidates=[],
                            )
                            for phase in d06.PHASES
                        ]
                    )
                    continue
                prompt_box = [float(value) for value in candidates[0]["bbox_xyxy"]]

            phase_local = {phase: index - start for phase, index in phases.items()}
            masks_by_local_index = {}
            state = predictor.init_state(
                video_path=str(temporary / f"ep{episode:04d}"),
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.asarray(prompt_box, dtype=np.float32),
                )
                for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
                    masks_by_local_index[int(frame_index)] = (
                        mask_logits[0].detach().float().cpu().numpy().squeeze() > 0.0
                    )
            predictor.reset_state(state)

            phase_masks = {
                phase: masks_by_local_index.get(
                    phase_local[phase], np.zeros((args.input_size, args.input_size), dtype=bool)
                )
                for phase in d06.PHASES
            }
            metrics = mask_metrics(phase_masks)
            episode_results.append(
                {
                    "episode": episode,
                    "prompt_source": prompt_source,
                    "prompt_available": True,
                    "prompt_box_xyxy": prompt_box,
                    **metrics,
                    "manual_mask_is_cube": None,
                }
            )
            panels = []
            for phase in d06.PHASES:
                local_index = phase_local[phase]
                panels.append(
                    draw_mask_panel(
                        Image.open(frames[local_index]).convert("RGB"),
                        phase_masks[phase],
                        episode=episode,
                        phase=phase,
                        prompt_box=prompt_box if phase == "approach" else None,
                        prompt_source=prompt_source,
                    )
                )
                csv_rows.append(
                    {
                        "episode": episode,
                        "phase": phase,
                        "prompt_source": prompt_source,
                        "prompt_box_xyxy": prompt_box,
                        "mask_area_fraction": metrics["area_fraction_by_phase"][phase],
                        "mask_centroid_xy": metrics["centroid_xy_by_phase"][phase],
                        "automatic_mask_gate": metrics["automatic_mask_gate"],
                        "manual_mask_is_cube": "",
                    }
                )
            sheet_rows.append(panels)
            torch.cuda.empty_cache()

    cell = 250
    sheet = Image.new("RGB", (cell * len(d06.PHASES), cell * len(sheet_rows)), (20, 20, 20))
    for row_index, row in enumerate(sheet_rows):
        for column_index, panel in enumerate(row):
            sheet.paste(panel, (column_index * cell, row_index * cell))
    sheet_path = output / "primary_sam2_mask_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    csv_path = output / "sam2_mask_phase_metrics.csv"
    fields = [
        "episode",
        "phase",
        "prompt_source",
        "prompt_box_xyxy",
        "mask_area_fraction",
        "mask_centroid_xy",
        "automatic_mask_gate",
        "manual_mask_is_cube",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    passed = sum(bool(result.get("automatic_mask_gate")) for result in episode_results)
    rate = passed / len(selected)
    payload = {
        "status": "PASS",
        "robot_commands_sent": 0,
        "dataset_root": str(root),
        "episode_indices": selected,
        "sam2_repo": str(sam2_repo),
        "sam2_git_commit": subprocess.check_output(
            ["git", "-C", str(sam2_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sam2_checkpoint": str(checkpoint),
        "sam2_checkpoint_sha256": sha256_file(checkpoint),
        "model_config": args.model_config,
        "prompt_override_episodes": sorted(overrides),
        "automatic_mask_gate_rate": rate,
        "automatic_evidence_gate": rate >= 0.8,
        "manual_review_required": True,
        "manual_acceptance_target": "at least 9/10 masks remain on the physical cube",
        "wrist_temporal_tracking_used": False,
        "episode_results": episode_results,
    }
    result_path = output / "phase_d08_sam2_prompt_pilot.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    review_template_path = output / "manual_review_decisions.json"
    review_template_path.write_text(
        json.dumps({str(episode): None for episode in selected}, indent=2) + "\n"
    )
    print("PHASE_D08_SAM2_PROMPT_PILOT=PASS")
    print(f"AUTOMATIC_MASK_GATE_RATE={rate:.6f}")
    print(f"AUTOMATIC_EVIDENCE_GATE={'PASS' if payload['automatic_evidence_gate'] else 'FAIL'}")
    print("MANUAL_REVIEW_REQUIRED=TRUE")
    print("WRIST_TEMPORAL_TRACKING_USED=FALSE")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"OUTPUT_DIR={output}")
    print(f"CONTACT_SHEET={sheet_path}")
    print(f"CSV={csv_path}")
    print(f"RESULT={result_path}")
    print(f"MANUAL_REVIEW_TEMPLATE={review_template_path}")


if __name__ == "__main__":
    main()
