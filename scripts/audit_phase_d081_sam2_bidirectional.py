#!/usr/bin/env python3
"""Audit bidirectional, two-anchor SAM2.1 cube tracking on primary video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
import audit_phase_d08_sam2_prompt_pilot as d08


ANCHOR_PHASES = ("approach", "release")
REVIEW_LABELS = ("correct", "occluded_abstain", "visible_missing", "wrong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--anchor-overrides", type=Path, default=None)
    parser.add_argument("--visible-anchor-overrides", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--box-padding-px", type=float, default=4.0)
    parser.add_argument("--minimum-agreement-iou", type=float, default=0.15)
    parser.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    return parser.parse_args()


def validate_box(raw_box: Any, image_size: int, context: str) -> list[float]:
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise ValueError(f"{context} box must contain x0,y0,x1,y1")
    box = [float(value) for value in raw_box]
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 < image_size and 0 <= y0 < y1 < image_size):
        raise ValueError(f"{context} has invalid box {box}")
    return box


def load_anchor_overrides(
    path: Path | None, image_size: int
) -> dict[int, dict[str, list[float]]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text())
    if not isinstance(payload, dict):
        raise ValueError("anchor override JSON must be an episode-to-anchor object")
    result = {}
    for raw_episode, raw_anchors in payload.items():
        if not isinstance(raw_anchors, dict):
            raise ValueError(f"Episode {raw_episode} anchors must be an object")
        unexpected = set(raw_anchors) - set(ANCHOR_PHASES)
        if unexpected:
            raise ValueError(f"Episode {raw_episode} has unexpected anchors {unexpected}")
        result[int(raw_episode)] = {
            phase: validate_box(raw_box, image_size, f"Episode {raw_episode} {phase}")
            for phase, raw_box in raw_anchors.items()
        }
    return result


def load_visible_anchor_overrides(
    path: Path | None, image_size: int
) -> dict[int, dict[str, dict[str, Any]]]:
    """Load frame-aware anchors selected from fully visible nearby frames.

    Schema:
      {"0": {"release": {"dataset_frame_index": 205,
                            "box_xyxy": [x0, y0, x1, y1]}}}
    """
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text())
    if not isinstance(payload, dict):
        raise ValueError("visible-anchor override JSON must be an episode object")
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for raw_episode, raw_anchors in payload.items():
        if not isinstance(raw_anchors, dict):
            raise ValueError(f"Episode {raw_episode} anchors must be an object")
        unexpected = set(raw_anchors) - set(ANCHOR_PHASES)
        if unexpected:
            raise ValueError(f"Episode {raw_episode} has unexpected anchors {unexpected}")
        episode = int(raw_episode)
        result[episode] = {}
        for phase, raw_anchor in raw_anchors.items():
            if not isinstance(raw_anchor, dict):
                raise ValueError(f"Episode {raw_episode} {phase} anchor must be an object")
            if set(raw_anchor) != {"dataset_frame_index", "box_xyxy"}:
                raise ValueError(
                    f"Episode {raw_episode} {phase} requires dataset_frame_index and box_xyxy"
                )
            frame_index = raw_anchor["dataset_frame_index"]
            if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
                raise ValueError(f"Episode {raw_episode} {phase} has invalid frame index")
            result[episode][phase] = {
                "dataset_frame_index": frame_index,
                "box_xyxy": validate_box(
                    raw_anchor["box_xyxy"], image_size, f"Episode {raw_episode} {phase}"
                ),
            }
    return result


def validate_visible_anchor_direction(
    phase: str, anchor_frame: int, evaluation_frame: int
) -> None:
    if phase == "approach" and anchor_frame > evaluation_frame:
        raise ValueError(
            f"Approach anchor frame {anchor_frame} must not follow evaluation frame {evaluation_frame}"
        )
    if phase == "release" and anchor_frame < evaluation_frame:
        raise ValueError(
            f"Release anchor frame {anchor_frame} must not precede evaluation frame {evaluation_frame}"
        )


def pad_box(box: list[float], padding: float, image_size: int) -> list[float]:
    if padding < 0 or image_size <= 1:
        raise ValueError("Invalid box padding")
    x0, y0, x1, y1 = box
    return [
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(float(image_size - 1), x1 + padding),
        min(float(image_size - 1), y1 + padding),
    ]


def mask_area(mask: np.ndarray | None) -> float:
    if mask is None:
        return 0.0
    return float(np.asarray(mask, dtype=bool).mean())


def plausible_mask(mask: np.ndarray | None) -> bool:
    area = mask_area(mask)
    return 1e-5 <= area <= 0.12


def mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else None


def fuse_directional_masks(
    forward: np.ndarray | None,
    backward: np.ndarray | None,
    *,
    phase: str,
    minimum_agreement_iou: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if phase not in d06.PHASES or not 0 <= minimum_agreement_iou <= 1:
        raise ValueError("Invalid fusion configuration")
    forward_ok = plausible_mask(forward)
    backward_ok = plausible_mask(backward)
    iou = mask_iou(forward, backward) if forward_ok and backward_ok else None
    preferred = "forward" if phase in ("approach", "pre_grasp", "grasp") else "backward"

    if forward_ok and backward_ok:
        if iou is not None and iou >= minimum_agreement_iou:
            selected = forward if preferred == "forward" else backward
            source = f"agreement_{preferred}"
        elif phase == "approach":
            selected, source = forward, "forward_anchor"
        elif phase == "release":
            selected, source = backward, "backward_anchor"
        else:
            selected, source = None, "abstain_direction_disagreement"
    elif forward_ok:
        selected, source = forward, "forward_only"
    elif backward_ok:
        selected, source = backward, "backward_only"
    else:
        selected, source = None, "abstain_no_plausible_mask"

    return selected, {
        "source": source,
        "abstained": selected is None,
        "forward_plausible": forward_ok,
        "backward_plausible": backward_ok,
        "forward_area_fraction": mask_area(forward),
        "backward_area_fraction": mask_area(backward),
        "direction_iou": iou,
        "fused_area_fraction": mask_area(selected),
    }


def run_direction(
    predictor,
    *,
    video_dir: Path,
    anchor_index: int,
    anchor_box: list[float],
    reverse: bool,
    phase_indices: set[int],
    torch,
) -> dict[int, np.ndarray]:
    state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )
    results = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.add_new_points_or_box(
            state,
            frame_idx=anchor_index,
            obj_id=1,
            box=np.asarray(anchor_box, dtype=np.float32),
        )
        for frame_index, _, mask_logits in predictor.propagate_in_video(
            state,
            start_frame_idx=anchor_index,
            reverse=reverse,
        ):
            frame_index = int(frame_index)
            if frame_index in phase_indices:
                results[frame_index] = (
                    mask_logits[0].detach().float().cpu().numpy().squeeze() > 0.0
                )
    predictor.reset_state(state)
    return results


def boundary(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    raw = np.asarray(mask, dtype=bool)
    eroded = (
        np.roll(raw, 1, 0)
        & np.roll(raw, -1, 0)
        & np.roll(raw, 1, 1)
        & np.roll(raw, -1, 1)
    )
    return raw & ~eroded


def draw_panel(
    image: Image.Image,
    *,
    forward: np.ndarray | None,
    backward: np.ndarray | None,
    fused: np.ndarray | None,
    episode: int,
    phase: str,
    source: str,
    anchor_box: list[float] | None,
    size: int = 250,
) -> Image.Image:
    source_image = image.convert("RGB")
    width, height = source_image.size
    panel = source_image.resize((size, size), Image.Resampling.BICUBIC).convert("RGBA")
    if fused is not None:
        fused_image = Image.fromarray((np.asarray(fused, dtype=np.uint8) * 255)).resize(
            (size, size), Image.Resampling.NEAREST
        )
        tint = Image.new("RGBA", (size, size), (0, 255, 0, 90))
        panel.alpha_composite(
            Image.composite(tint, Image.new("RGBA", (size, size)), fused_image)
        )
    for raw_mask, color in ((forward, (0, 220, 255, 255)), (backward, (255, 0, 255, 255))):
        raw_boundary = boundary(raw_mask)
        if raw_boundary is None:
            continue
        resized = Image.fromarray((raw_boundary.astype(np.uint8) * 255)).resize(
            (size, size), Image.Resampling.NEAREST
        )
        layer = np.zeros((size, size, 4), dtype=np.uint8)
        layer[np.asarray(resized) > 0] = color
        panel.alpha_composite(Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(panel)
    if anchor_box is not None:
        x0, y0, x1, y1 = anchor_box
        draw.rectangle(
            (x0 * size / width, y0 * size / height, x1 * size / width, y1 * size / height),
            outline=(255, 220, 0, 255),
            width=3,
        )
    draw.rectangle((0, 0, size, 29), fill=(0, 0, 0, 255))
    draw.text((4, 5), f"ep{episode:04d} {phase} {source[:18]}", fill=(255, 255, 255, 255))
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
    overrides = load_anchor_overrides(args.anchor_overrides, args.input_size)
    visible_overrides = load_visible_anchor_overrides(
        args.visible_anchor_overrides, args.input_size
    )
    overlap = sorted(set(overrides) & set(visible_overrides))
    if overlap:
        raise ValueError(
            f"Episodes cannot use both fixed and visible anchor overrides: {overlap}"
        )

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

    with tempfile.TemporaryDirectory(prefix="phase_d081_sam2_") as raw_temporary:
        temporary = Path(raw_temporary)
        for episode in selected:
            parquet = d06.episode_path(root, str(info["data_path"]), episode, chunks_size)
            gripper_values = d06.read_gripper_values(parquet)
            phases = d06.infer_phases(
                gripper_values,
                fps=float(info["fps"]),
                threshold=0.5,
                window=3,
            )
            episode_visible = visible_overrides.get(episode, {})
            anchor_dataset_indices = {
                phase: int(
                    episode_visible.get(phase, {}).get(
                        "dataset_frame_index", phases[phase]
                    )
                )
                for phase in ANCHOR_PHASES
            }
            for phase in ANCHOR_PHASES:
                validate_visible_anchor_direction(
                    phase, anchor_dataset_indices[phase], phases[phase]
                )
                if not 0 <= anchor_dataset_indices[phase] < len(gripper_values):
                    raise ValueError(
                        f"Episode {episode} {phase} anchor frame is outside the episode"
                    )
            start = min(phases["approach"], *anchor_dataset_indices.values())
            end = max(phases["release"], *anchor_dataset_indices.values())
            video = d06.episode_path(
                root,
                str(info["video_path"]),
                episode,
                chunks_size,
                video_key=primary_key,
            )
            video_dir = temporary / f"ep{episode:04d}"
            frames = d08.extract_continuous_segment(
                video, start=start, end=end, size=args.input_size, output=video_dir
            )
            phase_local = {phase: index - start for phase, index in phases.items()}
            anchor_local = {
                phase: index - start for phase, index in anchor_dataset_indices.items()
            }
            anchor_boxes = {}
            anchor_sources = {}
            for phase in ANCHOR_PHASES:
                local_index = anchor_local[phase]
                if phase in episode_visible:
                    raw_box = episode_visible[phase]["box_xyxy"]
                    anchor_sources[phase] = "visible_frame_override"
                elif phase in overrides.get(episode, {}):
                    raw_box = overrides[episode][phase]
                    anchor_sources[phase] = "override"
                else:
                    image = Image.open(frames[local_index]).convert("RGB")
                    candidates = d06.rank_cube_candidates(image, "primary", 1)
                    raw_box = (
                        [float(value) for value in candidates[0]["bbox_xyxy"]]
                        if candidates
                        else None
                    )
                    anchor_sources[phase] = "automatic"
                anchor_boxes[phase] = (
                    pad_box(raw_box, args.box_padding_px, args.input_size)
                    if raw_box is not None
                    else None
                )

            phase_indices = set(phase_local.values())
            forward = {}
            backward = {}
            if anchor_boxes["approach"] is not None:
                forward = run_direction(
                    predictor,
                    video_dir=video_dir,
                    anchor_index=anchor_local["approach"],
                    anchor_box=anchor_boxes["approach"],
                    reverse=False,
                    phase_indices=phase_indices,
                    torch=torch,
                )
            if anchor_boxes["release"] is not None:
                backward = run_direction(
                    predictor,
                    video_dir=video_dir,
                    anchor_index=anchor_local["release"],
                    anchor_box=anchor_boxes["release"],
                    reverse=True,
                    phase_indices=phase_indices,
                    torch=torch,
                )

            panels = []
            phase_results = {}
            for phase in d06.PHASES:
                local_index = phase_local[phase]
                fused, metrics = fuse_directional_masks(
                    forward.get(local_index),
                    backward.get(local_index),
                    phase=phase,
                    minimum_agreement_iou=args.minimum_agreement_iou,
                )
                phase_results[phase] = metrics
                anchor_box = (
                    anchor_boxes.get(phase)
                    if anchor_dataset_indices.get(phase) == phases[phase]
                    else None
                )
                panels.append(
                    draw_panel(
                        Image.open(frames[local_index]).convert("RGB"),
                        forward=forward.get(local_index),
                        backward=backward.get(local_index),
                        fused=fused,
                        episode=episode,
                        phase=phase,
                        source=metrics["source"],
                        anchor_box=anchor_box,
                    )
                )
                csv_rows.append(
                    {
                        "episode": episode,
                        "phase": phase,
                        "approach_anchor_source": anchor_sources["approach"],
                        "release_anchor_source": anchor_sources["release"],
                        "approach_anchor_dataset_frame": anchor_dataset_indices["approach"],
                        "release_anchor_dataset_frame": anchor_dataset_indices["release"],
                        **metrics,
                        "manual_review": "",
                    }
                )
            sheet_rows.append(panels)
            episode_results.append(
                {
                    "episode": episode,
                    "anchor_boxes_xyxy": anchor_boxes,
                    "anchor_sources": anchor_sources,
                    "anchor_dataset_frame_indices": anchor_dataset_indices,
                    "evaluation_phase_frame_indices": phases,
                    "extracted_segment_dataset_frames": [start, end],
                    "phase_results": phase_results,
                }
            )
            torch.cuda.empty_cache()

    cell = 250
    sheet = Image.new("RGB", (cell * len(d06.PHASES), cell * len(sheet_rows)), (20, 20, 20))
    for row_index, row in enumerate(sheet_rows):
        for column_index, panel in enumerate(row):
            sheet.paste(panel, (column_index * cell, row_index * cell))
    sheet_path = output / "primary_sam2_bidirectional_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    csv_path = output / "sam2_bidirectional_phase_metrics.csv"
    fields = list(csv_rows[0]) if csv_rows else []
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    total_phases = len(selected) * len(d06.PHASES)
    abstentions = sum(
        result["phase_results"][phase]["abstained"]
        for result in episode_results
        for phase in d06.PHASES
    )
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
        "sam2_checkpoint_sha256": d08.sha256_file(checkpoint),
        "model_config": args.model_config,
        "box_padding_px": args.box_padding_px,
        "minimum_agreement_iou": args.minimum_agreement_iou,
        "anchor_override_episodes": sorted(overrides),
        "visible_anchor_override_episodes": sorted(visible_overrides),
        "structural_non_abstention_rate": 1.0 - abstentions / total_phases,
        "manual_review_required": True,
        "manual_review_labels": list(REVIEW_LABELS),
        "preregistered_manual_gate": {
            "wrong_object_count": 0,
            "visible_phase_accuracy_minimum": 0.9,
            "anchor_accuracy_minimum": 0.9,
            "occluded_abstain_is_allowed": True,
        },
        "wrist_temporal_tracking_used": False,
        "episode_results": episode_results,
    }
    result_path = output / "phase_d081_sam2_bidirectional.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    review_path = output / "manual_bidirectional_review.json"
    review_path.write_text(
        json.dumps(
            {
                "allowed_labels": list(REVIEW_LABELS),
                "episodes": {
                    str(episode): {phase: None for phase in d06.PHASES}
                    for episode in selected
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("PHASE_D081_SAM2_BIDIRECTIONAL=PASS")
    print(f"STRUCTURAL_NON_ABSTENTION_RATE={payload['structural_non_abstention_rate']:.6f}")
    print("SEMANTIC_GATE=MANUAL_REVIEW_REQUIRED")
    print("WRIST_TEMPORAL_TRACKING_USED=FALSE")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"OUTPUT_DIR={output}")
    print(f"CONTACT_SHEET={sheet_path}")
    print(f"CSV={csv_path}")
    print(f"RESULT={result_path}")
    print(f"MANUAL_REVIEW_TEMPLATE={review_path}")


if __name__ == "__main__":
    main()
