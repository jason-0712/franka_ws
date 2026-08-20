#!/usr/bin/env python3
"""Select a primary-camera cube query by VGGT temporal task role.

Candidate identity is not inferred from color alone.  A cube candidate should
remain nearly static before grasp and move after grasp.  This offline audit
does not train a policy and sends no robot commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_phase_d0_vggt_capability as d0
import audit_phase_d06_cube_candidates as d06


COLORS = ((0, 255, 0), (255, 190, 0), (255, 0, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--points-per-candidate", type=int, default=5)
    parser.add_argument("--max-early-motion-px", type=float, default=20.0)
    parser.add_argument("--min-late-motion-px", type=float, default=8.0)
    parser.add_argument("--min-in-bounds-fraction", type=float, default=0.8)
    return parser.parse_args()


def points_for_candidate(
    mask: np.ndarray,
    candidate: dict[str, Any],
    count: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    x0, y0, x1, y1 = candidate["bbox_xyxy"]
    selected = np.zeros_like(mask, dtype=bool)
    selected[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    points = d0.sample_query_points(selected, count)
    if len(points):
        return points
    return np.asarray(
        [[(x0 + x1) / 2.0, (y0 + y1) / 2.0]], dtype=np.float32
    )


def temporal_role_metrics(
    track: torch.Tensor,
    *,
    width: int,
    height: int,
    candidate_score: float,
) -> dict[str, Any]:
    """Score static-before-grasp and moving-after-grasp behavior."""

    values = track.detach().float().cpu()
    if values.ndim != 3 or values.shape[0] < 5 or values.shape[-1] != 2:
        raise ValueError(f"Expected track [S>=5,N,2], got {tuple(values.shape)}")
    finite = torch.isfinite(values).all(dim=-1)
    in_bounds = (
        finite
        & (values[..., 0] >= 0)
        & (values[..., 0] <= width - 1)
        & (values[..., 1] >= 0)
        & (values[..., 1] <= height - 1)
    )
    centers = []
    dispersions = []
    for phase in range(5):
        valid = finite[phase]
        if not valid.any():
            centers.append(torch.tensor([float("nan"), float("nan")]))
            dispersions.append(float("inf"))
            continue
        center = values[phase, valid].median(dim=0).values
        centers.append(center)
        dispersions.append(
            float(torch.linalg.vector_norm(values[phase, valid] - center, dim=-1).median().item())
        )
    centers_tensor = torch.stack(centers)

    def distance(first: int, second: int) -> float:
        value = torch.linalg.vector_norm(centers_tensor[second] - centers_tensor[first])
        return float(value.item())

    approach_to_pregrasp = distance(0, 1)
    pregrasp_to_grasp = distance(1, 2)
    grasp_to_transport = distance(2, 3)
    grasp_to_release = distance(2, 4)
    early_motion = max(approach_to_pregrasp, pregrasp_to_grasp)
    late_motion = max(grasp_to_transport, grasp_to_release)
    static_score = math.exp(-max(early_motion, 0.0) / 12.0) if math.isfinite(early_motion) else 0.0
    movement_score = 1.0 - math.exp(-max(late_motion, 0.0) / 15.0) if math.isfinite(late_motion) else 0.0
    in_bounds_fraction = float(in_bounds.float().mean().item())
    dispersion = float(np.median(dispersions))
    coherence_score = math.exp(-max(dispersion, 0.0) / 12.0) if math.isfinite(dispersion) else 0.0
    role_score = (
        0.18 * float(candidate_score)
        + 0.30 * static_score
        + 0.34 * movement_score
        + 0.10 * in_bounds_fraction
        + 0.08 * coherence_score
    )
    return {
        "role_score": role_score,
        "candidate_score": float(candidate_score),
        "approach_to_pregrasp_px": approach_to_pregrasp,
        "pregrasp_to_grasp_px": pregrasp_to_grasp,
        "grasp_to_transport_px": grasp_to_transport,
        "grasp_to_release_px": grasp_to_release,
        "early_motion_px": early_motion,
        "late_motion_px": late_motion,
        "in_bounds_fraction": in_bounds_fraction,
        "median_point_dispersion_px": dispersion,
        "phase_centers_xy": [[float(value) for value in center.tolist()] for center in centers_tensor],
        "track_points_xy": values.tolist(),
    }


def passes_role_gate(
    metrics: dict[str, Any],
    *,
    max_early: float,
    min_late: float,
    min_in_bounds: float,
) -> bool:
    return (
        math.isfinite(metrics["early_motion_px"])
        and math.isfinite(metrics["late_motion_px"])
        and metrics["early_motion_px"] <= max_early
        and metrics["late_motion_px"] >= min_late
        and metrics["in_bounds_fraction"] >= min_in_bounds
    )


def draw_episode_overlay(
    images: list[Image.Image],
    candidates: list[dict[str, Any]],
    candidate_points: list[np.ndarray],
    tracks: list[torch.Tensor],
    metrics: list[dict[str, Any]],
    selected_index: int,
    episode: int,
    cell_size: int = 250,
) -> list[Image.Image]:
    panels = []
    for phase_index, (phase, image) in enumerate(zip(d0.PHASES, images)):
        width, height = image.size
        panel = image.convert("RGB").resize((cell_size, cell_size), Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(panel)
        sx, sy = cell_size / width, cell_size / height
        for candidate_index, (candidate, points, track, metric) in enumerate(
            zip(candidates, candidate_points, tracks, metrics)
        ):
            color = COLORS[min(candidate_index, len(COLORS) - 1)]
            if phase_index == 0:
                x0, y0, x1, y1 = candidate["bbox_xyxy"]
                draw.rectangle(
                    (x0 * sx, y0 * sy, x1 * sx, y1 * sy),
                    outline=color,
                    width=5 if candidate_index == selected_index else 2,
                )
                for x, y in points:
                    x, y = float(x) * sx, float(y) * sy
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            for x, y in track[phase_index].detach().float().cpu().numpy():
                if not np.isfinite((x, y)).all():
                    continue
                x, y = float(x) * sx, float(y) * sy
                radius = 5 if candidate_index == selected_index else 3
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            center = metric["phase_centers_xy"][phase_index]
            if np.isfinite(center).all():
                x, y = center[0] * sx, center[1] * sy
                draw.line((x - 7, y, x + 7, y), fill=(255, 255, 255), width=2)
                draw.line((x, y - 7, x, y + 7), fill=(255, 255, 255), width=2)
        draw.rectangle((0, 0, cell_size, 27), fill=(0, 0, 0))
        draw.text((5, 7), f"ep{episode:04d} {phase}", fill=(255, 255, 255))
        if phase_index == 0 and candidates:
            summary = metrics[selected_index]
            draw.rectangle((0, cell_size - 27, cell_size, cell_size), fill=(0, 0, 0))
            draw.text(
                (4, cell_size - 20),
                f"selected={selected_index + 1} early={summary['early_motion_px']:.1f} late={summary['late_motion_px']:.1f}",
                fill=COLORS[min(selected_index, len(COLORS) - 1)],
            )
        panels.append(panel)
    return panels


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    weight = args.weight.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase D-0.7 requires an available CUDA device")
    if d0.sha256_file(weight) != args.expected_weight_sha256.lower().strip():
        raise RuntimeError("VGGT weight SHA256 mismatch")

    info = d06.read_json(root / "meta/info.json")
    episode_rows = d06.read_jsonl(root / "meta/episodes.jsonl")
    available = sorted(int(row["episode_index"]) for row in episode_rows)
    selected_episodes = d06.choose_episode_indices(
        available, args.episodes, args.episode_indices
    )
    video_key = d06.resolve_video_keys(info)["primary"]
    chunks_size = int(info.get("chunks_size", 1000))

    from vggt.models.vggt import VGGT

    model = VGGT(
        img_size=args.input_size,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=True,
        feature_only=False,
    )
    state = d0.unwrap_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    incompatible = model.load_state_dict(state, strict=False)
    missing_track_keys = [key for key in incompatible.missing_keys if key.startswith(("aggregator.", "track_head."))]
    if missing_track_keys:
        raise RuntimeError(f"Missing VGGT tracking weights: {missing_track_keys[:10]}")
    model = model.to(device).eval()
    output.mkdir(parents=True)

    episode_results = []
    rows_for_sheet = []
    csv_records = []
    with tempfile.TemporaryDirectory(prefix="phase_d07_tracking_") as raw_temporary:
        temporary = Path(raw_temporary)
        for episode in selected_episodes:
            parquet = d06.episode_path(root, str(info["data_path"]), episode, chunks_size)
            phases = d06.infer_phases(
                d06.read_gripper_values(parquet),
                fps=float(info["fps"]),
                threshold=0.5,
                window=3,
            )
            video = d06.episode_path(
                root,
                str(info["video_path"]),
                episode,
                chunks_size,
                video_key=video_key,
            )
            frames = d06.extract_frames(video, phases.values(), temporary / f"ep{episode}")
            images = [
                Image.open(frames[phases[phase]]).convert("RGB").resize(
                    (args.input_size, args.input_size), Image.Resampling.BICUBIC
                )
                for phase in d0.PHASES
            ]
            candidates = d06.rank_cube_candidates(images[0], "primary", args.top_k)
            if not candidates:
                episode_results.append(
                    {"episode": episode, "candidate_count": 0, "automatic_role_gate": False}
                )
                rows_for_sheet.append(
                    [d06.draw_candidates(image, view="primary", episode=episode, phase=phase, candidates=[])
                     for phase, image in zip(d0.PHASES, images)]
                )
                continue
            mask = d06.blue_mask(images[0])
            candidate_points = [
                points_for_candidate(mask, candidate, args.points_per_candidate)
                for candidate in candidates
            ]
            query = np.concatenate(candidate_points, axis=0)
            image_tensor = torch.stack(
                [torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1) for image in images]
            ).unsqueeze(0).to(device)
            query_tensor = torch.from_numpy(query).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(image_tensor, query_points=query_tensor)
            track = prediction["track"]
            if track.shape[:2] != (1, len(d0.PHASES)):
                raise RuntimeError(f"Unexpected track shape: {tuple(track.shape)}")
            tracks = []
            metrics = []
            offset = 0
            for candidate_index, (candidate, points) in enumerate(zip(candidates, candidate_points)):
                count = len(points)
                candidate_track = track[0, :, offset : offset + count]
                offset += count
                candidate_metrics = temporal_role_metrics(
                    candidate_track,
                    width=args.input_size,
                    height=args.input_size,
                    candidate_score=candidate["score"],
                )
                tracks.append(candidate_track)
                metrics.append(candidate_metrics)
                csv_records.append(
                    {
                        "episode": episode,
                        "candidate_rank": candidate_index + 1,
                        **{key: value for key, value in candidate_metrics.items() if not isinstance(value, list)},
                        "bbox_xyxy": candidate["bbox_xyxy"],
                    }
                )
            selected_index = max(range(len(metrics)), key=lambda index: metrics[index]["role_score"])
            role_gate = passes_role_gate(
                metrics[selected_index],
                max_early=args.max_early_motion_px,
                min_late=args.min_late_motion_px,
                min_in_bounds=args.min_in_bounds_fraction,
            )
            episode_results.append(
                {
                    "episode": episode,
                    "candidate_count": len(candidates),
                    "selected_candidate_rank": selected_index + 1,
                    "selected_bbox_xyxy": candidates[selected_index]["bbox_xyxy"],
                    "selected_metrics": metrics[selected_index],
                    "all_candidate_metrics": metrics,
                    "automatic_role_gate": role_gate,
                    "manual_selected_is_cube": None,
                }
            )
            rows_for_sheet.append(
                draw_episode_overlay(
                    images,
                    candidates,
                    candidate_points,
                    tracks,
                    metrics,
                    selected_index,
                    episode,
                )
            )
            del prediction, track, image_tensor, query_tensor
            torch.cuda.empty_cache()

    cell = 250
    sheet = Image.new("RGB", (cell * len(d0.PHASES), cell * len(rows_for_sheet)), (20, 20, 20))
    for row_index, row in enumerate(rows_for_sheet):
        for column_index, panel in enumerate(row):
            sheet.paste(panel, (column_index * cell, row_index * cell))
    sheet_path = output / "primary_temporal_role_tracking_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    csv_path = output / "temporal_role_candidate_metrics.csv"
    fieldnames = [
        "episode",
        "candidate_rank",
        "role_score",
        "candidate_score",
        "approach_to_pregrasp_px",
        "pregrasp_to_grasp_px",
        "grasp_to_transport_px",
        "grasp_to_release_px",
        "early_motion_px",
        "late_motion_px",
        "in_bounds_fraction",
        "median_point_dispersion_px",
        "bbox_xyxy",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_records)
    passed = sum(bool(row.get("automatic_role_gate")) for row in episode_results)
    automatic_rate = passed / len(selected_episodes)
    payload = {
        "status": "PASS",
        "robot_commands_sent": 0,
        "dataset_root": str(root),
        "episode_indices": selected_episodes,
        "thresholds": {
            "max_early_motion_px": args.max_early_motion_px,
            "min_late_motion_px": args.min_late_motion_px,
            "min_in_bounds_fraction": args.min_in_bounds_fraction,
        },
        "automatic_role_gate_passed_episodes": passed,
        "automatic_role_gate_rate": automatic_rate,
        "automatic_evidence_gate": automatic_rate >= 0.8,
        "manual_review_required": True,
        "manual_acceptance_target": "at least 9/10 selected tracks stay on the physical cube",
        "wrist_temporal_tracking_used": False,
        "episode_results": episode_results,
    }
    result_path = output / "phase_d07_temporal_role_tracking.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("PHASE_D07_TEMPORAL_ROLE_TRACKING_AUDIT=PASS")
    print(f"AUTOMATIC_ROLE_GATE_RATE={automatic_rate:.6f}")
    print(f"AUTOMATIC_EVIDENCE_GATE={'PASS' if payload['automatic_evidence_gate'] else 'FAIL'}")
    print("MANUAL_REVIEW_REQUIRED=TRUE")
    print("WRIST_TEMPORAL_TRACKING_USED=FALSE")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"OUTPUT_DIR={output}")
    print(f"CONTACT_SHEET={sheet_path}")
    print(f"CSV={csv_path}")
    print(f"RESULT={result_path}")


if __name__ == "__main__":
    main()
