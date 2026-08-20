#!/usr/bin/env python3
"""Plot matched StarVLA probe actions as a 2-D object-position vector field.

Input is a CSV manifest with one row per manually measured cube position.  It
must contain ``label,x,y`` and one probe-JSON column per model.  Example::

    label,x,y,control_probe,treatment_probe
    front,0.02,0.00,/path/front/control/probe.json,/path/front/treatment/probe.json

The tool is offline: it reads existing JSON files, creates SVG/CSV outputs,
and never imports ROS or contacts a policy server.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-column",
        action="append",
        required=True,
        metavar="NAME=COLUMN",
        help="Repeat for each model, e.g. control=control_probe.",
    )
    parser.add_argument(
        "--action-field",
        choices=("first_action", "translation_mean"),
        default="first_action",
    )
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--arrow-gain",
        type=float,
        default=None,
        help="Visual multiplier from action units to position units; default is automatic.",
    )
    return parser.parse_args()


def parse_model_columns(values: list[str]) -> list[tuple[str, str]]:
    output = []
    seen_names = set()
    seen_columns = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=COLUMN, got: {value}")
        name, column = (part.strip() for part in value.split("=", 1))
        if not name or not column:
            raise ValueError(f"Expected non-empty NAME=COLUMN, got: {value}")
        if name in seen_names or column in seen_columns:
            raise ValueError(f"Duplicate model name or manifest column: {value}")
        seen_names.add(name)
        seen_columns.add(column)
        output.append((name, column))
    return output


def resolve_probe_path(manifest: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve(strict=True)
    if path.is_dir():
        path = (path / "probe.json").resolve(strict=True)
    return path


def load_records(manifest: Path, models: list[tuple[str, str]], action_field: str):
    records = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "x", "y", *(column for _, column in models)}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=2):
            label = row["label"].strip()
            if not label:
                raise ValueError(f"Empty label at manifest row {row_index}")
            x = float(row["x"])
            y = float(row["y"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Non-finite position at manifest row {row_index}")
            actions = {}
            for model, column in models:
                probe_path = resolve_probe_path(manifest, row[column])
                probe = json.loads(probe_path.read_text())
                vector = probe.get(action_field)
                if not isinstance(vector, list) or len(vector) < 2:
                    raise ValueError(
                        f"{probe_path} lacks a valid {action_field} vector"
                    )
                dx, dy = float(vector[0]), float(vector[1])
                if not math.isfinite(dx) or not math.isfinite(dy):
                    raise ValueError(f"Non-finite action in {probe_path}")
                actions[model] = {
                    "dx": dx,
                    "dy": dy,
                    "norm": math.hypot(dx, dy),
                    "close_fraction": float(probe.get("close_fraction", float("nan"))),
                    "probe": str(probe_path),
                    "checkpoint": str(probe.get("metadata", {}).get("ckpt_path", "")),
                }
            records.append({"label": label, "x": x, "y": y, "actions": actions})
    if len(records) < 3:
        raise ValueError("At least three object positions are required")
    return records


def nonzero_span(values: list[float]) -> float:
    span = max(values) - min(values)
    return span if span > 1e-12 else 1.0


def automatic_gain(records, models) -> float:
    positions = [(record["x"], record["y"]) for record in records]
    pair_distances = [
        math.hypot(ax - bx, ay - by)
        for index, (ax, ay) in enumerate(positions)
        for bx, by in positions[index + 1 :]
        if math.hypot(ax - bx, ay - by) > 1e-12
    ]
    spacing = min(pair_distances) if pair_distances else 1.0
    max_action = max(
        record["actions"][model]["norm"]
        for record in records
        for model, _ in models
    )
    return 1.0 if max_action <= 1e-12 else 0.42 * spacing / max_action


def render_svg(records, models, action_field: str, gain: float) -> str:
    panel_width = 560
    panel_height = 560
    left_margin = 88
    top_margin = 125
    panel_gap = 45
    width = 36 + len(models) * (panel_width + panel_gap)
    height = top_margin + panel_height + 85
    xs = [record["x"] for record in records]
    ys = [record["y"] for record in records]
    x_span = nonzero_span(xs)
    y_span = nonzero_span(ys)
    span = max(x_span, y_span)
    x_min = min(xs) - 0.18 * span
    x_max = max(xs) + 0.18 * span
    y_min = min(ys) - 0.18 * span
    y_max = max(ys) + 0.18 * span

    def sx(value, x0):
        return x0 + (value - x_min) / (x_max - x_min) * panel_width

    def sy(value):
        return top_margin + panel_height - (value - y_min) / (y_max - y_min) * panel_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif}.title{font-size:24px;font-weight:700}'
        '.panel{font-size:18px;font-weight:700}.small{font-size:12px}.label{font-size:13px}'
        '</style>',
        '<defs>',
    ]
    for model_index, (model, _) in enumerate(models):
        color = COLORS[model_index % len(COLORS)]
        pieces.append(
            f'<marker id="arrow{model_index}" markerWidth="9" markerHeight="7" '
            f'refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" '
            f'fill="{color}"/></marker>'
        )
    pieces.extend(
        [
            '</defs>',
            '<text x="24" y="34" class="title">StarVLA object-position action vector field</text>',
            f'<text x="24" y="58" class="small">Action source: {html.escape(action_field)}; '
            f'arrows multiplied by {gain:.4g} for display. All panels use the same scale.</text>',
            '<text x="24" y="77" class="small">Dots are measured cube positions; arrows are '
            'predicted (dx, dy). This file is generated offline and sends no robot commands.</text>',
        ]
    )

    for model_index, (model, _) in enumerate(models):
        x0 = left_margin + model_index * (panel_width + panel_gap)
        color = COLORS[model_index % len(COLORS)]
        pieces.append(
            f'<text x="{x0 + panel_width / 2}" y="{top_margin - 24}" '
            f'text-anchor="middle" class="panel">{html.escape(model)}</text>'
        )
        pieces.append(
            f'<rect x="{x0}" y="{top_margin}" width="{panel_width}" height="{panel_height}" '
            'fill="#f8fafc" stroke="#94a3b8"/>'
        )
        zero_x = sx(0.0, x0) if x_min <= 0.0 <= x_max else None
        zero_y = sy(0.0) if y_min <= 0.0 <= y_max else None
        if zero_x is not None:
            pieces.append(
                f'<line x1="{zero_x}" y1="{top_margin}" x2="{zero_x}" '
                f'y2="{top_margin + panel_height}" stroke="#cbd5e1" stroke-dasharray="5 5"/>'
            )
        if zero_y is not None:
            pieces.append(
                f'<line x1="{x0}" y1="{zero_y}" x2="{x0 + panel_width}" '
                f'y2="{zero_y}" stroke="#cbd5e1" stroke-dasharray="5 5"/>'
            )
        for record in records:
            action = record["actions"][model]
            start_x = sx(record["x"], x0)
            start_y = sy(record["y"])
            end_x = sx(record["x"] + gain * action["dx"], x0)
            end_y = sy(record["y"] + gain * action["dy"])
            pieces.append(
                f'<line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" '
                f'stroke="{color}" stroke-width="4" marker-end="url(#arrow{model_index})"/>'
            )
            pieces.append(
                f'<circle cx="{start_x}" cy="{start_y}" r="5" fill="#111827"/>'
            )
            pieces.append(
                f'<text x="{start_x + 7}" y="{start_y - 8}" class="label">'
                f'{html.escape(record["label"])}</text>'
            )
            pieces.append(
                f'<text x="{start_x + 7}" y="{start_y + 15}" class="small" fill="{color}">'
                f'd=({action["dx"]:+.4f},{action["dy"]:+.4f})</text>'
            )
        pieces.append(
            f'<text x="{x0 + panel_width / 2}" y="{top_margin + panel_height + 38}" '
            'text-anchor="middle" class="label">cube-position x</text>'
        )
        pieces.append(
            f'<text x="{x0 - 54}" y="{top_margin + panel_height / 2}" '
            f'text-anchor="middle" class="label" transform="rotate(-90 {x0 - 54} '
            f'{top_margin + panel_height / 2})">cube-position y</text>'
        )
    pieces.append('</svg>')
    return "\n".join(pieces) + "\n"


def write_csv(path: Path, records, models, action_field: str, gain: float) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "label", "x", "y", "model", "action_field", "dx", "dy",
                "xy_norm", "close_fraction", "visual_arrow_gain", "probe", "checkpoint",
            ),
        )
        writer.writeheader()
        for record in records:
            for model, _ in models:
                action = record["actions"][model]
                writer.writerow(
                    {
                        "label": record["label"],
                        "x": record["x"],
                        "y": record["y"],
                        "model": model,
                        "action_field": action_field,
                        "dx": action["dx"],
                        "dy": action["dy"],
                        "xy_norm": action["norm"],
                        "close_fraction": action["close_fraction"],
                        "visual_arrow_gain": gain,
                        "probe": action["probe"],
                        "checkpoint": action["checkpoint"],
                    }
                )


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve(strict=True)
    models = parse_model_columns(args.model_column)
    records = load_records(manifest, models, args.action_field)
    gain = args.arrow_gain if args.arrow_gain is not None else automatic_gain(records, models)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("--arrow-gain must be finite and positive")
    output_svg = args.output_svg or manifest.with_name("action_vector_field.svg")
    output_csv = args.output_csv or manifest.with_name("action_vector_field.csv")
    output_svg = output_svg.expanduser().absolute()
    output_csv = output_csv.expanduser().absolute()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text(render_svg(records, models, args.action_field, gain))
    write_csv(output_csv, records, models, args.action_field, gain)
    print("ACTION_VECTOR_FIELD=PASS")
    print(f"INPUT={manifest}")
    print(f"SVG={output_svg}")
    print(f"CSV={output_csv}")
    print(f"ARROW_GAIN={gain}")
    print("ROBOT_COMMANDS_SENT=0")


if __name__ == "__main__":
    main()
