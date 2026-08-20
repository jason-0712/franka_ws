#!/usr/bin/env python3
"""Render position-distance heatmaps from a Spatial-Forcing audit JSON.

The output is a self-contained SVG and a long-form CSV.  The script uses only
the Python standard library so it can run in the StarVLA server environment
without matplotlib, seaborn, or a display server.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable


MODEL_TITLES = {
    "libero74": "Replay94 baseline",
    "alpha0": "Control alpha=0",
    "alpha01": "Treatment alpha=0.1",
    "teacher": "Frozen VGGT",
    "delta": "Treatment - Control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--scale-mode",
        choices=("row-shared", "per-panel"),
        default="row-shared",
        help=(
            "row-shared preserves absolute distance magnitudes within each camera; "
            "per-panel exposes relative rank structure used by RSA"
        ),
    )
    return parser.parse_args()


def _validate_matrix(name: str, value: object, size: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must be a {size}x{size} list")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{name} must be a {size}x{size} list")
        converted = [float(item) for item in row]
        if not all(math.isfinite(item) for item in converted):
            raise ValueError(f"{name} contains non-finite values")
        matrix.append(converted)
    return matrix


def load_matrices(path: Path):
    report = json.loads(path.read_text())
    labels = [str(label) for label in report["labels"]]
    per_model = report["results"]["per_model"]
    required = ("libero74", "alpha0", "alpha01")
    for model in required:
        if model not in per_model:
            raise KeyError(f"Audit is missing model={model}")

    views = list(per_model["alpha01"]["position_distance_matrices"])
    output = {}
    for view in views:
        models = {}
        for model in required:
            models[model] = _validate_matrix(
                f"{model}/{view}/student",
                per_model[model]["position_distance_matrices"][view]["student"],
                len(labels),
            )
        models["teacher"] = _validate_matrix(
            f"teacher/{view}",
            per_model["alpha01"]["position_distance_matrices"][view]["teacher"],
            len(labels),
        )
        models["delta"] = [
            [
                models["alpha01"][row][column] - models["alpha0"][row][column]
                for column in range(len(labels))
            ]
            for row in range(len(labels))
        ]
        output[view] = models
    rsa = {
        model: per_model[model]["position_rsa"]
        for model in required
    }
    return labels, views, output, rsa


def _interpolate(stops: Iterable[tuple[float, tuple[int, int, int]]], value: float):
    stops = list(stops)
    value = min(1.0, max(0.0, value))
    for (left_x, left), (right_x, right) in zip(stops, stops[1:]):
        if value <= right_x:
            ratio = 0.0 if right_x == left_x else (value - left_x) / (right_x - left_x)
            return tuple(round(a + ratio * (b - a)) for a, b in zip(left, right))
    return stops[-1][1]


def sequential_color(value: float, maximum: float) -> tuple[int, int, int]:
    normalized = 0.0 if maximum <= 0.0 else value / maximum
    return _interpolate(
        (
            (0.0, (247, 251, 255)),
            (0.35, (107, 174, 214)),
            (0.7, (33, 113, 181)),
            (1.0, (8, 48, 107)),
        ),
        normalized,
    )


def diverging_color(value: float, maximum_abs: float) -> tuple[int, int, int]:
    normalized = 0.5 if maximum_abs <= 0.0 else 0.5 + 0.5 * value / maximum_abs
    return _interpolate(
        (
            (0.0, (33, 102, 172)),
            (0.5, (247, 247, 247)),
            (1.0, (178, 24, 43)),
        ),
        normalized,
    )


def text_color(rgb: tuple[int, int, int]) -> str:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#ffffff" if luminance < 125 else "#111827"


def render_svg(labels, views, matrices, rsa, scale_mode: str) -> str:
    cell = 62
    matrix_size = cell * len(labels)
    left = 135
    top = 118
    panel_gap = 54
    row_gap = 128
    model_order = ("libero74", "alpha0", "alpha01", "teacher", "delta")
    width = left + len(model_order) * (matrix_size + panel_gap) + 20
    height = top + len(views) * (matrix_size + row_gap) + 40
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif}.title{font-size:24px;font-weight:700}'
        '.panel{font-size:16px;font-weight:700}.small{font-size:12px}.label{font-size:12px}'
        '</style>',
        '<text x="24" y="36" class="title">Spatial-Forcing position-distance audit</text>',
        '<text x="24" y="62" class="small">Cosine distance between mean-pooled visual tokens. '
        f'Color scale mode: {html.escape(scale_mode)}.</text>',
        '<text x="24" y="80" class="small">Delta: red means Treatment separates a pair more; '
        'blue means it separates the pair less than Control.</text>',
    ]

    for view_index, view in enumerate(views):
        y0 = top + view_index * (matrix_size + row_gap)
        pieces.append(
            f'<text x="24" y="{y0 - 22}" class="panel">View: {html.escape(view)}</text>'
        )
        row_values = [
            value
            for model in model_order[:-1]
            for row in matrices[view][model]
            for value in row
        ]
        row_max = max(row_values, default=1.0)
        delta_max = max(
            (abs(value) for row in matrices[view]["delta"] for value in row),
            default=1.0,
        )
        for panel_index, model in enumerate(model_order):
            x0 = left + panel_index * (matrix_size + panel_gap)
            title = MODEL_TITLES[model]
            if model in rsa:
                rsa_key = f"view_{view_index}"
                title += f"  RSA={float(rsa[model][rsa_key]):+.3f}"
            pieces.append(
                f'<text x="{x0 + matrix_size / 2}" y="{y0 - 22}" '
                f'text-anchor="middle" class="panel">{html.escape(title)}</text>'
            )
            panel_max = max(
                (value for row in matrices[view][model] for value in row),
                default=1.0,
            )
            max_value = panel_max if scale_mode == "per-panel" else row_max
            for index, label in enumerate(labels):
                x = x0 + index * cell + cell / 2
                pieces.append(
                    f'<text x="{x}" y="{y0 - 7}" text-anchor="end" class="label" '
                    f'transform="rotate(-38 {x} {y0 - 7})">{html.escape(label)}</text>'
                )
                if panel_index == 0:
                    pieces.append(
                        f'<text x="{x0 - 8}" y="{y0 + index * cell + cell / 2 + 4}" '
                        f'text-anchor="end" class="label">{html.escape(label)}</text>'
                    )
            for row_index, row in enumerate(matrices[view][model]):
                for column_index, value in enumerate(row):
                    rgb = (
                        diverging_color(value, delta_max)
                        if model == "delta"
                        else sequential_color(value, max_value)
                    )
                    color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                    x = x0 + column_index * cell
                    y = y0 + row_index * cell
                    pieces.append(
                        f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                        f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
                    )
                    pieces.append(
                        f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" '
                        f'text-anchor="middle" class="small" fill="{text_color(rgb)}">'
                        f'{value:+.3f}</text>'
                    )
    pieces.append('</svg>')
    return "\n".join(pieces) + "\n"


def write_csv(path: Path, labels, views, matrices) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("view", "model", "row_label", "column_label", "distance"),
        )
        writer.writeheader()
        for view in views:
            for model, matrix in matrices[view].items():
                for row_index, row in enumerate(matrix):
                    for column_index, value in enumerate(row):
                        writer.writerow(
                            {
                                "view": view,
                                "model": model,
                                "row_label": labels[row_index],
                                "column_label": labels[column_index],
                                "distance": f"{value:.10g}",
                            }
                        )


def main() -> None:
    args = parse_args()
    audit_json = args.audit_json.expanduser().resolve(strict=True)
    output_svg = args.output_svg or audit_json.with_name("position_distance_heatmaps.svg")
    output_csv = args.output_csv or audit_json.with_name("position_distance_matrices.csv")
    output_svg = output_svg.expanduser().absolute()
    output_csv = output_csv.expanduser().absolute()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    labels, views, matrices, rsa = load_matrices(audit_json)
    output_svg.write_text(render_svg(labels, views, matrices, rsa, args.scale_mode))
    write_csv(output_csv, labels, views, matrices)
    print("POSITION_DISTANCE_HEATMAP=PASS")
    print(f"INPUT={audit_json}")
    print(f"SVG={output_svg}")
    print(f"CSV={output_csv}")
    print(f"SCALE_MODE={args.scale_mode}")
    print("ROBOT_COMMANDS_SENT=0")


if __name__ == "__main__":
    main()
