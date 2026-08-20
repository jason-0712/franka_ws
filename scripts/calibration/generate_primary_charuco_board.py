#!/usr/bin/env python3
"""Generate a metric A4 ChArUco target for primary-camera calibration.

Run this script with the system Python inside the ``franka`` container, where
OpenCV's aruco module and Pillow are already installed.  The generated PNG is
tagged as 300 DPI; it must be printed at 100% / actual size with all printer
scaling disabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DEFAULT_DPI = 300
SQUARES_X = 7
SQUARES_Y = 10
SQUARE_LENGTH_MM = 24.0
MARKER_LENGTH_MM = 18.0
DICTIONARY_NAME = "DICT_5X5_1000"


def mm_to_px(value_mm: float, dpi: int) -> int:
    return int(round(value_mm * dpi / 25.4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    page_width_px = mm_to_px(A4_WIDTH_MM, args.dpi)
    page_height_px = mm_to_px(A4_HEIGHT_MM, args.dpi)
    board_width_mm = SQUARES_X * SQUARE_LENGTH_MM
    board_height_mm = SQUARES_Y * SQUARE_LENGTH_MM
    board_width_px = mm_to_px(board_width_mm, args.dpi)
    board_height_px = mm_to_px(board_height_mm, args.dpi)

    if board_width_px >= page_width_px or board_height_px >= page_height_px:
        raise RuntimeError("ChArUco board does not fit on A4")

    dictionary_id = getattr(cv2.aruco, DICTIONARY_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard_create(
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LENGTH_MM / 1000.0,
        MARKER_LENGTH_MM / 1000.0,
        dictionary,
    )
    board_image = board.draw(
        (board_width_px, board_height_px),
        marginSize=0,
        borderBits=1,
    )

    page = Image.new("L", (page_width_px, page_height_px), color=255)
    board_pil = Image.fromarray(board_image)
    offset_x = (page_width_px - board_width_px) // 2
    offset_y = (page_height_px - board_height_px) // 2
    page.paste(board_pil, (offset_x, offset_y))

    # A physical scale check outside the target. The line must measure exactly
    # 100 mm after printing; otherwise the print must not be used for metric
    # calibration.
    draw = ImageDraw.Draw(page)
    scale_length_px = mm_to_px(100.0, args.dpi)
    scale_y = page_height_px - mm_to_px(8.0, args.dpi)
    scale_x0 = (page_width_px - scale_length_px) // 2
    scale_x1 = scale_x0 + scale_length_px
    cap_px = max(4, mm_to_px(2.0, args.dpi))
    line_px = max(2, mm_to_px(0.35, args.dpi))
    draw.line((scale_x0, scale_y, scale_x1, scale_y), fill=0, width=line_px)
    draw.line((scale_x0, scale_y - cap_px, scale_x0, scale_y + cap_px), fill=0, width=line_px)
    draw.line((scale_x1, scale_y - cap_px, scale_x1, scale_y + cap_px), fill=0, width=line_px)
    draw.text((scale_x0, scale_y - mm_to_px(6.0, args.dpi)), "100 mm - PRINT AT 100%", fill=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    page.save(args.output, dpi=(args.dpi, args.dpi), optimize=True)

    # Validate that the generated raster can be detected before handing it to
    # the operator. Detection is performed on the cropped board so the scale
    # annotation cannot affect the result.
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(board_image, dictionary)
    if marker_ids is None:
        raise RuntimeError("Generated board failed ArUco marker detection")
    corner_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        board_image,
        board,
    )
    detected_markers = int(len(marker_ids))
    detected_corners = int(corner_count)
    expected_corners = (SQUARES_X - 1) * (SQUARES_Y - 1)
    if detected_corners != expected_corners:
        raise RuntimeError(
            f"Detected {detected_corners} ChArUco corners; expected {expected_corners}"
        )

    metadata = {
        "schema_version": 1,
        "page": {
            "format": "A4",
            "width_mm": A4_WIDTH_MM,
            "height_mm": A4_HEIGHT_MM,
            "dpi": args.dpi,
            "print_scale_percent": 100,
        },
        "charuco": {
            "dictionary": DICTIONARY_NAME,
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_m": SQUARE_LENGTH_MM / 1000.0,
            "marker_length_m": MARKER_LENGTH_MM / 1000.0,
            "board_width_m": board_width_mm / 1000.0,
            "board_height_m": board_height_mm / 1000.0,
        },
        "self_check": {
            "detected_markers": detected_markers,
            "detected_charuco_corners": detected_corners,
            "expected_charuco_corners": expected_corners,
            "pass": True,
        },
        "operator_checks": [
            "Print at 100% / actual size; disable fit-to-page and scaling.",
            "The verification line must measure 100.0 mm with a ruler.",
            "One chessboard square must measure 24.0 mm.",
            "Mount the sheet flat on a rigid board before calibration.",
        ],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("CHARUCO_BOARD_GENERATION=PASS")
    print(f"PNG={args.output}")
    print(f"METADATA={metadata_path}")
    print(f"MARKERS={detected_markers}")
    print(f"CHARUCO_CORNERS={detected_corners}/{expected_corners}")
    print("PRINT_SCALE_CHECK_MM=100.0")
    print("ROBOT_COMMANDS_SENT=0")


if __name__ == "__main__":
    main()
