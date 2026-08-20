#!/usr/bin/env python3
"""Add a Replay94 baseline to an existing matched action-vector-field session.

The script re-queries one baseline policy with the exact snapshots referenced
by ``positions.csv``, writes a three-model manifest, and renders first-action
and chunk-mean vector fields.  It never initializes ROS or sends robot
commands.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import socket
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline-host", default="192.168.1.113")
    parser.add_argument("--baseline-port", type=int, default=10112)
    parser.add_argument("--baseline-name", default="replay94_baseline")
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def verify_server(host: str, port: int, timeout: float) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        raise ConnectionError(f"Baseline policy server is unavailable: {host}:{port}") from exc


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"label", "x", "y", "control_probe", "treatment_probe"}
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def validate_existing_baseline(probe_path: Path, source_dir: Path, baseline_name: str) -> None:
    record = json.loads(probe_path.read_text())
    if Path(record.get("source_snapshot_dir", "")).resolve() != source_dir.resolve():
        raise RuntimeError(f"Existing baseline probe uses another snapshot: {probe_path}")
    if record.get("ros_command_publishers_created") != 0:
        raise RuntimeError(f"Existing baseline probe is not command-free: {probe_path}")
    print(f"REUSE_BASELINE_PROBE={baseline_name} {probe_path}")


def main() -> None:
    args = parse_args()
    if not 1 <= args.baseline_port <= 65535:
        raise ValueError("--baseline-port must be in [1,65535]")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    session = args.session.expanduser().resolve(strict=True)
    source_manifest = session / "positions.csv"
    rows = read_manifest(source_manifest)
    verify_server(args.baseline_host, args.baseline_port, args.timeout)

    script_dir = Path(__file__).resolve().parent
    requery = script_dir / "starvla_requery_saved_snapshot.py"
    plotter = script_dir / "plot_starvla_action_vector_field.py"
    for required in (requery, plotter):
        if not required.is_file():
            raise FileNotFoundError(f"Required helper is missing: {required}")

    output_rows = []
    for row in rows:
        label = row["label"]
        control_probe = (session / row["control_probe"]).resolve(strict=True)
        source_dir = control_probe.parent
        baseline_dir = session / label / args.baseline_name
        baseline_probe = baseline_dir / "probe.json"
        if baseline_probe.is_file():
            validate_existing_baseline(baseline_probe, source_dir, args.baseline_name)
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(requery),
                    "--source-snapshot-dir", str(source_dir),
                    "--policy-host", args.baseline_host,
                    "--policy-port", str(args.baseline_port),
                    "--label", label,
                    "--output-dir", str(baseline_dir),
                ],
                check=True,
            )
        output_rows.append(
            {
                "label": label,
                "x": row["x"],
                "y": row["y"],
                "baseline_probe": str(baseline_probe.relative_to(session)),
                "control_probe": row["control_probe"],
                "treatment_probe": row["treatment_probe"],
            }
        )

    output_manifest = session / "positions_with_replay94_baseline.csv"
    with output_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    outputs = []
    for action_field, stem in (
        ("first_action", "three_model_first_action_vector_field"),
        ("translation_mean", "three_model_chunk_mean_vector_field"),
    ):
        svg = session / f"{stem}.svg"
        output_csv = session / f"{stem}.csv"
        subprocess.run(
            [
                sys.executable,
                str(plotter),
                "--manifest", str(output_manifest),
                "--model-column", f"{args.baseline_name}=baseline_probe",
                "--model-column", "control=control_probe",
                "--model-column", "treatment=treatment_probe",
                "--action-field", action_field,
                "--output-svg", str(svg),
                "--output-csv", str(output_csv),
            ],
            check=True,
        )
        outputs.extend((svg, output_csv))

    print("REPLAY94_BASELINE_VECTOR_FIELD=PASS")
    print("MATCHED_SOURCE_SNAPSHOTS=PASS")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"MANIFEST={output_manifest}")
    for output in outputs:
        print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
