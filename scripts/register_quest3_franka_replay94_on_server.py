#!/usr/bin/env python3
"""Idempotently register the replay-94 dataset in the server data registry."""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


DATASET_NAME = "quest3_franka_dualcam_replay_94eps_v1"
MARKER = "# CODEX_REPLAY94_DATASET_REGISTRATION_V1"
REGISTRATION = f'''\n\n{MARKER}\nDATASET_NAMED_MIXTURES["{DATASET_NAME}"] = [\n    ("{DATASET_NAME}", 1.0, "quest3_franka_dualcam_delta_eef"),\n]\n'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(
            "/home/hanyu/starVLA/examples/realRobots/Franka/train_files/"
            "data_registry/data_config.py"
        ),
    )
    args = parser.parse_args()
    registry = args.registry.resolve()
    if not registry.is_file():
        raise FileNotFoundError(registry)

    source = registry.read_text(encoding="utf-8")
    if DATASET_NAME in source:
        print(f"REPLAY94_REGISTRY=ALREADY_PRESENT {registry}")
    else:
        backup = registry.with_name(registry.name + ".before_replay94")
        if not backup.exists():
            backup.write_text(source, encoding="utf-8")
        registry.write_text(source.rstrip() + REGISTRATION, encoding="utf-8")
        print(f"REPLAY94_REGISTRY=ADDED {registry}")
        print(f"REPLAY94_REGISTRY_BACKUP={backup}")

    py_compile.compile(str(registry), doraise=True)
    final_source = registry.read_text(encoding="utf-8")
    if DATASET_NAME not in final_source:
        raise RuntimeError("Replay-94 registration was not persisted")
    print("REPLAY94_REGISTRY_COMPILE=PASS")


if __name__ == "__main__":
    main()
