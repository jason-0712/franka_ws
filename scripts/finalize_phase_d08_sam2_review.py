#!/usr/bin/env python3
"""Finalize the manual 9/10 cube-mask gate for a Phase D-0.8 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = json.loads(args.pilot_result.expanduser().read_text())
    decisions = json.loads(args.decisions.expanduser().read_text())
    expected = [int(value) for value in pilot["episode_indices"]]
    if not isinstance(decisions, dict):
        raise ValueError("decisions must be an episode-to-boolean JSON object")
    normalized = {}
    for episode in expected:
        value = decisions.get(str(episode), decisions.get(episode))
        if not isinstance(value, bool):
            raise ValueError(f"Episode {episode} decision must be true or false")
        normalized[str(episode)] = value
    unexpected = set(map(str, decisions)) - set(map(str, expected))
    if unexpected:
        raise ValueError(f"Unexpected episode decisions: {sorted(unexpected)}")

    passed = sum(normalized.values())
    rate = passed / len(expected)
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    payload = {
        "status": "PASS",
        "robot_commands_sent": 0,
        "pilot_result": str(args.pilot_result.expanduser().absolute()),
        "decisions": normalized,
        "manual_cube_mask_pass_count": passed,
        "episode_count": len(expected),
        "manual_cube_mask_pass_rate": rate,
        "manual_cube_mask_gate": rate >= 0.9,
        "next_route": (
            "POOL_VGGT_GEOMETRY_INSIDE_PRIMARY_CUBE_MASKS"
            if rate >= 0.9
            else "FIX_PROMPTS_OR_STOP_TASK_RELATIVE_TEACHER"
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("PHASE_D08_MANUAL_REVIEW=PASS")
    print(f"MANUAL_CUBE_MASK_PASS_RATE={rate:.6f}")
    print(f"MANUAL_CUBE_MASK_GATE={'PASS' if payload['manual_cube_mask_gate'] else 'FAIL'}")
    print(f"NEXT_ROUTE={payload['next_route']}")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"RESULT={output}")


if __name__ == "__main__":
    main()
