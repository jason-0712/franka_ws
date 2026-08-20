#!/usr/bin/env python3
"""Finalize the preregistered semantic gate for Phase D-0.8.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASES = ("approach", "pre_grasp", "grasp", "transport", "release")
ANCHORS = ("approach", "release")
LABELS = {"correct", "occluded_abstain", "visible_missing", "wrong"}


def score_review(pilot: dict, review: dict) -> dict:
    expected = [str(value) for value in pilot["episode_indices"]]
    episodes = review.get("episodes")
    if not isinstance(episodes, dict) or set(episodes) != set(expected):
        raise ValueError("Review episode keys do not match pilot")
    counts = {label: 0 for label in LABELS}
    anchor_correct = 0
    for episode in expected:
        phase_labels = episodes[episode]
        if not isinstance(phase_labels, dict) or set(phase_labels) != set(PHASES):
            raise ValueError(f"Episode {episode} phase keys do not match")
        for phase in PHASES:
            label = phase_labels[phase]
            if label not in LABELS:
                raise ValueError(f"Episode {episode} {phase} has invalid label {label!r}")
            counts[label] += 1
            if phase in ANCHORS and label == "correct":
                anchor_correct += 1
    visible_total = counts["correct"] + counts["visible_missing"] + counts["wrong"]
    visible_accuracy = counts["correct"] / visible_total if visible_total else 0.0
    anchor_accuracy = anchor_correct / (len(expected) * len(ANCHORS))
    gate = (
        counts["wrong"] == 0
        and visible_accuracy >= 0.9
        and anchor_accuracy >= 0.9
    )
    return {
        "label_counts": counts,
        "wrong_object_count": counts["wrong"],
        "visible_phase_accuracy": visible_accuracy,
        "anchor_accuracy": anchor_accuracy,
        "manual_semantic_gate": gate,
        "next_route": (
            "POOL_VGGT_GEOMETRY_INSIDE_HIGH_CONFIDENCE_CUBE_MASKS"
            if gate
            else "DO_NOT_TRAIN_TASK_RELATIVE_TEACHER"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads(args.pilot_result.expanduser().read_text())
    review = json.loads(args.review.expanduser().read_text())
    score = score_review(pilot, review)
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    payload = {
        "status": "PASS",
        "robot_commands_sent": 0,
        "pilot_result": str(args.pilot_result.expanduser().absolute()),
        **score,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("PHASE_D081_MANUAL_REVIEW=PASS")
    print(f"WRONG_OBJECT_COUNT={score['wrong_object_count']}")
    print(f"VISIBLE_PHASE_ACCURACY={score['visible_phase_accuracy']:.6f}")
    print(f"ANCHOR_ACCURACY={score['anchor_accuracy']:.6f}")
    print(f"MANUAL_SEMANTIC_GATE={'PASS' if score['manual_semantic_gate'] else 'FAIL'}")
    print(f"NEXT_ROUTE={score['next_route']}")
    print("ROBOT_COMMANDS_SENT=0")
    print(f"RESULT={output}")


if __name__ == "__main__":
    main()
