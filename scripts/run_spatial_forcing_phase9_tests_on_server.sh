#!/usr/bin/env bash
set -euo pipefail

starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"

python tests/test_spatial_forcing_alignment.py -v

python - <<'PY'
from pathlib import Path

alignment = Path("starVLA/model/modules/spatial_forcing/alignment.py").read_text()
framework = Path("starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py").read_text()
trainer = Path("starVLA/training/train_starvla.py").read_text()
runner = Path(
    "examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh"
).read_text()
config = Path(
    "examples/realRobots/Franka/train_files/"
    "starvla_cotrain_quest3_franka_spatial_forcing.yaml"
).read_text()

required = {
    "alignment": [
        "class SceneRelationalMemoryBank",
        'persistent=False',
        "self._enqueue(student, teacher)",
    ],
    "framework": [
        '"scene_relational_alpha": 0.0',
        '"scene_queue_size": 0',
        "weighted_scene_relational_loss",
        "scene_queue_fill",
    ],
    "trainer": [
        '"scene_relational_loss"',
        '"weighted_scene_relational_loss"',
        '"scene_queue_fill"',
    ],
    "runner": ["SCENE_RELATIONAL_ALPHA", "SCENE_QUEUE_SIZE"],
    "config": ["scene_relational_alpha: 0.0", "scene_queue_size: 64"],
}
contents = {
    "alignment": alignment,
    "framework": framework,
    "trainer": trainer,
    "runner": runner,
    "config": config,
}
missing = {
    name: [needle for needle in needles if needle not in contents[name]]
    for name, needles in required.items()
}
missing = {name: needles for name, needles in missing.items() if needles}
if missing:
    raise SystemExit(f"Phase-9 integration is incomplete: {missing}")
print("SPATIAL_FORCING_PHASE9_STATIC_INTEGRATION=PASS")
PY

echo "SPATIAL_FORCING_PHASE9_TESTS=PASS"
