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

required = {
    "alignment": ["def relational_geometry_loss", "teacher_tokens.detach()"],
    "framework": [
        '"projected_alignment_alpha": None',
        '"relational_alignment_alpha": 0.0',
        "weighted_relational_alignment_loss",
        "total_spatial_alignment_loss",
    ],
    "trainer": [
        'total_loss = output_dict.get("total_loss", action_loss)',
        '"relational_alignment_loss"',
    ],
    "runner": ["PROJECTED_ALIGNMENT_ALPHA", "RELATIONAL_ALIGNMENT_ALPHA"],
}
contents = {
    "alignment": alignment,
    "framework": framework,
    "trainer": trainer,
    "runner": runner,
}
missing = {
    name: [needle for needle in needles if needle not in contents[name]]
    for name, needles in required.items()
}
missing = {name: needles for name, needles in missing.items() if needles}
if missing:
    raise SystemExit(f"Phase-8 integration is incomplete: {missing}")
print("SPATIAL_FORCING_PHASE8_STATIC_INTEGRATION=PASS")
PY

echo "SPATIAL_FORCING_PHASE8_TESTS=PASS"
