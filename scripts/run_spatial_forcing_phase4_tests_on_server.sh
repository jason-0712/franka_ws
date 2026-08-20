#!/usr/bin/env bash
set -euo pipefail

starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"

python - <<'PY'
import peft
import torch
import transformers

print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("peft=", peft.__version__)
PY

python -m unittest discover -v \
  -s tests \
  -p 'test_spatial_forcing_*.py'

python - <<'PY'
from pathlib import Path

trainer = Path("starVLA/training/train_starvla.py").read_text()
required = [
    'total_loss = output_dict.get("total_loss", action_loss)',
    '"alignment_loss", "weighted_alignment_loss"',
]
missing = [item for item in required if item not in trainer]
if missing:
    raise SystemExit(f"trainer total-loss integration missing: {missing}")
print("TRAINER_TOTAL_LOSS_INTEGRATION=PASS")
PY

echo "SPATIAL_FORCING_PHASE4_TESTS=PASS"
