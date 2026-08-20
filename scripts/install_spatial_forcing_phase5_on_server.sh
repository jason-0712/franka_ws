#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_spatial_forcing_phase5_on_server.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"

if [[ ! -d "${starvla_repo}/starVLA" ]]; then
  echo "StarVLA repository not found: ${starvla_repo}" >&2
  exit 1
fi
if [[ ! -f "${stage_root}/vggt/models/vggt.py" ]]; then
  echo "Pinned VGGT source missing from Phase-5 stage: ${stage_root}" >&2
  exit 1
fi

for relative_path in \
  starVLA/model/modules/spatial_forcing/__init__.py \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/modules/spatial_forcing/lora_student.py \
  starVLA/model/modules/spatial_forcing/vggt_teacher.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_spatial_forcing.yaml \
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh \
  tests/test_spatial_forcing_alignment.py \
  tests/test_spatial_forcing_lora.py; do
  install -D -m 0644 \
    "${stage_root}/third_party/starVLA/${relative_path}" \
    "${starvla_repo}/${relative_path}"
done
chmod 0755 \
  "${starvla_repo}/examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh"

while IFS= read -r source_file; do
  relative_path="${source_file#${stage_root}/}"
  install -D -m 0644 "${source_file}" "${starvla_repo}/${relative_path}"
done < <(find "${stage_root}/vggt" -type f -name '*.py' | sort)

trainer_file="${starvla_repo}/starVLA/training/train_starvla.py"
if grep -q 'gradient_norm/spatial_forcing_lora' "${starvla_repo}/starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py" \
  && grep -q 'metrics.update(gradient_metrics)' "${trainer_file}"; then
  echo "Spatial-Forcing gradient audit is already installed."
else
  patch --batch --forward -d "${starvla_repo}" -p1 \
    < "${stage_root}/scripts/spatial_forcing_phase5_gradient_metrics.patch"
fi

cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"
python -m py_compile \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/modules/spatial_forcing/lora_student.py \
  starVLA/model/modules/spatial_forcing/vggt_teacher.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  starVLA/training/train_starvla.py \
  vggt/models/vggt.py
python - <<'PY'
import peft
from vggt.models.vggt import VGGT

print("peft=", peft.__version__)
print("VGGT_IMPORT=PASS", VGGT.__name__)
PY

echo "SPATIAL_FORCING_PHASE5_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
echo "VGGT_SOURCE_COMMIT=372819b31ba9a2c8fc5989edc7d525cb187cecd5"
