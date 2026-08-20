#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_spatial_forcing_phase4_on_server.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"

if [[ ! -d "${starvla_repo}/starVLA" ]]; then
  echo "StarVLA repository not found: ${starvla_repo}" >&2
  exit 1
fi
if [[ ! -f "${stage_root}/third_party/starVLA/tests/test_spatial_forcing_lora.py" ]]; then
  echo "Incomplete Phase-4 stage: ${stage_root}" >&2
  exit 1
fi

for relative_path in \
  starVLA/model/modules/spatial_forcing/__init__.py \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/modules/spatial_forcing/lora_student.py \
  starVLA/model/modules/spatial_forcing/vggt_teacher.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  tests/test_spatial_forcing_alignment.py \
  tests/test_spatial_forcing_lora.py; do
  install -D -m 0644 \
    "${stage_root}/third_party/starVLA/${relative_path}" \
    "${starvla_repo}/${relative_path}"
done

trainer_file="${starvla_repo}/starVLA/training/train_starvla.py"
if grep -q 'total_loss = output_dict.get("total_loss", action_loss)' "${trainer_file}"; then
  echo "Spatial-Forcing total-loss trainer patch is already installed."
else
  patch --batch --forward -d "${starvla_repo}" -p1 \
    < "${stage_root}/scripts/spatial_forcing_phase4_train_loss.patch"
fi

python -m py_compile \
  "${starvla_repo}/starVLA/model/modules/spatial_forcing/alignment.py" \
  "${starvla_repo}/starVLA/model/modules/spatial_forcing/lora_student.py" \
  "${starvla_repo}/starVLA/model/modules/spatial_forcing/vggt_teacher.py" \
  "${starvla_repo}/starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py" \
  "${starvla_repo}/starVLA/training/train_starvla.py"

echo "SPATIAL_FORCING_PHASE4_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
