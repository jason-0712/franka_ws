#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_spatial_forcing_phase6_update_audit.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"

for relative_path in \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh; do
  install -D -m 0644 \
    "${stage_root}/third_party/starVLA/${relative_path}" \
    "${starvla_repo}/${relative_path}"
done
chmod 0755 \
  "${starvla_repo}/examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh"

trainer_file="${starvla_repo}/starVLA/training/train_starvla.py"
if grep -q 'metrics.update(parameter_update_metrics)' "${trainer_file}"; then
  echo "ZeRO-safe parameter-update audit is already installed."
else
  patch --batch --forward -d "${starvla_repo}" -p1 \
    < "${stage_root}/scripts/spatial_forcing_phase6_zero_safe_update.patch"
fi

python -m py_compile \
  "${starvla_repo}/starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py" \
  "${starvla_repo}/starVLA/training/train_starvla.py"

echo "SPATIAL_FORCING_PHASE6_UPDATE_AUDIT_INSTALL=PASS"
