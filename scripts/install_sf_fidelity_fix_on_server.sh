#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_sf_fidelity_fix_on_server.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"

if [[ ! -d "${starvla_repo}/starVLA" ]]; then
  echo "StarVLA repository not found: ${starvla_repo}" >&2
  exit 1
fi

relative_paths=(
  starVLA/model/modules/spatial_forcing/__init__.py
  starVLA/model/modules/spatial_forcing/alignment.py
  starVLA/model/modules/spatial_forcing/image_augmentation.py
  starVLA/model/modules/spatial_forcing/vggt_teacher.py
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingClean.py
  examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_spatial_forcing_clean.yaml
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh
  tests/test_spatial_forcing_alignment.py
  tests/test_spatial_forcing_clean.py
)

for relative_path in "${relative_paths[@]}"; do
  source_path="${stage_root}/third_party/starVLA/${relative_path}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing staged source: ${source_path}" >&2
    exit 1
  fi
done

backup_root="${starvla_repo}_sf_fidelity_backups/$(date +%Y%m%d_%H%M%S)"
for relative_path in "${relative_paths[@]}"; do
  target_path="${starvla_repo}/${relative_path}"
  if [[ -f "${target_path}" ]]; then
    install -D -m 0644 "${target_path}" "${backup_root}/${relative_path}"
  fi
  install -D -m 0644 \
    "${stage_root}/third_party/starVLA/${relative_path}" \
    "${target_path}"
done

chmod 0755 \
  "${starvla_repo}/examples/realRobots/Franka/train_files/"\
"run_qwengroot_spatial_forcing_clean_smoke.sh"
install -D -m 0755 \
  "${stage_root}/scripts/export_spatial_forcing_rgb_view.py" \
  "${starvla_repo}/export_spatial_forcing_rgb_view.py"
install -D -m 0755 \
  "${stage_root}/scripts/run_spatial_forcing_phase10_tests_on_server.sh" \
  "${starvla_repo}/run_spatial_forcing_phase10_tests_on_server.sh"
install -D -m 0755 \
  "${stage_root}/scripts/run_sf_fidelity_smoke_matrix_on_server.sh" \
  "${starvla_repo}/run_sf_fidelity_smoke_matrix_on_server.sh"

cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"
python -m py_compile \
  starVLA/model/modules/spatial_forcing/__init__.py \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/modules/spatial_forcing/image_augmentation.py \
  starVLA/model/modules/spatial_forcing/vggt_teacher.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingClean.py \
  tests/test_spatial_forcing_alignment.py \
  tests/test_spatial_forcing_clean.py
bash -n \
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh

echo "SF_FIDELITY_FIX_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
echo "BACKUP_ROOT=${backup_root}"
