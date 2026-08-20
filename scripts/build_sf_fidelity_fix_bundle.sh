#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-/tmp/starvla_sf_fidelity_fix_20260813.tar.gz}"
stage_root="$(mktemp -d)"
trap 'rm -rf "${stage_root}"' EXIT

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
  mode=0644
  if [[ "${relative_path}" == *.sh ]]; then
    mode=0755
  fi
  install -D -m "${mode}" \
    "${workspace_root}/third_party/starVLA/${relative_path}" \
    "${stage_root}/third_party/starVLA/${relative_path}"
done

for relative_path in \
  scripts/export_spatial_forcing_rgb_view.py \
  scripts/run_spatial_forcing_phase10_tests_on_server.sh \
  scripts/run_sf_fidelity_smoke_matrix_on_server.sh \
  scripts/install_sf_fidelity_fix_on_server.sh; do
  install -D -m 0755 \
    "${workspace_root}/${relative_path}" \
    "${stage_root}/${relative_path}"
done

tar -czf "${archive}" -C "${stage_root}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "SF_FIDELITY_FIX_BUNDLE=${archive}"
