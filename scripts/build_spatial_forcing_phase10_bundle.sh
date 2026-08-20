#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-/tmp/spatial_forcing_phase10_clean_20260805.tar.gz}"
stage_root="$(mktemp -d)"
trap 'rm -rf "${stage_root}"' EXIT

relative_paths=(
  starVLA/model/modules/spatial_forcing/__init__.py
  starVLA/model/modules/spatial_forcing/alignment.py
  starVLA/model/modules/spatial_forcing/image_augmentation.py
  starVLA/model/modules/spatial_forcing/lora_student.py
  starVLA/model/modules/spatial_forcing/vggt_teacher.py
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingClean.py
  starVLA/training/train_starvla.py
  starVLA/training/trainer_utils/trainer_tools.py
  examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_spatial_forcing_clean.yaml
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh
  examples/realRobots/Franka/train_files/data_registry/data_config.py
  tests/test_spatial_forcing_alignment.py
  tests/test_spatial_forcing_lora.py
  tests/test_spatial_forcing_clean.py
)

for relative_path in "${relative_paths[@]}"; do
  install -D -m 0644 \
    "${workspace_root}/third_party/starVLA/${relative_path}" \
    "${stage_root}/third_party/starVLA/${relative_path}"
done

for script_name in \
  export_spatial_forcing_rgb_view.py \
  install_spatial_forcing_phase10_clean.sh \
  run_spatial_forcing_phase10_tests_on_server.sh; do
  install -D -m 0755 \
    "${workspace_root}/scripts/${script_name}" \
    "${stage_root}/scripts/${script_name}"
done

install -D -m 0644 \
  "${workspace_root}/PHASE10_CLEAN_REPRODUCTION.md" \
  "${stage_root}/PHASE10_CLEAN_REPRODUCTION.md"

tar -czf "${archive}" -C "${stage_root}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "SPATIAL_FORCING_PHASE10_BUNDLE=${archive}"
