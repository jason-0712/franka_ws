#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-/tmp/spatial_forcing_phase9_scene_memory_20260805.tar.gz}"
stage_root="$(mktemp -d)"
trap 'rm -rf "${stage_root}"' EXIT

for relative_path in \
  starVLA/model/modules/spatial_forcing/__init__.py \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  starVLA/training/train_starvla.py \
  spatial_forcing_representation_audit.py \
  examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_spatial_forcing.yaml \
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh \
  tests/test_spatial_forcing_alignment.py; do
  install -D -m 0644 \
    "${workspace_root}/third_party/starVLA/${relative_path}" \
    "${stage_root}/third_party/starVLA/${relative_path}"
done

for script_name in \
  install_spatial_forcing_phase9_scene_memory.sh \
  run_spatial_forcing_phase9_tests_on_server.sh; do
  install -D -m 0755 \
    "${workspace_root}/scripts/${script_name}" \
    "${stage_root}/scripts/${script_name}"
done

tar -czf "${archive}" -C "${stage_root}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "SPATIAL_FORCING_PHASE9_BUNDLE=${archive}"
