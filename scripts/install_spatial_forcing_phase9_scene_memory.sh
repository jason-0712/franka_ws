#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_spatial_forcing_phase9_scene_memory.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"

if [[ ! -d "${starvla_repo}/starVLA" ]]; then
  echo "StarVLA repository not found: ${starvla_repo}" >&2
  exit 1
fi

for relative_path in \
  starVLA/model/modules/spatial_forcing/__init__.py \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  starVLA/training/train_starvla.py \
  spatial_forcing_representation_audit.py \
  examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_spatial_forcing.yaml \
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh \
  tests/test_spatial_forcing_alignment.py; do
  source_path="${stage_root}/third_party/starVLA/${relative_path}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing staged source: ${source_path}" >&2
    exit 1
  fi
  install -D -m 0644 "${source_path}" "${starvla_repo}/${relative_path}"
done

chmod 0755 \
  "${starvla_repo}/examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_smoke.sh"

install -D -m 0755 \
  "${stage_root}/scripts/run_spatial_forcing_phase9_tests_on_server.sh" \
  "${starvla_repo}/run_spatial_forcing_phase9_tests_on_server.sh"

cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"
python -m py_compile \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/modules/spatial_forcing/__init__.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  starVLA/training/train_starvla.py \
  spatial_forcing_representation_audit.py \
  tests/test_spatial_forcing_alignment.py

echo "SPATIAL_FORCING_PHASE9_SCENE_MEMORY_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
