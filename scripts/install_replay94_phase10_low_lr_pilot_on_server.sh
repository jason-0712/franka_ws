#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: $0 STAGE_ROOT}"
repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "third_party/starVLA/examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh:examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh:0755"
  "third_party/starVLA/tests/test_spatial_forcing_clean.py:tests/test_spatial_forcing_clean.py:0644"
  "scripts/run_spatial_forcing_phase10_tests_on_server.sh:run_spatial_forcing_phase10_tests_on_server.sh:0755"
  "scripts/start_replay94_phase10_low_lr_pilot.sh:start_replay94_phase10_low_lr_pilot.sh:0755"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  remainder="${mapping#*:}"
  target_rel="${remainder%%:*}"
  mode="${remainder##*:}"
  source_path="${stage}/${source_rel}"
  target_path="${repo}/${target_rel}"
  if [[ ! -s "${source_path}" ]]; then
    echo "Missing staged input: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.before_lowlr_pilot_${timestamp}"
  fi
  install -m "${mode}" "${source_path}" "${target_path}"
done

cd "${repo}"
bash -n \
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh \
  start_replay94_phase10_low_lr_pilot.sh

PATH="/home/hanyu/miniconda3/envs/starVLA/bin:${PATH}" \
  bash "${repo}/run_spatial_forcing_phase10_tests_on_server.sh"

echo "REPLAY94_PHASE10_LOWLR_INSTALL=PASS"
echo "LAUNCHER=${repo}/start_replay94_phase10_low_lr_pilot.sh"

