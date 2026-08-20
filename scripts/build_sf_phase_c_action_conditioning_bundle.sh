#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/sf_phase_c_action_conditioning_20260818.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "third_party/starVLA/starVLA/model/modules/spatial_forcing/action_conditioning.py"
  "third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingActionConditioned.py"
  "third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_sf_action_conditioned.yaml"
  "third_party/starVLA/examples/realRobots/Franka/train_files/run_qwengroot_sf_action_conditioned_smoke.sh"
  "third_party/starVLA/tests/test_spatial_forcing_action_conditioning.py"
  "scripts/audit_sf_phase_c_action_conditioning_smoke.py"
  "scripts/install_sf_phase_c_action_conditioning_on_server.sh"
)

for relative in "${files[@]}"; do
  source_path="${workspace}/${relative}"
  if [[ ! -s "${source_path}" ]]; then
    echo "ERROR: missing bundle input: ${source_path}" >&2
    exit 1
  fi
  mode=0644
  if [[ "${relative}" == *.sh || "${relative}" == scripts/*.py ]]; then
    mode=0755
  fi
  install -D -m "${mode}" "${source_path}" "${stage}/${relative}"
done

mkdir -p "$(dirname "${archive}")"
tar -czf "${archive}" -C "${stage}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "SF_PHASE_C_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
