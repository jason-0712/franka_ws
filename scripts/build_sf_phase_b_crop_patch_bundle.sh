#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/sf_phase_b_crop_patch_20260817.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "third_party/starVLA/starVLA/model/modules/spatial_forcing/crop_patch.py"
  "third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingCropPatch.py"
  "third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_sf_crop_patch.yaml"
  "third_party/starVLA/examples/realRobots/Franka/train_files/run_qwengroot_sf_crop_patch_smoke.sh"
  "third_party/starVLA/tests/test_spatial_forcing_crop_patch.py"
  "scripts/audit_sf_phase_b_crop_patch_smoke.py"
  "scripts/install_sf_phase_b_crop_patch_on_server.sh"
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
echo "SF_PHASE_B_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
