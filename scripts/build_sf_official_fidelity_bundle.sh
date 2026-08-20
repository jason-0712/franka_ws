#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_repo="${workspace}/third_party/starVLA"
output="${1:-${workspace}/artifacts/sf_official_fidelity_20260813.tar.gz}"
stage="$(mktemp -d /tmp/sf_official_fidelity_bundle.XXXXXX)"
trap 'rm -rf "${stage}"' EXIT

copy_file() {
  local relative="$1"
  mkdir -p "${stage}/$(dirname "${relative}")"
  cp "${source_repo}/${relative}" "${stage}/${relative}"
}

copy_file examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_sf_official_fidelity.yaml
copy_file examples/realRobots/Franka/train_files/run_qwengroot_sf_official_fidelity_smoke.sh
copy_file scripts/run_sf_official_fidelity_matched_smoke.sh
copy_file scripts/install_sf_official_fidelity_on_server.sh
copy_file tests/test_spatial_forcing_official_fidelity.py

mkdir -p "$(dirname "${output}")"
tar -czf "${output}" -C "${stage}" .
sha256sum "${output}"
echo "SF_OFFICIAL_FIDELITY_BUNDLE=${output}"
