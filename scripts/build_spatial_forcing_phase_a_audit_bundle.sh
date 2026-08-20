#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-/tmp/spatial_forcing_phase_a_audit_20260817.tar.gz}"
stage_root="$(mktemp -d)"
trap 'rm -rf "${stage_root}"' EXIT

declare -a files=(
  "third_party/starVLA/starVLA/model/modules/spatial_forcing/phase_a_audit.py"
  "third_party/starVLA/spatial_forcing_phase_a_audit.py"
  "third_party/starVLA/tests/test_spatial_forcing_phase_a_audit.py"
  "scripts/install_spatial_forcing_phase_a_audit_on_server.sh"
  "scripts/run_spatial_forcing_phase_a_audit_on_server.sh"
)

for relative_path in "${files[@]}"; do
  source_path="${workspace_root}/${relative_path}"
  if [[ ! -s "${source_path}" ]]; then
    echo "Missing bundle input: ${source_path}" >&2
    exit 1
  fi
  mode=0644
  if [[ "${relative_path}" == scripts/*.sh ]]; then
    mode=0755
  fi
  install -D -m "${mode}" "${source_path}" "${stage_root}/${relative_path}"
done

tar -czf "${archive}" -C "${stage_root}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "SPATIAL_FORCING_PHASE_A_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
