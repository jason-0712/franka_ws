#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/phase_d07_temporal_role_20260818.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "scripts/audit_phase_d0_vggt_capability.py"
  "scripts/audit_phase_d06_cube_candidates.py"
  "scripts/audit_phase_d07_temporal_role_tracking.py"
  "scripts/test_audit_phase_d07_temporal_role_tracking.py"
  "scripts/install_phase_d07_temporal_role_on_server.sh"
  "scripts/run_phase_d07_temporal_role_on_server.sh"
)

for relative in "${files[@]}"; do
  source="${workspace}/${relative}"
  [[ -s "${source}" ]] || { echo "ERROR: missing ${source}" >&2; exit 1; }
  install -D -m 0755 "${source}" "${stage}/${relative}"
done

mkdir -p "$(dirname "${archive}")"
tar -czf "${archive}" -C "${stage}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "PHASE_D07_TEMPORAL_ROLE_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
