#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/phase_d0_vggt_capability_v4_20260818.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "scripts/audit_phase_d0_vggt_capability.py"
  "scripts/test_audit_phase_d0_vggt_capability.py"
  "scripts/install_phase_d0_vggt_capability_on_server.sh"
  "scripts/run_phase_d0_vggt_capability_on_server.sh"
)

for relative in "${files[@]}"; do
  source_path="${workspace}/${relative}"
  [[ -s "${source_path}" ]] || {
    echo "ERROR: missing bundle input: ${source_path}" >&2
    exit 1
  }
  install -D -m 0755 "${source_path}" "${stage}/${relative}"
done

mkdir -p "$(dirname "${archive}")"
tar -czf "${archive}" -C "${stage}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "PHASE_D0_VGGT_CAPABILITY_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
