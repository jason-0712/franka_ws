#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/phase_d06_cube_candidates_20260818.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "scripts/audit_phase_d06_cube_candidates.py"
  "scripts/test_audit_phase_d06_cube_candidates.py"
  "scripts/install_phase_d06_cube_candidates_on_server.sh"
  "scripts/run_phase_d06_cube_candidates_on_server.sh"
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
echo "PHASE_D06_CUBE_CANDIDATES_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
