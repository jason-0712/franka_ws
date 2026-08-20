#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/phase_d082_oracle_anchor_20260820.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "scripts/audit_phase_d06_cube_candidates.py"
  "scripts/export_phase_d082_oracle_anchor_review.py"
  "scripts/test_export_phase_d082_oracle_anchor_review.py"
  "scripts/install_phase_d082_oracle_anchor_on_server.sh"
  "scripts/run_phase_d082_oracle_anchor_export_on_server.sh"
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
echo "PHASE_D082_ORACLE_ANCHOR_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
