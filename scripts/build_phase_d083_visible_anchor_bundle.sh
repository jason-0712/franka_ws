#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/phase_d083_visible_anchor_20260820.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "scripts/audit_phase_d06_cube_candidates.py"
  "scripts/audit_phase_d08_sam2_prompt_pilot.py"
  "scripts/audit_phase_d081_sam2_bidirectional.py"
  "scripts/export_phase_d082_oracle_anchor_review.py"
  "scripts/export_phase_d083_visible_anchor_neighborhoods.py"
  "scripts/test_audit_phase_d081_sam2_bidirectional.py"
  "scripts/test_export_phase_d083_visible_anchor_neighborhoods.py"
  "scripts/run_phase_d081_sam2_bidirectional_on_server.sh"
  "scripts/run_phase_d083_visible_anchor_export_on_server.sh"
  "scripts/install_phase_d083_visible_anchor_on_server.sh"
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
echo "PHASE_D083_VISIBLE_ANCHOR_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
