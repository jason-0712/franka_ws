#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace}/artifacts/phase_d081_sam2_bidirectional_20260820.tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

declare -a files=(
  "scripts/audit_phase_d06_cube_candidates.py"
  "scripts/audit_phase_d08_sam2_prompt_pilot.py"
  "scripts/audit_phase_d081_sam2_bidirectional.py"
  "scripts/finalize_phase_d081_sam2_review.py"
  "scripts/test_audit_phase_d081_sam2_bidirectional.py"
  "scripts/test_finalize_phase_d081_sam2_review.py"
  "scripts/install_phase_d081_sam2_bidirectional_on_server.sh"
  "scripts/run_phase_d081_sam2_bidirectional_on_server.sh"
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
echo "PHASE_D081_SAM2_BIDIRECTIONAL_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
