#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-/tmp/starvla_crop_coverage_audit_20260817.tar.gz}"
stage_root="$(mktemp -d)"
trap 'rm -rf "${stage_root}"' EXIT

declare -a files=(
  "scripts/starvla_crop_coverage_audit.py"
  "scripts/test_starvla_crop_coverage_audit.py"
  "scripts/install_starvla_crop_coverage_audit_on_server.sh"
  "scripts/run_starvla_crop_coverage_audit_on_server.sh"
)

for relative_path in "${files[@]}"; do
  source_path="${workspace_root}/${relative_path}"
  if [[ ! -s "${source_path}" ]]; then
    echo "ERROR: missing bundle input: ${source_path}" >&2
    exit 1
  fi
  mode=0644
  if [[ "${relative_path}" == scripts/*.sh ]]; then
    mode=0755
  fi
  install -D -m "${mode}" "${source_path}" "${stage_root}/${relative_path}"
done

mkdir -p "$(dirname "${archive}")"
tar -czf "${archive}" -C "${stage_root}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "STARVLA_CROP_COVERAGE_BUNDLE=PASS"
echo "ARCHIVE=${archive}"
