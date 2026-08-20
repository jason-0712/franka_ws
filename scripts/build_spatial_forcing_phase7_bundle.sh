#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-/tmp/spatial_forcing_phase7_representation_audit_20260804.tar.gz}"
stage_root="$(mktemp -d)"
trap 'rm -rf "${stage_root}"' EXIT

for relative_path in \
  starVLA/model/modules/spatial_forcing/representation_audit.py \
  tests/test_spatial_forcing_representation_audit.py \
  spatial_forcing_representation_audit.py; do
  install -D -m 0644 \
    "${workspace_root}/third_party/starVLA/${relative_path}" \
    "${stage_root}/third_party/starVLA/${relative_path}"
done

for script_name in \
  install_spatial_forcing_phase7_representation_audit.sh \
  run_spatial_forcing_representation_audit_on_server.sh \
  run_spatial_forcing_representation_audit_tests.sh; do
  install -D -m 0755 \
    "${workspace_root}/scripts/${script_name}" \
    "${stage_root}/scripts/${script_name}"
done

snapshot_source="${workspace_root}/dataset/.sensitivity/dav2_cube_shift_20260803_144954"
for label in center primary_front primary_back primary_left primary_right; do
  for view in primary wrist; do
    install -D -m 0644 \
      "${snapshot_source}/${label}/${view}_original.png" \
      "${stage_root}/audit_inputs/${label}/${view}_original.png"
  done
done

tar -czf "${archive}" -C "${stage_root}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "SPATIAL_FORCING_PHASE7_BUNDLE=${archive}"

