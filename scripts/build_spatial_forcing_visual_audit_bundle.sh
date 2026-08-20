#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${workspace}/artifacts/spatial_forcing_visual_audit_20260811.tar.gz}"

mkdir -p "$(dirname "${output}")"
cd "${workspace}"

tar -czf "${output}" \
  third_party/starVLA/starVLA/model/modules/spatial_forcing/representation_audit.py \
  third_party/starVLA/spatial_forcing_representation_audit.py \
  third_party/starVLA/tests/test_spatial_forcing_representation_audit.py \
  scripts/plot_spatial_forcing_position_heatmaps.py \
  scripts/plot_starvla_action_vector_field.py \
  scripts/starvla_requery_saved_snapshot.py \
  scripts/add_replay94_baseline_to_vector_field.py \
  scripts/install_spatial_forcing_visual_audit_on_server.sh

sha256sum "${output}"
ls -lh "${output}"
echo "SPATIAL_FORCING_VISUAL_AUDIT_BUNDLE=PASS"
echo "BUNDLE=${output}"
