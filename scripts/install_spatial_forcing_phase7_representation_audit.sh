#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_spatial_forcing_phase7_representation_audit.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
audit_input_root="${AUDIT_INPUT_ROOT:-/data/hanyu/starVLA_audit_inputs/dav2_cube_shift_20260803_144954}"

for relative_path in \
  starVLA/model/modules/spatial_forcing/representation_audit.py \
  tests/test_spatial_forcing_representation_audit.py \
  spatial_forcing_representation_audit.py; do
  source_path="${stage_root}/third_party/starVLA/${relative_path}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing staged source: ${source_path}" >&2
    exit 1
  fi
  install -D -m 0644 "${source_path}" "${starvla_repo}/${relative_path}"
done
chmod 0755 "${starvla_repo}/spatial_forcing_representation_audit.py"

for script_name in \
  run_spatial_forcing_representation_audit_on_server.sh \
  run_spatial_forcing_representation_audit_tests.sh; do
  install -D -m 0755 \
    "${stage_root}/scripts/${script_name}" \
    "${starvla_repo}/${script_name}"
done

for label in center primary_front primary_back primary_left primary_right; do
  for view in primary wrist; do
    source_image="${stage_root}/audit_inputs/${label}/${view}_original.png"
    if [[ ! -s "${source_image}" ]]; then
      echo "Missing held-out audit image: ${source_image}" >&2
      exit 1
    fi
    install -D -m 0644 \
      "${source_image}" \
      "${audit_input_root}/${label}/${view}_original.png"
  done
done

"/home/hanyu/miniconda3/envs/starVLA/bin/python" -m py_compile \
  "${starvla_repo}/starVLA/model/modules/spatial_forcing/representation_audit.py" \
  "${starvla_repo}/spatial_forcing_representation_audit.py" \
  "${starvla_repo}/tests/test_spatial_forcing_representation_audit.py"

echo "SPATIAL_FORCING_PHASE7_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
echo "AUDIT_INPUT_ROOT=${audit_input_root}"

