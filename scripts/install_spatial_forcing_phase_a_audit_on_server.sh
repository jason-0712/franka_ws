#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: $0 STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ ! -s "${starvla_repo}/spatial_forcing_representation_audit.py" ]]; then
  echo "Missing existing Phase-7 dependency: ${starvla_repo}/spatial_forcing_representation_audit.py" >&2
  exit 1
fi

declare -a mappings=(
  "third_party/starVLA/starVLA/model/modules/spatial_forcing/phase_a_audit.py:starVLA/model/modules/spatial_forcing/phase_a_audit.py"
  "third_party/starVLA/spatial_forcing_phase_a_audit.py:spatial_forcing_phase_a_audit.py"
  "third_party/starVLA/tests/test_spatial_forcing_phase_a_audit.py:tests/test_spatial_forcing_phase_a_audit.py"
  "scripts/run_spatial_forcing_phase_a_audit_on_server.sh:run_spatial_forcing_phase_a_audit_on_server.sh"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  target_rel="${mapping#*:}"
  source_path="${stage_root}/${source_rel}"
  target_path="${starvla_repo}/${target_rel}"
  if [[ ! -s "${source_path}" ]]; then
    echo "Missing staged file: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.before_phase_a_${timestamp}"
  fi
  install -m 0644 "${source_path}" "${target_path}"
done

chmod 0755 \
  "${starvla_repo}/spatial_forcing_phase_a_audit.py" \
  "${starvla_repo}/run_spatial_forcing_phase_a_audit_on_server.sh"

cd "${starvla_repo}"
"${python_bin}" -m py_compile \
  spatial_forcing_phase_a_audit.py \
  starVLA/model/modules/spatial_forcing/phase_a_audit.py \
  tests/test_spatial_forcing_phase_a_audit.py

PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_spatial_forcing_phase_a_audit.py

echo "SPATIAL_FORCING_PHASE_A_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
echo "RUNNER=${starvla_repo}/run_spatial_forcing_phase_a_audit_on_server.sh"
