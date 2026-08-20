#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: $0 STAGE_ROOT}"
repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "third_party/starVLA/starVLA/model/modules/spatial_forcing/action_conditioning.py:starVLA/model/modules/spatial_forcing/action_conditioning.py:0644"
  "third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingActionConditioned.py:starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingActionConditioned.py:0644"
  "third_party/starVLA/tests/test_spatial_forcing_action_conditioning.py:tests/test_spatial_forcing_action_conditioning.py:0644"
  "scripts/audit_sf_phase_c_bf16_survival.py:audit_sf_phase_c_bf16_survival.py:0755"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  remainder="${mapping#*:}"
  target_rel="${remainder%%:*}"
  mode="${remainder##*:}"
  source_path="${stage}/${source_rel}"
  target_path="${repo}/${target_rel}"
  [[ -s "${source_path}" ]] || {
    echo "ERROR: missing staged file: ${source_path}" >&2
    exit 1
  }
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.before_bf16_audit_${timestamp}"
  fi
  install -m "${mode}" "${source_path}" "${target_path}"
done

cd "${repo}"
"${python_bin}" -m py_compile \
  starVLA/model/modules/spatial_forcing/action_conditioning.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingActionConditioned.py \
  tests/test_spatial_forcing_action_conditioning.py \
  audit_sf_phase_c_bf16_survival.py
PYTHONPATH="${repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_spatial_forcing_action_conditioning.py
"${python_bin}" audit_sf_phase_c_bf16_survival.py --help >/dev/null

echo "SF_PHASE_C_BF16_AUDIT_INSTALL=PASS"
echo "ROBOT_COMMANDS_SENT=0"
