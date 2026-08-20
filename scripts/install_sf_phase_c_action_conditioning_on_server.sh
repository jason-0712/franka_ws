#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: $0 STAGE_ROOT}"
repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "third_party/starVLA/starVLA/model/modules/spatial_forcing/action_conditioning.py:starVLA/model/modules/spatial_forcing/action_conditioning.py:0644"
  "third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingActionConditioned.py:starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingActionConditioned.py:0644"
  "third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_sf_action_conditioned.yaml:examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_sf_action_conditioned.yaml:0644"
  "third_party/starVLA/examples/realRobots/Franka/train_files/run_qwengroot_sf_action_conditioned_smoke.sh:examples/realRobots/Franka/train_files/run_qwengroot_sf_action_conditioned_smoke.sh:0755"
  "third_party/starVLA/tests/test_spatial_forcing_action_conditioning.py:tests/test_spatial_forcing_action_conditioning.py:0644"
  "scripts/audit_sf_phase_c_action_conditioning_smoke.py:audit_sf_phase_c_action_conditioning_smoke.py:0755"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  remainder="${mapping#*:}"
  target_rel="${remainder%%:*}"
  mode="${remainder##*:}"
  source_path="${stage}/${source_rel}"
  target_path="${repo}/${target_rel}"
  if [[ ! -s "${source_path}" ]]; then
    echo "ERROR: missing staged file: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.before_phase_c_${timestamp}"
  fi
  install -m "${mode}" "${source_path}" "${target_path}"
done

cd "${repo}"
"${python_bin}" -m py_compile \
  starVLA/model/modules/spatial_forcing/action_conditioning.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingActionConditioned.py \
  tests/test_spatial_forcing_action_conditioning.py \
  audit_sf_phase_c_action_conditioning_smoke.py
bash -n examples/realRobots/Franka/train_files/run_qwengroot_sf_action_conditioned_smoke.sh
PYTHONPATH="${repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_spatial_forcing_action_conditioning.py

"${python_bin}" - <<'PY'
import starVLA.model.framework.VLM4A.QwenGR00TSpatialForcingActionConditioned
from starVLA.model.tools import FRAMEWORK_REGISTRY

assert "QwenGR00TSpatialForcingActionConditioned" in FRAMEWORK_REGISTRY.list()
print("SF_PHASE_C_FRAMEWORK_REGISTRY=PASS")
PY

echo "SF_PHASE_C_INSTALL=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "CONTROL_COMMAND=MATCHED_ARM=control GPU_ID=0 bash ${repo}/examples/realRobots/Franka/train_files/run_qwengroot_sf_action_conditioned_smoke.sh"
echo "TREATMENT_COMMAND=MATCHED_ARM=treatment GPU_ID=1 bash ${repo}/examples/realRobots/Franka/train_files/run_qwengroot_sf_action_conditioned_smoke.sh"
