#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: $0 STAGE_ROOT}"
repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "scripts/audit_phase_d0_vggt_capability.py:audit_phase_d0_vggt_capability.py:0755"
  "scripts/test_audit_phase_d0_vggt_capability.py:tests/test_audit_phase_d0_vggt_capability.py:0755"
  "scripts/run_phase_d0_vggt_capability_on_server.sh:run_phase_d0_vggt_capability_on_server.sh:0755"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  remainder="${mapping#*:}"
  target_rel="${remainder%%:*}"
  mode="${remainder##*:}"
  source_path="${stage_root}/${source_rel}"
  target_path="${repo}/${target_rel}"
  [[ -s "${source_path}" ]] || {
    echo "ERROR: missing staged file: ${source_path}" >&2
    exit 1
  }
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.before_phase_d0_${timestamp}"
  fi
  install -m "${mode}" "${source_path}" "${target_path}"
done

[[ -s "${repo}/vggt/models/vggt.py" ]] || {
  echo "ERROR: official VGGT source is absent: ${repo}/vggt/models/vggt.py" >&2
  exit 1
}

cd "${repo}"
"${python_bin}" -m py_compile \
  audit_phase_d0_vggt_capability.py \
  tests/test_audit_phase_d0_vggt_capability.py
PYTHONPATH="${repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_audit_phase_d0_vggt_capability.py
PYTHONPATH="${repo}:${PYTHONPATH:-}" "${python_bin}" - <<'PY'
import inspect
from vggt.models.vggt import VGGT

signature = inspect.signature(VGGT)
required = {"enable_point", "enable_depth", "enable_track", "feature_only"}
missing = sorted(required.difference(signature.parameters))
if missing:
    raise RuntimeError(f"VGGT constructor lacks full-head switches: {missing}")
print("VGGT_FULL_HEAD_INTERFACE=PASS")
print("VGGT_SIGNATURE=", signature)
PY

echo "PHASE_D0_VGGT_CAPABILITY_INSTALL=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "RUNNER=${repo}/run_phase_d0_vggt_capability_on_server.sh"
