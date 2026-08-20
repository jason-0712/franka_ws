#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: $0 STAGE_ROOT}"
repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "scripts/audit_phase_d06_cube_candidates.py:audit_phase_d06_cube_candidates.py:0755"
  "scripts/audit_phase_d08_sam2_prompt_pilot.py:audit_phase_d08_sam2_prompt_pilot.py:0755"
  "scripts/finalize_phase_d08_sam2_review.py:finalize_phase_d08_sam2_review.py:0755"
  "scripts/test_audit_phase_d08_sam2_prompt_pilot.py:tests/test_audit_phase_d08_sam2_prompt_pilot.py:0755"
  "scripts/test_finalize_phase_d08_sam2_review.py:tests/test_finalize_phase_d08_sam2_review.py:0755"
  "scripts/prepare_phase_d08_sam2_on_server.sh:prepare_phase_d08_sam2_on_server.sh:0755"
  "scripts/run_phase_d08_sam2_pilot_on_server.sh:run_phase_d08_sam2_pilot_on_server.sh:0755"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  remainder="${mapping#*:}"
  target_rel="${remainder%%:*}"
  mode="${remainder##*:}"
  source="${stage}/${source_rel}"
  target="${repo}/${target_rel}"
  [[ -s "${source}" ]] || { echo "ERROR: missing ${source}" >&2; exit 1; }
  mkdir -p "$(dirname "${target}")"
  if [[ -e "${target}" ]]; then
    cp -a "${target}" "${target}.before_phase_d08_${timestamp}"
  fi
  install -m "${mode}" "${source}" "${target}"
done

cd "${repo}"
"${python_bin}" -m py_compile \
  audit_phase_d06_cube_candidates.py \
  audit_phase_d08_sam2_prompt_pilot.py \
  finalize_phase_d08_sam2_review.py \
  tests/test_audit_phase_d08_sam2_prompt_pilot.py \
  tests/test_finalize_phase_d08_sam2_review.py
PYTHONPATH="${repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_audit_phase_d08_sam2_prompt_pilot.py
PYTHONPATH="${repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_finalize_phase_d08_sam2_review.py

echo "PHASE_D08_SAM2_PILOT_INSTALL=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "PREPARE=${repo}/prepare_phase_d08_sam2_on_server.sh"
echo "RUNNER=${repo}/run_phase_d08_sam2_pilot_on_server.sh"
