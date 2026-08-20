#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: $0 STAGE_ROOT}"
repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "scripts/audit_phase_d06_cube_candidates.py:audit_phase_d06_cube_candidates.py:0755"
  "scripts/export_phase_d082_oracle_anchor_review.py:export_phase_d082_oracle_anchor_review.py:0755"
  "scripts/test_export_phase_d082_oracle_anchor_review.py:tests/test_export_phase_d082_oracle_anchor_review.py:0755"
  "scripts/run_phase_d082_oracle_anchor_export_on_server.sh:run_phase_d082_oracle_anchor_export_on_server.sh:0755"
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
    cp -a "${target}" "${target}.before_phase_d082_${timestamp}"
  fi
  install -m "${mode}" "${source}" "${target}"
done

cd "${repo}"
"${python_bin}" -m py_compile \
  audit_phase_d06_cube_candidates.py \
  export_phase_d082_oracle_anchor_review.py \
  tests/test_export_phase_d082_oracle_anchor_review.py
PYTHONPATH="${repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_export_phase_d082_oracle_anchor_review.py

echo "PHASE_D082_ORACLE_ANCHOR_INSTALL=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "RUNNER=${repo}/run_phase_d082_oracle_anchor_export_on_server.sh"
