#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
dataset="${DATASET_ROOT:-/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_replay_94eps_v1}"
tag="${TAG:-$(date +%Y%m%d_%H%M%S)}"
output="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/phase_d06_cube_candidates_${tag}}"
log="${LOG:-/home/hanyu/starVLA_logs/phase_d06_cube_candidates_${tag}.log}"

[[ -d "${dataset}" ]] || { echo "ERROR: dataset absent: ${dataset}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "ERROR: output exists: ${output}" >&2; exit 1; }
mkdir -p "$(dirname "${log}")"

set -o pipefail
"${python_bin}" "${repo}/audit_phase_d06_cube_candidates.py" \
  --dataset-root "${dataset}" \
  --output-dir "${output}" \
  --episodes "${EPISODES:-10}" \
  --input-size 518 \
  --top-k 3 \
  2>&1 | tee "${log}"

tar -czf "${output}/phase_d06_review_artifacts.tar.gz" \
  -C "${output}" \
  primary_cube_candidate_contact_sheet.jpg \
  wrist_cube_candidate_contact_sheet.jpg \
  cube_candidates_manual_review.csv \
  phase_d06_cube_candidate_audit.json

echo "PHASE_D06_CUBE_CANDIDATES_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "OUTPUT_DIR=${output}"
echo "REVIEW_ARCHIVE=${output}/phase_d06_review_artifacts.tar.gz"
echo "LOG=${log}"
