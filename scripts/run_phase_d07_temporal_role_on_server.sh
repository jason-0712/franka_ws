#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
dataset="${DATASET_ROOT:-/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_replay_94eps_v1}"
weight="${VGGT_WEIGHT:-/data/hanyu/starVLA_checkpoints/VGGT-1B/model.pt}"
sha="${VGGT_WEIGHT_SHA256:-d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0}"
gpu_id="${GPU_ID:-1}"
minimum_free_mib="${MINIMUM_FREE_MIB:-50000}"
tag="${TAG:-$(date +%Y%m%d_%H%M%S)}"
output="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/phase_d07_temporal_role_${tag}}"
log="${LOG:-/home/hanyu/starVLA_logs/phase_d07_temporal_role_${tag}.log}"

[[ -d "${dataset}" ]] || { echo "ERROR: dataset absent: ${dataset}" >&2; exit 1; }
[[ -s "${weight}" ]] || { echo "ERROR: VGGT weight absent: ${weight}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "ERROR: output exists: ${output}" >&2; exit 1; }
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < minimum_free_mib )); then
  echo "ERROR: GPU ${gpu_id} free=${free_mib:-unknown} MiB; require ${minimum_free_mib} MiB" >&2
  exit 1
fi
mkdir -p "$(dirname "${log}")"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${repo}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set -o pipefail
"${python_bin}" "${repo}/audit_phase_d07_temporal_role_tracking.py" \
  --dataset-root "${dataset}" \
  --weight "${weight}" \
  --expected-weight-sha256 "${sha}" \
  --output-dir "${output}" \
  --episodes "${EPISODES:-10}" \
  --device cuda:0 \
  --top-k 3 \
  --points-per-candidate 5 \
  2>&1 | tee "${log}"

tar -czf "${output}/phase_d07_review_artifacts.tar.gz" \
  -C "${output}" \
  primary_temporal_role_tracking_contact_sheet.jpg \
  temporal_role_candidate_metrics.csv \
  phase_d07_temporal_role_tracking.json

echo "PHASE_D07_TEMPORAL_ROLE_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "OUTPUT_DIR=${output}"
echo "REVIEW_ARCHIVE=${output}/phase_d07_review_artifacts.tar.gz"
echo "LOG=${log}"
