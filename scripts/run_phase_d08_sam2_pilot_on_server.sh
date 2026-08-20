#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
sam2_commit="${SAM2_COMMIT:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
sam2_repo="${SAM2_REPO:-/home/hanyu/third_party/sam2-${sam2_commit:0:8}}"
checkpoint="${SAM2_CHECKPOINT:-/data/hanyu/starVLA_checkpoints/SAM2.1/sam2.1_hiera_large.pt}"
dataset="${DATASET_ROOT:-/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_replay_94eps_v1}"
gpu_id="${GPU_ID:-1}"
minimum_free_mib="${MINIMUM_FREE_MIB:-20000}"
tag="${TAG:-$(date +%Y%m%d_%H%M%S)}"
output="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/phase_d08_sam2_prompt_pilot_${tag}}"
log="${LOG:-/home/hanyu/starVLA_logs/phase_d08_sam2_prompt_pilot_${tag}.log}"

[[ -d "${dataset}" ]] || { echo "ERROR: dataset absent: ${dataset}" >&2; exit 1; }
[[ -d "${sam2_repo}/.git" ]] || { echo "ERROR: run SAM2 prepare first: ${sam2_repo}" >&2; exit 1; }
[[ -s "${checkpoint}" ]] || { echo "ERROR: SAM2 checkpoint absent: ${checkpoint}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "ERROR: output exists: ${output}" >&2; exit 1; }
actual_commit="$(git -C "${sam2_repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${sam2_commit}" ]] || {
  echo "ERROR: SAM2 commit mismatch: ${actual_commit}" >&2
  exit 1
}
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < minimum_free_mib )); then
  echo "ERROR: GPU ${gpu_id} free=${free_mib:-unknown} MiB; require ${minimum_free_mib} MiB" >&2
  exit 1
fi

declare -a extra_args=()
if [[ -n "${PROMPT_OVERRIDES:-}" ]]; then
  [[ -s "${PROMPT_OVERRIDES}" ]] || { echo "ERROR: prompt overrides absent: ${PROMPT_OVERRIDES}" >&2; exit 1; }
  extra_args+=(--prompt-overrides "${PROMPT_OVERRIDES}")
fi
if [[ -n "${EPISODE_INDICES:-}" ]]; then
  extra_args+=(--episode-indices "${EPISODE_INDICES}")
fi

mkdir -p "$(dirname "${log}")"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${sam2_repo}:${repo}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set -o pipefail
"${python_bin}" "${repo}/audit_phase_d08_sam2_prompt_pilot.py" \
  --dataset-root "${dataset}" \
  --sam2-repo "${sam2_repo}" \
  --sam2-checkpoint "${checkpoint}" \
  --output-dir "${output}" \
  --episodes "${EPISODES:-10}" \
  --device cuda:0 \
  "${extra_args[@]}" \
  2>&1 | tee "${log}"

tar -czf "${output}/phase_d08_review_artifacts.tar.gz" \
  -C "${output}" \
  primary_sam2_mask_contact_sheet.jpg \
  sam2_mask_phase_metrics.csv \
  phase_d08_sam2_prompt_pilot.json \
  manual_review_decisions.json

echo "PHASE_D08_SAM2_PROMPT_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "OUTPUT_DIR=${output}"
echo "REVIEW_ARCHIVE=${output}/phase_d08_review_artifacts.tar.gz"
echo "LOG=${log}"
