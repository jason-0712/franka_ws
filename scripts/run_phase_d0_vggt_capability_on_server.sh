#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
dataset_root="${DATASET_ROOT:-/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_replay_94eps_v1}"
weight="${VGGT_WEIGHT:-/data/hanyu/starVLA_checkpoints/VGGT-1B/model.pt}"
expected_sha="${VGGT_WEIGHT_SHA256:-d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0}"
gpu_id="${GPU_ID:-1}"
episode_index="${EPISODE_INDEX:-0}"
minimum_free_mib="${MINIMUM_FREE_MIB:-60000}"
tag="${TAG:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/phase_d0_vggt_capability_${tag}}"
log="${LOG:-/home/hanyu/starVLA_logs/phase_d0_vggt_capability_${tag}.log}"

[[ -s "${repo}/audit_phase_d0_vggt_capability.py" ]] || {
  echo "ERROR: audit script is absent" >&2
  exit 1
}
[[ -d "${dataset_root}" ]] || {
  echo "ERROR: dataset is absent: ${dataset_root}" >&2
  exit 1
}
[[ -s "${weight}" ]] || {
  echo "ERROR: VGGT weight is absent: ${weight}" >&2
  exit 1
}
[[ ! -e "${output_dir}" ]] || {
  echo "ERROR: output already exists: ${output_dir}" >&2
  exit 1
}

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < minimum_free_mib )); then
  echo "ERROR: GPU ${gpu_id} free=${free_mib:-unknown} MiB; require at least ${minimum_free_mib} MiB" >&2
  exit 1
fi

mkdir -p "$(dirname "${log}")"
echo "GPU_ID=${gpu_id} free_mib=${free_mib}"
echo "DATASET_ROOT=${dataset_root}"
echo "EPISODE_INDEX=${episode_index}"
echo "OUTPUT_DIR=${output_dir}"
echo "LOG=${log}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${repo}:${PYTHONPATH:-}"

set -o pipefail
"${python_bin}" "${repo}/audit_phase_d0_vggt_capability.py" \
  --dataset-root "${dataset_root}" \
  --weight "${weight}" \
  --expected-weight-sha256 "${expected_sha}" \
  --output-dir "${output_dir}" \
  --episode-index "${episode_index}" \
  --device cuda:0 \
  --input-size 518 \
  --query-points 8 \
  2>&1 | tee "${log}"

"${python_bin}" - "${output_dir}/phase_d0_vggt_capability.json" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path))
print("===== PHASE D-0 DECISION =====")
for key in (
    "weight_coverage_pass",
    "geometry_heads_pass",
    "tracking_head_pass",
    "task_relative_signal_pass",
    "capability_ready",
):
    print(f"{key}={payload[key]}")
for view, result in payload["view_results"].items():
    geometry = result["query_geometry"]
    print(
        f"view={view} query_phase={result['query_phase']} "
        f"reprojection_px={geometry.get('first_frame_reprojection_error_px_mean')} "
        f"in_bounds={geometry.get('in_bounds_fraction')} "
        f"positive_depth={geometry.get('positive_depth_fraction')}"
    )
print("NEXT_ROUTE=IMPLEMENT_TASK_RELATIVE_TEACHER" if payload["capability_ready"] else "NEXT_ROUTE=STOP_AND_DIAGNOSE_VGGT_HEADS")
PY

echo "PHASE_D0_VGGT_CAPABILITY_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "RESULT=${output_dir}/phase_d0_vggt_capability.json"
echo "LOG=${log}"
