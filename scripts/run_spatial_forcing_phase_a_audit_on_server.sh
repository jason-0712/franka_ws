#!/usr/bin/env bash
set -euo pipefail

starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
gpu_id="${GPU_ID:-0}"
tag="${PHASE_A_TAG:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/spatial_forcing_phase_a_audit_${tag}}"
log_dir="${LOG_DIR:-/home/hanyu/starVLA_logs}"
log_path="${log_dir}/spatial_forcing_phase_a_audit_${tag}.log"

baseline_checkpoint="${BASELINE_CHECKPOINT:-/data/hanyu/starVLA_runs/replay94_from_successful_libero74_5k_retry1_20260810/final_model/pytorch_model.pt}"
control_checkpoint="${CONTROL_CHECKPOINT:-/data/hanyu/starVLA_runs/sf_official_fidelity_control_alpha0_500_seed42_20260813_official500_seed42/final_model/pytorch_model.pt}"
treatment_checkpoint="${TREATMENT_CHECKPOINT:-/data/hanyu/starVLA_runs/sf_official_fidelity_treatment_alpha05_500_seed42_20260813_official500_seed42/final_model/pytorch_model.pt}"
teacher_weight="${VGGT_WEIGHT:-/data/hanyu/starVLA_checkpoints/VGGT-1B/model.pt}"
snapshot_root="${SNAPSHOT_ROOT:-/data/hanyu/starVLA_audit_inputs/dav2_cube_shift_20260803_144954}"

for path in \
  "${python_bin}" \
  "${starvla_repo}/spatial_forcing_phase_a_audit.py" \
  "${baseline_checkpoint}" \
  "${control_checkpoint}" \
  "${treatment_checkpoint}" \
  "${teacher_weight}"; do
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: required file missing or empty: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${snapshot_root}" ]]; then
  echo "ERROR: snapshot directory missing: ${snapshot_root}" >&2
  exit 1
fi

used_mib="$(
  nvidia-smi \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits \
    -i "${gpu_id}" | tr -d ' '
)"
echo "GPU ${gpu_id} currently uses ${used_mib} MiB"
if [[ "${used_mib}" -gt 1024 ]]; then
  echo "ERROR: GPU ${gpu_id} is occupied; refusing to start Phase-A audit." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${log_dir}"
cd "${starvla_repo}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"

set +e
"${python_bin}" spatial_forcing_phase_a_audit.py \
  --baseline-checkpoint "${baseline_checkpoint}" \
  --control-checkpoint "${control_checkpoint}" \
  --spatial-checkpoint "${treatment_checkpoint}" \
  --teacher-weight "${teacher_weight}" \
  --snapshot-root "${snapshot_root}" \
  --hidden-indices 8 16 24 32 \
  --projection-steps "${PROJECTION_STEPS:-200}" \
  --projection-learning-rate "${PROJECTION_LR:-0.001}" \
  --projection-hidden-dim "${PROJECTION_HIDDEN_DIM:-2048}" \
  --device cuda:0 \
  --use-bf16 \
  --output-dir "${output_dir}" \
  2>&1 | tee "${log_path}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "SPATIAL_FORCING_PHASE_A_RUN=FAIL status=${status}" >&2
  echo "LOG=${log_path}" >&2
  exit "${status}"
fi
if ! grep -q '^SPATIAL_FORCING_PHASE_A_AUDIT=PASS$' "${log_path}"; then
  echo "ERROR: PASS marker missing from ${log_path}" >&2
  exit 1
fi
if grep -Eiq 'Traceback|OutOfMemoryError|CUDA out of memory|(^|[^[:alnum:]_])(nan)([^[:alnum:]_]|$)' "${log_path}"; then
  echo "ERROR: suspicious failure output in ${log_path}" >&2
  exit 1
fi

echo "SPATIAL_FORCING_PHASE_A_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "OUTPUT_DIR=${output_dir}"
echo "LOG=${log_path}"
