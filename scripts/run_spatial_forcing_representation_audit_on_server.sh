#!/usr/bin/env bash
set -euo pipefail

starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
snapshot_root="${SNAPSHOT_ROOT:-/data/hanyu/starVLA_audit_inputs/dav2_cube_shift_20260803_144954}"
teacher_weight="${VGGT_WEIGHT:-/data/hanyu/starVLA_checkpoints/VGGT-1B/model.pt}"

if [[ -n "${GPU_ID:-}" ]]; then
  physical_gpu="${GPU_ID}"
else
  physical_gpu="$({
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -F, '
        {
          gsub(/ /, "", $1)
          gsub(/ /, "", $2)
          gsub(/ /, "", $3)
          if (($2 + 0) < 1024 && ($3 + 0) < 10) {
            print $1
            exit
          }
        }'
  } || true)"
fi

if [[ -z "${physical_gpu}" ]]; then
  echo "ERROR: no GPU with <1024 MiB memory use and <10% utilization is available." >&2
  exit 1
fi

used_mib="$(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "${physical_gpu}" | tr -d ' '
)"
if (( used_mib > 1024 )); then
  echo "ERROR: physical GPU ${physical_gpu} uses ${used_mib} MiB; refusing to share it." >&2
  exit 1
fi

for required in \
  "${teacher_weight}" \
  "${snapshot_root}/center/primary_original.png" \
  "/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/final_model/pytorch_model.pt" \
  "/data/hanyu/starVLA_runs/qwengroot_spatial_forcing_libero74_control500_alpha0_20260804_074900_rgb_inference/final_model/pytorch_model.pt" \
  "/data/hanyu/starVLA_runs/qwengroot_spatial_forcing_libero74_pilot500_alpha01_20260804_064832_rgb_inference/final_model/pytorch_model.pt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required audit input: ${required}" >&2
    exit 1
  fi
done

tag="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/spatial_forcing_representation_audit_${tag}}"

cd "${starvla_repo}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "PHYSICAL_GPU=${physical_gpu}"
echo "LOGICAL_DEVICE=cuda:0"
echo "SNAPSHOT_ROOT=${snapshot_root}"
echo "OUTPUT_DIR=${output_dir}"

"${python_bin}" "${starvla_repo}/spatial_forcing_representation_audit.py" \
  --teacher-weight "${teacher_weight}" \
  --snapshot-root "${snapshot_root}" \
  --device cuda:0 \
  --output-dir "${output_dir}"

echo "SPATIAL_FORCING_PHASE7_AUDIT_RUN=PASS"
echo "OUTPUT_DIR=${output_dir}"

