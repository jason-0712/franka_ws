#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
runner="${repo}/examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh"
baseline="${PRETRAINED_CHECKPOINT:-/data/hanyu/starVLA_runs/replay94_from_successful_libero74_5k_retry1_20260810/final_model/pytorch_model.pt}"
data_root="${DATA_ROOT_DIR:-/data/hanyu/quest3_franka_real/snkdjn}"
data_mix="${DATA_MIX:-quest3_franka_dualcam_replay_94eps_v1}"
run_root="${RUN_ROOT_DIR:-/data/hanyu/starVLA_runs}"
gpu="${GPU_ID:-0}"
port_base="${MAIN_PROCESS_PORT_BASE:-29830}"
tag="${SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"

for required in "${runner}" "${baseline}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done
if [[ ! -d "${data_root}" ]]; then
  echo "Dataset root does not exist: ${data_root}" >&2
  exit 1
fi

read -r used_mib total_mib < <(
  nvidia-smi \
    --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits \
    -i "${gpu}" | tr -d ','
)
echo "GPU ${gpu} currently uses ${used_mib}/${total_mib} MiB"
if (( used_mib > 1024 )); then
  echo "ERROR: GPU ${gpu} is occupied; no smoke jobs were started." >&2
  exit 1
fi

read -r -a sources <<< "${STUDENT_FEATURE_SOURCES:-llm_hidden vision_projector vision_encoder}"
if (( ${#sources[@]} == 0 )); then
  echo "ERROR: STUDENT_FEATURE_SOURCES selected no smoke jobs." >&2
  exit 1
fi
for source in "${sources[@]}"; do
  case "${source}" in
    llm_hidden|vision_projector|vision_encoder) ;;
    *)
      echo "ERROR: unsupported student feature source: ${source}" >&2
      exit 1
      ;;
  esac
done
index=0
for source in "${sources[@]}"; do
  run_id="sf_fidelity_${source}_treatment_smoke20_${tag}"
  log="/home/hanyu/starVLA_logs/${run_id}.log"
  checkpoint="${run_root}/${run_id}/final_model/pytorch_model.pt"
  mkdir -p "$(dirname "${log}")"

  if [[ -s "${checkpoint}" ]] \
      && [[ -f "${log}" ]] \
      && grep -Fq 'SPATIAL_FORCING_CLEAN_SMOKE=PASS' "${log}"; then
    echo "SF_FIDELITY_SMOKE=SKIP_ALREADY_PASS source=${source} checkpoint=${checkpoint}"
    index=$((index + 1))
    continue
  fi

  echo "===== START ${source} ====="
  MATCHED_ARM=treatment \
  PROJECTED_ALIGNMENT_ALPHA=0.1 \
  STUDENT_FEATURE_SOURCE="${source}" \
  TEACHER_USE_POSITIONAL_EMBEDDING=false \
  TEACHER_VIEW_MODE=independent \
  IMAGE_AUGMENTATION_ENABLED=true \
  INFERENCE_CENTER_CROP_ENABLED=true \
  PRETRAINED_CHECKPOINT="${baseline}" \
  DATA_ROOT_DIR="${data_root}" \
  DATA_MIX="${data_mix}" \
  RUN_ROOT_DIR="${run_root}" \
  RUN_ID="${run_id}" \
  GPU_ID="${gpu}" \
  MAIN_PROCESS_PORT="$((port_base + index))" \
  MAX_TRAIN_STEPS=20 \
  NUM_WARMUP_STEPS=5 \
  SAVE_INTERVAL=21 \
  EVAL_INTERVAL=20 \
  ACTION_MODEL_LR=3e-5 \
  QWEN_VL_LR=1e-5 \
  ALIGNMENT_HEAD_LR=1e-4 \
  BASE_LR=1e-6 \
  WANDB_MODE=disabled \
  bash "${runner}" 2>&1 | tee "${log}"

  if [[ ! -s "${checkpoint}" ]]; then
    echo "ERROR: missing smoke checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if grep -Eiq \
      'Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory|(^|[^[:alnum:]_])(loss|mse_score)[^[:cntrl:]]*(:|=)[[:space:]]*(nan|[+-]?inf)([^[:alnum:]_]|$)' \
      "${log}"; then
    echo "ERROR: suspicious output in ${log}" >&2
    exit 1
  fi
  echo "SF_FIDELITY_SMOKE=PASS source=${source} checkpoint=${checkpoint}"
  index=$((index + 1))
done

echo "SF_FIDELITY_SMOKE_MATRIX=PASS"
echo "SMOKE_TAG=${tag}"
echo "ROBOT_COMMANDS_SENT=0"
