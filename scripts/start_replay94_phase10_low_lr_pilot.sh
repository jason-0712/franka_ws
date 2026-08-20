#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
matched_arm="${MATCHED_ARM:?Set MATCHED_ARM=control or treatment}"
case "${matched_arm}" in
  control)
    default_alpha=0.0
    default_gpu=0
    default_port=29910
    ;;
  treatment)
    default_alpha=0.1
    default_gpu=1
    default_port=29911
    ;;
  *)
    echo "MATCHED_ARM must be control or treatment, got: ${matched_arm}" >&2
    exit 1
    ;;
esac

export STARVLA_REPO="${repo}"
export PATH="/home/hanyu/miniconda3/envs/starVLA/bin:${PATH}"
export MATCHED_ARM="${matched_arm}"
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/data/hanyu/starVLA_runs/replay94_from_successful_libero74_5k_retry1_20260810/final_model/pytorch_model.pt}"
export DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data/hanyu/quest3_franka_real/snkdjn}"
export DATA_MIX="${DATA_MIX:-quest3_franka_dualcam_replay_94eps_v1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-${MAX_TRAIN_STEPS}}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-50}"
export ACTION_MODEL_LR="${ACTION_MODEL_LR:-3e-5}"
export QWEN_VL_LR="${QWEN_VL_LR:-1e-5}"
export ALIGNMENT_HEAD_LR="${ALIGNMENT_HEAD_LR:-1e-4}"
export BASE_LR="${BASE_LR:-1e-6}"
export IMAGE_AUGMENTATION_ENABLED="${IMAGE_AUGMENTATION_ENABLED:-true}"
export PROJECTED_ALIGNMENT_ALPHA="${PROJECTED_ALIGNMENT_ALPHA:-${default_alpha}}"
export GPU_ID="${GPU_ID:-${default_gpu}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${default_port}}"
export ALLOW_80GB_GPU="${ALLOW_80GB_GPU:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RUN_ID="${RUN_ID:-replay94_phase10_lowlr_${matched_arm}_alpha${PROJECTED_ALIGNMENT_ALPHA}_${MAX_TRAIN_STEPS}step_seed42_$(date +%Y%m%d_%H%M%S)}"
export TMPDIR="${TMPDIR:-/data/hanyu/tmp_phase10_lowlr_${matched_arm}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/hanyu/torch_extensions_phase10_lowlr_${matched_arm}}"

mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"

echo "REPLAY94_PHASE10_LOWLR_PILOT=START"
echo "MATCHED_ARM=${MATCHED_ARM}"
echo "RUN_ID=${RUN_ID}"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "DATA_MIX=${DATA_MIX}"
echo "GPU_ID=${GPU_ID}"
echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS}"
echo "ACTION_MODEL_LR=${ACTION_MODEL_LR}"
echo "QWEN_VL_LR=${QWEN_VL_LR}"
echo "ALIGNMENT_HEAD_LR=${ALIGNMENT_HEAD_LR}"
echo "BASE_LR=${BASE_LR}"
echo "IMAGE_AUGMENTATION_ENABLED=${IMAGE_AUGMENTATION_ENABLED}"
echo "PROJECTED_ALIGNMENT_ALPHA=${PROJECTED_ALIGNMENT_ALPHA}"

exec bash "${repo}/start_spatial_forcing_replay94_from_74_train.sh"

