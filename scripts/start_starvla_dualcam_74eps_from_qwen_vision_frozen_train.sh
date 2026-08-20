#!/usr/bin/env bash
set -euo pipefail

# Tutor-requested experiment:
#   - mix all vetted 74 real Franka episodes;
#   - initialize only from the Qwen3-VL base model;
#   - do not load LIBERO or previous real-Franka VLA checkpoints;
#   - freeze only the Qwen visual tower, not the full Qwen-VL interface;
#   - initialize the GR00T/DiT action model from scratch.

repo=/home/hanyu/starVLA
data_root=/data/hanyu/quest3_franka_real/snkdjn
dataset_name=quest3_franka_dualcam_pickplace_74eps
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
run_root=/data/hanyu/starVLA_runs
run_id=quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k

cd "${repo}"

if [[ -f /home/hanyu/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyu/miniconda3/etc/profile.d/conda.sh
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate starVLA

dataset_path="${data_root}/${dataset_name}"
for required in \
  "${dataset_path}/meta/info.json" \
  "${dataset_path}/meta/modality.json" \
  "${dataset_path}/meta/merge_manifest.json" \
  "${repo}/${base_vlm}/config.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

output_dir="${run_root}/${run_id}"
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing output: ${output_dir}" >&2
  exit 1
fi
mkdir -p "${run_root}"

train_script=examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh
if [[ ! -f "${train_script}" ]]; then
  train_script=examples/Franka/train_files/run_quest3_franka_train_delta_eef.sh
fi
if [[ ! -f "${train_script}" ]]; then
  echo "Quest3 Franka training entry point was not found." >&2
  exit 1
fi

echo "SCHEME=QWEN_BASE_TO_REAL_DUALCAM74_VISION_FROZEN"
echo "Dataset: ${dataset_path} (all vetted 74 episodes mixed together)"
echo "Qwen base: ${repo}/${base_vlm}"
echo "LIBERO checkpoint: NOT USED"
echo "Previous real-Franka checkpoint: NOT USED"
echo "Frozen module: qwen_vl_interface.model.model.visual"
echo "Action model: randomly initialized"
echo "Output: ${output_dir}"
echo "GPU: physical 1"
echo "Optimizer: fresh AdamW"

DATA_ROOT="${data_root}" \
DATA_MIX="${dataset_name}" \
RUN_ROOT_DIR="${run_root}" \
RUN_ID="${run_id}" \
MAX_TRAIN_STEPS=20000 \
SAVE_INTERVAL=2000 \
GPU_IDS=1 \
ACCELERATE_GPU_IDS=0 \
MAIN_PROCESS_PORT=0 \
PER_DEVICE_BATCH_SIZE=1 \
NO_PRETRAINED_CHECKPOINT=1 \
base_vlm="${base_vlm}" \
REPEATED_DIFFUSION_STEPS=8 \
ACTION_MODEL_LR=1e-4 \
QWEN_VL_LR=1e-7 \
BASE_LR=1e-6 \
NUM_WARMUP_STEPS=1000 \
FREEZE_MODULES=qwen_vl_interface.model.model.visual \
WANDB_PROJECT=starVLA_Quest3_Franka \
bash "${train_script}"
