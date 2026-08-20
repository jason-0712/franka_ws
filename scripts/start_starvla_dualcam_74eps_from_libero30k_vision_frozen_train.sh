#!/usr/bin/env bash
set -euo pipefail

# Fair initialization comparison against the completed Qwen-base run:
#   - use the same vetted 74 real-Franka episodes;
#   - load the full compatible LIBERO 30k StarVLA checkpoint;
#   - do not load the previous real-Franka 50-episode checkpoint;
#   - freeze only the Qwen visual tower;
#   - keep the Qwen-baseline training budget and learning rates unchanged;
#   - start with a fresh optimizer (this is initialization, not trainer resume).

repo=${STARVLA_REPO:-/home/hanyu/starVLA}
data_root=${DATA_ROOT:-/data/hanyu/quest3_franka_real/snkdjn}
dataset_name=${DATA_MIX:-quest3_franka_dualcam_pickplace_74eps}
checkpoint=${PRETRAINED_CHECKPOINT:-/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt}
base_vlm=${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
run_root=${RUN_ROOT_DIR:-/data/hanyu/starVLA_runs}
run_id=${RUN_ID:-quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3}
physical_gpu=${GPU_ID:-0}
main_process_port=${MAIN_PROCESS_PORT:-29502}
max_train_steps=${MAX_TRAIN_STEPS:-20000}
# A full checkpoint is roughly 9-10 GiB.  Saving every 2k steps creates about
# 100 GiB of model copies over a 20k run.  Five-thousand-step checkpoints keep
# useful intermediate models while reducing the expected storage requirement.
save_interval=${SAVE_INTERVAL:-5000}

cd "${repo}"

if [[ -f /home/hanyu/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyu/miniconda3/etc/profile.d/conda.sh
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate starVLA

dataset_path=${data_root}/${dataset_name}
for required in \
  "${checkpoint}" \
  "${dataset_path}/meta/info.json" \
  "${dataset_path}/meta/modality.json" \
  "${dataset_path}/meta/merge_manifest.json" \
  "${repo}/${base_vlm}/config.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

output_dir=${run_root}/${run_id}
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing output: ${output_dir}" >&2
  echo "Choose a fresh RUN_ID; preserve failed runs for diagnosis." >&2
  exit 1
fi
mkdir -p "${run_root}"

if ! [[ "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative physical GPU index: ${physical_gpu}" >&2
  exit 1
fi

gpu_memory_used=$(nvidia-smi \
  --query-gpu=memory.used \
  --format=csv,noheader,nounits \
  -i "${physical_gpu}" | tr -d ' ')
if [[ "${ALLOW_BUSY_GPU:-0}" != "1" && "${gpu_memory_used}" -gt 1024 ]]; then
  echo "Refusing to use physical GPU ${physical_gpu}: ${gpu_memory_used} MiB is already allocated." >&2
  exit 1
fi

if ss -ltn | grep -q ":${main_process_port} "; then
  echo "Accelerate/DeepSpeed port ${main_process_port} is already in use." >&2
  exit 1
fi

train_script=examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh
if [[ ! -f "${train_script}" ]]; then
  train_script=examples/Franka/train_files/run_quest3_franka_train_delta_eef.sh
fi
if [[ ! -f "${train_script}" ]]; then
  echo "Quest3 Franka training entry point was not found." >&2
  exit 1
fi

echo "SCHEME=LIBERO30K_TO_REAL_DUALCAM74_VISION_FROZEN_FAIR_20K"
echo "Dataset: ${dataset_path}"
echo "Initialization checkpoint: ${checkpoint}"
echo "Previous real-Franka checkpoint: NOT USED"
echo "Frozen module: qwen_vl_interface.model.model.visual"
echo "Output: ${output_dir}"
echo "Physical GPU: ${physical_gpu} (${gpu_memory_used} MiB used before launch)"
echo "Accelerate/DeepSpeed port: ${main_process_port}"
echo "Train/save steps: ${max_train_steps}/${save_interval}"
echo "Optimizer: fresh AdamW"

DATA_ROOT="${data_root}" \
DATA_MIX="${dataset_name}" \
RUN_ROOT_DIR="${run_root}" \
RUN_ID="${run_id}" \
MAX_TRAIN_STEPS="${max_train_steps}" \
SAVE_INTERVAL="${save_interval}" \
GPU_IDS="${physical_gpu}" \
ACCELERATE_GPU_IDS=0 \
MAIN_PROCESS_PORT="${main_process_port}" \
PER_DEVICE_BATCH_SIZE=1 \
PRETRAINED_CHECKPOINT="${checkpoint}" \
base_vlm="${base_vlm}" \
REPEATED_DIFFUSION_STEPS=8 \
ACTION_MODEL_LR=1e-4 \
QWEN_VL_LR=1e-7 \
BASE_LR=1e-6 \
NUM_WARMUP_STEPS=1000 \
FREEZE_MODULES=qwen_vl_interface.model.model.visual \
WANDB_PROJECT=starVLA_Quest3_Franka \
bash "${train_script}"
