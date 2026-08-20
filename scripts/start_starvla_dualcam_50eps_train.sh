#!/usr/bin/env bash
set -euo pipefail

# Run this on the StarVLA GPU server after the merged dataset and the updated
# Franka data registry have been copied there.
cd /home/hanyu/starVLA

if [[ -f /home/hanyu/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyu/miniconda3/etc/profile.d/conda.sh
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate starVLA

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

data_root=/data/hanyu/quest3_franka_real/snkdjn
dataset_name=quest3_franka_dualcam_pickplace_50eps
dataset_path="${data_root}/${dataset_name}"

if [[ ! -f "${dataset_path}/meta/info.json" ]]; then
  echo "Missing merged dataset: ${dataset_path}" >&2
  exit 1
fi

checkpoint=""
for candidate in \
  results/Checkpoints/quest3_franka_fedelta_eef_qwenbase_100eps_lateral11x3_weighted_3k/final_model/pytorch_model.pt \
  results/Checkpoints/quest3_franka_delta_eef_qwenbase_100eps_freezevision_3k/final_model/pytorch_model.pt \
  results/Checkpoints/quest3_franka_delta_eef_from_69eps3k_89eps_dagger20_2k/final_model/pytorch_model.pt; do
  if [[ -f "${candidate}" ]]; then
    checkpoint="${candidate}"
    break
  fi
done

if [[ -z "${checkpoint}" ]]; then
  echo "No compatible Quest3 Franka checkpoint was found." >&2
  exit 1
fi

echo "Dataset: ${dataset_path}"
echo "Checkpoint: ${checkpoint}"

train_script=examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh
if [[ ! -f "${train_script}" ]]; then
  train_script=examples/Franka/train_files/run_quest3_franka_train_delta_eef.sh
fi
if [[ ! -f "${train_script}" ]]; then
  echo "Quest3 Franka training entry point was not found." >&2
  exit 1
fi
echo "Training entry point: ${train_script}"

DATA_ROOT="${data_root}" \
DATA_MIX=quest3_franka_dualcam_pickplace_50eps \
RUN_ID=quest3_franka_dualcam_50eps_finetune_3k \
MAX_TRAIN_STEPS=3000 \
SAVE_INTERVAL=500 \
GPU_IDS=0 \
PER_DEVICE_BATCH_SIZE=1 \
PRETRAINED_CHECKPOINT="${checkpoint}" \
bash "${train_script}"
