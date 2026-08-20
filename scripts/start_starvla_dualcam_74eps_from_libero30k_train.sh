#!/usr/bin/env bash
set -euo pipefail

# Train a fresh real-robot adaptation from the original LIBERO 30k model.
# The previous 50 real episodes are included in the 74-episode dataset, but
# the previous 50-episode optimizer/model state is deliberately not reused.

repo=/home/hanyu/starVLA
data_root=/data/hanyu/quest3_franka_real/snkdjn
dataset_name=quest3_franka_dualcam_pickplace_74eps
checkpoint=/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt
run_root=/data/hanyu/starVLA_runs
run_id=quest3_franka_dualcam_74eps_from_libero30k_10k

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
  "${checkpoint}" \
  "${dataset_path}/meta/info.json" \
  "${dataset_path}/meta/modality.json" \
  "${dataset_path}/meta/merge_manifest.json"; do
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

echo "SCHEME=A_LIBERO30K_TO_REAL_DUALCAM74"
echo "Dataset: ${dataset_path} (previous 50 + vetted new 24)"
echo "Pretrained checkpoint: ${checkpoint}"
echo "Previous real-Franka checkpoint: NOT USED"
echo "Output: ${output_dir}"
echo "GPU: physical 1"
echo "Optimizer: fresh AdamW"

DATA_ROOT="${data_root}" \
DATA_MIX="${dataset_name}" \
RUN_ROOT_DIR="${run_root}" \
RUN_ID="${run_id}" \
MAX_TRAIN_STEPS=10000 \
SAVE_INTERVAL=2000 \
GPU_IDS=1 \
ACCELERATE_GPU_IDS=0 \
MAIN_PROCESS_PORT=0 \
PER_DEVICE_BATCH_SIZE=1 \
PRETRAINED_CHECKPOINT="${checkpoint}" \
REPEATED_DIFFUSION_STEPS=8 \
ACTION_MODEL_LR=1e-5 \
NUM_WARMUP_STEPS=500 \
FREEZE_MODULES=qwen_vl_interface \
WANDB_PROJECT=starVLA_Quest3_Franka \
bash "${train_script}"
