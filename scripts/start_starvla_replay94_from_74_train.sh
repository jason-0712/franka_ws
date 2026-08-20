#!/usr/bin/env bash
set -euo pipefail

# Replay fine-tuning: preserve the successful 74-episode policy while adding
# 10 Front + 10 Back demonstrations.  All 94 episodes are replayed together;
# this script never fine-tunes on the 20 new episodes alone.

repo=${STARVLA_REPO:-/home/hanyu/starVLA}
data_root=${DATA_ROOT:-/data/hanyu/quest3_franka_real/snkdjn}
dataset_name=${DATA_MIX:-quest3_franka_dualcam_replay_94eps_v1}
checkpoint=${PRETRAINED_CHECKPOINT:-/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/final_model/pytorch_model.pt}
base_vlm=${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
run_root=${RUN_ROOT_DIR:-/data/hanyu/starVLA_runs}
run_id=${RUN_ID:-quest3_franka_replay94_from_libero74_5k_seed42_$(date +%Y%m%d_%H%M%S)}
physical_gpu=${GPU_ID:-0}
main_process_port=${MAIN_PROCESS_PORT:-29810}
max_train_steps=${MAX_TRAIN_STEPS:-5000}
save_interval=${SAVE_INTERVAL:-$((max_train_steps + 1))}
warmup_steps=${NUM_WARMUP_STEPS:-250}
action_model_lr=${ACTION_MODEL_LR:-3e-5}
qwen_vl_lr=${QWEN_VL_LR:-1e-7}
base_lr=${BASE_LR:-1e-6}

cd "${repo}"

if [[ -f /home/hanyu/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyu/miniconda3/etc/profile.d/conda.sh
fi
conda activate starVLA

dataset_path=${data_root}/${dataset_name}
for required in \
  "${checkpoint}" \
  "${dataset_path}/meta/info.json" \
  "${dataset_path}/meta/modality.json" \
  "${dataset_path}/meta/merge_manifest.json" \
  "${repo}/${base_vlm}/config.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

python - "${dataset_path}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.load(open(root / "meta" / "info.json"))
manifest = json.load(open(root / "meta" / "merge_manifest.json"))
assert info["total_episodes"] == 94, info["total_episodes"]
assert info["total_videos"] == 188, info["total_videos"]
assert manifest["base_episode_count"] == 74
assert manifest["append_group_counts"] == {"front": 10, "back": 10}
assert len(manifest["episodes"]) == 94
identities = [(x["source_dataset"], x["source_episode_index"]) for x in manifest["episodes"]]
assert len(set(identities)) == 94
print("REPLAY94_PREFLIGHT=PASS")
PY

output_dir=${run_root}/${run_id}
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing output: ${output_dir}" >&2
  exit 1
fi
mkdir -p "${run_root}"

available_kb=$(df -Pk "${run_root}" | awk 'NR==2 {print $4}')
if (( available_kb < 15728640 )); then
  echo "ERROR: less than 15 GiB is free on the run filesystem." >&2
  df -h "${run_root}"
  exit 1
fi

if ! [[ "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative physical GPU index: ${physical_gpu}" >&2
  exit 1
fi
gpu_memory_used=$(nvidia-smi \
  --query-gpu=memory.used \
  --format=csv,noheader,nounits \
  -i "${physical_gpu}" | tr -d ' ')
if [[ "${ALLOW_BUSY_GPU:-0}" != "1" && "${gpu_memory_used}" -gt 1024 ]]; then
  echo "Refusing GPU ${physical_gpu}: ${gpu_memory_used} MiB is already allocated." >&2
  exit 1
fi
if ss -ltn | grep -q ":${main_process_port} "; then
  echo "Accelerate/DeepSpeed port ${main_process_port} is already in use." >&2
  exit 1
fi

train_script=examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh
if [[ ! -f "${train_script}" ]]; then
  echo "Quest3 Franka training entry point was not found: ${train_script}" >&2
  exit 1
fi

echo "SCHEME=REPLAY94_FROM_SUCCESSFUL_LIBERO74"
echo "DATASET=${dataset_path}"
echo "INITIALIZATION=${checkpoint}"
echo "TRAINING_DATA=74_OLD_PLUS_20_NEW_REPLAYED_TOGETHER"
echo "PHYSICAL_GPU=${physical_gpu}"
echo "RUN_ID=${run_id}"
echo "MAX_STEPS=${max_train_steps}"
echo "LEARNING_RATES=action:${action_model_lr},qwen:${qwen_vl_lr},base:${base_lr}"
echo "FROZEN_MODULE=qwen_vl_interface.model.model.visual"
echo "OPTIMIZER=fresh_AdamW"

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
ACTION_MODEL_LR="${action_model_lr}" \
QWEN_VL_LR="${qwen_vl_lr}" \
BASE_LR="${base_lr}" \
NUM_WARMUP_STEPS="${warmup_steps}" \
FREEZE_MODULES=qwen_vl_interface.model.model.visual \
WANDB_PROJECT=starVLA_Quest3_Franka \
bash "${train_script}"

final_checkpoint=${output_dir}/final_model/pytorch_model.pt
if [[ ! -s "${final_checkpoint}" ]]; then
  echo "Training finished without expected final checkpoint: ${final_checkpoint}" >&2
  exit 1
fi
echo "REPLAY94_TRAINING=PASS"
echo "REPLAY94_RUN_DIR=${output_dir}"
echo "REPLAY94_CHECKPOINT=${final_checkpoint}"
