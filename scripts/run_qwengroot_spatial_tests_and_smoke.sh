#!/usr/bin/env bash
set -euo pipefail

# Server-side validation gate for QwenGR00TSpatial.  It runs cheap CPU unit
# tests first, exercises the real frozen Depth Anything V2-Small model, and
# only then starts a 20-optimization-step StarVLA smoke training run.

repo=${STARVLA_REPO:-/home/hanyu/starVLA}
data_root=${DATA_ROOT:-/data/hanyu/quest3_franka_real/snkdjn}
dataset_name=${DATA_MIX:-quest3_franka_dualcam_pickplace_74eps}
checkpoint=${PRETRAINED_CHECKPOINT:-/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt}
base_vlm=${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
run_root=${RUN_ROOT_DIR:-/data/hanyu/starVLA_runs}
run_id=${RUN_ID:-quest3_franka_dualcam_74eps_qwengroot_spatial_smoke20}
gpu_id=${GPU_ID:-0}
main_process_port=${MAIN_PROCESS_PORT:-29610}
min_free_gib=${MIN_FREE_GIB:-30}
smoke_freeze_modules=${SMOKE_FREEZE_MODULES:-qwen_vl_interface}

cd "${repo}"
if [[ -f /home/hanyu/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyu/miniconda3/etc/profile.d/conda.sh
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate starVLA

config_yaml=examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef_spatial.yaml
train_entry=examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh
for required in \
  "${checkpoint}" \
  "${data_root}/${dataset_name}/meta/info.json" \
  "${data_root}/${dataset_name}/meta/modality.json" \
  "${data_root}/${dataset_name}/meta/merge_manifest.json" \
  "${repo}/${base_vlm}/config.json" \
  "${config_yaml}" \
  "${train_entry}" \
  tests/test_qwengroot_spatial.py; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

if [[ -e "${run_root}/${run_id}" ]]; then
  echo "Refusing to overwrite existing smoke run: ${run_root}/${run_id}" >&2
  echo "Set RUN_ID to a fresh name." >&2
  exit 1
fi
if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer, got ${gpu_id}" >&2
  exit 1
fi

gpu_memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')
if [[ "${ALLOW_BUSY_GPU:-0}" != "1" && "${gpu_memory_used}" -gt 1024 ]]; then
  echo "GPU ${gpu_id} is occupied (${gpu_memory_used} MiB); refusing to start smoke training." >&2
  exit 1
fi
available_kib=$(df --output=avail "${run_root}" | tail -1 | tr -d ' ')
required_kib=$((min_free_gib * 1024 * 1024))
if [[ "${available_kib}" -lt "${required_kib}" ]]; then
  echo "Insufficient free storage under ${run_root}: need at least ${min_free_gib} GiB." >&2
  exit 1
fi
if ss -ltn | grep -q ":${main_process_port} "; then
  echo "Accelerate/DeepSpeed port ${main_process_port} is already in use." >&2
  exit 1
fi

echo "[1/3] Running CPU zero-gate/freeze/gradient unit tests"
PYTHONPATH="${repo}:${PYTHONPATH:-}" python tests/test_qwengroot_spatial.py

echo "[2/3] Exercising the real frozen Depth Anything V2-Small model on GPU ${gpu_id}"
CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${repo}:${PYTHONPATH:-}" python - <<'PY'
import torch
from PIL import Image

from starVLA.model.modules.spatial import FrozenDepthAnythingV2Encoder

encoder = FrozenDepthAnythingV2Encoder(
    model_id="depth-anything/Depth-Anything-V2-Small-hf",
    grid_size=(14, 14),
).cuda()
images = [[Image.new("RGB", (224, 224), color=(32, 64, 96)), Image.new("RGB", (224, 224), color=(96, 64, 32))]]
geometry, valid = encoder(images)
assert geometry.shape == (1, 2, 196, 3), geometry.shape
assert valid.shape == (1, 2, 196), valid.shape
assert bool(valid.all())
assert not geometry.requires_grad
assert all(not parameter.requires_grad for parameter in encoder.depth_model.parameters())
print("REAL_DEPTH_ENCODER_SMOKE=PASS", tuple(geometry.shape))
PY

echo "[3/3] Starting 20-step QwenGR00TSpatial training smoke"
WANDB_MODE=disabled \
Framework_name=QwenGR00TSpatial \
config_yaml="${config_yaml}" \
DATA_ROOT="${data_root}" \
DATA_MIX="${dataset_name}" \
RUN_ROOT_DIR="${run_root}" \
RUN_ID="${run_id}" \
MAX_TRAIN_STEPS=20 \
SAVE_INTERVAL=20 \
GPU_IDS="${gpu_id}" \
ACCELERATE_GPU_IDS="${gpu_id}" \
MAIN_PROCESS_PORT="${main_process_port}" \
PER_DEVICE_BATCH_SIZE=1 \
PRETRAINED_CHECKPOINT="${checkpoint}" \
base_vlm="${base_vlm}" \
REPEATED_DIFFUSION_STEPS=1 \
ACTION_MODEL_LR=1e-4 \
QWEN_VL_LR=1e-7 \
GEOMETRY_PROJECTOR_LR=1e-4 \
SPATIAL_FUSER_LR=1e-4 \
BASE_LR=1e-6 \
NUM_WARMUP_STEPS=5 \
FREEZE_MODULES="${smoke_freeze_modules}" \
bash "${train_entry}"

final_checkpoint=${run_root}/${run_id}/final_model/pytorch_model.pt
if [[ ! -s "${final_checkpoint}" ]]; then
  echo "Smoke training ended without a final checkpoint: ${final_checkpoint}" >&2
  exit 1
fi

FINAL_CHECKPOINT="${final_checkpoint}" python - <<'PY'
import os
import torch

path = os.environ["FINAL_CHECKPOINT"]
try:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
except TypeError:
    state = torch.load(path, map_location="cpu")
required = {
    "spatial_fuser.gates",
    "geometry_projector.camera_embedding",
}
missing = sorted(required - set(state))
assert not missing, f"Missing spatial checkpoint keys: {missing}"
assert any(key.startswith("geometry_encoder.depth_model.") for key in state), "Frozen depth model was not saved"
gates = torch.tanh(state["spatial_fuser.gates"].float())
assert torch.isfinite(gates).all(), gates
assert bool((gates.abs() > 0).any()), f"Spatial gates never moved away from zero: {gates.tolist()}"
print("SPATIAL_SMOKE_CHECKPOINT=PASS", path)
print("effective_gates=", gates.tolist())
PY

echo "RESULT=PASS"
echo "RUN_DIR=${run_root}/${run_id}"
