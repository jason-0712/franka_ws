#!/usr/bin/env bash
set -euo pipefail

# Install the tutor-requested Qwen-base / frozen-vision training entry points.
# This intentionally does not start training, because the server NVIDIA driver
# must be healthy and GPU 1 must be free first.

stage="${1:-/tmp/starvla_dualcam_74eps_qwen_vision_frozen_patch}"
repo=/home/hanyu/starVLA
data_root=/data/hanyu/quest3_franka_real/snkdjn
dataset_name=quest3_franka_dualcam_pickplace_74eps
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
timestamp=$(date +%Y%m%d_%H%M%S)

train_source="${stage}/third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh"
launcher_source="${stage}/scripts/start_starvla_dualcam_74eps_from_qwen_vision_frozen_train.sh"
for required in \
  "${train_source}" \
  "${launcher_source}" \
  "${data_root}/${dataset_name}/meta/info.json" \
  "${data_root}/${dataset_name}/meta/modality.json" \
  "${data_root}/${dataset_name}/meta/merge_manifest.json" \
  "${repo}/${base_vlm}/config.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

grep -q '"total_episodes": 74' "${data_root}/${dataset_name}/meta/info.json"

train_dir=${repo}/examples/realRobots/Franka/train_files
if [[ ! -d "${train_dir}" ]]; then
  train_dir=${repo}/examples/Franka/train_files
fi
if [[ ! -d "${train_dir}" ]]; then
  echo "Franka training directory not found under ${repo}/examples." >&2
  exit 1
fi

train_target=${train_dir}/run_quest3_franka_train_delta_eef.sh
launcher_target=${repo}/start_starvla_dualcam_74eps_from_qwen_vision_frozen_train.sh
for target in "${train_target}" "${launcher_target}"; do
  if [[ -f "${target}" ]]; then
    cp -a "${target}" "${target}.pre_qwen_vision_frozen_${timestamp}"
  fi
done

install -m 0755 "${train_source}" "${train_target}"
install -m 0755 "${launcher_source}" "${launcher_target}"

bash -n "${train_target}"
bash -n "${launcher_target}"

echo "Installed: ${train_target}"
echo "Installed: ${launcher_target}"
echo "Dataset: ${data_root}/${dataset_name}"
echo "Qwen base: ${repo}/${base_vlm}"
echo "Training was NOT started. After the NVIDIA driver is repaired, run:"
echo "  bash ${launcher_target}"
