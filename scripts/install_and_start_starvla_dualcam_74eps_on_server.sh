#!/usr/bin/env bash
set -euo pipefail

# Run this script on server1cps after extracting the transfer bundle.  The
# optional first argument is the bundle extraction directory.
stage="${1:-/tmp/starvla_dualcam_74eps_stage}"
repo=/home/hanyu/starVLA
data_root=/data/hanyu/quest3_franka_real/snkdjn
dataset_name=quest3_franka_dualcam_pickplace_74eps
run_root=/data/hanyu/starVLA_runs
run_id=quest3_franka_dualcam_74eps_from_libero30k_10k
checkpoint=/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt
dataset_archive="${stage}/${dataset_name}.tar.gz"
timestamp=$(date +%Y%m%d_%H%M%S)

for required in \
  "${dataset_archive}" \
  "${checkpoint}" \
  "${stage}/scripts/start_starvla_dualcam_74eps_from_libero30k_train.sh" \
  "${stage}/third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py" \
  "${stage}/third_party/starVLA/starVLA/dataloader/gr00t_lerobot/registry.py" \
  "${stage}/third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh" \
  "${stage}/third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef.yaml"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${data_root}" "${run_root}"
dataset_path="${data_root}/${dataset_name}"
if [[ ! -e "${dataset_path}" ]]; then
  echo "Extracting ${dataset_name} ..."
  tar -xzf "${dataset_archive}" -C "${data_root}"
fi

parquet_count=$(find "${dataset_path}/data" -type f -name '*.parquet' | wc -l)
video_count=$(find "${dataset_path}/videos" -type f -name '*.mp4' | wc -l)
if [[ "${parquet_count}" -ne 74 || "${video_count}" -ne 148 ]]; then
  echo "Dataset validation failed: parquet=${parquet_count}, videos=${video_count}" >&2
  exit 1
fi
grep -q '"total_episodes": 74' "${dataset_path}/meta/info.json"
grep -q '"total_frames": 17328' "${dataset_path}/meta/info.json"
test -f "${dataset_path}/meta/modality.json"
test -f "${dataset_path}/meta/merge_manifest.json"

loader_target=${repo}/starVLA/dataloader/gr00t_lerobot/registry.py
test -f "${loader_target}"
cp -a "${loader_target}" "${loader_target}.pre_dualcam74_${timestamp}"
install -m 0644 \
  "${stage}/third_party/starVLA/starVLA/dataloader/gr00t_lerobot/registry.py" \
  "${loader_target}"

train_dir=${repo}/examples/realRobots/Franka/train_files
if [[ ! -d "${train_dir}/data_registry" ]]; then
  train_dir=${repo}/examples/Franka/train_files
fi
test -d "${train_dir}/data_registry"

registry_target=${train_dir}/data_registry/data_config.py
train_target=${train_dir}/run_quest3_franka_train_delta_eef.sh
yaml_target=${train_dir}/starvla_cotrain_quest3_franka_delta_eef.yaml
for target in "${registry_target}" "${train_target}" "${yaml_target}"; do
  if [[ -f "${target}" ]]; then
    cp -a "${target}" "${target}.pre_dualcam74_${timestamp}"
  fi
done
install -m 0644 \
  "${stage}/third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py" \
  "${registry_target}"
install -m 0755 \
  "${stage}/third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh" \
  "${train_target}"
install -m 0644 \
  "${stage}/third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef.yaml" \
  "${yaml_target}"
install -m 0755 \
  "${stage}/scripts/start_starvla_dualcam_74eps_from_libero30k_train.sh" \
  "${repo}/start_starvla_dualcam_74eps_from_libero30k_train.sh"
python3 -m py_compile "${registry_target}"

output_dir="${run_root}/${run_id}"
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing training output: ${output_dir}" >&2
  exit 1
fi

gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
if (( gpu_used > 2048 )); then
  echo "GPU 1 is not free: ${gpu_used} MiB used" >&2
  nvidia-smi -i 1 >&2
  exit 1
fi

log_file="${run_root}/${run_id}.launcher.log"
if [[ -f "${log_file}" ]]; then
  mv "${log_file}" "${log_file}.pre_${timestamp}"
fi
nohup bash "${repo}/start_starvla_dualcam_74eps_from_libero30k_train.sh" \
  >"${log_file}" 2>&1 < /dev/null &
pid=$!
echo "TRAIN_PID=${pid}"
echo "TRAIN_LOG=${log_file}"
sleep 20
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Training launcher exited during startup." >&2
  tail -n 160 "${log_file}" >&2
  exit 1
fi

echo "Training is running. Initial log:"
tail -n 80 "${log_file}"
echo
echo "Monitor with: tail -f ${log_file}"
