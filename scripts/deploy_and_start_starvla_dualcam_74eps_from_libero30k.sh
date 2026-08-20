#!/usr/bin/env bash
set -euo pipefail

server="${STARVLA_SERVER:-hanyu@192.168.1.113}"
control_path="${STARVLA_SSH_CONTROL_PATH:-/tmp/starvla-codex-ssh}"
remote_repo="${STARVLA_REMOTE_REPO:-/home/hanyu/starVLA}"
remote_data_root="${STARVLA_REMOTE_DATA_ROOT:-/data/hanyu/quest3_franka_real/snkdjn}"
run_root="${STARVLA_RUN_ROOT:-/data/hanyu/starVLA_runs}"

workspace=/home/dase-hw101/franka_ws
dataset_name=quest3_franka_dualcam_pickplace_74eps
run_id=quest3_franka_dualcam_74eps_from_libero30k_10k
archive="${workspace}/${dataset_name}.tar.gz"
registry="${workspace}/third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py"
registry_loader="${workspace}/third_party/starVLA/starVLA/dataloader/gr00t_lerobot/registry.py"
train_entry="${workspace}/third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh"
train_yaml="${workspace}/third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef.yaml"
launcher="${workspace}/scripts/start_starvla_dualcam_74eps_from_libero30k_train.sh"

for path in "${archive}" "${registry}" "${registry_loader}" "${train_entry}" "${train_yaml}" "${launcher}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required local file: ${path}" >&2
    exit 1
  fi
done

ssh_opts=(-o "ControlPath=${control_path}" -o BatchMode=yes)
if ! ssh "${ssh_opts[@]}" "${server}" true; then
  echo "No authenticated SSH master at ${control_path}." >&2
  echo "Create it first with:" >&2
  echo "  ssh -M -S ${control_path} -o ControlPersist=8h -fnNT ${server}" >&2
  exit 1
fi

remote_stage=/tmp/quest3_franka_dualcam_74eps_deploy
ssh "${ssh_opts[@]}" "${server}" "mkdir -p '${remote_stage}' '${remote_data_root}' '${run_root}'"
scp "${ssh_opts[@]}" \
  "${archive}" \
  "${registry}" \
  "${registry_loader}" \
  "${train_entry}" \
  "${train_yaml}" \
  "${launcher}" \
  "${server}:${remote_stage}/"

ssh "${ssh_opts[@]}" "${server}" bash -s -- \
  "${remote_repo}" "${remote_data_root}" "${run_root}" "${dataset_name}" "${run_id}" "${remote_stage}" <<'REMOTE'
set -euo pipefail

remote_repo=$1
remote_data_root=$2
run_root=$3
dataset_name=$4
run_id=$5
remote_stage=$6
checkpoint=/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt
timestamp=$(date +%Y%m%d_%H%M%S)

test -f "${checkpoint}"
if [[ -e "${run_root}/${run_id}" ]]; then
  echo "Training output already exists: ${run_root}/${run_id}" >&2
  exit 1
fi

gpu1_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
if (( gpu1_used > 2048 )); then
  echo "GPU 1 is not free: ${gpu1_used} MiB already used" >&2
  nvidia-smi -i 1 >&2
  exit 1
fi

target_dataset=${remote_data_root}/${dataset_name}
if [[ ! -e "${target_dataset}" ]]; then
  tar -xzf "${remote_stage}/${dataset_name}.tar.gz" -C "${remote_data_root}"
fi
for required in \
  "${target_dataset}/meta/info.json" \
  "${target_dataset}/meta/modality.json" \
  "${target_dataset}/meta/merge_manifest.json"; do
  test -f "${required}"
done

loader_target=${remote_repo}/starVLA/dataloader/gr00t_lerobot/registry.py
test -f "${loader_target}"
cp -a "${loader_target}" "${loader_target}.pre_dualcam74_${timestamp}"
install -m 0644 "${remote_stage}/registry.py" "${loader_target}"

train_dir=${remote_repo}/examples/realRobots/Franka/train_files
if [[ ! -d "${train_dir}/data_registry" ]]; then
  train_dir=${remote_repo}/examples/Franka/train_files
fi
test -d "${train_dir}/data_registry"

for mapping in \
  "data_config.py:data_registry/data_config.py:0644" \
  "run_quest3_franka_train_delta_eef.sh:run_quest3_franka_train_delta_eef.sh:0755" \
  "starvla_cotrain_quest3_franka_delta_eef.yaml:starvla_cotrain_quest3_franka_delta_eef.yaml:0644"; do
  IFS=: read -r source relative mode <<<"${mapping}"
  target=${train_dir}/${relative}
  if [[ -f "${target}" ]]; then
    cp -a "${target}" "${target}.pre_dualcam74_${timestamp}"
  fi
  install -m "${mode}" "${remote_stage}/${source}" "${target}"
done
install -m 0755 "${remote_stage}/start_starvla_dualcam_74eps_from_libero30k_train.sh" \
  "${remote_repo}/start_starvla_dualcam_74eps_from_libero30k_train.sh"

python3 -m py_compile "${train_dir}/data_registry/data_config.py"
log_file=${run_root}/${run_id}.launcher.log
nohup bash "${remote_repo}/start_starvla_dualcam_74eps_from_libero30k_train.sh" \
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

grep -E \
  "SCHEME=|Dataset:|Pretrained checkpoint:|Previous real-Franka|Using pretrained checkpoint:|Using data mix:|Repeated diffusion steps:|Step [0-9]+, Loss" \
  "${log_file}" | tail -n 40 || true
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
REMOTE
