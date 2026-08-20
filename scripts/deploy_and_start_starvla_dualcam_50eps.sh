#!/usr/bin/env bash
set -euo pipefail

# Run from /home/dase-hw101/franka_ws after an SSH master connection to the
# training server has been authenticated. Override these with environment
# variables if the server layout changes.
server="${STARVLA_SERVER:-hanyu@192.168.1.113}"
control_path="${STARVLA_SSH_CONTROL_PATH:-/tmp/starvla-codex-ssh}"
remote_repo="${STARVLA_REMOTE_REPO:-/home/hanyu/starVLA}"
remote_data_root="${STARVLA_REMOTE_DATA_ROOT:-/data/hanyu/quest3_franka_real/snkdjn}"

workspace=/home/dase-hw101/franka_ws
archive="${workspace}/quest3_franka_dualcam_pickplace_50eps.tar.gz"
registry="${workspace}/third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py"
registry_loader="${workspace}/third_party/starVLA/starVLA/dataloader/gr00t_lerobot/registry.py"
train_entry="${workspace}/third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh"
train_yaml="${workspace}/third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef.yaml"
launcher="${workspace}/scripts/start_starvla_dualcam_50eps_train.sh"

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
  echo "  ssh -M -S ${control_path} -o ControlPersist=2h -fnNT ${server}" >&2
  exit 1
fi

remote_stage=/tmp/quest3_franka_dualcam_50eps_deploy
ssh "${ssh_opts[@]}" "${server}" \
  "mkdir -p '${remote_stage}' '${remote_data_root}'"

scp "${ssh_opts[@]}" \
  "${archive}" \
  "${registry}" \
  "${registry_loader}" \
  "${train_entry}" \
  "${train_yaml}" \
  "${launcher}" \
  "${server}:${remote_stage}/"

ssh "${ssh_opts[@]}" "${server}" bash -s -- \
  "${remote_repo}" "${remote_data_root}" "${remote_stage}" <<'REMOTE'
set -euo pipefail

remote_repo=$1
remote_data_root=$2
remote_stage=$3
dataset_name=quest3_franka_dualcam_pickplace_50eps
run_id=quest3_franka_dualcam_50eps_finetune_3k
timestamp=$(date +%Y%m%d_%H%M%S)

registry_loader=${remote_repo}/starVLA/dataloader/gr00t_lerobot/registry.py
if [[ ! -f "${registry_loader}" ]]; then
  echo "StarVLA registry loader not found: ${registry_loader}" >&2
  exit 1
fi
cp -a "${registry_loader}" "${registry_loader}.pre_dualcam_${timestamp}"
install -m 0644 "${remote_stage}/registry.py" "${registry_loader}"

train_dir=${remote_repo}/examples/realRobots/Franka/train_files
if [[ ! -d "${train_dir}/data_registry" ]]; then
  train_dir=${remote_repo}/examples/Franka/train_files
fi
if [[ ! -d "${train_dir}/data_registry" ]]; then
  echo "Franka train_files directory not found under ${remote_repo}" >&2
  exit 1
fi

for filename in \
  data_registry/data_config.py \
  run_quest3_franka_train_delta_eef.sh \
  starvla_cotrain_quest3_franka_delta_eef.yaml; do
  target=${train_dir}/${filename}
  if [[ -f "${target}" ]]; then
    cp -a "${target}" "${target}.pre_dualcam_${timestamp}"
  fi
done

install -m 0644 "${remote_stage}/data_config.py" \
  "${train_dir}/data_registry/data_config.py"
install -m 0755 "${remote_stage}/run_quest3_franka_train_delta_eef.sh" \
  "${train_dir}/run_quest3_franka_train_delta_eef.sh"
install -m 0644 "${remote_stage}/starvla_cotrain_quest3_franka_delta_eef.yaml" \
  "${train_dir}/starvla_cotrain_quest3_franka_delta_eef.yaml"
install -m 0755 "${remote_stage}/start_starvla_dualcam_50eps_train.sh" \
  "${remote_repo}/start_starvla_dualcam_50eps_train.sh"

if [[ ! -e "${remote_data_root}/${dataset_name}" ]]; then
  tar -xzf "${remote_stage}/${dataset_name}.tar.gz" -C "${remote_data_root}"
elif [[ ! -f "${remote_data_root}/${dataset_name}/meta/info.json" ]]; then
  echo "Incomplete dataset target already exists: ${remote_data_root}/${dataset_name}" >&2
  exit 1
fi

python3 -m py_compile "${train_dir}/data_registry/data_config.py"
test -f "${remote_data_root}/${dataset_name}/meta/info.json"
test -f "${remote_data_root}/${dataset_name}/meta/modality.json"

cd "${remote_repo}"
log_file=${remote_repo}/results/Checkpoints/${run_id}.launcher.log
mkdir -p "$(dirname "${log_file}")"
nohup bash "${remote_repo}/start_starvla_dualcam_50eps_train.sh" \
  >"${log_file}" 2>&1 < /dev/null &
pid=$!
echo "TRAIN_PID=${pid}"
echo "TRAIN_LOG=${log_file}"
sleep 8
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Training launcher exited during startup." >&2
  tail -n 100 "${log_file}" >&2
  exit 1
fi
tail -n 40 "${log_file}"
REMOTE
