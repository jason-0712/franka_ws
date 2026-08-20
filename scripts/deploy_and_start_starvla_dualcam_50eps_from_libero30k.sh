#!/usr/bin/env bash
set -euo pipefail

server="${STARVLA_SERVER:-hanyu@192.168.1.113}"
control_path="${STARVLA_SSH_CONTROL_PATH:-/tmp/starvla-codex-ssh}"
remote_repo="${STARVLA_REMOTE_REPO:-/home/hanyu/starVLA}"
run_root="${STARVLA_RUN_ROOT:-/data/hanyu/starVLA_runs}"
run_id=quest3_franka_dualcam_50eps_from_libero30k_10k

workspace=/home/dase-hw101/franka_ws
train_entry="${workspace}/third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh"
launcher="${workspace}/scripts/start_starvla_dualcam_50eps_from_libero30k_train.sh"

for path in "${train_entry}" "${launcher}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required local file: ${path}" >&2
    exit 1
  fi
done

ssh_opts=(-o "ControlPath=${control_path}" -o BatchMode=yes)
if ! ssh "${ssh_opts[@]}" "${server}" true; then
  echo "No authenticated SSH master at ${control_path}." >&2
  exit 1
fi

remote_stage=/tmp/quest3_franka_dualcam_from_libero30k_deploy
ssh "${ssh_opts[@]}" "${server}" "mkdir -p '${remote_stage}' '${run_root}'"
scp "${ssh_opts[@]}" "${train_entry}" "${launcher}" "${server}:${remote_stage}/"

ssh "${ssh_opts[@]}" "${server}" bash -s -- \
  "${remote_repo}" "${run_root}" "${run_id}" "${remote_stage}" <<'REMOTE'
set -euo pipefail

remote_repo=$1
run_root=$2
run_id=$3
remote_stage=$4
checkpoint=/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt
dataset=/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_pickplace_50eps
timestamp=$(date +%Y%m%d_%H%M%S)

test -f "${checkpoint}"
test -f "${dataset}/meta/info.json"
test -f "${dataset}/meta/modality.json"

if [[ -e "${run_root}/${run_id}" ]]; then
  echo "Scheme A run already exists: ${run_root}/${run_id}" >&2
  exit 1
fi

gpu1_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
if (( gpu1_used > 2048 )); then
  echo "GPU 1 is not free: ${gpu1_used} MiB already used" >&2
  exit 1
fi

train_dir=${remote_repo}/examples/realRobots/Franka/train_files
if [[ ! -d "${train_dir}" ]]; then
  train_dir=${remote_repo}/examples/Franka/train_files
fi
test -d "${train_dir}"

remote_train=${train_dir}/run_quest3_franka_train_delta_eef.sh
if [[ -f "${remote_train}" ]]; then
  cp -a "${remote_train}" "${remote_train}.pre_scheme_a_${timestamp}"
fi
install -m 0755 "${remote_stage}/run_quest3_franka_train_delta_eef.sh" "${remote_train}"
install -m 0755 "${remote_stage}/start_starvla_dualcam_50eps_from_libero30k_train.sh" \
  "${remote_repo}/start_starvla_dualcam_50eps_from_libero30k_train.sh"

log_file=${run_root}/${run_id}.launcher.log
nohup bash "${remote_repo}/start_starvla_dualcam_50eps_from_libero30k_train.sh" \
  >"${log_file}" 2>&1 < /dev/null &
pid=$!
echo "TRAIN_PID=${pid}"
echo "TRAIN_LOG=${log_file}"
sleep 20
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Scheme A launcher exited during startup." >&2
  tail -n 160 "${log_file}" >&2
  exit 1
fi

grep -E \
  "SCHEME=|Pretrained checkpoint:|Previous single-camera|Using pretrained checkpoint:|Repeated diffusion steps:|Action-model learning rate|Warmup-step override|loaded <full_model>|Loaded pretrained|Step [0-9]+, Loss" \
  "${log_file}" | tail -n 40 || true
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
REMOTE
