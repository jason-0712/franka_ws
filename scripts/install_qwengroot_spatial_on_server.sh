#!/usr/bin/env bash
set -euo pipefail

stage=${1:-/tmp/qwengroot_spatial_stage}
repo=${STARVLA_REPO:-/home/hanyu/starVLA}
timestamp=$(date +%Y%m%d_%H%M%S)

relative_files=(
  starVLA/model/framework/VLM4A/QwenGR00TSpatial.py
  starVLA/model/modules/spatial/__init__.py
  starVLA/model/modules/spatial/depth_anything_v2.py
  starVLA/model/modules/spatial/gated_cross_attention.py
  examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef_spatial.yaml
  tests/test_qwengroot_spatial.py
)

install_existing_if_pristine() {
  local relative=$1
  local expected_original_sha=$2
  local source_path=${stage}/third_party/starVLA/${relative}
  local target_path=${repo}/${relative}
  local source_sha current_sha

  if [[ ! -f "${source_path}" || ! -f "${target_path}" ]]; then
    echo "Missing existing-file source/target for safe update: ${relative}" >&2
    exit 1
  fi
  source_sha=$(sha256sum "${source_path}" | awk '{print $1}')
  current_sha=$(sha256sum "${target_path}" | awk '{print $1}')
  if [[ "${current_sha}" == "${source_sha}" ]]; then
    echo "Already installed: ${target_path}"
    return
  fi
  if [[ "${current_sha}" != "${expected_original_sha}" ]]; then
    echo "Refusing to overwrite a locally modified server file: ${target_path}" >&2
    echo "Current SHA256: ${current_sha}" >&2
    echo "Expected original SHA256: ${expected_original_sha}" >&2
    exit 1
  fi
  cp -a "${target_path}" "${target_path}.pre_spatial_${timestamp}"
  install -m 0644 "${source_path}" "${target_path}"
}

install_existing_if_pristine \
  starVLA/training/trainer_utils/trainer_tools.py \
  ecd1d7d2f7a5f3607feec30b8e9ba78e2978f9dd1aa701f37642a11ef8133306
install_existing_if_pristine \
  examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh \
  d1b733279e5e9d6cd02c2b6ebbe9f761d18e01b7573c72ad84a3ac42a1e52edb

for relative in "${relative_files[@]}"; do
  source_path=${stage}/third_party/starVLA/${relative}
  target_path=${repo}/${relative}
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing staged source: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -f "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.pre_spatial_${timestamp}"
  fi
  install -m 0644 "${source_path}" "${target_path}"
done
chmod 0755 "${repo}/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh"

install -m 0755 \
  "${stage}/scripts/run_qwengroot_spatial_tests_and_smoke.sh" \
  "${repo}/run_qwengroot_spatial_tests_and_smoke.sh"

python -m py_compile \
  "${repo}/starVLA/model/framework/VLM4A/QwenGR00TSpatial.py" \
  "${repo}/starVLA/model/modules/spatial/depth_anything_v2.py" \
  "${repo}/starVLA/model/modules/spatial/gated_cross_attention.py" \
  "${repo}/tests/test_qwengroot_spatial.py"
bash -n "${repo}/run_qwengroot_spatial_tests_and_smoke.sh"
bash -n "${repo}/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh"

echo "Installed QwenGR00TSpatial into ${repo}"
echo "Backups of replaced files use suffix .pre_spatial_${timestamp}"
echo "Training has not started yet. Run:"
echo "  GPU_ID=0 bash ${repo}/run_qwengroot_spatial_tests_and_smoke.sh"
