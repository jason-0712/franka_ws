#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:-${workspace}/qwengroot_spatial_smoke_20260803.tar}

cd "${workspace}"
tar -cf "${output}" \
  third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00TSpatial.py \
  third_party/starVLA/starVLA/model/modules/spatial/__init__.py \
  third_party/starVLA/starVLA/model/modules/spatial/depth_anything_v2.py \
  third_party/starVLA/starVLA/model/modules/spatial/gated_cross_attention.py \
  third_party/starVLA/starVLA/training/trainer_utils/trainer_tools.py \
  third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh \
  third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef_spatial.yaml \
  third_party/starVLA/tests/test_qwengroot_spatial.py \
  scripts/run_qwengroot_spatial_tests_and_smoke.sh \
  scripts/install_qwengroot_spatial_on_server.sh

echo "${output}"
