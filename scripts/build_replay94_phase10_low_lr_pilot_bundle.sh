#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${workspace}/artifacts/replay94_phase10_low_lr_pilot_20260811.tar.gz}"

mkdir -p "$(dirname "${output}")"
cd "${workspace}"
tar -czf "${output}" \
  third_party/starVLA/examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh \
  third_party/starVLA/tests/test_spatial_forcing_clean.py \
  scripts/run_spatial_forcing_phase10_tests_on_server.sh \
  scripts/start_replay94_phase10_low_lr_pilot.sh \
  scripts/install_replay94_phase10_low_lr_pilot_on_server.sh

sha256sum "${output}"
ls -lh "${output}"
echo "REPLAY94_PHASE10_LOWLR_BUNDLE=PASS"
echo "BUNDLE=${output}"
