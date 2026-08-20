#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:-${workspace_root}/artifacts/starvla_seeded_inference_20260813.tar.gz}"
stage_root="$(mktemp -d)"
trap 'rm -rf "${stage_root}"' EXIT

install -D -m 0644 \
  "${workspace_root}/third_party/starVLA/deployment/model_server/policy_wrapper.py" \
  "${stage_root}/third_party/starVLA/deployment/model_server/policy_wrapper.py"
install -D -m 0755 \
  "${workspace_root}/scripts/starvla_open_loop_l2_eval.py" \
  "${stage_root}/scripts/starvla_open_loop_l2_eval.py"
install -D -m 0755 \
  "${workspace_root}/scripts/compare_starvla_seeded_open_loop.py" \
  "${stage_root}/scripts/compare_starvla_seeded_open_loop.py"
install -D -m 0755 \
  "${workspace_root}/scripts/install_starvla_seeded_inference_on_server.sh" \
  "${stage_root}/scripts/install_starvla_seeded_inference_on_server.sh"

mkdir -p "$(dirname "${archive}")"
tar -czf "${archive}" -C "${stage_root}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "STARVLA_SEEDED_INFERENCE_BUNDLE=${archive}"
