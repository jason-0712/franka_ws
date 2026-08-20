#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_starvla_seeded_inference_on_server.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
source_file="${stage_root}/third_party/starVLA/deployment/model_server/policy_wrapper.py"
target_file="${starvla_repo}/deployment/model_server/policy_wrapper.py"

if [[ ! -f "${source_file}" ]]; then
  echo "Missing staged source: ${source_file}" >&2
  exit 1
fi
if [[ ! -f "${target_file}" ]]; then
  echo "StarVLA policy wrapper not found: ${target_file}" >&2
  exit 1
fi

backup="${target_file}.before_seeded_inference_$(date +%Y%m%d_%H%M%S)"
cp -a "${target_file}" "${backup}"
install -m 0644 "${source_file}" "${target_file}"

cd "${starvla_repo}"
python -m py_compile deployment/model_server/policy_wrapper.py
grep -q 'supports_inference_seed' deployment/model_server/policy_wrapper.py

echo "STARVLA_SEEDED_SERVER_INSTALL=PASS"
echo "BACKUP=${backup}"
echo "Restart every policy server that should accept inference_seed."
