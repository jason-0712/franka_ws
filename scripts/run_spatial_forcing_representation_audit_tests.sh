#!/usr/bin/env bash
set -euo pipefail

starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"

cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"

"${python_bin}" -m unittest discover \
  -s tests \
  -p 'test_spatial_forcing_representation_audit.py' \
  -v

echo "SPATIAL_FORCING_REPRESENTATION_AUDIT_TESTS=PASS"

