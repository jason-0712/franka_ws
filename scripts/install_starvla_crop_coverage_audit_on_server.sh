#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: $0 STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "scripts/starvla_crop_coverage_audit.py:starvla_crop_coverage_audit.py"
  "scripts/test_starvla_crop_coverage_audit.py:test_starvla_crop_coverage_audit.py"
  "scripts/run_starvla_crop_coverage_audit_on_server.sh:run_starvla_crop_coverage_audit_on_server.sh"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  target_rel="${mapping#*:}"
  source_path="${stage_root}/${source_rel}"
  target_path="${starvla_repo}/${target_rel}"
  if [[ ! -s "${source_path}" ]]; then
    echo "ERROR: missing staged file: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.before_crop_audit_${timestamp}"
  fi
  install -m 0644 "${source_path}" "${target_path}"
done

chmod 0755 \
  "${starvla_repo}/starvla_crop_coverage_audit.py" \
  "${starvla_repo}/run_starvla_crop_coverage_audit_on_server.sh"

if [[ ! -x "${python_bin}" ]]; then
  echo "ERROR: StarVLA Python not executable: ${python_bin}" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required but not installed" >&2
  exit 1
fi

cd "${starvla_repo}"
"${python_bin}" -m py_compile \
  starvla_crop_coverage_audit.py \
  test_starvla_crop_coverage_audit.py
"${python_bin}" test_starvla_crop_coverage_audit.py
"${python_bin}" - <<'PY'
import numpy
import PIL
import pyarrow
print("CROP_AUDIT_DEPENDENCIES=PASS")
print("numpy=", numpy.__version__)
print("Pillow=", PIL.__version__)
print("pyarrow=", pyarrow.__version__)
PY

echo "STARVLA_CROP_COVERAGE_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
echo "RUNNER=${starvla_repo}/run_starvla_crop_coverage_audit_on_server.sh"
