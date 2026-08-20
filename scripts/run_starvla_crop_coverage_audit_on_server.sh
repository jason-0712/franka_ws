#!/usr/bin/env bash
set -euo pipefail

starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
dataset_root="${DATASET_ROOT:-/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_replay_94eps_v1}"
tag="${CROP_AUDIT_TAG:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/replay94_crop_coverage_audit_${tag}}"
log_dir="${LOG_DIR:-/home/hanyu/starVLA_logs}"
log_path="${log_dir}/replay94_crop_coverage_audit_${tag}.log"
primary_crop="${PRIMARY_CROP:-0.20,0.48,0.85,1.00}"
wrist_crop="${WRIST_CROP:-0.00,0.18,1.00,1.00}"

for path in \
  "${python_bin}" \
  "${starvla_repo}/starvla_crop_coverage_audit.py" \
  "${dataset_root}/meta/info.json" \
  "${dataset_root}/meta/episodes.jsonl"; do
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: required file missing or empty: ${path}" >&2
    exit 1
  fi
done
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required but not installed" >&2
  exit 1
fi

mkdir -p "${output_dir}" "${log_dir}"
echo "DATASET_ROOT=${dataset_root}"
echo "PRIMARY_CROP=${primary_crop}"
echo "WRIST_CROP=${wrist_crop}"
echo "OUTPUT_DIR=${output_dir}"
df -h /data /

cd "${starvla_repo}"
set +e
"${python_bin}" starvla_crop_coverage_audit.py \
  --dataset-root "${dataset_root}" \
  --output-dir "${output_dir}" \
  --primary-crop "${primary_crop}" \
  --wrist-crop "${wrist_crop}" \
  --confirmation-window "${CONFIRMATION_WINDOW:-3}" \
  --episodes-per-sheet "${EPISODES_PER_SHEET:-10}" \
  2>&1 | tee "${log_path}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "STARVLA_CROP_COVERAGE_RUN=FAIL status=${status}" >&2
  echo "LOG=${log_path}" >&2
  exit "${status}"
fi
if ! grep -q '^STARVLA_CROP_COVERAGE_AUDIT=PASS$' "${log_path}"; then
  echo "ERROR: PASS marker missing from ${log_path}" >&2
  exit 1
fi
if grep -Eiq 'Traceback|OutOfMemoryError|CUDA out of memory' "${log_path}"; then
  echo "ERROR: suspicious failure output in ${log_path}" >&2
  exit 1
fi

sheet_count="$(find "${output_dir}/contact_sheets" -maxdepth 1 -type f -name 'sheet_*.jpg' | wc -l)"
echo "STARVLA_CROP_COVERAGE_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "GPU_REQUIRED=0"
echo "CONTACT_SHEETS=${sheet_count}"
echo "OUTPUT_DIR=${output_dir}"
echo "SUMMARY=${output_dir}/crop_coverage_summary.json"
echo "CSV=${output_dir}/crop_coverage.csv"
echo "LOG=${log_path}"
