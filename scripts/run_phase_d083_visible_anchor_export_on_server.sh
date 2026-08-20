#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
dataset="${DATASET_ROOT:-/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_replay_94eps_v1}"
tag="${TAG:-$(date +%Y%m%d_%H%M%S)}"
output="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/phase_d083_visible_anchor_review_${tag}}"
log="${LOG:-/home/hanyu/starVLA_logs/phase_d083_visible_anchor_review_${tag}.log}"

[[ -d "${dataset}" ]] || { echo "ERROR: dataset absent: ${dataset}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "ERROR: output exists: ${output}" >&2; exit 1; }
mkdir -p "$(dirname "${log}")"
export PYTHONPATH="${repo}:${PYTHONPATH:-}"

set -o pipefail
"${python_bin}" "${repo}/export_phase_d083_visible_anchor_neighborhoods.py" \
  --dataset-root "${dataset}" \
  --output-dir "${output}" \
  --anchors "${ANCHORS:-0:release,10:release,31:release,41:release,82:approach}" \
  --input-size "${INPUT_SIZE:-518}" \
  --approach-before-frames "${APPROACH_BEFORE_FRAMES:-45}" \
  --release-after-frames "${RELEASE_AFTER_FRAMES:-45}" \
  --sample-stride "${SAMPLE_STRIDE:-5}" \
  --top-k "${TOP_K:-3}" \
  2>&1 | tee "${log}"

temporary_archive="${output}.review.tar.gz"
tar -czf "${temporary_archive}" -C "${output}" .
mv "${temporary_archive}" "${output}/phase_d083_visible_anchor_review.tar.gz"

echo "PHASE_D083_VISIBLE_ANCHOR_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "OUTPUT_DIR=${output}"
echo "REVIEW_ARCHIVE=${output}/phase_d083_visible_anchor_review.tar.gz"
echo "LOG=${log}"
