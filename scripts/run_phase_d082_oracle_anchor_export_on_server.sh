#!/usr/bin/env bash
set -euo pipefail

repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
dataset="${DATASET_ROOT:-/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_replay_94eps_v1}"
tag="${TAG:-$(date +%Y%m%d_%H%M%S)}"
output="${OUTPUT_DIR:-/data/hanyu/starVLA_runs/phase_d082_oracle_anchor_review_${tag}}"
log="${LOG:-/home/hanyu/starVLA_logs/phase_d082_oracle_anchor_review_${tag}.log}"

[[ -d "${dataset}" ]] || { echo "ERROR: dataset absent: ${dataset}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "ERROR: output exists: ${output}" >&2; exit 1; }
mkdir -p "$(dirname "${log}")"
export PYTHONPATH="${repo}:${PYTHONPATH:-}"

set -o pipefail
"${python_bin}" "${repo}/export_phase_d082_oracle_anchor_review.py" \
  --dataset-root "${dataset}" \
  --output-dir "${output}" \
  --anchors "${ANCHORS:-0:release,10:release,31:release,41:release,82:approach}" \
  --input-size 518 \
  --top-k 5 \
  2>&1 | tee "${log}"

tar -czf "${output}/phase_d082_oracle_anchor_review.tar.gz" \
  -C "${output}" \
  phase_d082_oracle_anchor_contact_sheet.jpg \
  phase_d082_oracle_anchor_manifest.json \
  oracle_anchor_overrides_template.json \
  ep0000_release_native.png \
  ep0000_release_model_input_518.png \
  ep0010_release_native.png \
  ep0010_release_model_input_518.png \
  ep0031_release_native.png \
  ep0031_release_model_input_518.png \
  ep0041_release_native.png \
  ep0041_release_model_input_518.png \
  ep0082_approach_native.png \
  ep0082_approach_model_input_518.png

echo "PHASE_D082_ORACLE_ANCHOR_RUN=PASS"
echo "ROBOT_COMMANDS_SENT=0"
echo "OUTPUT_DIR=${output}"
echo "REVIEW_ARCHIVE=${output}/phase_d082_oracle_anchor_review.tar.gz"
echo "LOG=${log}"
