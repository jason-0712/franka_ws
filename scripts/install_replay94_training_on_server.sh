#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: install_replay94_training_on_server.sh STAGE_DIR}
repo=${STARVLA_REPO:-/home/hanyu/starVLA}
data_root=${DATA_ROOT:-/data/hanyu/quest3_franka_real/snkdjn}
dataset_name=quest3_franka_dualcam_replay_94eps_v1
dataset_archive=${stage}/${dataset_name}.tar.gz

for required in \
  "${dataset_archive}" \
  "${stage}/scripts/register_quest3_franka_replay94_on_server.py" \
  "${stage}/scripts/start_starvla_replay94_from_74_train.sh" \
  "${stage}/scripts/start_spatial_forcing_replay94_from_74_train.sh"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing staged input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${data_root}"
dataset_path=${data_root}/${dataset_name}
if [[ -e "${dataset_path}" ]]; then
  echo "Dataset already exists; refusing to overwrite: ${dataset_path}" >&2
  echo "Audit or move the existing directory before reinstalling." >&2
  exit 1
fi

tar -xzf "${dataset_archive}" -C "${data_root}"

python "${stage}/scripts/register_quest3_franka_replay94_on_server.py" \
  --registry "${repo}/examples/realRobots/Franka/train_files/data_registry/data_config.py"

install -m 0755 \
  "${stage}/scripts/start_starvla_replay94_from_74_train.sh" \
  "${repo}/start_starvla_replay94_from_74_train.sh"
install -m 0755 \
  "${stage}/scripts/start_spatial_forcing_replay94_from_74_train.sh" \
  "${repo}/start_spatial_forcing_replay94_from_74_train.sh"

python - "${dataset_path}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.load(open(root / "meta" / "info.json"))
manifest = json.load(open(root / "meta" / "merge_manifest.json"))
assert info["total_episodes"] == 94
assert info["total_frames"] == 22449
assert info["total_videos"] == 188
assert len(list((root / "data" / "chunk-000").glob("*.parquet"))) == 94
assert len(list((root / "videos" / "chunk-000").glob("*/*.mp4"))) == 188
assert manifest["append_group_counts"] == {"front": 10, "back": 10}
print("REPLAY94_SERVER_DATASET_AUDIT=PASS")
PY

echo "REPLAY94_SERVER_INSTALL=PASS"
echo "REPLAY94_DATASET=${dataset_path}"
echo "REPLAY94_BASELINE_LAUNCHER=${repo}/start_starvla_replay94_from_74_train.sh"
echo "REPLAY94_SF_LAUNCHER=${repo}/start_spatial_forcing_replay94_from_74_train.sh"
