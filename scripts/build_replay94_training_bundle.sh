#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dataset_name=quest3_franka_dualcam_replay_94eps_v1
dataset=${workspace}/dataset/snkdjn/${dataset_name}
archive=${1:-${workspace}/artifacts/starvla_replay94_training_bundle_20260810.tar.gz}
stage=$(mktemp -d)
trap 'rm -rf "${stage}"' EXIT

for required in \
  "${dataset}/meta/info.json" \
  "${dataset}/meta/merge_manifest.json" \
  "${workspace}/scripts/register_quest3_franka_replay94_on_server.py" \
  "${workspace}/scripts/start_starvla_replay94_from_74_train.sh" \
  "${workspace}/scripts/start_spatial_forcing_replay94_from_74_train.sh" \
  "${workspace}/scripts/install_replay94_training_on_server.sh"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing bundle input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${stage}/scripts"
tar -czf "${stage}/${dataset_name}.tar.gz" -C "${workspace}/dataset/snkdjn" "${dataset_name}"
for script_name in \
  register_quest3_franka_replay94_on_server.py \
  start_starvla_replay94_from_74_train.sh \
  start_spatial_forcing_replay94_from_74_train.sh \
  install_replay94_training_on_server.sh; do
  install -m 0755 "${workspace}/scripts/${script_name}" "${stage}/scripts/${script_name}"
done
install -m 0644 \
  "${workspace}/dataset_manifests/quest3_franka_dualcam_replay_94eps_v1.json" \
  "${stage}/quest3_franka_dualcam_replay_94eps_v1.json"

mkdir -p "$(dirname "${archive}")"
tar -czf "${archive}" -C "${stage}" .
sha256sum "${archive}"
ls -lh "${archive}"
echo "REPLAY94_TRAINING_BUNDLE=${archive}"
