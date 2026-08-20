#!/usr/bin/env bash
set -euo pipefail

target="${VGGT_WEIGHT:-/data/hanyu/starVLA_checkpoints/VGGT-1B/model.pt}"
expected_sha256="d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"

mkdir -p "$(dirname "${target}")"
if [[ -s "${target}" ]]; then
  actual=$(sha256sum "${target}" | awk '{print $1}')
  if [[ "${actual}" == "${expected_sha256}" ]]; then
    echo "VGGT_WEIGHT_ALREADY_VERIFIED=${target}"
    exit 0
  fi
  echo "Existing VGGT file has the wrong SHA256; refusing to overwrite it: ${target}" >&2
  echo "actual=${actual}" >&2
  exit 1
fi

available_kb=$(df -Pk "$(dirname "${target}")" | awk 'NR==2 {print $4}')
if (( available_kb < 7340032 )); then
  echo "At least 7 GiB free space is required before downloading VGGT." >&2
  df -h "$(dirname "${target}")"
  exit 1
fi

TARGET_PATH="${target}" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path(os.environ["TARGET_PATH"])
downloaded = Path(
    hf_hub_download(
        repo_id="facebook/VGGT-1B",
        filename="model.pt",
        local_dir=str(target.parent),
    )
)
if downloaded.resolve() != target.resolve():
    raise SystemExit(f"Unexpected download path: {downloaded} (wanted {target})")
print(f"VGGT_WEIGHT_DOWNLOADED={downloaded}")
PY

echo "${expected_sha256}  ${target}" | sha256sum --check --status
ls -lh "${target}"
echo "VGGT_WEIGHT_VERIFY=PASS"
