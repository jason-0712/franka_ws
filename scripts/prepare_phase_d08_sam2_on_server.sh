#!/usr/bin/env bash
set -euo pipefail

python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
sam2_commit="${SAM2_COMMIT:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
sam2_repo="${SAM2_REPO:-/home/hanyu/third_party/sam2-${sam2_commit:0:8}}"
checkpoint="${SAM2_CHECKPOINT:-/data/hanyu/starVLA_checkpoints/SAM2.1/sam2.1_hiera_large.pt}"
checkpoint_url="${SAM2_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt}"

[[ -x "${python_bin}" ]] || { echo "ERROR: Python absent: ${python_bin}" >&2; exit 1; }
command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }

mkdir -p "$(dirname "${sam2_repo}")" "$(dirname "${checkpoint}")"
if [[ ! -d "${sam2_repo}/.git" ]]; then
  [[ ! -e "${sam2_repo}" ]] || {
    echo "ERROR: non-git path already exists: ${sam2_repo}" >&2
    exit 1
  }
  git clone https://github.com/facebookresearch/sam2.git "${sam2_repo}"
fi
git -C "${sam2_repo}" fetch --depth 1 origin "${sam2_commit}"
git -C "${sam2_repo}" checkout --detach "${sam2_commit}"
actual_commit="$(git -C "${sam2_repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${sam2_commit}" ]] || {
  echo "ERROR: SAM2 commit mismatch: ${actual_commit}" >&2
  exit 1
}

"${python_bin}" - <<'PY'
import torch
import torchvision
from packaging.version import Version

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
assert Version(torch.__version__.split("+")[0]) >= Version("2.5.1")
assert Version(torchvision.__version__.split("+")[0]) >= Version("0.20.1")
PY

SAM2_BUILD_CUDA=0 "${python_bin}" -m pip install \
  --no-build-isolation \
  -e "${sam2_repo}"

if [[ ! -s "${checkpoint}" ]]; then
  partial="${checkpoint}.partial.$$"
  trap 'rm -f "${partial:-}"' EXIT
  curl --fail --location --retry 5 --retry-delay 3 \
    --output "${partial}" "${checkpoint_url}"
  bytes="$(stat -c '%s' "${partial}")"
  (( bytes >= 500000000 )) || {
    echo "ERROR: downloaded checkpoint is unexpectedly small: ${bytes} bytes" >&2
    exit 1
  }
  mv "${partial}" "${checkpoint}"
  trap - EXIT
fi
bytes="$(stat -c '%s' "${checkpoint}")"
(( bytes >= 500000000 )) || {
  echo "ERROR: checkpoint is unexpectedly small: ${bytes} bytes" >&2
  exit 1
}

PYTHONPATH="${sam2_repo}:${PYTHONPATH:-}" "${python_bin}" - <<'PY'
from sam2.build_sam import build_sam2_video_predictor
print("SAM2_IMPORT=PASS")
PY

echo "SAM2_PREPARE=PASS"
echo "SAM2_REPO=${sam2_repo}"
echo "SAM2_COMMIT=${actual_commit}"
echo "SAM2_CHECKPOINT=${checkpoint}"
echo "SAM2_CHECKPOINT_BYTES=${bytes}"
echo "SAM2_CHECKPOINT_SHA256=$(sha256sum "${checkpoint}" | awk '{print $1}')"
echo "ROBOT_COMMANDS_SENT=0"
