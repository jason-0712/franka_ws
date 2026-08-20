#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$ROOT_DIR/patches/nested_repos"
mkdir -p "$PATCH_DIR"

repos=(
  "src/crisp_controllers"
  "src/crisp_controllers_demos"
  "src/crisp_gym"
  "src/crisp_py"
  "src/franka_broadcasters"
  "src/libfranka"
  "src/piper-vr-teleop"
  "third_party/RLinf"
  "third_party/lingbot-vla"
  "third_party/starVLA"
  "third_party/starVLA_rl_libero"
)

for repo in "${repos[@]}"; do
  repo_path="$ROOT_DIR/$repo"
  patch_name="${repo//\//__}.patch"
  patch_path="$PATCH_DIR/$patch_name"

  if ! git -C "$repo_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "skip $repo: not a git repository"
    continue
  fi

  (
    cd "$repo_path"
    git diff --binary HEAD > "$patch_path"

    git ls-files --others --exclude-standard \
      ':!:**/__pycache__/**' \
      ':!:**/*.pyc' \
      ':!:.gradle/**' \
      ':!:app/.cxx/**' \
      ':!:app/build/**' \
      ':!:**/.ruff_cache/**' \
      ':!:**/*.egg-info/**' |
    while IFS= read -r file; do
      [ -f "$file" ] || continue
      case "$file" in
        */__pycache__/*|*.pyc|*.pyo) continue ;;
        .gradle/*|*/.gradle/*|.cxx/*|*/.cxx/*) continue ;;
        build/*|*/build/*|install/*|*/install/*|log/*|*/log/*) continue ;;
        *.o|*.so|*.a|*.apk|*.jar|*.class) continue ;;
        *.png|*.jpg|*.jpeg|*.mp4|*.avi|*.mov) continue ;;
        *.bag|*.db3|*.mcap|*.parquet|*.npy|*.npz) continue ;;
        *.tar|*.tar.gz|*.tgz|*.zip) continue ;;
        observation/images/*|observation/observations_*.json) continue ;;
      esac
      git diff --binary --no-index /dev/null "$file" >> "$patch_path" || true
    done
  )

  if [ -s "$patch_path" ]; then
    echo "wrote $patch_path"
  else
    rm -f "$patch_path"
    echo "clean $repo"
  fi
done

git -C "$ROOT_DIR" status --short
