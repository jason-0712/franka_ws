#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$ROOT_DIR/patches/nested_repos"

if [ ! -d "$PATCH_DIR" ]; then
  echo "No patch directory found: $PATCH_DIR"
  exit 0
fi

for patch in "$PATCH_DIR"/*.patch; do
  [ -f "$patch" ] || continue
  name="$(basename "$patch" .patch)"
  repo="${name//__//}"
  repo_path="$ROOT_DIR/$repo"

  if ! git -C "$repo_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "skip $repo: not a git repository"
    continue
  fi

  echo "applying $(basename "$patch") -> $repo"
  git -C "$repo_path" apply "$patch"
done
