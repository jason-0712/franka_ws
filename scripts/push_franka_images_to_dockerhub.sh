#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "usage: $0 DOCKERHUB_USER [TAG]"
  echo "example: $0 snkdjn 20260820"
  exit 2
fi

DOCKERHUB_USER="$1"
BACKUP_TAG="${2:-20260820}"

images=(
  "crisp_controllers_demos:franka-overlay|franka-overlay-${BACKUP_TAG}"
  "franka:latest|franka-base-${BACKUP_TAG}"
  "realsense_ros2:latest|realsense-ros2-${BACKUP_TAG}"
)

for spec in "${images[@]}"; do
  source_image="${spec%%|*}"
  target_tag="${spec##*|}"
  target_image="${DOCKERHUB_USER}/franka-ws:${target_tag}"

  echo "tagging ${source_image} -> ${target_image}"
  docker image inspect "$source_image" >/dev/null
  docker tag "$source_image" "$target_image"

  echo "pushing ${target_image}"
  docker push "$target_image"
done

echo "DockerHub backup complete:"
for spec in "${images[@]}"; do
  target_tag="${spec##*|}"
  echo "  ${DOCKERHUB_USER}/franka-ws:${target_tag}"
done
