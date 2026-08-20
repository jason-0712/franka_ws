#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: $0 STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
python_bin="${STARVLA_PYTHON:-/home/hanyu/miniconda3/envs/starVLA/bin/python}"
timestamp="$(date +%Y%m%d_%H%M%S)"

declare -a mappings=(
  "third_party/starVLA/starVLA/model/modules/spatial_forcing/representation_audit.py:starVLA/model/modules/spatial_forcing/representation_audit.py"
  "third_party/starVLA/spatial_forcing_representation_audit.py:spatial_forcing_representation_audit.py"
  "third_party/starVLA/tests/test_spatial_forcing_representation_audit.py:tests/test_spatial_forcing_representation_audit.py"
  "scripts/plot_spatial_forcing_position_heatmaps.py:plot_spatial_forcing_position_heatmaps.py"
)

for mapping in "${mappings[@]}"; do
  source_rel="${mapping%%:*}"
  target_rel="${mapping#*:}"
  source_path="${stage_root}/${source_rel}"
  target_path="${starvla_repo}/${target_rel}"
  if [[ ! -s "${source_path}" ]]; then
    echo "Missing staged file: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" ]]; then
    cp -a "${target_path}" "${target_path}.before_visual_audit_${timestamp}"
  fi
  install -m 0644 "${source_path}" "${target_path}"
done

chmod 0755 \
  "${starvla_repo}/spatial_forcing_representation_audit.py" \
  "${starvla_repo}/plot_spatial_forcing_position_heatmaps.py"

cd "${starvla_repo}"
PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}" \
  "${python_bin}" tests/test_spatial_forcing_representation_audit.py

"${python_bin}" -m py_compile \
  spatial_forcing_representation_audit.py \
  plot_spatial_forcing_position_heatmaps.py \
  starVLA/model/modules/spatial_forcing/representation_audit.py

echo "SPATIAL_FORCING_VISUAL_AUDIT_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
echo "HEATMAP_TOOL=${starvla_repo}/plot_spatial_forcing_position_heatmaps.py"

