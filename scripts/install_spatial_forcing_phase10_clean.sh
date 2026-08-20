#!/usr/bin/env bash
set -euo pipefail

stage_root="${1:?usage: install_spatial_forcing_phase10_clean.sh STAGE_ROOT}"
starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"

if [[ ! -d "${starvla_repo}/starVLA" ]]; then
  echo "StarVLA repository not found: ${starvla_repo}" >&2
  exit 1
fi

if ! python - <<'PY'
import peft
assert peft.__version__ == "0.18.1", peft.__version__
PY
then
  echo "Phase 10 requires peft==0.18.1 in the active StarVLA environment." >&2
  echo "Run: python -m pip install --no-deps peft==0.18.1" >&2
  exit 1
fi

relative_paths=(
  starVLA/model/modules/spatial_forcing/__init__.py
  starVLA/model/modules/spatial_forcing/alignment.py
  starVLA/model/modules/spatial_forcing/image_augmentation.py
  starVLA/model/modules/spatial_forcing/lora_student.py
  starVLA/model/modules/spatial_forcing/vggt_teacher.py
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingClean.py
  starVLA/training/train_starvla.py
  starVLA/training/trainer_utils/trainer_tools.py
  examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_spatial_forcing_clean.yaml
  examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh
  examples/realRobots/Franka/train_files/data_registry/data_config.py
  tests/test_spatial_forcing_alignment.py
  tests/test_spatial_forcing_lora.py
  tests/test_spatial_forcing_clean.py
)

for relative_path in "${relative_paths[@]}"; do
  source_path="${stage_root}/third_party/starVLA/${relative_path}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing staged source: ${source_path}" >&2
    exit 1
  fi
done

backup_root="${starvla_repo}_phase10_backups/$(date +%Y%m%d_%H%M%S)"
for relative_path in "${relative_paths[@]}"; do
  target_path="${starvla_repo}/${relative_path}"
  if [[ -f "${target_path}" ]]; then
    install -D -m 0644 "${target_path}" "${backup_root}/${relative_path}"
  fi
done

for relative_path in "${relative_paths[@]}"; do
  install -D -m 0644 \
    "${stage_root}/third_party/starVLA/${relative_path}" \
    "${starvla_repo}/${relative_path}"
done

chmod 0755 \
  "${starvla_repo}/examples/realRobots/Franka/train_files/"\
"run_qwengroot_spatial_forcing_clean_smoke.sh"

install -D -m 0755 \
  "${stage_root}/scripts/run_spatial_forcing_phase10_tests_on_server.sh" \
  "${starvla_repo}/run_spatial_forcing_phase10_tests_on_server.sh"
install -D -m 0755 \
  "${stage_root}/scripts/export_spatial_forcing_rgb_view.py" \
  "${starvla_repo}/export_spatial_forcing_rgb_view.py"
install -D -m 0644 \
  "${stage_root}/PHASE10_CLEAN_REPRODUCTION.md" \
  "${starvla_repo}/PHASE10_CLEAN_REPRODUCTION.md"

cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"
python -m py_compile \
  starVLA/model/modules/spatial_forcing/image_augmentation.py \
  starVLA/model/modules/spatial_forcing/lora_student.py \
  starVLA/model/modules/spatial_forcing/vggt_teacher.py \
  starVLA/model/modules/spatial_forcing/alignment.py \
  starVLA/model/modules/spatial_forcing/__init__.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py \
  starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingClean.py \
  starVLA/training/train_starvla.py \
  starVLA/training/trainer_utils/trainer_tools.py \
  tests/test_spatial_forcing_alignment.py \
  tests/test_spatial_forcing_lora.py \
  tests/test_spatial_forcing_clean.py

echo "SPATIAL_FORCING_PHASE10_CLEAN_INSTALL=PASS"
echo "STARVLA_REPO=${starvla_repo}"
echo "BACKUP_ROOT=${backup_root}"
