#!/usr/bin/env bash
set -euo pipefail

starvla_repo="${STARVLA_REPO:-/home/hanyu/starVLA}"
cd "${starvla_repo}"
export PYTHONPATH="${starvla_repo}:${PYTHONPATH:-}"

python tests/test_spatial_forcing_alignment.py -v
python tests/test_spatial_forcing_lora.py -v
python tests/test_spatial_forcing_clean.py -v

python - <<'PY'
from pathlib import Path

framework = Path(
    "starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingClean.py"
).read_text()
augmentation = Path(
    "starVLA/model/modules/spatial_forcing/image_augmentation.py"
).read_text()
lora = Path("starVLA/model/modules/spatial_forcing/lora_student.py").read_text()
config = Path(
    "examples/realRobots/Franka/train_files/"
    "starvla_cotrain_quest3_franka_spatial_forcing_clean.yaml"
).read_text()
runner = Path(
    "examples/realRobots/Franka/train_files/"
    "run_qwengroot_spatial_forcing_clean_smoke.sh"
).read_text()
data_registry = Path(
    "examples/realRobots/Franka/train_files/data_registry/data_config.py"
).read_text()

requirements = {
    "framework": (
        "batch_images = self.paired_image_augmentation(batch_images)",
        "captured_visual, visual_hook = self._capture_qwen_visual_features()",
        "projected_alignment_loss = self._projected_alignment_loss(",
    ),
    "augmentation": (
        "_randperm(4, generator)",
        "crop_area_scale: float = 0.9",
        "class SpatialForcingCenterCrop",
    ),
    "lora": ('targets != "all-linear"', 'init_lora_weights="gaussian"'),
    "config": (
        "objective: projected_only",
        "name: QwenGR00TSpatialForcingClean",
        "projected_alignment_alpha: 0.5",
        "relational_alignment_alpha: 0.0",
        "scene_relational_alpha: 0.0",
        "lora_target_modules: all-linear",
        "image_augmentation_enabled: true",
        "inference_center_crop_enabled: true",
        "student_feature_source: llm_hidden",
        "teacher_use_positional_embedding: false",
        "teacher_view_mode: independent",
        "include_state: false",
        "libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt",
    ),
    "runner": (
        "MATCHED_ARM",
        'default_alpha="0.0"',
        'default_alpha="0.5"',
        "--framework.spatial_forcing.objective projected_only",
        "--framework.name QwenGR00TSpatialForcingClean",
        'action_model_lr="${ACTION_MODEL_LR:-1e-4}"',
        'qwen_vl_lr="${QWEN_VL_LR:-1e-4}"',
        'alignment_head_lr="${ALIGNMENT_HEAD_LR:-1e-4}"',
        '--trainer.learning_rate.action_model "${action_model_lr}"',
        '--trainer.learning_rate.qwen_vl_interface "${qwen_vl_lr}"',
        '--trainer.learning_rate.alignment_head "${alignment_head_lr}"',
        'save_interval="${SAVE_INTERVAL:-$((max_steps + 1))}"',
        '--trainer.save_interval "${save_interval}"',
        '--framework.spatial_forcing.inference_center_crop_enabled "${inference_center_crop_enabled}"',
        '--framework.spatial_forcing.teacher_use_positional_embedding "${teacher_use_positional_embedding}"',
        '--framework.spatial_forcing.teacher_view_mode "${teacher_view_mode}"',
        '--framework.spatial_forcing.student_feature_source "${student_feature_source}"',
        '--gpu_ids "${physical_gpu}"',
    ),
    "data_registry": (
        '"quest3_franka_spatial_balanced_30eps_v1"',
        '"quest3_franka_dualcam_delta_eef"',
    ),
}
contents = {
    "framework": framework,
    "augmentation": augmentation,
    "lora": lora,
    "config": config,
    "runner": runner,
    "data_registry": data_registry,
}
missing = {
    name: [needle for needle in needles if needle not in contents[name]]
    for name, needles in requirements.items()
}
missing = {name: needles for name, needles in missing.items() if needles}
if missing:
    raise SystemExit(f"Phase-10 static integration is incomplete: {missing}")

for forbidden in (
    "relational_geometry_loss(",
    "self.scene_memory(",
    '"relational_alignment_loss"',
    '"scene_relational_loss"',
):
    if forbidden in framework:
        raise SystemExit(f"Clean framework still contains forbidden objective: {forbidden}")

print("SPATIAL_FORCING_PHASE10_STATIC_INTEGRATION=PASS")
PY

echo "SPATIAL_FORCING_PHASE10_TESTS=PASS"
