# Spatial Forcing fidelity correction (2026-08-13)

This change prepares a matched StarVLA/Spatial-Forcing ablation without adding
demonstrations or changing the robot deployment safety layer.

## Corrected defaults for new runs

```yaml
student_feature_source: llm_hidden
teacher_use_positional_embedding: false
teacher_view_mode: independent
image_augmentation_enabled: true
inference_center_crop_enabled: true
```

The `llm_hidden` value preserves the previous layer-24 arm. Two additional
student sources are now available:

```text
vision_encoder    final pre-merger Qwen3-VL patch tokens
vision_projector  final post-merger Qwen3-VL image embeddings
```

The implementation captures the visual features from the same Qwen forward
used by the action loss. It does not run a second student vision pass.
For Transformers 4.57 compatibility, capture occurs on Qwen3-VL's main
``visual.merger``: its input is the final pre-merger encoder representation
and its output is the post-merger image embedding. This avoids relying on the
version-dependent return container of the enclosing vision tower.

## Why each correction exists

1. The extra VGGT position embedding is disabled to avoid an unverified image-
   grid shortcut in the teacher target.
2. Primary and wrist images are run as independent one-view VGGT samples. The
   project has not established calibrated static-to-wrist multiview geometry.
3. The exact crop-area scale used for random paired train augmentation is
   applied as a deterministic center crop at RGB-only inference.
4. Qwen feature-source selection is explicit, enabling a matched comparison of
   visual/projector tokens against the historical LLM hidden-state-24 path.

## Backward compatibility

Existing checkpoints keep their behavior when their saved config explicitly
contains the old values. The default `inference_center_crop_enabled` in the
framework is `false` for old configs. The corrected YAML and runner explicitly
enable the new crop for newly trained models.

## Required smoke sequence on the server

After installing the bundle, run:

```bash
cd /home/hanyu/starVLA
conda activate starVLA
bash /home/hanyu/starVLA/run_spatial_forcing_phase10_tests_on_server.sh
```

Then run three 20-step treatment smoke jobs, changing only:

```text
STUDENT_FEATURE_SOURCE=llm_hidden
STUDENT_FEATURE_SOURCE=vision_projector
STUDENT_FEATURE_SOURCE=vision_encoder
```

Each smoke must demonstrate:

- finite action and projected-alignment losses;
- nonzero alignment-head update;
- nonzero student LoRA-B update;
- no VGGT tensors in the saved policy checkpoint;
- successful RGB-only export;
- a policy-server dry inference with no robot commands.

Only after all three pass should the 500-step matched experiment begin. The
recommended first comparison is `alpha=0` versus `alpha=0.1` with
`vision_projector`, because those tokens already have one-to-one correspondence
with Qwen's merged image-token grid. `vision_encoder` is a separate pre-merger
ablation and should not silently replace it.
