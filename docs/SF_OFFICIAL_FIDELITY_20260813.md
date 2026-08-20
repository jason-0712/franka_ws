# Spatial Forcing official-fidelity experiment

This experiment is the final fidelity audit before deciding whether Spatial
Forcing is useful for the StarVLA Franka project.

The implementation contract is derived from the official repository at commit
`372819b31ba9a2c8fc5989edc7d525cb187cecd5` (2026-07-07), especially
`openvla-SF/train.sh` and `openvla-SF/vla-scripts/finetune_align.py`.

## Why another matched experiment is necessary

The earlier formal StarVLA experiment used post-merger `vision_projector`
features, `alpha=0.1`, independent per-camera VGGT inference, and disabled
VGGT positional embedding.  The official OpenVLA-SF recipe instead uses:

- layer-24 LLM hidden states at visual-token positions;
- final VGGT feature stage;
- two images processed jointly by VGGT;
- 0.1-scaled VGGT 2-D positional embeddings before bilinear resampling;
- LayerNorm on the VLA tokens;
- all-linear rank-32 LoRA;
- cosine alignment coefficient 0.5;
- image augmentation during training.

Therefore the earlier negative spatial result is valid for the implemented
`vision_projector/alpha=0.1/independent` variant, but is not a definitive test
of the official recipe.

## Matched design

Both arms start from the same Replay94 checkpoint, use the same Replay94
dataset, seed, optimizer, image augmentation, all-linear LoRA, action-model
learning rate, Qwen learning rate and alignment-head learning rate.

- Control: projected alignment alpha = 0.0
- Treatment: projected alignment alpha = 0.5

The frozen VGGT forward and alignment loss are still computed in the control
so data flow and memory behavior remain matched.  Only the loss coefficient
and resulting alignment gradient differ.

## Gates

1. Static/unit tests pass.
2. Twenty-step control and treatment complete without NaN/OOM.
3. Control alignment-head update norm remains zero.
4. Treatment alignment-head and LoRA-B update norms are finite and nonzero.
5. At 500 steps, run the existing five-seed multi-position offline gate.
6. Continue to 2000 steps only if both aggregate first-step and chunk XYZ L2
   improve by at least 3%, with the same direction on at least two positions.
7. Do not run a large real-robot A/B when the spatial gate fails.
