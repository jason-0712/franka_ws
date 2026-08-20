# Phase 10: clean Spatial-Forcing reproduction

This package isolates the published projected feature-alignment idea from the
earlier relational and scene-memory ablations.

## Matched experiment

Both arms use exactly the same:

- Libero30k StarVLA source checkpoint;
- 74 human-collected dual-camera episodes;
- QwenGR00T DiT action head and `include_state: false` setting;
- Qwen all-linear LoRA, rank 32, alpha 16, Gaussian no-op initialization;
- frozen VGGT teacher and alignment projection head;
- paired random-resized-crop and ColorJitter observation augmentation;
- optimizer, learning rates, random seed, batch size and step count.

The only intended difference is projected cosine-alignment weight:

- control: `projected_alignment_alpha=0.0`;
- treatment: `projected_alignment_alpha=0.5`.

The independently registered `QwenGR00TSpatialForcingClean` execution path
sets the objective to `projected_only`. Its forward graph and metric output do
not contain relational alignment, scene-relational alignment or memory queues.
The same augmented PIL images are materialized once and passed to both the
Qwen student and VGGT teacher, preserving token correspondence.

## Scope

The 20-step jobs are wiring and gradient smoke tests. They do not establish
better policy quality and their checkpoints must not be deployed on Franka.
After both jobs pass, compare their step-20 metrics before deciding whether to
run a longer matched pilot.

The runner defaults `save_interval` to `max_steps + 1`, so pilot runs save only
their final model instead of writing a roughly 9.4GB checkpoint every 1000
steps. Set `SAVE_INTERVAL` explicitly only when intermediate checkpoints are
required and disk space has been audited first.

GPU selection is forwarded consistently to both `CUDA_VISIBLE_DEVICES` and
Accelerate's `--gpu_ids`; this prevents a nonzero `GPU_ID` from being silently
overridden by an Accelerate device-zero argument.

Expected control behavior:

- `weighted_projected_alignment_loss = 0`;
- LoRA and action-head parameters update;
- alignment-head update norm remains zero.

Expected treatment behavior:

- `weighted_projected_alignment_loss = 0.5 * projected_alignment_loss`;
- LoRA, action-head and alignment-head parameters update;
- all losses and update norms remain finite.

For open-loop or real-policy evaluation, use
`scripts/export_spatial_forcing_rgb_view.py` to create a metadata-only
inference view. It disables the frozen teacher and training augmentation while
symlinking, rather than copying, the policy checkpoint.
