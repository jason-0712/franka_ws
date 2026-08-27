# Vendored StarVLA QwenGR00T inference runtime

This directory is a small, self-contained StarVLA runtime committed directly
into `franka_ws`. It exists so a normal clone contains the GPU policy server;
it is deliberately not a Git submodule.

## Included

- `deployment/model_server`: WebSocket server, client protocol, deterministic
  request seeding, and server-side state/action normalization;
- `starVLA/model/framework/VLM4A/QwenGR00T.py`: Qwen-VL plus DiT policy
  framework used by the Franka checkpoints;
- the Qwen VLM and action-head modules transitively required at inference;
- normalization transforms and the Franka registry required to reconstruct the
  exact training-time state/action contract from `config.yaml` and
  `dataset_statistics.json`.

The registry remains at
`examples/realRobots/Franka/train_files/data_registry/data_config.py` because
StarVLA discovers it at that location during checkpoint loading. It is runtime
metadata here, not a bundled training dataset.

## Excluded

This snapshot does not include model weights, datasets, generic training entry
points/configurations, benchmark examples, Spatial Forcing, VGGT, SAM2, or RL
experiments. It supports the QwenGR00T checkpoints used by this Franka project;
it is not intended as a replacement for the full upstream StarVLA development
repository.

## Provenance

- Upstream: <https://github.com/starVLA/starVLA>
- Upstream development base: `2e5f239bc0b1661d7d556bdba5071f3041544cc6`
- Project snapshot: `31d91bfbd5ff974ddf3c1d042972940eb7f54916`
- Additional local deployment change: request-local `inference_seed` support in
  `deployment/model_server/policy_wrapper.py`

The upstream MIT license is preserved in `LICENSE`.

## Run

From this directory in the existing StarVLA CUDA environment:

```bash
python -m pip install -e .

python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/final_model/pytorch_model.pt \
  --port 10096 \
  --use_bf16 \
  --idle_timeout -1
```

The checkpoint must remain beside its matching `config.yaml` and
`dataset_statistics.json` in the layout produced by StarVLA training.
