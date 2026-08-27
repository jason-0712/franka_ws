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

## Runtime boundary

`pip install -e .` only registers this source tree. It does **not** create the
large, CUDA-specific PyTorch environment. The policy server requires:

- a GPU host with an NVIDIA driver and CUDA-capable PyTorch;
- Python 3.10 (the checkpoint environment used by this project);
- the inference libraries listed in `requirements-inference.txt`.

Do not run `server_policy.py` from the robot laptop's `base` environment. The
robot laptop runs the ROS client; the GPU host runs the policy server and the
two communicate over WebSocket.

## Run on the GPU host

Activate the existing StarVLA CUDA environment first. For this project's GPU
server, the environment is normally activated as follows:

```bash
source /home/hanyu/miniconda3/etc/profile.d/conda.sh
conda activate starVLA
python --version  # must report Python 3.10.x

cd /absolute/path/to/franka_ws/third_party/starVLA
python -m pip install -e . --no-deps
python deployment/model_server/check_runtime.py

python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/final_model/pytorch_model.pt \
  --port 10096 \
  --use_bf16 \
  --idle_timeout -1
```

The preflight must print `STARVLA_RUNTIME_CHECK=PASS`. For a genuinely new GPU
environment, first install a CUDA-compatible PyTorch/torchvision pair, then run
`python -m pip install -r requirements-inference.txt`. Do not let pip replace a
working server's CUDA PyTorch build merely to fix one missing module.

The checkpoint must remain beside its matching `config.yaml` and
`dataset_statistics.json` in the layout produced by StarVLA training.
