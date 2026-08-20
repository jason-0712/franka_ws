# StarVLA Franka Next Implementation

This guide starts after the frozen-VLM 8D delta-joint fine-tuning run has completed.

Current model meaning:

```text
input:  one RGB image + language instruction
output: 8D action chunk = [delta_joint_0..delta_joint_6, gripper]
```

The model must not be sent directly to Franka before an offline sanity check.

## 1. Locate The Fine-Tuned Checkpoint

On `server1cps`:

```bash
cd /home/hanyu/starVLA

find /data/hanyu/starVLA_checkpoints/crisp_franka_delta_from_20k_gpu3_freeze_vlm \
  -type f | sort
```

Use the checkpoint inside either:

```text
.../final_model/pytorch_model.pt
```

or the latest step checkpoint under:

```text
.../checkpoints/steps_XXXX_pytorch_model.pt
```

Also confirm the run directory contains:

```text
config.yaml
config.full.yaml
dataset_statistics.json
```

## 2. Start The StarVLA Policy Server

Use the 95GB H100:

```bash
cd /home/hanyu/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):$PYTHONPATH

CKPT=/data/hanyu/starVLA_checkpoints/crisp_franka_delta_from_20k_gpu3_freeze_vlm/final_model/pytorch_model.pt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" \
  --port 10093 \
  --use_bf16
```

Keep this terminal open.

## 3. Run Offline Smoke Test

In another server terminal:

```bash
cd /home/hanyu/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):$PYTHONPATH

python /home/hanyu/starvla_delta_joint_policy_smoke_test.py \
  --host 127.0.0.1 \
  --port 10093 \
  --video /data/hanyu/franka_real_delta/snkdjn_delta/franka_test_161/videos/chunk-000/observation.images.primary/episode_000000.mp4 \
  --task "pick up the cube and place it on the bowl" \
  --unnorm-key franka_test_161
```

Expected output:

```text
shape: (8, 8)
joint_delta_abs_max: small value, ideally below 0.3 rad
gripper_values: values near 0 or 1
```

Do not deploy if:

```text
joint_delta_abs_max > 0.5 rad
nan / inf appears
action shape is not (8, 8)
```

## 4. Robot Deployment Logic

The client must convert the model output to a joint target:

```python
delta = action[:7]
gripper = action[7]

delta = np.clip(delta, -0.05, 0.05)  # first real-robot test: very conservative
target_joint = current_or_last_target_joint + delta
robot.set_target_joint(target_joint)
```

Start with only one action at a time. Do not execute the full action chunk at full speed first.

Recommended first safety settings:

```text
max joint delta per action: 0.03 to 0.05 rad
command rate: 2 to 5 Hz
interpolate with existing joint velocity / acceleration filter
gripper: disabled for first arm-motion test
```

## 5. Safe Real-Robot Test Order

1. Start Franka controller.
2. Confirm joint impedance controller is active.
3. Move Franka to the same standard start pose used in data collection.
4. Start camera.
5. Start policy server.
6. Run robot client in dry-run mode: print actions only.
7. Enable arm motion with gripper disabled.
8. Execute one predicted action only.
9. If stable, execute one action chunk slowly.
10. Only then enable gripper.

## 6. Why Not Direct Full Deployment

The current trained model uses:

```text
frozen qwen_vl_interface from 20k checkpoint
new / adapted 8D delta-joint action head
20 real-world episodes
```

This proves the training pipeline works, but it does not guarantee high real-world success rate.
The next accuracy improvement is partial VLM unfreeze, not full VLM co-training immediately.

