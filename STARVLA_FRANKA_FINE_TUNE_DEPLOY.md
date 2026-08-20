# StarVLA Fine-Tuning And Deployment On Franka

This guide connects the current CRISP/Franka teleoperation dataset to StarVLA fine-tuning and real-robot deployment.

It is based on:

- `third_party/starVLA/docs/starVLA_guideline.md`
- `third_party/starVLA/examples/realRobots/Franka/README.md`
- `third_party/starVLA/deployment/readme-deployment.md`
- the current CRISP Franka dataset saved under `dataset/snkdjn/franka_test_*`

## 0. Current Situation

You have:

- A trained StarVLA checkpoint from about 20k steps.
- About 20 real-world Franka teleoperation episodes.
- Real-world data in LeRobot v2.1 format.
- One camera stream saved as `observation.images.primary`.
- Actions recorded as 8D joint target commands:

```text
action = [
  joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,
  gripper
]
```

Important: in the current servo-leader recorder, `action[:7]` is an absolute target joint configuration, not a joint delta.

Your known 20k StarVLA checkpoint:

```text
repo branch: starVLA_dev
repo commit: e8f8fbb1a60f521b6075a63258d898ae987f02a0
checkpoint:
  /home/hanyu/starVLA/playground/Checkpoints/libero_all_gr00t_official_30000/checkpoints/steps_20000_pytorch_model.pt
also mirrored at:
  /data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000/checkpoints/steps_20000_pytorch_model.pt
config:
  /home/hanyu/starVLA/playground/Checkpoints/libero_all_gr00t_official_30000/config.yaml
  /home/hanyu/starVLA/playground/Checkpoints/libero_all_gr00t_official_30000/config.full.yaml
dataset stats:
  /home/hanyu/starVLA/playground/Checkpoints/libero_all_gr00t_official_30000/dataset_statistics.json
framework: QwenGR00T
action_dim: 7
state_dim: 7
action_horizon: 8
data_mix: libero_all
action_type: delta_qpos
training steps: 30000 max, using steps_20000 checkpoint
```

That means deployment should not use `env.step(action)` in `ManipulatorJointEnv`, because that method treats action as a delta and adds it to `robot.target_joint`.

For the current data, the safest deployment path is:

```text
StarVLA predicts absolute target joints -> client rate-limits/smooths -> robot.set_target_joint()
```

## 1. Keep Environments Separate

Do not install StarVLA into the existing `lerobot` recording environment.

Recommended separation:

- CRISP / Franka / recorder environment: Docker container `franka`, conda env `lerobot`.
- StarVLA training/deployment environment: a separate conda env, for example `starVLA`.

StarVLA repo path:

```bash
/home/dase-hw101/franka_ws/third_party/starVLA
```

## 2. Decide Action Representation

There are two possible routes.

### Current Decision: Convert The Franka Data To 8D Delta Joint Actions

After testing the absolute-joint route, training produced `nan` loss even when
the raw samples were finite. The current recommended route is therefore to keep
the Franka controller/deployment in joint space, but train StarVLA on delta
joint actions:

```text
state:  [joint_0, ..., joint_6, gripper]
action: [delta_joint_0, ..., delta_joint_6, gripper]
```

The original absolute data is kept under:

```text
dataset/snkdjn/franka_test_*
```

The converted delta data is written under:

```text
dataset/snkdjn_delta/franka_test_*
```

Conversion formula:

```text
delta_joint[t] = target_joint[t + 1] - target_joint[t]
last-frame delta_joint = 0
gripper is copied unchanged
```

Deployment should then apply the model output as:

```python
next_target_joint = current_or_last_target_joint + predicted_delta[:7]
robot.set_target_joint(next_target_joint)
```

Keep the existing safety filters in the deployment client: clamp per-step joint
delta, rate-limit large jumps, and smooth the command stream before publishing
to Franka.

Local conversion command:

```bash
docker exec franka bash -lc '
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
python /home/ros/ros2_ws/src/crisp_gym/crisp_gym/scripts/convert_lerobot_abs_joint_to_delta_joint.py \
  --input-root /home/ros/.cache/huggingface/lerobot/snkdjn \
  --output-root /home/ros/.cache/huggingface/lerobot/snkdjn_delta \
  --media-mode copy
'
```

Server-side conversion command:

```bash
cd /home/hanyu/starVLA
python /home/hanyu/convert_lerobot_abs_joint_to_delta_joint.py \
  --input-root /data/hanyu/franka_real/snkdjn \
  --output-root /data/hanyu/franka_real_delta/snkdjn_delta \
  --media-mode symlink
```

Server-side training command:

```bash
cd /home/hanyu/starVLA
conda activate starVLA

DATA_ROOT=/data/hanyu/franka_real_delta/snkdjn_delta \
PRETRAINED_CHECKPOINT=/home/hanyu/starVLA/playground/Checkpoints/libero_all_gr00t_official_30000/checkpoints/steps_20000_pytorch_model.pt \
RUN_ID=crisp_franka_delta_from_20k \
MAX_TRAIN_STEPS=1000 \
PER_DEVICE_BATCH_SIZE=1 \
GPU_IDS=1 \
bash examples/Franka/train_files/run_crisp_franka_train_delta_joints.sh
```

### Route A: Keep The 20k 7D LIBERO Action Head

This route preserves your existing 20k checkpoint most directly.

The 20k checkpoint was trained with LIBERO-style 7D actions:

```text
[x, y, z, roll, pitch, yaw, gripper]
```

To fine-tune this exact model on real Franka data, the real dataset must also provide 7D Cartesian/end-effector actions.

You would need to convert each real episode from current joint target actions to:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

Pros:

- Best compatibility with the trained 20k QwenGR00T checkpoint.
- You can continue fine-tuning the existing 7D action head.

Cons:

- Your current teleop data was not recorded in this action space.
- You must compute Cartesian delta actions from the robot pose sequence.
- Real Franka Cartesian deployment may be less stable than the current joint-space setup.

### Route B: Use The Current 8D Joint Target Data, Recommended For Current Franka Setup

Use the current recorded data directly:

```text
state:  observation.state.joints + observation.state.gripper
action: absolute joint target + gripper
image:  observation.images.primary
```

Set StarVLA:

```yaml
framework:
  action_model:
    action_dim: 8
    state_dim: 8
```

Deploy with:

```python
robot.set_target_joint(predicted_action[:7])
gripper.set_target(predicted_action[7])
```

Pros:

- Matches current CRISP joint teleoperation path.
- Reuses the smoother joint impedance controller that is already working.
- Avoids writing a Cartesian action converter.

Cons:

- If your 20k-step StarVLA checkpoint was trained with 7D Cartesian actions, the action head shape is different. You may not be able to load the full checkpoint directly. You can still reuse the VLM/backbone and train or reload the action head carefully.

For the current lab setup, start with Route B unless you explicitly want to convert the teleop data to 7D Cartesian actions.

## 3. Inspect The Current Dataset

Available local episodes with both parquet and video include:

```text
franka_test_132
franka_test_135
franka_test_136
franka_test_137
franka_test_139
franka_test_140
franka_test_141
franka_test_144
franka_test_146
franka_test_148
franka_test_149
franka_test_150
franka_test_151
franka_test_152
franka_test_153
franka_test_154
franka_test_155
franka_test_156
franka_test_157
franka_test_158
franka_test_161
```

From the screenshot, the selected 20 successful Hugging Face private datasets appear to be:

```text
snkdjn/franka_test_135
snkdjn/franka_test_136
snkdjn/franka_test_137
snkdjn/franka_test_139
snkdjn/franka_test_140
snkdjn/franka_test_141
snkdjn/franka_test_144
snkdjn/franka_test_146
snkdjn/franka_test_148
snkdjn/franka_test_149
snkdjn/franka_test_150
snkdjn/franka_test_151
snkdjn/franka_test_152
snkdjn/franka_test_153
snkdjn/franka_test_154
snkdjn/franka_test_155
snkdjn/franka_test_156
snkdjn/franka_test_157
snkdjn/franka_test_158
snkdjn/franka_test_161
```

Check one dataset:

```bash
DATA=/home/dase-hw101/franka_ws/dataset/snkdjn/franka_test_161
find $DATA -type f | sort
sed -n '1,220p' $DATA/meta/info.json
sed -n '1,80p' $DATA/meta/tasks.jsonl
```

The current schema is:

```text
observation.images.primary: 256x256 RGB video
observation.state.cartesian: 6D
observation.state.gripper: 1D
observation.state.joints: 7D
observation.state.target: 7D
observation.state: 21D
action: 8D
task_index: int
```

## 4. Add StarVLA Modality Metadata

StarVLA requires `meta/modality.json` in each dataset directory.

For the current CRISP Franka joint data, use this modality:

```json
{
  "state": {
    "joints": {
      "start": 0,
      "end": 7,
      "original_key": "observation.state.joints"
    },
    "target_joints": {
      "start": 0,
      "end": 7,
      "original_key": "observation.state.target"
    },
    "gripper": {
      "start": 0,
      "end": 1,
      "original_key": "observation.state.gripper"
    }
  },
  "action": {
    "target_joints": {
      "start": 0,
      "end": 7,
      "original_key": "action",
      "absolute": true
    },
    "gripper": {
      "start": 7,
      "end": 8,
      "original_key": "action",
      "absolute": true
    }
  },
  "video": {
    "primary_image": {
      "original_key": "observation.images.primary"
    }
  },
  "annotation": {
    "human.action.task_description": {
      "original_key": "task_index"
    }
  }
}
```

Create it for one dataset first:

```bash
cd /home/dase-hw101/franka_ws
mkdir -p dataset/snkdjn/franka_test_161/meta
nano dataset/snkdjn/franka_test_161/meta/modality.json
```

After one dataset works, copy the same file to all selected episode folders.

## 5. Register The Dataset In StarVLA

StarVLA automatically discovers files under:

```text
third_party/starVLA/examples/*/train_files/data_registry/data_config.py
```

Use the existing Franka registry:

```text
third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py
```

Add a new data config for CRISP absolute joint actions:

```python
class CrispFrankaAbsJointsDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA

    video_keys = ["video.primary_image"]
    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.target_joints", "action.gripper"]

    action_key_dims = {
        "action.target_joints": 7,
        "action.gripper": 1,
    }
    state_key_dims = {
        "state.joints": 7,
        "state.gripper": 1,
    }

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "min_max",
                    "state.gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.target_joints": "min_max",
                    "action.gripper": "binary",
                },
            ),
        ])
```

Then add it to `ROBOT_TYPE_CONFIG_MAP`:

```python
ROBOT_TYPE_CONFIG_MAP = {
    ...
    "crisp_franka_abs_joints": CrispFrankaAbsJointsDataConfig(),
}
```

Add a mixture:

```python
DATASET_NAMED_MIXTURES = {
    ...
    "crisp_franka_pick_cube_place_bowl_20eps": [
        ("franka_test_135", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_136", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_137", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_139", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_140", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_141", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_144", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_146", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_148", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_149", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_150", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_151", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_152", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_153", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_154", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_155", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_156", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_157", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_158", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_161", 1.0, "crisp_franka_abs_joints"),
    ],
}
```

If you want a validation episode, remove `franka_test_161` from the training mixture and keep it for evaluation.

## 6. Create A Fine-Tuning Config

Create:

```text
third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_crisp_franka_abs_joints.yaml
```

Use this as a starting point:

```yaml
run_id: crisp_franka_pick_cube_place_bowl_abs_joints
run_root_dir: results/Checkpoints
seed: 42
wandb_entity: your_wandb_entity
wandb_project: starVLA_crisp_franka
is_debug: false
version_id: "0.21"

framework:
  name: QwenOFT
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
    attn_implementation: flash_attention_2
  action_model:
    action_dim: 8
    state_dim: 8
    action_horizon: 8

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    include_state: false
    data_root_dir: /home/dase-hw101/franka_ws/dataset/snkdjn
    data_mix: crisp_franka_pick_cube_place_bowl_20eps
    action_type: absolute_joint
    sequential_step_sampling: false
    per_device_batch_size: 2
    load_all_data_for_training: true
    obs_image_size: [224, 224]
    delete_pause_frame: false
    video_backend: torchvision_av

trainer:
  max_train_steps: 1000
  num_warmup_steps: 50
  save_interval: 200
  eval_interval: 200
  learning_rate:
    base: 1.0e-05
    qwen_vl_interface: 1.0e-06
    action_model: 5.0e-05
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: "qwen_vl"
  loss_scale:
    vla: 1.0
    vlm: 0.0
  max_grad_norm: 1.0
  weight_decay: 0.0
  logging_frequency: 10
  gradient_clipping: 1.0
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08
```

If you want to continue from your 20k checkpoint, add:

```yaml
trainer:
  pretrained_checkpoint: /path/to/your/steps_20000_pytorch_model.pt
```

Important: if the 20k checkpoint was trained with `action_dim: 7`, it may not load cleanly into an `action_dim: 8` model. In that case use the checkpoint only for compatible modules, or train the 8D action head from the base VLM. Do not force-load mismatched action-head weights.

## 7. Validate DataLoader Before Training

From the StarVLA environment:

```bash
cd /home/dase-hw101/franka_ws/third_party/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/realRobots/Franka/train_files/starvla_cotrain_crisp_franka_abs_joints.yaml
```

Expected result:

- It loads the 20 selected datasets.
- It can iterate a few batches.
- It writes:

```text
results/debug/dataset_statistics.json
```

If this step fails, do not start training yet. Fix `modality.json`, registry, or paths first.

## 8. Fine-Tune

For a first small run:

```bash
cd /home/dase-hw101/franka_ws/third_party/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/Franka/train_files/starvla_cotrain_crisp_franka_abs_joints.yaml \
  --framework.name QwenOFT \
  --trainer.max_train_steps 1000 \
  --trainer.save_interval 200 \
  --run_root_dir results/Checkpoints \
  --run_id crisp_franka_pick_cube_place_bowl_abs_joints_test \
  --wandb_project starVLA_crisp_franka
```

For only 20 episodes:

- Use a small learning rate.
- Freeze most VLM layers first.
- Watch for overfitting.
- Do not assume high real-world success from 20 demos; use it first as adaptation/smoke test.

## 9. Start StarVLA Policy Server

After fine-tuning, choose a checkpoint:

```text
results/Checkpoints/crisp_franka_pick_cube_place_bowl_abs_joints_test/checkpoints/steps_1000_pytorch_model.pt
```

Start server:

```bash
cd /home/dase-hw101/franka_ws/third_party/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

python deployment/model_server/server_policy.py \
  --ckpt_path results/Checkpoints/crisp_franka_pick_cube_place_bowl_abs_joints_test/checkpoints/steps_1000_pytorch_model.pt \
  --port 5694 \
  --use_bf16
```

The server returns `response["data"]["actions"]` in the new server-side unnormalization API.

## 10. Build A Franka Deployment Client

The client should run in the Franka/ROS environment or in another environment that can access ROS and the camera topics.

High-level client loop:

```python
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from crisp_gym.envs.manipulator_env import make_env
import numpy as np
import time

client = WebsocketClientPolicy(host="127.0.0.1", port=5694)
meta = client.get_server_metadata()

env = make_env(env_type="franka", control_type="joint", namespace="")
env.wait_until_ready()
env.home()
env.reset()

instruction = "pick up the cube and place it on the bowl"

while True:
    obs = env.get_obs()
    image = obs["observation.images.primary"]  # uint8, H,W,3

    request = {
        "examples": [{
            "image": [image],
            "lang": instruction,
        }],
        "unnorm_key": "crisp_franka_abs_joints",
    }

    result = client.predict_action(request)
    actions = np.asarray(result["data"]["actions"][0])  # [T, 8]

    for action in actions:
        target_joint = action[:7]
        gripper = action[7]

        # Safety: clip to a conservative joint box before sending.
        target_joint = np.clip(target_joint, -np.pi, np.pi)

        # IMPORTANT: absolute target joint, not env.step().
        env.robot.set_target_joint(target_joint)
        env._set_gripper_action(float(gripper))
        time.sleep(1.0 / 8.0)
```

Before using the real robot, replace direct `set_target_joint()` with a safety wrapper:

- joint limit clipping
- max joint velocity clipping
- max joint acceleration clipping
- emergency stop condition
- workspace monitoring
- gripper command debounce

You can reuse the smoothing logic from:

```text
src/crisp_gym/crisp_gym/teleop/teleop_robot_servo.py
```

Relevant parameters:

```python
command_period = 0.01
max_joint_velocity = 0.8
max_joint_acceleration = 1.2
joint_target_tolerance = 0.002
```

## 11. Deployment Safety Checklist

Before robot execution:

```bash
ros2 topic list | grep -E "current_pose|joint_states|servo_angles"
ros2 control list_controllers
ros2 topic info -v /target_joint
```

Make sure:

- `joint_impedance_controller` is active.
- No recorder/teleop process is still publishing commands.
- `/target_joint` does not have multiple publishers.
- Franka Desk has no active error.
- FCI is active.
- Emergency stop is reachable.

Kill stale processes:

```bash
pkill -f record_lerobot_format_leader_follower.py
pkill -f teleop_robot_servo.py
pkill -f joint_control_node
```

Start with slow and short tests:

1. Server-only debug request.
2. Client predicts but does not execute.
3. Execute only first action with a very small safety clamp.
4. Execute one short chunk.
5. Full rollout.

## 12. What To Implement First

Recommended implementation order:

1. Add `modality.json` to one dataset.
2. Add `CrispFrankaAbsJointsDataConfig`.
3. Add one-dataset mixture first, for example `franka_test_161`.
4. Run dataloader smoke test.
5. Add the full 20-episode mixture.
6. Fine-tune for a small number of steps.
7. Start policy server and send one fake/debug request.
8. Write a dry-run client that prints predicted actions.
9. Add safe joint-target executor.
10. Test on real Franka with one action at a time.

Do not go directly from training to full autonomous rollout.

## 13. Files Added For The 8D Joint Route

The local workspace now contains the 8D joint-action setup files:

```text
scripts/starvla_crisp_franka_modality.json
third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_crisp_franka_abs_joints.yaml
third_party/starVLA/examples/realRobots/Franka/train_files/run_crisp_franka_train_abs_joints.sh
```

The StarVLA Franka registry was extended here:

```text
third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py
```

Added registry entries:

```text
CrispFrankaAbsJointsDataConfig
crisp_franka_abs_joints
crisp_franka_pick_cube_place_bowl_20eps
crisp_franka_pick_cube_place_bowl_debug
```

The `modality.json` template was copied to these selected datasets:

```text
franka_test_135
franka_test_136
franka_test_137
franka_test_139
franka_test_140
franka_test_141
franka_test_144
franka_test_146
franka_test_148
franka_test_149
franka_test_150
franka_test_151
franka_test_152
franka_test_153
franka_test_154
franka_test_155
franka_test_156
franka_test_157
franka_test_158
franka_test_161
```

Before training on the server, copy or reproduce these same files there, and make sure the dataset root path matches the server path.
