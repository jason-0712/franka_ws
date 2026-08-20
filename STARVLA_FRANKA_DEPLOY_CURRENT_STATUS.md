# StarVLA Franka Deployment Current Status

Last updated: 2026-07-06

This document summarizes the current state of deploying the fine-tuned StarVLA model to the real Franka FR3 robot, what has already been done, what problem we are seeing now, and the next plan.

## 1. Current Goal

The goal is to deploy a StarVLA policy on the real Franka robot for the task:

```text
pick up the cube and place it on the bowl
```

The intended control loop is:

```text
camera image + current Franka joint state
        -> StarVLA policy server
        -> 8D action chunk
        -> safe Franka client
        -> /target_joint + gripper command
        -> Franka controller
```

The action format is:

```text
[delta_joint_0, delta_joint_1, ..., delta_joint_6, gripper]
```

The current stage is **safe integration testing**, not full autonomous task success yet.

## 2. Machines And Environments

### Franka workstation

Host workspace:

```bash
/home/dase-hw101/franka_ws
```

Docker container:

```bash
franka
```

ROS workspace inside Docker:

```bash
/home/ros/ros2_ws
```

ROS domain:

```bash
export ROS_DOMAIN_ID=30
```

The StarVLA Franka client currently exists in both places:

```bash
/home/dase-hw101/franka_ws/scripts/starvla_franka_delta_joint_client.py
/home/ros/ros2_ws/scripts/starvla_franka_delta_joint_client.py
```

Important: the client must be run inside the Docker / ROS environment, not in the host `base` environment.

### StarVLA server machine

Policy server host:

```text
192.168.1.113
```

Policy server port:

```text
10093
```

Current deployed checkpoint:

```bash
/data/hanyu/starVLA_checkpoints/crisp_franka_delta_from_20k_gpu3_freeze_vlm_actionmode_abs_clean/final_model/pytorch_model.pt
```

The server metadata confirmed this checkpoint:

```text
ckpt_path: /data/hanyu/starVLA_checkpoints/crisp_franka_delta_from_20k_gpu3_freeze_vlm_actionmode_abs_clean/final_model/pytorch_model.pt
action_chunk_size: 8
available_unnorm_keys: ['franka']
action_keys: ['action.delta_joints', 'action.gripper']
state_keys: ['state.joints', 'state.gripper']
```

## 3. What Has Been Done

### 3.1 Real-world data collection

Around 20 real-world Franka teleoperation episodes were collected. The episodes are Hugging Face / LeRobot-style datasets under repos such as:

```text
snkdjn/franka_test_135
...
snkdjn/franka_test_161
```

The behavior is roughly:

```text
pick up the cube and place it on the bowl
```

The local data was copied to the training server and prepared for StarVLA fine-tuning.

### 3.2 Dataset conversion

The original real Franka data used absolute target joints. For deployment safety, we converted the dataset to delta joint actions:

```text
target_joint[t + 1] - target_joint[t]
```

The converted action format is:

```text
[delta_joint_0, ..., delta_joint_6, gripper]
```

Important fix:

The converted parquet already stores delta actions, so the StarVLA dataloader must not apply another delta operation. Therefore the StarVLA config was changed to:

```yaml
datasets:
  vla_data:
    action_mode: abs
```

This means: read the action values as stored. It does **not** mean the robot command is absolute. The stored values are still delta joints.

### 3.3 Bad output problem fixed

Before the fix, the policy output looked unsafe, with values like:

```text
-2.315, 1.981, 0.710
joint_delta_abs_max ~= 2.3
```

That was caused by the action/statistics mismatch.

After fixing the dataset/action mode/statistics, the policy output became much smaller, for example:

```text
joint_delta_min: [-0.0456 -0.0606 -0.0179 -0.0193  0.     -0.0164 -0.0534]
joint_delta_max: [ 0.0240  0.0866  0.0303  0.0314  0.      0.0616  0.0423]
joint_delta_abs_max: 0.0866
```

This is much safer for real robot testing.

### 3.4 Franka client implemented

The deployment client is:

```bash
scripts/starvla_franka_delta_joint_client.py
```

It does the following:

- subscribes to the camera image
- subscribes to `/joint_states`
- sends the image and task instruction to the StarVLA policy server
- receives an action chunk
- prints action statistics
- clamps each joint delta with `--max-delta`
- refuses very large raw policy outputs with `--max-abs-action`
- publishes to `/target_joint` only when `--execute` is passed
- runs in dry-run mode by default

The client was also modified so it no longer requires the full StarVLA repo inside the Docker container. It has a small fallback WebSocket client built in.

### 3.5 Dry-run test succeeded

This command succeeded:

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --max-delta 0.02 \
  --rate 2 \
  --max-steps 4
```

The client received policy output and computed target joints, but did not move the robot because `--execute` was not passed:

```text
Dry-run mode. No /target_joint commands will be published.
execute=False
```

### 3.6 First execute test ran

This command was run:

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --execute \
  --max-delta 0.01 \
  --rate 1 \
  --max-steps 1
```

The client entered execute mode and published one very small target:

```text
EXECUTE mode enabled. Keep hand on E-stop. max_delta=0.01, rate=1.0, max_steps=1
step=000 delta=[ 0.01 -0.01  0.01  0.01  0.    0.01 -0.01] ... execute=True
```

However, the robot did not visibly move.

## 4. Current Problem

The current issue is:

```text
The StarVLA client runs and publishes in execute mode, but Franka does not visibly move.
```

There are two likely reasons.

### Reason A: the commanded movement is too small

The first execute test used:

```bash
--max-delta 0.01
--max-steps 1
--rate 1
```

This means each joint moved by at most:

```text
0.01 rad ~= 0.57 degrees
```

For only one step, the end-effector motion may be too small to see clearly.

### Reason B: the controller is not actually accepting `/target_joint`

After the execute test, `ros2 control list_controllers` timed out:

```text
Failed getting a result from calling /controller_manager/list_controllers in 10.0
```

But the ROS graph still showed:

```text
ROS_DOMAIN_ID=30
/controller_manager
/joint_impedance_controller
/cartesian_impedance_controller
/joint_trajectory_controller
/controller_manager/list_controllers
```

This means the ROS domain is correct and the controller manager exists, but the controller manager service may be unhealthy, busy, or partially stuck.

If the controller is not active, or `/target_joint` has no subscriber, publishing commands will not move the robot.

## 5. Immediate Debug Plan

Run these checks inside the Docker / ROS terminal:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=30
```

### 5.1 Check controller manager service directly

```bash
ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers "{}"
```

If this hangs or times out, restart the Franka controller launch.

### 5.2 Check controller list

```bash
ros2 control list_controllers
```

Expected healthy state:

```text
joint_impedance_controller     ... active
joint_state_broadcaster        ... active
pose_broadcaster               ... active
twist_broadcaster              ... active
```

For this client, `joint_impedance_controller` should be active.

### 5.3 Check `/target_joint`

```bash
ros2 topic info -v /target_joint
```

Expected:

```text
Type: sensor_msgs/msg/JointState
Subscription count: 1
Publisher count: 0
```

Before running the client, `Publisher count` should be 0. During client execution, it should become 1.

If `Subscription count: 0`, the controller is not listening to `/target_joint`, so the robot will not move.

### 5.4 Check joint state is alive

```bash
ros2 topic echo /joint_states --once
```

If this hangs, the controller system is probably unhealthy.

### 5.5 Check camera is alive

```bash
ros2 topic hz /right/right_third_person_camera/color/image_raw/compressed
```

Expected around:

```text
30 Hz
```

If using compressed image in the client, add:

```bash
--compressed-image \
--image-topic /right/right_third_person_camera/color/image_raw/compressed
```

## 6. If Controller Is Unhealthy

Restart the Franka controller launch inside Docker:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=30

ros2 launch crisp_controllers_robot_demos franka.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

Then in another Docker / ROS terminal:

```bash
export ROS_DOMAIN_ID=30
ros2 control list_controllers
ros2 topic info -v /target_joint
ros2 topic echo /joint_states --once
```

Only continue if these commands return normally.

## 7. Next Safe Execution Plan

Do not run leader arm, teleop, or recorder during policy deployment.

Kill old command sources if needed:

```bash
pkill -f record_lerobot_format_leader_follower.py
pkill -f teleop_robot_servo.py
pkill -f starvla_franka_delta_joint_client.py
```

### Step 1: Dry-run

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --max-delta 0.02 \
  --rate 2 \
  --max-steps 4
```

Expected:

```text
execute=False
joint_delta_abs_max < 0.3
```

### Step 2: Execute one tiny step

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --execute \
  --max-delta 0.01 \
  --rate 1 \
  --max-steps 1
```

Expected:

```text
execute=True
one published /target_joint command
very small motion
no controller crash
```

### Step 3: Execute four tiny steps

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --execute \
  --max-delta 0.01 \
  --rate 1 \
  --max-steps 4
```

Only if this is stable should we increase `--max-steps`. Do not increase `--max-delta` yet.

## 8. Terminal Layout

Recommended terminals:

### Terminal 1: Franka controller, Docker

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=30
ros2 launch crisp_controllers_robot_demos franka.launch.py \
  arm_id:=fr3 robot_ip:=172.16.0.2 use_fake_hardware:=false use_rviz:=false
```

### Terminal 2: StarVLA policy server, server1cps

```bash
cd /home/hanyu/starVLA
conda activate starVLA
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=3 \
python deployment/model_server/server_policy.py \
  --ckpt_path /data/hanyu/starVLA_checkpoints/crisp_franka_delta_from_20k_gpu3_freeze_vlm_actionmode_abs_clean/final_model/pytorch_model.pt \
  --port 10093
```

Expected:

```text
server listening on 0.0.0.0:10093
```

### Terminal 3: StarVLA Franka client, Docker

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=30
python scripts/starvla_franka_delta_joint_client.py ...
```

### Terminal 4: Debug terminal, Docker

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=30
ros2 control list_controllers
ros2 topic info -v /target_joint
ros2 topic echo /joint_states --once
```

Leader arm is not needed for StarVLA deployment.

## 9. Safety Rules

- Keep hand near the emergency stop.
- Do not run teleoperation and StarVLA client at the same time.
- Do not run recorder during policy deployment.
- Before `--execute`, check `/target_joint` has no other publisher.
- Start with `--max-steps 1`.
- Keep `--max-delta 0.01` until controller behavior is verified.
- If controller terminal dies, stop deployment and recover Franka through Franka Desk / controller relaunch.
- Do not trust task success yet. First verify the command chain is stable.

## 10. Current Bottom Line

The StarVLA model server is reachable and gives reasonable delta-joint action chunks.

The Franka deployment client can receive observations, call the policy server, and compute safe target joints.

One tiny `--execute` test ran without client-side error, but the robot did not visibly move.

The most important next check is whether the Franka controller is healthy and whether `/target_joint` has an active subscriber. If `ros2 control list_controllers` or the direct `/controller_manager/list_controllers` service call times out, restart the Franka controller before any more execution tests.
