# Franka Teleoperation And Data Collection Guide

This guide documents the current lab procedure for Franka FR3 leader-arm teleoperation and LeRobot-format data collection.

The system has four main parts:

- Franka controller
- Leader arm serial reader
- RealSense camera
- Recorder script

Run each part in a separate terminal unless stated otherwise.

## 1. Important Paths

Host workspace:

```bash
/home/dase-hw101/franka_ws
```

Docker workspace:

```bash
/home/ros/ros2_ws
```

Local dataset path inside Docker:

```bash
/home/ros/.cache/huggingface/lerobot/<user_or_org>/<dataset_name>
```

Local dataset path on host:

```bash
/home/dase-hw101/franka_ws/dataset/<user_or_org>/<dataset_name>
```

Example:

```bash
/home/ros/.cache/huggingface/lerobot/snkdjn/franka_test_147
/home/dase-hw101/franka_ws/dataset/snkdjn/franka_test_147
```

## 2. Common Environment Setup

Use the same ROS domain in all Docker terminals:

```bash
export ROS_DOMAIN_ID=30
```

Standard Docker terminal setup:

```bash
docker exec -it franka bash

cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=30
```

For recorder terminal, also activate the LeRobot conda environment:

```bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
unset XAUTHORITY
```

## 3. Terminal 1: Camera

Run on the Ubuntu host, not inside Docker:

```bash
cd /home/dase-hw101/franka_ws/src/camera_driver
docker-compose up -d
docker-compose restart
docker logs --tail 80 realsense_ros2
```

Good sign:

```text
RealSense Node Is Up!
```

Check camera topics inside Docker:

```bash
ros2 topic list | grep right_third_person_camera
ros2 topic hz /right/right_third_person_camera/color/image_raw/compressed
```

Expected camera rate is about 30 Hz.

## 4. Terminal 2: Franka Controller

Start Docker if needed:

```bash
docker start franka
docker exec -it franka bash
```

Inside Docker:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=30

ros2 launch crisp_controllers_robot_demos franka.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

Leave this terminal running.

Before launch, Franka Desk should be ready:

- Robot unlocked
- No active error or reflex
- FCI activated
- Franka hand initialized if gripper is used

Check controllers from another Docker terminal:

```bash
ros2 control list_controllers
```

For joint teleoperation, expected important state:

```text
joint_impedance_controller active
joint_trajectory_controller inactive
joint_state_broadcaster active
pose_broadcaster active
twist_broadcaster active
```

If needed, switch to joint impedance:

```bash
ros2 control switch_controllers \
  --deactivate joint_trajectory_controller \
  --activate joint_impedance_controller
```

## 5. Terminal 3: Leader Arm

On the host, make sure the serial port is accessible:

```bash
sudo chmod 666 /dev/ttyUSB0
```

Inside Docker:

```bash
docker exec -it franka bash

cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=30

ros2 run leader_arm servo_reader.py --ros-args \
  -p serial_port:=/dev/ttyUSB0 \
  -p publish_rate:=2.0
```

Important: `publish_rate` must be a float such as `2.0`, not `2`.

Check leader arm output:

```bash
ros2 topic echo /servo_angles --once
```

At the initialized leader-arm pose, the data should be close to:

```yaml
data:
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
```

The first 7 values are joint offsets. The last value is gripper state, usually `0.0` for open or around `361.0` for closed.

If the values are not close to zero at the initial pose, restart `servo_reader.py` while the leader arm is physically held at the correct initial pose. The zero reference is captured during startup.

## 6. Terminal 4: Recorder

Inside Docker:

```bash
docker exec -it franka bash

cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
unset XAUTHORITY
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=30
```

Local-only recording command:

```bash
python src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py \
  --repo-id snkdjn/franka_test_XXX \
  --tasks "pick up the cube and place it on the bowl" \
  --num-episodes 2 \
  --fps 8 \
  --recording-manager-type keyboard \
  --follower-config franka \
  --no-push-to-hub
```

Replace `franka_test_XXX` with a new dataset name, for example `franka_test_148`.

Keyboard controls in the recorder terminal:

```text
r: start / stop recording
s: save current episode
d: delete current episode
q: quit
```

Use `--recording-manager-type keyboard` when manually pressing `r`.

## 7. Uploading To Hugging Face

Check current Hugging Face account:

```bash
hf auth whoami
```

Example current account:

```text
user: snkdjn
orgs: HKUCDS
```

To upload to your own account, use:

```bash
--repo-id snkdjn/franka_test_XXX
```

To upload to an organization, use:

```bash
--repo-id HKUCDS/franka_test_XXX
```

Only use the organization if your account has permission to create datasets there.

To upload, remove `--no-push-to-hub`:

```bash
python src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py \
  --repo-id snkdjn/franka_test_XXX \
  --tasks "pick up the cube and place it on the bowl" \
  --num-episodes 2 \
  --fps 8 \
  --recording-manager-type keyboard \
  --follower-config franka
```

If upload fails with `403 Forbidden`, create a Hugging Face token with Write permission:

```text
https://huggingface.co/settings/tokens
```

Then login again:

```bash
hf auth login
hf auth whoami
```

View uploaded dataset on the web:

```text
https://huggingface.co/datasets/snkdjn/franka_test_XXX
```

Delete a remote Hugging Face dataset with Python API:

```bash
python -c "from huggingface_hub import HfApi; HfApi().delete_repo(repo_id='snkdjn/franka_test_XXX', repo_type='dataset')"
```

Some versions of `hf` do not support `hf repo delete`, so the Python API is more reliable.

## 8. Checking Saved Data

Inside Docker:

```bash
DATA=/home/ros/.cache/huggingface/lerobot/snkdjn/franka_test_XXX
find $DATA -type f | sort
find $DATA -name "*.parquet" -exec ls -lh {} \;
find $DATA -name "*.mp4" -exec ls -lh {} \;
```

On host:

```bash
DATA=/home/dase-hw101/franka_ws/dataset/snkdjn/franka_test_XXX
find $DATA -type f | sort
```

Typical files:

```text
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.primary/episode_000000.mp4
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/crisp_meta.json
```

Meaning:

- `episode_000000.parquet`: robot state, action, timestamps, task labels
- `episode_000000.mp4`: camera video
- `info.json`: dataset feature/schema information
- `tasks.jsonl`: task descriptions
- `episodes.jsonl`: episode index metadata
- `episodes_stats.jsonl`: episode statistics
- `crisp_meta.json`: CRISP-specific metadata

## 9. Viewing Video

Find the video:

```bash
find /home/dase-hw101/franka_ws/dataset/snkdjn/franka_test_XXX -name "*.mp4"
```

Open it on the host:

```bash
xdg-open /home/dase-hw101/franka_ws/dataset/snkdjn/franka_test_XXX/videos/chunk-000/observation.images.primary/episode_000000.mp4
```

If the desktop has no default video player, install or use a player such as VLC. If the video format is not displayable, convert it with `ffmpeg`:

```bash
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output_h264.mp4
```

## 10. Common Problems And Fixes

### Pressing r Does Nothing

Use keyboard recording mode:

```bash
--recording-manager-type keyboard
```

Make sure the recorder terminal is focused. The recorder should log:

```text
Keyboard command received: r
```

### Dataset Already Exists

If the local dataset folder exists and you are creating a new dataset, use a new repo id:

```bash
--repo-id snkdjn/franka_test_149
```

Or delete the local folder:

```bash
rm -r /home/ros/.cache/huggingface/lerobot/snkdjn/franka_test_XXX
```

Use `--resume` only when you intentionally continue an existing dataset.

### Hugging Face 403 Forbidden

Example:

```text
403 Forbidden: You don't have the rights to create a dataset under the namespace
```

Causes:

- Wrong namespace, such as using `hku/...` without permission
- Token does not have Write permission
- Account cannot create dataset under that organization

Fix:

```bash
hf auth login
hf auth whoami
```

Use your own namespace:

```bash
--repo-id snkdjn/franka_test_XXX
```

### Stuck At Pushing Dataset

If the recorder prints:

```text
Pushing dataset to Hugging Face Hub...
```

and stays there, either the network is slow or the token/repo permissions are wrong. For normal local experiments, add:

```bash
--no-push-to-hub
```

### paplay Not Found

Example:

```text
Failed to play sound for episode deletion: No such file or directory: 'paplay'
```

This only means the sound notification failed. It does not affect recording.

### /servo_angles Does Not Publish

Check:

```bash
ros2 topic info -v /servo_angles
```

Expected:

```text
Publisher count: 1
Node name: servo_reader_node
```

If publisher count is 0, start the leader arm terminal again.

### Leader Arm Not Zero At Initial Pose

Check:

```bash
ros2 topic echo /servo_angles --once
```

If values are far from zero at the physical initial pose, restart `servo_reader.py` while the leader arm is held at the initial pose. Bad zero calibration can cause Franka to jump at teleop start.

### Servo Response Error

Example:

```text
Servo 5 response error
Servo 6 response error
Servo 7 response error
```

Likely causes:

- Loose wire after an earlier servo in the bus
- Power issue
- Wrong serial port
- Multiple processes accessing `/dev/ttyUSB0`
- Servo ID mismatch

Check old processes:

```bash
ps aux | grep -E "servo_reader|servo_zero|servo_changeid" | grep -v grep
```

Kill stale ones:

```bash
pkill -f servo_reader.py
pkill -f servo_zero.py
pkill -f servo_changeid.py
```

### Controller Terminal Dies Without Moving

Important line:

```text
ros2_control_node process has died, exit code -6
```

This means the C++ controller aborted. It is not necessarily caused by leader arm motion.

Search for the real reason:

```bash
LOG=$(ls -td ~/.ros/log/2026-* | head -1)
grep -R -n -B 120 -A 20 "Aborted\|exit code -6\|terminate called\|what():\|libfranka\|UDP\|FCI\|Timeout\|motion aborted" $LOG
```

If no detailed reason is saved, copy the controller terminal output immediately before:

```text
ros2_control_node-3] Aborted
```

Common causes:

- FCI not activated in Franka Desk
- Robot has active reflex/error
- Network timeout to `172.16.0.2`
- Another controller process is still running
- Multiple command publishers

Clean old processes:

```bash
pkill -f record_lerobot_format_leader_follower.py
pkill -f ros2_control_node
pkill -f franka.launch.py
pkill -f crisp_py_franka_hand_adapter
pkill -f franka_gripper
pkill -f robot_state_publisher
pkill -f joint_state_publisher
```

Check Franka network:

```bash
ping 172.16.0.2
```

### Multiple Target Publishers

Warning:

```text
Topic 'target_joint' has 2 publishers
Topic 'target_pose' has 2 publishers
```

Cause:

- Multiple recorder or teleop processes are still alive.

Fix:

```bash
pkill -f record_lerobot_format_leader_follower.py
pkill -f teleop_robot_servo.py
```

Then check:

```bash
ros2 topic info -v /target_joint
ros2 topic info -v /target_pose
```

### Camera Connected But No Video Saved

Use:

```bash
--follower-config franka
```

Check camera rate:

```bash
ros2 topic hz /right/right_third_person_camera/color/image_raw/compressed
```

If the camera topic exists and runs at about 30 Hz, the recorder should be able to save video.

### RealSense Incomplete Video Frame

Warning:

```text
Incomplete video frame detected
Frame Corrupted
```

This means RealSense dropped a frame, usually from USB bandwidth or camera connection. Occasional warnings are usually acceptable. Frequent warnings may require checking USB cable, USB port, resolution, or camera process load.

## 11. Current Teleoperation Settings

File:

```text
src/crisp_gym/crisp_gym/teleop/teleop_robot_servo.py
```

Important values:

```python
self.command_period = 0.01
self.max_joint_velocity = 0.8
self.max_joint_acceleration = 1.2
self.angle_smoothing_alpha = 0.45
self.angle_deadband_deg = 0.02
self.joint_target_tolerance = 0.002
```

Purpose:

- `command_period`: publish at 100 Hz for smoother motion
- `max_joint_velocity`: limit sudden fast joint movement
- `max_joint_acceleration`: reduce acceleration spikes that can abort Franka controller
- `angle_smoothing_alpha`: smooth leader arm serial noise
- `angle_deadband_deg`: ignore tiny hand/servo jitter
- `joint_target_tolerance`: stop near target to avoid oscillation

The code also uses braking-distance and overshoot prevention so Franka slows near target and does not bounce back and forth.

## 12. Build And Install Rules

For Python package changes in `crisp_gym`:

```bash
cd /home/ros/ros2_ws
python -m pip install -e src/crisp_gym
```

For ROS packages:

```bash
cd /home/ros/ros2_ws
colcon build --packages-select leader_arm crisp_controllers_robot_demos --symlink-install
source install/setup.bash
```

Use `colcon build` for ROS packages, not for pure Python editable changes in `crisp_gym`.

## 13. Recommended Start Order

1. Start camera.
2. Start Franka controller.
3. Start leader arm reader while leader arm is in initial pose.
4. Check `/servo_angles` is near zero.
5. Start recorder.
6. Press `r` to start recording.
7. Press `r` again to stop recording.
8. Press `s` to save.
9. Press `q` to quit.
10. Check saved `.parquet` and `.mp4`.

## 14. Safety Notes

- Keep the emergency stop reachable.
- Do not stand in the robot workspace during automatic homing.
- If Franka Desk reports reflex/error, stop recording and recover in Desk first.
- If the controller terminal dies repeatedly without motion, debug FCI/network/Desk state before running teleoperation.
- If leader arm zero is wrong, restart `servo_reader.py` before recording.

## 15. Planned Quest 3 Controller Version

The Quest 3 controller version should be added as a new teleop input path, not
as a replacement for the current leader-arm path.

Tracking note:

```text
QUEST3_FRANKA_TELEOP.md
```

The internal Lark reference for this work is:

```text
https://cjpvz8h53x23.jp.larksuite.com/wiki/TchvwT41FinWqlk5kwhjLkFXpxf?fromScene=spaceOverview
```

From the local Franka side, preserve the current safety-critical behavior:

- Use `joint_impedance_controller` for the existing 8D joint-action dataset
  route.
- Reuse the 100 Hz command timer, velocity limit, acceleration limit,
  braking-distance logic, and overshoot prevention from
  `teleop_robot_servo.py`.
- Keep gripper commands on `/gripper/gripper_position_controller/commands`.
- Require a Quest controller deadman/enable input before any robot motion.
- Add a reset/re-anchor input so the operator can redefine the Quest zero pose.

The fastest prototype can use the existing streamed-pose path in
`teleop_sensor_stream.py`, remapping Quest pose/gripper topics to
`/phone_pose` and `/phone_gripper`. That records Cartesian actions, so keep the
dataset separate from the current 8D joint-action StarVLA route. For
joint-action data collection, add a Quest-specific adapter that feeds the same
filtered joint target path used by the leader-arm node.
