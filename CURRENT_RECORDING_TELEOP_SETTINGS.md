# Current Recording And Teleoperation Settings

This is the current working setup for Franka leader-arm recording.

## Recorder Command

Use the Python package directly, not `colcon build`, for `crisp_gym` changes:

```bash
cd /home/ros/ros2_ws
python -m pip install -e src/crisp_gym
```

Typical recorder command:

```bash
python src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py \
  --repo-id hku/franka_test_XXX \
  --tasks "pick up the cube and place it on the bowl" \
  --num-episodes 2 \
  --fps 8 \
  --recording-manager-type keyboard \
  --no-push-to-hub
```

Keyboard controls:

- `r`: start / stop recording
- `s`: save current episode after stopping
- `d`: delete current episode after stopping
- `q`: quit

The keyboard manager has a stdin fallback, so pressing keys in the recorder terminal should work even if `pynput` cannot capture keyboard events in Docker.

## ROS Domain

Use the same ROS domain in all terminals:

```bash
export ROS_DOMAIN_ID=30
```

Useful check:

```bash
echo $ROS_DOMAIN_ID
ros2 topic list | grep -E "servo_angles|current_pose|joint_states|gripper"
```

## Leader Arm

Run the leader arm publisher:

```bash
ros2 run leader_arm servo_reader.py --ros-args \
  -p serial_port:=/dev/ttyUSB0
```

Expected topic:

```bash
ros2 topic echo /servo_angles --once
```

Expected `/servo_angles` format is 8 values:

- first 7 values: leader joint angle offsets
- 8th value: gripper state, `0.0` open or about `361.0` closed

## Franka Joint Teleoperation Settings

Current file:

```text
src/crisp_gym/crisp_gym/teleop/teleop_robot_servo.py
```

Current important parameters:

```python
self.command_period = 0.01
self.max_joint_velocity = 0.8
self.max_joint_acceleration = 1.2
self.angle_smoothing_alpha = 0.45
self.angle_deadband_deg = 0.02
self.joint_target_tolerance = 0.002
```

Meaning:

- `command_period = 0.01`: publish joint targets at 100 Hz.
- `max_joint_velocity = 0.8`: cap joint speed to avoid dangerous fast target changes.
- `max_joint_acceleration = 1.2`: cap joint acceleration, the main protection against Franka controller aborts.
- `angle_smoothing_alpha = 0.45`: smooth leader-arm angle readings.
- `angle_deadband_deg = 0.02`: ignore tiny leader-arm noise.
- `joint_target_tolerance = 0.002`: stop when close to target to prevent small oscillations.

The command profile also uses braking-distance logic and overshoot prevention, so the arm slows down near the target and should not bounce back and forth.

## Gripper Settings

The gripper command topic is:

```text
/gripper/gripper_position_controller/commands
```

Current direct publisher is inside `JointControlNode`:

```python
self.gripper_command_publisher = self.create_publisher(
    Float64MultiArray,
    "/gripper/gripper_position_controller/commands",
    10,
)
```

Gripper trigger:

```python
target_gripper = 0.0 if gripper >= 300 else 1.0
```

When open/close is detected, the command is repeatedly published for 1 second so the adapter and `ros2 topic echo` can see it.

Manual gripper tests:

```bash
ros2 action send_goal /franka_gripper/homing franka_msgs/action/Homing "{}"
ros2 topic pub --once /gripper/gripper_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [1.0]}"
ros2 topic pub --once /gripper/gripper_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0]}"
```

If the manual commands work but Franka Desk says end effector unconnected, the ROS gripper path is still usable for recording.

## Dataset Location

Datasets are saved inside Docker at:

```text
/home/ros/.cache/huggingface/lerobot/hku/franka_test_XXX
```

A saved episode appears as:

```text
data/chunk-000/episode_000000.parquet
```

Count saved episodes:

```bash
find /home/ros/.cache/huggingface/lerobot/hku/franka_test_XXX/data -name "*.parquet"
```

No `.mp4` output is expected when using `no_cam_franka`, because it records state/action only and no camera video.

## Quest 3 Controller Input

Quest 3 controller teleoperation is available as a separate Cartesian streamed
input path. The tracking note is:

```text
QUEST3_FRANKA_TELEOP.md
```

Run the adapter after installing `crisp_gym` editable:

```bash
crisp-quest3-stream-adapter --ros-args \
  -p quest_pose_topic:=/quest/right_controller/pose \
  -p quest_joy_topic:=/quest/right_controller/joy
```

Record with:

```bash
crisp-record-leader-follower \
  --repo-id hku/franka_quest3_cartesian_XXX \
  --tasks "pick up the cube and place it on the bowl" \
  --num-episodes 2 \
  --fps 8 \
  --recording-manager-type keyboard \
  --follower-config franka \
  --use-quest3-controller \
  --no-push-to-hub
```

Keep Quest Cartesian datasets separate from the current 8D joint-action
leader-arm datasets until an IK/joint-target route is added and tested.
