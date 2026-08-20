# Teleoperation Adjustment Process Summary

This summarizes the debugging and tuning process used to make Franka recording and leader-arm teleoperation work.

## 1. Initial Recording Start Problem

Symptom:

- Recorder showed `Waiting to start recording...`
- Pressing `r` did nothing.

Cause:

- Keyboard mode used `pynput`, which can fail inside Docker or terminal sessions.
- In ROS recording mode, `r` is not used; `/record_transition` is used instead.

Fix:

- Added stdin fallback to `KeyboardRecordingManager`.
- Keyboard mode now polls the recorder terminal directly.
- Added logging:

```text
Keyboard command received: r (is_waiting -> recording)
```

Result:

- `--recording-manager-type keyboard` can start/stop/save with `r/s/d/q`.

## 2. Dataset Writer / Repo ID Problems

Symptoms:

- `repo_id already exists`
- dataset writer timeout
- Hugging Face `Repository Not Found`

Cause:

- LeRobot datasets are stored locally under:

```text
/home/ros/.cache/huggingface/lerobot/<repo_id>
```

- If the local folder already exists and `--resume` is not used, creation fails.
- If local folder is deleted but `--resume` is used, LeRobot tries to find the remote Hugging Face dataset.

Rule:

- New recording: use a new repo id, no `--resume`.
- Continue existing local dataset: use `--resume` and set `--num-episodes` larger than existing count.
- If deleted locally, do not use `--resume`.

## 3. Robot Availability And ROS Domain

Symptoms:

- `Timeout waiting for robot to be available`
- `/current_pose` or `/joint_states` missing
- `/servo_angles` invisible in one terminal but visible in another

Cause:

- Terminals were not always using the same `ROS_DOMAIN_ID`.

Fix:

```bash
export ROS_DOMAIN_ID=30
```

Check:

```bash
ros2 topic list | grep -E "current_pose|joint_states|servo_angles"
ros2 control list_controllers
```

## 4. Multiple Target Publishers

Symptoms:

```text
Topic 'target_joint' has 2 publishers
Topic 'target_pose' has 2 publishers
```

Cause:

- Multiple recorder/teleop processes or multiple `robot_client` nodes were publishing commands.

Fix:

- Reuse the recorder environment robot inside `JointControlNode` instead of creating a second robot client.
- Kill stale processes when needed:

```bash
pkill -f record_lerobot_format_leader_follower.py
pkill -f teleop_robot_servo.py
```

## 5. Using Servo Leader In Recorder

Goal:

- Recorder should use `JointControlNode` from:

```text
src/crisp_gym/crisp_gym/teleop/teleop_robot_servo.py
```

Fix:

- Added default servo leader mode.
- Recorder creates:

```python
leader = JointControlNode(robot=env.robot, gripper=env.gripper)
```

- Recorder uses joint control when servo leader is active:

```python
ctrl_type = "joint" if args.use_servo_leader or args.joint_control else "cartesian"
```

## 6. Smoothness Tuning

Early symptom:

- Franka movement was very unsmooth.

First direction:

- Increase command update rate.
- Reduce lag.

Working settings:

```python
self.command_period = 0.01
self.angle_smoothing_alpha = 0.45
self.angle_deadband_deg = 0.02
```

Effect:

- 100 Hz target updates make motion more continuous.
- Leader-arm filtering removes serial/servo noise.
- Deadband removes tiny resting jitter.

## 7. Angular Velocity / Controller Death Problem

Symptom:

- When leader arm moved too fast, the Franka controller terminal died.

Cause:

- Direct joint target changes can create large joint velocity and acceleration jumps.
- Franka is sensitive to rapid changes in commanded motion.

Fix:

- Added joint velocity limit:

```python
self.max_joint_velocity = 0.8
```

- Added joint acceleration limit:

```python
self.max_joint_acceleration = 1.2
```

- Commanded velocity ramps gradually:

```python
self.commanded_joint_velocity += velocity_delta
self.filtered_joint_target += self.commanded_joint_velocity * self.command_period
```

Result:

- More robust against fast leader-arm motion.
- Less likely to trigger Franka controller aborts.

## 8. Overshoot / Repeated Back-And-Forth Problem

Symptom:

- One leader-arm movement caused Franka to move back and forth several times with smaller span.

Cause:

- A simple acceleration limiter can coast past the target and then correct back.

Fix:

- Added braking-distance logic:

```python
braking_velocity = np.sqrt(2.0 * self.max_joint_acceleration * np.abs(position_error))
```

- Added overshoot prevention:

```python
overshoot = np.abs(step) > np.abs(position_error)
step[overshoot] = position_error[overshoot]
```

- Added target tolerance:

```python
self.joint_target_tolerance = 0.002
```

Result:

- Franka slows down near the target.
- It stops instead of bouncing around the final target.

## 9. Gripper Debugging

Symptoms:

- Recorder printed `Open gripper (stable)` and `Close gripper (stable)`.
- Real Franka gripper did not move.
- `ros2 topic echo /gripper/gripper_position_controller/commands` showed nothing.

Findings:

- Direct Franka hand action worked:

```bash
ros2 action send_goal /franka_gripper/homing franka_msgs/action/Homing "{}"
```

- Direct adapter topic commands worked:

```bash
ros2 topic pub --once /gripper/gripper_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [1.0]}"
ros2 topic pub --once /gripper/gripper_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0]}"
```

Cause:

- The generic `no_cam_franka` gripper config used the wrong default command topic.
- The servo teleop gripper publish path depended on the CRISP gripper wrapper.

Fix:

- Set `no_cam_franka` command topic to:

```text
gripper/gripper_position_controller/commands
```

- Added direct publisher in `JointControlNode` to:

```text
/gripper/gripper_position_controller/commands
```

- Removed dependency on `self.gripper is not None` for direct gripper command publishing.
- Repeated gripper command publication for 1 second after a change.

## 10. Build / Install Lesson

Important:

`crisp_gym` is a Python package, not a colcon package in this workspace.

Wrong:

```bash
colcon build --packages-select crisp_gym --symlink-install
```

Correct:

```bash
cd /home/ros/ros2_ws
python -m pip install -e src/crisp_gym
```

Use `colcon build` for ROS packages such as:

```text
leader_arm
crisp_controllers_robot_demos
```

## Final Summary

Smooth motion came from:

- 100 Hz command timer
- leader-angle low-pass filtering
- small deadband

Robustness against angular velocity/controller abort came from:

- joint velocity limit
- joint acceleration limit
- braking-distance limiter
- overshoot prevention
- target tolerance stop condition

Reliable recording came from:

- keyboard stdin fallback
- consistent `ROS_DOMAIN_ID=30`
- avoiding duplicate command publishers
- installing `crisp_gym` with `pip install -e`

Reliable gripper control came from:

- direct publish to `/gripper/gripper_position_controller/commands`
- checking `/franka_gripper/homing`
- verifying adapter commands separately from recorder commands
