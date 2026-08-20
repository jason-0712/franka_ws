# Quest 3 Controller Franka Teleoperation

Status: first Cartesian streamed prototype added.

Internal reference:

```text
https://cjpvz8h53x23.jp.larksuite.com/wiki/TchvwT41FinWqlk5kwhjLkFXpxf?fromScene=spaceOverview
```

Note: this Lark page is access-controlled from this workspace session. The
implementation below uses configurable defaults so the exact topic names and
button mappings can be adjusted after opening the page in an authenticated
browser.

## Goal

Add a Meta Quest 3 controller input path for Franka FR3 teleoperation and data
collection without replacing the current leader-arm workflow.

The Quest 3 version should reuse the same Franka-side safety behavior that is
already working for leader-arm teleoperation:

- `joint_impedance_controller` for the stable joint-space Franka path.
- 100 Hz target publishing from `teleop_robot_servo.py`.
- Joint velocity, joint acceleration, braking-distance, and overshoot limits.
- Direct gripper publishing to `/gripper/gripper_position_controller/commands`.
- LeRobot recording through `record_lerobot_format_leader_follower.py`.

Do not stream raw Quest controller motion directly to Franka until the bridge
has been tested in fake hardware or RViz.

## Existing Local Baseline

Current leader-arm control lives in:

```text
src/crisp_gym/crisp_gym/teleop/teleop_robot_servo.py
```

The current recorder creates `JointControlNode`, reads `leader.last_action`, and
records 8D actions:

```text
[joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

The Quest 3 path should either preserve this 8D joint-action format, or clearly
create a new Cartesian dataset route with separate StarVLA training settings.

## Implemented Local Architecture

```text
Quest 3 controller bridge
  -> ROS 2 pose/button topics
  -> crisp-quest3-stream-adapter
  -> /phone_pose and /phone_gripper
  -> TeleopStreamedPose
  -> existing Cartesian CRISP environment step
  -> LeRobot recorder
```

Added files/entry points:

```text
src/crisp_gym/crisp_gym/scripts/quest3_stream_adapter.py
src/crisp_gym/crisp_gym/teleop/teleop_sensor_stream.py
src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py
src/crisp_gym/pyproject.toml
```

The Quest adapter expects, at minimum:

- A right-controller or dominant-hand `geometry_msgs/PoseStamped` stream.
- A `sensor_msgs/Joy` stream for buttons/axes.
- Trigger axis input for gripper open/close.
- A deadman/enable input that must be held before robot motion is allowed.
- A reset/re-anchor input that captures the current Quest controller pose as
  the new zero reference.

Preferred adapter behavior:

- Hold-to-move: when the deadman is released, the virtual pose stops moving.
- First valid controller pose becomes the Quest zero reference.
- A reset button re-anchors raw Quest motion without jumping the robot target.
- A pause button freezes motion until the deadman is released and held again.
- Translational and optional rotational scale factors are ROS parameters.
- Workspace bounds and table-height limits remain in the CRISP environment.

## Two Implementation Routes

### Route A: Cartesian Pilot

Use the existing streamed-pose path first if the Quest bridge publishes
`PoseStamped` and `Joy` topics.

```text
src/crisp_gym/crisp_gym/scripts/quest3_stream_adapter.py
src/crisp_gym/crisp_gym/teleop/teleop_sensor_stream.py
```

Terminal 1: run the Quest bridge from the Lark page.

Terminal 2: adapt Quest topics to the CRISP streamed teleop topics:

```bash
cd /home/ros/ros2_ws
python -m pip install -e src/crisp_gym

crisp-quest3-stream-adapter --ros-args \
  -p quest_pose_topic:=/quest/right_controller/pose \
  -p quest_joy_topic:=/quest/right_controller/joy \
  -p deadman_button:=4 \
  -p reset_button:=0 \
  -p pause_button:=1 \
  -p trigger_axis:=5 \
  -p trigger_threshold:=-0.5 \
  -p translation_scale:=0.15 \
  -p max_translation_step:=0.03
```

If the console script is not refreshed yet, run the module directly:

```bash
python -m crisp_gym.scripts.quest3_stream_adapter --ros-args \
  -p quest_pose_topic:=/quest/right_controller/pose \
  -p quest_joy_topic:=/quest/right_controller/joy
```

Terminal 3: record with the Quest controller path:

```bash
crisp-record-leader-follower \
  --repo-id snkdjn/franka_quest3_cartesian_XXX \
  --tasks "pick up the cube and place it on the bowl" \
  --num-episodes 2 \
  --fps 8 \
  --recording-manager-type keyboard \
  --follower-config franka \
  --use-quest3-controller \
  --skip-home \
  --no-push-to-hub
```

Current recommended Quest 3 workflow: first use the leader-arm workflow to move
Franka back to the lab standard start pose, then start the Quest 3 recorder with
`--skip-home`. In this mode the recorder treats the current Franka pose as the
episode start and does not command an additional homing motion.

This avoids the Quest 3 recorder depending on `joint_trajectory_controller` or
`/controller_manager/list_controllers` for start-pose homing. If you want a
fresh standard pose before Quest 3 collection, do it manually through the
leader-arm workflow first.

The older explicit form also works:

```bash
python src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py \
  --repo-id snkdjn/franka_quest3_cartesian_XXX \
  --tasks "pick up the cube and place it on the bowl" \
  --num-episodes 2 \
  --fps 8 \
  --recording-manager-type keyboard \
  --follower-config franka \
  --use-streamed-teleop \
  --no-use-servo-leader \
  --skip-home \
  --no-push-to-hub
```

This route is useful for quickly validating Quest tracking, gripper mapping,
and operator ergonomics. It records Cartesian relative actions, so it is not
drop-in compatible with the current 8D joint-action StarVLA dataset route.

Keep `--skip-home` for the current Quest 3 route unless you intentionally want
the recorder itself to run the normal non-leader homing path.

### Route B: Joint-Action Dataset Route

Use this route when the goal is to collect data compatible with the current
Franka joint-action datasets.

Planned code shape:

- Extract the reusable safety limiter from `JointControlNode`.
- Add a Quest-specific target provider, for example `Quest3JointControlNode`.
- Convert Quest controller delta pose into a safe Franka joint target through a
  tested IK or Cartesian-to-joint stage.
- Publish through the same filtered joint target path.
- Record `last_action` as `[filtered_joint_target, gripper]`.

This route needs more validation because the Quest controller is a 6D pose
device, while the current stable path is 7D joint target control.

## Suggested Button Mapping

Start conservative and keep one input as a hard software gate:

| Quest input | Franka action |
| --- | --- |
| Grip button held | Enable robot motion |
| Grip released | Freeze robot target, zero commanded velocity |
| Trigger pressed | Close gripper |
| Trigger released | Open gripper |
| A button | Reset/re-anchor Quest zero pose |
| B button | Pause motion until grip is released and held again |

Confirm these names against the Lark bridge because controller input labels,
axis polarity, and topic payloads vary by bridge. The adapter defaults are
parameters, not hard-coded assumptions.

## Safety Parameters To Start With

Start below the leader-arm limits and increase only after fake-hardware and
slow real-robot tests:

```text
translation_scale: 0.25
rotation_scale: 0.5
max_joint_velocity: 0.3
max_joint_acceleration: 0.6
min_z: table height + margin
```

Current Cartesian prototype defaults are more conservative:

```text
translation_scale: 0.15
rotation_scale: 0.0
max_translation_step: 0.03
max_rotation_step: 0.08
```

Keep the existing direct gripper topic:

```text
/gripper/gripper_position_controller/commands
```

## Test Checklist

1. Confirm Quest bridge topics with `ros2 topic list` and `ros2 topic echo`.
2. Start `crisp-quest3-stream-adapter` and verify `/phone_pose` and
   `/phone_gripper`.
3. Verify pose frame, units, and handedness without Franka connected.
4. Test deadman, gripper, reset, and pause behavior with printed targets only.
5. Test in fake hardware or RViz.
6. Test on the real Franka with gripper disabled and one axis at a time.
7. Enable gripper after arm motion is stable.
8. Record a short local-only episode and inspect `.parquet` action shape.
9. Only then collect a real Quest 3 dataset.

## Open Items

- Fill in exact Quest bridge package name and launch command from the Lark page.
- Fill in exact ROS topic names, button indices, and axis polarity from the
  Lark page.
- Decide whether the first usable version records Cartesian actions or the
  current 8D joint-action format. The implemented first version records
  Cartesian actions.
- If using the 8D joint-action route, add and test the IK or target-generation
  layer before recording training data.
