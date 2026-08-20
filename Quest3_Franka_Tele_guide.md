# Quest3 Franka Teleoperation Guide

This guide describes the working Quest 3 to Franka FR3 teleoperation and recording workflow.

The current intended use is:

```text
Quest 3 right controller
-> Franka Cartesian end-effector teleoperation
-> LeRobot/HuggingFace recording
-> 7D delta end-pose action data for StarVLA-style fine-tuning
```

## 1. System Overview

The full system is split into five terminals:

```text
Terminal 1: Franka controller
Terminal 2: Quest MR passthrough app launch
Terminal 3: Quest ROS bridge
Terminal 4: Quest-to-Franka recording adapter
Terminal 5: Recorder
```

The data path is:

```text
Quest headset/right controller
        |
        | adb logcat, tag wE9ryARX
        v
quest_reader_ros_bridge.py
        |
        | /quest/right_controller/pose
        | /quest/right_controller/joy
        v
quest3_stream_adapter.py
        |
        | /phone_pose
        | /phone_gripper
        | /record_transition
        | /gripper/gripper_position_controller/commands
        v
record_lerobot_format_leader_follower.py
        |
        | /target_pose
        v
cartesian_impedance_controller
        |
        v
Franka FR3
```

Current action space:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

Current recording setup disables Quest orientation deltas:

```text
rotation_scale = 0.0
```

So the recorded rotation components are usually near zero, while translation and gripper are active.

## 2. Safety Rules

Before operating:

```text
Keep hands clear of Franka.
Keep objects/cables clear of the arm.
Do not force the robot into the cube/table.
Stop if there is shaking near contact.
Stop if Franka Desk reports red robot errors.
Do not continue teleop after force/torque threshold errors.
```

If controller dies:

```text
1. Stop adapter and recorder.
2. Clean stale ROS nodes.
3. Recover/restart Franka safely.
4. Return to the intended standard pose.
5. Only then restart recorder.
```

Do not restart recorder while Franka is stuck near the cube, because:

```bash
--use-current-pose-as-episode-start
```

captures the current pose as the new standard pose.

## 3. Controls

Quest right controller mapping:

```text
right grip:
  hold to enable teleop and recording
  release to pause recording and wait for save/delete

trigger:
  gripper control
  held = close
  released = open

A:
  save current episode

B:
  delete current episode
```

Expected gripper stream:

```text
trigger held:
  /phone_gripper data: 0.0

trigger released:
  /phone_gripper data: 1.0
```

## 4. Clean Start Before Teleop

Before starting a new session, clean old processes.

On host:

```bash
pkill -9 -f quest3_stream_adapter
pkill -9 -f record_lerobot_format_leader_follower
pkill -9 -f quest_reader_ros_bridge
pkill -9 -f crisp_gym.scripts.record_lerobot_format_leader_follower
pkill -9 -f crisp_gym.scripts.quest3_stream_adapter
```

Kill leader arm residue if it was used:

```bash
docker exec franka bash -lc 'pkill -9 -f servo_reader.py'
```

Refresh ROS daemon:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

ros2 daemon stop
ros2 daemon start
```

Check that no old Quest/recorder nodes remain:

```bash
ros2 node list | grep -E "quest3_stream_adapter|recording_manager|robot_client|gripper_client|quest_reader_ros_bridge|servo_reader_node"
```

Expected:

```text
no output
```

## 5. Terminal 1: Franka Controller

Run inside docker:

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=30

ros2 launch /home/ros/ros2_ws/src/crisp_controllers_demos/crisp_controllers_robot_demos/launch/franka_cartesian_impedance.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

This is the correct controller for Quest Cartesian body teleoperation.

Do not use `franka.launch.py` for Quest body teleop. With the wrong launch, the gripper may move but the Franka body may not follow `/target_pose`.

If launch fails with:

```text
PackageNotFoundError: package 'franka_description' not found
```

then the workspace was not sourced. Run:

```bash
source /home/ros/ros2_ws/install/setup.bash
```

inside the docker terminal before launching.

## 6. Terminal 2: Quest MR Passthrough App

Run on host:

```bash
cd /home/ros/ros2_ws/src/piper-vr-teleop
scripts/launch_quest_mr_passthrough_app.sh
```

Expected output includes:

```text
List of devices attached
2G97C5ZHB603FS device

Launching Piper MR Passthrough Teleop (com.rail.oculus.teleop)
Events injected: 1
```

The line below is not a problem for USB ADB mode:

```text
Network stats: ... wifi, ... not connected
```

Check Quest tracking:

```bash
adb logcat | grep wE9ryARX
```

Wear the headset and move the right controller. Expected:

```text
I/wE9ryARX: right: ...
I/wE9ryARX: hmd: ...
```

If no tracking lines appear:

```text
wear headset
make sure app is active
make sure right controller is awake/tracked
check adb devices shows "device"
```

## 7. Terminal 3: Quest ROS Bridge

Run on host:

```bash
cd /home/ros/ros2_ws/src/piper-vr-teleop
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export PYTHONPATH=/home/ros/ros2_ws/src/piper-vr-teleop:$PYTHONPATH

python3 scripts/quest_reader_ros_bridge.py \
  --transport adb_logcat \
  --rate 30
```

Check topics in another terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

ros2 topic echo /quest/right_controller/pose --once
ros2 topic echo /quest/right_controller/joy --once
```

If these hang:

```text
Quest bridge may not be running.
Headset may not be worn.
Quest app may not be active.
ADB logcat may not contain wE9ryARX tracking lines.
```

Important:

```text
Quest app launch success does not automatically mean ROS topic data exists.
quest_reader_ros_bridge.py must be running.
```

## 8. Terminal 4: Recording Adapter

Use this adapter for normal data recording.

It does not publish `/target_pose` directly. The recorder/environment owns `/target_pose`.

Current safer parameters:

```text
translation_scale = 0.45
x_scale = 1.1
y_scale = 1.1
z_scale = 1.4
max_translation_step = 0.006
rotation_scale = 0.0
```

Command:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export PYTHONPATH=/home/ros/ros2_ws/src/crisp_gym:/home/ros/ros2_ws/src/crisp_py:$PYTHONPATH

python -m crisp_gym.scripts.quest3_stream_adapter --ros-args \
  -p quest_pose_topic:=/quest/right_controller/pose \
  -p quest_joy_topic:=/quest/right_controller/joy \
  -p translation_scale:=0.45 \
  -p x_scale:=1.1 \
  -p y_scale:=1.1 \
  -p z_scale:=1.4 \
  -p max_translation_step:=0.006 \
  -p rotation_scale:=0.0 \
  -p 'x_axis:="-x"' \
  -p 'y_axis:="z"' \
  -p 'z_axis:="y"' \
  -p deadman_button:=4 \
  -p deadman_latch_button:=-1 \
  -p reset_button:=-1 \
  -p pause_button:=-1 \
  -p gripper_button:=6 \
  -p trigger_axis:=5 \
  -p trigger_threshold:=0.8 \
  -p gripper_release_delay:=0.3 \
  -p record_transition_topic:=/record_transition \
  -p record_button:=4 \
  -p save_button:=0 \
  -p delete_button:=1 \
  -p gripper_requires_deadman:=false \
  -p franka_gripper_command_topic:=/gripper/gripper_position_controller/commands
```

Verify adapter output:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

ros2 topic echo /phone_pose --once
ros2 topic echo /phone_gripper
```

Trigger check:

```text
held trigger: data: 0.0
released trigger: data: 1.0
```

If ROS parameter parsing gives:

```text
Trying to set parameter 'z_axis' to 'True' of type 'BOOL', expecting type 'STRING'
```

the axis parameters were not quoted correctly. Use:

```bash
-p 'x_axis:="-x"' \
-p 'y_axis:="z"' \
-p 'z_axis:="y"' \
```

## 9. Terminal 5: Recorder

Start recorder only after:

```text
Franka controller is active
Quest app is active
Quest bridge publishes /quest/right_controller/*
Adapter publishes /phone_pose and /phone_gripper
```

Recorder command:

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export PYTHONPATH=/home/ros/ros2_ws/src/crisp_gym:/home/ros/ros2_ws/src/crisp_py:$PYTHONPATH

python -m crisp_gym.scripts.record_lerobot_format_leader_follower \
  --use-quest3-controller \
  --follower-config no_cam_franka \
  --streamed-pose-topic /phone_pose \
  --streamed-gripper-topic /phone_gripper \
  --recording-manager-type ros \
  --repo-id snkdjn/quest3_franka_data \
  --num-episodes 10 \
  --fps 15 \
  --push-to-hub \
  --use-current-pose-as-episode-start \
  --streamed-teleop-timeout 60 \
  --disable-recorder-gripper-control
```

Meaning of key arguments:

```text
--use-quest3-controller:
  use streamed Quest input

--follower-config no_cam_franka:
  Franka without camera observations

--recording-manager-type ros:
  Quest buttons control record/save/delete

--repo-id snkdjn/quest3_franka_data:
  HuggingFace dataset repo

--num-episodes 10:
  save 10 successful episodes

--push-to-hub:
  upload to HuggingFace

--use-current-pose-as-episode-start:
  capture current Franka pose as standard episode start

--disable-recorder-gripper-control:
  recorder records gripper action but does not execute gripper commands
```

## 10. Recording Workflow

Before starting recorder:

```text
Move Franka to the desired standard pose.
Do not touch/move Franka after that.
Start recorder.
```

Then:

```text
Hold right grip:
  starts teleop and starts recording

Move right controller:
  moves Franka end effector

Hold trigger:
  close gripper

Release trigger:
  open gripper

Release right grip:
  pause episode and wait for decision

Press A:
  save episode

Press B:
  delete episode
```

After save/delete:

```text
Franka returns to the captured standard pose.
Recorder waits for the next episode.
```

Seeing:

```text
Waiting to start recording...
```

after saving an episode is correct.

## 11. Setting or Changing Standard Pose

The standard pose is not a fixed config file by default. It is captured when recorder starts:

```bash
--use-current-pose-as-episode-start
```

To change it:

```text
1. Stop recorder.
2. Stop recording adapter.
3. Move Franka to the desired pose.
4. Stop direct adapter if used.
5. Start recording adapter.
6. Start recorder.
```

### 11.1 Direct adapter for standard pose adjustment

Use this only for moving to the standard pose. It directly publishes `/target_pose`.

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export PYTHONPATH=/home/ros/ros2_ws/src/crisp_gym:/home/ros/ros2_ws/src/crisp_py:$PYTHONPATH

python -m crisp_gym.scripts.quest3_stream_adapter --ros-args \
  -p quest_pose_topic:=/quest/right_controller/pose \
  -p quest_joy_topic:=/quest/right_controller/joy \
  -p translation_scale:=0.45 \
  -p x_scale:=1.2 \
  -p y_scale:=1.2 \
  -p z_scale:=1.4 \
  -p max_translation_step:=0.007 \
  -p rotation_scale:=0.20 \
  -p max_rotation_step:=0.04 \
  -p 'x_axis:="-x"' \
  -p 'y_axis:="z"' \
  -p 'z_axis:="y"' \
  -p deadman_button:=4 \
  -p deadman_latch_button:=-1 \
  -p reset_button:=-1 \
  -p pause_button:=-1 \
  -p gripper_button:=6 \
  -p trigger_axis:=5 \
  -p trigger_threshold:=0.8 \
  -p gripper_release_delay:=0.3 \
  -p gripper_requires_deadman:=false \
  -p publish_target_pose:=true \
  -p target_pose_topic:=/target_pose \
  -p current_pose_topic:=/current_pose \
  -p franka_gripper_command_topic:=/gripper/gripper_position_controller/commands
```

After reaching the desired pose:

```text
Ctrl+C this direct adapter.
Start recording adapter.
Start recorder.
```

### 11.2 Joint 7 wrist nudge

Use this only if you need to rotate wrist joint 7 directly for standard pose.

First stop Quest adapter/recorder:

```bash
pkill -9 -f quest3_stream_adapter
pkill -9 -f record_lerobot_format_leader_follower
```

Stop Cartesian controller with `Ctrl+C`, then start joint impedance controller:

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=30

ros2 launch crisp_controllers_robot_demos franka_joint_impedance.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

In another terminal, nudge joint 7:

```bash
docker exec -i franka bash <<'EOF'
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=30

python3 <<'PY'
import time
import rclpy
from sensor_msgs.msg import JointState

DELTA = 0.05  # rad. Change to -0.05 for opposite direction.

rclpy.init()
node = rclpy.create_node("nudge_fr3_joint7")
pub = node.create_publisher(JointState, "/target_joint", 10)

latest = {"msg": None}

def cb(msg):
    latest["msg"] = msg

node.create_subscription(JointState, "/joint_states", cb, 10)

start = time.time()
while rclpy.ok() and latest["msg"] is None and time.time() - start < 3.0:
    rclpy.spin_once(node, timeout_sec=0.1)

msg = latest["msg"]
if msg is None:
    raise RuntimeError("No /joint_states received")

names = [f"fr3_joint{i}" for i in range(1, 8)]
pos_map = dict(zip(msg.name, msg.position))
target = [float(pos_map[n]) for n in names]

target[6] += DELTA

out = JointState()
out.name = names
out.position = target

for _ in range(80):
    out.header.stamp = node.get_clock().now().to_msg()
    pub.publish(out)
    rclpy.spin_once(node, timeout_sec=0.01)
    time.sleep(0.02)

print(f"Moved fr3_joint7 by {DELTA} rad")
node.destroy_node()
rclpy.shutdown()
PY
EOF
```

Then stop joint controller and restart Cartesian controller before Quest teleop.

## 12. Tuning Parameters

### 12.1 Translation sensitivity

Main knobs:

```text
translation_scale:
  overall translation gain

x_scale, y_scale, z_scale:
  per-axis gain after Quest-to-Franka mapping

max_translation_step:
  maximum single publish-step translation
```

Safer recording values:

```text
translation_scale = 0.45
x_scale = 1.1
y_scale = 1.1
z_scale = 1.4
max_translation_step = 0.006
```

If z is still too weak:

```text
z_scale = 1.6
max_translation_step = 0.006
```

If motion is too slow generally:

```text
translation_scale = 0.55
max_translation_step = 0.007
```

If shaking near cube/table:

```text
translation_scale down
max_translation_step down
move standard pose slightly higher
do not keep pushing down into contact
```

### 12.2 Rotation sensitivity

Recording adapter:

```text
rotation_scale = 0.0
```

Direct standard-pose adapter:

```text
rotation_scale = 0.20
max_rotation_step = 0.04
```

If wrist rotation feels too slow during standard pose adjustment:

```text
rotation_scale = 0.25
```

Do not enable large rotation during recording unless the action space and data goal explicitly need orientation deltas.

### 12.3 Gripper timing

Adapter parameter:

```text
gripper_release_delay = 0.3
```

If gripper opens too slowly:

```text
reduce gripper_release_delay
```

If gripper auto-opens from trigger flicker:

```text
increase gripper_release_delay to 0.5 or 0.8
```

The Franka hand adapter also has open-command stabilization after today's code change.

## 13. Diagnostics

### 13.1 ADB

```bash
adb devices
```

Expected:

```text
2G97C5ZHB603FS device
```

If unauthorized:

```text
wear headset
accept USB debugging
try another USB cable/port
```

### 13.2 Quest app tracking

```bash
adb logcat | grep wE9ryARX
```

Expected:

```text
right: ...
hmd: ...
```

### 13.3 Quest ROS topics

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

ros2 topic echo /quest/right_controller/pose --once
ros2 topic echo /quest/right_controller/joy --once
```

### 13.4 Adapter output

```bash
ros2 topic echo /phone_pose --once
ros2 topic echo /phone_gripper
```

### 13.5 Target pose

```bash
ros2 topic info -v /target_pose
ros2 topic echo /target_pose --field pose.position
```

Recording mode expectation:

```text
/target_pose publisher should be robot_client from recorder/env.
```

Direct standard-pose mode expectation:

```text
/target_pose publisher should be quest3_stream_adapter.
```

Do not let both publish `/target_pose` at the same time.

### 13.6 Gripper command

```bash
ros2 topic info -v /gripper/gripper_position_controller/commands
```

The Quest adapter should be the active gripper command source. The recorder command must include:

```bash
--disable-recorder-gripper-control
```

If gripper auto-opens:

```text
kill stale recorder/adapter
refresh ROS daemon
check old gripper_client
restart cleanly
```

## 14. Troubleshooting

### 14.1 `/quest/right_controller/pose` has no output

Possible causes:

```text
Quest bridge is not running.
Headset is not worn.
Quest app is asleep/not active.
ADB logcat has no wE9ryARX data.
```

Fix:

```text
wear headset
move right controller
check adb logcat
restart quest_reader_ros_bridge.py
```

### 14.2 `/phone_pose` has no output

Possible causes:

```text
quest3_stream_adapter is not running
/quest/right_controller/pose has no data
ROS_DOMAIN_ID mismatch
```

Fix:

```text
check /quest/right_controller/pose first
restart adapter
make sure ROS_DOMAIN_ID=30 everywhere
```

### 14.3 Gripper moves but Franka body does not

Possible causes:

```text
wrong Franka launch
recorder not running
recording adapter is running without direct /target_pose
cartesian_impedance_controller not active
```

Fix:

```text
use franka_cartesian_impedance.launch.py
for recording: start recorder
for direct standard pose adjustment: use publish_target_pose:=true
```

### 14.4 Controller dies near cube

Likely:

```text
force/torque threshold
contact with cube/table
target moving into blocked space
motion too aggressive near contact
```

Do:

```text
stop teleop
clean stale nodes
recover/restart controller
return to standard pose
lower max_translation_step
avoid pushing down into contact
```

Do not:

```text
restart recorder at the stuck pose
increase z-scale aggressively
```

### 14.5 Robot shakes near cube

Likely:

```text
Cartesian impedance fighting contact or safety boundary
```

Try:

```text
move standard pose lower but not touching cube
use z_scale 1.4 but max_translation_step 0.006
approach cube slowly
release downward motion when contact begins
```

### 14.6 Movement feels strange after joint nudge

Possible causes:

```text
joint 7 changed wrist/nullspace posture
adapter was not restarted/re-anchored
old /target_pose publisher remains
```

Fix:

```bash
pkill -9 -f quest3_stream_adapter
```

Then restart adapter while holding the controller in a comfortable neutral pose.

### 14.7 `z_axis` parameter type error

Error:

```text
Trying to set parameter 'z_axis' to 'True' of type 'BOOL', expecting type 'STRING'
```

Fix:

```bash
-p 'x_axis:="-x"' \
-p 'y_axis:="z"' \
-p 'z_axis:="y"' \
```

## 15. HuggingFace Upload

Dataset repo:

```text
snkdjn/quest3_franka_data
```

Recorder uses:

```bash
--push-to-hub
```

If login is needed:

```bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
huggingface-cli login
```

Use a write token. Do not paste the token into chat or logs.

## 16. Recommended Normal Session Checklist

1. Clean old nodes.
2. Start Franka Cartesian controller.
3. Start Quest app.
4. Wear headset and verify passthrough/tracking.
5. Start Quest bridge.
6. Verify `/quest/right_controller/pose`.
7. Start recording adapter.
8. Verify `/phone_pose` and `/phone_gripper`.
9. Move Franka to intended standard pose if needed.
10. Start recorder.
11. Hold right grip to record.
12. Release right grip to pause.
13. Press A to save or B to delete.
14. Let Franka return to standard pose before next episode.

## 17. Key Principles

```text
Only one source should publish /target_pose at a time.
Only one active source should execute gripper commands.
Wear the headset for tracking.
Do not restart recorder at an unintended pose.
Use conservative step size near cube/table.
Clean stale nodes after any controller crash.
```

