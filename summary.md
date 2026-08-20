# 2026-07-08 Quest 3 -> Franka Teleoperation Summary

## 1. 今日目标

今天主要目标是继续把 Quest 3 controller 接到 Franka FR3 上，完成一个可以用于数据采集的 teleoperation pipeline。

目标链路是：

```text
Quest 3 right controller
-> Quest OpenXR native tracking app
-> adb logcat
-> quest_reader_ros_bridge
-> /quest/right_controller/pose and /quest/right_controller/joy
-> quest3_stream_adapter
-> /phone_pose and /phone_gripper
-> record_lerobot_format_leader_follower
-> /target_pose and gripper command
-> Franka cartesian_impedance_controller
```

长期数据目标仍然是和 StarVLA/LIBERO 对齐的 7D delta end pose：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

当前 Quest 3 teleop 本质上走的是 end-effector pose control，不是 delta joint control。

## 2. Quest 3 侧进展

### ADB 和 Developer Mode

今天 Quest 3 的 ADB 已经正常连上。最终正常状态是：

```bash
adb devices
```

应该看到：

```text
2G97C5ZHB603FS    device
```

之前出现过这些问题：

```text
adb: command not found
unauthorized
no permissions
```

处理结果：

- 安装/启用了 adb。
- 打开 Quest Developer Mode。
- 换 USB 线后，`unauthorized` 变成了 `device`。
- 如果再次出现 `unauthorized`，优先换线或重新插拔，然后戴上 headset 看是否弹出 USB debugging 授权。

### Quest passthrough / 黑屏问题

今天一开始戴 headset 时经常看到全黑，后来确认不是左 controller 电量导致。

含义：

```text
黑屏/透明屏 != tracking 一定失败
```

真正判断 Quest app 是否在工作，要看 logcat：

```bash
adb logcat -v time -s wE9ryARX:I '*:S'
```

后来 passthrough 已经恢复，可以在 headset 里看到 Franka 和环境。

如果再次黑屏，优先检查：

```bash
adb shell pidof com.rail.oculus.teleop
adb shell dumpsys power | grep -Ei 'mWakefulness|Display Power'
adb logcat -d -t 300 | grep -Ei "Passthrough|XR_FB|xrCreatePassthrough|xrPassthrough|wE9ryARX"
```

如果 `pidof` 有进程，并且 `wE9ryARX` 有 tracking line，说明 app 在跑；黑屏多半是 passthrough rendering/UI 层的问题。

## 3. Quest Tracking App

启动 Quest app：

```bash
cd /home/ros/ros2_ws/src/piper-vr-teleop
scripts/launch_quest_mr_passthrough_app.sh
```

正常输出：

```text
[Quest MR] Launching Piper MR Passthrough Teleop (com.rail.oculus.teleop)...
[Quest MR] Watch live tracking payloads with:
  adb logcat | grep wE9ryARX
```

package:

```text
com.rail.oculus.teleop
```

log tag:

```text
wE9ryARX
```

今天已经看到有效 tracking payload，例如：

```text
I/wE9ryARX: right: ... |hmd: ...
```

说明 Quest right controller tracking 是通的。

## 4. Quest ROS Bridge

启动 bridge：

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

正常日志：

```text
Publishing PoseStamped on /quest/right_controller/pose
Publishing Joy on /quest/right_controller/joy
```

检查 pose：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

ros2 topic hz /quest/right_controller/pose
```

今天已经看到：

```text
average rate: 30.000
```

这说明 Quest pose topic 已经稳定 30 Hz。

检查 joy：

```bash
ros2 topic echo /quest/right_controller/joy --once
```

今天看到的 button array 是 7 个：

```text
buttons:
- 0
- 0
- 0
- 0
- 0
- 0
- 0
```

我们扩展后的 mapping 预期是：

```text
buttons[0] = A
buttons[1] = B
buttons[2] = X
buttons[3] = Y
buttons[4] = rightGrip
buttons[5] = right joystick click
buttons[6] = rightTrig
```

但是今天实际测试发现 right joystick click 没有稳定变成 1，所以暂时不用右摇杆按下做 teleop toggle。

## 5. Adapter 当前推荐设置

当前使用 `quest3_stream_adapter` 把 Quest topic 转成 `/phone_pose` 和 `/phone_gripper`。

推荐命令：

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
  -p translation_scale:=0.4 \
  -p max_translation_step:=0.008 \
  -p rotation_scale:=0.0 \
  -p 'x_axis:=-x' \
  -p 'y_axis:=z' \
  -p 'z_axis:="y"' \
  -p deadman_button:=4 \
  -p deadman_latch_button:=-1 \
  -p reset_button:=-1 \
  -p pause_button:=-1 \
  -p record_transition_topic:=/record_transition \
  -p record_button:=4 \
  -p save_button:=0 \
  -p delete_button:=1 \
  -p gripper_requires_deadman:=false \
  -p franka_gripper_command_topic:=/gripper/gripper_position_controller/commands
```

含义：

```text
right grip 按住 = teleop enabled + recording active
right grip 松开 = teleop stop + recorder pause, 等待 A/B
A = save episode
B = delete episode
```

注意：

- 现在不是 toggle 模式。
- 必须一直按住 right grip，Franka body 才会跟随。
- 松开 right grip 后，下一次 record 前必须按 A 保存或 B 删除。

## 6. Franka Controller

今天确认原来的 launch：

```bash
ros2 launch crisp_controllers_robot_demos franka.launch.py ...
```

会导致“只有 gripper 动，Franka body 不动”。

原因：

```text
原 launch 没有让 cartesian_impedance_controller 处在正确 active path。
```

所以 Quest 3 teleop 应该使用新建/修改后的 Cartesian launch：

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

不要主要依赖：

```bash
ros2 control list_controllers
```

因为今天 `/controller_manager/list_controllers` 经常 timeout。

更可靠的检查是：

```bash
ros2 topic info -v /target_pose
ros2 topic echo /current_pose --once
```

期望：

```text
/target_pose 有 publisher 和 cartesian_impedance_controller subscriber
/current_pose 能正常输出
```

## 7. Recorder 当前推荐命令

启动 recorder：

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
  --repo-id quest3_997 \
  --num-episodes 1 \
  --fps 15 \
  --no-push-to-hub \
  --skip-home
```

今天加了 `--skip-home`，避免每次开始 recording 都强制回 home。之前尝试“开始录制自动回初始位”导致 motion/coordinate 行为变怪，所以已经改回不自动回初始位。

## 8. 今天的按钮逻辑设计

最初想做：

```text
右摇杆按下一次 = teleop 开始
再按一次 = teleop 关闭
中指 grip 按下 = 开始 recording
松开 grip = 结束 recording
A = save
B = delete
```

但实际测试发现 right joystick click 没有稳定发布成 button，所以今天最终改成：

```text
按住 right grip = teleop + record
松开 right grip = pause and wait save/delete
A = save
B = delete
```

这个方案更简单，也更安全，因为松手机器人就停。

## 9. 坐姿和坐标系

今天坐标方向调过几次。比较接近可用的当前 mapping 是：

```text
x_axis = -x
y_axis = z
z_axis = y
```

当前观察：

```text
up/down 已经正确
left/right 和 front/back 之前出现过反向
```

如果再次发现：

```text
controller left -> Franka right
controller front -> Franka back
```

说明 `x_axis` 或 `z_axis` 还需要再翻转。

当前 scale：

```text
translation_scale = 0.4
max_translation_step = 0.008
rotation_scale = 0.0
```

今天试过 scale 太大时容易敏感，也触发过力/力矩安全错误，所以目前 scale 先保守。

## 10. Gripper 状态

今天一开始 gripper 不动，Desk 上显示 end effector not connected。

后来 gripper 可以动，说明 gripper command path 已经基本恢复。

adapter 里额外加了直接 Franka gripper command publisher：

```text
franka_gripper_command_topic = /gripper/gripper_position_controller/commands
```

如果再次只有 gripper 动、body 不动，优先怀疑 controller launch 错了，不是 Quest pose 错。

## 11. Franka Safety/Error Recovery

今天触发过：

```text
configured force threshold reached
configured torque threshold reached
Joint Position Error detected
X3.1 Triggered
X4 Not Enabled
```

做过的恢复：

```bash
docker exec -it franka bash -lc 'source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && export ROS_DOMAIN_ID=30 && ros2 action send_goal /action_server/error_recovery franka_msgs/action/ErrorRecovery "{}"'
```

该命令成功过：

```text
Goal finished with status: SUCCEEDED
```

但 Desk 后来进入 Joint Position Error recovery，需要在 Desk UI 里按照 safety flow 恢复 Joint 2/Joint 4。

重要安全结论：

- 如果 Desk 出现红色 robot error，先停止 adapter / recorder / controller。
- 不要继续 teleop。
- X3.1 是 emergency/safety device，不是普通按钮。
- X4 是 external enabling device，需要按住物理 enabling device 才能进行某些 recovery step。
- Joint Position Error recovery 应该按 Desk 提示和 tutor/safety operator 指导操作。

## 12. 当前最后状态

最后用户检查到：

```bash
ros2 topic info /record_transition
ros2 topic info /target_pose
```

结果：

```text
/record_transition
Type: std_msgs/msg/String
Publisher count: 1
Subscription count: 1

/target_pose
Type: geometry_msgs/msg/PoseStamped
Publisher count: 1
Subscription count: 1
```

这说明：

```text
adapter -> recorder 的 record control topic 已连接
recorder/client -> Franka controller 的 target_pose topic 已连接
```

但是用户仍然说 Franka body 不动。

所以现在问题不再是 topic 没有连接，而是要判断：

```text
1. right grip 是否真的被识别成 buttons[4] = 1
2. /target_pose 的数值是否在 controller 移动时变化
3. /current_pose 是否跟随 /target_pose 变化
4. Desk 是否还有 safety/recovery 状态阻止 body motion
```

## 13. 下一步排查顺序

### Step 1: 检查 right grip

按住 right grip 时运行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30

ros2 topic echo /quest/right_controller/joy --once
```

期望：

```text
buttons[4] = 1
```

如果 `buttons[4]` 仍然是 0，说明 adapter 一直没有进入 teleop enabled。

### Step 2: 检查 /target_pose 是否变化

按住 right grip 并移动 controller：

```bash
ros2 topic echo /target_pose --field pose.position
```

如果数值不变：

```text
Quest/adapter/recorder 侧没有把运动变成 target_pose。
```

如果数值在变：

```text
Quest/adapter/recorder 侧是好的。
```

### Step 3: 检查 /current_pose 是否变化

```bash
ros2 topic echo /current_pose --field pose.position
```

如果 `/target_pose` 在变但 `/current_pose` 不变：

```text
问题在 Franka controller / Desk safety / robot execution 侧。
```

如果两个都在变但肉眼觉得小：

```text
scale 太小，可以之后把 translation_scale 从 0.4 调到 0.6 或 0.8。
```

## 14. 今天改过的代码

### quest3_stream_adapter.py

路径：

```text
/home/ros/ros2_ws/src/crisp_gym/crisp_gym/scripts/quest3_stream_adapter.py
```

增加/修改：

- 支持 `x_axis/y_axis/z_axis` 字符串参数。
- 支持 `record_transition_topic`。
- 支持 `record_button/save_button/delete_button`。
- 支持 `deadman_latch_button`。
- 支持直接发布 Franka gripper command。
- 当前推荐使用 `rightGrip` 作为 deadman + recording hold。

### quest_reader_ros_bridge.py

路径：

```text
/home/ros/ros2_ws/src/piper-vr-teleop/scripts/quest_reader_ros_bridge.py
```

增加了 Joy button mapping：

```text
A, B, X, Y, rightGrip, rightJS, rightTrig
```

但 rightJS 实测不稳定，所以暂时不用。

### record_lerobot_format_leader_follower.py

路径：

```text
/home/ros/ros2_ws/src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py
```

增加：

```text
--skip-home
```

用于跳过开始 recording 时的 home/reset motion。

### franka_cartesian_impedance.launch.py

路径：

```text
/home/ros/ros2_ws/src/crisp_controllers_demos/crisp_controllers_robot_demos/launch/franka_cartesian_impedance.launch.py
```

作用：

```text
直接启动适合 /target_pose 的 cartesian_impedance_controller path。
```

这个 launch 是 Quest 3 body teleop 应该用的，不要用普通 `franka.launch.py` 来测 Quest body movement。

## 15. 明天建议从这里继续

1. 确保 Desk 没有红色 error，robot 在 Work/Execution 可用状态。
2. 启动 Franka Cartesian controller。
3. 启动 Quest app。
4. 启动 quest_reader_ros_bridge。
5. 启动 quest3_stream_adapter。
6. 启动 recorder。
7. 按住 right grip，先不要录正式数据，只检查：

```bash
ros2 topic echo /quest/right_controller/joy --once
ros2 topic echo /target_pose --field pose.position
ros2 topic echo /current_pose --field pose.position
```

8. 如果 `/target_pose` 和 `/current_pose` 都正常变化，再录 1 个短 episode。
9. 如果方向仍然反，优先只改 `x_axis/z_axis`，不要同时改 scale 和 reset/home 逻辑。

