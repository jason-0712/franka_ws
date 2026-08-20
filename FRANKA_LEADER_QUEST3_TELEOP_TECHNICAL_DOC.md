# Franka Leader Arm Teleop and Quest3 Teleop Technical Document

本文档解释当前 `franka_ws` 里两套 Franka 遥操链路的技术原理：

1. **leader arm teleop**：用物理 leader arm 的关节角控制 Franka。
2. **Quest3 teleop**：用 Meta Quest3 右手柄的 6D pose 和按键控制 Franka 末端。

重点不是只列命令，而是解释背后的 fundamental rationale：人类输入为什么要被转换成 target command，为什么需要坐标映射、限速、deadman、re-anchor，为什么 leader arm 和 Quest3 数据不能随便混，为什么 StarVLA 部署必须和采集 action space 对齐。

## 1. 核心思想

Franka 遥操不是把人的手或者设备信号直接变成电机力矩。当前系统采用的是更安全、可记录、可训练的分层设计：

```text
human intent
  -> input device signal
  -> teleop adapter
  -> robot target command
  -> low-level controller
  -> Franka motion
  -> observation/action recording
  -> LeRobot dataset
  -> StarVLA fine-tuning/deployment
```

这套设计的基本原则是：

- 人只负责给出“想往哪里动”的意图。
- teleop adapter 把这个意图变成标准 action 或 target。
- Franka controller 负责实际动力学、阻抗、滤波和安全边界。
- recorder 记录 observation 和 action，供 imitation learning / VLA 训练。

所以系统里最关键的问题是：**采集时的人类动作表示，必须和训练/部署时模型输出的动作表示一致。**

如果采集的是 joint action，模型部署时也应该输出 joint action。  
如果采集的是 Cartesian delta EEF action，模型部署时也应该输出 Cartesian delta EEF action。

## 2. 两种遥操的本质差异

| 项目 | Leader arm teleop | Quest3 teleop |
| --- | --- | --- |
| 人类输入 | 物理 leader arm 关节角 | Quest3 controller 6D pose + buttons |
| 控制空间 | 通常是 joint space | Cartesian end-effector space |
| action 形状 | 8D: 7 joints + gripper | 7D: dx dy dz droll dpitch dyaw gripper |
| 坐标映射难度 | 低，关节对应较直接 | 高，需要 Quest frame 到 Franka base frame 映射 |
| 直觉 | 像操作另一个机械臂 | 像用手柄在空间里拖动 gripper |
| 主要风险 | 关节标定、噪声、速度突变 | 左右反、前后反、re-anchor、tracking 漂移、RPY 抖动 |
| 数据适配 | 更适合 joint-action policy | 更适合 Cartesian delta EEF policy / StarVLA |

当前你用于 StarVLA 的主线是 **Quest3 -> 7D Cartesian delta EEF action**，不是 leader arm 的 8D joint action。

## 3. 当前代码里的关键文件

Leader arm 相关：

- `src/crisp_gym/crisp_gym/teleop/teleop_robot_servo.py`
- `src/crisp_gym/crisp_gym/record/record_functions.py`
- `src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py`
- `src/crisp_gym/crisp_gym/config/teleop/right_leader.yaml`

Quest3 相关：

- `scripts/quest_reader_ros_bridge.py`
- `src/crisp_gym/crisp_gym/scripts/quest3_stream_adapter.py`
- `src/crisp_gym/crisp_gym/teleop/teleop_sensor_stream.py`
- `src/crisp_gym/crisp_gym/scripts/record_lerobot_format_leader_follower.py`

Franka environment / action execution：

- `src/crisp_gym/crisp_gym/envs/manipulator_env.py`
- `src/crisp_gym/crisp_gym/envs/manipulator_env_config.py`
- `src/crisp_gym/crisp_gym/config/envs/third_person_cam_franka.yaml`
- `src/crisp_gym/crisp_gym/config/envs/no_cam_franka.yaml`

## 4. Leader Arm Teleop

### 4.1 数据流

Leader arm 的典型数据流是：

```text
leader arm hardware / servo reader
  -> /servo_angles
  -> JointControlNode
  -> filtered target_joint
  -> robot.set_target_joint(...)
  -> /target_joint
  -> joint_impedance_controller
  -> Franka motion
```

recorder 这边不是重新解释 leader arm，而是直接读取：

```text
JointControlNode.last_action
```

然后保存进 LeRobot dataset。

### 4.2 输入格式

`/servo_angles` 是 `std_msgs/msg/Float64MultiArray`，至少 8 个数：

```text
[servo_0, servo_1, servo_2, servo_3, servo_4, servo_5, servo_6, gripper]
```

前 7 个是 leader arm 的关节角输入，第 8 个是 gripper 输入。

当前代码里 gripper 逻辑是：

```python
target_gripper = 0.0 if gripper >= 300 else 1.0
```

也就是：

```text
leader gripper >= 300 -> close -> action 0.0
leader gripper < 300  -> open  -> action 1.0
```

注意这里的 action 语义是：

```text
0.0 = close
1.0 = open
```

### 4.3 Zeroing / Home Anchor

leader arm 启动后第一帧会被当作 zero / initial position：

```python
self.initial_angles = angles.copy()
self.filtered_angles = angles.copy()
```

之后每一帧计算：

```text
angle_delta = current_filtered_angles - initial_angles
target_joint = robot_home_joint_values + angle_delta_rad
```

这意味着 leader arm 的意义不是“绝对关节角等于 Franka 关节角”，而是：

```text
leader arm 相对启动时姿态的变化
  -> 加到 Franka 当前/标准 home joint 上
  -> 得到 Franka target_joint
```

这就是为什么启动时 leader arm 和 Franka 的初始状态很重要。  
如果 zero pose 不干净，后续所有 target 都会偏。

### 4.4 Smoothing, Deadband, Velocity Limit

leader arm 有物理噪声和人手抖动，所以不能直接把角度差发给 Franka。

当前代码使用指数平滑：

```python
self.filtered_angles += alpha * (angles - self.filtered_angles)
```

其中：

```text
alpha = 0.45
```

如果角度变化小于 deadband，会被当作 0：

```text
angle_deadband_deg = 0.02 deg
```

joint target 发布频率：

```text
command_period = 0.01 s
publish rate ~= 100 Hz
```

速度和加速度限制：

```text
max_joint_velocity = 0.8 rad/s
max_joint_acceleration = 1.2 rad/s^2
```

核心控制逻辑是：不要一步跳到目标，而是每 0.01 秒往目标走一小步。

代码里使用 braking velocity：

```python
braking_velocity = sqrt(2 * max_acceleration * abs(position_error))
```

直观含义是：越接近目标，允许速度越低，避免冲过头。

还有 overshoot prevention：

```python
if abs(step) > abs(position_error):
    step = position_error
```

这保证了 target_joint 不会因为速度积分越过目标。

### 4.5 Leader Arm 记录的 Action

`make_servo_teleop_fn()` 会返回：

```python
action = leader.last_action
obs = env._get_obs()
```

而 `leader.last_action` 是：

```text
[target_joint_0, ..., target_joint_6, gripper]
```

所以 leader arm 录下来的数据本质是 **joint target action**，不是 Cartesian delta。

这也是为什么它不能直接和 Quest3 的 7D Cartesian delta EEF 数据混在一个 StarVLA action config 里。  
两者 action semantics 不同，即使都是 float array，含义完全不同。

### 4.6 Leader Arm 的优点

Leader arm 的优点：

- 物理 arm 和 Franka 都是 7-DoF manipulator，关节空间对应关系比较自然。
- 坐标系歧义少，不太会出现“我往左但机器人往右”的问题。
- joint target 可以被 smooth/limit 得很稳定。
- 适合作为安全 baseline 或把 Franka 返回 standard pose。

### 4.7 Leader Arm 的主要问题

Leader arm 的问题：

- 需要良好的初始 zero pose。
- 关节 offset 和真实 Franka workspace 不一定完全一致。
- 人操作的是关节空间，不一定等价于任务空间最自然的运动。
- 如果想训练视觉语言模型做 pick/place，joint action 对模型来说更难泛化，因为同一个视觉任务可能有多组 joint 解。

## 5. Quest3 Teleop

### 5.1 数据流

当前 Quest3 数据流是：

```text
Quest3 right controller
  -> OpenXR pose/button sample
  -> adb logcat / Piper QuestReader
  -> scripts/quest_reader_ros_bridge.py
  -> /quest/right_controller/pose
  -> /quest/right_controller/joy
  -> quest3_stream_adapter.py
  -> /phone_pose
  -> /phone_gripper
  -> record_lerobot_format_leader_follower.py
  -> ManipulatorCartesianEnv.step(action)
  -> robot.set_target(pose=target_pose)
  -> /target_pose
  -> cartesian_impedance_controller
  -> Franka motion
```

Quest3 bridge 只负责把 QuestReader 里的 controller transform 和 button 状态变成 ROS topic。

核心 topic：

```text
/quest/right_controller/pose   geometry_msgs/msg/PoseStamped
/quest/right_controller/joy    sensor_msgs/msg/Joy
```

adapter 再把它转换成：

```text
/phone_pose                         geometry_msgs/msg/PoseStamped
/phone_gripper                      std_msgs/msg/Float32
/quest/right_controller/delta_end_pose  std_msgs/msg/Float64MultiArray
/record_transition                  std_msgs/msg/String
/gripper/gripper_position_controller/commands
```

如果启用 direct mode，还会发布：

```text
/target_pose
```

### 5.2 Quest3 Bridge 的作用

`scripts/quest_reader_ros_bridge.py` 做几件事：

1. 从 `piper_vr.quest_reader.QuestReader` 读取 Quest sample。
2. 从 `sample.transforms_openxr["right"]` 取右手柄 4x4 transform。
3. 把 transform 的 translation 填进 `PoseStamped.position`。
4. 把 rotation matrix 转成 quaternion。
5. 读取按钮：
   - A/B/X/Y
   - rightGrip
   - rightTrig
6. 发布 Joy。

当前 Joy 里：

```text
buttons[0] = A
buttons[1] = B
buttons[2] = X
buttons[3] = Y
buttons[4] = rightGrip >= threshold
axes[5] = -1.0 if rightTrig pressed else 1.0
```

所以后面的 adapter 里常见：

```text
deadman_button = 4
trigger_axis = 5
trigger_threshold = 0.8 or -0.5 depending command
```

### 5.3 Quest3 Adapter 的本质

`quest3_stream_adapter.py` 的核心任务是：

```text
Quest controller raw delta
  -> axis mapping
  -> scale
  -> clipping
  -> virtual Cartesian pose delta
  -> /phone_pose and /phone_gripper
```

它不是一个 policy，也不是 recorder。  
它是一个 **input adapter / coordinate transformer / safety limiter**。

### 5.4 Deadman 的意义

deadman 是安全开关。  
只有 deadman pressed 时，Quest controller 的运动才会累积成 robot target。

在当前常用配置里：

```text
deadman_button:=4
```

也就是右手 grip。

这样做的原因：

- 松手时不会继续发送运动。
- 你可以移动自己的手柄姿态而不影响 Franka。
- re-anchor 时可以避免跳变。
- recorder 可以用同一个按钮触发 record transition。

### 5.5 Re-anchor 的意义

Quest3 controller 是一个自由空间设备，没有物理 home joint。  
如果系统直接使用 controller 的绝对位置，很容易出现 jump。

所以当前 adapter 使用的是相对运动：

```python
raw_delta = raw_pos - last_raw_pos
```

而 re-anchor 的作用是：

```text
把当前 controller pose 设为新的 last_raw_pose
但不移动 robot target
```

也就是说，re-anchor 是“重设手柄零点”，不是“让 Franka 回 home”。

如果你感觉方向突然变奇怪、controller 拿得不舒服、手柄位置快到身体边缘了，就应该 re-anchor。

### 5.6 Axis Mapping

Quest3 和 Franka 的坐标系不是同一个。

Quest3 raw delta 是：

```text
[quest_dx, quest_dy, quest_dz]
```

Franka base frame 需要：

```text
[franka_dx, franka_dy, franka_dz]
```

adapter 用 signed permutation matrix 做映射：

```python
mapped_delta = quest_to_franka @ raw_delta
```

常用参数类似：

```text
x_axis:="-x"
y_axis:="z"
z_axis:="y"
```

含义是：

```text
Franka x  <- - Quest x
Franka y  <-   Quest z
Franka z  <-   Quest y
```

这解释了你之前经常遇到的现象：  
从 camera perspective 看“左/右/前/后”不一定等于 Franka base frame 的 x/y/z。

真正决定机器人运动的是 Franka base frame，不是相机画面，也不是你坐的位置。

### 5.7 Scale 和 Clipping

Quest controller 原始位移会乘 scale：

```python
scaled_delta = mapped_delta * axis_scale * translation_scale
```

对应参数：

```text
translation_scale
x_scale
y_scale
z_scale
```

然后限制每一步最大位移：

```python
clipped_delta = clip_vector_norm(scaled_delta, max_translation_step)
```

这一步非常重要。  
如果没有 clipping，Quest tracking 抖动、USB/logcat 延迟、手柄瞬移都会让 Franka target jump。

### 5.8 Virtual Pose 和 Pending Delta

adapter 维护两个概念：

```text
virtual_pos / virtual_rot
pending_delta_pos / pending_delta_rot
```

`virtual_pos` 是从 teleop 开始后累计出来的虚拟位姿，发布到：

```text
/phone_pose
```

`pending_delta_pos` 是从上一次 publish 到这一次 publish 之间累计的增量，发布到：

```text
/quest/right_controller/delta_end_pose
```

每次 publish 后：

```python
self.pending_delta_pos = zeros
self.pending_delta_rot = identity
```

所以 Quest3 录制出来的 action 本质是每一帧的 delta，而不是绝对目标坐标。

### 5.9 Recorder 如何生成 Quest3 Action

Quest3 录制时，recorder 使用：

```python
TeleopStreamedPose
```

它订阅：

```text
/phone_pose
/phone_gripper
```

`make_teleop_streamer_fn()` 里计算：

```python
action_pose = pose - prev_pose
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
env.step(action)
```

所以 Quest3 dataset 的 action 是：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

这就是你 Hugging Face dataset 里看到的：

```json
"action": {
  "shape": [7],
  "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
}
```

### 5.10 Cartesian Env 如何执行 Action

`ManipulatorCartesianEnv.step()` 的核心逻辑是：

```python
target_position = robot.target_pose.position + action[:3]
target_orientation = delta_rotation * robot.target_pose.orientation
robot.set_target(pose=target_pose)
```

如果 `use_relative_actions=True`，action 就是 delta。  
这和 Quest3 recorder 生成的 delta action 是匹配的。

当前 Quest3 -> StarVLA 的关键就是：

```text
policy 输出 7D delta EEF action
deployment client 把 delta 加到当前/目标 target_pose
cartesian_impedance_controller 跟踪 /target_pose
```

### 5.11 Direct Target Pose Mode vs Recording Mode

Quest3 adapter 有两种模式。

Recording mode：

```text
publish_target_pose:=false
```

此时 adapter 只发布：

```text
/phone_pose
/phone_gripper
```

然后 recorder/env 拥有 `/target_pose` publisher。

Direct target pose mode：

```text
publish_target_pose:=true
```

此时 adapter 自己发布 `/target_pose`，可以用来调 standard pose。

这两种模式不能混用。  
同一时间只能有一个 `/target_pose` publisher。

如果同时存在两个 publisher，controller 会报：

```text
Ignoring target_pose message due to multiple publishers detected
```

此时 Franka 可能不动或者行为混乱。

### 5.12 Quest3 的 RPY

Quest3 controller 本身有 orientation，理论上可以控制 roll/pitch/yaw。

但是当前任务主要是 pick cube/place box，前期为了稳定，常用：

```text
rotation_scale:=0.0
```

这表示：

```text
禁用 Quest orientation streaming
```

所以数据里 action 的 roll/pitch/yaw 大多接近 0。  
dataset schema 里仍然有 `roll/pitch/yaw`，但这不代表你每次采集都真的用了明显 RPY。

如果后续要引入 wrist rotation，需要非常谨慎：

- 先确认 Quest rotation axis mapping。
- 先小 scale，比如 `rotation_scale:=0.05`。
- 设置 `max_rotation_step`。
- 单独录少量测试数据。
- 确认没有“移动左却出现大 yaw rotation”的问题。

## 6. Gripper 语义

这里最容易混淆。

### 6.1 Action 里的 gripper

在当前 Quest/leader action 里，常用语义是：

```text
1.0 = open
0.0 = close
```

Quest3 adapter 里：

```python
gripper_open_value = 1.0
gripper_closed_value = 0.0
```

trigger pressed 时 close，release 后 open。

### 6.2 Observation 里的 gripper

env observation 里有：

```python
gripper_value = 1 - np.array([self.gripper.value])
```

所以 observation state 的 gripper 可能和 action gripper 的直观语义不完全一样。

这就是为什么你之前看到：

```text
observation.state.gripper constant 0.0
action gripper constant 1.0
```

需要分别判断：

- action gripper 是人/模型想让 gripper open/close。
- observation.state.gripper 是系统读到的 gripper state，并且做了 `1 - value`。

训练时必须保证 modality/config 对的是正确 key 和正确语义。

## 7. 数据集和 StarVLA 的关系

你的 Quest3 Franka dataset 是 LeRobot v2.1 格式，典型 features 是：

```text
observation.images.primary: 256x256 RGB video
observation.state.cartesian: [x, y, z, roll, pitch, yaw]
observation.state.gripper: [gripper]
observation.state.joints: [joint_0 ... joint_6]
observation.state.target: [target_x ... target_yaw]
observation.state: concatenated 20D state
action: [x, y, z, roll, pitch, yaw, gripper]
```

StarVLA 训练时，我们把它解释成：

```text
state.eef_position      <- observation.state cartesian xyz
state.eef_rotation      <- observation.state cartesian rpy
state.gripper           <- observation/state gripper
action.delta_eef_position <- action xyz
action.delta_eef_rotation <- action rpy
action.gripper            <- action gripper
```

也就是说，StarVLA 输出的不是 absolute pose，而是：

```text
delta EEF position + delta EEF rotation + gripper
```

deployment client 再把它转成 `/target_pose`。

## 8. 为什么 Quest3 更适合当前 StarVLA 路线

当前任务是视觉 pick/place：

```text
camera image -> infer where cube/box is -> move gripper in Cartesian space
```

这和 Quest3 的控制方式比较一致：

```text
human moves controller in task space
-> robot EEF moves in task space
-> recorded action is EEF delta
-> StarVLA predicts EEF delta
```

leader arm joint action 虽然稳定，但它的数据是：

```text
visual scene -> joint target
```

对于模型来说，这通常更难，因为同一个视觉目标可能对应很多 joint configuration。

所以你现在做 StarVLA Franka delta EEF policy，选择 Quest3 数据是合理的。

## 9. 为什么 Quest3 容易出现方向反了

方向反通常不是模型问题，而是 frame 问题。

有三套 frame 容易混在一起：

1. Quest controller/OpenXR frame。
2. Franka base frame。
3. Camera image frame。

例如你说“从相机看 gripper 应该往右”，这不一定等于 Franka base 的 `+y`。  
如果相机是斜着看，image right 可能混合了 Franka `+x` 和 `+y`。

Quest3 adapter 只知道：

```text
raw Quest delta -> mapping matrix -> Franka base delta
```

它不知道你坐在什么位置，也不知道相机画面里哪个方向是右。

所以 debug 方向时，应该看：

```bash
ros2 topic echo /current_pose --once
```

然后观察：

```text
Franka x/y/z 数值变化
```

而不是只看相机画面里的左右。

## 10. 为什么会出现“移动左，Franka 旋转”的现象

常见原因：

1. `rotation_scale` 不是 0，Quest orientation 抖动被当成 RPY action。
2. controller 没有 re-anchor，raw rotation delta 积累异常。
3. 你身体方向、Quest frame、Franka base frame 不一致。
4. 旧 adapter 或 recorder 还在发布 `/target_pose`。
5. Franka target pose 已经靠近 workspace/contact/safety limit，translation 被限制后看起来像 rotation。

如果只是采集 pick/place 平移数据，建议：

```text
rotation_scale:=0.0
```

这样 dataset schema 里仍有 rpy，但 action rpy 接近 0，模型主要学 xyz/gripper。

## 11. 为什么 StarVLA 部署会“往前越过 cube”

从目前测试现象看，可能有几类原因：

1. **训练数据分布问题**：很多 episode 从 standard pose 开始，主要动作是 x+ 和 z-，lateral y correction 比例太少。
2. **视觉单相机限制**：第三人称相机对深度、前后位置和遮挡不敏感，模型可能把 cube/box 的相对深度估错。
3. **action chunk 开环漂移**：如果一次执行太多步，早期小误差会累积；所以 deployment 里更适合 `execution_horizon=1`。
4. **gripper phase 学得不好**：如果 gripper action 长时间是 1.0，模型不知道何时 close。
5. **standard pose/scene mismatch**：采集和部署时 cube/box 的相对位置、光照、相机角度、Franka 初始姿态不同。
6. **frame interpretation mismatch**：训练 action 的 y 是 Franka base y，不是 image right/left；如果人按图像直觉判断，容易误判。

这也是为什么 tutor 建议：

- 加 wrist camera / D435i。
- 做 DAgger-lite correction。
- open-loop 比较 predicted action 和 ground-truth action。
- 从 Qwen base + 全数据训练，并 freeze vision 检查是否有明显代码 bug。

## 12. DAgger-lite 的技术含义

DAgger 的核心思想是：

```text
让当前 policy 先跑
在它容易错的状态分布上，由人类接管给出正确动作
把这些 correction states/actions 加回训练集
```

普通 imitation learning 只看人类从 clean initial state 完成任务的数据。  
但部署时模型会进入自己造成的错误状态，比如：

```text
gripper 已经越过 cube
gripper 在 box 上方但还没抓 cube
z 太低但 y 没对齐
```

这些状态普通数据里很少。  
DAgger-lite 就是专门采集这些失败状态下的 correction。

你现在做的流程：

```text
StarVLA 从 standard pose 开始跑
发现方向错/越过 cube
停止 policy
从当前失败状态用 Quest3 teleop 接管完成任务
把这段 correction episode 录进 LeRobot
重新 fine-tune
```

这就是 DAgger-lite。

## 13. 两套遥操为什么不能随便混数据

即使两个数据集都叫 `action`，也可能完全不是同一个 action。

Leader arm：

```text
action = [target_joint_0 ... target_joint_6, gripper]
```

Quest3：

```text
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

如果把这两种 action 混进一个 dataset mixture，而 registry/modality 仍把它们都解释成 delta EEF，就会出现严重错误：

```text
模型以为 joint_0 是 dx
模型以为 joint_1 是 dy
...
```

训练 loss 可能还能下降，但部署行为会完全不可信。

所以必须坚持：

- Quest3 delta EEF 数据只进 `quest3_franka_delta_eef` 这类 robot_type。
- leader arm joint 数据只进 joint-action config。
- 两者要混，必须先转换到同一个 action space。

## 14. 采集时的标准模式

Quest3 采集建议模式：

```text
Franka controller terminal
Quest MR app terminal
Quest ROS bridge terminal
Quest3 stream adapter terminal
Recorder terminal
```

控制所有 terminal 都要：

```bash
export ROS_DOMAIN_ID=30
```

采集时 adapter 应该是 recording mode：

```text
publish_target_pose:=false
```

也就是让 recorder/env 拥有 `/target_pose`。

调 standard pose 时才使用 direct mode：

```text
publish_target_pose:=true
```

调完必须停掉 direct adapter，再启动 recorder。

## 15. 部署时的标准模式

StarVLA 部署时，Quest3 adapter 不应该继续发布 `/target_pose`。

部署链路应该是：

```text
RealSense image + /current_pose
  -> StarVLA policy server
  -> starvla_franka_delta_pose_client.py
  -> /target_pose
  -> cartesian_impedance_controller
```

部署前检查：

```bash
ros2 topic info -v /target_pose
```

应该只有一个 publisher：StarVLA client。  
如果看到 `quest3_stream_adapter`，就说明 Quest teleop 还在抢 `/target_pose`。

## 16. Fundamental Rationale 总结

Leader arm teleop 的本质是：

```text
用一个物理 7-DoF 输入设备给 Franka 产生 smooth joint target
```

它依赖的是 joint correspondence 和安全滤波。  
它稳定、直接，但 action 是 joint-space。

Quest3 teleop 的本质是：

```text
用 6D controller pose 产生 Cartesian EEF delta target
```

它依赖的是 coordinate transform、deadman、re-anchor、scale/clipping。  
它更适合视觉 pick/place 和 StarVLA delta EEF policy，但更容易出现 frame mapping 问题。

StarVLA 部署的本质是：

```text
把人类 Quest3 采集到的视觉条件下 delta EEF 行为
学习成 image/language -> action chunk 的函数
```

所以最重要的工程约束是：

```text
采集 action space == 训练 action space == 部署 execution space
```

一旦这个约束被破坏，哪怕 loss 看起来下降，真机也很可能不 work。

## 17. Practical Checklist

采集前：

```text
1. 确认 ROS_DOMAIN_ID=30。
2. 确认 Franka controller 是 cartesian_impedance_controller。
3. 确认只有 recorder/env 或 direct adapter 其中一个发布 /target_pose。
4. 确认 Quest bridge 有 /quest/right_controller/pose 和 /joy。
5. 确认 adapter 有 /phone_pose 和 /phone_gripper。
6. 确认 gripper open=1.0 close=0.0。
7. 确认 rotation_scale 是否按本次目的设置，普通 pick/place 建议 0.0。
8. 确认 standard pose、cube/box 位置、相机视角和训练分布一致。
```

部署前：

```text
1. 停掉 Quest3 stream adapter 或任何 teleop target publisher。
2. 启动 policy server。
3. 启动 StarVLA client dry-run。
4. 看 dpos 是否方向合理，尤其 x/y/z sign。
5. dry-run 合理后再 --execute。
6. execution_horizon 建议先用 1，持续闭环重规划。
7. max_trans_delta、rate、z_min 要保守。
8. 手放 E-stop。
```

debug 时：

```text
1. 不要只看相机画面判断左右，必须看 /current_pose 数值。
2. 如果 /target_pose 多 publisher，先清掉。
3. 如果 gripper 手动 close 后又 open，通常是还有 adapter/recorder 在持续发 open。
4. 如果 RPY 意外变化，先把 rotation_scale 设回 0.0。
5. 如果模型一直 x+ z-，说明它学到的是 approach prior，不是 object-conditioned correction。
6. 如果 y_abs 明显偏小，需要更多 lateral correction 或更强视觉输入。
```

