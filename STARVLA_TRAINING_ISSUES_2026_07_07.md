# StarVLA Training Issues - 2026-07-07

## 1. 这份文档要回答的问题

当前需要澄清两个核心问题：

1. 数据和控制到底是：
   - delta end pose
   - delta joint angle
   - absolute joint angle

2. 在当前 setup 下，StarVLA policy server 和 Franka laptop 的运行时间是多少：
   - Franka 端收到新 action chunk 的间隔
   - 模型推理时间
   - client 端 round-trip 时间

这两个问题必须先说清楚，否则后面训练、部署、Quest 3 teleop 会混在一起。

## 2. 当前结论总览

当前真实 Franka StarVLA 部署链路使用的是：

```text
StarVLA output: delta joint angle + gripper
Franka controller input: absolute joint target
Controller topic: /target_joint
Controller type: joint_impedance_controller
```

也就是说，模型输出不是 delta end pose，也不是直接输出 absolute joint angle。

当前模型输出：

```text
[delta_joint_0, delta_joint_1, ..., delta_joint_6, gripper]
```

Franka client 执行时会做：

```text
target_joint = current_target_joint + clipped_delta_joint
publish target_joint to /target_joint
```

所以从 policy 角度看是：

```text
delta joint angle
```

从 Franka controller topic 角度看是：

```text
absolute joint target
```

这两个不要混淆。

## 3. StarVLA LeRobot Training Dataset 是什么

这里要特别澄清：

```text
LeRobot 是一种 dataset 存储/加载格式，不等于 simulation dataset。
```

StarVLA guideline 里最常见的 LeRobot dataset 确实来自 simulation，例如 LIBERO 被转换成 LeRobot 格式：

```text
LIBERO simulation data -> LeRobot format -> StarVLA training
```

但真实机器人数据也可以用同样的 LeRobot 文件结构保存：

```text
real Franka teleop data -> LeRobot format -> StarVLA fine-tuning
```

所以这里说的 “StarVLA LeRobot training dataset” 不是指“simulation 数据本身”，而是指：

```text
StarVLA dataloader 能读取的 LeRobot-style dataset format。
```

当前有两类需要分开：

| 数据来源 | 是否 simulation | 是否 LeRobot 格式 | 当前 action semantic |
| --- | --- | --- | --- |
| LIBERO / original StarVLA 20k training | 是 | 是 | simulation action，通常是 delta pose / delta qpos 类定义，取决于 modality |
| CRISP Franka 原始 teleop data | 否，真实 Franka | 是 | absolute joint target + gripper |
| CRISP Franka delta copy | 否，真实 Franka | 是 | delta joint angle + gripper |

当前用于真实 Franka fine-tune / deploy 的数据有两套。

### 3.1 原始 CRISP Franka 数据

路径：

```bash
dataset/snkdjn/franka_test_*
```

对应 modality：

```json
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
}
```

所以原始 recorded LeRobot action 是：

```text
[target_joint_0, ..., target_joint_6, gripper]
```

这是 absolute joint target，不是 delta joint，也不是 delta end pose。

### 3.2 转换后的 delta joint 数据

路径：

```bash
dataset/snkdjn_delta/franka_test_*
```

对应 modality：

```json
"action": {
  "delta_joints": {
    "start": 0,
    "end": 7,
    "original_key": "action",
    "absolute": false
  },
  "gripper": {
    "start": 7,
    "end": 8,
    "original_key": "action",
    "absolute": true
  }
}
```

所以 delta dataset 的 action 是：

```text
[delta_joint_0, ..., delta_joint_6, gripper]
```

转换公式来自：

```bash
src/crisp_gym/crisp_gym/scripts/convert_lerobot_abs_joint_to_delta_joint.py
```

公式：

```text
delta_joint[t] = target_joint[t + 1] - target_joint[t]
last frame delta = 0
gripper copied from original action
```

因此，当前 StarVLA 部署 checkpoint 对应的数据理解应该是：

```text
训练数据: delta joint angle
模型输出: delta joint angle
部署执行: delta joint angle -> integrated into absolute joint target
```

## 4. Franka Record Data 是什么

Franka recording 使用 CRISP Gym / LeRobot format。

当前 servo leader arm 录制路径里，record function 是：

```bash
src/crisp_gym/crisp_gym/record/record_functions.py
```

关键函数：

```python
make_servo_teleop_fn(env, leader)
```

它直接保存：

```python
action = leader.last_action
```

而 `leader.last_action` 来自：

```bash
src/crisp_gym/crisp_gym/teleop/teleop_robot_servo.py
```

在 servo teleop 中，`last_action` 是：

```text
[target_joint_0, ..., target_joint_6, gripper]
```

因此，原始 Franka record data 是：

```text
absolute joint target + gripper
```

它不是 end-effector pose，也不是 delta end pose。

后面为了训练更稳定和便于部署，我们又把它转换成：

```text
delta joint angle + gripper
```

这就是 `dataset/snkdjn_delta`。

## 5. Leader Arm 发给 Franka Controller 的 Topic / Data

Leader arm 本身不是直接发 `/target_joint`。

Leader arm reader 发布：

```text
/servo_angles
```

消息类型：

```text
std_msgs/msg/Float64MultiArray
```

数据含义：

```text
[servo_angle_0, ..., servo_angle_6, gripper_raw]
```

这些是 leader arm servo 的角度读数，单位在 teleop 里按 degree 处理。

然后 `teleop_robot_servo.py` 订阅 `/servo_angles`，做下面几步：

1. 第一帧作为 leader arm 初始零点：

```text
initial_angles = angles.copy()
```

2. 后续计算 leader arm 相对初始位置的变化：

```text
angle_delta = filtered_angles - initial_angles
```

3. 把 degree 转成 rad：

```text
angle_delta_rad = np.deg2rad(angle_delta)
```

4. 加到 Franka home joint 上：

```text
target_joint = home_joint_values + angle_delta_rad
```

5. 内部经过速度/加速度平滑后调用：

```text
robot.set_target_joint(filtered_joint_target)
```

最终真正发给 Franka controller 的是：

```text
/target_joint
```

消息类型：

```text
sensor_msgs/msg/JointState
```

数据含义：

```text
absolute joint target positions
```

所以 leader arm 控制链路是：

```text
/servo_angles
-> teleop_robot_servo.py
-> absolute target_joint
-> /target_joint
-> joint_impedance_controller
```

一句话：

```text
leader arm 原始输入是 servo angle delta；
Franka controller 接收的是 absolute joint target。
```

## 6. 当前 StarVLA Deploy 发给 Franka Controller 的数据

当前部署脚本：

```bash
scripts/starvla_franka_delta_joint_client.py
```

模型输出：

```text
[delta_joint_0, ..., delta_joint_6, gripper]
```

客户端执行：

```python
delta = np.clip(action[:7], -args.max_delta, args.max_delta)
target_joint = np.clip(target_joint + delta, FR3_LOWER, FR3_UPPER)
publish /target_joint
```

所以当前 StarVLA deploy 链路是：

```text
StarVLA predicts delta joint angle
client integrates delta into absolute joint target
client publishes absolute joint target to /target_joint
joint_impedance_controller executes it
```

当前不是 end pose control。

当前不使用：

```text
/target_pose
cartesian_impedance_controller
```

## 7. Quest 3 TODO：建议使用 Delta End Pose

Quest 3 的当前 bridge 是：

```bash
scripts/quest_reader_ros_bridge.py
```

它发布：

```text
/quest/right_controller/pose
/quest/right_controller/joy
```

其中 `/quest/right_controller/pose` 是：

```text
geometry_msgs/msg/PoseStamped
```

当前它只是把 Quest controller pose 发到 ROS，还没有直接控制 Franka。

根据 @LI JI 的建议，Quest 3 teleop 应该使用：

```text
delta end pose
```

也就是：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

原因：

- Quest controller 本身天然是 6D pose input
- 人手移动更接近 end-effector Cartesian movement
- 用 delta end pose 更直观
- 不需要人手直接对应 Franka 7 个 joint

如果以后用 Quest 3 采数据，建议新建一条独立数据路线：

```text
Quest 3 pose
-> delta end pose
-> Cartesian target pose
-> /target_pose
-> cartesian_impedance_controller
```

对应训练数据应该是：

```text
delta end pose + gripper
```

而不是现在的：

```text
delta joint angle + gripper
```

注意：这会是新的 action space。

不能把 Quest 3 delta end pose 数据直接混到当前 delta joint angle 模型里，除非重新设计 dataset modality / action keys / controller wrapper。

## 8. 推荐统一方案

### 当前阶段：继续用 Delta Joint Angle

因为当前已经打通：

```text
StarVLA delta joint
-> /target_joint
-> joint_impedance_controller
-> real Franka moves
```

所以短期内建议继续沿用：

```text
delta joint angle + gripper
```

优点：

- 当前训练和部署已经跑通
- 不需要改 controller
- 不需要重写 action wrapper
- 安全限制可以直接用 `--max-delta`

### Quest 3 新数据阶段：用 Delta End Pose

如果后续使用 Quest 3 作为主要 teleop source，建议重新采集/转换为：

```text
delta end pose + gripper
```

并使用：

```text
cartesian_impedance_controller
/target_pose
```

这应该作为下一条路线，不要和当前 delta joint route 混在一个模型里。

## 9. 当前 StarVLA 训练配置

当前 delta joint config：

```bash
third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_crisp_franka_delta_joints.yaml
```

关键字段：

```yaml
action_dim: 8
state_dim: 8
action_horizon: 8
data_root_dir: /home/dase-hw101/franka_ws/dataset/snkdjn_delta
data_mix: crisp_franka_pick_cube_place_bowl_delta_20eps
action_type: delta_joint
action_mode: abs
```

这里最容易误解的是：

```yaml
action_mode: abs
```

这并不是说 action 是 absolute joint。

这里的含义是：

```text
parquet 里的 action 已经提前转换成 delta joint；
所以 StarVLA dataloader 不要再次做 delta transform。
```

如果这里设成 delta，可能会造成 double-delta，数据会坏掉。

当前 DataConfig：

```text
CrispFrankaDeltaJointsDataConfig
```

对应 action keys：

```text
action.delta_joints: 7
action.gripper: 1
```

## 10. 运行时间分析：要量哪些时间

现在要区分三个时间。

### 10.1 模型推理时间

字段：

```text
server_predict_action_s
```

含义：

```text
StarVLA server 内部 policy.predict_action 的时间。
```

这个由 StarVLA server 返回。

代码位置：

```bash
third_party/starVLA/deployment/model_server/tools/websocket_policy_server.py
```

server 在处理请求时会做：

```python
infer_start = time.perf_counter()
output_dict = self._policy.predict_action(**payload)
infer_s = time.perf_counter() - infer_start
```

然后返回：

```json
"timing": {
  "server_predict_action_s": infer_s
}
```

### 10.2 Franka laptop 看到的 round-trip 时间

字段：

```text
client_policy_roundtrip_s
```

含义：

```text
Franka laptop 从发送 WebSocket request 到收到 action response 的总时间。
```

它包含：

- client send
- network transfer
- server queue
- model predict_action
- server response
- client receive/unpack

它不完全等于纯模型时间。

如果：

```text
client_policy_roundtrip_s >> server_predict_action_s
```

说明网络、serialization 或 server queue 也有明显开销。

### 10.3 收到新 action chunk 的间隔

字段：

```text
request_interval_s
```

含义：

```text
两次向 server 请求 action chunk 之间的时间。
```

当前 client 每次请求一个 action chunk。

每个 chunk 是：

```text
8 steps
```

如果运行：

```bash
--rate 1
```

每个 action step 执行 1 秒，那么一个 8-step chunk 大约消耗：

```text
8 seconds
```

所以第二次 request 的间隔大约是：

```text
8 seconds + small overhead
```

如果运行：

```bash
--rate 2
```

每个 action step 执行 0.5 秒，那么一个 8-step chunk 大约消耗：

```text
4 seconds
```

所以 request interval 大约是：

```text
4 seconds + small overhead
```

注意：

```text
request_interval_s 不是单个 action step 的间隔。
```

单个 action step 的间隔由：

```bash
--rate
```

决定。

controller topic 的重复 publish 频率由：

```bash
--publish-rate
```

决定。

## 11. 如何测运行时间

使用当前 client 自带的：

```bash
--log-timing
```

### 11.1 Dry-run timing test

先不要移动机器人：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --max-delta 0.01 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 32 \
  --log-timing
```

为什么建议 `--max-steps 32`：

```text
action_horizon = 8
32 steps = 4 chunks
```

这样至少能看到 4 次 request，才能分析 request interval。

### 11.2 Real execute timing test

确认安全后再加：

```bash
--execute
```

命令：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --execute \
  --max-delta 0.01 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 16 \
  --log-timing
```

先用 16 steps，因为真实机器人会动。

如果一切稳定，再扩展到 32 steps。

## 12. Timing 输出怎么读

输出会有类似：

```text
timing_request: {
  "request_id": 0,
  "request_interval_s": null,
  "server_predict_action_s": 0.85,
  "client_policy_roundtrip_s": 0.93
}
```

解释：

- `request_id=0` 是第一次请求，所以 `request_interval_s=null`
- `server_predict_action_s=0.85` 表示 server 端模型推理约 0.85 秒
- `client_policy_roundtrip_s=0.93` 表示 Franka laptop 端看到完整 request-response 约 0.93 秒

第二次 request 可能类似：

```text
timing_request: {
  "request_id": 1,
  "request_interval_s": 8.95,
  "server_predict_action_s": 0.86,
  "client_policy_roundtrip_s": 0.94
}
```

如果 `--rate 1`，8-step chunk 执行约 8 秒，所以 request interval 约 8 到 9 秒是合理的。

如果 `--rate 2`，request interval 应该约 4 到 5 秒。

## 13. 当前 Setup 下的预期时间

在当前 policy server + Franka laptop setup 下，预期关系如下：

```text
模型推理时间 = server_predict_action_s
网络加 serialization overhead = client_policy_roundtrip_s - server_predict_action_s
新 action chunk 间隔 = request_interval_s
单步 action 间隔 = 1 / rate
target_joint 重复 publish 间隔 = 1 / publish_rate
```

例如：

```bash
--rate 1
--publish-rate 20
```

则：

```text
每个 action step: 1.0 s
每个 chunk: 8 steps ≈ 8 s
/target_joint publish: 20 Hz
```

如果 policy inference 约 0.9 s，那么每 8 步会有一次约 0.9 s 的等待。

当前 client 是：

```text
先等新 chunk 返回，再执行这个 chunk
```

还不是 fully asynchronous pipeline。

所以如果未来需要更流畅，可以考虑：

```text
执行当前 chunk 的同时，提前请求下一 chunk
```

这属于后续优化，不是今天必须做的。

## 14. 当前风险点

### 14.1 不要混用 action space

三种 action space 不能随便混：

```text
absolute joint angle
delta joint angle
delta end pose
```

如果 dataset 是 delta joint，但 config 当成 absolute end pose，训练会错。

如果 policy 输出 delta end pose，但 client 当成 delta joint，机器人会乱动。

### 14.2 Quest 3 路线需要新数据/新 wrapper

Quest 3 推荐 delta end pose，但当前 StarVLA real model 是 delta joint。

所以 Quest 3 下一步应该是：

```text
单独设计 delta end pose dataset + controller wrapper
```

不是直接复用当前 delta joint model。

### 14.3 当前 VLM frozen，任务成功率不一定高

当前 deployment pipeline 已经打通。

但是任务成功率还取决于：

- 数据量
- 视觉泛化
- VLM 是否 fine-tune
- cube/bowl/camera layout 是否和训练一致

当前模型可以先作为 deployment prototype，不应该直接视作最终稳定 policy。

## 15. TODO

### TODO 1：确认当前 20 episodes 的 action space

已确认：

```text
dataset/snkdjn: absolute joint target
dataset/snkdjn_delta: delta joint angle
```

### TODO 2：记录 timing log

运行：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --max-delta 0.01 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 32 \
  --log-timing
```

把输出里的这些字段保存：

```text
request_interval_s
server_predict_action_s
client_policy_roundtrip_s
action_period_actual_s
```

### Quest 3 delta end pose implementation

已在 `src/crisp_gym/crisp_gym/scripts/quest3_stream_adapter.py` 实现：

- Quest pose frame 到 Franka base frame：默认使用 `x_axis/y_axis/z_axis`
  signed-axis map，也可用 `quest_to_franka_matrix` 传入 row-major 3x3 transform。
- delta translation scale：`translation_scale`，并用 `max_translation_step`
  限制单步平移。
- delta rotation representation：内部用 rotvec 积分，输出 action 默认
  `rotation_representation:=euler_xyz`，也支持 `rotvec`。
- clutch/reset button：`deadman_button` hold-to-move，`reset_button`
  re-anchor without target jump，`pause_button` 暂停到松开 deadman 后恢复。
- gripper mapping：`trigger_axis` + `trigger_threshold` 映射到
  `gripper_open_value/gripper_closed_value`。
- 默认仍通过 CRISP streamed env step：`/phone_pose` + `/phone_gripper`。
- 同时发布 7D delta end-pose action：
  `/quest/right_controller/delta_end_pose = [dx, dy, dz, droll, dpitch, dyaw, gripper]`。
- 如需直接测试 `cartesian_impedance_controller`，设置
  `publish_target_pose:=true` 后会从 `/current_pose` 初始化并发布 `/target_pose`；
  此模式需要外部先启动 `cartesian_impedance_controller`，且不要同时运行其他
  `target_pose` publisher。

### TODO 4：决定下一轮训练 action space

短期：

```text
继续 delta joint angle，因为已经部署打通。
```

Quest 3 采集后：

```text
重新训练 delta end pose model。
```

## 16. 最终一句话结论

当前 StarVLA-Franka 真实部署是：

```text
delta joint angle policy
```

当前 Franka controller 接收的是：

```text
absolute joint target on /target_joint
```

Quest 3 未来建议做：

```text
delta end pose teleop / dataset / policy
```

运行时间分析可以直接用当前 client 的：

```bash
--log-timing
```

其中：

```text
server_predict_action_s = 模型推理时间
client_policy_roundtrip_s = Franka laptop 端 request-response 时间
request_interval_s = 收到新 action chunk 的间隔
```
