# 今日总结 - 2026-07-06

## 1. 今天的总体目标

今天的核心目标是把 StarVLA 从“训练和离线测试”推进到“真实 Franka 部署”。

也就是说，我们不只是检查模型能不能输出 action，而是要真正完成下面这条链路：

```text
真实相机图像 + 任务文字
-> StarVLA policy server
-> 输出 8D action chunk
-> Franka workstation 客户端
-> /target_joint
-> joint_impedance_controller
-> 真实 Franka 机械臂运动
```

今天最终确认了一件很重要的事情：

```text
StarVLA 已经可以通过客户端让真实 Franka 动起来。
```

目前真正剩下的问题已经不是“底层控制完全不通”，而是“模型策略是否足够好，能不能稳定完成任务”。

今天统一后的真实任务名称是：

```text
pick up the cube and place it on the bowl
```

之前很多地方还叫：

```text
pick up the cube
```

这个名字不完整，因为你的真实 episode 是“拿起 cube 并放到 bowl 上/里”。今天已经把本地脚本、文档、数据 metadata 里的任务名尽量改成完整版本。

## 2. StarVLA Policy Server 当前状态

StarVLA policy server 在服务器端运行，地址是：

```text
192.168.1.113:10093
```

当前部署用的 checkpoint 是：

```bash
/data/hanyu/starVLA_checkpoints/crisp_franka_delta_from_20k_gpu3_freeze_vlm_actionmode_abs_clean/final_model/pytorch_model.pt
```

客户端成功连接后，server 返回的 metadata 是：

```text
env: starvla_policy_server
action_chunk_size: 8
available_unnorm_keys: ['franka']
default_unnorm_key: franka
action_keys: ['action.delta_joints', 'action.gripper']
state_keys: ['state.joints', 'state.gripper']
```

这个信息说明：

- server 不是空的，确实加载了模型
- action chunk 长度是 8
- 使用的是 Franka 的 unnormalization key
- action 是 `delta_joints + gripper`
- state 是 `joints + gripper`

因此，今天 StarVLA server 端不是主要问题。它可以正常接收 observation，也可以返回 action chunk。

## 3. Franka 部署客户端

今天主要使用的客户端脚本是：

```bash
scripts/starvla_franka_delta_joint_client.py
```

这个脚本也被复制到了 Docker 容器内：

```bash
/home/ros/ros2_ws/scripts/starvla_franka_delta_joint_client.py
```

它的功能是：

- 读取 Franka 当前 `/joint_states`
- 读取相机图像
- 连接 StarVLA WebSocket policy server
- 发送 observation 和 task instruction
- 接收 8D action chunk
- 把前 7 维作为 joint delta
- 第 8 维作为 gripper command
- 用 `--max-delta` 限制每一步最大动作
- 如果没有 `--execute`，只打印动作，不控制机器人
- 如果有 `--execute`，发布到 `/target_joint`
- 每一步会持续重复发布目标，而不是只发一次
- 执行前会检查 `/target_joint` 有没有 subscriber

今天还改了客户端，使它不再必须依赖完整的 StarVLA repo。之前在 Docker 里报错：

```text
RuntimeError: Could not find StarVLA repo
```

原因是 Docker 里没有完整的 `third_party/starVLA`。后来在脚本里加入了 fallback WebSocket/msgpack client，所以只要 Docker 里有 `msgpack` 和 `websockets` 就可以连接 policy server。

Docker 里需要安装：

```bash
python -m pip install msgpack websockets
```

这个问题解决后，客户端可以正常连接 server。

## 4. Dry Run 测试结果

一开始我们先做 dry-run，也就是不真正控制 Franka，只看模型输出。

命令类似：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --max-delta 0.02 \
  --rate 2 \
  --max-steps 4
```

输出显示：

```text
action_chunk_shape: (8, 8)
joint_delta_abs_max: about 0.08 to 0.13
execute=False
```

这说明：

- StarVLA 可以输出 8 步 action chunk
- 每个 action 是 8 维
- 输出值不是 NaN
- gripper 也有输出
- dry-run 时不会控制真实 Franka

这一阶段证明了：

```text
模型推理链路是通的。
```

## 5. 第一次 Execute 测试的问题

后来我们加上 `--execute`，用很小的动作测试真实机器人：

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

客户端显示：

```text
EXECUTE mode enabled
step=000 ...
execute=True
```

但是 Franka 没有明显运动。

一开始看起来像是 StarVLA 输出不对，或者 client 没有 publish 成功。但后面检查发现，真正的问题是 controller 没有正确执行 `/target_joint`。

## 6. Controller Debugging 过程

我们检查了 `/target_joint`：

```bash
ros2 topic info -v /target_joint
```

有时候它显示：

```text
Unknown topic '/target_joint'
```

有时候又显示有 subscriber：

```text
Type: sensor_msgs/msg/JointState
Subscription count: 3

joint_impedance_controller
cartesian_impedance_controller
gravity_compensation
```

这说明 `/target_joint` 这个 topic 本身在 controller 启动后是存在的。

但是仅仅有 subscriber 不代表机器人一定会执行，因为关键还要看当前 active controller 是哪个。

我们尝试：

```bash
ros2 control list_controllers
```

但经常报：

```text
Failed getting a result from calling /controller_manager/list_controllers in 10.0
```

这说明 controller manager service 很不稳定，不能稳定用 `ros2 control switch_controllers` 来切换 controller。

## 7. 手动 `/target_joint` 测试

为了排除 StarVLA 的影响，我们手动发布一个很小的 joint target：

```bash
ros2 topic pub -r 20 /target_joint sensor_msgs/msg/JointState "{
  name: ['fr3_joint1','fr3_joint2','fr3_joint3','fr3_joint4','fr3_joint5','fr3_joint6','fr3_joint7'],
  position: [0.006, -0.7828, 0.0071, -2.3682, -0.0071, 1.5610, 0.7680],
  velocity: [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
}"
```

终端显示消息在持续 publishing，但是 `/joint_states` 里实际 joint1 基本没有变化。

这证明：

```text
StarVLA 不是当时机器人不动的原因。
```

真正原因是：

```text
/target_joint 的消息没有被 active controller 执行。
```

## 8. 找到 Root Cause

检查默认 Franka launch：

```bash
src/crisp_controllers_demos/crisp_controllers_robot_demos/launch/franka.launch.py
```

发现默认启动状态通常是：

```text
joint_impedance_controller      inactive
joint_trajectory_controller     active
cartesian_impedance_controller  inactive
gravity_compensation            inactive
```

但是当前 StarVLA client 发布的是：

```text
/target_joint
```

它需要：

```text
joint_impedance_controller active
```

因此之前机器人不动的根本原因是：

```text
默认 active 的 controller 不是 joint_impedance_controller。
```

## 9. Controller 解决方案

因为 `ros2 control switch_controllers` 经常因为 controller manager service 超时而失败，所以我们没有继续依赖手动切换 controller。

今天创建了一个新的 launch 文件：

```bash
src/crisp_controllers_demos/crisp_controllers_robot_demos/launch/franka_joint_impedance.launch.py
```

它的目的很简单：

```text
启动 Franka 时直接让 joint_impedance_controller active。
```

同时让：

```text
joint_trajectory_controller inactive
```

这样不需要启动后再切 controller。

Docker 内对应路径是：

```bash
/home/ros/ros2_ws/src/crisp_controllers_demos/crisp_controllers_robot_demos/launch/franka_joint_impedance.launch.py
```

并且在 Docker install/share 路径下做了 symlink，让 `ros2 launch` 可以找到它。

启动命令是：

```bash
ros2 launch crisp_controllers_robot_demos franka_joint_impedance.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

这个 launch 是今天 controller 问题的关键修复。

## 10. Controller 修复后的验证

修复后再次手动 publish `/target_joint`。

这次 `/joint_states` 变化了：

```text
target joint1: about 0.026
actual joint1: about 0.0248
```

这说明：

```text
/target_joint -> joint_impedance_controller -> Franka
```

这条底层控制链路已经打通。

之后再运行 StarVLA client，Franka 也可以动了。

因此今天可以认为：

```text
controller problem was solved for the current joint-control deployment path.
```

更准确地说：

```text
只要使用 franka_joint_impedance.launch.py 启动，/target_joint 可以控制 Franka。
```

## 11. StarVLA 真实执行结果

Controller 修好以后，执行命令类似：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --task "pick up the cube and place it on the bowl" \
  --execute \
  --max-delta 0.005 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 4
```

输出示例：

```text
step=000 delta=[ 0.005  0.005  0.005 -0.005  0.    -0.005  0.004]
target=[...]
gripper=1.0
execute=True
```

后来你确认：

```text
now franka could move
```

这就是今天最大的进展。

## 12. 当前安全执行建议

现在可以继续尝试 StarVLA real rollout，但要从小动作开始。

保守测试：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --execute \
  --max-delta 0.01 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 16
```

稍微大一点：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --execute \
  --max-delta 0.015 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 24
```

不建议一下子把 `--max-delta` 调得太大。

原因是：

- 如果模型方向是对的，稍微增大可以让动作明显
- 如果模型方向不稳定，增大会放大错误
- 当前模型还没有证明能完整完成任务

因此推荐顺序是：

```text
max_delta 0.005 -> 0.01 -> 0.015
```

每一步都先观察运动是否平滑、是否朝 cube/bowl 方向移动。

## 13. 当前模型质量判断

当前模型已经能输出动作，并能驱动 Franka 运动。

但是它还不能被认为已经可靠完成任务。

主要原因是：

1. 真实数据只有约 20 episodes
2. 当前 real-world fine-tune 时 VLM 是 frozen 的
3. 你之前的经验是：冻结 VLM 时 success rate 低，而加入 VLM co-training 后 success rate 接近 100%
4. 当前模型看到真实相机、真实 cube/bowl、真实 Franka 后，泛化能力还需要验证
5. 输出 action 有时会出现来回抵消的趋势

所以当前模型的定位应该是：

```text
deployment pipeline prototype
```

而不是最终可稳定执行任务的 policy。

## 14. 关于 Frozen VLM 的影响

你之前提到：

```text
previous StarVLA training: freeze VLM -> low SR
VLM co-training -> near 100% SR
```

这对当前 real Franka fine-tuning 很重要。

当前因为 GPU memory 问题，我们先用了 frozen VLM，这样训练更容易跑通，但视觉理解能力可能不足。

这意味着：

- 它可能学到一些动作分布
- 但不一定能正确根据图像定位 cube 和 bowl
- 对相机角度、物体位置、光照变化会更敏感
- 如果 cube/bowl 位置稍微变动，policy 可能不稳

后续如果想提高成功率，建议尝试：

- LoRA fine-tuning
- 只 unfreeze vision projector / VLM interface
- 减少 vision tokens
- batch size 1
- gradient accumulation
- 使用空闲 H100 GPU
- 避免占用被 root process 占住的 GPU

## 15. 训练数据和 Action Format

当前真实数据来自你采集的 Franka teleoperation episodes。

数据曾经经历过格式调整：

- 原始数据是 absolute joint action
- 后来转换成 delta-joint action
- 但为了适配 StarVLA dataloader，metadata 中保留 `action_mode: abs`
- 这样 dataloader 不会再做第二次 delta

这点非常重要。

如果数据已经是 delta joint，但 dataloader 又把它当 absolute 再做差分，就会得到错误 action，模型输出会很乱。

当前正确理解是：

```text
parquet 里的 action 已经是 delta joint
metadata / config 不能让 dataloader 再 double-delta
```

之前看到的正常 delta 范围大概是：

```text
joint delta mostly within about 0.01 to 0.13 rad
```

这比早期错误统计里出现的：

```text
2.0 rad, 3.0 rad
```

要合理很多。

## 16. 任务命名清理

今天把任务名从：

```text
pick up the cube
```

改成：

```text
pick up the cube and place it on the bowl
```

本地改过的主要文件包括：

```text
scripts/starvla_franka_delta_joint_client.py
scripts/starvla_delta_joint_policy_smoke_test.py
DATA_COLLECTION.md
franka.md
CURRENT_RECORDING_TELEOP_SETTINGS.md
QUEST3_FRANKA_TELEOP.md
STARVLA_FRANKA_DEPLOY_CURRENT_STATUS.md
STARVLA_FRANKA_FINE_TUNE_DEPLOY.md
STARVLA_FRANKA_NEXT_IMPLEMENTATION.md
dataset/snkdjn/... metadata
dataset/snkdjn_delta/... metadata
```

StarVLA 相关 data mix / run id 也改成了更完整的名字，例如：

```text
crisp_franka_pick_cube_place_bowl_20eps
crisp_franka_pick_cube_place_bowl_debug
crisp_franka_pick_cube_place_bowl_delta_20eps
crisp_franka_pick_cube_place_bowl_delta_debug
```

注意：

```text
已经训练好的 checkpoint 名字和内部 config 可能仍然带旧名字。
```

如果要让 checkpoint/config 也完全一致，需要之后用新命名重新训练或至少重新保存 config。

## 17. End Pose Control 代码检查

今天还检查了 `franka_ws` 里是否有 end pose control。

结论是有。

当前 StarVLA 部署用的是 joint control：

```text
/target_joint
joint_impedance_controller
sensor_msgs/msg/JointState
```

而 end pose / Cartesian control 相关的是：

```text
/target_pose
cartesian_impedance_controller
geometry_msgs/msg/PoseStamped
```

相关文件包括：

```text
src/crisp_controllers/src/cartesian_controller.cpp
src/crisp_py/crisp_py/robot/robot.py
src/crisp_py/examples/20_infinite_figure.py
src/crisp_py/examples/09_get_twist.py
src/crisp_py/examples/01_track.py
src/crisp_controllers_demos/crisp_controllers_robot_demos/crisp_controllers_robot_demos/target_publisher.py
```

但是今天没有切换到 end pose control。

原因是：

- 当前 StarVLA fine-tune 使用的是 8D joint data
- 当前 model 输出的是 delta joints
- 直接换成 end pose control 需要重新定义 action space
- 可能还需要重新训练或写 joint-to-pose wrapper

所以当前最稳的路线还是：

```text
继续使用 joint-control deployment。
```

## 18. 今天遇到的主要问题

### 问题 1：Docker 里找不到 StarVLA repo

报错：

```text
RuntimeError: Could not find StarVLA repo
```

解决：

- 在 client 中加入 fallback WebSocket client
- 不再强制依赖完整 StarVLA repo

### 问题 2：缺少 msgpack

报错：

```text
ModuleNotFoundError: No module named 'msgpack'
```

解决：

```bash
python -m pip install msgpack websockets
```

### 问题 3：Franka 不动

现象：

- client 显示 `execute=True`
- `/target_joint` 有 publishing
- Franka 没有明显动作

原因：

```text
joint_impedance_controller 没有 active。
```

解决：

```text
新建 franka_joint_impedance.launch.py，让 joint_impedance_controller 启动时就是 active。
```

### 问题 4：`ros2 control list_controllers` 经常超时

现象：

```text
Failed getting a result from calling /controller_manager/list_controllers in 10.0
```

处理方式：

- 不再依赖运行中 switch controller
- 改成 launch 阶段直接激活正确 controller

### 问题 5：动作太小看不见

一开始 `max_delta=0.005`，动作很安全但不明显。

后续可以逐渐调到：

```text
0.01
0.015
```

但不要直接大幅增加。

### 问题 6：模型策略还不一定能完成任务

底层控制已经打通，但模型是否能抓 cube、放 bowl，还需要继续测试和训练。

主要原因：

- real episodes 少
- VLM frozen
- 真实视觉环境和 LIBERO 不同
- object pose 变化可能影响很大

## 19. 当前推荐运行流程

### Terminal 1：进入 Docker，启动 Franka controller

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=30

ros2 launch crisp_controllers_robot_demos franka_joint_impedance.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

### Terminal 2：启动相机

如果相机已经在跑，就不用重复启动。

检查：

```bash
ros2 topic hz /right/right_third_person_camera/color/image_raw/compressed
```

正常应该接近：

```text
30 Hz
```

### Terminal 3：确认 `/target_joint`

```bash
docker exec -it franka bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=30

ros2 topic info -v /target_joint
```

应该能看到：

```text
joint_impedance_controller
```

### Terminal 4：运行 StarVLA client

先 dry-run：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --max-delta 0.01 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 8
```

确认输出合理后，再 execute：

```bash
python scripts/starvla_franka_delta_joint_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10093 \
  --execute \
  --max-delta 0.01 \
  --rate 1 \
  --publish-rate 20 \
  --max-steps 16
```

如果动作很小但方向看起来正确，再尝试：

```bash
--max-delta 0.015
--max-steps 24
```

## 20. 下一步计划

### 短期

1. 用当前模型继续小步测试 real Franka。
2. 观察 Franka 是否朝 cube 移动。
3. 记录每次输出的 delta 和实际运动。
4. 不要一次把动作调太大。
5. 保持 cube/bowl/camera/initial pose 和训练数据一致。

### 中期

1. 采集更多真实数据，建议至少 50 到 100 episodes。
2. 先保持环境固定，让模型学会稳定完成同一个任务。
3. 然后再逐渐加入位置变化。
4. 用完整任务名重新训练：

```text
pick up the cube and place it on the bowl
```

5. 检查 dataset statistics，确保 action delta 范围合理。
6. 确保不要 double-delta。

### 训练改进

下一轮训练建议不要完全依赖 frozen VLM。

可以尝试：

- LoRA
- partial VLM fine-tuning
- unfreeze vision projector
- reduce vision tokens
- batch size 1
- gradient accumulation
- 使用更空闲的 GPU

目标是让模型不仅学到 joint action pattern，也能更好理解真实图像里的 cube 和 bowl。

## 21. 今天最重要的结论

今天最重要的结论是：

```text
Franka 不动的主要原因不是 StarVLA，而是 controller 没有正确 active。
```

修复方式是：

```text
使用 franka_joint_impedance.launch.py，让 joint_impedance_controller 启动时直接 active。
```

修复后：

```text
/target_joint 可以让真实 Franka 运动。
StarVLA client 可以让真实 Franka 运动。
```

所以现在项目状态已经从：

```text
模型能推理，但机器人不动
```

推进到：

```text
模型能推理，机器人能动，下一步优化任务成功率
```

## 22. 当前一句话总结

今天完成了 StarVLA 到真实 Franka 的第一条可执行闭环：

```text
StarVLA policy server -> Franka client -> /target_joint -> joint_impedance_controller -> real Franka movement
```

下一阶段的重点是：

```text
让模型可靠完成 pick up the cube and place it on the bowl。
```
