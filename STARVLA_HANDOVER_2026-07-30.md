# StarVLA 项目完整交接文档

> 最后整理时间：2026-07-30（Asia/Hong_Kong）  
> 工作站：`/home/dase-hw101/franka_ws`  
> 目标读者：对前序对话一无所知、需要立即接手训练、部署和后续研究模块开发的新 Codex Agent

---

## 0. 一页结论

当前项目已经完成了以下主线：

1. 用 Quest 3 遥操作真实 Franka FR3，采集双 RGB 相机的 pick-and-place 示范。
2. 从独立单 episode 数据中人工筛选并合并出一个 **74 episodes、17,328 frames、双相机** 的 LeRobot v2.1 数据集。
3. 已经训练并真机测试过一个“LIBERO 30k → 旧 50 episodes”的模型。它能靠近方块，偶尔能抓住和抬升，但抓取时机、视觉定位和释放决策都不稳定。
4. 现在正在/刚刚完成导师要求的新基线：**把筛选后的 74 episodes 全部混合，从 Qwen3-VL-4B-Instruct 初始化，冻结 vision tower，其余模块训练 20k steps；不加载 LIBERO checkpoint，也不加载旧 50-episode 模型。**
5. 计划做一个公平的 LIBERO 初始化对照实验，但第一次在物理 GPU 2（80 GB）运行时 OOM；需要等待物理 GPU 0（96 GB）空闲后串行训练。
6. 真机客户端已经包含通用安全限制和真实夹爪宽度反馈，但已经移除了基于固定 XYZ 抓取区、自动抬升等 object-specific “作弊”逻辑。评估必须区分：
   - **纯 VLA 成功**：模型自主决定接近、闭合、抬升、搬运、释放；
   - **安全过滤成功**：安全层曾压制/改写模型动作，不能当作纯 VLA 成功。
7. 当前最大模型问题不是“机器人完全不会动”，而是：
   - 方块位置变化时，末端仍趋向相似的平均抓取位置；
   - 闭合时机经常过早或过晚；
   - 抓住后模型有时不抬升，或过早请求张开；
   - 数据中的物体位置分布较窄，视觉 grounding 不够强。
8. 后续研究最推荐先做 **Spatial Forcing 风格的 3D 表征蒸馏/对齐模块**，因为它正面针对空间 grounding，同时理想情况下无需在真机推理阶段加入深度模型。RLinf/HG-DAgger 可以作为第二条路线，但目前并不是把一个配置文件改掉就能直接接入现有 QwenGR00T + ROS 2 + Quest 3 系统。

最重要的执行原则：

- **先确认 74-episode Qwen 基线是否训练完成并做严格基线评估，再加模块。**
- **一次实验只改变一个变量。不要同时加入 3D 模块和 RL。**
- **不要直接覆盖 `QwenGR00T.py`；注册一个新的 framework/model variant。**
- **不要执行 `git reset --hard`、`git checkout -- .` 或盲目 pull/rebase。当前 StarVLA 工作树含有本项目的重要未提交修改。**
- **任何真机执行先 dry-run，再低速、短 horizon 实机验证，并确保急停可用。**

---

## 1. 机器、网络与进程拓扑

| 角色 | 地址/路径 | 说明 |
|---|---|---|
| Franka 控制/采集笔记本 | `dase-hw101@192.168.1.117` | 当前文档所在机器；Ubuntu、ROS 2 Humble、相机容器、控制器和部署客户端 |
| GPU 训练服务器 | `hanyu@192.168.1.113` | hostname `server1cps`；StarVLA 训练和 policy server |
| Franka 机器人 | `172.16.0.2` | 控制器/夹爪节点连接地址 |
| Quest 3 | `192.168.1.149` | ADB serial `2G97C5ZHB603FS`；无线 ADB 地址通常为 `192.168.1.149:5555` |
| ROS 2 Domain | `30` | `ROS_DOMAIN_ID=30` |
| ROS middleware | CycloneDDS | `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |

训练服务器时间显示通常是 UTC；香港时间约为服务器显示时间 `+8 h`。日志时间判断必须先确认 `date` 和 `timedatectl`，不要因时区误判训练是否停止。

### 1.1 GPU 服务器重启后的硬件状态

此前服务器出现 NVIDIA 内核模块和用户态库版本不一致：

- 已加载内核模块：`595.71.05`
- 磁盘 DKMS 模块和用户态 NVML：`595.84`
- 症状：`Failed to initialize NVML: Driver/library version mismatch`

用户获得授权后重启了服务器。重启后已验证：

```text
loaded NVRM: 595.84
modinfo nvidia: 595.84
nvidia-smi driver: 595.84
```

四张 GPU 均可见：

| 物理 GPU | 型号 | 显存 | 当前项目用途 |
|---|---|---:|---|
| 0 | NVIDIA H100 | 97,871 MiB | 74-episode Qwen baseline 实际运行位置；适合当前约 86 GB 峰值训练 |
| 1 | NVIDIA H100 PCIe | 81,559 MiB | 当前完整训练配置大概率不够 |
| 2 | NVIDIA H100 PCIe | 81,559 MiB | LIBERO 对照实验已 OOM，不应按原配置重试 |
| 3 | NVIDIA H100 | 97,871 MiB | 重启后曾有 `liji` 的进程；未经对方允许不要占用 |

注意：脚本里曾同时设置：

```bash
GPU_IDS=1
ACCELERATE_GPU_IDS=0
```

实际进程落在**物理 GPU 0**，而不是脚本输出所写的 physical GPU 1。原因是 `CUDA_VISIBLE_DEVICES` 后的逻辑编号与 Accelerate 的 `gpu_ids` 二次映射。新实验必须同时用 `nvidia-smi` 和 `/proc/<pid>/environ` 验证实际物理卡，不能相信 launcher 的 echo。

---

## 2. 本地代码仓库与重要文件

### 2.1 路径与版本

- 项目根目录：`/home/dase-hw101/franka_ws`
- StarVLA 子仓库：`/home/dase-hw101/franka_ws/third_party/starVLA`
- 当前 branch：`starVLA_dev`
- 当前基准 commit：`2e5f239bc0b1661d7d556bdba5071f3041544cc6`
- 服务器 StarVLA：`/home/hanyu/starVLA`

官方 StarVLA README 明确提示开发分支可能不稳定；本项目又包含自定义数据、训练和 policy-server 修改。因此开始新模块前，建议先创建可追踪的 baseline commit/tag 或独立 worktree，而不是在现有文件上继续堆叠修改。

### 2.2 当前工作树不是干净的

截至交接时：

```text
 M deployment/model_server/policy_norm_processor.py
 M deployment/model_server/policy_wrapper.py
 M deployment/model_server/tools/websocket_policy_server.py
 M examples/realRobots/Franka/train_files/data_registry/data_config.py
 M starVLA/dataloader/gr00t_lerobot/registry.py
 M starVLA/dataloader/lerobot_datasets.py
?? examples/realRobots/Franka/train_files/run_crisp_franka_train_abs_joints.sh
?? examples/realRobots/Franka/train_files/run_crisp_franka_train_delta_joints.sh
?? examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh
?? examples/realRobots/Franka/train_files/starvla_cotrain_crisp_franka_abs_joints.yaml
?? examples/realRobots/Franka/train_files/starvla_cotrain_crisp_franka_delta_joints.yaml
?? examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef.yaml
```

这些不是可以随意丢弃的临时文件。新 Agent 第一件事应该是：

```bash
cd /home/dase-hw101/franka_ws/third_party/starVLA
git status --short
git diff --stat
git diff -- examples/realRobots/Franka/train_files/data_registry/data_config.py
git diff -- starVLA/dataloader/gr00t_lerobot/registry.py
```

在理解差异后，把当前可工作的 baseline 保存到一个明确的分支/commit。不要自行覆盖用户更改。

### 2.3 关键代码入口

| 文件 | 用途 |
|---|---|
| `third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00T.py` | 当前主模型：Qwen3-VL + GR00T/DiT action head；未来 3D 模块的关键参照入口 |
| `third_party/starVLA/starVLA/model/framework/VLM4A/ABot_M0.py` | 已有 Qwen + VGGT + CrossAttention 的实验草图，但目前不完整，不能直接训练 |
| `third_party/starVLA/starVLA/model/tools.py` | 已有 `CrossAttention(image_feature, spatial_feature)` 工具，可参考复用 |
| `third_party/starVLA/examples/realRobots/Franka/train_files/run_quest3_franka_train_delta_eef.sh` | 74-episode 训练主入口 |
| `third_party/starVLA/examples/realRobots/Franka/train_files/starvla_cotrain_quest3_franka_delta_eef.yaml` | 当前任务训练配置 |
| `scripts/start_starvla_dualcam_74eps_from_qwen_vision_frozen_train.sh` | 导师要求的 Qwen-base、冻结 vision、20k launcher |
| `scripts/start_starvla_dualcam_74eps_from_libero30k_train.sh` | 旧版 LIBERO launcher；**不是公平 A/B 配置** |
| `scripts/build_quest3_franka_dualcam_74eps.py` | 74 episodes 数据集构建脚本 |
| `scripts/starvla_franka_delta_pose_client.py` | 当前真机 VLA 部署客户端及安全过滤逻辑 |
| `scripts/franka_return_to_standard.py` | Franka 返回标准初始位姿 |
| `src/camera_driver/docker-compose.dual-franka.yaml` | 双 RealSense 相机容器配置；此前已经为帧率问题修改过 |

### 2.4 历史说明文档

交接 Agent 应同时阅读：

- `/home/dase-hw101/franka_ws/7.23_7.24_7.27_summary.md`
- `/home/dase-hw101/franka_ws/728_summary.md`

前者记录采集、夹爪状态和早期部署；后者记录 7 月 28 日的数据合并、服务器训练准备和部署诊断。本文件优先级最高；若现场状态与文档冲突，以只读命令重新检查的实时结果为准。

---

## 3. 74-episode 数据集

### 3.1 唯一推荐训练集

本地：

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/quest3_franka_dualcam_pickplace_74eps
```

服务器：

```text
/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_pickplace_74eps
```

Hugging Face namespace 曾使用 `snkdjn`。如果要重新 push，先检查本地 HF 登录和 repo 权限；不要因为本地目录名存在就假定远端已经完整同步。

**不要改用名称包含 `100eps` 的混合集。** 那些目录可能包含未审查或明确有问题的 episode。当前实验的正式数据集只认 `quest3_franka_dualcam_pickplace_74eps`。

### 3.2 数据统计

来自 `meta/info.json`：

```text
episodes: 74
frames: 17,328
videos: 148 = 74 × 2 cameras
top-level fps: 15
image shape: 256 × 256 × 3
robot_type: franka
codebase_version: v2.1
task: pick up the cube and place it on the box
```

两个图像流：

- `observation.images.primary`
- `observation.images.wrist`

元数据里有一个需要新 Agent 留意的历史不一致：每个视频 feature 的 `video_info.video.fps` 显示 `30.0`，但顶层 `fps` 和内部 `info.video.fps` 是 `15`。当前训练按本项目的数据配置使用 15 Hz；如果重写 loader、重新编码视频或做时序模块，应先验证真实 frame timestamp 和 loader 采样，不要盲信某一个嵌套字段。

### 3.3 74 episodes 的组成

- 原始人工筛选基线：50 episodes
- 改善腕部相机位置后新增、再次筛选：24 episodes
- 合计：74 episodes

完整 source ID：

```text
0047 0048 0049 0050 0051 0052 0058 0060 0063 0064 0065 0069
0077 0078 0079 0080 0081 0082 0083 0084 0085 0086 0087 0088
0089 0090 0091 0092 0094 0095 0096 0097 0098 0099 0100 0101
0102 0103 0104 0105 0106 0107 0109 0110 0111 0112 0113 0114
0115 0116 0121 0124 0126 0127 0128 0131 0132 0134 0135 0136
0137 0138 0139 0140 0142 0143 0144 0145 0146 0147 0148 0149
0150 0151
```

明确排除：

| ID | 原因 |
|---|---|
| `0036` | 原始 50 episodes 中已经排除；夹爪 state 有问题 |
| `0125` | 录制不完整 |
| `0129` | wrist video 冻结 |
| `0130` | 录制不完整 |
| `0133` | 录制不完整 |
| `0141` | video/parquet frame count 不匹配 |
| `0152` | 录制不完整 |

权威 manifest：

```text
dataset/snkdjn/quest3_franka_dualcam_pickplace_74eps/meta/merge_manifest.json
```

### 3.4 Observation/action 语义

LeRobot 表中保留了完整 20-D observation state：

```text
[x, y, z, roll, pitch, yaw,
 gripper,
 joint_0 ... joint_6,
 target_x, target_y, target_z, target_roll, target_pitch, target_yaw]
```

StarVLA 任务实际映射：

| StarVLA modality | 数据来源 | 维数 |
|---|---|---:|
| `state.eef_position` | `observation.state.cartesian[0:3]` | 3 |
| `state.eef_rotation` | Cartesian RPY `[3:6]` | 3 |
| `state.gripper` | `observation.state.gripper` / combined state index 6 | 1 |
| `action.delta_eef_position` | `action[0:3]` | 3 |
| `action.delta_eef_rotation` | RPY delta `action[3:6]` | 3 |
| `action.gripper` | absolute gripper command `action[6]` | 1 |

动作总维度为 7，action chunk/horizon 为 8，因此 policy 通常输出 `[8, 7]`。

非常容易踩坑的一点：**数据表的 action 已经是 delta pose。** 当前 StarVLA loader 配置中 `action_mode: abs` 是有意的，意思是 loader 不再对已经是 delta 的 action 做一次差分；它不代表机器人执行的是 absolute Cartesian pose。若把 loader 改成 delta，会二次求差，动作量会错误。

夹爪约定：

- Policy action：`1 = open`，`0 = close`
- Observation state：基于真实总指间宽度归一化，张开约为 `0`，闭合程度越高越接近 `1`
- 真实总指间宽度：`finger_joint1 + finger_joint2`
- 空夹闭合接近 `0 m`
- 抓住当前方块时，历史成功宽度约 `0.028–0.030 m`

不要把 action gripper 和 state gripper 的开闭方向混淆。

### 3.5 数据采集链路与已经修复的问题

完整采集数据流是：

```text
Quest 3 right controller
  → Piper MR/OpenXR app + ADB logcat
  → quest_reader_ros_bridge.py
  → /quest/right_controller/pose + /quest/right_controller/joy
  → quest3_stream_adapter.py
  → /phone_pose + /phone_gripper + /record_transition
  → LeRobot recorder / ManipulatorCartesianEnv
  → /target_pose + Franka gripper command
  → Cartesian controller + dual RealSense observations
  → 独立单-episode LeRobot dataset
  → 保存后 push to Hugging Face
```

交互式终端通常为：

1. Franka Cartesian controller；
2. Quest MR app/ADB；
3. Quest ROS bridge；
4. Quest stream adapter；
5. LeRobot recorder。

双相机容器在后台运行，不占一个交互式终端。更完整的 teleop 操作和排错说明位于：

- `Quest3_Franka_Tele_guide.md`
- `FRANKA_LEADER_QUEST3_TELEOP_TECHNICAL_DOC.md`

Quest bridge 的无线模板：

```bash
cd /home/ros/ros2_ws/src/piper-vr-teleop
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=/home/ros/ros2_ws/src/piper-vr-teleop:$PYTHONPATH

python3 scripts/quest_reader_ros_bridge.py \
  --transport adb_logcat \
  --connection wireless \
  --quest-ip 192.168.1.149 \
  --rate 30
```

如果无线 ADB 不稳定，USB 模式更可靠：删除 `--connection wireless --quest-ip ...`，或明确写 `--connection usb`。

录制 adapter 的历史最终参数经历过多次调整。当前代码默认值和早期指南不完全一致，重新采集前要以 `quest3_stream_adapter.py`、实际 ROS parameter 和小幅动作测试三者为准，尤其核对：

- Quest-to-Franka axis mapping；
- translation/rotation scale 与单步 clipping；
- deadman/grip button；
- trigger axis 的开闭方向；
- adapter 是否直接控制夹爪；
- recording adapter 本身不应直接发布 `/target_pose`，应由 recorder/environment 发布。

双相机单 episode 录制模板为：

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH=/home/ros/ros2_ws/src/crisp_gym:/home/ros/ros2_ws/src/crisp_py:$PYTHONPATH

python -m crisp_gym.scripts.record_lerobot_format_leader_follower \
  --use-quest3-controller \
  --follower-config dual_cam_franka \
  --streamed-pose-topic /phone_pose \
  --streamed-gripper-topic /phone_gripper \
  --recording-manager-type ros \
  --repo-id snkdjn/<NEW_UNIQUE_SINGLE_EPISODE_ID> \
  --tasks "pick up the cube and place it on the box" \
  --num-episodes 1 \
  --fps 15 \
  --push-to-hub \
  --use-current-pose-as-episode-start \
  --streamed-teleop-timeout 60 \
  --disable-recorder-gripper-control
```

`<NEW_UNIQUE_SINGLE_EPISODE_ID>` 是占位符，不能原样粘贴。应使用一个尚不存在的独立 repo ID；只有明确继续同一数据集时才加 `--resume`。

这里的 `--disable-recorder-gripper-control` 只应该禁止 recorder 与 adapter **重复发送夹爪命令**，不能禁止读取真实夹爪 feedback。此前 `observation.state.gripper` 始终为 0 的根因正是把这两件事混在一起；现在修复为：即使 recorder 不发夹爪命令，environment 仍读取真实状态。相关代码：

- `src/crisp_gym/crisp_gym/envs/manipulator_env.py`
- `src/crisp_gym/crisp_gym/record/record_functions.py`
- `src/crisp_gym/tests/test_streamed_gripper_feedback.py`

录完每条数据必须检查：两路视频都运动、frame count 与 parquet 对齐、完整完成接近/抓取/抬升/搬运/释放、action gripper 和 observation gripper 都正确变化、没有 stale camera 或异常 recovery。不要仅因 recorder 成功 push 就把 episode 纳入训练。

---

## 4. 当前 StarVLA 模型结构

当前使用 `QwenGR00T` 双系统框架：

```text
primary RGB + wrist RGB + language + robot state
                         │
                         ▼
                   Qwen3-VL backbone
                         │ last_hidden [B, L, H]
                         ▼
             GR00T flow-matching DiT action head
                         │
                         ▼
                  action chunk [B, 8, 7]
```

关键事实：

- VLM：Qwen3-VL-4B-Instruct
- Action head：`FlowmatchingActionHead`
- State/action dimension：7
- Action horizon：8
- 当前重复 diffusion steps：训练配置为 8；部署/模型默认 denoise 次数需以 checkpoint config 和 server log 为准
- QwenGR00T 中存在未启用的 DINO 相关注释，但不能据此宣称已有 3D/额外视觉先验
- `forward()` 是训练路径；`predict_action()` 是推理路径。新模块必须保证两条路径使用相同的 preprocessing、token layout 和 feature fusion 语义

最自然的研究插入点是 `QwenGR00T.py` 中 Qwen 的 `last_hidden` 到 action head 之间。建议新增模型类，例如：

```text
QwenGR00TSpatialForcing.py
```

并在 registry/config 中注册。不要直接改坏 baseline 类。

### 4.1 `ABot_M0.py` 的状态

`ABot_M0.py` 看起来尝试：

```text
Qwen visual/language feature + VGGT spatial feature + CrossAttention + action head
```

但它目前只是研究草图：

- `VGGT` import 被注释，constructor 却可能仍调用它；
- action head/维度和当前 QwenGR00T 不完全一致；
- 没有证据表明当前数据配置、训练或部署路径可以运行；
- 没有完整测试或 checkpoint。

它可以帮助理解作者原先设想，但**不能当成可直接启用的现成 3D 模块**。

---

## 5. 已有训练与当前训练状态

### 5.1 已有 LIBERO 与旧 50-episode 模型

原始 LIBERO checkpoint：

```text
/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt
```

旧 50-episode real-Franka 模型：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_50eps_from_libero30k_10k/final_model/pytorch_model.pt
```

重启前曾运行两个 policy server：

- port `10093`：更早的 3k 模型
- port `10094`：旧 50-episode final model

服务器已经重启，因此这些 PID 和端口**不再保证存在**。部署前必须用 `ss`/`pgrep` 重查并手动启动所需 checkpoint。

### 5.2 当前导师要求的主基线

实验定义：

```text
数据：筛选后的 74 episodes 全混合
初始化：仅 Qwen3-VL-4B-Instruct
LIBERO checkpoint：不使用
旧 50-episode checkpoint：不使用
vision tower：冻结
Qwen 其余模块：训练，LR 1e-7
GR00T/DiT action model：随机初始化，LR 1e-4
base LR：1e-6
warmup：1000 steps
total：20,000 steps
save：每 2,000 steps
batch size：1
repeated diffusion steps：8
optimizer：fresh AdamW
```

Qwen base：

```text
/home/hanyu/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

Run：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k
```

Log：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k.launcher.log
```

Launcher PID file：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k.launcher.log.pid
```

冻结路径：

```text
qwen_vl_interface.model.model.visual
```

启动时参数统计：

```text
4599.289 M total
4183.941 M trainable
约 415.348 M frozen（约 9.0%）
```

这与“只冻结 visual tower”相符，而不是冻结整个 Qwen-VL。最后一次明确观察时训练健康推进，约在 step 1549，速度约 2.1–2.6 it/s；本 handover 写入时不能断言它已经完成，接手 Agent 必须现场确认。

### 5.3 立即检查 Qwen 基线是否完成

在笔记本：

```bash
ssh hanyu@192.168.1.113
```

在服务器：

```bash
RUN=/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k
LOG=${RUN}.launcher.log

date
nvidia-smi
test -f "${RUN}/final_model/pytorch_model.pt" \
  && echo "FINAL CHECKPOINT EXISTS" \
  || echo "FINAL CHECKPOINT NOT FOUND"

if test -f "${LOG}.pid"; then
  LAUNCHER_PID=$(cat "${LOG}.pid")
  ps -ww -fp "${LAUNCHER_PID}"
fi

pgrep -af 'train_starvla|accelerate|deepspeed|quest3_franka_dualcam_74eps_from_qwen'
tr '\r' '\n' < "${LOG}" | tail -n 80
find "${RUN}" -maxdepth 3 -type f \
  \( -name 'pytorch_model.pt' -o -name 'config.json' -o -name '*.yaml' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort
```

完成标准不是“进度条看起来到了 100%”，而是至少同时满足：

1. `final_model/pytorch_model.pt` 存在且大小合理；
2. log 无 traceback/OOM/NCCL error；
3. 训练进程正常退出；
4. 保存模型能被 policy server 实际加载并完成一次离线推理。

本地用于安装这次训练入口的 patch 包：

```text
/home/dase-hw101/franka_ws/starvla_dualcam_74eps_qwen_vision_frozen_patch_20260730.tar
SHA256 d805887dd08cab4e61f193feb3f7c4ebe638d1710e39613b576222cfcdea1722
```

### 5.4 LIBERO 初始化公平对照实验

研究问题：同一份 74 episodes、同样训练超参数下，以下初始化谁更好？

```text
A: Qwen base + random action head
B: LIBERO 30k full compatible checkpoint
```

已有旧 launcher `scripts/start_starvla_dualcam_74eps_from_libero30k_train.sh` **不能直接作为公平对照**，因为它使用：

- 10k steps，而不是 20k；
- action LR `1e-5`，而不是 `1e-4`；
- warmup 500，而不是 1000；
- 冻结整个 `qwen_vl_interface`，而不是只冻结 vision tower。

公平 B 实验应复制 Qwen 基线的训练参数，仅改变初始化 checkpoint。建议新 run ID：

```text
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry2
```

失败历史：

1. port `29500` 被占用；后来训练入口已支持 `MAIN_PROCESS_PORT`。
2. 使用自动端口时曾停在 NCCL init；应确认没有残留 rank/process，再用明确空闲端口。
3. 在物理 GPU 2（80 GB）重试时 OOM：PyTorch 已用约 70.68 GiB，又申请 14.99 GiB，峰值需求至少约 85.7 GiB。

因此不要在 GPU 1/2 上按原配置反复重试。为保持公平，优先等待 Qwen 基线释放物理 GPU 0，再在同一张 96 GB 卡上串行跑 B。不要为了塞进 80 GB 而只给 B 开 CPU offload、额外冻结模块或改 batch，否则比较失去意义。

如果端口需要确认：

```bash
ss -ltnp | grep -E ':(29500|29501|29502)\b' || true
pgrep -af 'accelerate|deepspeed|train_starvla'
```

---

## 6. 相机、ROS 2 与 Franka 控制链

### 6.1 双相机

容器名：

```text
franka_dual_realsense
```

相机 serial：

- primary/third person：`344522302659`
- wrist：`351322301561`

ROS topics：

```text
/right/right_third_person_camera/color/image_raw/compressed
/right/right_wrist_camera/color/image_raw/compressed
```

两路相机使用 sensor-data BEST_EFFORT QoS。部署客户端已经改用 `qos_profile_sensor_data`；如果出现 `Timed out waiting for observations: ['primary', 'wrist']`，首先检查容器、topic 名、QoS 和消息年龄，而不是立刻改模型。

笔记本宿主机上启动/重建相机容器：

```bash
cd /home/dase-hw101/franka_ws/src/camera_driver
docker-compose -f docker-compose.dual-franka.yaml up -d --force-recreate
```

本机只有 legacy `docker-compose` 可用；曾出现 `docker compose -f ...` 被旧 docker CLI 解析为 `unknown shorthand flag: 'f'`。

相机诊断：

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/dase-hw101/franka_ws/install/setup.bash 2>/dev/null || true

ros2 topic list | grep -E 'third_person|wrist'
ros2 topic hz /right/right_third_person_camera/color/image_raw/compressed
ros2 topic hz /right/right_wrist_camera/color/image_raw/compressed
```

### 6.2 Franka controller 与状态

关键 topics：

```text
/current_pose
/target_pose
/franka_gripper/joint_states
```

需要 active 的 controllers：

- `cartesian_impedance_controller`
- `pose_broadcaster`
- `joint_state_broadcaster`

在 `franka` Docker 容器内启动当前 Cartesian 控制链的历史命令：

```bash
cd /home/ros/ros2_ws
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch \
  /home/ros/ros2_ws/src/crisp_controllers_demos/crisp_controllers_robot_demos/launch/franka_cartesian_impedance.launch.py \
  arm_id:=fr3 \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  use_rviz:=false
```

这是会连接真实机器人并激活 controller 的命令。若 controller 中途死亡，先保留完整 traceback/controller-manager 状态，再重启；不要在旧进程是否仍占用硬件不明时重复启动多个 controller。

夹爪节点必须正确发布两指 joint states。之前曾因启动参数 `joint_names` 为空导致：

```text
Parameter 'joint_names' needs exactly two arguments, got 0 instead
```

应使用经过安装的正确 YAML，不要手写一个缺少 list 参数的启动命令。任何 `Timed out waiting for observations: ['gripper_state']` 先检查：

```bash
ros2 topic info /franka_gripper/joint_states
ros2 topic echo /franka_gripper/joint_states --once
```

部署前 `/target_pose` 必须没有其他 publisher：

```bash
ros2 topic info /target_pose
```

期望：

- 启动 VLA 客户端前：`Publisher count: 0`
- 启动客户端后：`Publisher count: 1`

若 teleop、录制 adapter、旧部署客户端同时发布，会产生控制竞争。录制/teleop 和自主部署不能同时执行。

### 6.3 标准初始位姿

参考末端位置：

```text
x = 0.308852 m
y = 0.000865 m
z = 0.584837 m
RPY ≈ [-3.11, 0.016, -0.753]
```

脚本：

```text
笔记本本地：/home/dase-hw101/franka_ws/scripts/franka_return_to_standard.py
franka 容器：/home/ros/ros2_ws/scripts/franka_return_to_standard.py
```

注意上下文：如果 shell 已经在 `franka` 容器内部，就没有 `docker` 命令；直接运行 Python。只有在宿主机才运行 `docker exec`。

宿主机：

```bash
docker exec -it franka bash -lc '
  export ROS_DOMAIN_ID=30
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  source /opt/ros/humble/setup.bash
  source /home/ros/ros2_ws/install/setup.bash
  python3 /home/ros/ros2_ws/scripts/franka_return_to_standard.py --execute
'
```

容器内：

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash
python3 /home/ros/ros2_ws/scripts/franka_return_to_standard.py --execute
```

这会真实移动机器人。执行前必须确认路径无障碍、夹爪/末端不会撞桌面或盒子、急停可触达。

---

## 7. 当前真机部署方式

### 7.1 启动 policy server

服务器重启后旧服务已停止。先选一个没有训练或其他用户任务的 GPU，再选空闲端口。示例：

```bash
ssh hanyu@192.168.1.113
cd /home/hanyu/starVLA
source /home/hanyu/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

ss -ltnp | grep -E ':(10093|10094|10095)\b' || true
nvidia-smi
```

启动一个 checkpoint 的模板：

```bash
CKPT=/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/final_model/pytorch_model.pt
PORT=10095
CUDA_VISIBLE_DEVICES=<确认空闲的物理GPU> \
python deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16 \
  --idle_timeout -1
```

不要照抄 `<确认空闲的物理GPU>`。先用 `nvidia-smi` 查实时占用，并确认不会影响其他用户。若需后台运行，使用本项目惯用的 tmux/nohup，并保存 PID、stdout log、checkpoint 路径和 port。

### 7.2 真机客户端

客户端文件：

```text
/home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py
```

建议先做 dry-run：使用与下面相同参数，但**不带 `--execute`**。确认图像、state、policy response、action shape、时序和目标 pose 均合理后，先用短 episode 实机测试。

历史使用过的完整模板：

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash
mkdir -p /home/ros/ros2_ws/deployment_logs

python3 /home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10095 \
  --task "pick up the cube and place it on the box" \
  --primary-image-topic /right/right_third_person_camera/color/image_raw/compressed \
  --wrist-image-topic /right/right_wrist_camera/color/image_raw/compressed \
  --compressed-image \
  --max-steps 400 \
  --execution-horizon 1 \
  --rate 5 \
  --publish-rate 40 \
  --translation-scale 1.25 \
  --max-trans-delta 0.009 \
  --max-rot-delta 0.003 \
  --max-observation-age 1.0 \
  --initial-gripper-state open \
  --gripper-switch-confirmations 3 \
  --gripper-chunk-consensus 0.75 \
  --max-grasp-attempts 1 \
  --log-timing \
  2>&1 | tee "/home/ros/ros2_ws/deployment_logs/dryrun_$(date +%Y%m%d_%H%M%S).log"
```

真机版本是在完成 dry-run 检查后添加 `--execute`。不要仅因 dry-run 没报错就一次运行 400 steps；新模型首次真机应从较短步数开始，操作者手放急停附近。

### 7.3 客户端的夹爪反馈与安全逻辑

客户端订阅真实 `/franka_gripper/joint_states`，计算：

```text
total_width_m = finger_joint_1 + finger_joint_2
state_gripper_closed = clip(1 - total_width_m / 0.08, 0, 1)
```

现有安全逻辑大体包括：

- observation stale 检查；
- workspace、minimum Z 和单步 delta clipping；
- gripper command 连续确认和 chunk consensus；
- 最多抓取次数限制；
- 真实夹爪宽度验证；
- 抓取后的真实抬升验证；
- 防止在未验证抓取时无限反复开合。

已经移除/不应重新加入的 task-specific 逻辑：

- 基于成功示范固定 XYZ 范围的 close gate；
- 一旦宽度合适就由客户端自动执行 `+30 mm` 抬升；
- 把模型的早期 open 长时间强制改成 closed 并把结果记成纯 VLA 成功。

原因：这些逻辑会让客户端自己完成任务阶段，掩盖模型是否真正学会空间定位和抬升。

通用物理安全过滤仍可保留，但必须记录过滤是否触发。出现下列情况时，结论必须写清楚：

- 模型自主抓住、抬升、释放且安全层未改写任务语义：纯 VLA 成功；
- 安全层曾压制早期 open，最后物体仍被放下：安全过滤部署成功，但不是严格纯 VLA 成功；
- 模型请求 open、未达到真实 lift，客户端 abort：纯 VLA 失败，即使夹爪机械延迟后看起来碰巧夹住。

---

## 8. 已知真机表现与模型诊断

### 8.1 已观察到的能力

- 模型能从标准起点向方块方向移动；`/current_pose` 证明末端确实有持续变化。
- 调整客户端 action timing 后轨迹明显更平滑。
- 在部分摆放中，模型能将夹爪移动到方块附近并闭合。
- 至少一次安全过滤部署中，真实宽度约 `28.25 mm`、抬升约 `32.92 mm`，随后释放成功。

### 8.2 不能过度解读的结果

上述成功 episode 中曾有两次过早 open action 被客户端压制。因此那一次应记为：

```text
safety-filtered success
```

而不是：

```text
strict pure-VLA success
```

另一些 episode 中，夹爪位置肉眼看起来很好，但模型没有持续 close/lift，或在验证前请求 open；这说明抓取控制时序仍未学稳。

### 8.3 最重要的失败模式

1. **相似终点/平均轨迹问题**  
   方块向前、向后或轻微侧移时，夹爪常去到相似的抓取区域。模型可能更依赖 proprioception、桌面背景或训练集平均轨迹，而没有充分使用图像中的方块位置。

2. **过早闭合**  
   方块稍微移到 primary camera 画面前方时，模型仍在熟悉的阶段闭合，说明 close timing 可能与轨迹进度绑定，而非与实际 object-relative pose 绑定。

3. **抓住但不抬升**  
   真实宽度进入方块兼容范围，但模型后续 Z action 不够正向，甚至请求 open。不能用自动抬升伪装修复；这应作为训练/模块的真实评估指标。

4. **数据变化范围窄**  
   旧 50 episodes 的第一次 close 位置大致：

   ```text
   mean XYZ = [0.50072, 0.01051, 0.13552] m
   std XYZ  = [0.00745, 0.01311, 0.00811] m
   ```

   另一次对后续数据的视觉统计显示，方块中心变化仅约 `8.4 × 4.8 px`。这会鼓励模型学习平均抓取轨迹。

5. **wrist camera 视角历史变化**  
   前 50 episodes 的腕部视角不够理想，新增 24 episodes 改善了位置。两个子集可能存在视觉域差异；训练模块研究时应分别检查性能，而不是只看合并 loss。

### 8.4 当前模型是否“学会了”

合理结论是：模型已经学到任务方向、粗略接近和部分夹爪阶段，但尚未证明学到了对方块位置具有充分敏感性的稳定闭环策略。LIBERO 仿真 checkpoint 的 98.3% success rate 只说明其在相应 LIBERO 分布上表现好，不能直接证明真实 Franka 双相机场景的 grounding、动作标定和夹爪时序已经解决。

---

## 9. 后续研究目标与推荐路线

用户下一阶段计划：

> 基于 StarVLA 框架增加一个已经发表论文中的经典模块，在真实 Franka 上验证该模块能否改善效果。候选包括 RL/RLinf 或 3D 先验。

这意味着下一阶段应该是一个可发表/可答辩的**受控增量研究**，而不是继续在 deployment client 里写任务脚本。

### 9.1 推荐优先级

| 优先级 | 路线 | 与当前问题匹配 | 工程风险 | 是否需要重新采深度 | 建议 |
|---:|---|---|---|---|---|
| 1 | Spatial Forcing 风格 3D 表征蒸馏/对齐 | 很高：直接改善空间 grounding | 中 | 理想情况下不需要 | **首选** |
| 2 | RLinf 中的 HG-DAgger/人类干预微调 | 高：针对抓取失败和分布偏移 | 高 | 不需要 | baseline 稳定后做 |
| 3 | PointVLA 风格 point-cloud injection | 高 | 高 | 需要 depth/calibration 或伪点云 | 第二阶段 3D 路线 |
| 4 | SpatialVLA 的 Ego3D/action-grid 思路 | 高 | 很高 | 通常需要明确几何信息 | 适合更大重构 |
| 5 | 在线真实机器人 RL | 中到高 | 极高、安全和 reward 困难 | 不一定 | 不作为第一个模块 |

### 9.2 为什么优先 Spatial Forcing 风格模块

[Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing) 的核心方向是：训练期间让 VLA 的中间视觉表征对齐到预训练 3D foundation model 的空间表征，从而把 3D 结构先验蒸馏进 VLA。论文/项目的目标属性是推理时不需要显式深度输入或额外 3D estimator，因此非常适合现有只存双 RGB、没有完整 depth/calibration 的 74-episode 数据。

它与当前失败模式直接对应：

- 模型对方块平移不够敏感；
- 数据分布窄，容易记平均轨迹；
- wrist/primary 双视角包含几何线索，但现有 action head 未显式受 3D 表征约束；
- 真机部署非常在意延迟，不希望在线再跑一个庞大 3D 模型。

但不能把官方 OpenVLA/OpenPI patch 原样复制就宣称完成。必须先读论文和官方实现，确认：teacher、对齐层、token/feature shape、loss、是否 stop-gradient、训练和推理时哪些分支保留。

### 9.3 与“冻结 vision tower”的关键冲突

导师当前 baseline 要求冻结 Qwen visual tower。如果把 spatial alignment loss 直接施加在**完全冻结且后面没有可训练路径**的视觉 token 上，该 loss 不会改善视觉表征。

推荐做法：

```text
冻结原始 Qwen vision tower
        │
        ▼
添加可训练 geometry adapter / projector / fuser
        │
        ├── 与预计算 3D teacher feature 做 alignment loss
        └── 融合后送入 GR00T action head
```

这样可以尊重“冻结 vision”的 baseline 约束，同时让新增模块确实可学习。另一个实验是只解冻若干上层 visual blocks，但这会改变训练策略、显存和可比性，需要导师明确同意，并应作为独立 ablation。

---

## 10. 推荐模块的具体实现计划

以下是建议，不是已经实现的事实。

### Phase 0：冻结并复现 baseline

1. 确认 Qwen 74-episode final checkpoint 完整。
2. 保存：commit、训练 YAML、launcher、完整 log、环境包版本、GPU、数据 manifest hash、checkpoint hash。
3. 用离线样本做 loader 和 policy-server smoke test。
4. 做固定场景 dry-run，保存原始 action chunk，而不是只看机器人视频。
5. 做第一轮严格真机 baseline，禁止 object-specific close gate 和自动抬升。

如果 baseline 都不能稳定加载或输入映射不一致，先修 pipeline，不要开始模块研究。

### Phase 1：复现 Spatial Forcing 的最小可验证版本

建议新增而非覆盖：

```text
starVLA/model/framework/VLM4A/QwenGR00TSpatialForcing.py
```

配置建议具备：

```yaml
spatial_forcing:
  enabled: true
  teacher_name: <按官方实现确认>
  teacher_feature_cache: <path>
  student_layer: <按实验确认>
  adapter_dim: <按官方/显存确认>
  loss_type: <按官方实现确认>
  loss_weight: <按论文和小规模 sweep 确认>
  stop_gradient_teacher: true
  use_primary: true
  use_wrist: true
```

不要在没有读官方代码的情况下凭空决定 teacher、loss 和 token pooling。复现工作的学术可信度来自明确说明：哪些部分忠实复现、哪些为适配 StarVLA 所做改变。

总 loss 的概念形式可写成：

```text
L_total = L_action_flow_matching + λ × L_spatial_alignment
```

但实际 alignment loss 的定义必须以官方实现为准。

### Phase 2：优先离线预计算 3D teacher feature

当前 QwenGR00T 训练本身峰值已经超过 80 GB。若再在线加载 VGGT/其他 3D teacher，单张 96 GB H100 也可能 OOM。因此建议：

1. 对 74 episodes 的 primary/wrist frames 离线运行 frozen teacher；
2. 缓存与 frame index 一一对应的 feature；
3. 训练时只读缓存，teacher 不驻留 GPU；
4. 保存 feature schema、teacher checkpoint hash、图像预处理版本和 index mapping；
5. 随机抽样核对视频帧、parquet frame 和 teacher feature 没有错位。

不要立即把所有 token 用 float32 无压缩保存。先根据官方特征层计算：

```text
storage = frames × cameras × tokens × feature_dim × bytes_per_value
```

再决定 fp16/bf16、pooling 或 chunked storage。空间 token pool 太激进会抹掉正要学习的位置关系。

### Phase 3：必须具备的单元/集成测试

至少加入：

1. `enabled: false` 时，模型输出与 baseline 路径相同；
2. primary/wrist feature 的 batch、time、token 顺序正确；
3. teacher feature 与 LeRobot `episode_index/frame_index` 一一对应；
4. teacher 参数 `requires_grad=False`；
5. adapter/fuser 有非零 gradient；
6. vision tower 在“冻结方案”下确实无 gradient；
7. `forward()` 和 `predict_action()` 的融合路径一致；
8. 输出仍为 `[B, 8, 7]`，gripper convention 不变；
9. checkpoint save/load 能恢复新增模块；
10. policy server 可加载新 framework，不依赖训练时 teacher；
11. 双图缺一、stale image、feature-cache miss 时明确失败，不能静默用零张量；
12. 推理 latency 不显著高于 baseline，或把增加量如实报告。

### Phase 4：小规模训练闸门

不要一开始再跑 20k：

1. 20–100 steps overfit/smoke；
2. 验证 loss 有限、无 NaN、gradient 合理；
3. 500–2,000 steps pilot；
4. 离线图像平移敏感性测试；
5. 只有 pilot 确实显示空间敏感性改善，再跑完整 20k。

### Phase 5：公平 A/B

对照应只有新增模块不同：

```text
Baseline: Qwen base + frozen vision + GR00T, 74 eps, 20k
Module:   同一初始化、同一数据、同一 seed/steps/LR + spatial module/loss
```

至少保存相同 step 的 checkpoint，例如 2k、4k、6k……20k。不能只挑表现最好的一次真机视频。

---

## 11. RLinf 路线应如何理解

[RLinf](https://github.com/RLinf/RLinf) 是面向 embodied/agentic AI 的 RL infrastructure，官方资料中确实包括：

- StarVLA 的 RL 示例；
- Franka 真机支持；
- GRPO/PPO/SAC/Cross-Q/RLPD 等；
- DAgger/HG-DAgger、人类干预采集；
- RealSense 与真实机器人流程。

但当前官方 turnkey 示例和本项目并不完全相同：

- StarVLA RL 教程主要是 LIBERO 上的 QwenOFT + GRPO；
- HG-DAgger 真机教程主要围绕 OpenPI `pi0/pi0.5` + SpaceMouse/operator；
- Franka 文档部分控制栈是 ROS Noetic/特定 libfranka 组合；本项目当前是 ROS 2 Humble + 已工作的 CRISP/Cartesian controller；
- 本项目是 QwenGR00T、双相机、Quest 3、8×7 delta action chunk。

所以 RLinf 集成是一个研究工程，不是一条现成命令。不要为了使用 RLinf 而替换已经能工作的 ROS 2 控制链。

### 11.1 更现实的第一步：HG-DAgger/干预数据

相较于直接在线 RL，当前问题更适合：

1. 用 baseline policy 自主运行；
2. 人类通过 Quest 3 在即将过早闭合、偏离方块或不抬升时接管；
3. 记录 intervention segments；
4. 只把高质量纠正动作合并进新数据版本；
5. SFT/DAgger 迭代训练；
6. 与原 74-episode baseline 公平对比。

需要自己实现/适配：

- Quest trigger 到 intervention flag；
- policy/teleop controller 平滑切换，保证 `/target_pose` 只有一个 publisher；
- 8×7 action chunk 与实时单步执行的转换；
- intervention 前后 observation/action timestamp 对齐；
- 数据里明确保存 `policy_action`、`human_action`、`executed_action` 和 takeover flag；
- 失败恢复和 episode outcome 标签。

### 11.2 直接在线 RL 的额外要求

当前 74 episodes 都是成功示范，没有可靠的失败 reward/advantage 标注。要做真实机器人在线 RL，至少需要：

- 自动或人工可重复的 success detector；
- 抓取、抬升、放置等 stage reward 的定义；
- 安全 workspace、碰撞、速度、力/扭矩限制；
- reset 机制和 cube/box 自动或人工复位协议；
- 策略更新期间的回滚 checkpoint；
- operator 始终在急停旁；
- 明确区分 reward shaping 与客户端替模型完成任务。

如果 reward 直接使用“进入固定 XYZ 后给分”，可能再次制造平均位置策略；应优先用物体相对状态、视觉检测或真实 task success。

建议先在 RLinf 官方 Docker/仿真中复现其 StarVLA 示例，再单独做本项目 adapter。不要直接在真实 Franka 上调试分布式 RL 基础设施。

---

## 12. 其他 3D 先验候选

### 12.1 SpatialVLA

[SpatialVLA](https://github.com/SpatialVLA/SpatialVLA) 引入 Ego3D position encoding 和 adaptive action grids，方向与当前空间定位问题高度相关。但完整复现对数据几何、深度/相机参数和 action representation 的要求更高。现有 74 episodes 的 LeRobot 数据只明确保存 RGB，没有为研究用途整理 depth、intrinsics、extrinsics，因此它属于更大规模重构。

### 12.2 PointVLA

[PointVLA](https://pointvla.github.io/) 的思路是把轻量 point-cloud 表征注入冻结 action expert，对少样本和几何变化有吸引力。但需要可靠点云：

- RealSense depth 是否与当前 RGB 同步；
- primary/wrist intrinsics/extrinsics；
- hand-eye calibration；
- depth filtering 和机器人本体点剔除；
- 训练数据是否需要重采。

截至本次资料核对，项目主页可读，但没有像 Spatial Forcing 那样明确、成熟的官方代码入口可直接使用；复现风险更高。

### 12.3 3D-VLA

[3D-VLA](https://github.com/UMass-Embodied-AGI/3D-VLA) 是更大范围的 generative world model/3D reasoning 框架，学术价值高，但不适合作为当前毕业项目中第一个“小模块 + 真机验证”。

---

## 13. 基线与模块的严格评估协议

### 13.1 先做离线视觉敏感性测试

在不移动机器人的情况下固定同一 robot state，向 policy 输入：

- 方块位于中心的图像；
- 方块向前/后/左/右移动后的真实图像；
- 可控图像 crop/translation 的合成对照；
- primary 变化、wrist 不变；
- wrist 变化、primary 不变。

记录每次完整 `[8,7]` chunk：

- translation vector 差异；
- close probability/二值动作变化；
- 预测抓取阶段随方块位置是否单调变化；
- baseline 与 3D 模块的 action sensitivity。

不要只看 action norm；应看移动方向是否与物体位移方向一致。

### 13.2 真机摆放分组

建议固定 camera、box、标准起点，只改变 cube：

1. ID center；
2. left/right；
3. forward/backward；
4. 小角度 askew；
5. 如安全且数据支持，再加入小高度变化。

初期可用约 `±30 mm`，确认安全后再测试 `±50 mm`。每个位置至少 5 次只是初步结果；若要报告有统计意义的 success rate，需要更多重复并给出置信区间。

### 13.3 每个 episode 的分阶段指标

必须同时记录：

| 阶段 | 指标 |
|---|---|
| approach | 与 cube center 的最终 XY/Z 误差、首次 close 时末端 pose |
| grasp | 是否接触、最小/稳定宽度、空抓、close timing |
| lift | 抓取后最大正 Z 位移、是否达到 30 mm、是否掉落 |
| transport | 是否朝 box 移动、是否碰撞 |
| place | 方块是否进入 box/目标区、首次 open 时 pose |
| safety | 哪个 filter 触发、改写了什么动作、abort 原因 |
| systems | policy RTT、action period、camera age、controller 是否掉线 |

主指标：**strict pure-VLA full-task success rate**。安全过滤成功率只能作为次要工程指标。

### 13.4 公平性

Baseline 与模块必须保持：

- 相同 74-episode dataset 和 split；
- 相同初始 checkpoint/随机种子；
- 相同训练 steps/LR/batch/diffusion steps；
- 相同 camera placement、lighting、cube/box；
- 相同 robot standard pose；
- 相同 deployment scaling、rate、safety config；
- 相同 trial 顺序或随机化顺序；
- 操作者不能只为某个模型手动把 cube 放回更容易的位置。

不要把新增数据、换相机位置、改 action scale 和新模块同时放在一个实验里。

---

## 14. 新 Agent 接手后的前 30–60 分钟

按这个顺序执行：

1. 阅读本文件、`7.23_7.24_7.27_summary.md` 和 `728_summary.md`。
2. `git status` 和 `git diff --stat`，确认未提交 baseline 修改；不要清理。
3. SSH 到服务器，刷新 Qwen 20k run 是否完成、实际 GPU/PID/log/final checkpoint。
4. 检查失败的 LIBERO retry 是否还有残留进程；只清理属于 `hanyu` 且确认是本项目失败任务的进程。
5. 为 baseline 生成一份不可变 run manifest：代码 commit/diff、dataset manifest hash、config、环境、checkpoint hash。
6. 在不连接机器人执行的情况下启动 baseline policy server，跑一条离线 inference smoke test。
7. 启动相机/controller，只读检查 topics、帧率、staleness、gripper joint state、`/target_pose` publisher。
8. 做 deployment dry-run；不要立即加 `--execute`。
9. 给用户提交一页研究设计：推荐 Spatial Forcing、精确复现来源、StarVLA 适配点、预计显存和 A/B 方案。
10. 用户确认后，新建独立 model variant 和实验分支；先实现 feature-cache/shape tests，再训练。

建议分支命名：

```text
research/spatial-forcing-qwengroot
```

若工作树无法直接切分支，先在不破坏用户修改的前提下创建 worktree 或保存 baseline commit。不要用 destructive git 命令。

---

## 15. 常见故障和解释

### Policy server `address already in use`

```text
OSError: [Errno 98] ... ('0.0.0.0', PORT): address already in use
```

检查：

```bash
ss -ltnp | grep ':10095\b'
pgrep -af server_policy.py
```

选择新端口或在确认 PID 属于自己的旧服务后正常终止。不要随意 kill 其他用户进程。

### Accelerate port 冲突

```text
ConnectionError: distributed communication on port 29500 ... in use
```

使用训练入口支持的 `MAIN_PROCESS_PORT=<空闲端口>`，单机也可尝试 0；但如果停在 NCCL init，检查残留 rank 和 GPU process。

### NCCL init 看似卡住

`Initializing TorchBackend ... backend nccl` 后短时间没输出不一定是死锁，可能在加载 9.3 GB checkpoint/初始化。并行检查：

```bash
pgrep -af 'accelerate|deepspeed|train_starvla'
nvidia-smi
ss -ltnp | grep 29502
```

若最终出现 OOM，根因是显存，不是 NCCL。

### CUDA OOM on “GPU 0” 但实际指定了 GPU 2

设置 `CUDA_VISIBLE_DEVICES=2` 后，进程内部只看到一张卡并把它称为 logical `cuda:0`。OOM 文本中的 GPU 0 可能是物理 GPU 2。用进程环境和 `nvidia-smi` 判定。

### 相机 observation timeout

```text
Timed out waiting for observations: ['primary', 'wrist']
```

先检查容器、topics、QoS、Domain ID、DDS implementation 和帧龄。policy 模型本身通常不是根因。

### Gripper observation timeout

```text
Timed out waiting for observations: ['gripper_state']
```

检查 `/franka_gripper/joint_states` publisher 和两指 joint names。不要用缺少 `joint_names` 的临时 node 命令。

### 机器人肉眼几乎没动

查看两次 `/current_pose` 数值。此前位置每次确实改变几毫米，只是 translation scale/max delta 较小。不要仅为“看起来快”立即提高速度；先核对单位、policy action 分布和 workspace。

### Bash 出现 `$'\E[200~python3' command not found`

这是 terminal bracketed-paste 控制码被粘入命令。按 `Ctrl+C` 清空当前行，重新从纯文本粘贴，不要把 `^[[200~` 或结尾 `~` 一起复制。

### Quest 无线 ADB offline/more than one device

USB serial 和无线地址同时存在时，命令要显式 `-s`：

```bash
adb -s 2G97C5ZHB603FS tcpip 5555
adb disconnect 192.168.1.149:5555
adb connect 192.168.1.149:5555
adb devices -l
```

Quest bridge 启动后没有不断输出不一定“卡住”；需要通过 ROS topic publisher/echo 判断。若 device visible but inaccessible，戴上头显允许 USB debugging，并清理 offline wireless entry。

---

## 16. 研究完成的最低标准

新增模块只有同时满足以下条件，才可以声称“改善了 StarVLA 真机表现”：

1. 有明确已发表论文和官方实现来源；
2. 说明 faithful reproduction 与 StarVLA-specific adaptation 的边界；
3. baseline 和 module 使用同一数据/初始化/训练预算；
4. 新模块有单元测试、shape/gradient/checkpoint/inference 测试；
5. 真机评估不是单次视频，而是多个固定位置、重复 trials；
6. 报告 strict pure-VLA success，不把安全层接管算成纯成功；
7. 显示至少一个与设计目标直接相关的中间指标改善，例如 cube shift 对 action direction 的敏感性、close pose error、grasp/lift stage success；
8. 报告失败、延迟、显存和额外训练成本；
9. 保存可复现实验的代码 commit、config、log、checkpoint hash、数据 manifest；
10. 不通过客户端自动抓取/自动抬升来制造模型成功。

---

## 17. 官方资料入口

- StarVLA 官方仓库：[github.com/starVLA/starVLA](https://github.com/starVLA/starVLA)
- RLinf 官方仓库：[github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)
- RLinf StarVLA/GRPO 教程：[StarVLA on LIBERO](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html)
- RLinf Franka 文档：[Franka real-world environment](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka.html)
- RLinf HG-DAgger：[Human-Guided DAgger](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/hg-dagger.html)
- RLinf RECAP：[RECAP](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- RLinf STEAM：[STEAM](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/steam.html)
- Spatial Forcing 官方代码：[OpenHelix-Team/Spatial-Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing)
- Spatial Forcing 论文：[arXiv:2510.12276](https://arxiv.org/abs/2510.12276)
- SpatialVLA 官方代码：[SpatialVLA/SpatialVLA](https://github.com/SpatialVLA/SpatialVLA)
- PointVLA 项目页：[pointvla.github.io](https://pointvla.github.io/)
- PointVLA 论文：[arXiv:2503.07511](https://arxiv.org/abs/2503.07511)
- 3D-VLA 官方代码：[UMass-Embodied-AGI/3D-VLA](https://github.com/UMass-Embodied-AGI/3D-VLA)

---

## 18. 给新 Codex Agent 的最终任务定义

接手后不要马上写大量代码。先把当前 Qwen 74-episode baseline 的训练与真机表现固化为可复现基线，然后向用户确认第一项研究增量。

默认推荐任务定义：

> 在不改变现有 74-episode 数据、动作空间、真机 controller 和通用安全边界的前提下，忠实复现 Spatial Forcing 的空间表征对齐思想，为 QwenGR00T 增加一个可训练 geometry adapter，并使用离线缓存的 frozen 3D-teacher features 训练。保持原 Qwen vision tower 冻结，确保推理阶段不依赖 3D teacher。以原 Qwen 74-episode 20k 模型为 baseline，做离线空间敏感性、分阶段抓取指标和严格纯 VLA 真机 A/B 评估。

如果用户改选 RLinf，先把任务收窄为：

> 复现 RLinf 官方仿真 StarVLA 示例，并设计一个 ROS 2/Quest 3 HG-DAgger adapter；先完成 intervention 数据格式和离线 SFT 闭环，不立即进行无保护的真实机器人在线 RL。

这两个方向都应先做一个独立、最小、可证伪的实验。不要把“能运行”误写成“模块有效”，也不要把安全控制器替模型完成任务误写成 VLA 成功。
