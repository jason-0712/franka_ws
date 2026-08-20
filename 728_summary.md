# 2026-07-28 Quest3–Franka–StarVLA 工作总结

## 1. 今日目标

今天主要完成了三条工作线：

1. 检查新腕部相机位置下录制的最新 episodes，并确定当前可训练数量；
2. 将原来筛选过的 50 条双相机示范与新筛选的 24 条示范合并成一个完整的 74-episode LeRobot 数据集，准备重新训练 StarVLA；
3. 在新训练因服务器 NVIDIA 驱动问题受阻后，恢复并验证原 50-episode 模型的真实 Franka 部署链路。

截至今天结束：

- 74-episode 数据集已经完成并通过完整性验证；
- 数据和训练配置已经上传、解压并安装到训练服务器；
- 74-episode 新训练尚未启动，唯一阻塞是服务器 NVIDIA 内核驱动与用户空间库版本不匹配；
- 原 50-episode policy server 仍能推理；
- Franka 双相机 observation timeout 已定位并修复；
- 原 50-episode 模型已完成一次成功的单步 dry-run，但今天尚未记录新的真实执行结果。

---

## 2. 新腕部相机数据筛选

### 2.1 当前合格数量

新腕部相机位置下，目前筛选出 24 条可用于训练的 episodes：

```text
0121, 0124, 0126, 0127, 0128,
0131, 0132,
0134, 0135, 0136, 0137, 0138, 0139,
0140,
0142, 0143, 0144, 0145, 0146, 0147,
0148, 0149, 0150, 0151
```

加上原来已经严格筛选的 50 条数据：

```text
50 old vetted episodes + 24 new vetted episodes = 74 usable episodes
```

如果只计算“新相机位置下收集 50 条”的目标，则当前进度是：

```text
24 / 50
```

还需要 26 条高质量新相机示范才能单独达到 50 条目标。

### 2.2 明确排除的数据

以下数据没有加入 74-episode 训练集：

| Source ID | 原因 |
|---|---|
| `0036` | 已从原 50 条筛选集中明确排除 |
| `0125` | 录制不完整 |
| `0129` | 腕部视频冻结，绝大多数相邻帧完全重复 |
| `0130` | 录制不完整 |
| `0133` | 录制不完整 |
| `0141` | Parquet 为 256 帧，但两路视频为 348 帧，数据与视觉时间轴不一致 |
| `0152` | 只创建了初始 metadata，录制未完成 |

### 2.3 `0140–0151` 最新检查结果

自动检查覆盖：

- metadata 是否完成；
- Parquet 是否存在且行数正确；
- primary/wrist 两路视频是否存在；
- 每路视频解码帧数是否与 Parquet 完全相等；
- 是否包含一次 close、一次 reopen；
- close 后是否有明显抬升；
- 是否最终把方块放到箱子上；
- 是否有长时间相机冻结。

`0140、0142–0147` 均通过自动检查和最终画面检查。

`0148–0151` 的结果：

| Episode | 帧数 | Primary 相邻重复帧 | Wrist 相邻重复帧 | 最长连续重复 | 结论 |
|---|---:|---:|---:|---:|---|
| `0148` | 259 | 10.9% | 0% | 2 帧 | 合格 |
| `0149` | 271 | 14.1% | 0% | 2 帧 | 合格 |
| `0150` | 275 | 4.4% | 0% | 2 帧 | 合格 |
| `0151` | 271 | 14.8% | 0% | 2 帧 | 合格 |

Primary 的少量孤立重复帧来自采集/编码节奏，不构成长时间冻结；最长只连续重复 2 帧。Wrist 四条数据均无相邻完全重复帧。最终第三人称画面确认四条都完成了 cube-to-box 放置。

---

## 3. 74-episode LeRobot 数据集合并

### 3.1 输出数据集

最终本地数据集：

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/quest3_franka_dualcam_pickplace_74eps
```

基本信息：

| 项目 | 数值 |
|---|---:|
| Episodes | 74 |
| Frames | 17,328 |
| Videos | 148 |
| Cameras per episode | 2 |
| FPS | 15 |
| Task count | 1 |
| Task | `pick up the cube and place it on the box` |
| LeRobot version | v2.1 |
| Dataset size | 约 183 MB |

完整验证结果：

```text
episodes: 74
frames: 17328
videos: 148
all_video_frames_match: True
global_index_contiguous: True
```

### 3.2 合并时发现并处理的 schema 差异

原 50-episode 合并数据中的向量列由旧 merger 写成可变长度 Arrow list：

```text
list<float32>
```

新单 episode 数据使用 Hugging Face/LeRobot 的固定长度 schema：

```text
fixed_size_list<float32>[N]
```

受影响的字段包括：

- `observation.state.cartesian`；
- `observation.state.joints`；
- `observation.state.target`；
- `observation.state`；
- `action`。

今天没有把两种 schema 直接混放，而是在输出 74-episode 数据集时，将旧数据安全转换为与新数据一致的 fixed-size-list schema，并验证每列数据类型和维度兼容。

同时重写了：

- `episode_index`；
- 全局连续 `index`；
- `task_index`；
- `episodes.jsonl`；
- `episodes_stats.jsonl`；
- `info.json`；
- `merge_manifest.json`。

每条输出 episode 的 `frame_index` 都重新验证为从 `0` 到 `length-1`，全局 `index` 从 0 连续到 17,327。

### 3.3 可复现构建脚本和 manifest

新增构建脚本：

```text
/home/dase-hw101/franka_ws/scripts/build_quest3_franka_dualcam_74eps.py
```

合并清单：

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/quest3_franka_dualcam_pickplace_74eps/meta/merge_manifest.json
```

manifest 记录：

- 原 50 条的 source IDs；
- 新增 24 条的 source IDs；
- 每个输出 episode 对应的 source dataset/source index；
- 所有明确排除的数据及排除原因；
- 每条 episode 的长度。

### 3.4 传输包

数据集压缩包：

```text
/home/dase-hw101/franka_ws/quest3_franka_dualcam_pickplace_74eps.tar.gz
```

大小约 180 MB，SHA-256：

```text
55e6a98e8bebd1918efabc2d206f94c7d61e41a80cfae1812f3bf29df498ae55
```

包含数据、registry、训练配置和服务器安装脚本的完整传输包：

```text
/home/dase-hw101/franka_ws/starvla_dualcam_74eps_training_bundle_20260728.tar
```

大小约 180 MB，最终 SHA-256：

```text
08e0fcf115591e41fa457a98b86e0c4f2caacf7b53d743e9b213d27024cae479
```

---

## 4. 74-episode StarVLA 训练方案

### 4.1 初始化方案

继续采用之前选定的 Scheme A：

- 从原始 LIBERO 30k checkpoint 初始化；
- 使用新的 AdamW optimizer；
- 不加载此前 50-episode Franka checkpoint；
- 不继承旧真实模型可能存在的过早 close、固定位置倾向和视觉 grounding 不足。

输入 checkpoint：

```text
/data/hanyu/starVLA_checkpoints/libero_all_gr00t_official_30000_rerun/final_model/pytorch_model.pt
```

服务器确认该文件存在，大小约 9.3 GB。

计划输出：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_libero30k_10k
```

### 4.2 计划训练参数

| 参数 | 设置 |
|---|---|
| Dataset mix | `quest3_franka_dualcam_pickplace_74eps` |
| Max steps | 10,000 |
| Save interval | 2,000 |
| Physical GPU | 1 |
| Per-device batch size | 1 |
| Repeated diffusion steps | 8 |
| Action-model learning rate | `1e-5` |
| Warmup steps | 500 |
| Frozen module | `qwen_vl_interface` |
| Optimizer | fresh AdamW |

### 4.3 新增/修改的训练文件

```text
scripts/start_starvla_dualcam_74eps_from_libero30k_train.sh
scripts/deploy_and_start_starvla_dualcam_74eps_from_libero30k.sh
scripts/install_and_start_starvla_dualcam_74eps_on_server.sh
third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py
```

Registry 新增：

```text
quest3_franka_dualcam_pickplace_74eps
```

### 4.4 服务器端当前数据状态

74-episode bundle 已经成功传到 `server1cps` 并解压。

服务器端数据路径：

```text
/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_pickplace_74eps
```

服务器安装脚本已验证：

```text
parquet_count = 74
video_count = 148
total_episodes = 74
total_frames = 17328
```

StarVLA registry、训练入口、YAML 和 launcher 也已经安装到 `/home/hanyu/starVLA`。

---

## 5. 新训练未启动的真正原因

### 5.1 直接报错

服务器运行：

```bash
nvidia-smi -i 1
```

返回：

```text
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 595.84
```

加载中的内核驱动版本：

```text
NVIDIA Open Kernel Module 595.71.05
```

因此当前冲突是：

```text
running NVIDIA kernel module = 595.71.05
userspace NVML library       = 595.84
```

### 5.2 冲突发生的位置

```text
nvidia-smi / PyTorch
        ↓
libnvidia-ml / libcuda 595.84
        ↓
/dev/nvidiactl
        ↓
running NVIDIA kernel module 595.71.05
        ↓
version handshake fails
```

`env -u LD_LIBRARY_PATH /usr/bin/nvidia-smi -i 1` 仍然失败，说明不是 conda 的 `LD_LIBRARY_PATH` 选择了错误库，而是宿主机系统级驱动组件本身不一致。

### 5.3 为什么原 50-episode 模型以前可以训练

50-episode 模型训练时，GPU 驱动仍可正常初始化。之后可能发生了用户空间 NVIDIA 库更新，但正在运行的旧内核模块没有被替换。

另一个重要现象是：已经在驱动更新前启动的进程可能继续使用内存中已加载的旧库，而今天新启动的 `nvidia-smi` 或训练进程会加载当前 595.84 库并失败。

空的 apt/dpkg 搜索结果不能否定版本冲突；驱动可能由其他方式安装，日志也可能已经轮转。两个直接检测到的版本已经足以证明不匹配。

### 5.4 为什么没有训练日志

服务器安装脚本在以下检查处停止：

```bash
gpu_used=$(nvidia-smi --query-gpu=memory.used ...)
```

因此：

- 没有进入 PyTorch；
- 没有加载 9.3 GB checkpoint；
- 没有创建输出目录；
- 没有创建 `.launcher.log`；
- 74-episode 训练没有开始过。

这不是数据集、checkpoint 或 StarVLA 配置错误。

### 5.5 当前不能直接重启的原因

`who` 显示服务器上有 `hanyu` 和 `liji` 的大量长期 tmux/SSH 会话。`sudo reboot` 会终止：

- 所有 GPU 作业；
- Python/Jupyter 进程；
- tmux 中的计算任务；
- SSH 会话；
- 未保存的工作。

tmux 不能让进程跨系统重启继续运行。因此在未联系其他用户和管理员前，不应直接重启。

### 5.6 可选解决方案

按推荐顺序：

1. 与所有服务器用户协调维护时间，然后重启，使新内核模块生效；
2. 由管理员将用户空间 NVIDIA 库恢复为与当前内核模块匹配的 `595.71.05`；
3. 由管理员停止所有 GPU 进程后重新加载驱动模块，但该方案同样会影响全体用户且风险更高；
4. 把 74-episode 训练迁移到另一台驱动正常的 GPU 服务器。

Docker、conda 或更换 CUDA toolkit 无法解决宿主机内核驱动与系统用户库的版本冲突。

协调修复前建议检查磁盘上的待加载模块版本：

```bash
modinfo -F version nvidia
modinfo -n nvidia
dkms status | grep -i nvidia
```

如果 `modinfo` 显示 `595.84`，说明磁盘上已有新模块，协调重启后通常可恢复。如果仍为 `595.71.05`，则需要管理员补装新内核模块或回退用户空间库。

---

## 6. 恢复原 50-episode 模型部署

### 6.1 初始错误

客户端报错：

```text
TimeoutError: Timed out waiting for observations: ['primary', 'wrist']
```

这发生在 policy inference 之前，表示客户端没有收到两路图像。

### 6.2 相机本身是正常的

相机容器：

```text
franka_dual_realsense
```

两路压缩 topic：

```text
/right/right_third_person_camera/color/image_raw/compressed
/right/right_wrist_camera/color/image_raw/compressed
```

实测速率：

```text
primary compressed ≈ 29.96–29.99 Hz
wrist compressed   ≈ 29.96–29.99 Hz
```

每路 topic 均有 1 个 publisher，当前 publisher QoS 是：

```text
Reliability: BEST_EFFORT
Durability: VOLATILE
History: KEEP_LAST(5)
```

### 6.3 第一个原因：ROS domain 不一致

Camera container 使用：

```text
ROS_DOMAIN_ID=30
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

但 `franka` 容器默认环境中两项均未设置，客户端因此默认进入 ROS domain 0，完全看不到 domain 30 的相机。

在 Franka 部署终端必须显式执行：

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 6.4 第二个原因：图像 QoS 不兼容

客户端原来通过整数 depth 创建图像 subscription，等价于默认 `RELIABLE` request。相机提供的是 `BEST_EFFORT`，因此即使 domain 正确，DDS 也可能拒绝连接。

今天修改：

```text
/home/dase-hw101/franka_ws/scripts/starvla_franka_delta_pose_client.py
```

加入：

```python
from rclpy.qos import qos_profile_sensor_data
```

并让 raw/CompressedImage 两种相机 subscription 都使用：

```python
qos_profile_sensor_data
```

修改已同步到运行中的 `franka` container：

```text
/home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py
```

两份文件同步后的 SHA-256：

```text
79f0b8f5234921dbc711943715f58351f36988ef989249fbeeab7679efc38ea0
```

### 6.5 Controller 和 ROS 控制链状态

检查时：

```text
cartesian_impedance_controller: active
pose_broadcaster: active
joint_state_broadcaster: active
```

`/target_pose`：

```text
Publisher count: 0
Subscription count: 1
```

因此当时没有 teleop/旧 VLA client 与新 client 争用 `/target_pose`，Controller 已准备接收命令。

服务器端口：

```text
192.168.1.113:10093 open
192.168.1.113:10094 open
```

正确的 50-episode model server 是 `10094`。

---

## 7. 50-episode 模型 dry-run 结果

### 7.1 Observation

修复 domain 和 QoS 后，单步 dry-run 成功收到：

```text
primary image: (720, 1280, 3)
wrist image:   (480, 640, 3)
```

初始末端位置：

```text
[0.313871, 0.000420, 0.584967] m
```

该位置接近训练标准起点：

```text
[0.308852, 0.000865, 0.584837] m
```

初始姿态：

```text
RPY xyz = [-3.11269, 0.01594, -0.75349] rad
```

夹爪反馈：

```text
total_width = 0.079989 m
state.gripper = 0.000135
```

因此夹爪处于正常全开状态。

### 7.2 Policy server metadata

`192.168.1.113:10094` 返回：

```text
checkpoint:
/data/hanyu/starVLA_runs/quest3_franka_dualcam_50eps_from_libero30k_10k/final_model/pytorch_model.pt

action_chunk_size: 8
action keys:
  action.delta_eef_position
  action.delta_eef_rotation
  action.gripper

state keys:
  state.eef_position
  state.eef_rotation
  state.gripper

supports_raw_state_normalization: True
```

这证明 dry-run 使用的是原 50-episode 双相机真实 Franka 模型，不是 LIBERO-only checkpoint，也不是另一个旧服务。

尽管当前新进程无法初始化 NVIDIA 驱动，已经启动并仍在运行的 50-episode policy server 可以继续响应推理请求。这与“旧进程可能仍使用已加载的旧驱动库”一致。

### 7.3 第一帧预测

Action chunk：

```text
shape = (8, 7)
gripper = [1, 1, 1, 1, 1, 1, 1, 1]
```

夹爪语义：

```text
1 = open
0 = close
```

第一步经过 scale/clamp 后：

```text
dpos = [0.00436, 0.00246, -0.00257] m
drpy = [0.0000799, -0.00000589, 0.0001998] rad
target position = [0.3182, 0.0029, 0.5824] m
```

推理 round trip 约：

```text
0.101 s
```

本次是 dry-run：

```text
execute=False
```

因此没有发布 `/target_pose`，Franka 没有因该测试发生运动。

### 7.4 当前部署安全逻辑

客户端当前启用了通用安全/反馈过滤：

- action chunk 至少 75% 同意 open/close；
- 需要 3 次连续 policy request 才切换夹爪状态；
- 使用真实 `/franka_gripper/joint_states`；
- 有 cube-compatible width 检查；
- 有 close timeout、stable width 和 lift 验证；
- 每个 episode 最多一次物理抓取尝试；
- 没有基于成功示范 XYZ 坐标的 object-specific close gate。

因此使用当前命令得到的是“50-episode VLA + 通用安全过滤”的部署表现，不是完全无过滤的 pure-VLA 指标。报告结果时应将二者区分。

---

## 8. 当前真实部署命令

在 `franka` container 的 `ros@...:~/ros2_ws` 终端执行：

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash

mkdir -p /home/ros/ros2_ws/deployment_logs

python3 /home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10094 \
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
  --execute \
  --log-timing \
  2>&1 | tee /home/ros/ros2_ws/deployment_logs/vla_50eps_test_$(date +%Y%m%d_%H%M%S).log
```

运行前要求：

- Franka 从标准初始位姿附近开始；
- 夹爪确认为全开；
- cube/box/相机摆放与训练分布一致；
- `/target_pose` publisher count 为 0；
- Cartesian controller active；
- 操作者手放在急停附近；
- 不同时运行 Quest teleop adapter/recorder。

---

## 9. 今日修改或新增文件

### 数据集构建

```text
scripts/build_quest3_franka_dualcam_74eps.py
dataset/snkdjn/quest3_franka_dualcam_pickplace_74eps/
quest3_franka_dualcam_pickplace_74eps.tar.gz
starvla_dualcam_74eps_training_bundle_20260728.tar
```

### 74-episode 训练

```text
scripts/start_starvla_dualcam_74eps_from_libero30k_train.sh
scripts/deploy_and_start_starvla_dualcam_74eps_from_libero30k.sh
scripts/install_and_start_starvla_dualcam_74eps_on_server.sh
third_party/starVLA/examples/realRobots/Franka/train_files/data_registry/data_config.py
```

### Franka 部署

```text
scripts/starvla_franka_delta_pose_client.py
```

部署客户端的实际 container 副本也已同步：

```text
franka:/home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py
```

---

## 10. 当前状态快照

### 数据

```text
Old vetted dataset: 50 episodes
New-camera vetted subset: 24 episodes
Merged train-ready dataset: 74 episodes
New-camera target remaining: 26 episodes
```

### 74-episode training

```text
Dataset uploaded: yes
Dataset validated on server: yes
Checkpoint present: yes
Training files installed: yes
Training started: no
Blocker: NVIDIA kernel 595.71.05 vs NVML 595.84
```

### Original 50-episode deployment

```text
Policy server 192.168.1.113:10094: reachable
Correct checkpoint metadata: confirmed
Camera container: running
Primary compressed stream: ~30 Hz
Wrist compressed stream: ~30 Hz
ROS domain/QoS bug: fixed
Cartesian controller: active at inspection time
/target_pose competing publisher: none at inspection time
Single-step dry-run: successful
New real rollout result: not yet recorded today
```

---

## 11. 下一步

### 11.1 先评估原 50-episode 模型

1. 确认 Franka 标准初始位姿、夹爪打开和场景摆放；
2. 使用本总结中的压缩双相机执行命令；
3. 全程保存 deployment log 和外部视频；
4. 记录 cube 初始位置；
5. 分别记录 approach、close、lift、transport 和 release 是否由 policy 正确完成；
6. 若失败，保留失败日志，不要通过手动移动 cube 把它描述为纯 VLA 成功；
7. episode 结束后安全返回标准位姿。

### 11.2 修复训练服务器

1. 联系 `liji` 和服务器管理员；
2. 检查磁盘中的 NVIDIA module 版本；
3. 协调维护时间重启，或由管理员恢复匹配的用户空间库；
4. 修复后验证 `nvidia-smi -i 1`；
5. 确认 GPU 1 空闲；
6. 启动 74-episode 10k-step 训练；
7. 在出现第一条 `Step ..., Loss ...` 后再确认训练真正开始。

驱动恢复后，服务器上可直接使用已安装的 launcher：

```bash
nohup bash /home/hanyu/starVLA/start_starvla_dualcam_74eps_from_libero30k_train.sh \
  > /data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_libero30k_10k.launcher.log \
  2>&1 < /dev/null &
```

### 11.3 新模型完成后的比较

应固定相同的：

- 标准起点；
- cube/box 位置集合；
- 相机位姿；
- 客户端 movement scale；
- gripper safety settings；
- 每个位置的重复次数。

对比：

```text
50-episode model vs 74-episode model
```

重点指标：

- 首次 approach 是否随 cube 位置变化；
- 首次 close 的空间误差；
- 有效抓取率；
- 抓取后抬升率；
- 完整 pick-and-place 成功率；
- pure-VLA 成功率；
- safety-filtered 成功率。

今天最关键的结论是：数据、模型部署链路和新训练准备已经分别验证；当前新训练失败不是数据问题，而是共享训练服务器的系统级 NVIDIA driver mismatch。与此同时，原 50-episode policy server 仍可用于真实部署评估，双相机 timeout 已通过 ROS domain 和 QoS 修复。
