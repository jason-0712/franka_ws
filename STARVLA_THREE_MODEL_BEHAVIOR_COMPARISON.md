# StarVLA 三模型训练与行为对比

日期：2026-08-03  
工作区：`/home/dase-hw101/franka_ws`  
任务：`pick up the cube and place it on the box`  
机器人：Franka FR3，primary + wrist 双 RGB 相机  
数据：筛选后的 74 个 Quest3 teleoperation episodes，15 FPS，`0036` 已排除

本文比较三个模型：

1. `Qwen-base`：通常口头称为“from scratch”；
2. `Libero-init`：从 LIBERO 30k StarVLA checkpoint 继续训练；
3. `Libero + 3D prior`：从同一 LIBERO checkpoint 继续训练，并增加冻结的 Depth Anything V2-Small 相对深度分支和 gated cross-attention。

本文同时记录训练差异、严格配对的 open-loop 指标、真实 Franka 行为、当前证据的限制，以及三种模型可复现的 server、offline evaluation、dry-run 和真机执行命令。

---

## 1. 最重要的结论

### 1.1 当前最佳真机基线仍是 Libero-init

当前最可信的真机结果是：

```text
Libero-init 74eps:
13/20 = 65% safety-filtered end-to-end success
```

该结果使用：

```text
gripper chunk consensus = 0.75
gripper switch confirmations = 3
close latch = enabled
physical lift validation = 30 mm
Cartesian temporal ensemble = disabled, window=1
max grasp attempts = 1
```

它不是 strict raw/pure VLA，因为客户端对夹爪命令做了通用去抖、close latch 和真实反馈验证；但它也不是 scripted pick-and-place：客户端没有固定方块坐标、没有自动补做 +30 mm 抬升，也没有替模型决定释放。

### 1.2 Qwen-base 明显弱于 Libero-init

Qwen-base 在同一 345-query open-loop 协议下表现出：

- XYZ 误差更大；
- Y 方向存在系统性偏置；
- 更频繁地提前 close；
- false close 和 missed close 都更多。

初步真机测试为 `0/3`。主要观察是末端相对方块存在明显偏位，并容易在方块上方提前关闭。

### 1.3 当前 3D-prior checkpoint 离线更好，但真机没有提升

`Libero + 3D prior` 在 demonstration-conditioned open-loop 中取得了三者中最低的 first-action XYZ mean，并显著降低 false close；但是最初三次真实 closed-loop 测试均失败，末端从 primary camera 视角越过方块并碰到方块后方桌面。

所以当前结论不是“3D prior 无效”，而是：

```text
当前这一个 DAv2 gated-fusion 实现和 checkpoint
尚未把离线提升转化为真实闭环提升。
```

在相同、原始 13/20 客户端配置下重新做严格 A/B 之前，不应宣称它优于 Libero baseline。

### 1.4 训练 loss 不能替代真机评估

3D-prior 模型最终 W&B summary 的 loss 很低，但真机行为更差。这说明训练集 loss 和 teacher-forced open-loop 都不足以覆盖：

- closed-loop observation drift；
- 相机视角与物体位置变化；
- 机器人控制延迟和目标累积；
- 接近、贴近桌面、close、lift 等阶段切换；
- 深度先验与机器人坐标系之间的对应关系。

---

## 2. “From scratch”这一名称的准确含义

本文沿用用户口头表达，把第一个模型写成“from scratch”，但它并不是所有参数随机初始化。

准确结构是：

```text
Qwen3-VL-4B-Instruct pretrained base
+ randomly initialized GR00T/DiT action head
+ 74 real-Franka episodes
```

因此更准确的名称是：

```text
Qwen-base + fresh action head
```

Libero-init 的差异是 Qwen/视觉到动作框架以及 action head 都从已经完成 30k LIBERO embodied training 的兼容 StarVLA checkpoint 初始化。

---

## 3. 三个模型的 checkpoint

### 3.1 Qwen-base / fresh action head

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/
final_model/pytorch_model.pt
```

建议 policy port：`10095`。

### 3.2 Libero-init

父 checkpoint：

```text
/data/hanyu/starVLA_checkpoints/
libero_all_gr00t_official_30000_rerun/
final_model/pytorch_model.pt
```

74-episode final checkpoint：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

建议 policy port：`10096`。

### 3.3 Libero-init + Depth Anything gated spatial prior

```text
/data/hanyu/starVLA_runs/
qwengroot_spatial_libero74_vision_frozen_20k_20260803_032202/
final_model/pytorch_model.pt
```

建议 policy port：`10097`。

端口只是约定，不是模型身份。每次测试必须查看 policy metadata 中的 `ckpt_path`，不能只相信端口号。

---

## 4. 训练设置对比

| 项目 | Qwen-base | Libero-init | Libero + 3D prior |
|---|---|---|---|
| Real-Franka data | 相同 74 episodes | 相同 74 episodes | 相同 74 episodes |
| Primary/wrist RGB | 是 | 是 | 是 |
| 初始化 | Qwen3-VL base | LIBERO 30k full checkpoint | LIBERO 30k full checkpoint |
| Action head 初始状态 | 随机初始化 | 已有 embodied action prior | 已有 embodied action prior |
| Qwen visual tower | frozen | frozen | frozen |
| Depth model | 无 | 无 | frozen DAv2-Small |
| Geometry projector | 无 | 无 | trainable |
| Spatial fuser | 无 | 无 | trainable gated cross-attention |
| Optimizer | fresh AdamW | fresh AdamW | fresh AdamW |
| Steps | 20,000 | 20,000 | 20,000 |
| Action LR | `1e-4` | `1e-4` | `1e-4` |
| Qwen interface LR | `1e-7` | `1e-7` | `1e-7` |
| Geometry/fuser LR | — | — | 各 `1e-4` |
| Warmup | 1,000 | 1,000 | 以 final server config 为准；正式 run 按公平 20k 配置启动 |
| Reported epoch | 约 1.15 | 约 1.15 | 约 1.15 |

三组都没有加载旧 50-episode real-Franka checkpoint。主要研究变量分别是：

```text
Qwen-base vs Libero-init:
是否具有 LIBERO embodied-action initialization

Libero-init vs Libero + 3D prior:
是否增加 frozen relative-depth branch + learned gated fusion
```

正式复现实验时，应以每个 final model 目录内保存的 `config.yaml`、`config.full.yaml` 和 policy metadata 为最终权威来源，不能只根据 run 名推断配置。

### 4.1 训练末尾的 W&B summary

| 指标 | Qwen-base | Libero-init | Libero + 3D prior |
|---|---:|---:|---:|
| action_dit_loss | 约 0.57323 | 约 0.42184 | 约 0.02877 |
| mse_score | 约 0.03988 | 约 0.02782 | 约 0.00447 |
| epoch | 约 1.15 | 约 1.15 | 约 1.15 |

这些数值是训练过程 summary/末批次附近的指标，不是 held-out test success。尤其不能因为 3D-prior loss 更低，就推断其真机成功率更高。

---

## 5. 3D prior 实际实现了什么

### 5.1 数据流

当前 `QwenGR00TSpatial` 的数据流为：

```text
primary RGB + wrist RGB
        │
        ├──────────────→ Qwen3-VL visual/language tokens
        │
        └→ frozen Depth Anything V2-Small
              → per-view relative depth
              → 14 × 14 grid
              → [relative_depth, image_x, image_y]
              → trainable geometry projector
              → view-aware gated cross-attention
              → residual fusion into Qwen image-token spans
              → existing GR00T/DiT action head
```

具体 geometry token shape 为：

```text
[batch, 2 cameras, 196 grid tokens, 3]
```

每个 token 的 3 个值是：

```text
normalized relative depth
normalized image-plane x
normalized image-plane y
```

### 5.2 Gate

每个相机有一个可学习 gate：

```python
fused_rgb = rgb_tokens + tanh(gate_view) * spatial_update
```

gate 从 0 初始化，所以刚加载 Libero baseline 时 spatial branch 是精确 identity，不会在训练前立刻破坏 baseline 输出。训练过程中 projector、cross-attention 和 gate 学习如何注入深度信息。

### 5.3 它不是什么

当前实现不是完整的 metric 3D perception：

- 没有使用 RealSense depth stream；
- 不需要每个 episode 的真实深度文件；
- 没有相机内参/外参标定输入；
- 没有把像素反投影为机器人坐标系点云；
- 没有显式估计方块的世界坐标；
- DAv2 输出是单目相对深度，不是可靠的绝对米制距离。

所以“3D prior”是方便的项目简称。更严格的名称是：

```text
frozen monocular relative-depth / 2.5D spatial prior
```

### 5.4 相关实现文件

```text
third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00TSpatial.py
third_party/starVLA/starVLA/model/modules/spatial/depth_anything_v2.py
third_party/starVLA/starVLA/model/modules/spatial/gated_cross_attention.py
third_party/starVLA/examples/realRobots/Franka/train_files/
  starvla_cotrain_quest3_franka_delta_eef_spatial.yaml
third_party/starVLA/tests/test_qwengroot_spatial.py
scripts/run_qwengroot_spatial_tests_and_smoke.sh
```

Baseline 模型类：

```text
third_party/starVLA/starVLA/model/framework/VLM4A/QwenGR00T.py
```

---

## 6. Open-loop 公平评估协议

三个模型应使用完全相同的协议：

```text
episodes: 0047 0077 0099 0121 0149 0150 0151
queries: 345
dataset FPS: 15
stride: 5 frames
action offset: 0
views: recorded primary RGB + recorded wrist RGB
task: pick up the cube and place it on the box
output: 8 × 7 delta-EEF action chunk
```

Open-loop evaluator每次重新使用成功 demonstration 中的 observation。模型前一步的错误不会改变下一帧输入，因此它是 teacher-forced 分析，不是 closed-loop success test。

---

## 7. Open-loop 三模型结果

| Metric | Qwen-base | Libero-init | Libero + 3D prior |
|---|---:|---:|---:|
| Queries | 345 | 345 | 345 |
| First XYZ L2 mean | 4.481 mm | 3.138 mm | **3.080 mm** |
| First total-action L2 mean | 0.143173 | 0.052319 | **0.040717** |
| Whole-chunk total L2 mean | 0.122011 | 0.041778 | **0.036373** |
| Y MAE | 1.674 mm | **1.078 mm** | 1.124 mm |
| Y sign accuracy | 70.88% | **89.01%** | 86.26% |
| Predicted mean Y/action | +0.429 mm | -0.625 mm | -0.503 mm |
| GT mean Y/action | -0.575 mm | -0.575 mm | -0.575 mm |
| Gripper binary accuracy | 86.09% | 95.07% | **96.23%** |
| False close rate, GT open | 14.42% | 6.05% | **1.40%** |
| Missed close rate, GT closed | 13.08% | **3.08%** | 7.69% |
| Mean first-close timing | 18.57 frames early | 7.14 frames early | **1.43 frames early** |

`total-action L2` 会被幅度为 1 的 binary gripper error 主导，所以 Cartesian 和 gripper 指标必须分开看。

### 7.1 Qwen-base 的离线行为

- 7/7 episodes 都比 demonstration 更早首次 close；
- 平均提前约 18.57 frames，即约 1.24 s；
- mean predicted Y 与 GT 符号相反；
- false close 与 missed close 都是三者中最高；
- 表明模型学到粗略任务阶段，但定位方向和夹爪阶段不稳定。

### 7.2 Libero-init 的离线行为

- XYZ mean 比 Qwen 低约 30%；
- Y sign accuracy 比 Qwen 高约 18.13 percentage points；
- gripper accuracy 明显更高；
- 仍有提前 close，不是无需过滤的 pure-VLA policy；
- 结果支持将其作为当前真机 baseline。

### 7.3 3D-prior 的离线行为

相对 Libero-init：

- first XYZ mean 从 3.138 mm 降到 3.080 mm，改善约 1.8%；
- false close 从 6.05% 降到 1.40%；
- gripper accuracy 从 95.07% 升到 96.23%；
- first-close timing 更接近 demonstration；
- 但 Y sign accuracy 从 89.01% 降到 86.26%；
- missed close 从 3.08% 增加到 7.69%。

因此 spatial model 的离线结果不是所有指标一致变好，而是：

```text
close precision / aggregate XYZ 略有改善，
但 Y direction 和 close recall 有退化。
```

### 7.4 权威 artifacts

```text
deployment_logs/open_loop/qwen74_open_loop_7eps_stride5_20260730.log
deployment_logs/open_loop/libero74_open_loop_7eps_full_stride5_20260730.log
deployment_logs/open_loop/dav2_gated_74eps_20k_full_stride5_20260803_141352.log

deployment_logs/open_loop/qwen74_open_loop_7eps_stride5_20260730.csv
deployment_logs/open_loop/libero74_open_loop_7eps_full_stride5_20260730.csv
deployment_logs/open_loop/dav2_gated_74eps_20k_full_stride5_20260803_141352.csv
```

不要使用只有 224 queries 的早期 Libero smoke artifact 做最终比较。

---

## 8. 真机 closed-loop 行为

### 8.1 Qwen-base

已记录的初步测试：

```text
success: 0/3
```

主要行为：

- primary camera 中末端相对方块有明显侧向/前后偏差；
- 定位不稳定；
- 容易在方块上方关闭；
- 与 open-loop 的 Y 偏置和平均提前 close 一致。

这个样本很小，不能估计精确成功率，但足以说明它不是当前最佳真机 baseline。

### 8.2 Libero-init

正式程度最高的一组测试：

```text
success: 13
failure: 7
success rate: 65%
```

主要能力：

- 可以从标准位接近方块；
- 可以在部分 trials 完成 close、真实抬升、搬运和释放；
- 相比 Qwen-base，定位和夹爪阶段明显更稳定；
- 仍会发生 alignment、no-lift、no-release、workspace 或 feedback 类失败。

该数字的正确名称是：

```text
65% safety-filtered end-to-end VLA success
```

20 trials 的样本仍然有限，Wilson 95% interval 约为 43%–82%。

### 8.3 Libero + 3D prior

最初三次真实 closed-loop 测试观察到：

```text
success: 0/3 observed initial trials
```

主要行为：

- 末端三次都从 primary-camera 视角越过方块；
- 继续下降并碰到方块后方桌面；
- 即使将方块沿 primary 前后方向移动，抓取目标也没有表现出足够可靠的修正；
- 后续一次运行中，模型已经输出全 close chunk，但只累计到 `pending_count=2`，随后 current pose 与 gripper feedback stale，中止发生在第三次 confirmation 之前。这一次不能简单归因于“模型不想 close”。

### 8.4 Cube-shift snapshot probe

在机器人不移动、EEF state 基本不变时，将方块放在 center/front/back/left/right，spatial model 的第一步 Cartesian 输出发生变化：

| Placement | First dx | First dy | First dz |
|---|---:|---:|---:|
| center | +5.917 mm | +1.181 mm | -7.708 mm |
| primary front | +2.913 mm | -0.081 mm | -1.816 mm |
| primary back | +4.647 mm | +0.968 mm | -5.838 mm |
| primary left | +2.784 mm | +0.115 mm | -2.451 mm |
| primary right | +4.452 mm | +0.328 mm | -5.026 mm |

这说明模型不是完全忽略图像；但是变化不等于正确 grounding。当前 probe 没有提供方块的精确机器人坐标，也没有验证预测方向是否应与实际位移单调对应。

Artifacts：

```text
deployment_logs/sensitivity/dav2_cube_shift_20260803_144954_*.log
```

### 8.5 必须排除的客户端混淆变量

在 spatial 真机测试后，客户端曾临时加入 XY/Z target tracking tube，其中 Z 限幅为 15 mm。该限幅会在命令目标领先真实末端时重投影 candidate target。

启用后观察到：

- spatial model 在方块上方停留/抖动；
- 随后 Libero baseline 也在方块上方停留；
- 因此这一阶段不能用于判断哪个模型更好。

该 tracking limiter 已经完整删除，客户端恢复到 13/20 测试前保存的版本。当前脚本 SHA256：

```text
ca576a2e5a263fec435785b67b124300c3f51a71ce751675b34bc9b2a8cca134
```

后续公平 A/B 命令中不要再包含：

```text
--max-target-xy-tracking-error
--max-target-z-tracking-error
```

---

## 9. 为什么 3D-prior 离线好、真机却差

当前证据支持以下可能性，但还不能确认唯一原因。

### 9.1 Relative depth 不等于 robot-frame 3D

DAv2 提供物体间相对远近和轮廓信息，但没有绝对尺度、相机外参或 EEF-to-camera transform。网络必须仅靠 74 条 demonstration 学会从两幅相对深度图映射到机器人 XYZ correction，这对小数据集仍然困难。

### 9.2 训练摆位分布较窄

74 条示范中，方块大多被手动放在相近区域。模型可能学习到场景模板和平均轨迹，而不是对方块位置做强闭环追踪。加入深度分支不会自动解决数据覆盖不足。

### 9.3 Open-loop 不惩罚误差累积

Open-loop 每次使用成功 demonstration 帧。即使单步只偏 3 mm，真实机器人多次执行后也可能偏离训练图像，随后进入模型未见过的 observation distribution。

### 9.4 Fusion 可能改变已有 Libero 表征

zero gate 保证初始化时不改变 baseline；20k training 后 gate 和 cross-attention 会改变 visual token。它可能降低训练 loss，却损伤原本对真实相机偏差更鲁棒的 feature。必须检查 final checkpoint 的 gate、attention magnitude 和消融结果。

### 9.5 深度模型的域偏差

DAv2 是通用单目深度模型。金属夹爪、反光桌面、小型蓝色方块、遮挡和 wrist-camera 近距离视角都可能造成相对深度误差。

### 9.6 不能把所有失败都归给模型

stale camera/current-pose/gripper feedback、controller target lag、临时 15 mm limiter 和人为摆位差异都是系统层变量。严格模型比较必须固定并记录它们。

---

## 10. 推荐的科学结论措辞

可以写：

> 在相同 74 条真机双相机示范上，LIBERO 初始化显著优于 Qwen-base 初始化。加入 frozen DAv2 relative-depth gated fusion 后，teacher-forced open-loop XYZ 与夹爪 precision 略有改善，但最初真实闭环测试未复现该提升，并出现系统性越过方块的失败。因此当前 spatial checkpoint 尚未优于 65% 的 Libero-init safety-filtered baseline。

暂时不要写：

```text
3D prior improves real-robot success rate
```

也不要写：

```text
3D prior does not work
```

正确说法是当前实现尚未通过公平、足量的真机验证。

---

## 11. Server 端：启动三个模型的代码

以下命令在服务器 `server1cps` 上运行：

```text
hanyu@192.168.1.113
```

它不会直接移动机器人，只启动 policy server。不要在不知道归属的 GPU/端口上杀进程。

### 11.1 定义统一启动函数

```bash
cd /home/hanyu/starVLA
source /home/hanyu/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

start_starvla_server() {
  local model="$1"
  local gpu="$2"
  local port
  local ckpt

  case "${model}" in
    qwen)
      port=10095
      ckpt=/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/final_model/pytorch_model.pt
      ;;
    libero)
      port=10096
      ckpt=/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/final_model/pytorch_model.pt
      ;;
    spatial)
      port=10097
      ckpt=/data/hanyu/starVLA_runs/qwengroot_spatial_libero74_vision_frozen_20k_20260803_032202/final_model/pytorch_model.pt
      ;;
    *)
      echo "Usage: start_starvla_server {qwen|libero|spatial} GPU_ID"
      return 2
      ;;
  esac

  if ! test -s "${ckpt}"; then
    echo "ERROR: checkpoint not found: ${ckpt}"
    return 1
  fi

  if ss -ltn | grep -q ":${port} "; then
    echo "ERROR: port ${port} is already occupied"
    ss -ltnp | grep ":${port} " || true
    return 1
  fi

  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
  echo "GPU ${gpu} currently uses ${used} MiB"
  if test "${used}" -gt 1024; then
    echo "ERROR: GPU ${gpu} is occupied; choose another authorized free GPU"
    return 1
  fi

  mkdir -p /data/hanyu/starVLA_runs/policy_server_logs
  local log=/data/hanyu/starVLA_runs/policy_server_logs/${model}_port${port}_$(date +%Y%m%d_%H%M%S).log

  echo "MODEL=${model}"
  echo "GPU=${gpu}"
  echo "PORT=${port}"
  echo "CKPT=${ckpt}"
  echo "LOG=${log}"

  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python deployment/model_server/server_policy.py \
    --ckpt_path "${ckpt}" \
    --port "${port}" \
    --use_bf16 \
    --idle_timeout -1 2>&1 | tee "${log}"
}
```

### 11.2 启动其中一个模型

先运行 `nvidia-smi`，把下面的 `2` 换成当前获准使用且空闲的物理 GPU。

Qwen-base：

```bash
start_starvla_server qwen 2
```

Libero-init：

```bash
start_starvla_server libero 2
```

Libero + 3D prior：

```bash
start_starvla_server spatial 2
```

server 成功后应看到对应端口：

```text
server listening on 0.0.0.0:10095
server listening on 0.0.0.0:10096
server listening on 0.0.0.0:10097
```

一次公平测试可以只运行一个 server。切换模型时，在该 server 自己的终端按 `Ctrl+C`，确认 GPU 和端口释放，再启动下一模型。

---

## 12. Robot 端：统一 open-loop 离线评估代码

在 `franka` Docker 内、`lerobot`/ROS Python 环境运行。该命令不发布 `/target_pose`，不会移动机器人。

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash

run_open_loop() {
  local model="$1"
  local port

  case "${model}" in
    qwen) port=10095 ;;
    libero) port=10096 ;;
    spatial) port=10097 ;;
    *) echo "Usage: run_open_loop {qwen|libero|spatial}"; return 2 ;;
  esac

  mkdir -p /home/ros/ros2_ws/deployment_logs/open_loop
  local tag=${model}_full_stride5_$(date +%Y%m%d_%H%M%S)

  python3 /home/ros/ros2_ws/scripts/starvla_open_loop_l2_eval.py \
    --policy-host 192.168.1.113 \
    --policy-port "${port}" \
    --dataset-root /home/ros/.cache/huggingface/lerobot/snkdjn \
    --ids 0047 0077 0099 0121 0149 0150 0151 \
    --task "pick up the cube and place it on the box" \
    --stride 5 \
    --action-offset 0 \
    --max-queries-per-episode 0 \
    --compare both \
    --output-csv "/home/ros/ros2_ws/deployment_logs/open_loop/${tag}.csv" \
    2>&1 | tee "/home/ros/ros2_ws/deployment_logs/open_loop/${tag}.log"
}
```

运行一个模型：

```bash
run_open_loop qwen
```

或：

```bash
run_open_loop libero
```

或：

```bash
run_open_loop spatial
```

程序开头打印的 `Policy metadata` 必须与所选 checkpoint 一致。

---

## 13. Robot 端：统一 dry-run 代码

Dry-run 使用实时相机和实时 robot state，但没有 `--execute`，因此不发送运动命令。

```bash
run_vla_dryrun() {
  local model="$1"
  local port

  case "${model}" in
    qwen) port=10095 ;;
    libero) port=10096 ;;
    spatial) port=10097 ;;
    *) echo "Usage: run_vla_dryrun {qwen|libero|spatial}"; return 2 ;;
  esac

  mkdir -p /home/ros/ros2_ws/deployment_logs
  local log=/home/ros/ros2_ws/deployment_logs/${model}_dryrun_$(date +%Y%m%d_%H%M%S).log

  python3 /home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py \
    --policy-host 192.168.1.113 \
    --policy-port "${port}" \
    --task "pick up the cube and place it on the box" \
    --primary-image-topic /right/right_third_person_camera/color/image_raw/compressed \
    --wrist-image-topic /right/right_wrist_camera/color/image_raw/compressed \
    --compressed-image \
    --max-steps 8 \
    --execution-horizon 1 \
    --rate 5 \
    --publish-rate 40 \
    --translation-scale 1.5 \
    --max-trans-delta 0.009 \
    --max-rot-delta 0.003 \
    --min-y -0.265 \
    --max-observation-age 1.0 \
    --initial-gripper-state open \
    --gripper-switch-confirmations 3 \
    --gripper-chunk-consensus 0.75 \
    --temporal-ensemble-window 1 \
    --max-grasp-attempts 1 \
    --grasp-close-width-timeout 5.0 \
    --log-timing 2>&1 | tee "${log}"
}
```

运行：

```bash
run_vla_dryrun qwen
run_vla_dryrun libero
run_vla_dryrun spatial
```

每次只运行当前 server 对应的一条命令。

---

## 14. Robot 端：三个模型统一真机执行代码

以下函数包含 `--execute`，会真实移动 Franka。执行前必须确保：

1. 机器人从相同标准位开始；
2. primary/wrist camera placement 不变；
3. 方块与箱子按同一 protocol 摆放；
4. `/target_pose` 的 publisher count 为 0；
5. controller、`/current_pose`、两路图像和 gripper feedback 正常更新；
6. server metadata 中的 `ckpt_path` 正确；
7. 操作者在机器人旁，急停可立即触达。

该函数使用原始 `13/20` 配置，不包含后来已删除的 15 mm target tracking limiter。

```bash
run_vla_real() {
  local model="$1"
  local port

  case "${model}" in
    qwen) port=10095 ;;
    libero) port=10096 ;;
    spatial) port=10097 ;;
    *) echo "Usage: run_vla_real {qwen|libero|spatial}"; return 2 ;;
  esac

  local target_publishers
  target_publishers=$(ros2 topic info /target_pose 2>/dev/null | awk '/Publisher count:/ {print $3}')
  if test "${target_publishers:-unknown}" != "0"; then
    echo "ERROR: /target_pose publisher count is ${target_publishers:-unknown}; do not start"
    return 1
  fi

  for topic in \
    /current_pose \
    /franka_gripper/joint_states \
    /right/right_third_person_camera/color/image_raw/compressed \
    /right/right_wrist_camera/color/image_raw/compressed; do
    echo "Checking ${topic}"
    if ! timeout 5 ros2 topic echo "${topic}" --once >/dev/null 2>&1; then
      echo "ERROR: no fresh message from ${topic}"
      return 1
    fi
  done

  mkdir -p /home/ros/ros2_ws/deployment_logs
  local log=/home/ros/ros2_ws/deployment_logs/${model}_real_$(date +%Y%m%d_%H%M%S).log
  echo "REAL ROBOT EXECUTION: model=${model}, port=${port}, log=${log}"

  set -o pipefail
  python3 /home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py \
    --policy-host 192.168.1.113 \
    --policy-port "${port}" \
    --task "pick up the cube and place it on the box" \
    --primary-image-topic /right/right_third_person_camera/color/image_raw/compressed \
    --wrist-image-topic /right/right_wrist_camera/color/image_raw/compressed \
    --compressed-image \
    --max-steps 600 \
    --execution-horizon 2 \
    --rate 10 \
    --publish-rate 40 \
    --translation-scale 1.5 \
    --max-trans-delta 0.009 \
    --max-rot-delta 0.003 \
    --min-y -0.265 \
    --max-observation-age 1.0 \
    --initial-gripper-state open \
    --gripper-switch-confirmations 3 \
    --gripper-chunk-consensus 0.75 \
    --temporal-ensemble-window 1 \
    --max-grasp-attempts 1 \
    --grasp-close-width-timeout 5.0 \
    --execute \
    --log-timing 2>&1 | tee "${log}"
}
```

运行 Qwen-base：

```bash
run_vla_real qwen
```

运行 Libero-init：

```bash
run_vla_real libero
```

运行 Libero + 3D prior：

```bash
run_vla_real spatial
```

不要连续执行这三条。每个 trial 后停止、记录结果、将 Franka 返回标准位、恢复方块/箱子摆位，再开始下一个 trial。

---

## 15. 推荐的公平真机 A/B 方案

### 15.1 先比较 Libero 与 spatial

Qwen 已明显较弱，因此下一轮优先比较：

```text
A: Libero-init 74eps
B: Libero-init + DAv2 gated spatial prior
```

固定以下所有变量：

- 同一客户端 SHA；
- 相同 controller；
- 相同 rate、horizon、translation scale 和 clamps；
- 相同 gripper filters；
- 相同 camera placement；
- 相同 standard pose；
- 使用预先定义的 cube-placement bins；
- trial 顺序随机交错，而不是先跑完 A 再跑 B；
- 每次保存外部视频和 terminal log。

建议至少先做：

```text
10 matched placements × 2 models = 20 trials
```

若没有明显安全问题，再扩到每个模型 20–30 trials。

### 15.2 必须记录的字段

| 字段 | 示例 |
|---|---|
| model | libero / spatial |
| checkpoint metadata | 完整路径 |
| trial ID | A03 / B03 |
| cube placement bin | center/front/back/left/right |
| standard pose verified | yes/no |
| success | yes/no |
| alignment failure | yes/no |
| early/late close | yes/no |
| empty grasp | yes/no |
| no lift | yes/no |
| no release | yes/no |
| workspace abort | yes/no |
| stale feedback | yes/no |
| controller failure | yes/no |
| log/video path | 完整路径 |

基础设施故障必须保留在 system-level success rate 中，同时额外报告 policy-conditional success，不能事后只删除不利 trials。

---

## 16. 下一步模型诊断

如果 spatial 在恢复原客户端后仍明显落后 Libero，应按以下顺序检查：

1. 从 final checkpoint 读取 primary/wrist effective gate；
2. 对同一图像比较 Libero 与 spatial 的 action delta；
3. 对每个 placement 重复推理 10–20 次，量化 diffusion variance；
4. 保存 DAv2 relative-depth visualization，确认方块、桌面和夹爪几何是否合理；
5. 做 `gate forced to zero` 推理，确认 spatial checkpoint 在关闭 branch 时是否恢复接近 Libero 的行为；
6. 分别只启用 primary-depth 和 wrist-depth，定位哪一路造成偏差；
7. 做 held-out placement open-loop，而不是只测训练分布 episode；
8. 若仍失败，再考虑 feature-alignment、metric depth/point cloud 或更宽位置分布的数据。

最关键的 ablation 是：

```text
same spatial checkpoint, gate active
vs
same spatial checkpoint, gate forced to zero
```

它能帮助区分“spatial branch 本身造成退化”和“20k fine-tuning 过程中 baseline 其他参数发生漂移”。

---

## 17. 当前模型排名

### 按严格配对 open-loop

```text
1. Libero + 3D prior：aggregate XYZ / close precision 最好
2. Libero-init：非常接近，Y direction 和 missed-close 更好
3. Qwen-base：明显最弱
```

### 按当前真实 Franka 证据

```text
1. Libero-init：13/20，当前唯一有正式多次成功记录的模型
2. Qwen-base / spatial：都只有很小的失败样本，不能精确排序
```

Qwen 和 spatial 都观察到 `0/3` 初始失败，但失败类型不同，不能据此声称二者真实成功率完全相同。

### 项目当前默认选择

```text
Deployment baseline: Libero-init 74eps, port convention 10096
Research candidate: Libero + DAv2 gated prior, port convention 10097
Weak initialization ablation: Qwen-base, port convention 10095
```

在 spatial 完成同配置真机 A/B 并超过 baseline 前，不应替换当前 Libero-init deployment baseline。

