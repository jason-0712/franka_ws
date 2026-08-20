# StarVLA RL 工作总结（2026-07-30）

## 1. 今日目标

今天的目标是为现有 StarVLA QwenGR00T Franka 模型增加强化学习能力，先完成 RL 前置工程，而不是直接启动真机在线训练：

1. 保存 dirty StarVLA baseline。
2. 生成可追溯 baseline manifest。
3. 验证本地 Franka 数据集与模型接口。
4. 建立独立 RLinf 环境。
5. 设计 QwenGR00T × RLinf × ROS 2 adapter。
6. 在 4×H100 服务器验证 checkpoint 和 RLinf QwenGR00T flow-RL 接口。

截至今天结束，没有启动 Ray、policy server、ROS 节点、真实机器人动作或 RL optimizer step。

## 2. Baseline 固化

### 2.1 为什么原 baseline 是 dirty

原 StarVLA 工作树包含尚未提交、但实际参与训练/部署的修改，例如：

- Franka 74-episode 数据注册与训练配置。
- Qwen-only / LIBERO-resumed launcher。
- 部署端 normalization 和 policy wrapper 修改。
- 双相机、Franka action/state registry 修改。
- `action_mode=abs` 修正，避免对已经是 delta 的动作再次差分。

因此 dirty 并不等于代码错误，而是“训练所依赖的真实修改还没有形成 Git 快照”。如果直接开始 RL，之后将无法区分 SFT baseline 改动与 RL 改动。

### 2.2 已保存的 Git 状态

StarVLA 仓库：

```text
/home/dase-hw101/franka_ws/third_party/starVLA
```

保存的 baseline：

```text
branch: baseline/starvla-franka-74eps-20260730
commit: d0b3282ed90685ca5c09c57b97dda56c1745ca05
message: chore: snapshot Franka 74-episode StarVLA baseline
```

RL 开发分支：

```text
branch: research/rlinf-qwengroot
```

今日 RL 分支提交：

```text
55fffc8 docs: define QwenGR00T RL integration baseline
aa2ad02 fix: support no-root RLinf server bootstrap
f018447 test: add strict QwenGR00T RL server smoke
c1068e2 fix: attach StarVLA framework metadata in smoke
1cff0f8 fix: bridge RLinf to training-time action normalization
```

当前 nested StarVLA Git 工作区干净。

## 3. 主模型与 LIBERO checkpoint 的决定

当前主 RL initialization 固定为已经完成的 Qwen-only run：

```text
Qwen3-VL-4B-Instruct initialization
+ Franka dual-camera 74 episodes
+ vision tower frozen
+ fresh AdamW
+ randomly initialized GR00T/DiT action head
+ 20,000 training steps
```

Run：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k
```

Checkpoint：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/final_model/pytorch_model.pt
```

尚未完成的 LIBERO-resumed run 不阻塞当前 RL 工作，也不应在主实验中途替换 Qwen-only initialization。它只在未来研究以下问题时需要完成：

```text
Qwen-only initialization vs LIBERO initialization
```

核心因果比较只需要：

| Initialization | Post-training | 作用 |
|---|---|---|
| Qwen-only 20k | none | 固定 SFT baseline |
| Qwen-only 20k | RL | 核心 RL 增益 |

LIBERO 两组可以作为后续可选 initialization ablation。

## 4. 数据集验证结果

数据集：

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/quest3_franka_dualcam_pickplace_74eps
```

服务器路径：

```text
/data/hanyu/quest3_franka_real/snkdjn/quest3_franka_dualcam_pickplace_74eps
```

完整验证结果：

- 74 episodes。
- 17,328 Parquet rows。
- 148 videos，两路相机各 74 个。
- 15 FPS。
- 256×256 RGB。
- AV1 codec。
- 148/148 视频帧数与对应 Parquet 行数一致。
- action gripper 值为 `{0, 1}`。
- state gripper 实测范围约为 `[0, 0.6552254]`。
- 数据与接口 validator 状态为 `PASS`。

Metadata hashes：

```text
info.json:
1865c4a0f5e197c964e8b36f22b2b7c7ac35247b89fa8fd5cb7a510c304f181a

modality.json:
5a5305a94cbb9c13c22d1cba1223f4c60fe1719cbbb8efb31030f18be4a34f36

merge_manifest.json:
70ef2d554395cfa227223f42bcd4dfe83cad8b97d3488344d6c1f7a59508bb63
```

已知 metadata warning：嵌套字段 `video_info.video.fps` 写成 30，但 top-level info、timestamps、ffprobe 都确认真实 canonical FPS 是 15。

## 5. 模型与动作契约

### 5.1 动作

QwenGR00T 输出：

```text
shape: [B, 8, 7]
action: [dx, dy, dz, droll, dpitch, dyaw, gripper_command]
```

关键语义：

- Cartesian 6D 是 Franka base frame 下的 EEF delta。
- gripper action 是绝对命令：`1=open, 0=close`。
- 真实部署每次只执行 action chunk 的第一个 action，然后重新观察和规划。
- Parquet action 已经是 delta，所以 loader 必须保持 `action_mode=abs`；若改成 `delta`，会再次差分并破坏动作与 normalization statistics。

### 5.2 State 的重要修正

数据记录的 state 是：

```text
[x, y, z, roll, pitch, yaw, measured_gripper_closed_amount]
```

虽然 action head config 保留：

```text
state_dim=7
```

但本次实际训练配置为：

```text
datasets.vla_data.include_state=false
```

因此该 Qwen-only checkpoint 的 base QwenGR00T policy **没有消费 proprio state**。RLinf 当前必须配置：

```text
enable_state_input=false
```

否则会启用训练时没有使用的 state conditioning，产生不可解释行为。

ROS/replay 仍必须记录 state，因为它用于：

- 安全检查。
- transition/replay。
- residual actor/critic。
- 后续 `include_state=true` ablation。

state gripper 与 action gripper 不是同一个量：state 是测量闭合量，action 是 open/close 命令。

## 6. 已生成文件

```text
examples/realRobots/Franka/rl/baseline_manifest.yaml
examples/realRobots/Franka/rl/validate_baseline.py
examples/realRobots/Franka/rl/QWENGROOT_ROS2_RL_ADAPTER_DESIGN.md
examples/realRobots/Franka/rl/bootstrap_rlinf_server.sh
examples/realRobots/Franka/rl/smoke_qwengroot_server.py
examples/realRobots/Franka/rl/run_qwengroot_server_smoke.sh
```

设计文档包含：

- observation/action contract。
- normalization exactly-once invariant。
- ROS 2 observe/shadow/execute 状态机。
- intervention、reward、transition schema。
- safety gates。
- residual RLPD/SAC 与 direct flow-GRPO 两条路线。
- 4×H100 初始资源布局建议。
- LIBERO checkpoint 的实验角色。

## 7. RLinf 环境

固定版本：

```text
tag: v0.3
commit: 0505431899574619da86f551bad70b71e0ea2177
```

本地轻量环境：

```text
/home/dase-hw101/franka_ws/.venvs/rlinf-v0.3-dev
```

本地环境仅用于源码、PyArrow、YAML 和数据检查，没有安装 Torch/Ray/GPU runtime，因为工作站磁盘空间不足。

服务器完整环境：

```text
source: /home/hanyu/RLinf-v0.3
venv:  /home/hanyu/RLinf-v0.3/.venv
```

服务器 bootstrap 最初因为官方安装器尝试系统依赖安装而要求 sudo。随后修正为官方 `--no-root` 路径，并安全续装已有 Python 3.11 partial environment。

已确认：

```text
Python: 3.11.14
Torch: 2.6.0+cu124
Torch CUDA: 12.4
CUDA available: true
Visible GPUs: 4
RLinf import: PASS
StarVLA module discovery: PASS
Checkpoint torch.load: PASS
```

NVIDIA driver 显示 CUDA 13.2，而 PyTorch wheel 使用 CUDA 12.4。这是驱动向后兼容，不是错误。

## 8. Checkpoint 与训练日志验收

审计报告：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/server_audit_20260730_085627.txt
```

服务器原路径：

```text
/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/rl_bootstrap_audit/server_audit_20260730_085627.txt
```

结果：

- checkpoint size：9,976,833,554 bytes。
- checkpoint payload：dict。
- top-level state-dict keys：962。
- `torch.load`：PASS。
- 训练完成 20,000 steps。
- final model 保存成功。
- training log error scan 没有发现 traceback、OOM、NaN、fatal 或 NCCL error。

Checkpoint SHA256 尚未回填。第二次审计为了避免重复读取 9.97GB 文件使用了 `HASH_CHECKPOINT=0`；第一次报告 `server_audit_20260730_082641.txt` 仍需复制并提取 hash。

## 9. 服务器 provenance 风险

审计时服务器训练仓库状态：

```text
branch: starVLA_dev
commit: e8f8fbb1a60f521b6075a63258d898ae987f02a0
dirty: true
```

它与工作站 handover baseline 的父 commit `2e5f239...` 不同，而且服务器存在 tracked/untracked 修改。

这不代表 checkpoint 无效，但表示：

- checkpoint 的实际训练实现应以服务器 `/home/hanyu/starVLA` 为准。
- 当前不能声称完整源码级 reproducibility。
- 在服务器源码继续变化前，应归档 server Git diff、关键源文件 hash 和相关 untracked training files。
- strict smoke 必须使用服务器实际 StarVLA，而不是直接用工作站版本替代。

## 10. GPU 与磁盘状态

服务器可见 4 张 H100。审计时：

- GPU 0：约 9.7GB 已用，利用率快照 0%。
- GPU 1：约 16.4GB 已用，利用率约 64%，应避开。
- GPU 2：约 9.6GB 已用，利用率快照 0%。
- GPU 3：约 2.9GB 已用，利用率快照 0%。

四张卡均有已有进程。选择 GPU 前必须确认 PID 所有者和用途，不自动终止或抢占进程。

磁盘审计：

- root filesystem 约剩 40GB。
- `/data` 约剩 55GB，使用率因取整显示 100%。

这足够做单次 smoke，但不足以无规划地保存大量 online replay、视频和频繁 checkpoint。正式 RL 前必须制定 retention 和存储路径。

## 11. Server smoke 进展与发现

Smoke 目标：

1. 读取保存的 config/statistics。
2. 完整实例化 QwenGR00T。
3. strict checkpoint load。
4. 解码 episode 0 primary/wrist observation。
5. native StarVLA `[1,8,7]` 推理。
6. RLinf deterministic rollout 与 native parity。
7. stochastic flow log-prob rollout/replay parity。
8. action head backward，检查有限非零梯度。
9. 不执行 optimizer step。

### 11.1 第一次兼容性错误：framework metadata

错误：

```text
Unable to infer VLM type for starVLA model
framework_name=None
```

原因：smoke 为避免重复加载 9.97GB checkpoint，直接用严格加载后的 model 构造 RLinf wrapper，但漏掉了 RLinf 公共 `get_model()` 会设置的：

```python
model.framework_name = "QwenGR00T"
```

已在 commit `c1068e2` 修复。

### 11.2 第二次兼容性错误：action unnormalization API

错误：

```text
type object 'baseframework' has no attribute 'unnormalize_actions'
```

原因：

```text
RLinf v0.3 假设：baseframework.unnormalize_actions()
服务器训练 StarVLA 实际接口：PolicyNormProcessor.unapply_actions()
```

这是真实的 RLinf × dirty StarVLA 版本兼容问题，不是 checkpoint 损坏。

已在 smoke 中加入 process-local compatibility shim：

- 不修改服务器 StarVLA/RLinf 源码。
- 复用部署端 `PolicyNormProcessor`。
- normalized action 先 clip 到 `[-1,1]`。
- 使用训练时的 composed transform 做反归一化。
- 保持 Franka gripper 语义。
- 在 smoke 报告中明确记录 shim。

修复 commit：`1cff0f8`。

截至本文档写入时，`1cff0f8` 修复后的 smoke 尚待服务器重跑，因此 deterministic parity、log-prob replay 和 backward 仍未最终 PASS。

## 12. RL 总体路线

### 12.1 近期路线

1. 完成 strict QwenGR00T/RLinf smoke。
2. 归档服务器训练源码 diff/hash。
3. 固定 checkpoint SHA256。
4. 实现 ROS 2 adapter 的 `observe` 模式。
5. 实现 `shadow` inference，禁止发布 `/target_pose`。
6. 固定真实机器人 baseline evaluation protocol。
7. 建立 intervention、reward、replay 数据闭环。

### 12.2 首个真机 RL 建议

首个真机 RL 推荐 residual RLPD/SAC：

- 冻结 4B QwenGR00T base policy。
- residual 初始输出为零。
- 首版只学习 6D Cartesian residual。
- gripper 继续使用 base policy，并加 hysteresis/debounce。
- demo buffer 使用已有 74 episodes。
- online buffer 使用 human-gated rollout。
- residual 有严格平移/旋转限幅，可随时置零回到原始 SFT baseline。

### 12.3 Direct QwenGR00T GRPO

RLinf v0.3 已实现实验性 flowmatching RL wrapper：

- 缓存 flow denoising chain。
- 在一个随机 denoise step 注入 Gaussian transition。
- rollout 保存 old log-prob。
- training replay 重算 log-prob/entropy。

但 RLinf 源码自己标注该 flow path 尚需完整端到端验证。必须先通过：

- native/RLinf deterministic parity。
- rollout/replay log-prob parity。
- gradient finite/non-zero。
- probability ratio 初始约为 1。
- normalization exactly once。
- 离线短优化和仿真/digital-twin gate。

这些通过前，不将 direct GRPO 连接到真实机器人 execute。

## 13. 下一步操作

将最新 smoke 覆盖到服务器：

```bash
cd /home/dase-hw101/franka_ws/third_party/starVLA

scp examples/realRobots/Franka/rl/smoke_qwengroot_server.py \
  hanyu@192.168.1.113:/home/hanyu/smoke_qwengroot_server.py
```

选择已获准使用的 GPU 后重跑，例如 GPU 3：

```bash
ssh -t hanyu@192.168.1.113 \
  'GPU_ID=3 bash /home/hanyu/run_qwengroot_server_smoke.sh'
```

最新代码会继续检查：

- `[1,8,7]` 输出。
- native/RLinf deterministic parity。
- normalized action 是否超出 `[-1,1]`。
- training-time normalization compatibility shim。
- stochastic flow log-prob parity。
- action-head backward。

还需复制第一次审计报告以获得 checkpoint SHA256：

```bash
scp hanyu@192.168.1.113:/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/rl_bootstrap_audit/server_audit_20260730_082641.txt \
  /home/dase-hw101/franka_ws/artifacts/rlinf_audits/
```

## 14. 当前状态一句话总结

已经把 Qwen-only Franka-74eps StarVLA baseline、数据契约、RLinf 环境和 ROS 2/RL 设计固定下来，并确认 checkpoint 与完整 GPU runtime 可用；当前正在解决服务器 dirty StarVLA 与 RLinf v0.3 的两个明确 adapter 兼容点，最新 normalization shim 待重跑验证，尚未进入真实机器人 RL。
