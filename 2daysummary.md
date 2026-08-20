# StarVLA Quest3–Franka 两日工作总结

日期：2026-07-30 至 2026-07-31  
时区：Asia/Hong_Kong  
工作区：`/home/dase-hw101/franka_ws`  
任务：`pick up the cube and place it on the box`

---

## 0. 两日结论

这两天完成了从训练、离线比较到真实 Franka 评估的一条完整闭环：

1. 修复服务器 NVIDIA 驱动版本不一致并恢复 4×H100 可用状态。
2. 完成两个使用相同 74 条双相机真机示范的 StarVLA checkpoint：
   - Qwen-base、vision frozen、随机初始化 action head、20k steps；
   - Libero-30k checkpoint 初始化、vision frozen、20k steps。
3. 对两个模型完成严格配对的 7-episode、345-query open-loop 对比。
4. 结果明确表明 Libero-init 在 Cartesian、Y 方向和夹爪时机上都优于 Qwen-base。
5. Qwen-base 真机初步测试为 `0/3`，主要问题是系统性定位偏差和提前关闭。
6. Libero-init 成为当前真实 Franka baseline。
7. 7 月 31 日实现并验证了：
   - physical gripper close latch；
   - 按绝对执行 step 对齐的 temporal action ensembling。
8. A/B 测试发现：对 XYZ/RPY 一起进行 temporal ensemble 会削弱阶段切换后的正 Z 抬升；关闭 Cartesian ensemble 后模型恢复抬升。
9. 当前推荐配置为：

   ```text
   Libero-init 74eps model
   close latch enabled
   30 mm physical lift validation enabled
   Cartesian temporal ensemble disabled (window=1)
   chunk consensus=0.75
   gripper switch confirmations=3
   ```

10. 用户在该配置下完成 20 次真机测试，结果为：

    ```text
    13/20 = 65% safety-filtered end-to-end VLA success rate
    ```

11. 这个 `65%` 不是 strict raw VLA：客户端对夹爪切换进行了通用去抖、close latch 和真实反馈验证；但客户端没有使用固定物体坐标、没有自动抬升，也没有自动决定释放。
12. 下一项最合理的部署改进是只对 gripper 做 temporal ensemble，XYZ/RPY 始终执行最新预测。该修改截至本文写入时尚未实现。

---

## 1. 固定术语和评估口径

### 1.1 Raw/pure VLA

这里的 strict raw VLA 指：模型输出直接决定 Cartesian 和夹爪切换，不使用跨推理 confirmation、close latch 或其他会改写模型夹爪命令的逻辑。

不建议直接用 strict raw VLA 做长时间真机测试，因为已有模型会出现提前 close、瞬时反向 open 和 workspace drift。

### 1.2 Safety-filtered VLA

当前真机结果属于 safety-filtered VLA：

- VLA 决定 XYZ/RPY 运动；
- VLA 决定何时开始 close；
- VLA 决定何时请求 open/release；
- 客户端对夹爪切换进行共识、连续确认和物理反馈验证；
- 客户端在抓取验证期间保持已经确认的 close；
- 客户端不会替模型生成自动抬升或自动释放动作。

### 1.3 成功 episode

本项目的 end-to-end 成功必须同时满足：

1. 从标准初始位开始；
2. 自主接近并抓住方块；
3. 自主抬升；
4. 自主搬运到箱子；
5. 模型请求释放；
6. 夹爪真实打开；
7. 方块最终留在箱子上；
8. 无人工纠正、无手动抓取或手动放置；
9. episode 中没有安全中止或控制器故障。

只抓住、只抬升或只移动到箱子上方都不是完整成功。

---

## 2. 2026-07-30：服务器、训练与模型比较

## 2.1 NVIDIA 驱动问题与修复

训练服务器此前出现内核模块与用户态 NVIDIA 库版本不匹配：

```text
loaded kernel module: 595.71.05
installed DKMS/NVML: 595.84
error: Failed to initialize NVML: Driver/library version mismatch
```

用户获得服务器重启许可后执行 reboot。重启后确认：

```text
NVRM loaded: 595.84
modinfo nvidia: 595.84
nvidia-smi driver: 595.84
CUDA reported by driver: 13.2
```

四张 H100 均重新可见。PyTorch 使用 CUDA 12.4，而驱动显示 CUDA 13.2，这是 NVIDIA 驱动向后兼容，不是错误。

### 服务器

```text
host: server1cps
SSH: hanyu@192.168.1.113
StarVLA repo: /home/hanyu/starVLA
```

### GPU

```text
GPU 0: H100, about 96 GB
GPU 1: H100 PCIe, about 80 GB
GPU 2: H100 PCIe, about 80 GB
GPU 3: H100, about 96 GB
```

不要根据 Accelerate 的逻辑 GPU 编号推断物理卡；必须同时检查 `CUDA_VISIBLE_DEVICES`、`nvidia-smi` 和进程环境。

---

## 2.2 正式 74-episode 数据集

本地数据集：

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/
quest3_franka_dualcam_pickplace_74eps
```

服务器数据集：

```text
/data/hanyu/quest3_franka_real/snkdjn/
quest3_franka_dualcam_pickplace_74eps
```

数据统计：

| 项目 | 数值 |
|---|---:|
| Episodes | 74 |
| Frames | 17,328 |
| Cameras | primary + wrist |
| Videos | 148 |
| FPS | 15 |
| Image size | 256×256 |
| Action chunk | 8 |
| Action dimension | 7 |

动作语义：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
gripper: 1=open, 0=close
```

正式训练集已经明确排除 `0036`。不要将它重新加入后续训练或评估。

74 条数据的 action gripper 本身是干净的：

- 74/74 都是一次 `open → close → open`；
- 每条恰好 2 次 gripper transition；
- action gripper 只有严格的 `0/1`；
- 没有长度小于等于 5 帧的短脉冲。

因此部署时相邻推理之间的夹爪抖动不是由示范标签逐帧乱跳直接造成，更可能来自 diffusion sampling、replanning、阶段边界不确定性和闭环 distribution shift。

---

## 2.3 Qwen-base 74eps 模型

Checkpoint：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/
final_model/pytorch_model.pt
```

训练设置：

- Qwen3-VL-4B-Instruct base；
- 不加载 Libero checkpoint；
- 不加载旧 50-episode Franka checkpoint；
- GR00T/DiT action head 随机初始化；
- 使用全部 74 条真机双相机数据；
- 冻结 visual tower；
- fresh AdamW；
- 20,000 optimization steps；
- repeated diffusion steps：8；
- action model LR：`1e-4`；
- Qwen-VL interface LR：`1e-7`；
- warmup：1,000 steps。

最终模型成功保存。审计记录显示 checkpoint 可被 `torch.load`，大小约 9.98 GB。

训练结束时报告过：

```text
epoch: about 1.15
action_dit_loss: about 0.573
mse_score: about 0.040
```

低 epoch 并不表示训练只跑了很少 step；StarVLA 的进度是按 20,000 optimizer steps 驱动，epoch 是相对于数据 loader 的换算值。

---

## 2.4 Libero-init 74eps 模型

初始化 checkpoint：

```text
/data/hanyu/starVLA_checkpoints/
libero_all_gr00t_official_30000_rerun/
final_model/pytorch_model.pt
```

最终 checkpoint：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

训练设置：

- 完整加载兼容的 Libero 30k checkpoint；
- 不加载旧 50-episode Franka checkpoint；
- fresh optimizer；
- 使用全部 74 episodes；
- 冻结 visual tower；
- 20,000 steps；
- 其余主要超参数与 Qwen-base 对照一致。

训练过程中发生的两个重要问题：

1. 在 80 GB GPU 上出现 OOM；完整配置峰值不适合该卡。
2. `retry2` 在 step 2000 保存约 8.17 GB checkpoint 时失败，因为 `/data` 只剩约 658 MB：

   ```text
   PytorchStreamWriter failed writing file
   /data: 100% used
   ```

删除不再需要的中间 checkpoints 后释放约 102 GB，再启动 `retry3`。

`retry3` 最终完成：

```text
20000/20000
wall time: about 2 h 11 min
action_dit_loss: 0.42184
mse_score: 0.02782
epoch: 1.15
```

Policy server 成功加载该 checkpoint，模型包含约 157M DiT 参数；完整框架总参数约 4.6B。

---

## 2.5 Open-loop 严格配对比较

评估 episode：

```text
0047 0077 0099 0121 0149 0150 0151
```

协议：

- 两个模型使用相同 7 个 episode；
- 345 个严格配对 observation query；
- stride 5；
- action offset 0；
- primary + wrist 双相机；
- 相同 raw 7D state；
- 相同 Franka normalization；
- 每次输出 `[8,7]` action chunk；
- 不使用 ROS，不移动机器人。

核心结果：

| Metric | Libero-init | Qwen-base |
|---|---:|---:|
| First XYZ L2 mean | 3.138 mm | 4.481 mm |
| First XYZ L2 median | 1.943 mm | 3.402 mm |
| Whole-chunk XYZ L2 mean | 3.078 mm | 4.416 mm |
| Y MAE | 1.078 mm | 1.674 mm |
| Y sign accuracy | 89.01% | 70.88% |
| Gripper accuracy | 95.07% | 86.09% |
| False-close rate | 6.05% | 14.42% |
| Missed-close rate | 3.08% | 13.08% |

首次 close 时机：

```text
Libero-init: mean 7.14 frames early ≈ 0.48 s
Qwen-base:   mean 18.57 frames early ≈ 1.24 s
```

Qwen 在 7/7 个 episode 都比示范更早 close。Libero 在 5/7 中略早、2/7 与 GT 对齐。

Y 方向偏差尤其重要：

```text
GT mean Y:     -0.000575 m/action
Libero mean Y: -0.000625 m/action
Qwen mean Y:   +0.000429 m/action
```

Qwen 的平均 Y 动作与 GT 符号相反，与真机看到的系统性侧向定位错误一致。

完整文档：

```text
deployment_logs/open_loop/
2026-07-30_libero_vs_qwen_complete_open_loop_results.md
```

权威 artifacts：

```text
deployment_logs/open_loop/
libero74_open_loop_7eps_full_stride5_20260730.csv
libero74_open_loop_7eps_full_stride5_20260730.log
qwen74_open_loop_7eps_stride5_20260730.csv
qwen74_open_loop_7eps_stride5_20260730.log
```

不要使用只有 224 queries、文件名不含 `full` 的 Libero smoke 结果作为最终比较。

---

## 2.6 真机模型选择

Qwen-base 做过 3 次初步真机测试：

```text
0/3 successful
```

现象：

- primary view 中末端明显偏到方块一侧；
- 抓取位置不稳定；
- 容易在方块上方提前关闭；
- 与 open-loop 的 Y 偏置和更早 close 一致。

因此停止继续扩大 Qwen 真机测试，将其保留为弱 initialization ablation。

Libero-init 成为当前真机 baseline。7 月 30 日较早一组未严格归档的测试约为：

```text
about 10/20 ≈ 50%
```

该早期结果已经包含 30 mm 真实抬升验证，但尚未包含 7 月 31 日新实现的 close latch。

---

## 2.7 RLinf 前置工程

7 月 30 日还完成了 RL 研究的基础工作，但没有执行真实机器人在线 RL：

- 保存 StarVLA Franka 74eps baseline commit；
- 建立 `research/rlinf-qwengroot` 分支；
- 固定 RLinf v0.3；
- 在服务器创建 no-root RLinf 环境；
- 验证 Torch、CUDA、4 GPU 和 checkpoint load；
- 设计 QwenGR00T × RLinf × ROS 2 adapter；
- 发现并修正 framework metadata 和 action unnormalization adapter 问题；
- 尚未完成 direct flow-RL 的全套 deterministic/log-prob/backward smoke；
- 尚未启动真实机器人 RL optimizer step。

详细 RL 记录见：

```text
/home/dase-hw101/franka_ws/730summary.md
```

当前更合理的 RL 对照是：

```text
Libero-init BC
vs
Libero-init BC + RL
```

不要用 `Libero BC` 对比 `Qwen BC + RL` 后把全部差异归因于 RL，因为 initialization 同时发生了变化。

---

## 3. 2026-07-31：夹爪时序、close latch 与真机消融

## 3.1 问题背景

模型在 closed-loop 真机部署中存在以下夹爪阶段问题：

- 有时提前 close；
- 已经抓住后，相邻 replanning 会短暂请求 open；
- 有时抓住但不持续抬升；
- 到达箱子上方后可能一直不 open；
- action chunk 内部和相邻 action chunk 可能出现不同阶段判断。

讨论过四类改进：

1. previous commanded gripper state 作为额外输入；
2. overlapping action chunks temporal ensembling；
3. gripper temporal-consistency training loss；
4. physical gripper close latch。

用户决定先实现第 2 和第 4 项，因为它们只涉及部署客户端，不要求重写 74 条数据或重新训练模型。

---

## 3.2 新实现：Physical gripper close latch

实现文件：

```text
/home/dase-hw101/franka_ws/scripts/
starvla_franka_delta_pose_client.py
```

同步到容器：

```text
franka:/home/ros/ros2_ws/scripts/
starvla_franka_delta_pose_client.py
```

当前文件 SHA256：

```text
ca576a2e5a263fec435785b67b124300c3f51a71ce751675b34bc9b2a8cca134
```

状态机：

```text
OPEN
  → CLOSING_VALIDATION
  → LIFT_VALIDATION
  → HOLDING_OBJECT
  → OPENING_VALIDATION
  → RELEASED
```

具体语义：

- close latch 不决定何时抓取；
- 只有模型经过共识与确认、实际触发 close 后，latch 才生效；
- 宽度和抬升验证完成前，模型的瞬时 open 请求会被记录但不会发给物理夹爪；
- 被抑制的旧 open 候选不会提前排队；
- 完成抬升验证后，必须由后续新的模型推理重新形成 open 共识；
- 无效宽度、空抓、反馈 stale 或 timeout 仍会中止。

重要：close latch 是 actuator transaction/safety state machine，不是自动任务策略。

---

## 3.3 30 mm 抬升验证不是自动抬升

当前逻辑是：

```text
StarVLA 输出正 Z action
→ 客户端执行模型 action
→ /current_pose 提供真实 Z
→ 实际 lift >= 0.030 m 才确认抓取完成
```

客户端没有执行：

```python
target_z += 0.030
```

所以它不会替 VLA 自动补足抬升。

如果模型只抬升 9 mm，客户端会报告：

```text
Abort: grasp did not reach the required lift before timeout:
lift=0.00914m < 0.03000m
```

这证明 30 mm 是验证条件，不是自动动作。

74 条训练示范的 Z 时序复核结果：

| 指标 | 结果 |
|---|---:|
| 完成 ≥30 mm 抬升 | 74/74 |
| close 到 30 mm lift 中位时间 | 2.30 s |
| close 到 30 mm lift 最慢时间 | 2.93 s |
| 5 s 时从最低点的最小 lift | 87.76 mm |

因此 5 s timeout 和 30 mm 阈值不是导致 9 mm 失败的主要原因；训练示范全部在约 3 s 内达到该阈值。

---

## 3.4 新实现：Aligned temporal action ensembling

Temporal ensemble 不是简单平均不同 chunk 的同一个数组 index。

如果旧 chunk 在 `steps_done=s` 预测：

```text
A_s[0], A_s[1], A_s[2], ...
```

执行若干 action 后，新 chunk 从 `steps_done=s+2` 开始，则：

```text
old A_s[2]
new A_(s+2)[0]
```

描述的是同一个绝对执行 step，只有这些对齐 action 才能组合。

实现参数：

```text
--temporal-ensemble-window
--temporal-ensemble-decay
--temporal-ensemble-gripper-threshold
```

首次测试设置：

```text
window=3
decay=0.8
weights=newest 1.0, older 0.8, oldest 0.64
```

初版实现对：

- XYZ：加权平均；
- RPY：加权平均；
- gripper：加权二值投票。

`window=1` 表示完全关闭 temporal ensemble。

---

## 3.5 软件验证

完成的无真机检查：

1. Python syntax/compile：PASS；
2. CLI `--help` 新参数：PASS；
3. 绝对 step 对齐单测：PASS；
4. 一次异常 open 被旧 close 抵消：PASS；
5. 持续 open 最终通过：PASS；
6. close latch 仅在已进入 close transaction 后抑制 open：PASS；
7. 本机与容器脚本 SHA256 一致：PASS；
8. 连接 Libero server `10096` 完成 6-step dry-run：PASS；
9. dry-run 没有发布 `/target_pose`，Franka 没有移动。

Dry-run 中 temporal candidate count 正确出现：

```text
first request:  1 candidate
second request: 2 candidates
third request:  3 candidates
```

客户端还新增：

```text
gripper_stability_summary
```

用于记录 raw/ensembled first-action switches、temporal chunk intent change、被 latch 抑制的 open request 和 episode completion。

---

## 3.6 真机消融：为什么完整 temporal ensemble 被关闭

使用以下组合进行首次实机测试：

```text
close latch enabled
temporal ensemble window=3
XYZ/RPY/gripper all ensembled
```

结果：

```text
Abort: grasp did not reach the required lift before timeout:
lift=0.00914m < 0.03000m
```

夹爪宽度验证已通过，close latch 正常保持 close，但真实 Z 只增加约 9.14 mm。

随后进行单变量 A/B：

```text
keep close latch
set temporal ensemble window=1
all other deployment parameters unchanged
```

结果：模型恢复了抓取后的抬升，并能移动到箱子区域。

因此当前最合理的解释是：

- 旧 chunk 中的下降/停留动作与新 chunk 中的正 Z lift action 被平均；
- 阶段从 grasp 转为 lift 时，Cartesian temporal ensemble 引入滞后；
- close latch 不是无法抬升的原因。

当前决定：

```text
Cartesian temporal ensemble disabled: window=1
close latch retained
```

下一步可实现：

```text
XYZ/RPY: always newest chunk
gripper: aligned temporal ensemble only
```

该 gripper-only 模式尚未实现，不应在报告中写成已经使用。

---

## 3.7 夹爪反馈 stale 中止

一次成功抓取和抬升后的运行出现：

```text
RuntimeError: Abort: gripper feedback became stale while validating grasp:
age=1.976s
```

含义：

- 客户端约 1.976 s 没收到新的 `/franka_gripper/joint_states`；
- 当前 `--max-observation-age` 为 1.0 s；
- 客户端不能继续相信旧的夹爪宽度，所以安全中止；
- 这不是模型不会抬升，也不表示抓取宽度一定错误。

中止后重新检查 topic：

```text
Publisher count: 1
rate: about 15 Hz
current width after reset: about 0.080 m
```

所以反馈目前正常，故障更像一次 transient pause、gripper action blocking 或 DDS delivery delay，而不是 publisher 永久死亡。

正确的后续代码改进应是增加独立阈值：

```text
camera/current_pose max age: 1.0 s
gripper feedback max age: 2.5–3.0 s
```

在短暂 gripper feedback gap 中继续保持已经确认的 close 并等待新反馈；超过独立阈值才中止。该独立参数尚未实现。

不建议仅把所有 observation 的全局 age 一起放宽到 3 s，因为陈旧相机和陈旧 Cartesian pose 的风险更高。

---

## 3.8 最新 20-trial 真机结果

用户在当前推荐部署组合下完成 20 次测试：

```text
success: 13
failure: 7
success rate: 13/20 = 65%
```

配置口径：

```text
model: Libero-init StarVLA trained on 74 real episodes
close latch: enabled
physical lift validation: 30 mm
Cartesian temporal ensemble: disabled, window=1
chunk consensus: 0.75
gripper switch confirmations: 3
max grasp attempts: 1
```

应报告为：

```text
Safety-filtered end-to-end VLA success rate: 13/20 = 65%
```

不能报告为 strict raw/pure VLA success rate。

相较较早约 `10/20 = 50%` 的未严格归档测试：

```text
absolute improvement: about +15 percentage points
failures: 10 → 7
```

这是积极但仍属 preliminary 的结果。`13/20` 的 Wilson 95% confidence interval 约为：

```text
43%–82%
```

20 次样本不足以统计上证明它一定优于 50%。应固定配置并扩大评估，而不是继续同时改变多个参数。

目前还缺少 7 次失败的严格分类。后续必须至少记录：

| Failure type | 说明 |
|---|---|
| Alignment | 末端未对准方块 |
| Early/late close | 夹爪时机错误 |
| Empty grasp | 宽度显示空抓 |
| No lift | 抓住但未达到 30 mm |
| No release | 到箱子上方但模型不 open |
| Workspace abort | XYZ target 超出限制 |
| Stale feedback | camera/pose/gripper 数据过期 |
| Controller failure | controller/node unexpectedly stopped |

建议同时报告两个指标：

1. system-level end-to-end success：所有 infrastructure failure 都计失败；
2. policy-conditional success：单独说明排除明确基础设施故障后的结果。

主结果必须仍使用第 1 个指标。

---

## 4. 当前安全过滤的精确定义

最新 `13/20` 使用了以下通用过滤：

### 4.1 Gripper chunk consensus

一个 8-action chunk 至少 75% 表达相同 open/close 意图，才形成候选：

```text
--gripper-chunk-consensus 0.75
```

### 4.2 Consecutive request confirmation

连续 3 次 policy inference 形成相同切换候选，客户端才真正改变夹爪状态：

```text
--gripper-switch-confirmations 3
```

### 4.3 Close latch

一旦模型已经确认 close：

- 保持 close；
- 等待真实宽度验证；
- 等待模型自己产生真实 30 mm lift；
- 在验证完成前抑制短暂 open；
- 验证完成后要求新的模型 open 共识。

只有加入以下参数才会关闭：

```text
--disable-gripper-close-latch
```

### 4.4 Physical grasp validation

```text
valid total finger width: 0.022–0.036 m
stable duration: 0.25 s
empty threshold: 0.015 m
required measured lift: 0.030 m
max physical grasp attempts: 1
```

### 4.5 Physical release validation

模型请求 open 后：

- 冻结当前 Cartesian target；
- 持续发送模型已经决定的 open；
- 等待真实夹爪总宽度至少 0.060 m；
- 宽度稳定 0.25 s 后判定物理释放完成。

### 4.6 Cartesian/general safety

- XYZ workspace bounds；
- raw translation/rotation action limits；
- per-step translation/rotation clamp；
- stale observation abort；
- existing `/target_pose` publisher conflict check；
- continuous held-command heartbeat；
- controller subscriber readiness check。

### 4.7 明确没有使用的 object-specific logic

当前没有：

- 基于成功示范固定 XYZ 范围的 close gate；
- 客户端自动 +30 mm 抬升；
- 客户端自动选择箱子位置；
- 客户端自动决定 release；
- 抓取失败后自动多次 retry；
- 根据已知方块坐标做 scripted pick-and-place。

因此它是 safety-filtered VLA，而不是 scripted pick-and-place。

---

## 5. 当前 checkpoint、server 与文件

### 5.1 当前推荐 checkpoint

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

### 5.2 当前弱基线 checkpoint

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_qwen_vision_frozen_20k/
final_model/pytorch_model.pt
```

### 5.3 Policy ports

截至 2026-07-31 最后核验：

```text
192.168.1.113:10096 open  → Libero-init 74eps
192.168.1.113:10095 closed → Qwen-base server not running
```

每次部署前仍必须通过 server metadata 核对 `ckpt_path`，不能只相信端口号。

### 5.4 本地关键文件

```text
scripts/starvla_franka_delta_pose_client.py
scripts/franka_return_to_standard.py
scripts/starvla_open_loop_l2_eval.py
deployment_logs/open_loop/2026-07-30_libero_vs_qwen_complete_open_loop_results.md
730summary.md
STARVLA_HANDOVER_2026-07-30.md
```

### 5.5 当前机器人现场状态

截至本文最后一次只读检查：

```text
current position:
x=0.30657
y=-0.00075
z=0.58186

gripper finger positions:
[0.0399976, 0.0399976]
total width ≈ 0.079995 m, open

/target_pose publisher count: 0
/target_pose subscription count: 1
/franka_gripper/joint_states publisher count: 1
```

这表示机器人位于标准高位附近、夹爪打开、当前没有部署客户端占用 `/target_pose`。

---

## 6. 当前推荐部署命令

以下命令会移动真实 Franka。执行前必须：

- 确认 policy metadata 是 Libero retry3；
- 确认双相机 topic 更新；
- 确认 `/current_pose` 和 gripper feedback 更新；
- 确认 `/target_pose` publisher count 为 0；
- 将方块和箱子按评估协议摆放；
- 人手保持在急停附近。

```bash
docker exec -it franka bash -lc '
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash

python3 /home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py \
  --policy-host 192.168.1.113 \
  --policy-port 10096 \
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
  --log-timing
'
```

注意：`--temporal-ensemble-window 1` 是当前经过 A/B 后的推荐值；不要恢复为 3 后仍宣称与 `13/20` 使用相同配置。

---

## 7. 下一步建议

### Priority 1：冻结并正式记录当前 65% baseline

下一批测试不要同时改模型、速度和过滤逻辑。固定：

- Libero retry3 checkpoint；
- camera placement；
- standard start pose；
- cube/box placement protocol；
- rate、scale、horizon；
- confirmations=3；
- consensus=0.75；
- close latch enabled；
- temporal ensemble window=1。

为每次 trial 保存：

- trial ID；
- cube/box initial coordinates or placement class；
- terminal log；
- external video；
- success/failure；
- failure category；
- whether an infrastructure fault occurred。

### Priority 2：实现 gripper-only temporal ensemble

建议新增独立模式：

```text
XYZ/RPY = newest policy chunk
gripper = aligned temporal ensemble
```

然后做严格 A/B：

```text
A: current baseline, window=1
B: gripper-only ensemble, window=3, decay=0.8
```

其他参数不变。每组至少先做 20 trials，并使用匹配或随机化的 cube placements。

### Priority 3：独立 gripper feedback stale threshold

新增：

```text
--max-gripper-feedback-age 2.5
```

相机和 pose 继续使用 1.0 s。先进行无真机、mock feedback gap 测试，再做短真机验证。

### Priority 4：失败定向补数据

若 7 次失败主要是定位和 close timing，收集额外 30–50 条针对性示范：

- 方块覆盖更宽 XY 分布；
- 不同 yaw/轻微歪放；
- 成功抓取中心和边缘情况；
- 明确、稳定的 close→lift→transport→open 阶段；
- 保持新腕部相机无遮挡。

不要只重复完全相同摆位。对 DAgger-style correction episode 单独标记来源。

### Priority 5：模型级时序模块

客户端实验稳定后再考虑：

1. previous commanded gripper state 作为额外输入；
2. gripper temporal-consistency loss；
3. state/no-state ablation；
4. 3D spatial prior；
5. RLinf residual RL 或 direct flow RL。

previous-command input 会把 state 从 7D 改为 8D，需要更新 dataset loader、normalization、state encoder、server metadata、client 和 evaluator，但不要求重新采集数据；可以从已有 action.gripper 向后平移一帧动态生成。

### Priority 6：RL 因果实验

建议保持：

```text
A. Libero → BC74
B. Libero → BC74 → RL
```

如果增加数据，再单独做：

```text
C. Libero → BC124
D. Libero → BC124 → RL
```

不能同时增加数据和加入 RL，然后把全部提升归因于 RL。

---

## 8. 禁止混淆的关键结论

1. `30 mm lift validation` 不是自动抬升。
2. `close latch` 会保持已经确认的 close，但不会决定抓取位置。
3. `13/20` 使用 close latch 和 30 mm validation。
4. `13/20` 没有使用 Cartesian temporal ensemble；`window=1`。
5. 当前完整 temporal ensemble 的代码仍存在，但实机推荐关闭。
6. gripper-only temporal ensemble 尚未实现。
7. object-specific XYZ close gate 已移除。
8. 客户端没有 scripted cube/box coordinates。
9. stale gripper feedback 是 infrastructure/safety abort，不等同于模型动作错误。
10. open-loop L2 只能支持相对模型比较，不能等同于真机成功率。
11. Qwen-base 的 0/3 和 Libero 的 13/20 不应被解释为只由训练 loss 决定；初始化 prior、closed-loop drift 和安全过滤都有影响。
12. 当前最可信的正式真机数字是 `13/20 = 65% safety-filtered end-to-end success`，但仍需更大样本和完整失败分类。

---

## 9. 一句话交接

截至 2026-07-31，Libero-30k 初始化并用 74 条双相机真机数据训练 20k steps 的 StarVLA 是当前最佳 baseline；在 close latch、30 mm 真实抬升验证、75% chunk consensus、3 次连续确认且关闭 Cartesian temporal ensemble 的配置下，用户报告真实 Franka `13/20 = 65%` 成功率。下一步应先正式记录失败类型，再实现 gripper-only temporal ensemble 和独立 gripper feedback stale threshold，之后才进入模型级 3D/RL 改进。
