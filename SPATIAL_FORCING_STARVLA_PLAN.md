# Spatial Forcing × StarVLA 复现与真机验证计划

日期：2026-08-04  
工作区：`/home/dase-hw101/franka_ws`  
服务器代码目录：`/home/hanyu/starVLA`  
任务：`pick up the cube and place it on the box`  
机器人：Franka FR3，primary + wrist 双 RGB 相机

## 1. 决策摘要

下一条研究路线是把 Spatial Forcing 的核心训练方法适配到 StarVLA/QwenGR00T，而不是继续扩大当前 DAv2 在线深度分支。

核心原则：

```text
训练时：RGB → StarVLA student
        RGB → frozen VGGT teacher
        student 中间视觉特征与 VGGT 空间特征对齐

部署时：RGB → StarVLA student → action
        VGGT、alignment head 全部删除
```

这项工作应称为：

> Reproduction and adaptation of Spatial Forcing to StarVLA/QwenGR00T.

它不是对论文 OpenVLA/π0 代码路径的逐行复现，但目标是忠实复现论文的核心算法：冻结 VGGT、对齐中间视觉表示、联合 action/alignment loss、部署时移除 teacher。

论文与官方代码：

- Spatial Forcing paper: <https://arxiv.org/abs/2510.12276>
- Spatial Forcing official code: <https://github.com/OpenHelix-Team/Spatial-Forcing>
- VGGT official code: <https://github.com/facebookresearch/vggt>

## 2. 当前基线与研究问题

### 2.1 数据

```text
dataset:
  /data/hanyu/quest3_franka_real/snkdjn/
  quest3_franka_dualcam_pickplace_74eps

episodes: 74
views: primary RGB + wrist RGB
FPS: 15
excluded: episode 0036
```

现有数据不需要加入 depth、point cloud 或相机标定文件。Spatial Forcing teacher 直接从训练 RGB 生成空间监督。

### 2.2 Student 初始化

使用当前表现最好的 Libero-init 74-episode checkpoint：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

该模型当前最可信的真机结果为：

```text
13/20 = 65% safety-filtered end-to-end success
```

端口约定：

```text
10096: Libero74 baseline
10097: 已有 DAv2 gated model
10098: 建议留给 Spatial Forcing model
```

端口不是模型身份。所有测试必须同时记录 checkpoint path 和 server metadata。

### 2.3 主要研究问题

在保持相同的 74 条数据、动作表示、训练步数和真机客户端配置时：

> 训练期 VGGT feature alignment 是否能让 StarVLA 获得更好的空间 grounding，并把提升转化为真实 closed-loop 成功率？

## 3. 为什么不继续把 DAv2 当在线 policy 分支

当前 `QwenGR00TSpatial` 的结构是：

```text
RGB → frozen DAv2 → relative-depth tokens
                    ↓
            gated cross-attention
                    ↓
              action policy
```

这意味着部署时仍需要 DAv2。它只有相对深度，没有 metric scale、相机外参或 robot-frame geometry。当前 checkpoint 虽然 open-loop 略有改善，但最初真机测试没有超过 Libero74 baseline。

Spatial Forcing 的差异是：

```text
VGGT 只提供训练监督，不是部署输入。
部署架构恢复为普通 RGB StarVLA。
```

因此它更适合没有 depth 的可扩展机器人数据，也避免在线 DAv2 的额外延迟和域偏差。

## 4. 目标训练管线

### 4.1 Teacher branch

同一时刻的 primary/wrist 图像一起送入冻结的 VGGT：

```text
I = [I_primary, I_wrist]
Z_teacher = VGGT(I)
```

使用 VGGT Transformer backbone 的 pixel-level latent spatial representations，不把最终 depth map 当作 policy 输入。

要求：

- `VGGT.eval()`；
- 所有 VGGT 参数 `requires_grad=False`；
- teacher forward 使用 `torch.no_grad()`；
- teacher 输出必须 `detach()`；
- teacher 不进入部署 checkpoint；
- primary/wrist 顺序在所有数据中固定；
- teacher/student 必须看到相同裁剪内容，避免空间错位。

### 4.2 Student branch

Student 保持现有 QwenGR00T 动作结构：

```text
primary/wrist RGB + task instruction
              ↓
       QwenGR00T visual tokens
              ↓
        GR00T/DiT action head
              ↓
          8 × 7 action chunk
```

从 Qwen 的中间层取得 image-token hidden states：

```text
Z_student^(l) = hidden_states[l][image_token_mask]
```

应选择“相对较深但不是最后一层”的候选层。第一阶段至少比较三个位置：

```text
25% depth
50% depth
75% depth
```

不要一开始同时对齐许多层，以免实验变量过多。

### 4.3 Token correspondence

Qwen image tokens 和 VGGT patch tokens 数量通常不同。必须按照每个 view 的二维网格进行对应：

1. 从 Qwen input 的 `image_grid_thw` 恢复每个相机的 token grid；
2. 将 VGGT feature map 保持为 `[H_t, W_t, D_t]`；
3. 使用二维插值映射到 Qwen 的 `[H_s, W_s]`；
4. 分别对 primary/wrist 对齐；
5. 加入二维位置编码和 view embedding；
6. 只对有效 image tokens 计算 loss。

禁止把两个不等长的 flattened token sequence 直接截断后对齐，因为这会破坏像素位置对应。

### 4.4 Alignment head and loss

Student features 经过 normalization 和两层 MLP。论文公式把归一化写作
`Gamma`；官方 OpenVLA/PI0 PyTorch 实现实际使用可选的 `LayerNorm`
（`use_vlm_norm=True`），不是 PyTorch `BatchNorm`：

```text
Z_projected = MLP(LayerNorm(Z_student))
```

对齐损失：

```text
L_align = mean(1 - cosine_similarity(Z_projected, Z_teacher + E_pos))
```

总损失：

```text
L_total = L_action + alpha * L_align
```

论文默认 `alpha=0.5`，但 StarVLA 的 action-loss 尺度不同。正式 20k 前先比较：

```text
alpha = 0.00  control
alpha = 0.02
alpha = 0.10
alpha = 0.50
```

如果 alignment loss 明显主导总梯度，应使用 warmup/ramp，而不是直接提高 alpha。

## 5. 最关键的训练约束

### 5.1 Student 表示必须能够被更新

如果产生 `Z_student` 的全部 Qwen 层被冻结，alignment MLP 会独自拟合 teacher，而 StarVLA 内部表示不会获得 3D knowledge。这不构成有效 Spatial Forcing。

最低要求：

- alignment MLP trainable；
- action head trainable；
- selected Qwen layers 或其 LoRA/adapters trainable；
- audit 实际 `requires_grad`，不能只看配置中的 learning-rate 名称；
- VGGT frozen。

推荐先用 LoRA/adapters 解冻 selected Qwen blocks，避免再次出现 full-model optimizer OOM。

### 5.2 Teacher 不进入部署图

建议实现训练 wrapper：

```text
SpatialForcingTrainingWrapper
├── student: Qwen_GR00T
├── teacher: FrozenVGGT
└── alignment_head
```

训练 checkpoint 可以保存恢复训练所需的 alignment head；部署 export 只保存 `student.state_dict()` 和普通 `QwenGR00T` config。policy server 不应加载 VGGT。

## 6. 预计代码改动

建议新增，而不是覆盖当前 baseline/DAv2 文件：

```text
third_party/starVLA/starVLA/model/modules/spatial_forcing/
  __init__.py
  vggt_teacher.py
  token_alignment.py
  alignment_head.py

third_party/starVLA/starVLA/model/framework/VLM4A/
  QwenGR00TSpatialForcing.py

third_party/starVLA/tests/
  test_qwengroot_spatial_forcing.py

third_party/starVLA/examples/realRobots/Franka/train_files/
  starvla_cotrain_quest3_franka_delta_eef_spatial_forcing.yaml

scripts/
  run_qwengroot_spatial_forcing_tests_and_smoke.sh
  export_qwengroot_spatial_forcing_student.py
```

需要修改的现有训练逻辑：

```text
third_party/starVLA/starVLA/training/train_starvla.py
```

当前 `_train_step()` 只读取 `action_loss`。需要支持：

```python
action_loss = output_dict["action_loss"]
align_loss = output_dict.get("alignment_loss", 0.0)
total_loss = action_loss + alpha * align_loss
```

并记录：

```text
action_dit_loss
spatial_alignment_loss
spatial_alignment_cosine
spatial_alpha
student_alignment_grad_norm
alignment_head_grad_norm
teacher_grad_count          # 必须为 0
```

## 7. 分阶段实施与停止条件

### Phase 0：锁定公平基线

保留：

- Libero74 checkpoint；
- 7-episode、345-query open-loop artifacts；
- 13/20 真机客户端版本与参数；
- 当前客户端 SHA/命令；
- 训练数据 manifest。

任何模型比较都不得临时改变控制速度、workspace、gripper confirmations 或 close latch。

### Phase 1：官方方法核对

阅读并记录官方 OpenVLA/π0 实现中的：

- VGGT feature extraction layer；
- student alignment layer；
- positional embedding；
- token resize/matching；
- alpha；
- frozen/trainable modules；
- deployment export。

以官方代码为方法权威来源，论文为数学定义来源。

### Phase 2：CPU/GPU 单元测试

必须全部通过：

1. primary/wrist view 顺序测试；
2. token grid 对应测试；
3. teacher output shape/finite test；
4. teacher parameter gradients 全为 `None`；
5. alignment head 有非零梯度；
6. selected student layer 有非零梯度；
7. `alpha=0` 时 total loss 等于 action loss；
8. deploy export 不包含 `teacher.*` 或 `alignment_head.*`；
9. 导出的 student 能由原 `QwenGR00T` policy server 加载；
10. 相同输入下 export 前后 action 数值一致到合理容差。

任一项失败，不得启动正式训练。

### Phase 3：20-step smoke

目的不是看 success rate，而是验证：

- 无 OOM、NaN、bf16 转换错误；
- loss 可以 backward；
- teacher 不更新；
- student 确实收到 alignment gradient；
- checkpoint/export 可加载；
- W&B 指标完整。

为了节省磁盘，20-step smoke 不应同时保留重复的 9–10 GB step checkpoint 和 final checkpoint。

### Phase 4：500-step pilot

先运行 `alpha={0.02, 0.1, 0.5}` 的短 pilot。选择标准不是最低训练 action loss，而是：

- alignment cosine 有改善；
- action loss 没有异常恶化；
- student gradient 稳定；
- open-loop XYZ/gripper 不明显退化；
- checkpoint 能导出为不依赖 VGGT 的 policy。

### Phase 5：正式训练

只对 pilot 胜出的一个设置运行正式训练，保持与 baseline 可比：

```text
initial checkpoint: Libero74 baseline
dataset: same 74 episodes
action representation: same 8 × 7 delta-EEF
training steps: 20,000
action LR: baseline-compatible
teacher: frozen VGGT
student adaptation: selected-layer LoRA/adapters
```

训练前检查：

```bash
df -h /data /
nvidia-smi
```

不要删除其他用户文件或进程。训练 GPU 上不能同时运行占用约 10 GB 的 policy server。

### Phase 6：离线评估

使用完全相同协议：

```text
episodes: 0047 0077 0099 0121 0149 0150 0151
queries: 345
stride: 5
views: primary + wrist
```

报告：

- first XYZ L2 mean/median/p90；
- Y MAE 与 sign accuracy；
- gripper binary accuracy；
- false-close/missed-close；
- first-close timing；
- per-stage metrics：approach/close/lift/transport/release。

不能仅用总体 action L2，因为 binary gripper error 会主导该指标。

### Phase 7：无动作 sensitivity probe

在相同机器人初始状态下，拍摄：

```text
center
primary_front
primary_back
primary_left
primary_right
```

机器人不移动，只记录 action chunk。目标是验证预测相对物体移动具有正确、稳定的方向响应，而不只是“数值发生变化”。

### Phase 8：真机验证

顺序：

1. policy-server load test；
2. recorded-image offline inference；
3. live dry-run；
4. gripper-disabled arm motion；
5. conservative full execution；
6. matched 20-trial A/B。

正式 A/B 必须保持与 `13/20` baseline 相同：

```text
gripper chunk consensus: 0.75
switch confirmations: 3
close latch: enabled
physical lift validation: 30 mm
Cartesian temporal ensemble: disabled/window=1
max grasp attempts: 1
same start pose
same cube placement protocol
same cameras and task text
```

主要指标是 end-to-end success，不是训练 loss。失败分类至少包括：

```text
missed target
close above cube
table contact
failed close
failed lift
workspace abort
failed release
stale observation/controller failure
```

## 8. 成功标准

### 工程成功

- teacher frozen 且不进入 deployment；
- student 收到 alignment gradients；
- final policy server 只依赖 RGB StarVLA；
- 相同客户端可以切换 baseline/SF checkpoint；
- 所有训练配置和 checkpoint provenance 可追踪。

### 研究成功

最低要求：

- 不降低 baseline 的 in-distribution 真机成功率；
- 对轻微 cube-position shift 的失败减少；
- 提升能够在相同安全过滤和测试协议下复现；
- 至少报告 20 trials，并保留逐 trial 结果。

理想目标：

```text
Spatial Forcing model > 13/20 baseline
并且改善主要来自定位/grounding，而不是客户端规则变化。
```

## 9. 风险与缓解措施

### 风险 1：alignment 只训练 MLP

缓解：每步记录 selected student block 的 gradient norm；如果为零立即停止。

### 风险 2：teacher/student token 错位

缓解：保存网格可视化，逐 view 检查空间对应，不使用一维截断。

### 风险 3：VGGT 导致 OOM

缓解：停止本人的 policy server，使用空闲且获准的 95 GB H100；teacher 使用 no-grad/bf16；必要时降低 batch size、使用 gradient accumulation 或 LoRA。不要用删除安全检查来解决 OOM。

### 风险 4：磁盘再次写满

缓解：每次训练前保留充足空间；smoke/pilot 不保存重复大 checkpoint；删除文件前逐项确认归属和可恢复性。

### 风险 5：74 条数据位置分布过窄

Spatial Forcing 不会凭空产生未见过的动作监督。即使空间表征改善，模型仍可能学习平均抓取位置。应把训练分布内测试与位置偏移测试分开报告；若偏移仍失败，再收集有计划的位置多样化数据。

### 风险 6：低 action loss 被误认为真机成功

缓解：保持 open-loop、sensitivity、closed-loop 三层评估；正式结论只依据配对真机结果。

## 10. 立即执行的下一步

当前只执行 Phase 1 和 Phase 2，不启动正式训练：

1. 克隆/审阅 Spatial Forcing 官方实现；
2. 定位官方 VGGT feature 与 alignment loss 代码；
3. 在本地新增独立 `spatial_forcing` 模块，不覆盖 DAv2 或 baseline；
4. 实现 student image-token extraction 和二维 token correspondence；
5. 实现 frozen VGGT wrapper、alignment MLP 和 loss；
6. 写完 10 项单元测试；
7. 通过代码审查后才运行 20-step smoke。

在上述步骤完成前，不应再次进行 Spatial Forcing 真机执行。

## 10.1 Phase 1 官方实现审阅结果（2026-08-04）

已核对官方 `openvla-SF` 和 `openpi-SF` 源码，而不只依据论文摘要。

官方 OpenVLA 论文配置为：

```text
vla_layers_align: 24        # OpenVLA 共约 33 层
vggt_layers_align: -1       # VGGT 最后一组 aggregated features
align_loss_type: cosine
align_loss_coeff: 0.5
use_vlm_norm: true
use_vggt_pe: true
num_images_in_input: 2
lora_rank: 32
```

官方实际数据流为：

1. 从 VLA 指定 hidden layer 中截取 visual-token span；
2. 将送入 VLA 的图像反归一化并 resize/crop 为 VGGT 所需尺寸；
3. 在 `torch.no_grad()` 下运行 feature-only VGGT；
4. 取 `features[-1]` 并去掉 VGGT camera/register tokens；
5. 保持每个 view 的二维 patch grid，使用 bilinear interpolation 将 VGGT
   token 数匹配到 VLA visual-token 数；
6. 可选地向 VGGT grid 加入缩放为 `0.1` 的二维位置编码；
7. 对 VLA tokens 使用可选 `LayerNorm` 和 `Linear → GELU → Linear`；
8. 对归一化后的 student/teacher tokens 计算 `1 - cosine similarity`；
9. 优化 `action_loss + 0.5 * align_loss`；
10. 官方使用 LoRA 更新 VLA，并单独保存 alignment projector。

映射到当前 StarVLA 后有一个阻塞性差异：

```text
当前 Franka runner 默认：freeze_modules=qwen_vl_interface
Spatial Forcing 要求：产生 aligned hidden state 的 student 路径可训练
```

因此正式实现必须选择以下之一：

```text
推荐：对选定 Qwen text-transformer blocks 加 LoRA
备选：以极低 LR 解冻选定 blocks
无效：冻结整个 qwen_vl_interface，只训练 alignment projector
```

本项目初版使用 Qwen 总层数约 75% 位置的一个 hidden layer，对应官方
`24/33` 的相对深度；实际 layer index 必须从加载后的
`len(output.hidden_states)` 解析并记录，不能硬编码假设 Qwen 与 OpenVLA 层数相同。

## 11. 最终实验命名建议

```text
Baseline:
  starvla_libero74_baseline

Existing online depth ablation:
  starvla_libero74_dav2_gated

Spatial Forcing adaptation:
  starvla_libero74_spatial_forcing_vggt
```

论文/汇报中的推荐描述：

> We adapt Spatial Forcing to StarVLA/QwenGR00T by aligning intermediate multi-view visual tokens with frozen VGGT spatial representations during behavior-cloning fine-tuning. The VGGT teacher and alignment head are removed for deployment, so the resulting policy retains the original RGB-only inference interface.
