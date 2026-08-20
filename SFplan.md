# Spatial Forcing × StarVLA 项目计划

日期：2026-08-06  
本地工作区：`/home/dase-hw101/franka_ws`  
服务器代码目录：`/home/hanyu/starVLA`  
机器人：Franka FR3  
任务：`pick up the cube and place it on the box`  
观测：primary RGB + wrist RGB  
动作：8 × 7 delta-action chunk，包含末端平移、旋转和夹爪状态

## 1. 项目摘要

本项目计划在 StarVLA/QwenGR00T 上复现并适配 Spatial Forcing 的核心思想：训练时使用冻结的 3D foundation model（VGGT）作为 teacher，把 StarVLA 中间视觉特征与具有几何信息的 teacher representation 对齐；部署时删除 VGGT 和 alignment head，只保留 RGB-only StarVLA policy。

核心流程：

```text
训练：
primary/wrist RGB ──→ StarVLA student ──→ action loss
          │                  │
          └──→ frozen VGGT   └──→ projected cosine alignment loss

部署：
primary/wrist RGB ──→ RGB-only StarVLA student ──→ Franka action
```

本项目不是 Spatial Forcing 官方 OpenVLA/π0 路径的逐行复现，而是：

> Reproduction and adaptation of Spatial Forcing to StarVLA/QwenGR00T for local real-world Franka pick-and-place.

论文与官方资源：

- Spatial Forcing paper: <https://arxiv.org/abs/2510.12276>
- Spatial Forcing official implementation: <https://github.com/OpenHelix-Team/Spatial-Forcing>
- VGGT official implementation: <https://github.com/facebookresearch/vggt>

## 2. 为什么要做 Spatial Forcing

### 2.1 当前真实问题

当前最可靠的 Libero-init 74-episode baseline 在带通用安全过滤的真机测试中取得过：

```text
13 / 20 = 65% success rate
```

这说明模型已经学习到了任务的大体结构：

- 从标准位向方块移动；
- 接近方块；
- 部分情况下成功关闭夹爪；
- 部分情况下成功抬升、运输和释放。

但失败仍集中在空间动作精度和闭环状态附近：

- 到达方块附近但 XY 对不准；
- 停在方块上方，不继续下降；
- 从 primary camera 视角看落在方块前方或后方；
- 在方块上方过早关闭夹爪；
- 抓取位置偏心，之后无法稳定抬升；
- 运输阶段累计偏移，靠近 workspace 边界；
- 方块位置稍有变化时，策略仍趋向相似轨迹。

其中一部分失败可能来源于 StarVLA 的视觉 backbone 主要从 2D 图像预训练，未必形成足够明确的几何表示。因此，使用 3D foundation-model teacher 监督中间特征是一个合理且可验证的研究方向。

### 2.2 方法上的研究价值

Spatial Forcing 要回答的核心问题不是“更大的网络是否更好”，而是：

> 在动作数据和 policy 架构基本不变时，训练期的 3D representation alignment 能否让 RGB VLA 更准确地理解物体与末端之间的相对位置，并改善闭环动作精度？

这个问题具有明确的 control/treatment 结构：

```text
Control:   StarVLA + LoRA，alignment weight alpha = 0
Treatment: StarVLA + LoRA + frozen VGGT alignment，alpha > 0
```

只要两组使用相同的数据、初始化、随机种子、增强、训练步数和部署客户端，就可以区分“增加数据带来的收益”和“Spatial Forcing 本身带来的额外收益”。

### 2.3 可扩展性价值

现有 LeRobot demonstrations 只有 RGB、机器人 state 和 action，没有完整的 metric depth、point cloud 或相机标定。Spatial Forcing 的优点是：

- 不要求所有训练数据都包含 depth；
- VGGT 只在训练期从 RGB 生成几何 teacher representation；
- 部署时不运行 VGGT；
- 不增加真机推理输入类型；
- 不依赖不同机器人之间一致的深度传感器；
- 可以继续使用现有 primary/wrist RGB 数据。

这比把 DAv2/VGGT 永久放进 online policy 更容易扩展到其他 RGB-only 数据源。

### 2.4 为什么不能只在完全固定位置验证

如果方块、箱子、相机和机器人初始位姿完全固定，模型可以通过记忆平均轨迹取得较高成功率，而不必真正使用 3D representation。此时，即使 alignment loss 显著降低，action head 也可能忽略这些几何特征。

因此，本项目的目标不是进行大范围 OOD 泛化，而是定义一个受控的局部空间任务：

> 在原始数据采集位置附近的 ±15 mm 范围内，提高 StarVLA 的定位、抓取和放置成功率，并测试对未训练中间位置的局部插值能力。

这仍然符合“在固定工作区附近提高 SR”的实际目标，同时提供了足够的空间变化，让 Spatial Forcing 的假设可以被识别和验证。

## 3. 研究问题与假设

### 3.1 主要研究问题

在相同的局部空间数据、训练预算和部署设置下：

> Spatial Forcing treatment 是否比 matched alpha=0 control 获得更好的 unseen local-position success rate？

### 3.2 主要假设

```text
H1: treatment 的 VGGT projected alignment loss 会在训练中明显下降。

H2: 当方块在 robot-frame X/Y 中移动时，treatment 的预测接近方向
    会比 control 更一致地随物体位置变化。

H3: treatment 在未用于训练的局部中间位置上具有更低 XYZ action error。

H4: treatment 的真机 local-position SR 高于 matched control，且不会降低
    原中心位置 SR。

H5: treatment 不应通过更激进地提前关闭夹爪来换取表面上的 missed-close
    改善；false-close 必须受到约束。
```

### 3.3 零假设与替代解释

```text
H0: alpha=0 与 alpha>0 没有稳定行为差异；收益完全来自新增数据。
```

可能的替代解释包括：

- alignment head 学会拟合 VGGT，但 student action representation 没有真正改变；
- student 表示改变了，但 action head 仍使用平均轨迹 shortcut；
- 几何改善存在，但夹爪时序或闭环控制仍是主要瓶颈；
- 训练位置变化不足，无法建立视觉位置与动作变化的对应；
- primary/wrist 的 crop、token correspondence 或 view order 不一致；
- 训练步数不足，expanded dataset 尚未完成一个有效 epoch；
- 真机失败主要来自控制器、延迟、接触或 stale observation，而不是视觉几何。

## 4. 已完成工作与当前证据

### 4.1 真机主 baseline

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

历史结果：

```text
13/20 = 65% safety-filtered end-to-end success
```

该 checkpoint 仍是真机性能 baseline。它不是 Phase 10 的 matched alpha=0 control。

### 4.2 Phase 10 clean reproduction

已经实现并验证：

- frozen VGGT teacher；
- projected cosine alignment；
- Qwen all-linear LoRA；
- student/teacher 共享相同图像增强；
- alignment-head 与 LoRA gradient/update audit；
- teacher state 在 RGB-only inference export 中移除；
- policy server 部署时不加载 VGGT；
- alpha=0、0.1、0.5 matched training；
- open-loop evaluator；
- live snapshot sensitivity probe；
- Franka dry-run 和 real execution 路径。

主要本地代码：

```text
/home/dase-hw101/franka_ws/third_party/starVLA/
starVLA/model/framework/VLM4A/QwenGR00TSpatialForcingClean.py

/home/dase-hw101/franka_ws/third_party/starVLA/
starVLA/model/modules/spatial_forcing/alignment.py

/home/dase-hw101/franka_ws/third_party/starVLA/
starVLA/model/modules/spatial_forcing/image_augmentation.py

/home/dase-hw101/franka_ws/third_party/starVLA/
starVLA/model/modules/spatial_forcing/lora_student.py

/home/dase-hw101/franka_ws/third_party/starVLA/
starVLA/model/modules/spatial_forcing/vggt_teacher.py

/home/dase-hw101/franka_ws/third_party/starVLA/examples/realRobots/Franka/
train_files/run_qwengroot_spatial_forcing_clean_smoke.sh

/home/dase-hw101/franka_ws/scripts/export_spatial_forcing_rgb_view.py
```

完整工程说明：

```text
/home/dase-hw101/franka_ws/PHASE10_CLEAN_REPRODUCTION.md
```

### 4.3 Phase 10 已完成的 5k runs

Control：

```text
/data/hanyu/starVLA_runs/
phase10_clean_control_alpha0_5k_20260805_042006/
final_model/pytorch_model.pt
```

Treatment alpha=0.1：

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha01_5k_20260805_080954/
final_model/pytorch_model.pt
```

Treatment alpha=0.5：

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha05_5k_gpufix_20260805_045615/
final_model/pytorch_model.pt
```

### 4.4 当前结果

三轮 matched open-loop 平均结果：

```text
Metric                   alpha0       alpha0.1     alpha0.5
first XYZ L2             0.004525     0.004591     0.004689
chunk XYZ L2             0.004587     0.004573     0.004617
first rotation L2        0.000349     0.000359     0.000365
gripper accuracy         94.20%       93.75%       93.60%
false-close              3.55%        5.46%        4.64%
missed-close             8.50%        7.19%        8.50%
```

当前证据应表述为：

```text
technical reproduction:          PASS
alignment optimization:          PASS
RGB-only inference export:       PASS
offline action improvement:      NOT DEMONSTRATED
real-robot SR improvement:       NOT DEMONSTRATED
```

这不是证明 Spatial Forcing 无效，而是说明在原 74 条空间范围较窄的数据上，3D alignment 没有稳定转化为更好的 action prediction。

## 5. 下一阶段的关键改变：扩大局部空间覆盖

### 5.1 不是单纯增加 episode 数

下一阶段要增加的是数据的空间信息量，而不是在同一个位置重复录制更多相似轨迹。

需要形成以下可学习关系：

```text
cube moves +X → approach/close position should move +X
cube moves -X → approach/close position should move -X
cube moves +Y → approach/close position should move +Y
cube moves -Y → approach/close position should move -Y
```

只有当视觉变化对应不同的正确动作时，action head 才有理由使用经过 VGGT 对齐的空间 representation。

### 5.2 坐标定义

所有位置必须使用 Franka base/robot frame 定义，而不是使用“从 primary camera 看向前、向后、向左、向右”作为最终标签。

建议在桌面建立一个可重复放置的物理网格，并记录：

```text
position_id
delta_x_mm
delta_y_mm
cube_yaw_deg
robot_initial_pose_id
primary_camera_config_id
wrist_camera_config_id
episode_id
quality_status
```

在第一阶段，cube yaw 固定，只改变 X/Y。

### 5.3 训练位置：3 × 3 网格

以现有成功率最高的 nominal cube position 为中心：

```text
delta_x ∈ {-15, 0, +15} mm
delta_y ∈ {-15, 0, +15} mm
```

共 9 个训练位置：

```text
(-15,+15)  (0,+15)  (+15,+15)
(-15,  0)  (0,  0)  (+15,  0)
(-15,-15)  (0,-15)  (+15,-15)
```

建议每个位置收集 5–8 条成功 demonstrations：

```text
最低：9 × 5 = 45 episodes
推荐：9 × 8 = 72 episodes
```

为了保证可解释性，第一轮不要同时改变：

- cube yaw；
- box position；
- camera extrinsic；
- lighting；
- robot initial pose；
- task instruction。

### 5.4 未见验证位置

保留训练网格之间的中间位置作为 holdout，例如：

```text
(-7.5,-7.5) mm
(-7.5,+7.5) mm
(+7.5,-7.5) mm
(+7.5,+7.5) mm
```

这些位置不能进入训练数据。

为了进行离线 action comparison，可以在每个 holdout 位置录制 2–3 条高质量示范，但必须存入独立 validation dataset。真机评估时也应在这些预定义位置进行重复测试。

### 5.5 旧 74 episodes 的使用方法

原 74 条数据大多位于相似区域，而且没有可靠的精确网格标签。如果直接与 45–72 条新数据等概率按 episode 混合，旧中心分布可能继续占主导。

建议采用以下一种方法：

1. 推荐：把旧数据标记为 `legacy_center`，在 sampler 中限制其总采样权重，使其不超过一个或两个网格 cell 的总权重；
2. 或者：从旧数据中筛选固定数量的高质量 episode 作为中心数据；
3. 不推荐：不加权地混合全部旧数据并假设数据已经平衡。

Control 和 treatment 必须使用完全相同的数据列表和采样权重。

### 5.6 Episode 质量标准

训练 episode 必须满足：

- 两个相机画面清晰、同步且无长时间 stale；
- 方块在指定网格位置；
- Franka 从同一标准位开始；
- 接近过程没有明显碰撞或人工推动方块；
- 夹爪在有效抓取位置关闭；
- 方块实际抬离桌面；
- 方块运输到盒子上方；
- 模型目标动作中的 release 阶段完整；
- 方块最终留在盒子上；
- state/action/gripper feedback 连续；
- 没有 controller restart、camera timeout 或 episode 截断。

失败 demonstrations 不应混入普通 BC dataset。它们应另存，供后续 HG-DAgger、failure recovery 或 RL 使用。

## 6. Matched training 设计

### 6.1 Primary scientific comparison

两组都从同一个 Libero30k checkpoint 独立启动：

```text
/data/hanyu/starVLA_checkpoints/
libero_all_gr00t_official_30000_rerun/
final_model/pytorch_model.pt
```

模型 A：

```text
name: Phase 11 local-grid control
framework: QwenGR00TSpatialForcingClean
alpha: 0
teacher: instantiated/frozen but alignment has zero weight
```

模型 B：

```text
name: Phase 11 local-grid treatment
framework: QwenGR00TSpatialForcingClean
alpha: 0.1
teacher: frozen VGGT
loss: L_action + 0.1 * L_projected_alignment
```

目前不优先重复 alpha=0.5，因为现有结果已经显示它具有更强 alignment，但没有带来动作改善。Alpha=0.1 是下一轮较保守的 treatment。

### 6.2 必须严格匹配的变量

```text
initial checkpoint
dataset manifest
position-balanced sampler
batch size
number of optimizer updates
seed
learning rates
warmup/scheduler
LoRA targets/rank/alpha/init
action-model settings
diffusion settings
student/teacher augmentation
image resolution
primary/wrist ordering
checkpoint interval
evaluation frames
deployment client settings
safety filters
```

唯一主要实验变量应为：

```text
projected_alignment_alpha = 0 versus 0.1
```

### 6.3 训练阶段

#### Stage A：数据与 sampler audit

- 验证每个 position cell 的 episode 数；
- 验证每个 cell 的实际采样概率；
- 检查 primary/wrist 顺序；
- 检查 action/state schema；
- 检查 gripper 0/1 定义；
- 检查旧 center 数据是否压倒新位置；
- 输出固定 dataset manifest 和 SHA256。

#### Stage B：20-step smoke

Control 预期：

```text
alignment-head update norm = 0
LoRA/action model update norm > 0
```

Treatment 预期：

```text
alignment-head update norm > 0
LoRA update norm > 0
VGGT parameter update = 0
```

#### Stage C：500-step pilot

目标：

- 排除 NaN/OOM/checkpoint/export 问题；
- treatment projected loss 应下降；
- action loss 不应明显劣于 control；
- RGB-only export 能被 policy server 加载；
- teacher keys 不应成为部署依赖。

#### Stage D：正式训练

不要机械地沿用固定 5k 或 20k，而要根据 expanded dataset 的有效样本数确定训练预算。原 74eps 的 20k 约为 1.15 epoch；数据量扩大后，相同 20k steps 可能不足一个 epoch。

建议：

```text
target effective epochs: 1.0–1.2 for the first full comparison
control/treatment: exactly the same optimizer-step count
save: final model only, or at most one midpoint checkpoint
```

第一组 seed 完成并通过离线标准后，再重复第二个 seed。不要在第一轮同时扫描多个 alpha、layer 或 loss。

### 6.4 部署 export

训练 checkpoint 可包含恢复训练需要的 alignment head/teacher metadata，但 deployment view 必须满足：

- 使用普通 RGB observation；
- 不加载 VGGT weights；
- 不执行 teacher forward；
- 不需要 depth；
- checkpoint/config/norm statistics 完整；
- server 日志打印源 checkpoint、RGB export path 和 framework identity。

端口只能表示进程监听地址，不能作为模型身份。所有测试必须保存 checkpoint path。

## 7. 评估计划

### 7.1 表征诊断

以下指标只能证明 alignment 是否发生，不能单独证明机器人能力提高：

- projected alignment loss；
- alignment-head update norm；
- LoRA-B update norm；
- CKA；
- position RSA；
- shared probe loss。

表征 audit 是诊断项，最终判断必须依赖 action metrics 和真机 SR。

### 7.2 Live snapshot spatial-sensitivity probe

固定 Franka 在标准位，在不同预设 cube positions 捕获同一时刻观测，禁止发送机器人命令。

每个模型记录：

```text
first action dxyz
8-action chunk translation mean
predicted close fraction
policy roundtrip latency
raw robot state
checkpoint identity
```

关键检查：

- cube +X 时，预测接近轨迹是否相对向 +X 改变；
- cube -X、+Y、-Y 时是否呈一致方向变化；
- treatment 的变化是否比 control 更单调、稳定；
- 不同位置是否仍输出近乎相同的平均轨迹。

建议拟合：

```text
predicted displacement response / cube displacement
```

而不是只比较单次动作大小。

### 7.3 Open-loop evaluation

分别报告：

1. 原 legacy center episodes；
2. 训练网格 seen positions；
3. 未训练的 ±7.5 mm holdout positions；
4. 每个 position cell，而不是只报告 pooled overall mean。

指标：

```text
first XYZ L2
chunk XYZ L2
first/chunk rotation L2
Y/X sign accuracy
gripper binary accuracy
false-close rate when GT open
missed-close rate when GT closed
first predicted close frame
first GT close frame
close-frame error
```

必须运行相同帧列表、相同 stride 和相同随机推理次数。由于 diffusion inference 有随机性，每个模型至少运行三轮。

### 7.4 Dry-run

在真实实时相机和 robot state 上运行，但不发布动作：

- 先从标准位测试 8–10 inference requests；
- 再进行较长 dry-run，观察 target trajectory；
- 检查 policy 是否随 cube grid position 改变；
- 检查 gripper close 是否明显提前；
- 检查 workspace target 是否持续漂向边界；
- 检查 observation age 和 controller topic 稳定性。

Dry-run 通过不等于真机成功，只用于排除明显不安全或错误模型。

### 7.5 真机评估

#### Pilot

每个模型先进行约 20 次：

- 使用相同的预设 position list；
- 随机交替 control/treatment；
- 每次从同一标准位开始；
- camera/box/cube yaw 保持一致；
- 使用相同 deployment client 与安全过滤；
- 不因某个模型表现差而临时改变 workspace 或速度。

#### Formal evaluation

Pilot 无明显安全问题后，每个模型至少进行约 50 次。建议覆盖：

```text
nominal center
seen grid edges/corners
unseen ±7.5 mm midpoint positions
```

成功定义：

```text
policy 自主接近方块
→ 自主关闭夹爪
→ 方块实际离开桌面
→ 自主运输到盒子上方
→ 自主打开夹爪
→ 方块最终稳定留在盒子上
```

以下均计为 failure/incomplete：

- 人手纠正或移动方块；
- 人工触发夹爪；
- workspace abort；
- stale observation 中止；
- controller 死亡；
- 抓住但没有有效抬升；
- 到达盒子但未释放；
- 方块掉落到盒子外。

系统故障应同时单独统计，以区分 policy failure 和 infrastructure failure，但不能直接从分母中静默删除。

### 7.6 Failure taxonomy

每次测试标注一个主要失败阶段：

```text
P0: perception/initial targeting
P1: approach XY error
P2: stopped above cube / insufficient descent
P3: premature close
P4: missed/offset grasp
P5: grasped but insufficient lift
P6: transport error/workspace drift
P7: failed or late release
P8: controller/camera/stale-observation infrastructure failure
```

Spatial Forcing 最应该改善 P0–P2 和一部分由定位引起的 P3/P4。如果主要变化只出现在 P7 或系统故障，不能归因于更好的 3D representation。

## 8. 预注册的成功与停止条件

### 8.1 进入正式真机测试的最低条件

Treatment 必须同时满足：

- projected alignment 正常学习；
- holdout first/chunk XYZ error 至少一个稳定优于 control；
- 三轮 inference 的改善方向基本一致；
- false-close 不明显高于 control；
- live sensitivity 对 cube displacement 的响应合理；
- dry-run 不出现明显 workspace drift 或提前 close。

### 8.2 支持 Spatial Forcing 有效的结果

建议把以下作为有实际意义的目标，而不是保证：

```text
unseen local-position SR: treatment 比 control 高至少约 10 percentage points
center SR degradation:    不超过约 5 percentage points
false-close:              不高于 control 超过约 1 percentage point
结果方向:                  第二个 seed 或重复测试中仍一致
```

最终报告应包含置信区间和原始成功/失败计数；不能只报告一个百分比。

### 8.3 如何解释不同结果

#### 情况 A：control 和 treatment 都提高，幅度相似

结论：主要收益来自空间范围更广的数据，尚无证据证明 Spatial Forcing 提供额外收益。

#### 情况 B：treatment 在 unseen positions 提高，center 不下降

结论：支持 training-only 3D alignment 改善局部空间泛化。进入第二 seed 和正式真机评估。

#### 情况 C：alignment 降低，但动作与 SR 不改善

结论：student 学到了 teacher feature，action head 没有有效利用；停止继续扫描 alpha。下一步可以研究 action-aware geometry auxiliary objective，例如预测 EEF-object relative displacement。

#### 情况 D：XYZ 改善，但 false-close 变差

结论：空间定位与夹爪时序发生冲突。需要把连续 Cartesian action 与离散 gripper phase 分开建模，不能用更高 SR 掩盖危险的提前关闭。

#### 情况 E：两个模型都对 cube shift 不敏感

结论：数据或 action supervision 仍允许平均轨迹 shortcut；扩大/重新平衡位置覆盖，或者增加明确的 object-relative supervision。

#### 情况 F：真机差异主要由 stale/controller abort 造成

结论：先修复 infrastructure；这类结果不能用来判断 Spatial Forcing。

## 9. 项目范围与非目标

本阶段包含：

- 训练期 frozen VGGT representation alignment；
- RGB-only StarVLA deployment；
- 局部 X/Y 空间变化；
- matched alpha=0/0.1 comparison；
- 离线和真机评估。

本阶段不包含：

- 大范围物体位置泛化；
- 多物体、多任务或语言泛化；
- 在线 VGGT/DAv2 inference；
- RealSense metric-depth policy input；
- camera-to-robot calibration geometry branch；
- relational/scene-memory loss；
- RL；
- DAgger；
- 同时改变 cube yaw、相机和箱子位置；
- 为避免失败而放宽 workspace safety limits。

这些项目可以成为后续独立实验，但不能同时加入 Phase 11，否则无法判断 improvement 来自哪一项。

## 10. 安全和实验纪律

- 每次执行前确认 controller、camera、gripper feedback 和 policy server；
- 每次从已验证的标准位开始；
- 首先 live probe，其次 dry-run，最后才 real execute；
- 不因模型 target 越界而临时扩大 workspace；
- safety abort 必须保留并记录；
- 不使用自动抬升、object-specific close gate 或手工释放来宣称“纯 VLA 成功”；
- 通用去抖、命令连续发布和 workspace protection 可以保留，但所有模型设置必须完全相同；
- 每次测试保存 checkpoint path、server port、client command、ROS topic state、日志和视频；
- 端口号不能代替 checkpoint identity；
- 机器人、方块或相机配置改变后，必须更新 configuration ID。

## 11. 数据扩展后的项目阶段

### Phase 11A：准备局部网格

- 确定 nominal cube position；
- 建立 robot-frame 3×3 placement grid；
- 固定 cube yaw、box、camera 和 robot start；
- 建立 episode metadata template；
- 完成 2–3 条试录并验证坐标方向。

### Phase 11B：数据采集与质检

- 每个 grid cell 收集 5–8 条成功示范；
- 单独收集 holdout validation demonstrations；
- 检查双相机视频、state、action 和 gripper；
- 删除截断、碰撞、人工纠正和系统异常 episode；
- 生成固定 dataset manifest。

### Phase 11C：matched smoke/pilot

- alpha=0 与 alpha=0.1 运行 20-step smoke；
- 运行 500-step pilot；
- 审计 gradient、loss、export 和 server load；
- 不进行真机执行。

### Phase 11D：full matched training

- 按有效 epoch 设置相同训练预算；
- 第一 seed 完成；
- 只保留必要 checkpoint；
- RGB-only export；
- policy server 启动并验证 metadata。

### Phase 11E：offline and sensitivity evaluation

- seen/holdout/legacy 分组 open-loop；
- 三轮 stochastic inference；
- live cube-shift sensitivity probe；
- 形成 go/no-go decision。

### Phase 11F：real-robot evaluation

- 每模型 20-trial pilot；
- 通过安全与行为标准后扩展到约 50 trials/model；
- 分位置报告 SR 和 failure taxonomy；
- 必要时重复第二 seed。

## 12. 预期产物

```text
SFplan.md
local_grid_definition.yaml
episode_position_manifest.csv
train_manifest.json
validation_manifest.json
dataset_balance_audit.json
phase11_alpha0_config.yaml
phase11_alpha01_config.yaml
training_loss_comparison.csv
spatial_sensitivity_report.json
open_loop_by_position.csv
real_robot_trial_manifest.csv
real_robot_failure_taxonomy.csv
phase11_final_report.md
```

模型产物：

```text
Phase 11 alpha=0 final model
Phase 11 alpha=0.1 final model
两者的 RGB-only inference view
checkpoint/config/dataset hash manifest
```

## 13. 本项目最终希望支持的结论

如果 treatment 达到预注册标准，可以谨慎表述为：

> Training-time alignment to frozen VGGT representations improved the local spatial robustness of a StarVLA/QwenGR00T Franka policy using only RGB observations at deployment.

不能仅凭 alignment loss 下降写成“Spatial Forcing 提高了真机性能”。最终结论必须由 matched control、unseen local-position evaluation 和真实 Franka SR 共同支持。

如果 treatment 没有超过 control，也仍能得到清晰结论：在当前任务、数据规模、局部变化范围和 StarVLA 适配方式下，表示对齐本身不足以改善行为；后续应转向 targeted intervention、phase-aware gripper learning、robot-frame geometry 或 bounded residual RL，而不是继续无约束地扫描 alpha。

## 14. 下一项立即执行的工作

在开始任何新训练之前，先完成：

1. 确定 nominal cube position，并在桌面建立 3×3、间隔 15 mm 的可重复网格；
2. 试录 center、+X、-X、+Y、-Y 各 1 条 episode；
3. 验证 robot-frame 标签方向与视频方向；
4. 检查这五条 episode 的两个相机、动作和夹爪数据；
5. 使用相同标准位做 live snapshot probe，确认原 baseline 对位置变化的响应；
6. 通过后再正式收集 45–72 条 balanced local-grid demonstrations。

在这一步完成前，不应继续训练新的 alpha，也不应开始 Spatial Forcing 正式真机 SR comparison。
