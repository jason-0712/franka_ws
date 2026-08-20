# StarVLA × Franka × Spatial Forcing 周总结

日期范围：2026-08-10 至 2026-08-13  
本地工作区：`/home/dase-hw101/franka_ws`  
服务器代码目录：`/home/hanyu/starVLA`  
服务器训练输出：`/data/hanyu/starVLA_runs`  
机器人：Franka FR3  
任务：`pick up the cube and place it on the box`  
输入：primary RGB + wrist RGB，部分配置包含机器人 state  
输出：8 × 7 delta-action chunk（EEF 平移、旋转和夹爪）

---

## 1. 本周执行摘要

本周围绕两个问题展开：

1. 如何在不单纯依赖增加大量示范的情况下，提高 StarVLA 在局部方块位置变化下的抓取与放置表现；
2. Spatial Forcing 是否能把冻结 VGGT 的几何知识迁移到 StarVLA，并进一步改善动作精度或真机成功率。

本周完成了以下主要工作：

1. 构建并训练了固定的 Spatial30 数据对照实验：10 Middle + 10 Front + 10 Back。
2. 发现 Spatial30 Control 和 SF 模型都会在方块后方闭合，排除了夹爪确认次数是主要原因。
3. 修复客户端在夹爪切换时的 target/measured pose 不同步问题，加入 measured-pose synchronized close hold。
4. 在历史上真机表现较好的 Libero74 模型基础上加入新增 20 条示范，建立 Replay94 模型。
5. Replay94 真机测试达到 `14/20 = 70%`，当前为最强经验 baseline；主要失败仍是夹爪在方块上方或后方关闭。
6. 完成 Replay94 上多轮 Spatial Forcing matched control/treatment 实验，包括 5k、低学习率 500-step、2000-step和多随机种子离线评估。
7. 完成 representation audit、位置距离 heatmap、action vector field 和三模型对照可视化。
8. 审计 Spatial Forcing 官方实现，确认此前版本与官方设置存在关键偏差。
9. 实现 official-fidelity 版本：`llm_hidden layer 24 + VGGT layer -1 + joint views + VGGT PE + VLM norm + all-linear LoRA + alpha=0.5`。
10. official-fidelity 20-step smoke 的所有梯度与信号门通过。
11. official-fidelity 500-step treatment 的 alignment loss 下降约 88.7%，且 action loss 未明显退化。
12. 在 4 个训练未包含的开发 episode、5 个 matched diffusion seeds、共 890 对 paired queries 上进行严格检验；空间 evidence gate 失败。
13. 当前客观结论是：Spatial Forcing 技术复现成功，VGGT alignment 目标被成功优化，但尚未证明动作空间精度或真机成功率提升。

---

## 2. 当前最重要结论

### 2.1 当前最强行为模型仍是 Replay94

Replay94 从历史成功的 Libero74 checkpoint 出发，在完整 74 条旧数据和新增 20 条 Front/Back 数据上进行低学习率 replay fine-tuning。

Checkpoint：

```text
/data/hanyu/starVLA_runs/
replay94_from_successful_libero74_5k_retry1_20260810/
final_model/pytorch_model.pt
```

真机结果：

```text
14 / 20 = 70% success rate
```

历史 Libero74 baseline：

```text
13 / 20 = 65% success rate
```

20 次样本量仍较小，`70%` 与 `65%` 的差异不能视为统计显著提升，但 Replay94 至少没有破坏原模型，并在当前实测中取得最高成功次数。

主要失败模式没有根本改变：

- 夹爪在方块上方关闭；
- 夹爪在方块后方关闭；
- 抓取后抬升不足；
- 接近目标时存在平均轨迹或位置偏差；
- 部分 episode 靠接触推动方块后才成功，不属于 clean grasp。

### 2.2 Spatial Forcing 的实现与优化是成功的

本周已经证明：

- frozen VGGT teacher 能正常运行；
- Control 和 Treatment 使用 matched initialization/data/seed/config；
- α=0 Control 的 alignment head 不更新；
- α>0 Treatment 的 alignment head 和 LoRA-B 都收到有限、非零梯度；
- teacher 可以在部署导出时删除；
- RGB-only inference checkpoint 能正常启动 policy server；
- matched inference seed 能控制 diffusion sampling；
- official-fidelity treatment 的 projected alignment loss 显著下降。

因此，当前问题不是“代码没有跑通”或“alignment 没有学习”。

### 2.3 但 Spatial Forcing 尚未改善主要行为指标

截至本周结束，以下结论均未得到证明：

- first-step XYZ action error 稳定降低；
- chunk XYZ action error 稳定降低；
- pre-grasp 定位误差稳定降低；
- 真机 pick-and-place SR 提升；
- 方块位置变化时预测轨迹更准确地随位置变化。

当前最准确的表述为：

```text
technical reproduction:                PASS
alignment optimization:                PASS
gradient/update audit:                 PASS
RGB-only inference export:             PASS
matched-seed evaluation:               PASS
offline spatial-action improvement:    NOT DEMONSTRATED
real-robot SR improvement:             NOT DEMONSTRATED
```

---

## 3. 8 月 10 日：Spatial30、客户端修复与 Replay94

### 3.1 Spatial30 数据集

数据集：

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/
quest3_franka_spatial_balanced_30eps_v1
```

Manifest：

```text
/home/dase-hw101/franka_ws/dataset_manifests/
quest3_franka_spatial_balanced_30eps_v1.json
```

组成：

```text
Middle: 10 episodes
Front:  10 episodes
Back:   10 episodes
Total:  30 episodes
Frames: 7,844
Videos: 60
```

Middle source IDs：

```text
0131 0135 0136 0138 0142
0146 0147 0148 0150 0151
```

Front IDs：

```text
030 031 032 033 036
037 039 040 041 042
```

Back IDs：

```text
034 045 046 047 049
050 051 052 053 054
```

Episode 055 被保留作为开发评估 episode，没有加入训练。

### 3.2 Spatial30 Control 与 SF

Control checkpoint：

```text
/data/hanyu/starVLA_runs/
spatial30_control_alpha0_5k_seed42_20260810/
final_model/pytorch_model.pt
```

Treatment checkpoint：

```text
/data/hanyu/starVLA_runs/
spatial30_sf_alpha01_5k_seed42_20260810/
final_model/pytorch_model.pt
```

第一轮 open-loop 显示 mixed result：SF 的部分 XYZ/y-direction 指标略好，但总体 L2 和夹爪指标更差。由于当时数据量、训练预算、framework 和历史 baseline 都不完全 matched，不能用该实验单独判断 SF 有效或无效。

### 3.3 真机共同失败：在方块后方关闭

Control 和 SF 多次在方块后方关闭。将：

```text
--gripper-switch-confirmations 3
```

改为：

```text
--gripper-switch-confirmations 1
```

仍不能消除偏差，说明主要问题不是三次确认造成的夹爪延迟。

SF 曾出现一次任务成功，但属于 contact-assisted success：夹爪先碰到方块边缘并把方块推入可抓位置后完成抓取。该 episode 应标记为：

```text
task_success = 1
clean_grasp = 0
contact_assisted = 1
```

不能将其作为干净的空间定位成功证据。

### 3.4 客户端 measured-pose synchronized close hold

本周修复了 close 切换时 target pose 领先 measured pose 的问题。修复前，客户端可能在真实末端尚未到达 policy target 时继续累积目标，使夹爪切换位置出现约 14–18 mm 的执行偏差。

修复后：

- close 候选确认后，目标与 measured pose 同步；
- close hold 期间持续发送已确认的夹爪状态；
- target 与 measured XY 在切换附近仅相差约 0.4–0.6 mm；
- 模型仍在相同方向出现抓取偏差。

因此，剩余主要问题更可能来自视觉定位、数据分布或 policy representation/action mapping，而不是客户端目标累积。

### 3.5 Replay94 建立与训练

为了保留 Libero74 已经学到的行为能力，没有只使用 30 条 spatial data 重新训练，而是：

```text
历史 74 episodes + 新增 20 Front/Back episodes = Replay94
```

训练设置核心点：

- 初始化：历史成功 Libero74 checkpoint；
- 训练步数：5,000；
- action model LR：`3e-5`；
- Qwen interface LR：`1e-7`；
- 使用完整 replay，避免灾难性遗忘。

Replay94 open-loop 主要结果（7 episodes，224 queries）：

```text
first_l2_mean:                 0.020724
first_l2_median:               0.001993
first_l2_p90:                  0.006809
chunk_l2_mean:                 0.022820
first_xyz_l2_mean:             0.002876
gripper_binary_accuracy:       0.982143
false_close_rate_when_gt_open: 0.033058
missed_close_rate_when_gt_closed: 0.000000
first_close_frame_error:       -5
```

在 reserve 055 上，Replay94 也明显优于原始 Libero74 的总体 L2/夹爪结果。该结果推动后续所有 SF 实验改为从 Replay94 出发。

---

## 4. 8 月 11 日：Replay94 上的 Phase10、低学习率修正与可视化

### 4.1 Replay94 + Phase10 5k 对照

Control：

```text
/data/hanyu/starVLA_runs/
replay94_phase10_control_alpha0_5k_seed42_retry1_20260811/
final_model/pytorch_model.pt
```

Treatment：

```text
/data/hanyu/starVLA_runs/
replay94_phase10_treatment_alpha01_5k_seed42_20260811/
final_model/pytorch_model.pt
```

两者相比直接 Replay94 都出现不同程度的动作退化，说明此前 Phase10 的 LoRA 学习率/训练预算本身可能改变了已训练好的 policy；因此不能把所有差异都归因于 α=0.1 alignment。

### 4.2 Representation audit

Replay94 Phase10 treatment 相对 control 的主要表示审计结果：

```text
delta linear CKA:          +0.023377
delta position RSA:        -0.259838
delta shared-probe loss:   -0.000188
```

该结果为 mixed evidence：

- CKA 方向为正；
- shared-probe loss 略有改善；
- position RSA 明显变差。

因此不能只根据 alignment head 自己的 loss 宣称几何知识已迁移。

### 4.3 位置 heatmap 与 action vector field

本周生成：

- position-distance heatmap；
- first-action vector field；
- chunk-mean action vector field；
- Replay94 baseline、Phase10 control、Phase10 treatment 三模型面板。

主要工具：

```text
/home/dase-hw101/franka_ws/scripts/
plot_spatial_forcing_position_heatmaps.py

/home/ros/ros2_ws/scripts/
plot_starvla_action_vector_field.py
```

这些可视化表明：alignment-induced representation change 不一定产生正确的 action-vector change；同时 Phase10 control 本身也可能相对直接 Replay94 发生漂移。

### 4.4 低学习率 500-step Phase10

为减少对 Replay94 原行为的破坏，训练改为低学习率、短预算 500 steps。

Control open-loop：

```text
queries:                         224
first_l2_mean:                   0.016514
chunk_l2_mean:                   0.017009
first_xyz_l2_mean:               0.003113
gripper_binary_accuracy:         0.986607
false_close_rate_when_gt_open:   0.016529
missed_close_rate_when_gt_closed:0.009709
first_close_frame_error:         0
```

Treatment open-loop：

```text
queries:                         224
first_l2_mean:                   0.012156
chunk_l2_mean:                   0.018190
first_xyz_l2_mean:               0.003214
gripper_binary_accuracy:         0.991071
false_close_rate_when_gt_open:   0.008264
missed_close_rate_when_gt_closed:0.009709
first_close_frame_error:         0
```

解释：

- Treatment 的 first L2 和夹爪指标更好；
- 但 first XYZ 和 chunk L2 更差；
- 仍然不能证明 SF 改善空间动作。

该实验说明学习率和步数确实重要，但“更小总 L2”可能主要来自夹爪维度，必须把 XYZ 与 gripper 分开报告。

---

## 5. 8 月 13 日：修正 SF fidelity、严格五种子实验和 official-fidelity

### 5.1 发现此前复现与官方实现的关键差异

审计 Spatial Forcing 官方仓库后，确认官方设置包含：

```text
student feature:       VLA LLM hidden state
VLA layer:             24
teacher:               frozen VGGT
VGGT feature layer:    -1
teacher views:         two images jointly processed
VGGT positional embed: enabled
VLM normalization:     enabled
LoRA targets:          all-linear
LoRA rank:             32
LoRA alpha:            16
projected cosine α:    0.5
image augmentation:    enabled during training
```

此前正式版本主要使用：

```text
student feature:       vision_projector
alpha:                 0.1
teacher views:         independent
VGGT PE:               disabled
```

因此此前结果可以说明该适配版本无明显收益，但不能作为对官方 SF 设定的最终检验。

官方资源：

- Spatial Forcing：<https://github.com/OpenHelix-Team/Spatial-Forcing>
- Spatial Forcing paper：<https://arxiv.org/abs/2510.12276>
- VGGT：<https://github.com/facebookresearch/vggt>

### 5.2 Official-fidelity 实现

新增主要文件：

```text
/home/dase-hw101/franka_ws/third_party/starVLA/examples/realRobots/Franka/
train_files/starvla_cotrain_quest3_franka_sf_official_fidelity.yaml

/home/dase-hw101/franka_ws/third_party/starVLA/examples/realRobots/Franka/
train_files/run_qwengroot_sf_official_fidelity_smoke.sh

/home/dase-hw101/franka_ws/third_party/starVLA/scripts/
run_sf_official_fidelity_matched_smoke.sh

/home/dase-hw101/franka_ws/third_party/starVLA/tests/
test_spatial_forcing_official_fidelity.py

/home/dase-hw101/franka_ws/third_party/starVLA/scripts/
install_sf_official_fidelity_on_server.sh

/home/dase-hw101/franka_ws/docs/
SF_OFFICIAL_FIDELITY_20260813.md
```

部署 bundle：

```text
/home/dase-hw101/franka_ws/artifacts/
sf_official_fidelity_20260813.tar.gz
```

SHA256：

```text
aac556489985583dd98f41f641d616a518e063db3c618ef52dbffa2ed31c5968
```

### 5.3 测试与 smoke

静态/单元测试验证了：

- official config contract；
- control/treatment α contract；
- two-view hidden-token extraction；
- spatial alignment utilities；
- LoRA gradient flow；
- RGB inference export 路径。

最初 unittest module path 受到 `tests` package shadowing 影响，改用 direct-file 或 discover 方式后测试通过。

20-step Control：

```text
action_dit_loss:                              0.0113903
total_loss:                                   0.0113903
projected_alignment_loss:                     1.0142392
weighted_projected_alignment_loss:            0
update_norm/spatial_forcing_lora_B:            0.0001423
update_norm/alignment_head:                    0
```

20-step Treatment：

```text
action_dit_loss:                              0.0115331
total_loss:                                   0.2355599
projected_alignment_loss:                     0.4480537
weighted_projected_alignment_loss:            0.2240268
update_norm/spatial_forcing_lora_B:            0.0001482
update_norm/alignment_head:                    0.0036657
```

信号门：

```text
all_required_values_finite:             PASS
control_weighted_alignment_zero:        PASS
control_alignment_head_update_zero:     PASS
control_lora_B_updates_from_action_loss: PASS
treatment_weighted_alignment_positive:  PASS
treatment_alignment_head_updates:       PASS
treatment_lora_B_updates:                PASS
OFFICIAL_FIDELITY_SIGNAL_GATE:           PASS
```

日志检测器曾把 DeepSpeed 的合法配置：

```text
steps_per_print: inf
```

误判为数值异常。Checkpoint zip integrity 检查通过，随后日志规则被收紧为只检测 fatal errors 或 loss/score/norm/metric 中的非有限值。

### 5.4 Official-fidelity 500-step training

Control：

```text
/data/hanyu/starVLA_runs/
sf_official_fidelity_control_alpha0_500_seed42_
20260813_official500_seed42/
final_model/pytorch_model.pt
```

Treatment：

```text
/data/hanyu/starVLA_runs/
sf_official_fidelity_treatment_alpha05_500_seed42_
20260813_official500_seed42/
final_model/pytorch_model.pt
```

训练曲线：

```text
Control total/action loss, last100:        0.111611
Treatment action loss, last100:            0.110735
Treatment alignment first50:               0.817164
Treatment alignment last50:                0.092570
Treatment alignment reduction:             88.672%
Treatment alignment-head update, last100:  0.002835
Control alignment-head update:              0
```

关键解释：

- Treatment 成功学习 VGGT projected alignment；
- alignment loss 大幅下降；
- Treatment action loss 没有明显退化，最后 100 steps 甚至比 Control 低约 0.8%；
- 但这只证明训练目标被成功优化，不代表动作或成功率提高。

### 5.5 Official-fidelity 500-step 严格 open-loop

评估数据：

```text
quest3_9_grids_055
quest3_9_grids_058
quest3_9_grids_059
quest3_9_grids_060
```

设置：

```text
matched diffusion seeds: 42, 314159, 271828, 20260813, 8675309
queries per seed:        178
paired queries total:    890
stride:                  5
robot commands sent:     0
```

五种子 aggregate：

```text
first_xyz_l2:
  Control:   0.003894707
  Treatment: 0.003909911
  change:    +0.39%（Treatment 更差）

chunk_xyz_l2_mean:
  Control:   0.003820166
  Treatment: 0.003836632
  change:    +0.43%（Treatment 更差）

gripper accuracy:
  Control:   0.934831
  Treatment: 0.938202
  change:    +0.337 percentage points

false-close rate:
  Control:   0.040000
  Treatment: 0.036522
  change:    -0.348 percentage points

missed-close rate:
  Control:   0.111111
  Treatment: 0.107937
  change:    -0.317 percentage points
```

逐数据集空间结果：

- 055：first/chunk XYZ 均恶化；
- 058：first/chunk XYZ 均恶化；
- 059：first/chunk XYZ 均恶化；
- 060：first/chunk XYZ 有极小改善，但仅约 0.45%/0.30%，远低于 3% threshold。

预注册 evidence gate：

```text
aggregate first XYZ improvement >= 3%
aggregate chunk XYZ improvement >= 3%
至少两个 dataset 同时满足两项 >= 3%
```

结果：

```text
SPATIAL_GATE=FAIL
```

因此不继续将相同 official-fidelity 配置扩展到 2000 steps，也不进行该模型的大规模真机 SR A/B。

---

## 6. 早期 2000-step vproj 版本的独立开发评估

在 official-fidelity 之前，还完成了 `vision_projector + alpha=0.1` 版本的 2000-step matched evaluation。

数据同样为 055/058/059/060，五个 matched seeds，共 890 paired queries。

Aggregate：

```text
first-step XYZ:
  0.003761435 -> 0.003759150
  improvement: 0.0608%

chunk XYZ:
  0.003738708 -> 0.003748350
  degradation: 0.2579%

gripper accuracy:
  0.906742 -> 0.908989
  +0.225 percentage points
```

结果：

```text
SPATIAL_GATE=FAIL
```

保留产物：

```text
/home/dase-hw101/franka_ws/artifacts/
sf2000_independent_holdout4_20260813/
```

该结果和 official-fidelity 500-step 结果方向一致：不同 SF 适配方式都成功优化 alignment，但没有形成稳定的 XYZ action gain。

---

## 7. 本周工程改动

### 7.1 Spatial Forcing framework

核心模块：

```text
/home/dase-hw101/franka_ws/third_party/starVLA/starVLA/model/framework/VLM4A/
QwenGR00TSpatialForcing.py

/home/dase-hw101/franka_ws/third_party/starVLA/starVLA/model/framework/VLM4A/
QwenGR00TSpatialForcingClean.py

/home/dase-hw101/franka_ws/third_party/starVLA/starVLA/model/modules/spatial_forcing/
alignment.py
image_augmentation.py
lora_student.py
representation_audit.py
vggt_teacher.py
```

实现内容：

- frozen VGGT teacher；
- projected cosine alignment；
- student feature layer/source routing；
- independent/joint teacher view support；
- VGGT positional embedding；
- VLM normalization；
- all-linear LoRA；
- alignment-head optimizer group；
- gradient/update norm logging；
- training-only teacher state filtering；
- RGB-only inference export。

### 7.2 Evaluation and visualization

```text
/home/dase-hw101/franka_ws/scripts/
starvla_open_loop_l2_eval.py
compare_starvla_seeded_open_loop.py
export_spatial_forcing_rgb_view.py
plot_spatial_forcing_position_heatmaps.py
```

重要能力：

- stable matched inference seeds；
- control/treatment paired CSV；
- aggregate 和 per-dataset 指标；
- 3% spatial evidence gate；
- position-distance heatmap；
- action vector field；
- teacher-free RGB inference view。

### 7.3 可靠性与资源管理

本周同时处理了：

- policy server GPU 占用识别和停止；
- 80 GB / 96 GB H100 的显存审计；
- DeepSpeed OOM 与进程冲突；
- `/data` 和根分区空间清理；
- 大型 checkpoint 的软链接 inference export；
- checkpoint zip central-directory integrity 验证；
- policy server 端口冲突与启动检查；
- `tests` module shadowing；
- Rich 多行日志导致 grep 只抓到键名的问题；
- DeepSpeed `steps_per_print: inf` 的假阳性日志检测。

---

## 8. 为什么当前不能宣称 Spatial Forcing 提高了 SR

本周有若干看起来“变好”的数字：

- alignment loss 降低约 88.7%；
- official-fidelity gripper accuracy 提高约 0.34 个百分点；
- false-close/missed-close 略有下降；
- 某些 seed 或单一 dataset 上的部分指标改善。

但这些都不足以支持 SR 提升，因为：

1. alignment loss 是训练目标本身，下降只能证明优化成功；
2. gripper 增益很小，且与 VGGT 几何机制的联系较弱；
3. aggregate first/chunk XYZ 都略有恶化；
4. 只有 060 出现极小空间改善，其他三个位置恶化；
5. 尚未有 matched real-robot control/treatment SR 证据；
6. 之前真机主要失败是抓取位置偏差，不只是 gripper binary 分类。

因此本周不能写：

```text
Spatial Forcing improved StarVLA success rate.
```

可以写：

```text
We reproduced and adapted Spatial Forcing to StarVLA and verified that the
VGGT alignment objective was optimized correctly. However, the representation
alignment did not yield a meaningful or consistent reduction in open-loop
spatial action error on the current Franka task.
```

---

## 9. 当前科学解释

目前存在两个最可能的瓶颈：

### 9.1 有效几何并未真正进入 student representation

虽然 projected alignment loss 降低，但可能是 alignment head 吸收了大部分拟合；StarVLA hidden feature 未必更显式地编码方块位置。

### 9.2 几何已进入 representation，但 action head 没有利用

即使 student hidden feature 中包含更容易读取的位置关系，现有 GR00T action head 仍可能继续使用平均轨迹、数据 shortcut 或旧的视觉特征子空间。

仅使用 action L2 无法区分以上两种情况。这就是下一步需要 frozen representation probe 的原因。

---

## 10. 下一阶段计划

### 10.1 首要实验：frozen object-relative position probe

冻结以下模型：

```text
Replay94 / matched Control
Official-fidelity SF Treatment
VGGT teacher（作为上限参考）
```

在相同静态图像上提取中间特征，只训练相同容量的线性 probe：

```text
frozen student feature
        ↓
small linear probe
        ↓
cube position relative to gripper in robot frame
(Δx, Δy, optionally Δz)
```

主要指标：

- held-out relative-position RMSE（mm）；
- X/Y MAE（mm）；
- R²；
- `P(XY error < 5 mm)`；
- `P(XY error < 10 mm)`；
- 3 个 probe seeds 的均值和标准差。

该实验不等于成功率，但能区分：

```text
SF没有迁移可用几何
vs
SF迁移了几何，但action head没有使用
```

### 10.2 辅助实验：pre-grasp 分阶段动作评估

把 episode 分为：

1. approach；
2. grasp transition；
3. transport；
4. release。

重点报告：

- pre-grasp XYZ error；
- approach directional cosine；
- close-frame spatial offset；
- first-close timing；
- false/missed-close。

如果发现探索性改善，必须在新的、未分析过的 holdout episodes 上确认。

### 10.3 跨视角与扰动鲁棒性

可进一步量化：

- primary/wrist same-position retrieval Recall@1/Recall@3；
- same-position vs different-position feature margin；
- brightness/contrast/blur 等几何不变扰动下的 action consistency；
- 方块移动后 action vector 的方向一致性。

### 10.4 决策门

如果 SF position probe 不优于 Control：

```text
结论：当前 alignment 没有迁移可读的物体几何。
动作：停止当前 SF 路线，检查 token correspondence/teacher target/layer。
```

如果 probe 改善但 action error 不改善：

```text
结论：存在 representation-to-action utilization bottleneck。
动作：实现 zero-gated geometry-to-action connector。
```

如果 probe、pre-grasp action 和独立 open-loop 都改善：

```text
动作：再进行小规模、matched、严格安全过滤的真机 A/B。
```

### 10.5 暂不进行的方向

- 不继续把相同 official-fidelity 500-step 配置盲目扩展到 2000 steps；
- 不降低 3% gate 来制造正结果；
- 不只选 060 或单一 seed 报告；
- 不立即进行 RL；
- 不把 alignment loss 下降写成行为提升；
- 不进行未经标定的 DAv2/VGGT 在线几何条件输入。

如果最终转向直接 geometry conditioning，建议优先使用：

- RealSense aligned metric depth；
- primary camera 标定 `T_B_Cp`；
- wrist hand–eye 标定 `T_E_Cw`；
- robot-frame point/relative geometry；
- zero-initialized gated connector；
- 先冻结原 policy，仅训练 connector；
- 使用相同 matched control/treatment 和 3% evidence gate。

---

## 11. 本周模型清单

### 真机/行为主 baseline

```text
Libero74:
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt

Replay94:
/data/hanyu/starVLA_runs/
replay94_from_successful_libero74_5k_retry1_20260810/
final_model/pytorch_model.pt
```

### Spatial30

```text
Control:
/data/hanyu/starVLA_runs/
spatial30_control_alpha0_5k_seed42_20260810/
final_model/pytorch_model.pt

SF alpha=0.1:
/data/hanyu/starVLA_runs/
spatial30_sf_alpha01_5k_seed42_20260810/
final_model/pytorch_model.pt
```

### Replay94 Phase10

```text
5k Control:
/data/hanyu/starVLA_runs/
replay94_phase10_control_alpha0_5k_seed42_retry1_20260811/
final_model/pytorch_model.pt

5k Treatment:
/data/hanyu/starVLA_runs/
replay94_phase10_treatment_alpha01_5k_seed42_20260811/
final_model/pytorch_model.pt
```

### Official-fidelity 500-step

```text
Control alpha=0:
/data/hanyu/starVLA_runs/
sf_official_fidelity_control_alpha0_500_seed42_20260813_official500_seed42/
final_model/pytorch_model.pt

Treatment alpha=0.5:
/data/hanyu/starVLA_runs/
sf_official_fidelity_treatment_alpha05_500_seed42_20260813_official500_seed42/
final_model/pytorch_model.pt
```

---

## 12. 最终周结论

本周最重要的成果不是证明 Spatial Forcing 已经提高真机成功率，而是完成了一条可审计、可复现、带严格 control/treatment 和 evidence gate 的实验链：

```text
数据与部署问题排查
    ↓
Replay94 强 baseline（14/20）
    ↓
旧 SF/Phase10 matched experiments
    ↓
representation 与 vector-field audit
    ↓
官方实现差异审计
    ↓
official-fidelity 复现
    ↓
smoke gradient gate PASS
    ↓
500-step alignment optimization PASS
    ↓
5-seed / 4-dataset / 890-query spatial gate FAIL
```

目前最客观的研究结论是：

> 在 StarVLA/Replay94 的 Franka 双相机 pick-and-place 任务上，我们成功复现并适配了 Spatial Forcing 的 VGGT feature-alignment 训练机制。该机制显著降低了 projected alignment loss，且未明显损害 BC action loss；然而，在训练未包含的局部位置开发集上，它没有带来稳定或具有实际意义的 XYZ action-error 改善。因此，当前证据不支持直接扩大训练或声称真机 SR 提升。下一步应通过 frozen object-relative position probe 判断几何信息是否已进入 student representation，再决定是修正 alignment，还是设计显式的 geometry-to-action connector。

