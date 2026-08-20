# StarVLA–Franka 2026-08-10 工作总结

## 1. 今日结论

今天完成了两条关键路线的收敛与切换：

1. 完成固定的 `30-episode` Spatial Forcing 对照数据集与 Control/SF 训练、离线评估和第一轮真机对照。
2. 确认两种 spatial30 模型都会在 cube 后方闭合；进一步排除了夹爪确认次数和客户端目标累积超前作为主要剩余原因。
3. 在部署客户端加入通用的 measured-pose synchronized close hold，消除了 close 时目标位姿领先真实末端约 14–18 mm 的问题。
4. 通过日志确认：修复后 close 时 target 与 measured XY 仅相差约 0.4–0.6 mm，但模型仍有相同方向的抓取空间偏差，因此剩余问题主要来自数据/视觉空间分布或模型定位，而非执行延迟。
5. 决定保留历史上真机表现较好的 `Libero30k + 74 episodes` 模型，并把新增 20 条 Front/Back 数据加入完整 replay，而不是只使用 spatial30 重新训练。
6. 构建并审计了固定的 `74 + 20 = 94 episodes` Replay 数据集。
7. 从历史成功的 74-episode checkpoint 进行低学习率 5k replay fine-tuning，训练成功完成。
8. 新 Replay94 模型的 open-loop regression 和 reserve-055 结果通过，8-step 真机 dry-run 也通过。
9. 新模型 policy server 已在 `192.168.1.113:10112` 运行，当前状态是：可以进行第一条保守真机 pilot，但尚未在本总结所覆盖的对话中报告该 pilot 的最终结果。

---

## 2. 重要模型与 checkpoint

### 2.1 历史主 baseline：Libero30k + 74 episodes

路径：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

关键历史结果：

```text
真机测试：13/20 = 65%
```

这仍是今天所有 replay 训练的初始化基础。

### 2.2 Spatial30 Control

```text
/data/hanyu/starVLA_runs/
spatial30_control_alpha0_5k_seed42_20260810/
final_model/pytorch_model.pt
```

RGB inference view：

```text
/data/hanyu/starVLA_runs/
spatial30_control_alpha0_5k_seed42_20260810_rgb_inference/
final_model/pytorch_model.pt
```

此前 policy server：

```text
port: 10110
PID: 313425
GPU: 0
```

该 server 后来为 Replay94 训练释放 GPU 0 而停止。

### 2.3 Spatial30 Spatial Forcing alpha=0.1

```text
/data/hanyu/starVLA_runs/
spatial30_sf_alpha01_5k_seed42_20260810/
final_model/pytorch_model.pt
```

RGB inference view：

```text
/data/hanyu/starVLA_runs/
spatial30_sf_alpha01_5k_seed42_20260810_rgb_inference/
final_model/pytorch_model.pt
```

policy server：

```text
port: 10111
PID: 313782
GPU: 3
```

### 2.4 今日新模型：Replay94 from successful Libero74

```text
/data/hanyu/starVLA_runs/
replay94_from_successful_libero74_5k_retry1_20260810/
final_model/pytorch_model.pt
```

当前 policy server：

```text
host: 192.168.1.113
port: 10112
GPU: 0
connection: PASS
```

---

## 3. Spatial30 数据集

### 3.1 数据集路径

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/
quest3_franka_spatial_balanced_30eps_v1
```

manifest：

```text
/home/dase-hw101/franka_ws/dataset_manifests/
quest3_franka_spatial_balanced_30eps_v1.json
```

构建器：

```text
/home/dase-hw101/franka_ws/scripts/
build_quest3_franka_spatial_balanced_30eps.py
```

### 3.2 固定组成

```text
Middle: 10
Front:  10
Back:   10
Total:  30 episodes
Frames: 7,844
Videos: 60
```

Middle 选择的 source IDs：

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

保留但未加入训练：

```text
055
```

### 3.3 近似抓取位置统计

基于每条成功示范第一次持续 close 附近的 Cartesian 状态：

```text
Middle mean: [0.5118, -0.0438, 0.1357]
Middle x range: 0.5018–0.5214
Middle y range: -0.0498–-0.0363

Front mean:  [0.6410, -0.0059, 0.1325]
Back mean:   [0.4748,  0.0110, 0.1336]
```

注意：Front mean x≈0.641 超出当前真机部署 `max-x=0.57`。这些坐标是抓取过程统计，不应直接当作新的 object-specific gate，但说明训练/部署工作区定义必须进一步统一。

### 3.4 参考图

```text
/home/dase-hw101/franka_ws/artifacts/
spatial30_front_middle_back_reference.png

/home/dase-hw101/franka_ws/artifacts/
spatial30_middle_representative_0131.png
```

---

## 4. Spatial30 Control 与 SF 离线结果

### 4.1 Control alpha=0

```text
queries: 224
first_l2_mean: 0.057434
first_l2_median: 0.003463
first_l2_p90: 0.010022
chunk_l2_mean: 0.036999
first_xyz_l2_mean: 0.003920
y_mae: 0.001534
y_sign_acc: 0.704918
gripper_binary_accuracy: 0.946429
false_close_rate_when_gt_open: 0.101852
missed_close_rate_when_gt_closed: 0.008621
first_close_frame_error: -5
```

### 4.2 SF alpha=0.1

```text
queries: 224
first_l2_mean: 0.079552
first_l2_median: 0.003575
first_l2_p90: 0.009167
chunk_l2_mean: 0.042530
first_xyz_l2_mean: 0.003762
y_mae: 0.001452
y_sign_acc: 0.827869
gripper_binary_accuracy: 0.924107
false_close_rate_when_gt_open: 0.138889
missed_close_rate_when_gt_closed: 0.017241
first_close_frame_error: -5
```

### 4.3 解释

SF 的 XYZ 和 y-direction 指标略好，但总体 first/chunk L2 和 gripper 指标更差。该结果属于 mixed signal，不能从 open-loop 宣称 Spatial Forcing 已提升真机 SR。

30条数据量较小，并且只保留10条原始 Middle。它与原74模型还存在以下混杂差异：

- 数据量：30 vs 74。
- 训练预算：5k vs 20k。
- Middle replay：10条 vs 原有约24条新腕相机 Middle。
- 模型结构：Phase-10 LoRA/SF framework vs 原始 QwenGR00T。
- 当前 cube 视觉位置并不严格等于旧训练 Middle。

因此 spatial30 结果不应简单解释为“SF无效”或“SF使模型变差”。

---

## 5. Spatial30 真机观察

### 5.1 Control

Control 多次在 cube 后方关闭夹爪，并发生 empty grasp。

### 5.2 SF

SF 有一次完成 pick-and-place，但存在明显 contact-assisted lucky factor：

- 初始夹爪略在 cube 后方。
- 左侧夹爪先触碰 cube 边缘。
- 接触把 cube 向后推了一点。
- 随后才成功闭合、抬升并放到 box 上。

该次记录可以标记：

```text
task_success = 1
clean_grasp = 0
contact_assisted = 1
```

它不能作为干净的视觉定位成功证据。

### 5.3 共同失败模式

用户随后在不同 Middle 位置、不同 `gripper-switch-confirmations` 设置下测试，Control 和 SF 仍倾向在 cube 后方关闭。

将 confirmations 从3改为1仍无法消除偏差，说明：

- 问题不只是3次连续确认造成的延迟。
- 问题不只是chunk consensus造成的晚闭合。
- 需要检查执行目标累积与模型空间定位本身。

---

## 6. synchronized close hold 修复

### 6.1 修改文件

Host：

```text
/home/dase-hw101/franka_ws/scripts/
starvla_franka_delta_pose_client.py
```

运行容器：

```text
/home/ros/ros2_ws/scripts/
starvla_franka_delta_pose_client.py
```

容器内修改前备份：

```text
/home/ros/ros2_ws/scripts/
starvla_franka_delta_pose_client.py.before_synchronized_close_hold_20260810
```

当前脚本 SHA256：

```text
b751402b5f75cdb0204875efa2486c7dca501253582f0846ddf39dc867deb594
```

### 6.2 行为

新增参数：

```text
--synchronized-close-hold
--no-synchronized-close-hold
```

默认启用。

逻辑：

1. 模型第一次产生可靠 close candidate。
2. 客户端把 target rebase 到最新 measured pose。
3. 在 debounce 和物理夹爪宽度验证阶段冻结末端位置与姿态。
4. 若 candidate 被取消，则 rebase 到 measured pose 并恢复模型运动。
5. 若真实 close 已发生，则即使后续模型短暂抖动也保持 close。
6. 宽度稳定验证完成后释放 hold，再由模型自行决定抬升。

该逻辑不包含：

- cube XYZ 坐标。
- object-specific close gate。
- 自动抬升。
- 自动放置或自动释放。

因此它属于通用通信/执行同步过滤，而不是任务作弊代码。

### 6.3 测试

测试文件：

```text
/home/dase-hw101/franka_ws/tests/
test_starvla_synchronized_close_hold.py
```

结果：

```text
6 unit tests: PASS
py_compile: PASS
CLI help check: PASS
```

### 6.4 修复前后证据

旧日志中，physical close 时 accumulated target 通常领先 measured pose：

```text
XY gap: approximately 14–18 mm
XYZ gap: approximately 18–22 mm
```

修复后 Control：

```text
measured: [0.53296, 0.01292, 0.12761]
target:   [0.53348, 0.01262, 0.12797]
XY gap: approximately 0.6 mm
```

修复后 SF：

```text
measured: [0.53165, 0.00574, 0.12458]
target:   [0.53207, 0.00568, 0.12486]
XY gap: approximately 0.4 mm
```

结论：客户端目标超前已经基本消除。

但是两边仍发生 edge contact 后 empty grasp：

```text
Control width: briefly 0.027958 m, then 0.008551 m
SF width:      briefly 0.027346 m, then 0.007976 m
```

这表明剩余主要问题是抓取空间定位或场景分布，而不是 close target tracking。

---

## 7. 当前相机画面与训练 Middle 差异

只读 snapshot probe 保存了当前画面：

```text
/home/dase-hw101/franka_ws/artifacts/current_middle_20260810/
primary_original.png
wrist_original.png
primary_224.png
wrist_224.png
```

训练 0131 首帧：

```text
/home/dase-hw101/franka_ws/artifacts/current_middle_20260810/train_0131/
primary_frame0.png
wrist_frame0.png
```

视觉检查：

- 当前 cube 在 primary 画面中比训练 Middle 0131 更靠右。
- 归一化横向位置差约为整幅图宽的4–5%。
- 在1280像素原图中约为50–60 px。
- wrist 图中 cube 可见，但尺寸较小且金属桌面反光明显。

因此用户主观认为的“另一个 Middle”不一定等于模型训练分布中的视觉 Middle。

该观察与两种模型共享相似方向偏差一致。

---

## 8. 为什么回到原74模型并加入20条新数据

原74模型真机历史结果约13/20，而 spatial30 模型更不稳定。主要原因可能是：

1. 原74模型看过更多 Middle 示范。
2. 原74模型训练20k steps；spatial30只训练5k。
3. spatial30丢弃了原74中大量有效动作与场景覆盖。
4. spatial30 的 Front/Back 数据占比达到2/3，可能降低固定 Middle 精度。
5. SF Control 虽然 alpha=0，仍然不是原始 QwenGR00T 的完全等价模型。

因此决定：

```text
保留完整旧74
+ 新增10 Front
+ 新增10 Back
= 94 unique episodes
```

不能直接把整个 spatial30 加到74，因为其中10条 Middle 已经存在于旧74中，会重复加权。

---

## 9. Replay94 数据集

### 9.1 路径

```text
/home/dase-hw101/franka_ws/dataset/snkdjn/
quest3_franka_dualcam_replay_94eps_v1
```

manifest：

```text
/home/dase-hw101/franka_ws/dataset_manifests/
quest3_franka_dualcam_replay_94eps_v1.json
```

构建器：

```text
/home/dase-hw101/franka_ws/scripts/
build_quest3_franka_dualcam_replay_94eps.py
```

构建器 SHA256：

```text
7859d24381e64ea3aeaad1c9fc41a79574babc34a96602122c0b60115c7d07c3
```

### 9.2 固定组成

```text
Base74: 74
Front:  10
Back:   10
Total:  94

Frames: 22,449
Videos: 188
Size: approximately 239 MB
```

### 9.3 审计结果

```text
REPLAY94_DATASET_AUDIT=PASS
episodes=94
frames=22449
videos=188
unique_source_identities=94
```

审计内容：

- 94条source identity唯一。
- base74与append20没有重叠。
- episode indices为0–93连续。
- global frame index连续。
- parquet长度与episodes metadata一致。
- 每条episode都有primary和wrist两个非空视频。
- task_index统一为0。
- modality继承自已验证的dual-camera 74数据集。

### 9.4 训练包

```text
/home/dase-hw101/franka_ws/artifacts/
starvla_replay94_training_bundle_20260810.tar.gz
```

大小：

```text
235 MB
```

SHA256：

```text
ddc9c5b8d11e619a046d2a3427154523e1063e18873ae261bd4c562f90575325
```

相关脚本：

```text
scripts/register_quest3_franka_replay94_on_server.py
scripts/install_replay94_training_on_server.sh
scripts/start_starvla_replay94_from_74_train.sh
scripts/start_spatial_forcing_replay94_from_74_train.sh
scripts/build_replay94_training_bundle.sh
```

---

## 10. Replay94 训练

### 10.1 训练配置

```text
initialization:
  successful Libero30k + 74eps 20k checkpoint

dataset:
  quest3_franka_dualcam_replay_94eps_v1

max steps:
  5000

action_model_lr:
  3e-5

qwen_vl_interface_lr:
  1e-7

base_lr:
  1e-6

vision tower:
  frozen

repeated diffusion steps:
  8

optimizer:
  fresh AdamW

save policy:
  final checkpoint only
```

这是低学习率 replay adaptation，不是从 Libero checkpoint 重新完整训练，也不是只在新增20条数据上做容易遗忘旧技能的 sequential fine-tuning。

### 10.2 第一次启动 OOM

第一次后台 launcher PID：

```text
319912
```

错误：

```text
torch.OutOfMemoryError
Tried to allocate 14.99 GiB
GPU total capacity: 95.08 GiB
free: approximately 13.20 GiB
```

原因：GPU 0 同时运行旧 Control policy server PID `313425`，占用约9.87 GB。训练进程已经使用约72 GB，optimizer step还需一次性申请约15 GB，总需求超过96 GB GPU容量。

GPU 1只有约80 GB，不适合该训练峰值。正确处理是释放自己的GPU 0 policy server，而不是把训练转移到80 GB GPU。

### 10.3 Retry1 成功

成功 run：

```text
replay94_from_successful_libero74_5k_retry1_20260810
```

结束结果：

```text
5000/5000 steps
elapsed: 31 min 20 sec
speed: 2.66 it/s
epoch: 0.22
action_dit_loss: 0.0721031
mse_score: 0.0347775
timing/model: 0.3566 s
```

`epoch=0.22` 是预期的，因为这是5k低学习率适配，而94条数据共22,449帧。它不是完整一轮重新训练。

最终 checkpoint：

```text
/data/hanyu/starVLA_runs/
replay94_from_successful_libero74_5k_retry1_20260810/
final_model/pytorch_model.pt
```

---

## 11. Replay94 open-loop 结果

### 11.1 7-episode regression

使用：

```text
0047 0077 0121 0131 0142 0150 0151
```

这些episode已经属于94条训练数据，因此只能用于检查是否遗忘旧74能力，不是真正holdout。

结果：

```text
queries: 224
first_l2_mean: 0.020724
first_l2_median: 0.001993
first_l2_p90: 0.006809
chunk_l2_mean: 0.022820
first_xyz_l2_mean: 0.002876
first_gripper_abs_mean: 0.017857
y_mae: 0.001070
y_sign_acc: 0.892562
pred_y_mean: -0.001156
gt_y_mean: -0.001002
gripper_binary_accuracy: 0.982143
false_close_rate_when_gt_open: 0.033058
missed_close_rate_when_gt_closed: 0.000000
first_pred_close_frame: 110
first_gt_close_frame: 115
first_close_frame_error: -5
```

与原74模型的相同规模历史结果相比：

```text
first_l2_mean: approximately 0.02585 -> 0.02072
first_xyz_l2_mean: approximately 0.00355 -> 0.00288 m
gripper accuracy: approximately 0.9777 -> 0.9821
false close: approximately 0.0410 -> 0.0331
```

说明 replay 没有在这些训练内轨迹上出现明显遗忘。

### 11.2 Reserve 055

055没有加入94训练集，是今天唯一明确冻结的空间reserve。

结果：

```text
queries: 41
first_l2_mean: 0.052534
first_l2_median: 0.002842
first_l2_p90: 0.008030
chunk_l2_mean: 0.058131
first_xyz_l2_mean: 0.003802
first_gripper_abs_mean: 0.048780
y_mae: 0.001130
y_sign_acc: 0.875000
gripper_binary_accuracy: 0.951220
false_close_rate_when_gt_open: 0.000000
missed_close_rate_when_gt_closed: 0.133333
first_pred_close_frame: 65
first_gt_close_frame: 60
first_close_frame_error: +5
```

解释：

- XYZ误差约3.8 mm，允许进入保守真机前检查。
- false close为0，比过早关闭更安全。
- missed close为13.3%，可能表现为在cube附近稍微犹豫或晚闭合。
- close比GT晚5帧。
- 只有一个reserve episode，不能据此证明广泛空间泛化。

---

## 12. Replay94 live/dry-run 状态

### 12.1 Policy server

```text
host: 192.168.1.113
port: 10112
POLICY_SERVER_CONNECTION=PASS
```

### 12.2 8-step dry-run

dry-run没有使用 `--execute`，机器人未移动。

最后状态：

```text
raw_policy_state:
[0.3088916, 0.0024954, 0.5855340,
 -3.1286872, 0.0190955, -0.7665595,
 0.0016040]

measured total gripper width:
0.0798717 m

raw/ensembled gripper values:
[1,1,1,1,1,1,1,1]

step 7 dpos:
[0.004608, 0.000003, -0.003843]

target_pos:
[0.3375, 0.0032, 0.5636]

execute:
False
```

gripper stability：

```text
policy_requests: 8
raw_first_action_switches: 0
ensembled_first_action_switches: 0
temporal_chunk_intent_changes: 0
latch_suppressed_open_requests: 0
synchronized_close_hold_enabled: True
synchronized_close_hold_active: False
synchronized_close_hold_activations: 0
physical_grasp_attempts: 0
grasp_confirmed: False
episode_completed: False
```

解释：

- 8次推理都要求保持夹爪打开。
- 没有提前close。
- 没有gripper action抖动。
- 动作方向为向前并下降，符合任务初期接近趋势。
- `synchronized_close_hold_active=False` 只表示当前还没有进入close阶段，不表示该功能被禁用。
- 当前末端在标准初始位置附近，夹爪正常全开。

---

## 13. 当前第一条真机 pilot 设置

建议只先执行一条，使用：

```text
policy port: 10112
max steps: 400
execution horizon: 1
temporal ensemble window: 1
rate: 5 Hz
publish rate: 40 Hz
translation scale: 1.0
max translation delta: 0.006 m
max rotation delta: 0.003 rad
gripper confirmations: 3
chunk consensus: 0.75
max grasp attempts: 1
synchronized close hold: enabled
minimum measured lift: 0.030 m
lift timeout: 8.0 s
```

工作区：

```text
x: [0.28, 0.57]
y: [-0.265, 0.10]
z: [0.03, 0.70]
```

重要：

- 不使用object-specific close XYZ gate。
- 不自动抬升。
- 不自动放置或强制release。
- synchronized close hold只补偿模型close确认期间的通信和跟踪延迟。
- 第一次真机只记录一条，不立即做20次SR测试。

真机日志应保存到：

```text
/home/ros/ros2_ws/deployment_logs/replay94_real/
```

---

## 14. 当前不能下的结论

截至本总结：

1. 不能宣称 Replay94 真机成功率提高，因为第一条新模型真机pilot结果尚未报告。
2. 不能宣称 SF 提高了空间泛化；spatial30 offline是mixed，真机只有一次contact-assisted成功。
3. 不能宣称 SF 无效；当前30条实验与原74 baseline并不完全matched。
4. 不能把 open-loop L2 改善等同于闭环真机SR改善。
5. 不能把当前系统性“behind cube”偏差继续归因于确认次数或target tracking；后者已通过同步hold修复和日志证据基本排除。

---

## 15. 下一步实验顺序

### Step 1：Replay94 单条真机 pilot

使用当前标记的 Middle cube 位置和端口10112执行一条完整任务。

记录：

- 是否到达cube中心。
- 是否edge contact。
- close measured pose。
- cube实际位置。
- 第一次有效接触方式。
- grasp width是否稳定在22–36 mm。
- 是否模型自行抬升至少30 mm。
- 是否模型自行移动到box并release。
- 是否出现workspace abort或stale observation。

### Step 2：与旧74做同位置paired comparison

若 Replay94 pilot行为合理，在完全相同的cube、box、标准初始位置和安全过滤下比较：

```text
old Libero74
vs
Replay94 from Libero74
```

先每边3–5条，不立即做20条。

### Step 3：决定是否继续 Replay94

若新模型：

- 保留原Middle能力；且
- 对Front/Back插值更稳定；且
- 没有增加过早close；

再做正式20-trial SR。

若仍固定在cube后方close，应优先：

1. 用teleop把打开的gripper置于cube正上方，记录实际 robot-frame cube XY。
2. 对照模型close measured XY。
3. 检查相机画面中的cube横向位置是否落在训练分布。
4. 不应立即增加人工XYZ offset或object-specific gate。

### Step 4：之后才继续Matched Spatial Forcing

已准备脚本：

```text
/home/hanyu/starVLA/start_spatial_forcing_replay94_from_74_train.sh
```

后续 matched pair：

```text
Control:
  same Libero74 initialization
  same Replay94 data
  same seed/steps/config
  projected alpha = 0

Treatment:
  same Libero74 initialization
  same Replay94 data
  same seed/steps/config
  projected alpha = 0.1
```

在 Replay94 标准 baseline 未完成真机验证前，不应先消耗GPU启动该pair。

---

## 16. 安全与恢复命令

### 16.1 若abort后夹爪仍持有或关闭

```bash
timeout 2 ros2 topic pub -r 20 \
  /gripper/gripper_position_controller/commands \
  std_msgs/msg/Float64MultiArray \
  "{data: [1.0]}"
```

### 16.2 返回标准位置

```bash
python3 /home/ros/ros2_ws/scripts/franka_return_to_standard.py \
  --safe-lift-z 0.50 \
  --execute
```

标准位置近似：

```text
x≈0.309
y≈0.001–0.003
z≈0.585
gripper=open
```

### 16.3 当前真机测试前必须运行

- Franka controller。
- 双RealSense camera container/topics。
- gripper state publisher。
- Replay94 policy server `10112`。

Quest bridge和recording process不是VLA部署必需项。

---

## 17. 今日生成或修改的关键文件

```text
/home/dase-hw101/franka_ws/810summary.md

/home/dase-hw101/franka_ws/dataset_manifests/
quest3_franka_spatial_balanced_30eps_v1.json
quest3_franka_dualcam_replay_94eps_v1.json

/home/dase-hw101/franka_ws/scripts/
build_quest3_franka_spatial_balanced_30eps.py
build_quest3_franka_dualcam_replay_94eps.py
register_quest3_franka_replay94_on_server.py
install_replay94_training_on_server.sh
start_starvla_replay94_from_74_train.sh
start_spatial_forcing_replay94_from_74_train.sh
build_replay94_training_bundle.sh
starvla_franka_delta_pose_client.py

/home/dase-hw101/franka_ws/tests/
test_starvla_synchronized_close_hold.py

/home/dase-hw101/franka_ws/artifacts/
spatial30_front_middle_back_reference.png
spatial30_middle_representative_0131.png
starvla_replay94_training_bundle_20260810.tar.gz
```

---

## 18. 最终状态一句话

今天把问题从“夹爪为何总在cube后面闭合”分解为执行同步与模型空间定位两部分，修复并验证了执行同步问题；随后保留原74模型能力、加入20条新空间数据构建Replay94并完成5k低学习率训练。Replay94的open-loop、reserve-055和8-step dry-run均通过，当前已准备进行第一条保守真机闭环pilot，但尚不能在得到该pilot及paired trials结果之前宣称真机SR提高。
