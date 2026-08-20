# 2026-08-05 StarVLA Spatial Forcing / Phase 10 / Franka 工作总结

日期：2026-08-05  
时区：Asia/Hong_Kong  
本地工作区：`/home/dase-hw101/franka_ws`  
服务器代码：`/home/hanyu/starVLA`  
机器人容器工作区：`/home/ros/ros2_ws`  
任务：`pick up the cube and place it on the box`

---

## 0. 今日最终结论

今天完成了 Spatial Forcing 路线从早期 relational/scene-memory 实验，到独立 Phase 10 clean reproduction，再到 `alpha=0/0.1/0.5` 三组 5k matched training、九次离线 open-loop 和一次真实 Franka pilot 的完整闭环。

最重要的结论如下：

1. 冻结 VGGT teacher、Qwen all-linear LoRA、projected cosine alignment、student/teacher 配对图像增强已经成功接入 StarVLA。
2. RGB-only inference export 和不加载 VGGT teacher 的 policy server 路径已经打通。
3. Phase 10 clean control/treatment 的唯一目标变量是 projected alignment 权重 `alpha`。
4. `alpha=0.5` 和 `alpha=0.1` 都能让 alignment head 和 Qwen LoRA 收到有效梯度。
5. `alpha=0.5` 的 projected alignment loss 在 5k 训练中显著下降，证明 teacher-to-student feature alignment 确实发生。
6. 但是 alignment 的优化成功没有转化为更好的离线动作预测。
7. 三轮 open-loop 平均结果中，`alpha=0` control 仍然是整体最稳定的 Phase 10 模型。
8. `alpha=0.1` 相比 `alpha=0`：

   ```text
   first XYZ L2:      +1.45%（更差）
   chunk XYZ L2:      -0.32%（极小改善）
   first rotation:    +3.01%（更差）
   gripper accuracy:  -0.45 percentage point
   false-close:       +53.85%
   missed-close:      -15.38%
   ```

9. `alpha=0.1` 的 missed-close 下降但 false-close 上升，表明模型更倾向于提前关闭，而不是整体夹爪预测更准确。
10. 因此当前结果只能写成：

    ```text
    technical reproduction = PASS
    optimization/alignment = PASS
    behavioral improvement = NOT DEMONSTRATED
    real-robot SR improvement = NOT DEMONSTRATED
    ```

11. 今天做了一次 `alpha=0.1` 真机 exploratory pilot；策略到达放置区域附近后触发 workspace abort：

    ```text
    target=[0.5708, -0.2508, 0.2230]
    x allowed=[0.28, 0.57]
    y allowed=[-0.265, 0.1]
    ```

    其中 x 超出上限约 `0.8 mm`。episode 没有完整结束，应记录为 failure/incomplete，而不是成功。
12. 真机中止后，默认回位脚本报告抬升后的单段回位距离 `0.410 m`，超过现有 `0.40 m` 安全限制。不能放宽限制；建议把垂直安全抬升高度由 `0.35 m` 提高到 `0.45 m` 后再回标准位。
13. 截至本文写入时，机器人是否已经完成 `z=0.45` 路径回位尚未确认，状态必须视为 pending。
14. 不建议继续盲目扫描更多 alpha，也不建议基于当前证据直接进行 Spatial Forcing 20k 或正式真机 SR 测试。
15. 真机性能主 baseline 仍然是旧的 Libero-init 74eps、vision-frozen、20k checkpoint，历史测试约 `13/20 = 65%`。
16. 后续更合理的方向是：固定 Libero74 baseline、整理 failure taxonomy、优先 HG-DAgger/gripper correction，再准备 bounded residual RL；如果继续 3D prior，则转向带相机标定和 robot-frame metric geometry 的路线。

---

## 1. 进入今天工作前的 baseline

### 1.1 当前最可靠的真机 checkpoint

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

主要设置：

- 初始化：Libero30k StarVLA checkpoint；
- 数据：74 条人工采集双相机 Franka demonstrations；
- 标准 `QwenGR00T` framework；
- 冻结 Qwen visual tower；
- Qwen-VL learning rate：`1e-7`；
- action-model learning rate：`1e-4`；
- repeated diffusion steps：8；
- 训练：20,000 optimizer steps，约 1.15 epoch；
- 不使用 VGGT；
- 不使用 Spatial Forcing LoRA；
- 不使用 Phase 10 paired crop/ColorJitter。

历史真机结果：

```text
13/20 success = 65%
```

该成功率属于带通用安全过滤的 VLA deployment，不是完全无过滤 raw policy。

### 1.2 Libero30k 初始化 checkpoint

```text
/data/hanyu/starVLA_checkpoints/
libero_all_gr00t_official_30000_rerun/
final_model/pytorch_model.pt
```

今天所有 Phase 10 matched 训练都从这个 checkpoint 独立开始，而不是从旧 Libero74 20k checkpoint 继续训练。

---

## 2. Phase 9 scene-memory 实验

今天早期继续了 8 月 4 日的 relational/scene-memory 探索。

### 2.1 20-step smoke

Run：

```text
/data/hanyu/starVLA_runs/
qwengroot_phase9_scene_memory_smoke20_20260805_013229
```

step 20 的主要结果：

```text
action_dit_loss:                 0.1054329
total_loss:                      0.1148348
projected_alignment_loss:        0.4647400
weighted_projected_loss:         0.0092948
relational_alignment_loss:       0.0395373
scene_relational_loss:           0.0053566
weighted_scene_relational_loss:  0.0001071
scene_queue_fill:                20
LoRA-B update norm:              0.0016383
alignment-head update norm:      0.0048529
```

Smoke test 证明 scene memory queue、loss、gradient 和 checkpoint 路径均能工作。

### 2.2 500-step scene-memory pilot

Run：

```text
/data/hanyu/starVLA_runs/
qwengroot_phase9_scene_memory500_proj002_scene04_20260805_014220
```

W&B：

```text
https://wandb.ai/u3666250-the-university-of-hong-kong/
starVLA_Quest3_Franka/runs/e80uuy3f
```

训练正常完成 500/500；最终 summary 包括：

```text
action_dit_loss:                   0.00781
alignment_loss:                    0.06338
mse_score:                         0.02701
alignment_head norm:               1.42355
spatial_forcing_lora_B norm:       0.13654
alignment_head update norm:        0.00239
```

### 2.3 representation audit

Phase 9 treatment 相对 alpha0 control：

```text
delta linear CKA:        -0.000125997
delta position RSA:      -0.006143615
delta shared-probe loss: +0.000015437
```

三个方向都没有提供有说服力的空间表示改善。由此决定停止继续叠加 relational/scene-memory loss，建立独立的 Phase 10 clean reproduction。

---

## 3. Phase 10 clean Spatial Forcing reproduction

### 3.1 设计目的

Phase 10 删除此前所有实验性附加项：

- 删除 relational alignment；
- 删除 scene-relational loss；
- 删除 memory queue；
- 不使用 DAv2 gated cross-attention；
- 只保留 training-only frozen VGGT projected feature alignment。

目标函数：

```text
L_total = L_action + alpha * L_projected_alignment
```

其中：

- `L_action`：StarVLA 7D action diffusion loss；
- `L_projected_alignment`：Qwen student token 与 frozen VGGT teacher token 的 projected cosine alignment；
- `alpha`：空间 teacher 梯度权重；
- `alpha=0`：matched control；
- `alpha>0`：Spatial Forcing treatment。

### 3.2 matched 设置

所有 Phase 10 control/treatment 共同使用：

- 同一 Libero30k 初始 checkpoint；
- 同一批 74 dual-camera episodes；
- `include_state: false`；
- `QwenGR00TSpatialForcingClean` framework；
- frozen VGGT teacher；
- Qwen all-linear LoRA；
- LoRA rank 32、LoRA alpha 16；
- Gaussian no-op LoRA initialization；
- student/teacher 共享同一个 random crop 和 ColorJitter realization；
- DiT-B 7D action head；
- batch size 1；
- seed 42；
- 相同 optimizer/LR/scheduler；
- 5,000 training steps；
- 最终 checkpoint only，避免频繁写入 9.4 GB 中间模型。

唯一预期变量：

```text
projected_alignment_alpha
```

### 3.3 主要代码

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

/home/dase-hw101/franka_ws/PHASE10_CLEAN_REPRODUCTION.md
```

服务器安装后的 RGB export 工具实际位置是：

```text
/home/hanyu/starVLA/export_spatial_forcing_rgb_view.py
```

不是：

```text
/home/hanyu/starVLA/scripts/export_spatial_forcing_rgb_view.py
```

今天曾因使用错误路径导致一次 `No such file or directory`，随后已纠正。

### 3.4 已完成的工程修复

包括：

- Spatial module import 路径；
- `peft==0.18.1` 环境；
- BF16 tensor 转 NumPy；
- teacher keys 在 RGB-only inference load 中的过滤；
- `strict=False` compatible loading；
- all-linear LoRA wiring；
- paired augmentation；
- alignment-head 和 LoRA update audit；
- Accelerate 非零物理 GPU 映射；
- checkpoint save interval；
- RGB-only metadata/symlink export；
- policy server 不加载训练期 VGGT teacher。

相关 alignment、LoRA、clean framework 单元测试和 smoke training 均已通过。

---

## 4. Phase 10 smoke 和 500-step pilot

### 4.1 20-step alpha0 control

```text
action_dit_loss:                   0.1743300
total_loss:                        0.1743300
projected_alignment_loss:          1.0012270
weighted_projected_alignment_loss: 0
LoRA-B update norm:                0.0013819
alignment-head update norm:        0
```

符合 control 预期：action/LoRA 更新，但 alignment head 不受梯度影响。

### 4.2 20-step alpha0.5 treatment

```text
action_dit_loss:                   0.1846437
total_loss:                        0.4073924
projected_alignment_loss:          0.4454975
weighted_projected_alignment_loss: 0.2227487
LoRA-B update norm:                0.0012914
alignment-head update norm:        0.0043660
```

符合 treatment 预期：teacher alignment 梯度进入 alignment head 和 Qwen LoRA。

### 4.3 500-step matched pilot

Control：

```text
run: phase10_clean_control_alpha0_pilot500_20260805_032518
W&B: https://wandb.ai/u3666250-the-university-of-hong-kong/
     starVLA_Quest3_Franka/runs/9x9lznrz
```

Treatment：

```text
run: phase10_clean_treatment_alpha05_pilot500_20260805_033422
W&B: https://wandb.ai/u3666250-the-university-of-hong-kong/
     starVLA_Quest3_Franka/runs/0yv2vcva
```

解析后的 500-step 统计：

```text
Control action loss mean:          0.343118
Control last100 mean:              0.225587
Control projected loss last100:    0.977643

Treatment action loss mean:        0.348308
Treatment last100 mean:            0.226514
Treatment projected loss last100:  0.080804
```

结论：alignment 显著学习，action loss 没有明显崩坏，因此进入 5k matched pilot。

---

## 5. Phase 10 5k control 和 alpha0.5 treatment

### 5.1 Control alpha0

Run：

```text
/data/hanyu/starVLA_runs/
phase10_clean_control_alpha0_5k_20260805_042006
```

Checkpoint：

```text
/data/hanyu/starVLA_runs/
phase10_clean_control_alpha0_5k_20260805_042006/
final_model/pytorch_model.pt
```

W&B：

```text
https://wandb.ai/u3666250-the-university-of-hong-kong/
starVLA_Quest3_Franka/runs/2sjptexd
```

### 5.2 Treatment alpha0.5

Run：

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha05_5k_gpufix_20260805_045615
```

Checkpoint：

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha05_5k_gpufix_20260805_045615/
final_model/pytorch_model.pt
```

W&B：

```text
https://wandb.ai/u3666250-the-university-of-hong-kong/
starVLA_Quest3_Franka/runs/k67hgsre
```

### 5.3 GPU 映射问题

第一次并行训练时，runner 的 Accelerate 参数曾把非零 `GPU_ID` 静默映射回 GPU 0，造成 control 和 treatment 同时占用 GPU 0。

处理：

- 停止错误启动的 treatment；
- 修正 runner，使 `CUDA_VISIBLE_DEVICES` 与 `accelerate --gpu_ids` 一致；
- control 保留在 GPU 0；
- treatment 在 GPU 1 重新训练；
- 两个最终 run 都完整结束。

### 5.4 5k 训练曲线比较

解析到 4,996 个常规 logging step：

```text
Control action loss mean:          0.230573
Treatment action loss mean:        0.231136

Control last1000 action:           0.198347
Treatment last1000 action:         0.196945

Control last500 action:            0.198142
Treatment last500 action:          0.195993

Control projected loss last500:    0.992766
Treatment projected loss last500:  0.048033

Control MSE checkpoints mean:       0.035804
Treatment MSE checkpoints mean:     0.036576
```

解释：

- teacher alignment 学习非常明显；
- treatment action loss 与 control 基本相同；
- MSE 没有稳定优于 control；
- 需要离线 closed-distribution action evaluation，而不能只看训练 loss。

---

## 6. alpha0 / alpha0.5 三轮 open-loop

### 6.1 评估协议

数据集：

```text
quest3_franka_dualcam_test_0047
quest3_franka_dualcam_test_0077
quest3_franka_dualcam_test_0099
quest3_franka_dualcam_test_0121
quest3_franka_dualcam_test_0149
quest3_franka_dualcam_test_0150
quest3_franka_dualcam_test_0151
```

统一设置：

```text
stride=5
max_queries_per_episode=32
compare=both
224 queries/run
3 independent stochastic inference runs/model
```

Control server：

```text
port 10100
```

Alpha0.5 server：

```text
port 10101
```

容器内 CSV：

```text
/home/ros/ros2_ws/deployment_logs/open_loop/
phase10_control5k_20260805_073218.csv
phase10_control5k_repeat2.csv
phase10_control5k_repeat3.csv

phase10_treatment5k_20260805_073218.csv
phase10_treatment5k_repeat2.csv
phase10_treatment5k_repeat3.csv
```

### 6.2 三轮结果

```text
Metric                   alpha0 control       alpha0.5
first XYZ L2             0.004525 ± 0.000156 0.004689 ± 0.000140
chunk XYZ L2             0.004587 ± 0.000064 0.004617 ± 0.000018
first rotation L2        0.000349 ± 0.000011 0.000365 ± 0.000011
chunk rotation L2        0.000351 ± 0.000002 0.000358 ± 0.000002
gripper accuracy         94.20% ± 0.89%       93.60% ± 1.29%
false-close              3.55% ± 0.95%        4.64% ± 2.50%
missed-close             8.50% ± 1.13%        8.50% ± 2.04%
```

Pooled gripper counts：

```text
alpha0:
  false-close 13/366
  missed-close 26/306

alpha0.5:
  false-close 17/366
  missed-close 26/306
```

结论：alpha0.5 没有表现出稳定空间动作改善，并增加 false-close。

---

## 7. 为什么又训练 alpha0.1

Alpha 控制 teacher alignment 在总 loss 中的权重：

```text
L_total = L_action + alpha * L_alignment
```

Alpha 不是 learning rate，也不是置信度。

在 alpha0.5 的最后阶段：

```text
projected loss ≈ 0.048
weighted loss ≈ 0.024
action loss ≈ 0.196
```

alignment 项约为 action loss 的 12%。因此提出 alpha0.1：仍然提供 VGGT 空间约束，但把同样规模的 weighted alignment 降至约 action loss 的 2.4%，测试是否能减少对原动作/夹爪表征的扰动。

这不是预期 alpha0.1 一定更好，而是一个预注册的弱约束诊断。

---

## 8. Alpha0.1 5k training

### 8.1 Run

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha01_5k_20260805_080954
```

Checkpoint：

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha01_5k_20260805_080954/
final_model/pytorch_model.pt
```

W&B：

```text
https://wandb.ai/u3666250-the-university-of-hong-kong/
starVLA_Quest3_Franka/runs/6uxfvvkv
```

### 8.2 训练结束 summary

```text
5000/5000
wall time:                    51m24s
epoch:                        0.29
action_dit_loss:              0.12045
mse_score:                    0.06293
alignment_head norm:          1.44825
LoRA-B norm:                  0.40199
alignment-head update norm:   0.00286
LoRA-B update norm:           0.00168
```

### 8.3 训练结束后的 SIGKILL

Trainer 已先打印：

```text
Training complete. Final model saved
... and that's all, folks!
```

约 10 秒后 Accelerate/Elastic 报：

```text
Signal 9 (SIGKILL) received by PID 4175522
TRAIN_STATUS=1
```

审计结果：

```text
checkpoint size:          10,157,101,954 bytes
size check:              PASS
zip members:             1684
zip central directory:   PASS
kernel OOM record:       none found
```

之后 policy server 完整加载 checkpoint 并监听端口 `10102`，进一步证明保存的 checkpoint 可用。

因此该 SIGKILL 被判断为训练完成后的 launcher/teardown 异常，而不是训练中断、CUDA OOM 或 checkpoint 损坏。

### 8.4 RGB-only export 路径问题

最初错误使用：

```text
/home/hanyu/starVLA/scripts/export_spatial_forcing_rgb_view.py
```

实际服务器安装路径：

```text
/home/hanyu/starVLA/export_spatial_forcing_rgb_view.py
```

纠正后创建 RGB-only inference view，checkpoint 通过绝对符号链接引用 source 权重，没有复制额外 10 GB 模型。

Alpha0.1 policy server：

```text
host: 192.168.1.113
port: 10102
status observed: server listening on 0.0.0.0:10102
```

---

## 9. alpha0.1 三轮 open-loop

容器内 CSV：

```text
/home/ros/ros2_ws/deployment_logs/open_loop/
phase10_alpha01_5k_repeat1.csv
phase10_alpha01_5k_repeat2.csv
phase10_alpha01_5k_repeat3.csv
```

每轮 224 queries。

### 9.1 每轮结果

```text
Run 1:
  first XYZ:       0.004432
  chunk XYZ:       0.004539
  gripper acc:     94.64%
  false-close:     5.74%
  missed-close:    4.90%

Run 2:
  first XYZ:       0.004727
  chunk XYZ:       0.004574
  gripper acc:     92.41%
  false-close:     4.92%
  missed-close:    10.78%

Run 3:
  first XYZ:       0.004614
  chunk XYZ:       0.004605
  gripper acc:     94.20%
  false-close:     5.74%
  missed-close:    5.88%
```

### 9.2 三模型统一结果

```text
Metric                   alpha0             alpha0.1           alpha0.5
first XYZ L2             0.004525           0.004591           0.004689
chunk XYZ L2             0.004587           0.004573           0.004617
first rotation L2        0.000349           0.000359           0.000365
chunk rotation L2        0.000351           0.000361           0.000358
first total L2           0.062415           0.066939           0.068521
chunk total L2           0.075833           0.074739           0.074392
Y MAE                    0.001633           0.001606           0.001671
Y sign accuracy          78.14%             73.50%             77.87%
gripper accuracy         94.20%             93.75%             93.60%
false-close              3.55%              5.46%              4.64%
missed-close             8.50%              7.19%              8.50%
```

### 9.3 Alpha0.1 相对 control

```text
first XYZ:       +1.452% worse
chunk XYZ:       -0.317% better
first rotation:  +3.007% worse
chunk rotation:  +2.748% worse
first total L2:  +7.248% worse
chunk total L2:  -1.443% better
Y MAE:           -1.654% better
gripper accuracy:-0.474% relative
false-close:     +53.846% relative
missed-close:    -15.385% relative
```

三轮配对 first XYZ 差值：

```text
-0.0002343
+0.0001742
+0.0002571
```

即只有第一轮优于 control，方向不稳定。

Pooled gripper counts：

```text
alpha0:
  false-close 13/366
  missed-close 26/306

alpha0.1:
  false-close 20/366
  missed-close 22/306

alpha0.5:
  false-close 17/366
  missed-close 26/306
```

Alpha0.1 的行为更像把 close timing 向前移动：减少一部分 missed-close，但制造更多 false-close。对真实抓取而言，false-close 是关键风险，因为它对应在到达有效抓取区域前关闭。

---

## 10. Control alpha0 与旧 Libero74 checkpoint 的区别

这两者都没有有效的 VGGT alignment，但不是同一个模型。

```text
Setting                   old Libero74             Phase10 alpha0
framework                 QwenGR00T                QwenGR00TSpatialForcingClean
initial checkpoint        Libero30k                Libero30k
data                      same 74 episodes         same 74 episodes
training steps            20,000                   5,000
Qwen update               visual frozen/LR 1e-7   all-linear LoRA/LR 1e-4
action LR                 1e-4                     1e-4
diffusion repeats         8                        1
VGGT                      absent                   present but weight=0
alignment head            absent                   present, no update
paired augmentation       absent                   enabled
real robot evidence       13/20                    no formal SR
```

因此：

- old Libero74 是真机 deployment baseline；
- Phase10 alpha0 是科学 matched control；
- Phase10 alpha0.1/0.5 是 Spatial Forcing treatments。

不能用 Phase10 alpha0 直接替代旧 Libero74，也不能把两者的差异归因于 Spatial Forcing。

---

## 11. 一次 alpha0.1 真实 Franka pilot

### 11.1 使用设置

Policy server：

```text
192.168.1.113:10102
```

部署尽量复用原 `13/20` 配置：

```text
max_steps=600
execution_horizon=2
rate=10
publish_rate=40
translation_scale=1.5
max_trans_delta=0.009
max_rot_delta=0.003
min_y=-0.265
max_observation_age=1.0
gripper_switch_confirmations=3
gripper_chunk_consensus=0.75
temporal_ensemble_window=1
max_grasp_attempts=1
grasp_close_width_timeout=5.0
```

这是 safety-filtered VLA deployment；通用安全过滤保留，未加入固定物体坐标或自动任务动作。

### 11.2 结果

真实执行中止：

```text
RuntimeError: Abort: policy target left the XY workspace:
target=[0.5708, -0.2508, 0.2230]
x=[0.28, 0.57]
y=[-0.265, 0.1]
```

解释：

- target x 比上限多约 `0.0008 m`；
- y 仍在当前扩展下限内；
- workspace safety filter 正常工作；
- 不能因为只超出 0.8 mm 就把它记为成功；
- episode 没有通过完整 place/release/stability 成功条件；
- 此 trial 应记录为 failure/incomplete。

这一次 pilot 不能用于计算正式 SR，也不能推翻三轮 open-loop 结论。

---

## 12. Franka 回标准位问题

### 12.1 默认回位失败

真机 rollout 中止后运行标准回位脚本，得到：

```text
RuntimeError: 单段位移 0.410 m 超过 0.40 m 安全限制
```

`franka_return_to_standard.py` 的默认逻辑：

- 如果当前 z 低于 `0.35`，先垂直抬升到 `z=0.35`；
- 然后从该点直接插值到训练标准位；
- 任意单段平移不得超过 `0.40 m`；
- 任意单段旋转不得超过 90 度。

当前位置较靠近放置区域。抬升到 `z=0.35` 后，到标准位仍需 0.410 m，所以脚本安全拒绝。

### 12.2 正确处理

不能将 `0.40 m` 限制放宽到 0.42 或更大。

推荐把 safe lift 提高到 `z=0.45`：

```bash
python3 /home/ros/ros2_ws/scripts/franka_return_to_standard.py \
  --safe-lift-z 0.45 \
  --lift-duration 7.0 \
  --move-duration 10.0
```

这只是 dry-run。确认没有 segment-limit error 后，才运行：

```bash
python3 /home/ros/ros2_ws/scripts/franka_return_to_standard.py \
  --safe-lift-z 0.45 \
  --lift-duration 7.0 \
  --move-duration 10.0 \
  --execute
```

执行前条件：

- `/target_pose` publisher count 为 0；
- controller 仍订阅 `/target_pose`；
- `/current_pose` 正常更新；
- 垂直上方无障碍；
- 夹爪没有抓着方块，否则脚本会在标准高位打开夹爪；
- 操作者可以立即触达急停。

截至总结写入时，以上回位是否已经实际完成尚未收到确认。

---

## 13. 当前服务器进程/端口记录

今天观察到：

```text
GPU 0 PID 3980356 ~9.87 GB
  likely Phase10 alpha0 5k policy server
  port 10100

GPU 1 PID 3981671 ~9.79 GB
  likely Phase10 alpha0.5 5k policy server
  port 10101

GPU 2 PID 4175522 ~19.46 GB
  alpha0.1 5k training child（训练结束后 SIGKILL）

GPU 3 PID 2281432 ~9.76 GB
  old alpha0 control500 policy server
  port 10099

GPU 3 PID 4458 ~2.93 GB
  未确认的旧 Python 进程
```

Alpha0.1 训练完成后，GPU 2 用于启动新 policy server：

```text
port 10102
```

旧 PID/端口在后续操作前应重新用 `ps` 和 `ss` 验证，不能假定永久不变。

---

## 14. 今天讨论的 RL 路线

结论：可以与 Spatial Forcing 并行进行 RL 的准备工作，但不能同时混合训练或混合真机测试。

推荐 RL baseline：

```text
old Libero74 20k checkpoint
```

不是：

```text
Phase10 alpha0
Phase10 alpha0.1
Phase10 alpha0.5
```

推荐第一条 RL 主路线：

```text
frozen Libero74 StarVLA
       +
zero-initialized bounded 6D residual actor
       +
human-gated data / RLPD or SAC
```

首版 residual 不直接修改 gripper；gripper timing 应优先通过 HG-DAgger、supervised correction 或独立 temporal loss/head 处理。

当前可以并行完成：

- 固定 Libero74 checkpoint manifest/SHA/config/norm stats；
- 整理已有 20 次真机 failure taxonomy；
- observe-only logger；
- intervention/reward/replay schema；
- residual actor/critic mock interface；
- recorded-observation replay。

当前不能启动：

- direct PPO/GRPO 真机训练；
- 在 Spatial model 上直接加 RL；
- 未通过 log-prob parity 的 direct flow RL；
- 无 reward/replay/rollback 的在线探索。

最低 RL 实验矩阵：

```text
Libero74 BC
Libero74 BC + residual RL
```

只有 RL 和 Spatial model 各自独立证明有效后，才测试：

```text
Spatial model BC
Spatial model BC + residual RL
```

---

## 15. 今日研究判断

### 15.1 已经证明的内容

- frozen VGGT teacher 能接入训练；
- student/teacher token 能执行 projected alignment；
- all-linear LoRA 能接收 alignment gradient；
- paired augmentation 路径工作；
- alpha 能正确控制 weighted alignment loss；
- RGB-only inference view 能移除 teacher；
- policy server 能加载 alpha0.1 最终 checkpoint；
- 九次 open-loop 使用相同 7 episodes 和协议完成；
- alpha0.1/0.5 没有表现出稳定优于 alpha0 的离线行为。

### 15.2 尚未证明的内容

- VGGT alignment 提高真机 SR；
- alpha0.1 提高目标定位；
- alpha0.5 提高目标定位；
- Spatial Forcing treatment 优于旧 Libero74 20k；
- 5k 结果可以外推到完整 20k；
- RL 会提高当前真机 SR；
- 当前机器人已经安全返回标准位。

### 15.3 对负结果的正确表述

不能写：

```text
Spatial Forcing 无效。
```

应该写：

```text
在当前 74 条固定区域双相机 Franka 数据、Libero30k 初始化、
5k all-linear-LoRA fine-tuning 和 projected-cosine alignment 设置下，
alpha=0.1 与 alpha=0.5 均未在三轮 held-out open-loop 中
稳定优于 matched alpha=0 control，并表现出更高 false-close 风险。
```

---

## 16. 下一步优先级

### Priority 0：确认机器人安全状态

1. 确认 alpha0.1 deployment client 已退出；
2. 确认 `/target_pose` publisher count 为 0；
3. 确认 gripper 是否抓着方块；
4. 运行 `safe-lift-z=0.45` dry-run；
5. dry-run 通过后执行回标准位；
6. 记录最终 `/current_pose` 与 gripper width。

### Priority 1：冻结 Spatial Forcing Phase 10 结果

保存：

- 三个 5k final checkpoints；
- W&B run IDs；
- 九个 open-loop CSV；
- Phase 10 config；
- RGB inference manifest；
- 训练和 server logs；
- 本总结。

不再盲目尝试更多 alpha。

### Priority 2：回到真机 baseline

将 old Libero74 20k checkpoint 继续作为 deployment baseline。Phase10 alpha0 只是科学 control，不是替代 baseline。

### Priority 3：failure taxonomy 和 HG-DAgger

把真机失败分为：

- 未到达方块；
- 在方块上方停住；
- 过早关闭；
- 抓取失败；
- 抓住但未抬升；
- 到盒子但未释放；
- workspace abort；
- stale observation/controller failure；
- timeout。

针对过早关闭和抓取附近错误收集 human correction。

### Priority 4：如果继续 3D prior

不要继续只调 alignment alpha。转向：

- primary camera calibration `T_B_Cp`；
- wrist hand-eye calibration `T_E_Cw`；
- RealSense aligned metric depth；
- robot-frame point/geometry representation；
- object-relative EEF displacement auxiliary objective；
- 或 training-only 3D teacher distillation，但必须验证其对 robot-frame action sensitivity 的作用。

### Priority 5：RL

先完成 observe-only、replay、intervention 和 reward plumbing；优先 HG-DAgger，再做 frozen-StarVLA bounded residual RL。不要从当前 Spatial treatment 直接进入 online RL。

---

## 17. 当前关键路径速查

### 真机主 baseline

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

### Phase10 alpha0 5k

```text
/data/hanyu/starVLA_runs/
phase10_clean_control_alpha0_5k_20260805_042006/
final_model/pytorch_model.pt
```

### Phase10 alpha0.5 5k

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha05_5k_gpufix_20260805_045615/
final_model/pytorch_model.pt
```

### Phase10 alpha0.1 5k

```text
/data/hanyu/starVLA_runs/
phase10_clean_treatment_alpha01_5k_20260805_080954/
final_model/pytorch_model.pt
```

### Open-loop evaluator（机器人容器）

```text
/home/ros/ros2_ws/scripts/starvla_open_loop_l2_eval.py
```

### 真机 deployment client（机器人容器）

```text
/home/ros/ros2_ws/scripts/starvla_franka_delta_pose_client.py
```

### 回标准位脚本（机器人容器）

```text
/home/ros/ros2_ws/scripts/franka_return_to_standard.py
```

### Phase10 本地说明

```text
/home/dase-hw101/franka_ws/PHASE10_CLEAN_REPRODUCTION.md
```

### 前一天总结

```text
/home/dase-hw101/franka_ws/8.4summary.md
```

---

## 18. 最终一句话

今天成功完成了 Spatial Forcing-style VGGT teacher alignment 的独立、可部署、严格 matched 工程复现，但 `alpha=0.1` 和 `alpha=0.5` 都没有在三轮 held-out open-loop 中稳定优于 `alpha=0`，一次 alpha0.1 真机 pilot 也因 workspace 边界中止；因此下一步应先安全回位并冻结该负结果，再回到 Libero74 baseline，转向 failure-driven HG-DAgger、residual RL 准备或带标定的 robot-frame 3D geometry。
