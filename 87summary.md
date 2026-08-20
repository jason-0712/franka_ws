# 2026-08-07 StarVLA × Residual SAC × Franka 工作总结

日期：2026-08-07  
时区：Asia/Hong_Kong  
工作区：`/home/dase-hw101/franka_ws`  
RL worktree：`/home/dase-hw101/franka_ws/third_party/starVLA_rl_libero`  
分支：`research/rlinf-libero-residual`  
机器人容器：`franka`，用户目录 `/home/ros/qwengroot_rl_execution`  
GPU 服务器：`hanyu@192.168.1.113`  
任务：`pick up the cube and place it on the box`

---

## 0. 今日最终结论

今天完成了 residual SAC 离线训练闭环的 smoke、旧 74 集 demonstration 的 provenance/reward/action/runtime
契约审计、候选 5 Hz zero-residual 真机校准，以及 StarVLA 推理从约 600 ms 级到约 115–135 ms 模型侧
延迟的系统优化和分段诊断。

最重要的结果如下：

1. 主 baseline checkpoint 保持固定，SHA256 未变化。
2. residual SAC actor、critic ensemble、target critic、entropy temperature、Polyak update、checkpoint
   restore 和 exact-next-base replay 已通过 3-step 纯离线 optimizer smoke。
3. 今天确实运行了 3 个 **离线 CPU optimizer steps**；没有在线训练、没有加载 4B base 做梯度更新、没有
   ROS、没有机器人命令。
4. 74 个历史 episode 已正式证明是 curated successful demonstrations，并生成 terminal-success sparse reward。
5. 旧 recorder 的 action 行语义已确定为：

   ```text
   observation row t
   -> source action row t+1
   -> next/post-step observation row t+1
   ```

6. 旧 74 集与当前 5 Hz、9 mm/3 mrad、10 mm tracking gate 的 execution MDP 不兼容：完整兼容
   episode 为 `0/74`，因此正式决定不把旧数据放入 residual SAC Bellman replay。
7. 这不表示 RL 失败。旧数据仍适合 StarVLA supervised baseline、provenance、视觉和动作诊断；只是不能
   假装它们是由当前 safety/execution adapter 产生的 transition。
8. 完成一次 3-step、5 Hz 候选、zero-residual 真机 calibration；三步均发送成功，最大 tracking error
   `7.417 mm`，人工确认机器人行为正常并 accept。
9. calibration 证明当前 controller/safety/recording 链路能执行短程动作，但因为每步仍有人工 typed prompt，
   它不等于连续 5 Hz scheduler 已通过。
10. StarVLA 模型侧推理已优化到典型 `115–135 ms`，动作与相同物理 GPU reference 严格一致，
    `max_abs=0.0`。
11. NUMA 和线程修正把 policy server 从 419 threads 降到 15 threads，并把主要内存放到 GPU1 所在的
    NUMA node 0。
12. WebSocket 之外实现了持久 length-prefixed raw TCP transport，并完成逐阶段网络计时。
13. 最终发现机器人电脑到 GPU server 的路由走 Wi-Fi，而 Franka 使用的有线口位于独立
    `172.16.0.0/24`，不能到 GPU server。
14. 关闭 Wi-Fi powersave 后，roundtrip p50 从约 `204 ms` 降到 `138 ms`，但 p95 仍为
    `228.8 ms`、max `485.5 ms`，没有通过稳定 5 Hz 门槛。
15. 当前主要 blocker 已不是 StarVLA 计算，而是 Wi-Fi 长尾。推荐新增第二个有线 NIC/USB Ethernet 连接
    GPU LAN，同时保留现有 Franka control Ethernet。
16. 今天没有启用非零 residual 真机动作，没有完成 RL training run，没有得到“RL 提高成功率”的实验结论。

当前准确状态：

```text
fixed Libero-init StarVLA baseline:              PASS
formal QwenGR00TRLPolicy adapter:                PASS
residual SAC actor/critic/trainer smoke:         PASS (offline 3 steps only)
74-demo provenance and sparse reward:            PASS
74-demo current execution-MDP compatibility:     FAIL / EXCLUDED FROM SAC REPLAY
5 Hz candidate zero-residual calibration:        PASS (3 human-gated steps)
same-GPU optimized action parity:                PASS, max_abs=0.0
model-side inference latency:                    PASS, typical ~115–135 ms
end-to-end stable 5 Hz latency gate:              FAIL, Wi-Fi p95/max tail
continuous 5 Hz live shadow scheduler:           NOT YET VALIDATED
new on-contract success/failure rollout:         NOT YET COLLECTED
nonzero residual robot execution:                NOT AUTHORIZED
RL success-rate improvement:                     NOT TRAINED / NOT MEASURED
```

---

## 1. 固定 baseline 与身份

主 baseline：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

固定身份：

```text
size:   9,976,833,554 bytes
sha256: 7f018ea7d83f0e87f0f900fec9d8c3cbc7125af1a75ca843cb06b48d74f243d7
```

模型契约：

```text
framework:          QwenGR00T
base VLM:           Qwen3-VL-4B-Instruct
action head:        FlowmatchingActionHead / DiT-B
action chunk:       [B, 8, 7]
action semantics:   [dx,dy,dz,droll,dpitch,dyaw,gripper]
execution horizon:  1
base uses state:    false
gripper residual:   disabled
```

安全配置 identity：

```text
safety_limits_sha256:
d1eaa3acc9b62776287f1936148cdae803cd06b7715b282075234fc817ea88d9
```

当前 RL worktree 是单独创建的研究 worktree；今日 latency 和 residual 代码仍未 commit，不能把工作树误写成
clean release。原 baseline checkpoint 由历史 dirty server checkout 训练得到，其 provenance warning 仍然存在，
但 checkpoint 文件和 SHA 已固定。

---

## 2. Residual SAC 离线训练闭环

主路线保持 frozen base + compact residual actor：

```text
current dual-camera observation
  -> frozen StarVLA base action

current cameras + measured state + base action
  -> compact squashed-Gaussian actor
  -> bounded 6D Cartesian residual

base action + residual
  -> robot-side safety filter
  -> actually executed 7D action
  -> exact measured next observation

critic:
Q(current observation, current base action, actually executed action)

SAC target:
frozen base(exact next observation)
  -> next base action
  -> residual actor
  -> target critic bootstrap
```

关键语义：

- base StarVLA 不被 residual SAC optimizer 更新；
- residual 只作用于 6D Cartesian 维度；
- gripper 永远复制 base policy command；
- critic 必须消费 `executed_action`，不能消费 filter 前 proposal；
- time-limit truncation 仍 bootstrap，真正 termination 才屏蔽 bootstrap；
- SAC 是 off-policy，不需要 PPO 的 rollout-time old logprob；
- exact-next-base 必须在 transition 的真实 `next_observation` 上重新计算，不能把下一条 before observation
  的 base action 左移代替。

### 2.1 `_04` 纯离线 optimizer smoke

输入 episode：

```text
franka_zero_residual_3step_20260806_04
```

结果：

```text
gate:                              AUDITED_RESIDUAL_SAC_OFFLINE_OPTIMIZER
status:                            PASS
device:                            CPU
optimizer steps:                   3
sample count per update:           3
base model loaded:                 false
robot commands sent:               0
action execution:                  NOT_RUN
checkpoint restore:                PASS_EXACT_IDENTITY
actor parameter max change:        8.998e-4
critic parameter max change:       9.015e-4
initial deterministic residual:    exactly 0
final deterministic residual max:  6.849e-6
```

离线 smoke 使用的 residual limits：

```text
[2 mm, 2 mm, 2 mm, 0.5 mrad, 0.5 mrad, 0.5 mrad]
```

这些 limits 只用于证明 actor 能收到非零 Q-gradient，不是部署授权。部署 adapter 的 residual limits 仍是 0。

报告：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
qwengroot_residual_sac_optimizer_smoke_20260807.json
```

trainer state：

```text
sha256: 4aa7d3c8053f5e2b45bd1425a9cf26873495b85324355d8f2ea4676c462d7b0d
```

这个 smoke 只证明公式、gradient、optimizer、target update、checkpoint restore 和数据 plumbing 正确；3 条
transition 不足以证明学习质量，也不能用于 success-rate claim。

---

## 3. 74 集旧 demonstration：成功 provenance、reward 与 action 语义

### 3.1 Curated-success attestation

项目 provenance 明确说明当前 74 个 episode 都是经过筛选的成功示范，因此不再机械地逐集重看 74 次视频。
今天创建了 dataset-level attestation：

```text
all_selected_episodes_are_successful_demonstrations: true
episode_count:                                     74
failure_demonstrations_present:                    false
attestation_id:
b30f8b7d2c6f8d38c8956050220bbe951d9b21974cae4722a24adee3f5250406
```

路径：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
franka_74eps_curated_success_v1.attestation.json
```

episode 0 已完成的双视频人审仍保留为额外证据，但不再要求对余下 73 集重复审核。

### 3.2 Sparse reward bundle

正式 reward 定义为 `terminal_success_sparse_v1`：

```text
17,254 transitions
non-terminal reward: 0
每集最后一条 transition reward: +1
terminations: 74
truncations: 0
reward sum: 74
```

路径：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
franka_74eps_sparse_rewards_v1
```

这一步只解决 outcome/reward provenance，不自动让旧数据满足当前 residual SAC action/dynamics contract。

### 3.3 Source action 的真实行对齐

旧 collector 的执行顺序是先 `env.step(action)`，再把 post-step observation 与 action 写进同一 row。因此：

```text
transition observation[t] -> observation[t+1]
uses source action[t+1]
```

不能使用 `action[t]`。正式 sidecar 保存 controller-target command，但不会把旧 source action 夸大为带有当前
safety adapter receipt 的 `actually_executed_action`。

路径：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
franka_74eps_action_semantics_v1
```

---

## 4. 为什么旧 74 集不能直接进入 residual SAC replay

正式 runtime-contract 审计结果：

```text
total transitions:                         17,254
current-step runtime-exact:                 423
next-state runtime-continuable:             470
strict transition compatible intersection: 330
strict terminal successes:                  5
episodes with any strict coverage:          64
longest contiguous compatible run:          33
fully compatible episodes:                  0 / 74
```

这里的 423、470、330 是不同集合：

- 423 表示当前 action 能原样落在当前 runtime envelope；
- 470 表示 next state 独立满足继续执行条件；
- 330 是 current action、next state 和 tracking/dynamics 条件同时成立的严格 transition；
- 只有 5 条严格 transition 同时是成功 terminal；
- 没有任何完整 episode 从头到尾满足当前 execution MDP。

正式策略：

```text
EXCLUDE_LEGACY_DEMOS_FROM_SAC_BELLMAN_REPLAY_V1
```

禁止：

- 把旧 74 集放入 residual SAC Bellman replay；
- 把旧 source action 直接当 residual actor BC target；
- clip/scale 旧 action 后仍配原 next state；
- 声称旧 transition 具有当前 safety filter、controller receipt 和 tracking evidence。

允许：

- 已完成的 StarVLA supervised training 和 baseline provenance；
- visual/task representation diagnostics；
- action/statistics/offline diagnostics。

路径：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
franka_74eps_runtime_contract_strategy_v1
```

因此 RL 路线没有失败；它要求用当前 execution adapter 重新采集 on-contract rollout。

---

## 5. 候选 5 Hz zero-residual 真机校准

今天创建并使用了：

```text
episode_id:
franka_zero_residual_5hz_calibration_20260807_01
```

preflight：

```text
publisher_counts.target_pose:      0
publisher_counts.gripper_command:  0
subscriber_counts.target_pose:     1
subscriber_counts.gripper_command: 1
stream_skew_ms:                    34.142
robot_commands_sent before start:  0
status:                            PASS
```

真实执行结果：

```text
commands sent:          3
residual max abs:       0.0
target error max:       0.00741729699 m
safety limit identity:  d1eaa3...88d9
```

操作者确认：

```text
I_REVIEWED_THE_CALIBRATION_AND_ROBOT_BEHAVIOR_WAS_NORMAL
```

审核结果：

```text
accepted_for_audited_dataset: true
accepted_for_training:        true
audit scope:                  baseline_validation
training scope:               AUDITED_DATASET_ONLY_NOT_ON_POLICY
```

audit：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
franka_zero_residual_5hz_calibration_20260807_01.audit.json
```

episode evidence SHA256：

```text
48be2b2dd72dbbbdcd2bbe3026caaa946be4d7c5bb9c2861e4b33e48b4fbb279
```

这个 episode 仍不满足 direct PPO/GRPO policy-gradient gate，缺少：

```text
rl_rollout.prev_logprobs
rl_rollout.forward_inputs
rl_rollout.forward_inputs.action_for_logprob
rl_rollout.model_weights_id
```

Residual SAC 是 off-policy，不依赖 old logprob，但仍需要足量、任务级、有明确
SUCCESS/FAILURE/INTERVENTION 的新 on-contract replay。

重要限制：这三步之间有人工 prompt，不是连续 200 ms 周期调度测试。

---

## 6. 推理优化路线 V1–V7

优化目标：在动作严格不变的前提下，使 StarVLA + 网络 roundtrip 稳定满足：

```text
p95 <= 180 ms
absolute max <= 220 ms
```

### 6.1 原始 baseline

原始 10098 WebSocket server 的 30-call roundtrip 约为：

```text
p50: 203.84 ms
p95: 311.28 ms
max: 417.53 ms
```

GPU0/GPU3 数值路径的 action reference SHA256：

```text
198332750d49eac3633fd1be1c8f43e2db2ec580432ebd09403f4b149b0390dc
```

### 6.2 V1/V2：直接 backbone bypass，拒绝

V1/V2 试图绕过 Qwen 内部 conditional-generation 路径，但改变了动作：

```text
max_abs action difference: 0.4600
```

因此两版均被拒绝，没有进入机器人或正式 server。

### 6.3 V3：只 bypass LM head，严格通过

V3 保留完整 conditional-generation forward，只把最终 vocabulary `lm_head.forward` 替换成空 logits，
避免计算机器人 policy 不使用的超大词表投影。

相同物理 GPU1 上：

```text
legacy full LM vs V3:
reference_max_abs: 0.0
```

GPU1 action reference SHA256：

```text
00701cb61b888dc01a358d61a880d7b1df1ae5a01f468a3e6ddbcc43e9449451
```

不同 H100 型号/物理卡之间曾出现：

```text
normalized action max_abs: 0.0137571
physical first-action difference: approximately 0.153 mm max
```

这是硬件数值路径差异，不能用跨 GPU reference 判断代码 parity。

### 6.4 V4：NUMA 与线程治理

GPU1 位于 NUMA node 0，对应 CPU：

```text
0-23,96-119
```

旧 server 状态：

```text
threads:            419
CPU affinity:       0-191
memory:             大量散布在 node 1/2/3
```

重启参数：

```text
numactl --cpunodebind=0 --preferred=0
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
```

重启后：

```text
threads:       15
CPU affinity:  0-23,96-119
RSS node 0:    approximately 3.26 GB
```

V4 compute：

```text
service policy p50: 131.53 ms
service policy p95: 140.05 ms
service policy max: 140.36 ms
```

动作 parity 仍为 0。

### 6.5 V5：可审计 WebSocket 分段

V5 证明：

```text
request unpack mean:      ~0.10 ms
response pack mean:       ~0.045 ms
server route p95:         ~141.17 ms
roundtrip p95:            ~417.63 ms
outside-server gap mean:  ~103.43 ms
```

因此 model wrapper、MsgPack unpack 和 response pack 不是主要长尾来源。

### 6.6 V6：持久 raw TCP

V6 实现：

- 4-byte big-endian length prefix；
- persistent connection；
- `TCP_NODELAY`；
- same MsgPack NumPy schema；
- same policy route；
- no ROS/no command capability。

初次使用 `CUDA_VISIBLE_DEVICES=1` 时，CUDA 默认 ordinal 顺序把 server 放到了物理 GPU3。证据：

```text
process PID 839890
physical GPU UUID: GPU-f16a68c8-8e24-9ac3-0713-b9bdbc3af6f6
```

因此动作落在 GPU0/GPU3 reference `198332...`，不是 transport 回归。

最终使用 GPU UUID 固定物理 GPU1：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=GPU-ccf45489-f640-c333-5e47-fefd51eed0ec
```

GPU1 strict parity：

```text
reference_max_abs: 0.0
action reference:  00701c...9451
```

但某一轮 V6 GPU1 网络长尾为：

```text
model p95:      152.4 ms
roundtrip p95:  849.6 ms
roundtrip max:  1060.1 ms
```

### 6.7 V7：逐阶段 TCP 计时

V7 新增逐 call 记录：

- client request pack；
- client send；
- server request header wait；
- server request payload receive；
- request unpack；
- server policy route；
- response pack probe；
- client response header wait；
- response payload receive；
- client response unpack；
- request/response frame bytes。

真实 frame 大小：

```text
request:  393,655–393,656 bytes
response: 2,686–2,687 bytes
```

V7 powersave-on 的主要指标：

```text
reference_max_abs:                    0.0
model p50/p95:                        130.95 / 144.42 ms
roundtrip p50/p95/max:                203.88 / 276.52 / 306.83 ms
client request send p50/p95:          0.479 / 0.653 ms
response payload receive p50/p95:     0.013 / 0.019 ms
server request payload p50/p95:       11.32 / 53.05 ms
response header wait p50/p95:         203.07 / 275.58 ms
```

结论：action response 本身只有约 2.7 KB，接收几乎为零；长尾主要发生在 Wi-Fi request/response 到达与
唤醒，而不是 StarVLA compute。

---

## 7. Wi-Fi 根因与 powersave A/B

### 7.1 实际路由

机器人电脑到 GPU server：

```text
192.168.1.113 dev wlp0s20f3 src 192.168.1.117
```

工作站已有的有线口：

```text
enp0s31f6: 172.16.0.1/24
```

该口服务 Franka 独立控制网络，无法到达 `192.168.1.113`。不能为了 GPU latency 随意改变 Franka
control Ethernet 的地址或路由。

Wi-Fi 状态：

```text
SSID:       HKU-CPS
BSSID:      F8:CE:21:CE:14:F2
channel:    36
frequency:  5180 MHz
rate:       540 Mbit/s
signal:     97%
```

即使信号强，实际 ICMP 仍出现 `114/174/205/309 ms` 尖峰，说明 AP contention、packet scheduling 或
retransmission 长尾仍存在。

### 7.2 Powersave 设置

系统默认配置：

```text
/etc/NetworkManager/conf.d/default-wifi-powersave-on.conf
wifi.powersave = 3
```

今天将 HKU-CPS connection profile 改为：

```text
802-11-wireless.powersave = disabled
```

`nmcli device reapply` 不支持热更新该属性，因此重新激活了 Wi-Fi connection。此设置当前是持久化配置。

如以后需要恢复默认：

```bash
nmcli connection modify HKU-CPS 802-11-wireless.powersave 0
nmcli connection up HKU-CPS ifname wlp0s20f3
```

### 7.3 Powersave-off 正式 V7 结果

动作仍严格一致：

```text
reference_max_abs: 0.0
```

关闭前后对比：

| 指标 | powersave on | powersave off |
|---|---:|---:|
| model p95 | 144.5 ms | 134.3 ms |
| roundtrip p50 | 203.9 ms | 138.3 ms |
| roundtrip p95 | 276.5 ms | 228.8 ms |
| roundtrip max | 306.8 ms | 485.5 ms |
| outside-server mean gap | 72.3 ms | 44.4 ms |

典型延迟明显改善，但 tail gate 仍失败：

```text
required: p95 <= 180 ms and max <= 220 ms
actual:   p95 228.8 ms and max 485.5 ms
```

### 7.4 纯 transport 对照

不调用模型、只使用 V7 ping route：

```text
96-byte request:
  p50 3.22 ms, p95 11.65 ms, max 20.32 ms

400,108-byte request:
  p50 10.56 ms, p95 30.56 ms, max 66.80 ms
```

这说明 TCP framing 本身工作正常；推理期间约 110–150 ms 静默使 Wi-Fi/调度长尾更容易暴露。

### 7.5 无损压缩评估

对 calibration 的双 RGB 使用 zlib：

```text
compressed size: 约原始的 71–73%
saved:           约 27–29%
level 3 encode:  约 6.0 ms / step
level 3 decode:  约 1.6 ms / step
```

无损压缩只能减少约 28% payload，却增加约 7.6 ms CPU 时间，不能消除 100–300 ms Wi-Fi spikes，
因此今天没有把压缩引入 production protocol。Lossy JPEG 会改变模型输入和 action，不可在没有新 parity/
behavior validation 的情况下直接使用。

---

## 8. 当前 server、部署与 artifacts

截至本文写入时，服务器运行的是 V7 shadow-only TCP candidate：

```text
directory: /home/hanyu/qwengroot_latency_candidate_v7_20260807
port:      10099
pid file:  /home/hanyu/qwengroot_v7_10099.pid
log:       /home/hanyu/qwengroot_v7_10099.log
GPU:       physical GPU1
GPU UUID:  GPU-ccf45489-f640-c333-5e47-fefd51eed0ec
NUMA:      node 0
transport: persistent raw TCP
```

安全边界：

```text
GPU_POLICY_ONLY
NO_OPTIMIZER
NO_ROS
NO_ROBOT_COMMAND
command capability: NONE
```

关键 artifacts：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
```

| Artifact | SHA256 |
|---|---|
| `qwengroot_low_jitter_transport_v5_20260807.json` | `31429a65599d730a5aed812ed914eeb4a08f208bed82e71cdfcced2ad7c2ce6d` |
| `qwengroot_low_jitter_tcp_v6_gpu1_strict_20260807.json` | `6eeb7c47248ced51d9c5225e539ac235a17beefe438dac7928b5d3f62b1c6e31` |
| `qwengroot_tcp_stage_timing_v7_gpu1_strict_20260807.json` | `4691c34ff7ec9a914869d019cc9ce5cf3065e04e43f61731cfcf83c9083d6e05` |
| `qwengroot_tcp_v7_wifi_powersave_off_gpu1_strict_20260807.json` | `0dcf991c53c3bedc99e11bf37dc7909bb1812fcb186cbbb446fcc4aba38fecaa` |

部署 bundles：

```text
qwengroot_low_jitter_tcp_candidate_v6_20260807.tar.gz
sha256: 64d4661b7c59f6df228ca8889538d296f46072d9a75b8835a9ed84f2b4311228

qwengroot_tcp_stage_timing_candidate_v7_20260807.tar.gz
sha256: df13a1865dadf2e492e493789643cccd497ff7c7c7fe37df5bd49cac636f2be6
```

服务器 root filesystem 曾为约 98% 使用率，`/data` 约 96%。这不是今天延迟尖峰的主因，但后续训练前必须
监控空间，避免 checkpoint/replay 写入失败。

---

## 9. 今日代码变更与测试状态

核心新增：

```text
examples/realRobots/Franka/rl/qwen_backbone_inference.py
examples/realRobots/Franka/rl/benchmark_qwengroot_shadow_latency.py
examples/realRobots/Franka/rl/qwengroot_low_jitter_websocket_server.py
examples/realRobots/Franka/rl/qwengroot_low_jitter_tcp_server.py
tests/test_qwen_backbone_inference.py
tests/test_qwengroot_low_jitter_server.py
tests/test_qwengroot_low_jitter_tcp.py
```

核心修改：

```text
examples/realRobots/Franka/rl/qwengroot_rl_policy.py
examples/realRobots/Franka/rl/qwengroot_shadow_policy_server.py
examples/realRobots/Franka/rl/qwengroot_shadow_protocol.py
examples/realRobots/Franka/rl/franka_ros2_shadow_adapter.py
tests/test_qwengroot_rl_policy.py
tests/test_qwengroot_shadow_adapter.py
```

V7 instrumentation 后验证：

```text
52 tests run
status: OK
skipped: 13 optional/environment-dependent tests
py_compile: PASS
git diff --check: PASS
```

注意：上述变更仍在 working tree 中，尚未 commit。不要 reset、checkout 或覆盖这个 worktree。应在有线网络
最终 latency validation 后统一审查并 commit。

---

## 10. 当前 blocker 的准确含义

当前 blocker 不是：

- checkpoint 损坏；
- StarVLA 不能输出 action；
- QwenGR00TRLPolicy adapter 错误；
- controller 无法执行；
- safety adapter 失败；
- residual SAC 公式不能更新；
- 旧 74 集不是成功示范。

当前 blocker 是：

```text
机器人电脑 -> GPU server 只能走具有长尾的共享 Wi-Fi，
无法稳定保证每个 synchronous replan 在 200 ms 内完成。
```

如果直接忽略这个问题进入 RL rollout，会造成：

- stale action；
- observation/action/next-observation 时间错位；
- safety abort 增多；
- accepted replay 偏向网络恰好正常的 transition；
- Bellman transition 不再代表固定控制周期 MDP；
- 可能把网络问题误当成 policy failure 或 reward signal。

因此当前停止点是正确的工程 gate，不是 RL 算法失败。

---

## 11. 下一步推荐路线

### 11.1 首选：新增 GPU LAN 有线链路

保持：

```text
enp0s31f6 / 172.16.0.1 -> Franka controller network
```

新增：

```text
USB 3.0 Gigabit/2.5GbE adapter -> GPU server 192.168.1.x LAN
```

不要把现有 Franka Ethernet 改到 `192.168.1.x`，也不要让 robot control traffic 和 GPU image traffic
共享同一错误路由。

有线连接后首先检查：

```bash
ip route get 192.168.1.113
ping -c 30 192.168.1.113
```

必须看到 route 走新增有线 interface，再重复完全相同的 V7 strict benchmark。验收仍是：

```text
reference_max_abs == 0.0
roundtrip p95 <= 180 ms
roundtrip max <= 220 ms
```

通过后再做 100-call confirmation，不能只凭一次 30-call best case 放行。

### 11.2 网络通过后的执行顺序

1. 固化 GPU1 UUID + NUMA0 + 4-thread server launcher；
2. 运行 100-call strict parity/latency confirmation；
3. commit V3–V7 production code与报告 identity；
4. 实现/验证无人工等待的连续 5 Hz shadow scheduler；
5. 在 live 双相机和 state stream 下验证 freshness、skew、deadline、stale rejection；
6. 先采集 zero-residual、完整任务级 SUCCESS/FAILURE/INTERVENTION episodes；
7. 保存每步 before observation、base action、filtered/executed action、controller receipt、exact next
   observation、outcome/reward；
8. 对 exact next observation 预计算 frozen next-base action；
9. 建立新的 on-contract replay 和 replay checkpoint；
10. 用足量新 replay 做多步 residual SAC offline training/restore smoke；
11. residual deployment limits 保持 0 做 stochastic shadow；
12. 独立人审和安全审批后，才考虑极小非零 residual 的单步 human-gated rollout；
13. 最终用固定任务、固定初始条件和 blinded A/B 评估 baseline vs baseline+RL success rate。

### 11.3 如果短期无法增加有线网络

不推荐简单放宽 stale threshold 或假装 3/5 Hz 稳定。可选架构是：

```text
GPU 异步生成 8-step base action chunk
-> 机器人本地以固定频率消费缓存
-> 提前请求下一 chunk
-> 小型 residual actor 在本地运行
```

但这会把当前 `execution_horizon=1` 的 MDP 改成 chunk/asynchronous contract，必须重新设计：

- chunk-level transition；
- base action age；
- residual credit assignment；
- safety filter 与 stale semantics；
- exact next-base definition；
- replay schema 和 reward attribution。

在完成这些设计与校准前，不能把异步 chunk 当成现有 residual SAC 的无缝替换。

---

## 12. 对“RL 是否能提高成功率”的当前回答

理论上，residual SAC 可以通过真实失败、干预和成功 transition 学习修正 frozen StarVLA 在真实 Franka 环境
中的系统性误差，例如末端位置偏差、抓取接近误差和放置偏差。

但截至 2026-08-07：

```text
new on-contract task episodes: 0
nonzero residual robot steps:   0
full RL training run:           0
baseline-vs-RL evaluation:      0
measured success improvement:   unknown
```

今天完成的是让未来的 RL 数据在因果、时序、安全和 provenance 上可信。它显著降低了“训练跑起来但学到错误
信号”的风险，但本身不等于成功率已经提高。

---

## 13. 明日恢复工作时的最小检查

1. 确认 V7 server 是否仍在：

   ```bash
   ssh hanyu@192.168.1.113
   v7_pid=$(cat /home/hanyu/qwengroot_v7_10099.pid)
   kill -0 "$v7_pid" && echo V7_RUNNING
   ```

2. 确认物理 GPU：

   ```bash
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
     --format=csv,noheader | grep ",$v7_pid," 
   ```

   必须是：

   ```text
   GPU-ccf45489-f640-c333-5e47-fefd51eed0ec
   ```

3. 确认工作站到 GPU 的 route。如果仍是 `wlp0s20f3`，不要进入连续 5 Hz 真机 rollout。
4. 保持 residual deployment limits 为 0。
5. 不要把旧 74 集重新放入 SAC Bellman replay。
6. 不要清理或重置 `starVLA_rl_libero` 未提交工作树。

---

## 14. 一句话交接

```text
Residual SAC 的代码、审计、离线 optimizer 和短程零 residual 真机链路已建立；
StarVLA 模型侧已优化到约 115–135 ms 且同 GPU action parity 为 0；
当前唯一直接阻止稳定同步 5 Hz rollout 的主 blocker 是机器人电脑到 GPU server 的 Wi-Fi 长尾，
下一步应增加独立有线 GPU LAN，再做 100-call latency gate 和连续 5 Hz shadow scheduler。
```
