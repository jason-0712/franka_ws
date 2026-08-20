# 2026-08-06 StarVLA × RLinf × Franka RL 安全执行工作总结

日期：2026-08-06  
时区：Asia/Hong_Kong  
本地工作区：`/home/dase-hw101/franka_ws`  
RL 开发 worktree：`/home/dase-hw101/franka_ws/third_party/starVLA_rl_libero`  
机器人容器：`franka`，用户工作区 `/home/ros/qwengroot_rl_execution`  
GPU 服务器：`192.168.1.113`，RLinf `/home/hanyu/RLinf-v0.3`  
任务：`pick up the cube and place it on the box`

---

## 0. 今日最终结论

今天完成了从 production QwenGR00T RL policy adapter，到 shadow-only GPU policy service、
ROS 2 安全 execution adapter、真实 Franka 三步执行、完整 transition 记录、人审门和
audited dataset loader 的工程闭环。

最重要的结论如下：

1. Libero-init 74eps、vision-frozen、20k checkpoint 已通过正式 QwenGR00TRLPolicy adapter smoke。
2. GPU policy service 仍是 proposal-only：只能返回 StarVLA base action 和 zero residual，不能连接 ROS 或执行动作。
3. 真实机器人动作只能由机器人侧 execution adapter 发出，并受本地 permit、TTY、逐步 `EXECUTE n`、freshness、latency、workspace、delta、tracking 和 feedback 门禁控制。
4. execution adapter 已记录完整链路：

   ```text
   before observation
   -> StarVLA base proposed action
   -> residual proposal/applied residual
   -> safety-filtered action
   -> controller command
   -> measured next state
   -> outcome/reward/terminated/truncated
   ```

5. command write-ahead journal 已实现并验证：

   ```text
   PREPARED_NOT_YET_CONFIRMED_SENT
   -> SENT_AWAITING_TRANSITION_RECORD
   -> TRANSITION_RECORDED
   ```

6. teleop publisher collision、controller 中断、腕部相机 USB/stream 问题、operator 停顿导致 stale action、跨流 skew 等真实故障都被门禁正确拒绝，没有静默绕过。
7. 成功完成两次单步和两次三步 zero-residual 真机安全验证；最终 `_04` 三步 episode 完整通过。
8. 最终 `_04` 的 post-command target error 为：

   ```text
   step 0: 3.899 mm
   step 1: 6.734 mm
   step 2: 7.556 mm
   ```

   全部低于新的 `10 mm` tracking hard gate。
9. `_04` 的 manifest、transition、command journals、双相机图像和 action stages 完整性检查为 PASS。
10. 已实现独立 human-audit record 和 audited dataset loader。原 episode 永远保持 `accepted_for_training=false`，人工决定写到独立、hash-bound、create-only 的 audit JSON。
11. 当前 `_04` 可以作为 baseline evaluation、safety validation 和 offline analysis 数据，但不能直接作为 PPO/on-policy batch。
12. 当前缺少 PPO 必需的 rollout-time 数据：

    ```text
    rl_rollout.prev_logprobs
    rl_rollout.forward_inputs
    rl_rollout.forward_inputs.action_for_logprob
    rl_rollout.model_weights_id
    ```

13. 截至本文写入时，`_04` 的人工 accept audit **尚未创建**；只读完整性检查和 contact sheet 导出已完成。
14. 今天没有启动 RL optimizer step，没有使用真实机器人做在线梯度更新，也没有得到“RL 已提高成功率”的结论。

当前准确状态应写为：

```text
production StarVLA RL policy adapter:       PASS
zero-residual policy/server smoke:          PASS
ROS 2 safety execution adapter:             PASS
single-step real execution:                 PASS
three-step real execution:                  PASS
immutable episode evidence audit:           PASS
human accept audit:                         PENDING OPERATOR
audited dataset loader:                     PASS
PPO/on-policy rollout data contract:        BLOCKED / NOT YET RECORDED
RL optimizer update:                        NOT STARTED
RL success-rate improvement:                NOT MEASURED
```

---

## 1. 固定的主 baseline

当前 RL 主 baseline 是已完成的 Libero-init Franka checkpoint：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
final_model/pytorch_model.pt
```

文件信息：

```text
size:   9,976,833,554 bytes
sha256: 7f018ea7d83f0e87f0f900fec9d8c3cbc7125af1a75ca843cb06b48d74f243d7
modify: 2026-07-30 08:08:28 UTC
```

这个 checkpoint 已完成，因此今天后续的 adapter、server、execution 和 audit 都绑定它，
不再使用未完成状态的模型，也不在实验中途更换 initialization。

模型动作语义保持：

```text
StarVLA physical action chunk: [B, 8, 7]
single executed action:        [dx,dy,dz,droll,dpitch,dyaw,gripper]
execution horizon:             1
gripper residual:              disabled
base policy uses state:        false
```

真实部署每次只执行 action chunk 的第一个 action，然后重新观测、重新推理、重新过全部安全门。

---

## 2. 正式 QwenGR00TRLPolicy adapter

今天完成并验证了 production `QwenGR00TRLPolicy` adapter，不再依赖 smoke 脚本中的临时 monkey patch。

adapter 的核心职责：

- 加载 frozen StarVLA QwenGR00T checkpoint；
- 使用训练时完全一致的 normalization/unnormalization；
- 输出 formal action stages；
- 只允许 6D Cartesian residual；
- gripper 永远来自 base policy，不接受 residual 修改；
- residual 在物理空间逐维 clip；
- zero residual 时 proposed action 与 base first action 完全相同；
- 保留未来 RL replay 所需的 policy metadata；
- 对错误 shape、NaN/Inf、stateful-base mismatch 和七维 residual fail closed。

正式 policy smoke 结果：

```text
requested_residual_max_abs: 0.0
residual_add_max_abs:       0.0
residual_clip_max_abs:      0.0
zero_proposal_max_abs:      0.0
status:                     PASS
zero-residual inference:    ~0.628 s
```

对应 report：

```text
/data/hanyu/starVLA_runs/
quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/
rl_smoke/qwengroot_rl_policy_smoke_20260806_064338.json
```

GPU policy service 随后监听：

```text
0.0.0.0:10098
```

网络检查曾确认：

```text
nc -zv 192.168.1.113 10098
Connection succeeded
```

重要边界：10098 service 只提供 shadow proposal，没有 ROS、controller 或 execute API。

---

## 3. ROS/controller/camera 的真实问题与处理

### 3.1 Teleop publisher 冲突

第一次 execution preflight 发现 command topics 上已有 publisher。后来确认这些 publisher 来自
正在进行的 teleop，而不是 execution adapter。

结论：

- teleop 与 execution adapter 不能同时占用 command topics；
- teleop 停止后 command publisher count 应为 0；
- publisher count 为 0 只是必要条件，不代表 controller 正常；
- 还必须有 controller subscribers、current pose、gripper feedback 和两路相机数据。

execution adapter 从未自动杀死或替换 teleop/controller 节点。

### 3.2 Controller 中断与恢复

期间 controller 曾死亡，随后由操作者手动重启。恢复后验证：

- wired interface 恢复；
- robot network 可达；
- controller subscribers 存在；
- pose/gripper feedback 实际推进；
- command publisher 初始为 0。

不能仅根据 GPU 空闲、TCP service 可达或 publisher count 判断 controller 可执行。

### 3.3 腕部相机 USB/stream 问题

一次三步尝试出现 `STREAM_SKEW`，期间用户拔过 USB，腕部相机 stream 曾停止/重启。

处理方式不是放宽 `250 ms` skew 门限，而是：

- 恢复相机容器/USB 数据；
- 每步等待 coherent snapshot；
- 最多等待 2 秒；
- 只有 observation age 与 stream skew 同时满足原门限才继续；
- 等待超时仍然 fail closed。

这避免了两路相机和 robot state callback 更新相位不同导致的偶发假拒绝，同时没有削弱安全阈值。

---

## 4. 安全 execution adapter

机器人侧部署目录：

```text
/home/ros/qwengroot_rl_execution
```

核心文件：

```text
franka_execution_safety.py
franka_ros2_execution_adapter.py
create_franka_execution_permit.py
run_franka_ros2_human_gated_execution.sh
```

安全链路固定为：

```text
coherent observation
-> shadow-only StarVLA proposal
-> robot-side freshness/latency checks
-> hard action checks
-> delta/workspace/gripper filtering
-> local typed EXECUTE n
-> fresh observation + fresh inference + safety recheck
-> controller command heartbeat
-> measured next feedback
-> tracking gate
-> event reward + atomic transition record
```

### 4.1 当前安全参数

```text
workspace X:                       [0.28, 0.57] m
workspace Y:                       [-0.23, 0.10] m
workspace Z:                       [0.03, 0.70] m
max translation delta norm:        0.009 m
max rotation delta norm:           0.003 rad
raw translation hard reject:       0.15 m
raw rotation hard reject:          0.25 rad
max workspace clip:                0.003 m
max target tracking error:         0.010 m
max observation age:               500 ms
max stream skew:                   250 ms
max policy roundtrip:              1500 ms
max action age:                    1750 ms
gripper consensus:                 75%
gripper switch confirmations:      3 executed proposals
```

canonical safety-limit SHA256：

```text
d1eaa3acc9b62776287f1936148cdae803cd06b7715b282075234fc817ea88d9
```

`10 mm` tracking gate 来自首次完整三步 baseline 的实测校准。旧的 50 mm 上限明显过宽，
因此没有保留。超过 10 mm 时 adapter 会立即停止 command heartbeat 并记录 `safety_abort`。

### 4.2 授权机制

真实动作要求同时满足：

1. `--arm-human-gated`；
2. 本地一次性 permit；
3. permit 有效期不超过 30 分钟；
4. permit 与 episode/checkpoint/max steps/safety hash 精确绑定；
5. 本地交互式 TTY；
6. command publisher 初始为 0；
7. controller subscriber 和 feedback 存在；
8. 每步由操作者输入 `EXECUTE <step_id>`；
9. operator 停顿后重新采集、重新推理、重新做全部 safety check。

远端 GPU service 不能创建 permit，也不能通过网络绕过本地逐步授权。

### 4.3 数据记录

每个 transition 记录：

- before/next 主相机和腕部 RGB；
- before/next Franka state；
- stream timestamps、age、skew；
- raw/normalized/physical/base/residual/proposed action；
- filtered action 和 executed action；
- controller absolute target 和 gripper command；
- command publish counts；
- pose/gripper feedback advancement；
- measured target tracking error；
- operator decision 和 outcome；
- event reward 和 reward components；
- terminated/truncated/truncation reason；
- checkpoint、permit、safety-limit identity；
- 图像、transition 和 journal SHA256。

原始 transition 和 manifest 永远写：

```text
accepted_for_training=false
```

不能由 execution adapter 自动进入训练。

---

## 5. 真机执行过程与故障分类

### 5.1 单步验证 `_02`

```text
episode: franka_zero_residual_step1_20260806_02
command sent: true
filter: UNCHANGED
target error: 3.394 mm
result: PASS_RECORDED_PENDING_HUMAN_AUDIT
```

该次暴露了 `max_steps` 的语义问题：达到 step limit 应记录 `truncated=true`，不能看起来像
仍可继续的非终止 transition。随后修复。

### 5.2 单步验证 `_03`

```text
episode: franka_zero_residual_step1_20260806_03
command sent: true
filter: UNCHANGED
target error: 3.288 mm
terminated: false
truncated: true
truncation_reason: max_steps
```

该次验证 step-limit 语义修复正确。

### 5.3 第一次三步尝试 `_01`

```text
step 0: command sent, target error 3.546 mm
step 1: rejected, STALE_ACTION
```

原因：操作者在 step 0 后输入 feedback/下一步授权期间停顿，旧 observation/action 继续被使用。

修复：每次 operator pause 之后必须重新采集 coherent observation、重新请求 policy、重新过 safety。

### 5.4 第二次三步尝试 `_02`

```text
step 0: rejected, STREAM_SKEW
commands sent: 0
```

原因与腕部相机 stream/USB 恢复阶段有关。门禁正确保持 robot 不动。

修复：恢复相机，并实现最多 2 秒的 coherent snapshot waiter；没有放宽原始 age/skew 门限。

### 5.5 第一次完整三步 `_03`

```text
episode: /home/ros/qwengroot_rl_execution_logs/
         franka_zero_residual_3step_20260806_03

commands: 3/3
residual: exactly zero
target error:
  step 0: 3.80 mm
  step 1: 6.57 mm
  step 2: 8.18 mm
```

该次证明：

- 三步 cumulative target semantics 与 training/deployment 一致；
- action stages exact；
- feedback 每步推进；
- 旧 50 mm tracking gate 不够严格。

由此将 tracking hard gate 收紧为 10 mm。

### 5.6 最终三步 `_04`

```text
episode: /home/ros/qwengroot_rl_execution_logs/
         franka_zero_residual_3step_20260806_04
```

最终结果：

| step | max obs age | stream skew | policy | action age | target error |
|---:|---:|---:|---:|---:|---:|
| 0 | 44.60 ms | 50.34 ms | 265.93 ms | 273.11 ms | 3.899 mm |
| 1 | 58.72 ms | 56.44 ms | 290.51 ms | 302.77 ms | 6.734 mm |
| 2 | 69.97 ms | 73.40 ms | 341.80 ms | 343.29 ms | 7.556 mm |

全部满足门限：

- `manifest_status=COMMITTED`；
- 3 transitions；
- 3 commands；
- 3 journals 均为 `TRANSITION_RECORDED`；
- `execution_data_complete=true`；
- incomplete journals 为空；
- transition、journal、所有图像 hash 一致；
- residual requested/applied 全为 0；
- base/proposed/filtered/executed action exact；
- filter flag 均为 `UNCHANGED`；
- pose 和 gripper feedback 每步推进；
- 每步各发布 21 个 pose 和 21 个 gripper command messages；
- gripper 保持 open command `1.0`；
- 每步 reward 为 `-0.01`；
- 最后一步 `terminated=false`；
- 最后一步 `truncated=true`；
- `truncation_reason=max_steps`。

注意：三个 outcome 都是 `CONTINUE`。这不是任务成功 episode，也不能用于宣称成功率提升；
它是三步 zero-residual execution/safety baseline。

---

## 6. Human audit 与 immutable evidence

今天新增：

```text
examples/realRobots/Franka/rl/franka_execution_audit.py
examples/realRobots/Franka/rl/audit_franka_execution_episode.py
examples/realRobots/Franka/rl/export_franka_execution_review.py
examples/realRobots/Franka/rl/franka_execution_dataset.py
examples/realRobots/Franka/rl/FRANKA_AUDITED_ROLLOUT_DATA.md
tests/test_franka_execution_audit.py
```

### 6.1 为什么 audit 必须独立

人审不会修改原 episode。原始：

```text
manifest.json
transitions.jsonl
command_journal/*.json
observations/*.npy
COMMITTED
```

保持不可变。独立 audit JSON 绑定：

- COMMITTED marker hash；
- manifest file hash；
- transitions file hash；
- 每个 command journal file hash；
- 每张 NPY file hash；
- 每张图像 array-content hash；
- checkpoint hash；
- safety-limit hash；
- reviewer、review time、notes 和三项人工确认。

audit 文件使用 mode `0600`、create-only、不可覆盖。任意原始证据或 audit 内容变化，loader
都会重新计算并拒绝。

### 6.2 `_04` 只读 audit 结果

```text
status:                    PASS
episode_id:                franka_zero_residual_3step_20260806_04
steps/commands:            3/3
residual_max_abs:          0.0
target_error_max_m:        0.007555802818387747
episode_evidence_sha256:   7df82eb7bbbf9c0788a6c0fde332dcae8728bfb43f602f55c0e27e65378471d6
human_audit_written:       false
```

contact sheet：

```text
/home/dase-hw101/franka_ws/artifacts/rlinf_audits/
franka_zero_residual_3step_20260806_04_contact_sheet.png
```

contact sheet SHA256：

```text
d50da031c34fa2abb2714489648500669a5390237b252a12e2818e6d0ef8d57a
```

画面检查结果：三步的 before/next 主相机与腕部相机均可读，没有黑帧或格式错误；最终行为和
`CONTINUE` 标签仍必须由现场操作者确认，不能由自动程序代签。

截至本文写入时：

```text
/home/ros/qwengroot_rl_execution_audits/
franka_zero_residual_3step_20260806_04.audit.json
```

仍不存在，这是正确状态。

---

## 7. Audited dataset loader 与 RLinf 边界

`franka_execution_dataset.py` 只接受：

1. 完整 evidence validation PASS；
2. 独立 human audit decision 为 accept；
3. audit evidence hash 与重新计算结果完全一致。

输出 NumPy trajectory 使用 RLinf 的 `[T,B,...]` 轴语义，当前 `B=1`：

| field | shape | meaning |
|---|---|---|
| `actions` | `[T,1,7]` | 实际 executed physical action |
| `rewards` | `[T,1]` | event reward |
| `terminations` | `[T,1]` | terminal flag |
| `truncations` | `[T,1]` | truncation flag |
| `dones` | `[T,1]` | termination OR truncation |
| `intervene_flags` | `[T,1,7]` | clean episode 中为 false |
| `curr_obs/main_images` | `[T,1,H,W,3]` | before 主相机 |
| `curr_obs/wrist_images` | `[T,1,H,W,3]` | before 腕部相机 |
| `curr_obs/states` | `[T,1,7]` | before robot state |
| `next_obs/*` | `[T,1,...]` | measured next observation |

### 7.1 为什么 current episode 不能直接 PPO

PPO ratio 需要：

```text
ratio = exp(new_logprob - old_logprob)
```

其中 `old_logprob` 必须是实际采样该 action 时的策略概率。当前 execution log 只有 action stages，
没有 rollout-time old logprob 和完整可重放 model inputs。

不能用以下方式伪造：

- 把 old logprob 填成 0；
- 事后重新推理一个 logprob；
- 用 physical action 距离近似概率；
- 只保留 seed 而不保留 sampling/replay tensors。

否则 PPO ratio 不再对应实际 behavior policy，更新方向可能错误。loader 因此显式报告：

```text
policy_gradient_eligible=false
policy_gradient_gate=BLOCKED_MISSING_ROLLOUT_REPLAY_DATA
```

这不是 execution pipeline 失败，而是下一阶段必须补齐的 RL 数据合同。

---

## 8. Tests 与验证

今天最终回归：

```text
QwenGR00T policy/shadow/replay tests: 26 PASS
Franka safety + audit tests:          10 PASS
total related tests:                  36 PASS
```

新增的 4 个 audit/dataset 测试覆盖：

- accept audit 后可加载；
- 原 manifest/transition 仍为 false；
- audit 文件 mode 为 0600；
- 当前数据进入 PPO 时 fail closed；
- 图像被修改后 audit 立即失效；
- reject audit 不能加载；
- evidence hash 不匹配不能加载；
- accept 缺少人工确认不能创建；
- 已存在的 audit 路径不能覆盖。

真实 `_04` 也在本机副本和 `franka` 容器中分别完成只读验证，结果一致。

---

## 9. Git 状态与今日提交

RL worktree：

```text
/home/dase-hw101/franka_ws/third_party/starVLA_rl_libero
branch: research/rlinf-libero-residual
HEAD:   7f83f19c7ec9008d958bab8b6d71a3a932695111
status: clean
```

今日相关提交按时间顺序：

```text
f1bc098 docs: record Libero deterministic smoke
cab4dfb feat: add production QwenGR00T RL policy adapter
5ac4616 test: add production QwenGR00T adapter smoke
806dd04 docs: record production adapter smoke pass
ada1f8f feat: add Franka observe-only and shadow adapters
2ca9509 docs: record ROS shadow container preflight
bafcdd7 docs: record ROS state-stack preflight blocker
769ff52 docs: record live ROS observe-only smoke
4084a47 feat: verify committed shadow replay parity
ef74988 feat: add human-gated Franka execution adapter
2189ca8 fix: mark step-limit execution transitions truncated
3348067 fix: refresh observations after operator pauses
9a1daa5 safety: tighten Franka target tracking gate
91ded23 fix: wait for coherent Franka observations
7f83f19 feat: gate audited Franka rollout data
```

---

## 10. 容器部署 identity

当前 execution 关键文件：

```text
franka_execution_safety.py
c51b29a6c407bd3a38a7facbca9f6694f56b9cdef943f66ddb4169aaf93dad0d

franka_ros2_execution_adapter.py
6b7e87f23bb69c62c60832a25c4fbabc3969fc4683c4c67e9d16c131bf7df506
```

新增 audit/dataset 文件的本地与容器 SHA256 已核对一致：

```text
franka_execution_audit.py
0a9570890e213bfca05faaad9d6344b00a5eec3904b28bd255abb9e076ca3c0e

audit_franka_execution_episode.py
3f73f194c6a44a6de44c92d9448cf942e08bcab5936759cc2f339bf9867b9e37

franka_execution_dataset.py
bed1362a06a9686045fc1b1e7ba85242e24922a7b0c558753354bd213eb79fae

export_franka_execution_review.py
b7642ab24fa394ae6603a97a865cb55004c8221dfa209d6bc1835ed78fac6cfd
```

---

## 11. 下一步

### Step 1：由现场操作者完成人工 accept

先查看 contact sheet，并确认：

- 三步 before/next 图像正确；
- 三步真实机器人行为与现场观察一致；
- 三个 `CONTINUE` outcome 标签正确；
- 最后一步确实因 `max_steps` 截断；
- 该 episode 只作为 zero-residual baseline/safety validation。

然后运行：

```bash
docker exec -it --user ros franka bash -lc "
cd /home/ros/qwengroot_rl_execution
python3 audit_franka_execution_episode.py \
  --episode /home/ros/qwengroot_rl_execution_logs/franka_zero_residual_3step_20260806_04 \
  --decision accept \
  --audit-root /home/ros/qwengroot_rl_execution_audits \
  --reviewer-id hanyu \
  --scope baseline_validation \
  --require-zero-residual \
  --notes '3-step zero-residual safety baseline; outcomes are CONTINUE, final truncation is max_steps'
"
```

准确输入：

```text
ACCEPT franka_zero_residual_3step_20260806_04 FOR AUDITED DATASET
```

### Step 2：audited loader smoke

accept 后重新验证：

- audit file permissions；
- audit/evidence hash binding；
- trajectory shapes；
- rewards/dones；
- PPO gate 仍然正确拒绝当前 zero-residual baseline。

### Step 3：扩展真正的 RL rollout protocol

GPU policy response/sidecar 必须正式记录：

```text
sampled residual action
requested residual
clipped/applied residual
sampling seed and sampling parameters
rollout-time prev_logprobs
complete forward_inputs
action_for_logprob
model_weights_id / policy version
checkpoint identity
```

并验证：

```text
recorded action replay parity
recorded old-logprob replay parity
same action-space and same clipping semantics
policy-version binding
```

### Step 4：只做 stochastic shadow rollout

在 residual limits 非零之前，先让 residual actor 产生 stochastic proposal，但：

- 不连接 ROS；
- 不发 controller command；
- 不启动真机动作；
- 验证 logprob/replay/trajectory 完整性；
- 验证 RLinf actor 能重算相同 logprob。

### Step 5：离线最小 optimizer smoke

只有 Step 3/4 全部通过后，才在 recorded synthetic/shadow batch 上做：

- 1 个 optimizer step；
- loss finite；
- ratio/clip/KL/entropy 可解释；
- 只更新 residual policy/value head；
- frozen StarVLA base 权重 hash 不变。

### Step 6：未来的 bounded residual 真机试验

只有离线训练、shadow rollout 和 safety review 都通过后，才考虑：

- 极小 residual limits；
- 单步 human-gated；
- zero residual control 与 RL residual treatment 配对；
- 明确 success/failure/intervention reward；
- 先少量 pilot，再统计成功率。

不应直接从今天的三步 baseline 跳到自动多步 online RL。

---

## 12. 对成功率问题的准确回答

RL 理论上可能改善当前约 65% 的历史 task success rate，尤其可能针对：

- 临近抓取时的细小 XYZ correction；
- gripper close timing；
- 放置区域附近的末端修正；
- demonstration 分布中较少见的恢复状态。

但今天的结果只证明：

```text
policy integration works
safety execution works
transition recording works
human audit/data gate works
```

尚未证明：

```text
RL optimization works on this task
reward is sufficient
residual policy improves action quality
real-robot success rate improves
```

未来必须用固定 Libero baseline、相同初始状态分布、相同 safety filter 和足够 trial 数比较：

```text
zero-residual baseline vs trained bounded-residual policy
```

才能回答成功率是否真正提高。

---

## 13. 明日恢复工作时的最短检查清单

```text
[ ] 阅读本文件和 FRANKA_AUDITED_ROLLOUT_DATA.md
[ ] 不重复做真机三步 baseline
[ ] 查看 _04 contact sheet
[ ] 由现场操作者创建 accept audit
[ ] 运行 audited loader smoke
[ ] 固定 audit/evidence SHA256
[ ] 设计 rollout logprob/replay sidecar schema
[ ] 先做 GPU-only stochastic shadow + parity
[ ] parity 通过前不启动 optimizer
[ ] offline optimizer smoke 通过前不增加 residual limit
[ ] 所有阶段通过前不进行自动多步真机 RL
```

今天已经完成的核心成果不是“RL 已经训练好”，而是建立了一个不会把不完整、未审核或
概率语义错误的数据送进 RL 的安全起点。
