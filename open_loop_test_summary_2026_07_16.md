# Open-Loop Test Summary - 2026-07-16

## Purpose

We added an open-loop test to evaluate StarVLA without moving the real Franka.
The goal was to separate policy prediction quality from real-robot execution issues.

The test uses recorded LeRobot observations:

- read image frames from recorded episodes;
- send each image + language instruction to the StarVLA websocket policy server;
- get predicted 7D delta EEF action:
  `[dx, dy, dz, droll, dpitch, dyaw, gripper]`;
- compare predicted action with dataset GT action using L2 metrics.

This is useful because the real-robot failure looked like weak lateral/y correction and unstable gripper timing.

## Script

The open-loop script is:

```text
scripts/starvla_open_loop_l2_eval.py
```

It reports:

- `first_l2_mean`: L2 of the first predicted action vs GT action.
- `chunk_l2_mean`: average L2 over the predicted action chunk.
- `first_xyz_l2_mean`: position-only error for `[dx, dy, dz]`.
- `first_gripper_abs_mean`: gripper open/close mismatch rate-like error.
- `y_mae`: absolute error on y action.
- `y_sign_acc`: whether predicted y direction matches GT y direction.
- `pred_y_abs_mean` and `gt_y_abs_mean`: predicted vs GT y action magnitude.

We mainly care about:

```text
y_sign_acc
pred_y_abs_mean / gt_y_abs_mean
first_gripper_abs_mean
first_xyz_l2_mean
```

## Models Tested

### 1. 100eps lateral11 model

Checkpoint:

```text
results/Checkpoints/quest3_franka_delta_eef_from_89eps2k_100eps_lateral11_1500/final_model/pytorch_model.pt
```

Training data:

- previous 89 episodes;
- plus 11 lateral correction episodes:

```text
0150 0151 0152 0153 0154 0157 0158 0159 0160 0161 0162
```

We tested both `action-offset 0` and `action-offset 1`.

`action-offset 1` was better for gripper timing on some episodes, but did not fundamentally solve the issue.

Important overall result for offset 1:

```text
queries: 171
first_l2_mean:          0.306810
first_l2_median:        0.004898
chunk_l2_mean:          0.278823
first_xyz_l2_mean:      0.004248
first_gripper_abs_mean: 0.304094
y_mae:                  0.002477
y_sign_acc:             0.613445
pred_y_abs_mean:        0.000904
gt_y_abs_mean:          0.002477
```

Interpretation:

- Position error was not terrible: `first_xyz_l2_mean` was about 4.2 mm.
- Full L2 was high mainly because gripper mismatch contributes about `1.0` whenever open/close is wrong.
- Gripper timing remained unstable: about 30% sampled points had gripper mismatch.
- Y direction was only slightly better than random: `y_sign_acc = 0.613`.
- Y magnitude was too small:

```text
pred_y_abs_mean / gt_y_abs_mean = 0.000904 / 0.002477 ≈ 0.36
```

So the model predicted only about one third of the needed lateral movement.

### 2. Lateral11 x3 weighted model

Checkpoint:

```text
results/Checkpoints/quest3_franka_delta_eef_from_100eps1500_lateral11x3_1000/final_model/pytorch_model.pt
```

Training data:

- old 89 episodes once;
- lateral 11 episodes repeated 3 times;
- total sampling items: `89 + 11 * 3 = 122`.

The purpose was to increase the weight of lateral correction data.

Overall result:

```text
queries: 171
first_l2_mean:          0.289108
first_l2_median:        0.004293
chunk_l2_mean:          0.272266
first_xyz_l2_mean:      0.004242
first_gripper_abs_mean: 0.286550
y_mae:                  0.002522
y_sign_acc:             0.588235
pred_y_abs_mean:        0.000806
gt_y_abs_mean:          0.002477
```

Comparison with previous 100eps lateral11 model:

```text
Metric                  100eps lateral11    lateral11 x3
first_xyz_l2_mean       0.004248            0.004242
first_gripper_abs_mean  0.304094            0.286550
y_sign_acc              0.613445            0.588235
pred_y_abs_mean         0.000904            0.000806
gt_y_abs_mean           0.002477            0.002477
```

Interpretation:

- Position error stayed almost identical.
- Gripper mismatch improved slightly, but was still too high.
- Y correction became slightly worse:
  - y sign accuracy decreased;
  - predicted y magnitude decreased.
- Repeating lateral data x3 did not make the model more decisive in y.

Therefore, simply repeating the 11 lateral episodes is not an effective fix.

## Main Findings

### 1. XYZ prediction is acceptable but not task-successful

The `first_xyz_l2_mean` stayed around:

```text
0.0042 m
```

This means the low-level delta position prediction is not wildly wrong. However, small average xyz error does not guarantee successful grasping because the failure is phase-sensitive:

- approach;
- lateral align;
- descend;
- close;
- lift;
- move to box;
- open.

The policy can have low average position error while still failing at the critical close/lift moment.

### 2. Y/lateral correction is still too weak

Across tests, GT y magnitude was about:

```text
gt_y_abs_mean ≈ 0.00248
```

But predicted y magnitude was only:

```text
pred_y_abs_mean ≈ 0.0008 - 0.0009
```

So the model predicted only about one third of the necessary lateral correction.

This matches the real-robot observation:

- robot moves mostly forward/downward;
- lateral motion toward cube is weak;
- gripper may pass beside or in front of the cube.

### 3. Gripper timing is unstable

The gripper mismatch remained around:

```text
0.286 - 0.304
```

This means many sampled observations predicted the wrong open/close state.

This also matches the real-robot observation:

- gripper often stayed open too long;
- sometimes closed/opened at the wrong phase;
- model did not robustly switch from approach to grasp.

### 4. Weighting lateral11 x3 did not solve the problem

The lateral x3 model did not increase y magnitude. It made y slightly more conservative.

This suggests the current 11 lateral episodes are not enough or not diverse enough. The model may need more visually clear, phase-isolated correction data rather than repeated copies.

## Recommended Next Steps

### Deployment testing

It is reasonable to do dry-run testing, but execute should remain conservative.

For real robot:

- dry-run first;
- if executing, use `--disable-gripper`;
- use short horizon such as `--max-steps 16`;
- inspect `dpos=[x, y, z]`, especially the second component `y`.

If cube is visibly left/right of gripper and `dpos[1]` remains small, the model is still not ready for full execution.

### Data collection

Collect more targeted DAgger data. The next batch should be more extreme and cleaner than the previous lateral11.

Recommended collection pattern:

- start with gripper already near cube;
- cube is clearly left or right of gripper by about 8-12 cm;
- first 1-2 seconds should be mostly lateral y correction;
- avoid descending immediately;
- after lateral alignment, descend;
- close;
- lift;
- place on box.

Suggested batch:

```text
10 episodes: cube clearly to one side
10 episodes: cube clearly to the other side
5-10 episodes: gripper has overshot cube and must move back/laterally before close
```

These should teach:

```text
visual lateral error -> large y correction
alignment -> descend
near cube -> close
closed -> lift
```

### Model/training

Do not continue training the lateral x3 model for many more steps unless new data is added. The open-loop metrics did not improve in the desired direction.

Better options:

1. Add more targeted correction data.
2. Consider using wrist camera, because third-person camera may not provide enough information for precise gripper-cube alignment.
3. If continuing without wrist camera, make DAgger data more phase-isolated and visually obvious.

## Current Decision

The 100eps and lateralx3 models are useful diagnostics, but neither has clearly solved the true real-robot failure.

The best current checkpoint for cautious dry-run comparison is:

```text
quest3_franka_delta_eef_from_89eps2k_100eps_lateral11_1500
```

The lateralx3 checkpoint slightly improved gripper mismatch but worsened y metrics, so it should not be assumed better for deployment.

Before full execution, use dry-run and verify:

```text
predicted y direction is correct
predicted y magnitude is not too small
gripper is disabled unless close timing is being explicitly tested
```
