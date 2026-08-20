# Consultation Package for the Spatial Forcing Authors

Date: 2026-08-12  
Project: Spatial Forcing adaptation to StarVLA/QwenGR00T on a real Franka robot  
Task: `pick up the cube and place it on the box`

## 1. Purpose of this document

This document is a technical consultation package for contacting the authors of
[Spatial Forcing](https://arxiv.org/abs/2510.12276) about our reproduction and adaptation.
It contains:

1. the minimum context the authors need to understand the experiment;
2. the exact differences between our implementation and their official implementation;
3. the evidence already collected;
4. the questions whose answers would most directly change our next experiment;
5. an email that can be sent with only minor editing;
6. a checklist of files and visual evidence to share.

The official references used by this project are:

- Spatial Forcing paper: <https://arxiv.org/abs/2510.12276>
- Official Spatial Forcing repository: <https://github.com/OpenHelix-Team/Spatial-Forcing>
- Official VGGT repository: <https://github.com/facebookresearch/vggt>

## 2. One-sentence description

We adapted the central Spatial Forcing idea—training-time alignment of an RGB VLA's
intermediate representation with a frozen VGGT teacher—to a StarVLA/QwenGR00T policy for
dual-camera real-world Franka pick-and-place, while removing the teacher at deployment.

## 3. Important scope statement

This work must not be described as a line-by-line reproduction of the official OpenVLA or
Pi0 implementation. The accurate description is:

> A reproduction and adaptation of the core Spatial Forcing method to
> StarVLA/QwenGR00T for local real-world Franka pick-and-place.

The official method and our adaptation share the main training idea:

```text
RGB images -> VLA student -> action loss
      |            |
      +-> frozen VGGT teacher
                   |
student tokens -> projection -> cosine alignment with VGGT features
```

At deployment:

```text
dual RGB images -> RGB-only StarVLA policy -> robot action
```

No VGGT model, depth map, point cloud, or explicit 3D input is used online.

## 4. Why we are asking the authors

Our implementation passes unit, gradient, checkpoint, and RGB-only export tests. The
treatment clearly reduces the VGGT alignment loss and changes the student representation.
However, the current behavioral evidence does not show a consistent Cartesian-action or
real-robot success-rate improvement over a matched control.

The authors' advice would be most valuable for determining whether this is caused by:

- a non-faithful layer or token choice in the StarVLA/QwenGR00T adaptation;
- an incorrect teacher/student spatial-token correspondence;
- inappropriate handling of the primary and wrist camera as VGGT views;
- a mismatch between training augmentation and deployment preprocessing;
- unsuitable LoRA targets, loss weight, or second-stage training schedule;
- insufficient position diversity or evaluation power;
- a limitation of applying the method to a small, locally concentrated real-robot dataset.

## 5. Robot, observations, actions, and data

### 5.1 Robot and task

- Robot: Franka FR3 with a parallel gripper.
- Task instruction: `pick up the cube and place it on the box`.
- Object: approximately 20 mm cube.
- Environment: one tabletop workspace with a box as the placement target.
- Initial robot configuration: a repeated standard initial pose.

### 5.2 Observations

- Primary/static third-person RGB camera.
- Wrist-mounted RGB camera.
- Images are resized to 224 x 224 for the StarVLA student.
- Cameras are ordered consistently as primary, then wrist.
- No metric depth is supplied to the policy.
- Camera intrinsics/extrinsics are not supplied to the alignment objective.
- The primary and wrist images are approximately synchronized through the ROS observation
  pipeline, but are not geometrically calibrated for training.
- Current StarVLA dataset setting: `include_state: false`.

The lack of calibration is intentional because our interpretation of Spatial Forcing is
that the 3D teacher provides representation supervision from RGB, rather than requiring
explicit calibrated geometry at deployment. We nevertheless need the authors to confirm
whether this interpretation remains valid for a static-plus-moving dual-camera setup.

### 5.3 Action representation

- Action chunk shape: 8 x 7.
- Seven action dimensions:
  - delta end-effector XYZ;
  - delta end-effector roll/pitch/yaw;
  - binary gripper state.
- Deployment normally executes a short horizon from every predicted chunk and replans.

### 5.4 Demonstration data

- Data format: LeRobot episodes collected by human teleoperation.
- Original useful set: 74 dual-camera episodes concentrated near one cube position.
- Replay94 set: the successful 74-episode set plus 20 additional locally shifted episodes.
- The added episodes expand the cube distribution in the front/back direction near the
  original position; they do not constitute broad workspace coverage.
- The local evaluation is intended to test interpolation and small position shifts, not
  broad out-of-distribution generalization.

The training-data manifest, selected episode IDs, and object-position definitions should be
shared with the authors. Raw robot data or private Hugging Face credentials should not be
included in the first email.

## 6. Starting checkpoints and baselines

### 6.1 Successful 74-episode baseline

The historically strongest real-robot model was initialized from a Libero-trained StarVLA
checkpoint and then trained on 74 real Franka demonstrations. With the same generic
deployment safety layer, it previously achieved:

```text
13 / 20 = 65% safety-filtered end-to-end success
```

This result is useful context but is not a matched Spatial Forcing control because it does
not include the same LoRA adaptation and training schedule.

### 6.2 Replay94 starting checkpoint

The current Phase 10 experiments start from a successful Replay94 checkpoint:

```text
/data/hanyu/starVLA_runs/
replay94_from_successful_libero74_5k_retry1_20260810/
final_model/pytorch_model.pt
```

This checkpoint was obtained by continuing the successful Libero-initialized real-robot
model on Replay94 for 5,000 steps, using approximately:

```text
action-model LR: 3e-5
Qwen interface LR: 1e-7
```

The authors should be told that Spatial Forcing is currently applied as a second adaptation
stage after successful behavior cloning, not from the original Libero checkpoint. This may
be important because the representation might already be highly specialized before the
alignment objective is introduced.

## 7. Our Phase 10 clean implementation

### 7.1 Objective

The clean treatment intentionally removes experimental relational and scene-memory losses.
It retains only action loss and the projected cosine alignment objective:

```text
L_total = L_action + alpha * L_projected_alignment
```

Matched control:

```text
alpha = 0
```

Spatial Forcing treatment:

```text
alpha > 0
```

The control and treatment use the same starting checkpoint, data order, seed, student
architecture, LoRA configuration, augmentation, optimizer, and number of steps. The intended
causal difference is the nonzero alignment weight.

### 7.2 Student policy

- Framework: StarVLA/QwenGR00T.
- Visual-language model: Qwen3-VL-4B-Instruct.
- Action model: DiT-B action head.
- Selected student hidden-state index: 24.
- Student features are image-token hidden states from the multimodal Qwen sequence.
- The selected hidden-state index counts the embedding output, so the semantic mapping to
  an OpenVLA layer number may not be equivalent.

This layer choice is one of the most important questions for the authors.

### 7.3 Frozen VGGT teacher

- Teacher: official VGGT-1B checkpoint.
- Teacher is frozen and used in feature-only mode.
- Teacher precision: BF16.
- Teacher input resolution: 518 x 518.
- Selected feature: the last aggregated feature stage.
- Teacher feature dimension: 2048.
- Camera, depth, point, and tracking heads are disabled.
- Special tokens before the patch-token region are removed.
- An additional scaled 2D positional embedding is currently used with ratio 0.1.

The last item may differ from the intended official method and must be disclosed explicitly.

### 7.4 Token correspondence

For each view:

1. Qwen image-token spans are identified in the multimodal sequence.
2. The Qwen token grid is inferred from `image_grid_thw` and the Qwen spatial merge size.
3. VGGT patch tokens are reshaped into their teacher patch grid.
4. The teacher grid is bilinearly resampled to the Qwen visual-token grid.
5. Resampling currently uses `align_corners=True`.
6. Projected Qwen tokens and resampled VGGT tokens are compared token by token.
7. The loss is averaged within each view and then averaged across the two views.

This is an engineering interpretation of spatial correspondence. It is not yet confirmed
that it matches the authors' recommended token mapping for a Qwen-based VLA.

### 7.5 Projection head and alignment loss

The student projection head is:

```text
LayerNorm(student_dim)
-> Linear(student_dim, 2048)
-> GELU
-> Linear(2048, teacher_dim=2048)
```

The projected student tokens and frozen teacher tokens are compared using mean tokenwise
cosine distance:

```text
mean(1 - cosine(projected_student_token, teacher_token))
```

The teacher target is detached. The alignment head and student LoRA receive gradients.

### 7.6 LoRA and trainable modules

- The Qwen base model is frozen.
- LoRA is applied to all linear layers in the Qwen student.
- LoRA rank: 32.
- LoRA alpha: 16.
- LoRA dropout: 0.
- LoRA-B starts at zero, making the adapter initially close to a no-op.
- The DiT action head remains trainable.
- The alignment projection head is trainable.
- The VGGT teacher is frozen.

We need to confirm whether all-linear LoRA is appropriate, or whether Spatial Forcing should
target the visual encoder, multimodal projector, attention Q/K/V/O layers, or a different
subset so that the spatial gradient reaches the relevant visual representation.

### 7.7 Paired image augmentation

The same materialized image transform is supplied to the student and teacher for each
sample. The current transform includes:

- random square crop with approximately 0.9 retained area;
- resize back to the model input size;
- brightness, contrast, saturation, and hue jitter;
- randomized color-operation order.

The exported RGB-only inference configuration currently sets:

```text
teacher_enabled: false
image_augmentation_enabled: false
```

A potentially important unresolved issue is whether inference should apply a deterministic
center crop corresponding to the training random crop, rather than simply disabling the
augmentation. This is a high-priority question for the authors.

## 8. Matched experiments and current evidence

### 8.1 High-learning-rate 5k experiment

Both the alpha=0 control and alpha=0.1 treatment were trained for 5,000 steps with Qwen/LoRA
and action learning rates near 1e-4. Both substantially regressed relative to the Replay94
starting checkpoint.

Matched open-loop results over 224 queries:

| Metric | alpha=0 control | alpha=0.1 treatment |
|---|---:|---:|
| First-action L2 mean | 0.075788 | 0.088834 |
| First XYZ L2 mean | 0.004466 | 0.004261 |
| Chunk L2 mean | 0.080621 | 0.085327 |
| Gripper accuracy | 0.928571 | 0.915179 |
| False-close rate | 0.049180 | 0.049180 |
| Missed-close rate | 0.098039 | 0.127451 |
| First close-frame error | -20 | -20 |

Interpretation: the shared second-stage LoRA/action training schedule was too aggressive.
This experiment cannot isolate Spatial Forcing as the source of the regression because the
alpha=0 control also regressed.

Relevant W&B runs:

- Control: <https://wandb.ai/u3666250-the-university-of-hong-kong/starVLA_Quest3_Franka/runs/a0kjwrp9>
- Treatment: <https://wandb.ai/u3666250-the-university-of-hong-kong/starVLA_Quest3_Franka/runs/sit3zl6k>

### 8.2 Low-learning-rate 500-step experiment

To preserve the successful policy, we then used a short, lower-learning-rate matched stage:

```text
steps: 500
seed: 42
action-model LR: 3e-5
Qwen LoRA LR: 1e-5
alignment-head LR: 1e-4
base LR: 1e-6
warmup steps: 50
control alpha: 0
treatment alpha: 0.1
```

Matched open-loop results over the same 224 queries:

| Metric | Replay94 start | Low-LR control | Low-LR treatment |
|---|---:|---:|---:|
| First-action L2 mean | 0.020724 | 0.016514 | **0.012156** |
| First-action L2 median | 0.001993 | **0.002393** | 0.002522 |
| First-action L2 p90 | **0.006809** | 0.006920 | 0.007076 |
| First XYZ L2 mean | **0.002876** | 0.003113 | 0.003214 |
| Chunk L2 mean | 0.022820 | **0.017009** | 0.018190 |
| Gripper absolute error | 0.017857 | 0.013393 | **0.008929** |
| Gripper accuracy | 0.982143 | 0.986607 | **0.991071** |
| False-close rate | 0.033058 | 0.016529 | **0.008264** |
| Missed-close rate | **0.000000** | 0.009709 | 0.009709 |
| First close-frame error | -5 | 0 | 0 |

Interpretation:

- the low learning rate avoided the severe 5k regression;
- the treatment improved aggregate first-action L2 and gripper classification;
- the treatment did not improve first-action Cartesian XYZ error over the control;
- therefore the current result is promising for gripper prediction but is not evidence of
  improved spatial action accuracy.

### 8.3 Representation audit

For an earlier matched alpha=0 versus alpha=0.1 comparison, treatment-minus-control metrics
were:

```text
linear CKA:              +0.023377
position RSA:            -0.259838
shared-probe loss:       -0.000188
```

The alignment treatment made the student representation more linearly similar to the VGGT
teacher and easier for a shared probe, but the position-distance structure measured by RSA
became worse. The accurate conclusion is:

> The treatment learned a teacher-correlated representation, but transfer of useful spatial
> geometry to policy behavior has not been demonstrated.

The following visualization is available:

```text
/home/dase-hw101/franka_ws/artifacts/
replay94_phase10_visual_audit_20260811/
position_distance_heatmaps.svg
```

### 8.4 Stochastic action-vector-field audit

We probed the same robot state and three local object positions with repeated policy samples.
Five-sample front-to-back slopes were:

| Model | First-action dx slope mean | Chunk-mean dx slope mean |
|---|---:|---:|
| Replay94 baseline | -0.001242 | -0.000516 |
| Low-LR control | -0.000494 | -0.001967 |
| Low-LR treatment | -0.001581 | -0.002673 |

The diffusion sampling variance was larger than the effect of the approximately 3 cm object
shift. Consequently, these slopes are not yet a valid comparison. The next evaluator should
use common random numbers: the same explicit inference seed for every model and object
position, repeated over a predefined seed list.

This is also a question for the authors: how was stochastic policy inference controlled in
their ablations and real-world evaluation?

### 8.5 Real-robot evidence

- Historical 74-episode model: 13/20 under a generic safety-filtered deployment protocol.
- Earlier Spatial Forcing pilots: mixed outcomes and confounded by object contact, workspace
  aborts, or lucky cube displacement.
- Current low-LR control and treatment: no formal paired real-robot success-rate evaluation
  has been completed.

We should not report a Spatial Forcing real-robot SR improvement until the control and
treatment are evaluated at the same predefined placements, with the same seeds/settings and
an episode-level success rubric.

## 9. Deployment logic that must be disclosed

The robot does not execute completely raw policy outputs. Both matched models use the same
generic deployment layer, including:

- Cartesian workspace bounds;
- maximum translation and rotation deltas;
- stale-observation aborts;
- action-chunk gripper consensus;
- consecutive gripper switch confirmation;
- close latch during grasp validation;
- measured gripper-width validation;
- minimum measured lift validation;
- maximum grasp-attempt rules.

These filters are safety and command-stability mechanisms, not object-coordinate scripts,
but they can materially affect measured success rate and gripper timing. The exact client
command and filter thresholds must accompany any real-robot result sent to the authors.

## 10. Highest-priority questions for the authors

The first email should ask only the highest-priority questions. The remaining questions can
be placed in the attached technical note.

### Q1. Which StarVLA/Qwen intermediate representation should be aligned?

> In your OpenVLA/Pi0 implementations, what property guided the selection of the aligned
> student layer? For a Qwen3-VL-based VLA, should we align visual-encoder patch tokens before
> the multimodal projector, projected image tokens before the LLM, LLM hidden states at image
> token positions, or action-conditioned tokens? We currently align Qwen multimodal hidden
> state 24 at the image-token positions. Is this semantically comparable to the layer used in
> your implementation?

Why this matters: using the same integer layer index across architectures does not guarantee
the same visual or geometric role.

### Q2. Are we using the correct VGGT feature and special-token handling?

> We use the last VGGT aggregated feature stage, remove tokens preceding the patch grid, and
> align the remaining patch tokens. Which exact VGGT layer/output and token subset do you
> recommend? Should camera/register tokens participate in the objective? Did your method add
> an extra 2D positional embedding to the teacher target, or should we remove our additional
> positional embedding with scale 0.1?

Why this matters: the teacher target may currently contain a positional shortcut that is not
part of the intended method.

### Q3. Is our student/teacher token correspondence faithful?

> Our Qwen and VGGT token grids have different resolutions. We bilinearly resample each VGGT
> patch grid to the merged Qwen visual-token grid with `align_corners=True`, then apply
> tokenwise cosine alignment. Is this the intended correspondence? Would you recommend
> average pooling, a learned resampler, `align_corners=False`, feature-pyramid features, or a
> different treatment of Qwen's spatial merge?

Why this matters: a mathematically valid interpolation can still destroy the geometric
correspondence the loss is meant to transfer.

### Q4. How should a static primary camera and moving wrist camera be handled?

> Should primary and wrist RGB frames be passed jointly to VGGT as two views of one scene,
> or should VGGT be run independently per camera? Does the method assume known or stable
> cross-view geometry, camera calibration, synchronized frames, or similar intrinsics? Is a
> moving wrist view appropriate for the same alignment loss as a static external view?

Why this matters: VGGT's multi-view features may use relationships that do not match our
uncalibrated static-plus-moving camera setup.

### Q5. What preprocessing should be used at deployment?

> During training, the student and teacher receive exactly the same random crop and color
> augmentation. Our RGB-only export currently disables augmentation entirely. Should
> deployment instead apply the corresponding deterministic center crop used by your
> evaluation pipeline? What crop scale and resize order should remain identical between
> training and deployment?

Why this matters: an otherwise correct Spatial Forcing model can fail if image coordinates
shift between training and inference.

### Q6. Which LoRA modules should receive the alignment gradient?

> We freeze Qwen and add rank-32, alpha-16 LoRA to all Qwen linear layers; the action head and
> alignment head are trainable. Is all-linear LoRA appropriate, or should LoRA be restricted
> to visual-attention Q/K/V/O, the visual encoder, or the multimodal projector? Should the
> action head be fully trainable during the alignment stage?

Why this matters: our alignment head can improve while the action-relevant visual path
changes too little—or while broad language-model adaptation damages a previously successful
policy.

### Q7. How should alpha be selected across VLA architectures?

> Is the reported alignment weight expected to transfer directly across VLA architectures,
> or should it be chosen from action/alignment gradient norms? Do you recommend an alignment
> warm-up or ramp, normalization by token/view count, or gradient balancing? In our StarVLA
> adaptation, alpha=0.1 lowers alignment loss but has not improved Cartesian action error.

Why this matters: identical alpha values do not imply identical gradient influence when
action loss scales and token counts differ.

### Q8. When should Spatial Forcing be introduced?

> Should control and treatment start from the same original pretrained VLA and receive the
> 94 demonstrations jointly with/without alignment, or is it valid to add Spatial Forcing as
> a short second stage after successful real-data behavior cloning? Could the second-stage
> setup prevent the teacher signal from influencing the relevant visual representation?

Why this matters: our current low-LR treatment begins from an already successful Replay94
checkpoint and runs for only 500 additional steps.

### Q9. What evidence did you use to verify spatial transfer?

> Besides alignment loss, which diagnostics best predicted downstream success in your
> experiments: linear CKA, RSA, depth/point probes, attention maps, action-vector fields, or
> held-out pose success? How do you rule out the projection head absorbing the teacher signal
> without meaningful student adaptation?

Why this matters: our CKA and shared-probe metrics improve while position RSA and Cartesian
action metrics do not consistently improve.

### Q10. How should stochastic policy evaluation be controlled?

> StarVLA uses stochastic diffusion action sampling, and repeated predictions at the same
> observation can vary more than the effect of a 3 cm cube shift. Did your matched evaluations
> use fixed inference seeds/common random numbers, deterministic sampling, or multiple
> rollouts per observation? How many seeds or real-robot trials would you recommend?

Why this matters: single-sample action-vector comparisons can reverse sign across repeats.

## 11. Secondary questions

### Q11. Is our projection head capable of hiding a failed transfer?

> Is LayerNorm -> MLP -> cosine loss an appropriate head, or should the projection be linear
> and lower-capacity? Do you stop-gradient or normalize features differently? What result
> would indicate that only the head, rather than the student, learned the teacher target?

### Q12. How much spatial diversity is necessary?

> Our data contains 94 demonstrations, mostly near one location, with 20 additional local
> front/back placements. Is this sufficient for local interpolation, or does Spatial Forcing
> require a broader distribution of object/camera poses to make geometric supervision
> behaviorally identifiable? Would you recommend a 3x3 grid and held-out cell centers?

### Q13. Should proprioceptive state be included?

> Our current StarVLA setting has `include_state: false`. Would you expect geometric feature
> alignment to improve action prediction without explicit end-effector state, or should
> proprioception be included so that the policy can convert visual geometry into robot-frame
> deltas?

### Q14. How should view losses be weighted?

> We average the primary and wrist alignment losses equally. Should the primary view receive
> more weight, should visibility masks be used, or should alignment be disabled when the
> object/end effector is cropped from the wrist view?

### Q15. Are our real-robot filters comparable to your protocol?

> Did your real-world experiments apply temporal gripper confirmation, action temporal
> ensembling, close latching, contact/lift validation, or workspace clipping? Should SR be
> reported with raw-policy execution, with a generic safety controller, or both?

### Q16. What is the most informative next ablation?

> Given alignment-loss reduction but no clear Cartesian improvement, which one experiment
> would you prioritize: student-layer sweep, VGGT-layer sweep, per-view alignment, removing
> the added teacher positional embedding, deterministic center-crop inference, training from
> the pre-real-data checkpoint, or a longer low-LR matched run?

## 12. What to send in the first contact

Do not attach every checkpoint and log in the initial message. Send:

1. the short email in Section 15;
2. this document or a shortened PDF version;
3. one architecture diagram;
4. the exact matched control/treatment configuration diff;
5. the low-LR open-loop comparison table;
6. the alignment-loss and parameter-update curves;
7. the position-distance heatmap;
8. links to two short paired real-robot videos only if they use matched conditions.

Offer the following on request:

- minimal code patch against StarVLA;
- complete configuration files;
- dataset episode manifest and split definitions;
- open-loop per-query CSV files;
- common-seed probe outputs;
- policy-server and deployment logs;
- RGB-only exported checkpoints;
- full 9.5 GB checkpoints.

## 13. Reproducibility information to include

Before sending, fill in every `[TO FILL]` field:

```text
StarVLA source commit:                 [TO FILL]
Spatial Forcing reference commit:     [TO FILL]
VGGT source/checkpoint version:        [TO FILL]
PyTorch version:                       2.6.0+cu124
Transformers version:                  4.57.0
PEFT version:                          [TO FILL]
DeepSpeed version:                     0.16.9
CUDA runtime used by PyTorch:          12.4
NVIDIA driver:                         595.84
GPU used for the matched run:          NVIDIA H100/H100 PCIe [TO FILL]
Random seed:                           42
Dataset manifest checksum:             [TO FILL]
Control config checksum:               [TO FILL]
Treatment config checksum:             [TO FILL]
Starting checkpoint checksum:          [TO FILL]
```

Also provide:

- batch size and gradient accumulation;
- optimizer, betas, weight decay, and scheduler;
- exact number of training samples/frames;
- steps per effective epoch;
- image normalization and resize/crop order;
- exact student and teacher tensor shapes at the aligned layer;
- exact hidden-state indexing convention;
- full trainable/frozen parameter counts;
- alignment/action gradient norms, if available.

## 14. Minimal architecture/tensor table to complete

This table will help the authors find a shape or semantic mismatch quickly:

| Stage | Primary shape | Wrist shape | Notes |
|---|---|---|---|
| Raw RGB | `[B,3,H,W]` | `[B,3,H,W]` | `[TO FILL H,W]` |
| Student input | `[B,3,224,224]` | `[B,3,224,224]` | same paired augmentation |
| Qwen image grid | `[TO FILL]` | `[TO FILL]` | before/after spatial merge |
| Student image tokens | `[B,Ns,Ds]` | `[B,Ns,Ds]` | hidden-state index 24 |
| VGGT input | `[B,3,518,518]` | `[B,3,518,518]` | joint or per-view: `[TO FILL]` |
| VGGT patch grid | `[B,Ht,Wt,2048]` | `[B,Ht,Wt,2048]` | last aggregated feature |
| Resampled teacher | `[B,Ns,2048]` | `[B,Ns,2048]` | bilinear, align_corners=True |
| Projected student | `[B,Ns,2048]` | `[B,Ns,2048]` | tokenwise cosine loss |

## 15. Ready-to-send email draft

**Subject:** Questions about adapting Spatial Forcing to StarVLA/QwenGR00T for real-world Franka manipulation

Dear Spatial Forcing authors,

Thank you for releasing Spatial Forcing. We are adapting the core method to a
StarVLA/QwenGR00T policy for dual-camera real-world Franka pick-and-place, and we would be
very grateful for your advice on several implementation choices.

This is not a line-by-line reproduction of your OpenVLA/Pi0 code. Our student is a
Qwen3-VL-based VLA with a DiT action head. During training, we freeze VGGT-1B, project an
intermediate Qwen image-token representation to the VGGT feature dimension, and optimize:

```text
L = L_action + alpha * L_projected_cosine_alignment
```

The primary and wrist RGB images receive the same paired augmentation in the student and
teacher paths. VGGT and the alignment head are removed at deployment, so inference remains
RGB-only.

Our matched alpha=0 and alpha=0.1 runs confirm that the alignment loss decreases and the
student becomes more teacher-correlated. A low-learning-rate 500-step alpha=0.1 treatment
also improved aggregate first-action L2 and gripper accuracy over its matched control.
However, it did not improve Cartesian XYZ action error, and our current representation
metrics are mixed. We therefore do not yet claim a real-robot SR improvement.

Our most important questions are:

1. For a Qwen3-VL VLA, should alignment target visual-encoder tokens, projected visual
   tokens, or LLM hidden states at image-token positions? We currently use Qwen hidden state
   24.
2. Which exact VGGT feature stage and token subset should be used? Should camera/register
   tokens or an additional 2D positional embedding be included?
3. Is bilinear resampling of the VGGT patch grid to Qwen's merged visual-token grid an
   appropriate token correspondence?
4. For a static primary camera plus moving wrist camera, should VGGT process the views
   jointly or independently? Is camera calibration expected for training-time alignment?
5. If paired random crop is used in training, should RGB-only deployment use a corresponding
   deterministic center crop rather than simply disabling augmentation?
6. Do you recommend all-linear Qwen LoRA, or a narrower visual/attention LoRA target? Should
   the action head remain trainable during alignment?
7. How should alpha be calibrated across architectures with different action-loss scales?
8. Is Spatial Forcing best introduced jointly from the pre-real-data checkpoint, rather than
   as a short second stage after successful behavior cloning?
9. How did you control stochastic policy sampling in matched offline and real-world
   comparisons?

We have attached a concise technical note containing architecture details, tensor mapping,
matched results, and our current failure analysis. We can also provide the minimal StarVLA
patch, full configs, per-query evaluation CSVs, and checkpoints if useful.

Thank you very much for your time and for releasing the work.

Best regards,  
Hanyu  
The University of Hong Kong  
[email address]

## 16. Suggested concise message for GitHub Issues

Use this only if the repository permits implementation questions. Do not paste the complete
technical report into an issue.

> Thank you for releasing Spatial Forcing. We are adapting the method to a Qwen3-VL-based
> StarVLA policy for dual-camera real-world Franka manipulation. Our frozen-VGGT projected
> cosine loss trains correctly, but teacher-correlated representation gains have not yet
> translated into better Cartesian action error. Could you clarify three choices: (1) whether
> to align pre-projector visual tokens or LLM image-token hidden states in a new VLA, (2) how
> you resample/match VGGT and student patch grids, and (3) whether static and wrist camera
> views should be processed jointly by VGGT? A compact implementation note and config diff
> are available here: [link].

## 17. Evidence-status statement

Use this exact wording in presentations or correspondence:

```text
Implementation integration:                 PASS
Frozen-teacher and gradient audits:          PASS
Alignment-loss optimization:                PASS
RGB-only deployment export:                 PASS
Teacher-correlated representation change:   OBSERVED
Improved Cartesian action accuracy:          NOT YET DEMONSTRATED
Improved real-robot success rate:            NOT YET DEMONSTRATED
```

Do not write “Spatial Forcing learned useful 3D geometry” based only on lower alignment loss
or higher CKA. The defensible statement is that the treatment learned a representation more
correlated with the frozen VGGT teacher; usefulness for robot behavior remains under test.

## 18. Privacy and sharing checklist

Before uploading this document or a reproduction bundle:

- remove IP addresses and SSH usernames if they should not be public;
- remove passwords, Hugging Face tokens, W&B API keys, and robot credentials;
- remove Quest/ADB device serial numbers;
- replace internal absolute paths with relative paths in the public reproduction package;
- confirm that demonstration images may be shared;
- blur people, screens, and lab-sensitive information in robot videos;
- do not upload 9.5 GB checkpoints until the authors request them;
- include checksums for any shared checkpoint or archive;
- state clearly which code is ours and which code originates from the official repositories.

## 19. Recommended sequence after receiving a reply

1. Correct any layer, token, view, or preprocessing mismatch identified by the authors.
2. Add explicit inference seeds and common-random-number evaluation.
3. Run a small layer/token/preprocessing ablation before another 5k training job.
4. Select one control and one treatment using only predefined offline criteria.
5. Perform paired real-robot trials at predefined middle and held-out local positions.
6. Report raw success, safety-filtered success, failure category, and confidence intervals.
7. Only after the matched comparison should additional data, RL, or temporal-policy changes
   be introduced; otherwise the Spatial Forcing effect cannot be isolated.

