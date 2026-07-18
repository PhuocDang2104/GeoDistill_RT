# GeoDistill-RT Ablation Protocol

## Architecture Search, Evaluation Matrix, and Final-Model Selection

> **Document status.** This is an executable experiment specification and a results template, not a table of claimed results. Cells marked `—` are filled only from measured runs. The final architecture is selected by the frozen rules in Section 11; it is not chosen manually after viewing the test set.

---

## 1. Questions the ablation must answer

The experiment suite tests five hypotheses independently.

| Hypothesis | Required evidence |
|---|---|
| H1: sparse median normalization improves scale robustness | lower scale-equivariance error and better scaled/cross-sensor evaluation without material in-domain regression |
| H2: SE-MSAR is better than a fixed sparse prior | lower far-range and large-hole error at a better accuracy-latency Pareto point than local propagation or multi-step SPN |
| H3: SP-BRP improves boundaries without moving global scale | lower 3 px/5 px boundary error and gradient error, with stable global RMSE and $|\mathrm{Bias}_{\log}|$ |
| H4: separated, conflict-aware distillation is better than uniform teacher use | lower RMSE and teacher-conflict-region error without putting DMD3C into its own agreement reference |
| H5: the full student is deployable | batch-1 P95 latency, peak memory, parameter count, and export status meet a target fixed before final training |

The main comparison separates architecture gain from distillation gain:

$$
\text{student architecture}\quad\perp\quad\text{teacher supervision}.
$$

No test-set result is used for module selection or hyperparameter tuning.

---

## 2. Implementation gate before training

The checked-in code is the executable reference baseline. It does not yet expose the complete GeoRT-SAR search space. An ablation row is valid only when its switch changes the actual forward/loss graph and is recorded in the checkpoint.

### 2.1 Code readiness matrix

| Area | Current repository behavior | Requirement for V2 ablation |
|---|---|---|
| Sparse scale | `model_geort.py` uses `log1p(depth) / log(max_depth)` | add per-sample valid median $m_S$, normalized sparse depth, and metric reconstruction by $m_S\exp z$ |
| Sparse prior | `fast_sparse_propagation` uses local normalized fill or global analytic KNN at $1/4$ | add multi-radius analytic banks and validity at $1/8$ and $1/4$ |
| Routing | `SparseRayInjection` gates projected prior features | add learned radius logits, invalid-radius masking, per-pixel gate, and bounded log-depth update |
| Boundary path | guided convex upsampling plus an ordinary full-resolution residual | add half/full edge-gated high-pass residual heads with bounded amplitudes |
| Sparse correction | fixed `sparse_anchor_lambda=0.7` | add none, fixed, adaptive, and hard modes |
| Metric teacher | `D_cm` overlays GT on a DMD3C-dominant map | retain raw $D_{\mathrm{DMD}}/C_{\mathrm{DMD}}$ and compute GT, sparse, and DMD losses independently |
| Geometry teacher | generator can pass DMD3C into `build_geometry_teacher`; config prior is 0.25 | main V2 target excludes DMD3C; set the prior to zero/remove the candidate and regenerate pseudo labels |
| Losses | GT, sparse, metric, auxiliary, range, SSI, ordinal, confidence, smoothness are implemented | add edge-band, gradient, scale-equivariance, and optional 3D terms behind independent flags |
| Metrics | RMSE, MAE, AbsRel, $\delta$, range, and RGB-edge metrics exist | add iRMSE/iMAE, GT-depth boundary bands, gradient error, ordinal accuracy, anchor error, log bias, confidence, robustness, latency, and memory |

Adding unused YAML keys does not implement an ablation. `GeoRTStudentS.from_config` and `geort_loss` must consume every declared switch, and each run must save the fully resolved configuration.

### 2.2 Required contract tests

| Test ID | Contract | Pass condition |
|---|---|---|
| U0 | output shapes | `D_full/C_full = [B,1,H,W]`; `D_1_4/C_1_4 = [B,1,H/4,W/4]` |
| U1 | finite empty-anchor path | no NaN/Inf when one sample has no valid sparse point; anchor updates are zero |
| U2 | constant-anchor preservation | a constant valid sparse field produces $A_{s,r}=c/m_S$ for every valid radius, within numerical tolerance |
| U3 | invalid-radius masking | an invalid proposal receives zero routing probability and softmax remains finite |
| U4 | bounded routing | $|z_s^+-z_s|\le\tau_s+10^{-6}$ at every pixel |
| U5 | scale equivariance | in eval mode and away from depth clipping, mean $|\log D_\beta-\log(\beta D_1)|<10^{-4}$ for $\beta\in\{0.5,0.75,1.5,2\}$ |
| U6 | high-pass DC rejection | interior mean of $\mathcal H_k(c)$ is zero within $10^{-6}$ for a constant tensor |
| U7 | residual amplitude | $|r_{1/2}^{\mathrm{raw}}|\le a_{1/2}$ and $|r_1^{\mathrm{raw}}|\le a_1$ |
| U8 | sparse correction | none/fixed/adaptive/hard modes match their equations; hard mode has zero valid-anchor error |
| U9 | teacher separation | main `R_G` is unchanged when only DMD3C geometry input changes; DMD confidence agreement uses a non-DMD reference |
| U10 | inference isolation | the exported student loads and runs without teacher repositories or teacher-output files |
| U11 | export parity | PyTorch and exported backend outputs meet declared absolute/relative tolerances on at least 20 samples |

Run these tests before every screening batch. Failed rows are implementation failures, not negative research results.

---

## 3. Frozen experimental protocol

### 3.1 Data and preprocessing

Record these fields before the first run:

| Field | Frozen value |
|---|---|
| Dataset | KITTI Depth Completion |
| Train split | repository `train.txt`; current depth-selection layout contains 800 training samples |
| Validation split | repository `val.txt`; current depth-selection layout contains 200 validation samples |
| Test split | official test server or a held-out split used once after selection |
| Input resolution | $352\times1216$ unless a separate resolution ablation is declared |
| Depth unit | metres internally; report KITTI RMSE/MAE in millimetres when comparing with benchmark convention |
| Valid depth | $0.001<D<120$ m |
| Evaluation mask/crop | exact KITTI protocol; save mask version with results |
| Teacher files | one immutable pseudo-label version/hash shared by compared runs |
| Base seeds | 42 for screening; 42, 3407, 9191 for finalists |

For a nominal 1,000-image screen, use $N_{\mathrm{screen}}=\min(1000,N_{\mathrm{train}})$; with the current split this is 800. A 10,000-image phase requires a larger training split and must keep the same validation set or define a new frozen split before any result is viewed.

### 3.2 Optimization controls

All rows in one comparison block use

- equal optimizer steps rather than merely equal epochs;
- the same sampler order for the same seed;
- the same batch size, optimizer, LR schedule, AMP mode, gradient clipping, and initialization policy;
- the same pretrained encoder checkpoint;
- the same augmentation stream unless augmentation itself is the factor;
- the same curriculum boundaries when comparing student modules;
- no test-set tuning.

If a row runs out of memory, reduce batch size for the whole comparison block or use gradient accumulation so the effective batch and step count remain equal.

### 3.3 Latency controls

Fill the deployment header before benchmarking:

| Field | Value |
|---|---|
| Target device | `[DEVICE MODEL]` |
| Driver / CUDA / cuDNN | `[VERSIONS]` |
| PyTorch / ONNX / TensorRT | `[VERSIONS]` |
| Precision | FP32 and FP16 |
| Batch | 1 |
| Resolution | $352\times1216$ |
| Warm-up | 100 iterations |
| Timed iterations | at least 500 |
| Timing | device synchronization before and after every measured iteration |
| Included path | normalization, anchor bank, neural forward, sparse correction, and required device transfers |
| Report | median, P95, FPS, peak allocated memory |

Latency from another paper, GPU, resolution, framework, or excluded preprocessing path is not placed in the same numerical column.

---

## 4. Configuration and artifact contract

Every run uses a unique ID and saves

```text
student_outputs/ablations/<RUN_ID>/
  resolved_config.yaml
  environment.json
  git_state.txt
  train.csv
  val_per_sample.csv
  summary.json
  latency.json
  checkpoints/best.pth
  predictions/<split>/*.npz
```

The run manifest records dataset/split hashes, teacher-output hashes, code commit plus dirty diff hash, seed, trainable parameter count, optimizer steps, best-checkpoint selection metric, and failure status.

A proposed configuration namespace is

```yaml
experiment:
  id: A09_seed42
  seed: 42

normalization:
  mode: sparse_median          # none | sparse_median
  representation: log_depth   # linear | inverse | log_depth | log_residual

anchor_bank:
  enabled: true
  scales: [8, 4]
  radii_s8: [3, 7, 15, 31]
  radii_s4: [1, 3, 7, 15]

routing:
  enabled: true
  scales: [8, 4]
  updates_per_scale: 1
  gate: per_pixel
  validity_logit: true
  use_distance: true
  use_ray: true
  tau_s8: 0.30
  tau_s4: 0.15

boundary:
  mode: sp_brp               # bilinear | convex | ordinary | sp_brp
  levels: [2, 1]
  highpass_kernel: 5
  width: 16
  amp_half: 0.10
  amp_full: 0.05
  edge_gate: true

correction:
  mode: adaptive_soft        # none | fixed_soft | adaptive_soft | hard
  fixed_lambda: 0.7
```

These keys become valid only after being wired into model construction, checkpoint metadata, and inference.

---

## 5. Main incremental architecture matrix

This is the primary architecture table. Rows A00--A09 use exactly the same loss set and base supervision: GT + sparse + uniform DMD3C. A common GT edge-band loss may be enabled for every row, but no loss is activated only because a particular architecture row contains a boundary head. Geometry distillation stays disabled in this block and is tested later. Row C00 measures the current checked-in student as an external reference and is not part of the additive chain.

Legend: `U` = fixed valid-uniform anchor update, `L` = learned SE-MSAR, `HP` = high-pass SP-BRP.

| ID | Median/log | Sparse prior | Route $1/8$ | Route $1/4$ | Upsampling / residual | Correction | Full confidence head | Question |
|---|:---:|---|:---:|:---:|---|---|:---:|---|
| C00 |  | local propagation |  |  | guided convex + ordinary full residual | fixed 0.7 |  | checked-in reference student; coarse confidence is only upsampled |
| A00 |  | none |  |  | bilinear | none |  | clean mobile encoder + additive FPN baseline |
| A01 | ✓ | none |  |  | bilinear | none |  | value of normalized log-depth alone |
| A02 | ✓ | analytic bank |  | U | bilinear | none |  | value of a fixed analytic anchor update |
| A03 | ✓ | analytic bank |  | L | bilinear | none |  | learned routing versus fixed uniform routing |
| A04 | ✓ | analytic bank | L | L | bilinear | none |  | value of the second routing scale |
| A05 | ✓ | analytic bank | L | L | ordinary full residual | none |  | unconstrained full-resolution refinement reference |
| A06 | ✓ | analytic bank | L | L | HP at $1/2$ only | none |  | high-pass boundary correction at one level |
| A07 | ✓ | analytic bank | L | L | HP at $1/2$ and $1$ | none |  | full SP-BRP |
| A08 | ✓ | analytic bank | L | L | full SP-BRP | adaptive soft |  | real-anchor preservation under sensor noise |
| A09 | ✓ | analytic bank | L | L | full SP-BRP | adaptive soft | ✓ | full GeoRT-SAR-S output contract |

A08 uses the upsampled coarse reliability $C_{1/4}$ for adaptive correction. A09 replaces or refines it with the dedicated full-resolution reliability head; this keeps the depth architecture comparison separate from confidence calibration.

### Promotion rule for the incremental chain

A module advances when all conditions hold on the same validation split:

1. the primary target metric improves in the expected region;
2. global RMSE does not regress by more than 0.5% unless latency improves by at least 10%;
3. P95 latency and memory remain inside the declared budget;
4. the improvement direction is reproduced by at least two finalist seeds;
5. no unit-contract or stability failure occurs.

Expected target regions are scale tests for A01, far/large-hole regions for A02--A04, boundary metrics for A05--A07, anchor/noise tests for A08, and calibration metrics for A09.

---

## 6. Component search matrices

### 6.1 SE-MSAR search

Run this block after fixing the encoder and before tuning SP-BRP.

| ID | Proposal / router | Update | Extra inputs | Purpose |
|---|---|---|---|---|
| R00 | raw sparse concat | none | mask | no analytic proposal |
| R01 | nearest-neighbor fill | fixed | mask | cheap fixed fill |
| R02 | single-radius masked average | bounded | validity | one-scale analytic prior |
| R03 | multi-radius uniform | bounded | validity | bank without learned selection |
| R04 | learned radius logits | direct replacement | validity | test unsafe replacement |
| R05 | learned radius logits | unbounded residual | validity | isolate correction bound |
| R06 | learned radius logits | bounded | validity | minimal SE-MSAR |
| R07 | R06 | bounded | + distance $d_M$ | test sparse-distance cue |
| R08 | R07 | bounded | + ray | full router input |
| R09 | R08 at $1/4$ | one update | all | single-scale routing |
| R10 | R08 at $1/8,1/4$ | one per scale | all | main routing configuration |
| R11 | R08 at $1/8,1/4$ | two per scale | all | accuracy/latency reference |

Radius-bank candidates:

| ID | $\mathcal R_8$ | $\mathcal R_4$ |
|---|---|---|
| RB0 | $\{3,7,15\}$ | $\{1,3,7\}$ |
| RB1 | $\{3,7,15,31\}$ | $\{1,3,7,15\}$ |
| RB2 | $\{1,3,7,15,31\}$ | $\{1,3,7,15,31\}$ |

Bound/gate candidates:

| Factor | Values |
|---|---|
| $\tau_8$ | 0.15, 0.30, 0.45 |
| $\tau_4$ | 0.075, 0.15, 0.225 |
| gate | fixed 1, learned scalar, learned per-pixel, learned per-pixel + confidence |
| update count | one per scale, two per scale |

Select the smallest radius bank and fewest updates within 0.5% of the best RMSE and 1% of the best far-range RMSE.

### 6.2 SP-BRP search

Run with the selected router frozen.

| ID | Upsample/refinement | Edge gate | Levels | Conv | Purpose |
|---|---|:---:|---|---|---|
| B00 | bilinear |  | none | none | lower-cost reference |
| B01 | learned convex upsampling |  | full | standard | current-style learned upsampling reference |
| B02 | ordinary residual |  | full | DW+PW | unconstrained residual reference |
| B03 | ordinary residual | ✓ | full | DW+PW | isolate edge gate |
| B04 | high-pass residual |  | full | DW+PW | isolate high-pass projection |
| B05 | high-pass residual | ✓ | full | DW+PW | one-level SP-BRP |
| B06 | high-pass residual | ✓ | half | DW+PW | half-resolution only |
| B07 | high-pass residual | ✓ | half + full | standard | convolution reference |
| B08 | high-pass residual | ✓ | half + full | DW+PW | main SP-BRP |

| Hyperparameter | Values |
|---|---|
| full amplitude $a_1$ | 0.025, 0.05, 0.10, 0.20 |
| half amplitude $a_{1/2}$ | 0.05, 0.10, 0.20 |
| high-pass kernel $k$ | 3, 5, 7, 11 |
| residual width | 8, 16, 24 |

SP-BRP is accepted only if it improves both boundary-band error and depth-gradient error over B02, while global RMSE regresses by no more than 0.5% and absolute log-scale bias regresses by no more than 10% relative.

### 6.3 Backbone search

Keep inputs, routing, decoder, SP-BRP, and training losses identical. Width-match candidates or report the Pareto curve.

| ID | Encoder | Global blocks | Target |
|---|---|---|---|
| E00 | pure mobile CNN | none | local-only baseline |
| E01 | MobileViTv2-0.5 | $1/16$ only | runtime-first |
| E02 | MobileViTv2-0.5 | $1/8,1/16$ | GeoRT-SAR-S default candidate |
| E03 | MobileViTv2-0.75 | $1/8,1/16$ | accuracy-oriented candidate |
| E04 | width-matched ConvNeXt-style mobile blocks | declared stages | modern CNN comparison |

The backbone does not become a contribution because it wins this table; it is selected only as the best carrier for SE-MSAR and SP-BRP under the deployment budget.

### 6.4 Input representation search

| ID | RGB | Sparse | Mask | UV | Ray | Distance | Depth representation |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| I00 | ✓ |  |  |  |  |  | none |
| I01 |  | ✓ | ✓ |  |  | ✓ | median log |
| I02 | ✓ | ✓ |  |  |  |  | raw linear |
| I03 | ✓ | ✓ | ✓ |  |  |  | raw linear |
| I04 | ✓ | ✓ | ✓ | ✓ |  | ✓ | median log |
| I05 | ✓ | ✓ | ✓ |  | ✓ | ✓ | median log |
| I06 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | median log |

Additionally compare raw versus median-normalized sparse input and linear versus inverse versus log-depth output using the same architecture. Remove UV or ray when its three-seed gain is within noise and its removal improves latency or export simplicity.

### 6.5 Sparse-correction search

| ID | Mode | Clean anchors | 0.5% outliers | 1--3 px misalignment | Required report |
|---|---|---:|---:|---:|---|
| C0 | none | — | — | — | RMSE, anchor error |
| C1 | fixed soft, $\lambda=0.7$ | — | — | — | same |
| C2 | adaptive soft, $\lambda\in[0.5,0.9]$ | — | — | — | same + lambda histogram |
| C3 | hard replacement | — | — | — | same + boundary artifacts |

Adaptive correction wins only when it improves clean anchor error over no correction and degrades less than fixed/hard modes under sparse corruption.

---

## 7. Distillation and loss matrices

### 7.1 Metric-teacher reliability

Use the selected architecture with geometry losses disabled.

| ID | Supervision | DMD confidence factors |
|---|---|---|
| T00 | GT + sparse | none |
| T01 | T00 + DMD3C | uniform valid mask |
| T02 | T01 | sparse consistency |
| T03 | T02 | + edge risk |
| T04 | T03 | + range risk |
| T05 | T04 | + leave-one-out geometry agreement |
| T06 | T05 | + optional DMD-to-GT calibration |

T05 is the main theoretical candidate. T06 is reported separately because generation-time calibration changes the teacher and can use GT overlap.

### 7.2 Relative-geometry supervision

| ID | Geometry target | SSI | Ordinal | DSINE reliability | DMD in geometry |
|---|---|:---:|:---:|:---:|:---:|
| G00 | none |  |  |  |  |
| G01 | DA2 | ✓ |  |  |  |
| G02 | DA2 | ✓ | ✓ |  |  |
| G03 | DA2 | ✓ | ✓ | ✓ |  |
| G04 | DA2 + Metric3D uniform | ✓ | ✓ |  |  |
| G05 | DA2 + Metric3D conflict-aware | ✓ | ✓ |  |  |
| G06 | DA2 + Metric3D conflict-aware | ✓ | ✓ | ✓ |  |
| G07 | G06 diagnostic | ✓ | ✓ | ✓ | ✓, leave-one-out only |

G06 is the main candidate. G07 tests correlated knowledge but cannot be used to compute DMD agreement from a geometry map containing DMD itself.

### 7.3 Loss search

Start from Huber metric losses and add one group at a time.

| ID | SSI | Ordinal | Edge band | Gradient | Equivariance | 3D | Curriculum |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| L00 |  |  |  |  |  |  |  |
| L01 | ✓ |  |  |  |  |  |  |
| L02 | ✓ | ✓ |  |  |  |  |  |
| L03 | ✓ | ✓ | ✓ |  |  |  |  |
| L04 | ✓ | ✓ | ✓ | ✓ |  |  |  |
| L05 | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |
| L06 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| L07 | selected terms | selected | selected | selected | selected | selected | ✓ |

Also compare L1, Huber, and BerHu under L00 with identical normalization. Keep the optional 3D loss only when its three-seed accuracy gain exceeds its training-memory and stability cost; it adds no inference output.

### 7.4 Architecture-distillation interaction matrix

This $2\times2$ table is mandatory because it separates where the gain comes from.

| ID | Student | Supervision | Interpretation |
|---|---|---|---|
| X00 | clean baseline A00 | GT + sparse | lower bound |
| X01 | selected GeoRT-SAR | GT + sparse | architecture-only gain |
| X10 | clean baseline A00 | selected conflict-aware distillation | distillation-only gain |
| X11 | selected GeoRT-SAR | selected conflict-aware distillation | full method and interaction |

Report whether

$$
\Delta_{\mathrm{interaction}}=
(m_{X11}-m_{X10})-(m_{X01}-m_{X00})
$$

is positive or negative for each lower-is-better metric $m$; do not assume architectural and distillation gains are additive.

---

## 8. Metrics

### 8.1 Global and range accuracy

Report RMSE, MAE, iRMSE, iMAE, AbsRel, $\delta_1$, $\delta_2$, and $\delta_3$. Use fixed range bins $[0,20)$, $[20,40)$, $[40,60)$, $[60,80)$, and $[80,120]$ metres. Also report error versus distance to the nearest sparse point, using fixed bins declared before evaluation.

Sparse-anchor error is

$$
E_{\mathrm{anchor}}=
\frac1{|\Omega_M|}\sum_{p\in\Omega_M}|D(p)-S(p)|.
$$

Global log-scale bias is

$$
\mathrm{Bias}_{\log}=
\frac1{|\Omega_{\mathrm{gt}}|}
\sum_{p\in\Omega_{\mathrm{gt}}}
[\log D(p)-\log D_{\mathrm{gt}}(p)].
$$

### 8.2 Boundary accuracy

Derive the main boundary mask from valid GT depth discontinuities, not RGB edges alone. Dilate it to 3 px and 5 px bands. Report

- boundary-band RMSE and MAE at 3 px and 5 px;
- log-depth gradient MAE;
- foreground/background ordinal accuracy;
- thin-object error on a fixed semantic or geometry-derived mask;
- non-boundary RMSE to detect edge-only overfitting.

RGB-edge metrics from the current evaluator may be retained as diagnostics but do not replace GT-depth boundary metrics.

### 8.3 Scale and sparse-pattern robustness

| Family | Conditions |
|---|---|
| metric scaling | $\beta\in\{0.5,0.75,1.0,1.5,2.0\}$ with clipping-aware masks |
| LiDAR density | simulated 64, 32, 16, and 8 lines |
| random dropout | 25%, 50%, and 75% |
| outliers | 0.5%, 1%, and 3% |
| range noise | fixed sensor-noise model and seed |
| calibration | RGB-LiDAR shifts of 1, 2, and 3 pixels |
| appearance | frozen day/night or corruption split |
| cross-dataset | evaluate without retuning when data is available |

For every condition, report absolute metrics and degradation relative to the clean condition.

### 8.4 Confidence

Report error-confidence Spearman correlation, risk-coverage curve, AUSE, AURG, NLL, and RMSE by confidence quantile. A confidence head is selected by calibration plus overhead; it is not selected merely because its training loss decreases.

### 8.5 Efficiency

Report trainable parameters, MACs/FLOPs with the counting convention, median/P95 latency, FPS, peak memory, preprocessing share, and PyTorch/ONNX/TensorRT export status. SE-MSAR comparisons also report the number of routing updates; SPN references report their iteration count.

---

## 9. Run plan by data budget

### Phase 0: smoke and contract checks

Use 16--32 samples and no research conclusions. Run U0--U11, overfit four samples, inspect routing entropy/gates/residual maps, and verify that each ablation flag changes parameters or graph outputs as intended.

### Phase 1: screening with up to 1,000 images

With the current repository split, use all 800 training samples and the frozen 200-sample validation set. Use seed 42 and a fixed small step budget.

Priority runs:

1. C00 and A00--A04;
2. A05, A06, A07, A08, A09;
3. R03, R06, R08, R10, R11;
4. B02, B05, B06, B08;
5. E00, E02, E03;
6. T00, T01, T05;
7. G00, G02, G06.

This phase rejects broken, dominated, unstable, or clearly slow options. It does not produce final paper claims.

### Phase 2: 10,000-image confirmation

Use a frozen 10,000-image training subset, equal steps, and seeds 42/3407/9191. Carry forward

- A00;
- the best single-scale routing model;
- the best two-scale routing model;
- the best ordinary residual model;
- the best SP-BRP model;
- X00, X01, X10, and X11.

Report mean $\pm$ standard deviation and latency for every row.

### Phase 3: full training and one-time test evaluation

Retain exactly three configurations:

1. lightweight baseline;
2. selected GeoRT-SAR architecture with GT + sparse only;
3. full GeoRT-SAR with selected conflict-aware distillation.

Lock weights, hyperparameters, and checkpoint selection before test inference.

---

## 10. Results tables

### 10.1 Main architecture results

| ID | Params M | GFLOPs | RMSE mm ↓ | MAE mm ↓ | Far RMSE ↓ | Edge 3 px ↓ | Grad MAE ↓ | $|\mathrm{Bias}_{\log}|$ ↓ | Anchor MAE ↓ | P50 ms ↓ | P95 ms ↓ | Mem MB ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C00 | — | — | — | — | — | — | — | — | — | — | — | — |
| A00 | — | — | — | — | — | — | — | — | — | — | — | — |
| A01 | — | — | — | — | — | — | — | — | — | — | — | — |
| A02 | — | — | — | — | — | — | — | — | — | — | — | — |
| A03 | — | — | — | — | — | — | — | — | — | — | — | — |
| A04 | — | — | — | — | — | — | — | — | — | — | — | — |
| A05 | — | — | — | — | — | — | — | — | — | — | — | — |
| A06 | — | — | — | — | — | — | — | — | — | — | — | — |
| A07 | — | — | — | — | — | — | — | — | — | — | — | — |
| A08 | — | — | — | — | — | — | — | — | — | — | — | — |
| A09 | — | — | — | — | — | — | — | — | — | — | — | — |

### 10.2 Distillation results

| ID | RMSE ↓ | Far RMSE ↓ | Edge 3 px ↓ | Conflict-region RMSE ↓ | DMD coverage | Geometry coverage | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| T00 | — | — | — | — | — | — | — |
| T01 | — | — | — | — | — | — | — |
| T05 | — | — | — | — | — | — | — |
| G02 | — | — | — | — | — | — | — |
| G06 | — | — | — | — | — | — | — |
| X11 | — | — | — | — | — | — | — |

### 10.3 Robustness results

| Model | Clean RMSE | 16-line $\Delta$ | 8-line $\Delta$ | 75% dropout $\Delta$ | 1% outlier $\Delta$ | 2 px shift $\Delta$ | Scale $\times2$ equiv. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | — | — | — | — | — | — | — |
| SE-MSAR | — | — | — | — | — | — | — |
| Full GeoRT-SAR | — | — | — | — | — | — | — |

### 10.4 Confidence results

| Model | Spearman ↑ | AUSE ↓ | AURG ↑ | NLL ↓ | 50% coverage RMSE ↓ | P95 overhead ms ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Coarse upsampled confidence | — | — | — | — | — | — |
| Full reliability head | — | — | — | — | — | — |
| Full + calibration loss | — | — | — | — | — | — |

---

## 11. Rule for selecting the best architecture

### 11.1 Eligibility

Before seeing final results, declare

```text
T_RT      = maximum allowed batch-1 P95 latency
M_MAX     = maximum peak deployment memory
P_MAX     = 10M parameters for the S variant
R_MAX     = maximum tolerated export relative error
```

A configuration is eligible only if

- all contract tests pass;
- no NaN/Inf or silent missing-teacher batch occurs;
- parameter, P95 latency, memory, and export parity meet the declared limits;
- three-seed mean and standard deviation are available for a finalist;
- teacher coverage and evaluation masks match the frozen protocol.

### 11.2 Pareto filter

Remove any configuration dominated by another on all four axes:

$$
(\mathrm{RMSE},\ \mathrm{EdgeRMSE}_{3px},\ \mathrm{P95Latency},\ \mathrm{PeakMemory}).
$$

The surviving set is the accuracy-efficiency Pareto frontier.

### 11.3 Frozen ranking score

For each eligible Pareto model $c$, normalize lower-is-better metrics by the clean baseline A00:

$$
q_m(c)=\frac{m(c)}{m(\mathrm{A00})}.
$$

Rank with

$$
\begin{aligned}
J(c)={}&
0.30q_{\mathrm{RMSE}}
+0.10q_{\mathrm{MAE}}
+0.15q_{\mathrm{FarRMSE}}
+0.15q_{\mathrm{EdgeRMSE}_{3px}}\\
&+0.10q_{\mathrm{GradMAE}}
+0.05q_{\mathrm{AnchorMAE}}
+0.10q_{\mathrm{P95Latency}}
+0.05q_{\mathrm{PeakMemory}}.
\end{aligned}
$$

Lower $J$ is better. Confidence is selected in a second step using AUSE, AURG, NLL, and overhead so that a confidence head cannot mask a worse depth architecture.

### 11.4 Tie and stability rules

If two models differ by less than 0.5% in RMSE and 1% in boundary RMSE, choose the one with lower P95 latency; if still tied, choose fewer parameters; if still tied, choose the simpler graph with fewer routing updates and residual levels.

A module is not retained when its mean gain is smaller than the combined seed variation. Report all excluded finalist rows rather than only the winner.

### 11.5 Final architecture record

Fill this table after Phase 2 and before the test evaluation:

| Component | Selected value | Evidence row | Rejected alternative | Reason |
|---|---|---|---|---|
| Encoder | — | — | — | — |
| Input representation | — | — | — | — |
| $\mathcal R_8$ | — | — | — | — |
| $\mathcal R_4$ | — | — | — | — |
| Routing scales / updates | — | — | — | — |
| $\tau_8,\tau_4$ | — | — | — | — |
| SP-BRP levels | — | — | — | — |
| High-pass kernel | — | — | — | — |
| Residual width/amplitudes | — | — | — | — |
| Sparse correction | — | — | — | — |
| Confidence head | — | — | — | — |
| Metric reliability | — | — | — | — |
| Geometry teacher | — | — | — | — |
| Loss set | — | — | — | — |
| Params / P95 / memory | — | — | — | — |

The row is named **GeoRT-SAR-S** only after this record is complete and both SE-MSAR and SP-BRP are enabled. If either module fails its hypothesis, the final model is named by the modules it actually contains rather than preserving the intended name.
