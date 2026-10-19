# Why test decoder capacity, not simply add parameters?

Evidence: completed G30 CSV in `evidence/G_train_log.csv`, exact historical
comparison snapshot and the corrected U-removal table supplied by the user.
No checkpoint/GT/prediction tensor was supplied locally; no claimed new metric.

## 1. What the logs do and do not show

G native best epoch27: RMSE0.990139440m / iRMSE3.171911570km⁻¹.
TRT FP32 graph:0.990150559m /3.172073529km⁻¹ /MAE0.258984267m;
boundary3 RMSE1.685667680m; median3.826771ms. Epoch29 RMSE0.990432:
the curve has flattened, but two later epochs are not proof of strong overfitting.
Only one seed and repeatedly used400val; no independent test accuracy.

At epoch27, weighted training loss fractions are approximately:

| Term | Fraction of summed training loss |
|---|---:|
| Direct RMSE |36.7%|
| Multi-scale metric |21.1%|
| Range-balanced metric |15.4%|
| Inverse RMSE |8.4%|
| Boundary |6.2%|
| Tail |5.9%|
| Metric teacher KD |1.0%|
| Relative |**0% — disabled**|

Fractions are NOT gradient attribution or a measure of a loss's importance.
Raw metric-KD≈2.116 is not dominant after weighting≈0.01376. Relative zero is
the intentional treatment, not a missing teacher failure. Zero query diagnostics
in training CSV can come from disabled telemetry; not proof the U path is unused.

Train losses D16≈0.668, D8≈0.504, D4≈0.235, D2≈0.150 differ in pooled GT,
weights and tasks: **cannot infer the largest one is the causal bottleneck**.
We add native-stage validation traces to test the hypothesis directly.

## 2. Most RMSE energy is in rare failures

Error>10m:45,232pixels, only0.1779% support, **57.12% SSE**.
Error>20m:9,155pixels,29.86% SSE. Increasing only final smoothing/edge detail
may not fix wrong surface/far-range assignments. Stronger low-resolution RGB/sparse
fusion could improve contextual seed/scale before jets; that is a **hypothesis**,
not a diagnosis supported by per-pixel examples/profiler evidence.

For reference, reducing RMSE from0.990151 to0.98 needs about2.04% less total SSE;
if all reduction came from the >10m tail, about3.57% less tail SSE would suffice.
This is an algebraic budget, not an achievable performance forecast.

Corrected same-checkpoint intervention:
U present0.990151/3.172073/0.258984 versus U-zero0.990889/3.191544/0.261689.
U input helps this trained checkpoint, mostly inverse/MAE. Mean output change
0.03668m is prediction difference, NOT error reduction. Preserve U-PIR in all
three arms; do not confound another loss/cue change with decoder capacity.

## 3. Parameter count is not the compute budget

Current583,880params: encoder340,992 (58.4%); decoder149,748 (25.6%);
F32 context52,144; dynamics10,891; detail/U-PIR6,113. A larger encoder changes
pretraining/representation and a larger field is paid4times. Fine NODE has
already shown unfavorable runtime trade-off. Start with once-per-frame decoder.

Existing block C→M→M(depthwise3×3)→C has approximately
\(M(2C+9)\) convolution weights, but
\(HW\,M(2C+9)\) MACs. Thus identical params cost16× more at D4 than D16.

| Arm | Hidden D16 / D8 / D4 | Params | Added conv MACs |
|---|---|---:|---:|
| G0 |192/128/96|583,880|0|
| H |384/256/96|641,288 (+9.83%)|181.81M|
| I |384/256/192|651,752 (+11.62%)|451.47M|

H→I adds only10,464params but269.66M MACs, **more** than all G0→H extra
compute. H is the primary edge candidate; I explicitly tests whether higher-res
capacity earns its cost. Same layer count and external feature widths48/64/96;
no new field evaluation, attention, boundary network or full-res feature CNN.
MAC counts exclude BN/activation/data movement and do not predict TRT latency.

## 4. Initialization, fairness and failure conditions

Construct full G first (including removed historical fine constructor RNG draws).
Copy every old parameter slice/BN statistic into the wider existing blocks.
Additional hidden lanes receive zero outgoing projection, so initial function is
preserved in exact arithmetic. New lanes have learnable gradients after updates;
they are not a fixed zero ablation. Common hashes crop to retained old slices,
not a false claim that different-shaped models have identical full state.

Same30epochs, seed42, data/augmentation, ImageNet weights, optimizer, objectives,
teacher treatment and best_policy rule; only decoder hidden width differs.
Run fresh G0 too. An old-session runtime/metric row alone is not the control.
No early stop; epoch-boundary resume remains strict. No loss retuning to save a
losing architecture. No TensorRT training or unsupported train-speed claim.

Predeclare accuracy not worse than G0 and latency≤1.05×G0. Report MAE, boundaries,
near/far, >10m tail, params, median/P95 and failed gates even if a variant loses.
If H/I do not earn their cost, retain G; more parameters are not intrinsically
an upgrade. No new baseline is promoted before measured results.

Function-preserving widening is established prior art:
[Net2Net](https://arxiv.org/abs/1511.05641). Here zero-outgoing extra lanes are a
simple implementation choice, not a novel method or a pretrained widening run.
Actual launch/memory/fusion behavior must be measured, per
[NVIDIA TensorRT performance guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html).
