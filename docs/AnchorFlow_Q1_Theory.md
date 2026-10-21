# Q1: code-matched mathematical specification

Current subset/development baseline: **Q1_s4_n1_fast,582,888 learned parameters,
coarse4NFE/fine0**. The [baseline record](AnchorFlow_V11_Baseline.md) contains the
reported metrics. The old G equations are not silently reused as a Q1 architecture
claim: Q1 inherits N1's trims/Fusion512/Jet-context128 and adds a deployment fastpath.

## 1. Three distinct objects

1. Neural model: N1 architecture with fresh30 Q1 weights. No packed/local branch,
   fine NODE, beta controller, new loss or extra uncertainty network.
2. Numerical model: FP32 chart/jet RHS, two explicit-midpoint steps, four calls
   to a shared state/time-dependent field.
3. Execution model: algebraic rewrites + TRT builderlevel5 + CUDA Graph; reference
   uses original operators/builderlevel3. Both already use CUDA Graph/auxstreams0.
   Measured speedup belongs to this deployment bundle, not isolated Graph,
   operator-only or builder-only effects.

Updating theory means making assumptions/attribution precise, not renaming an
implementation optimization as a new neural method. No source/config/weights were
changed during baseline promotion.

## 2. Architecture and IO

```text
RGB → MobileNetV4 features ──────┐
Sparse S / mask M / K ──────────┴→ gated multiscale decoder → D0 (1/4)
 → bounded inverse-depth jet → S4 JetODE → D4
 → phase lift + five-jet consensus + sparse innovation → D2, disagreement U
 → phase lift/detail → U-conditioned PIR → soft sensor fusion → D_full
```

Encoder: `mobilenetv4_conv_small_050.e3000_r224_in1k`, ImageNet initialization.
Actual internal bottlenecks, inherited from N1:

| Block | Feature channels | Hidden channels |
|---|---:|---:|
| D16 fusion refine |96|96|
| D8 fusion refine |64|64|
| D8 top-down refine |64|64|
| Context32 LiteBlock |64|32|
| **D4 RGB/sparse fusion refine** |48|**512**|
| **Jet context LiteBlock** |32|**128**|

D4 top-down refine remains hidden96. Hidden channels are feature expansion widths,
not additional depth maps. Q1 has **zero additional learned parameters versus N1**.
The frozen package retains fine-NODE source for constructor/RNG parity before
removing that module; `fine_node=None` at training/inference. Deleting that source
is not a parameter/runtime optimization and would invalidate the frozen contract.

Authoritative code: [constructor](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/experiment.py),
[jet field](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/vendor/geometry.py),
[U-PIR](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/vendor/phase_readout.py),
[fast operators](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/fast_ops.py),
[deployment](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/deployment.py),
[qualification/timing](../drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/benchmark_fast.py).

Inference inputs: RGB FP32[B,3,352,1216] in[0,1]; S/M FP32[B,1,352,1216] in metres
and binary support; K FP32[B,3,3] adjusted for crop/resize. RGB normalization is
inside the model. Static deployment B1. Invalid/nonfinite sparse inputs are removed.
Output D_full FP32[B,1,352,1216], clamped to[0.1,120]m. GT/teachers are never
inference inputs; sensor depth is not validation GT.

## 3. Local jet, chart and field

At a quarter-grid cell:

\[
j=(v,g_x,g_y,h_{xx},h_{xy},h_{yy}),\quad
q_j(\delta)=v+g_x\delta_x+g_y\delta_y+
\tfrac12h_{xx}\delta_x^2+h_{xy}\delta_x\delta_y+\tfrac12h_{yy}\delta_y^2.
\]

Offsets use quarter-grid cells. This is a local polynomial representation, not
proof that evolved coefficients are globally integrable derivatives of one surface.
The seed uses inverse D0 and minmod differences; derivative seeding detaches from
initial depth, as implemented.

The smooth chart for finite coordinates z is

\[
v=\exp[\log(1/120)+\log(1200)\sigma(z_0)],\quad
j_{1:5}=v\,(.5,.5,.25,.25,.25)\odot\tanh(z_{1:5}).
\]

Center inverse depth and derivative coefficients are bounded. q_j is not guaranteed
positive at arbitrary offsets; readout explicitly clamps proposals to[1/120,10].
The entire model is not globally smooth merely because its chart is smooth.

Let Aθ be learned reaction, Bθ sparse innovation forcing, and Tθ local geometric
transport, weighted by learned conductance/depth disagreement/barrier:

\[
\dot z=2\tanh\!\left[\frac{A_\theta+B_\theta+T_\theta}{2v\kappa}\right],
\quad\kappa=(1,.5,.5,.25,.25,.25),
\]
\[
T_{\theta,i}=\sum_{d\in\{E,W,S,N\}}w_{i,d}
[\mathcal T_{\delta_d}j_{i+d}-j_i].
\]

Context is computed once; the RHS still depends on current state and solver time.
The ODE is defined **directly in chart coordinates**, not an exact Jacobian
pullback of a physical-jet PDE. Bounded RHS does not prove stability/accuracy of
every discrete step, mass conservation or a physical law. No global calibration,
integrability, optimality or safe-improvement guarantee is asserted.

## 4. S4 is four field evaluations, not RK4

Two explicit midpoint RK2 steps with h=1/2,T=1:

\[
k_1=f_\theta(t_n,z_n),\quad k_2=f_\theta(t_n+h/2,z_n+hk_1/2),\quad
z_{n+1}=z_n+hk_2,\quad n=0,1.
\]

Calls at0,0.25,0.5,0.75 → **4NFE**, same field parameters. The horizon is latent
refinement time, not elapsed scene time. No adaptive stop, learned step controller
or fine solver. PyTorch trains; TensorRT is an inference compiler/runtime.

## 5. Consensus disagreement and U-PIR

Center/E/W/S/N jets give inverse-depth proposals at four subpixel phases. A learned
softmax includes geometric disagreement, barrier and legal-neighbour penalties:

\[
\bar q=\sum_a\alpha_aq_a,\quad
U=\frac{\sum_a\alpha_a(q_a-\bar q)^2}{\max(\bar q^2,10^{-6})},\quad
c(U)=\frac{\max(U,0)}{\max(U,0)+0.002}.
\]

U is **relative proposal disagreement**, not calibrated error probability,
aleatoric uncertainty or a formal confidence bound. It is already computed for
consensus, not predicted by another uncertainty network. PixelShuffle maps four
phase values to one half-grid channel.

PIR inputs:24 hidden features +16 same-phase innovation/mass/contrast features
+1 bounded U cue. Neural core:1×1 projection41→16,SiLU,depthwise3×3,SiLU,head16→8:

\[
(a,b)=\operatorname{Head}(\operatorname{DWBlock}(W_xx+w_uc(U)+b_x)),
\quad
\Delta D=(.5+.05D)\odot\operatorname{PixelShuffle}_2[\tanh(a)\odot\sigma(b)].
\]

Add the bounded residual to detail depth, clamp, then softly fuse sensor input.
U can change sign/amplitude and gates; Q1 has no explicit JPIR beta. U-PIR
historically added16 weights over PIR, **not another16 in Q1 versus N1**.
No Q1-specific no-U retraining/intervention is available; G's intervention is
historical evidence, not transferred as proof for Q1.

## 6. Fastpath equivalence and its limits

### Cardinal transport

For a quadratic jet, changing origin gives

\[
\mathcal T_\delta j=(q_j(\delta),
g_x+h_{xx}\delta_x+h_{xy}\delta_y,
g_y+h_{xy}\delta_x+h_{yy}\delta_y,h_{xx},h_{xy},h_{yy}).
\]

Shift the neighbouring jet to the target cell:

| Neighbour / offset | Value | x derivative | y derivative |
|---|---|---|---|
| E /(-1,0) |v−gx+½hxx|gx−hxx|gy−hxy|
| W /(+1,0) |v+gx+½hxx|gx+hxx|gy+hxy|
| S /(0,-1) |v−gy+½hyy|gx−hxy|gy−hyy|
| N /(0,+1) |v+gy+½hyy|gx+hxy|gy+hyy|

Hessian components are unchanged. Substitution proves equivalence for finite
states; the local map's analytic Jacobian is unchanged. Known-zero products are
removed, not neighbours/state feedback/NFE. This is specialized evaluation of
the existing polynomial, not a newly learned geometric transport.

### Fixed stencils and moments

One-hot3×3 phase-upsample Conv selects nine neighbours. Replicate padding plus
nine row-major slices is the same linear selection. Only the fixed stencil
changes; learned Conv weights do not. Mass/numerator/squared-numerator moment
maps are concatenated and pooled with the same2D kernel/padding. Channels are
independent: no new spatial/phase mixing is introduced.

Separable7×7 pooling was rejected: FP32 variance cancellation amplified differences
at low support. It is NOT in the fastpath. Identities hold in real arithmetic;
backend rounding can differ. No blanket bitwise identity or automatic speedup.
Deployment uses an eval copy; training/checkpoint/BN state remain unchanged.

### Compilation and replay

Strongly-typed FP32/TF32-OFF ONNX22; original builderlevel3 versus fastlevel5,
auxstreams0 in both. More tactic search is not a speed guarantee
([NVIDIA builder documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/builder-performance.html)).
Both use a dedicated stream/stable buffers, warm enqueue and CUDA Graph capture;
new inputs are copied D2D before replay. Graph reduces submission overhead, not
NFE or parameters ([NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)).

## 7. Training and evaluation contracts

1600train/400val from2000images,seed42,B4,ImageNet encoder,30epochs/no early stop.
Eager BF16CNN with FP32geometry/loss; AdamW fused,LR3e−4,encoderLR×0.5,warmup1+
cosine,minLRratio0.05. Frozen code/config is authoritative, not prose alone.

Loss: multiscale GT Huber +0.6globalRMSE +0.1range-balancedRMSE +0.02observed
sensor +0.05holdout +0.02trust +metricKD +0.05boundary +0.01barrier +tail +0.1
dynamics auxiliary +inverseRMSE. KD coefficient0.012→0.006,tail coefficient0.25
ramps2epochs,inverseRMSE coefficient0.04 ramps4epochs. **Relative OFF/weight0**;
teacher train-only, never fused into GT/inference. Selection first prefers native
iRMSE≤3.2, then lower RMSE; final acceptance uses strict iRMSE<3.2/RMSE<0.98.

Global accumulation over valid pixels, not average per-image RMSE:

\[
\mathrm{RMSE}=\sqrt{\sum e_p^2/N},\quad\mathrm{MAE}=\sum|e_p|/N,\quad
\mathrm{iRMSE}=1000\sqrt{\sum(1/\widehat D_p-1/D_p)^2/N}.
\]

Boundary3: valid adjacent GT jump>max(1m,5% of nearer depth), then Chebyshev-radius3
band intersected with valid GT. It is not dense occlusion ground truth.

## 8. Attribution, evidence and future validation

Q1 same weights, original→fast:4.229391→3.846668ms (about9.05% lower wall), RMSE
drift≈9.41e−9m. This supports deployment efficiency, not higher accuracy.
Q1 versus oldN1 fast improves RMSE≈0.0951%, but changes trained weights; no
isolated architecture/loss gain, multi-seed significance or determinism claim.

Actual400/25,424,992 valid pixels,B1×352×1216. A/B/A freshness; mean pixel≤1mm,
max≤6cm versus original PyTorchFP32; nativeBF16 metric drift RMSE≤0.002m,
iRMSE≤0.02km⁻¹. Primary100warmup/300iterations/100 changing GPU images; wall
includesD2D, excludes decode/preprocess/H2D/build/capture/metrics. FPS=1000/mean.
Reverse-order repeat is separate, never best-of. Torch memory is not fullTRT
footprint; both controls already use CUDA Graph.

Promotion is user-authorized. The [raw console and evidence level](../benchmarks/q1_n1fast_2026_10_21/README.md)
are retained; actual checkpoint/ONNX/engine SHA and selected epoch were not supplied
locally. They are not guessed. CPU/ONNX checks are independent source tests, not
a rerun of the A100 dataset/metrics.

Keep Drive's `S4_UPIR_N1Fast_Fresh30_pair01` checkpoint/runtime_config/frozen_bundle/
data_contract/full report/status/log and original/fast export hashes. Read selected
epoch from the real checkpoint/report, not N1's old epoch. Packaging verifies bytes;
reverification writes outside the frozen bundle. New source requires a new run.

FullKITTI needs a separately locked split and preferably another seed. Current400
frames have repeatedly guided development; do not include them in training and
still report them as independent validation. Official pretrained MSG/fullKITTI
is not a matched supervision control. Q1 Orin is unmeasured; renaming a variant
does not make the legacy G exporter Q1-compatible.
