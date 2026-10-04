> Historical research proposal. Implementation hiện tại: [AnchorFlow Research](../drive_upload/AnchorFlow_Research/ARCHITECTURE.md). Không dùng tài liệu đề xuất này như contract train/deploy của v3/v4.

from pathlib import Path

md = r"""# AnchorFlow-DC
## ShapeFlow-Inspired Ray-Constrained Continuous Deformation for Neural Depth Completion

**Status:** Research architecture proposal  
**Primary target:** KITTI Depth Completion, with extension to robust / cross-domain depth completion  
**Core inputs at inference:** dense relative depth prior + sparse metric LiDAR  
**Training-only input:** metric ground-truth depth where available  
**Optional upstream input:** RGB is used only by the frozen monocular-depth model that generates the dense relative prior; the core completion network does not require RGB features.

---

## 0. Executive summary

This proposal reformulates depth completion as **continuous deformation of an already-dense relative-depth surface into a dense metric-depth surface**, rather than as direct sparse-to-dense regression.

The central ShapeFlow-inspired idea is:

\[
\text{source geometry}
\;\xrightarrow{\text{learned continuous flow}}\;
\text{target geometry}.
\]

For depth completion:

\[
D_{\mathrm{rel}}
\;\rightarrow\;
D_0
\;\xrightarrow[\text{sparse metric anchors}]
{\text{ray-constrained neural flow}}\;
\hat D_{\mathrm{metric}}.
\]

The dense relative-depth prediction acts as the **source shape**. Sparse LiDAR samples act as **metric anchors**. The unknown dense metric depth is the target shape.

The key design is to **not move image points freely in XYZ**, because that causes occlusion, re-projection and topology problems. Instead, every image pixel remains on its original camera ray and only its depth changes. The learned dynamics therefore deform a 2.5D surface while preserving pixel correspondence:

\[
P_t(u)=D_t(u)\,K^{-1}\tilde u.
\]

The model evolves a **log-depth correction field**

\[
c_t(u)=\log D_t(u)-\log D_0(u),
\]

so that

\[
D_t(u)=D_0(u)\exp(c_t(u)).
\]

The proposed flow is a **neural reaction-diffusion field**:

\[
\frac{\partial c_t}{\partial t}
=
R_\theta(c_t,F,t)
+
\lambda_t
\nabla\cdot
\left(
G_\theta(F,t)\odot\nabla c_t
\right),
\]

where:

- \(R_\theta\) learns non-local/local corrections that cannot be explained by smooth propagation alone;
- \(G_\theta\) is an anisotropic conductance field that determines where metric corrections may propagate;
- sparse LiDAR observations are repeatedly enforced through a confidence-aware Dirichlet projection;
- the same vector-field network is reused across a small number of integration steps.

The recommended model name in this proposal is **AnchorFlow-DC**.

The strongest research hypothesis is not simply “ODEs help depth completion.” It is:

> **A dense monocular geometry prior can be metricized more accurately by learning an anchor-constrained, ray-preserving deformation field than by either one-shot calibration, generic residual regression, or unconstrained spatial propagation.**

This hypothesis must be falsified against strong direct-residual and recurrent baselines with matched parameter counts.

---

# 1. Motivation

## 1.1 The depth-completion problem

Let:

\[
D_s \in \mathbb{R}^{H\times W}
\]

be a sparse LiDAR depth map and

\[
M_s \in \{0,1\}^{H\times W}
\]

its valid mask.

A monocular foundation model provides a dense relative-depth map

\[
R \in \mathbb{R}^{H\times W}.
\]

The target is dense metric depth

\[
D^* \in \mathbb{R}^{H\times W},
\]

although real datasets such as KITTI provide metric supervision only on a validity mask \(M_{gt}\).

The conventional formulation is

\[
\hat D = f_\theta(I,D_s,M_s),
\]

or

\[
\hat D = f_\theta(R,D_s,M_s).
\]

This proposal instead decomposes the task into:

\[
R
\rightarrow
D_0
\rightarrow
D_t
\rightarrow
D_1=\hat D.
\]

Here \(D_0\) is a coarse metricized version of the dense relative prior, while the learned flow handles local metric distortion, object-level correction, edge preservation and sparse-anchor propagation.

---

## 1.2 Why a dense relative prior changes the problem

A modern monocular-depth foundation model already provides much of the information that old depth-completion networks had to learn from RGB:

- foreground/background ordering;
- object boundaries;
- road and surface geometry;
- fine structural details;
- dense scene coverage;
- long-range contextual relationships.

What is still missing is reliable **metric scale and local metric consistency**.

This suggests a different decomposition:

\[
\boxed{
\text{Depth completion}
\approx
\text{dense geometry prior}
+
\text{sparse metric correction}.
}
\]

Instead of asking a network to reconstruct the entire scene from sparse measurements, we ask it to learn:

\[
\boxed{
\text{where and how much is the dense prior metrically wrong?}
}
\]

That is a significantly better-conditioned problem.

---

# 2. What is borrowed from ShapeFlow

ShapeFlow (NeurIPS 2020) learns a continuous deformation field that advects a source 3D geometry toward a target geometry. Its conceptual contribution is to model a class of shapes as a **deformation space**, rather than directly decoding a latent code into a new shape.

The useful analogy is:

| ShapeFlow | AnchorFlow-DC |
|---|---|
| source 3D shape | dense relative-depth surface |
| target 3D shape | dense metric-depth surface |
| latent deformation field | anchor-conditioned depth-correction field |
| continuous ODE flow | fixed-ray continuous correction dynamics |
| preserve source geometric detail | preserve monocular-prior geometry where metric evidence does not contradict it |

The important adaptation is that depth completion is a **2.5D camera-ray problem**, not free-form 3D shape deformation.

A full 3D deformation

\[
\frac{dP_t}{dt}=v_\theta(P_t)
\]

would allow points to move tangentially to image rays, producing:

- changing pixel correspondences;
- occlusions;
- many-to-one projections;
- holes after projection;
- unnecessary geometric freedom.

Instead, AnchorFlow-DC enforces:

\[
P_t(u)=D_t(u)r_u,\qquad r_u=K^{-1}\tilde u.
\]

Hence

\[
\frac{dP_t(u)}{dt}
=
r_u \frac{dD_t(u)}{dt}.
\]

Every point is permitted to move **only along its camera ray**.

This is the depth-completion equivalent of restricting ShapeFlow to a geometry-preserving deformation manifold.

---

# 3. Literature boundary and novelty risks

The proposal must be positioned carefully because several recent methods occupy adjacent territory.

## 3.1 ShapeFlow — NeurIPS 2020

**ShapeFlow: Learnable Deformations Among 3D Shapes**

Key idea: learned continuous flow fields deform source geometries toward target geometries.

Relevance:
- direct conceptual inspiration;
- demonstrates that deformation can be a better representation than generation;
- not designed for outdoor metric depth completion;
- does not formulate dense relative depth + sparse metric anchors as a ray-constrained image-space flow.

Paper: https://arxiv.org/abs/2006.07982

---

## 3.2 OGNI-DC — ECCV 2024

**OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations**

It iteratively refines a **depth-gradient field**, followed by a differentiable depth integrator.

Relevance:
- strong evidence that iterative geometric-field refinement is useful;
- closest older precedent for "learn a field, then integrate it";
- however its state is a depth-gradient field, not a ShapeFlow-style deformation of a dense monocular source geometry under sparse metric anchors.

Paper: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/319_ECCV_2024_paper.php

---

## 3.3 BP-Net — CVPR 2024

**Bilateral Propagation Network for Depth Completion**

BP-Net propagates sparse depth early using a learned nonlinear bilateral model and demonstrates the importance of propagation before later fusion/refinement.

Relevance:
- strong propagation baseline;
- AnchorFlow-DC must beat a matched propagation baseline to justify the flow formulation.

Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Bilateral_Propagation_Network_for_Depth_Completion_CVPR_2024_paper.html

---

## 3.4 DMD3C — CVPR 2025

**Distilling Monocular Foundation Model for Fine-grained Depth Completion**

DMD3C distills monocular foundation-model geometry into a depth-completion network and uses synthetic/data-distillation pretraining followed by real metric fine-tuning.

Relevance:
- demonstrates the value of monocular foundation priors;
- shows that synthetic/full geometry supervision can compensate for sparse real ground truth;
- currently a key accuracy reference on KITTI.

Paper: https://arxiv.org/abs/2503.16970

As of 2026-10-01, KITTI's official depth-completion leaderboard lists DMD3C++ at approximately:

- RMSE: 676.38 mm
- MAE: 193.86 mm
- iRMSE: 1.79
- iMAE: 0.83

Leaderboard: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion

These numbers are only a reference target; fair comparison must match additional-data settings.

---

## 3.5 DepthFM — AAAI 2025 Oral

**DepthFM: Fast Generative Monocular Depth Estimation with Flow Matching**

DepthFM transports image latent representations toward depth latent representations using flow matching.

Relevance:
- prevents a claim such as "first use of continuous flow for depth";
- its flow acts in latent generative space, not as explicit metric deformation of an existing dense depth surface;
- AnchorFlow-DC should be positioned as **geometry-space correction**, not generic flow matching.

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/32330

---

## 3.6 Marigold-DC — ICCV 2025

**Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion**

It views depth completion as dense image-conditioned generation guided by sparse depth observations.

Relevance:
- supports the view that dense monocular priors should be central;
- diffusion is comparatively expensive;
- AnchorFlow-DC aims for a deterministic, few-step alternative.

Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Viola_Marigold-DC_Zero-Shot_Monocular_Depth_Completion_with_Guided_Diffusion_ICCV_2025_paper.html

---

## 3.7 OMNI-DC — ICCV 2025

**OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration**

It emphasizes robustness across varying sparse-depth patterns and zero-shot environments.

Relevance:
- robust sparsity generalization is an important evaluation target;
- AnchorFlow-DC should not be judged only on standard 64-line KITTI input.

Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_OMNI-DC_Highly_Robust_Depth_Completion_with_Multiresolution_Depth_Integration_ICCV_2025_paper.html

---

## 3.8 Midas Touch for Metric Depth — CVPR 2026

**The Midas Touch for Metric Depth**

It explicitly converts relative depth into metric depth from extremely sparse 3D observations through segment-wise sparse graph optimization and discontinuity-aware geodesic refinement.

Relevance:
- directly overlaps the "relative depth + sparse metric anchors" problem decomposition;
- AnchorFlow-DC cannot claim novelty merely for relative-to-metric conversion;
- the distinction must be the **learned ray-constrained continuous deformation dynamics** and learned anchor propagation.

Paper: https://arxiv.org/abs/2605.11578

---

## 3.9 Any2Full — 2026

**Any to Full: Prompting Depth Anything for Depth Completion in One Stage**

It injects sparse scale cues into a pretrained monocular-depth model through a scale-aware prompt encoder.

Relevance:
- shows that one-stage scale prompting can avoid explicit prior-alignment artifacts;
- AnchorFlow-DC must justify why a post-prior learned deformation is preferable or complementary.

Paper: https://arxiv.org/abs/2603.05711

---

## 3.10 LDCM — ICLR 2026

**Large Depth Completion Model from Sparse Observations**

LDCM combines a monocular geometry prior with multiscale Poisson completion and a transformer-based point-map regressor.

Relevance:
- extremely close in high-level decomposition;
- importantly, it already performs metric alignment of a monocular prior using sparse measurements;
- AnchorFlow-DC must not claim the first "dense prior + sparse metric initialization" approach;
- its possible novelty lies in replacing static Poisson initialization + large refinement with a learned anchor-constrained deformation field.

Paper: https://arxiv.org/abs/2605.30115

---

## 3.11 Adaptive Response Geometry — arXiv 2026-09-28

**Boosting Metric Depth Completion via Training-Free Adaptive Response Geometry**

This very recent preprint argues that the best calibration response may vary between depth, log-depth and disparity-like coordinates, then uses training-free residual reconstruction with hard Dirichlet constraints.

Relevance:
- a major novelty threat for any claim based purely on "relative prior + response calibration + hard sparse anchors";
- AnchorFlow-DC should treat response coordinates as an ablation / design choice rather than its primary novelty;
- the learned continuous neural dynamics remain a meaningful distinction.

Paper: https://arxiv.org/abs/2609.36168

---

# 4. Proposed method: AnchorFlow-DC

## 4.1 High-level pipeline

```text
RGB image
   │
   ▼
Frozen / precomputed monocular depth foundation model
   │
   ▼
Dense relative depth R
   │
   ├────────────────────────────────────────┐
   │                                        │
Sparse metric LiDAR S, mask M               │
   │                                        │
   └──────────────┬─────────────────────────┘
                  ▼
       Robust coarse metric bootstrap
                  │
                  ▼
          Dense metric-ish D0
                  │
        ┌─────────┴──────────┐
        │                    │
 relative-geometry       sparse-anchor
     encoder               encoder
        │                    │
        └─────────┬──────────┘
                  ▼
       Anchor-conditioned features F
                  │
                  ▼
         c0 = 0 correction field
                  │
                  ▼
      Shared neural deformation field
                  │
          dc/dt = Vθ(c,F,t)
                  │
        fixed 2-4 solver steps
                  │
                  ▼
        confidence-aware anchor
            projection each step
                  │
                  ▼
                 c1
                  │
                  ▼
        D_hat = D0 * exp(c1)
                  │
                  ▼
         dense metric depth
```

The ground-truth depth \(D^*\) is **never an inference input**. It is used only to supervise training.

---

# 5. Stage A — dense relative-depth prior

Use a strong frozen monocular-depth model to compute:

\[
R = f_{\mathrm{MDE}}(I).
\]

Possible upstream priors include Depth Anything V2, MoGe or another dense monocular model.

For a controlled research experiment, **precompute \(R\)** and freeze the prior model. This has three benefits:

1. isolates whether AnchorFlow-DC itself improves completion;
2. makes training substantially cheaper;
3. permits an honest parameter/runtime analysis of the completion head separately from the upstream prior.

A later end-to-end experiment can optionally fine-tune only lightweight adapters in the prior model.

---

# 6. Stage B — robust metric bootstrap

A pure relative map may have arbitrary global scale/shift or response distortion.

The flow should not waste most of its capacity learning trivial global calibration. Therefore compute a coarse metric initialization \(D_0\) from sparse pairs:

\[
\left\{
R(u_i),D_s(u_i)
\right\}_{i=1}^{N_s}.
\]

The simplest recommended bootstrap is a robust two-parameter fit in a chosen response coordinate \(\phi\):

\[
a^*,b^*
=
\arg\min_{a,b}
\sum_{u_i\in\Omega_s}
\rho
\left(
a\phi(R(u_i))+b-\phi(D_s(u_i))
\right),
\]

followed by:

\[
D_0(u)
=
\phi^{-1}
\left(
a^*\phi(R(u))+b^*
\right).
\]

Use Huber/Tukey weighting or IRLS so isolated LiDAR outliers do not dominate calibration.

### Response-coordinate recommendation

Do **not** assume one response coordinate is universally optimal.

Ablate:

\[
\phi(d)=d,
\]

\[
\phi(d)=\log(d+\epsilon),
\]

and

\[
\phi(d)=\frac{1}{d+\epsilon}.
\]

Because the 2026 Adaptive Response Geometry paper specifically shows that response choice matters, AnchorFlow-DC should report this ablation explicitly.

Recommended first implementation: **log-depth** because it:

- guarantees positive reconstructed depth;
- makes multiplicative corrections additive;
- reduces dynamic-range imbalance between near and far KITTI regions.

The bootstrap is intentionally simple. The neural flow, not the bootstrap, should carry the central contribution.

---

# 7. Stage C — formulate completion as a correction flow

Define:

\[
z_0(u)=\log(D_0(u)+\epsilon),
\]

\[
z_t(u)=z_0(u)+c_t(u).
\]

Therefore:

\[
D_t(u)=\exp(z_t(u)).
\]

Initialization:

\[
c_0(u)=0.
\]

Final output:

\[
\hat D(u)
=
D_0(u)\exp(c_1(u)).
\]

The network therefore learns a dense metric **correction surface**, not a depth map from scratch.

---

# 8. Sparse LiDAR as metric boundary conditions

At every valid LiDAR pixel:

\[
u\in\Omega_s,
\]

the correction implied by the sensor is:

\[
c_s(u)
=
\log(D_s(u)+\epsilon)
-
\log(D_0(u)+\epsilon).
\]

This creates a sparse correction map:

\[
C_s=M_s\odot c_s.
\]

The central interpretation is:

\[
\boxed{
\text{sparse LiDAR does not supply the full surface;
it supplies metric boundary conditions for the correction flow.}
}
\]

---

# 9. Confidence-aware Dirichlet anchor projection

A pure hard constraint is attractive:

\[
c_t(u)=c_s(u),
\qquad
u\in\Omega_s.
\]

But real LiDAR may contain:

- projection/calibration noise;
- moving-object misalignment;
- mixed pixels;
- reflective-surface errors;
- outliers.

Therefore use a predicted anchor confidence:

\[
q(u)\in[0,1].
\]

After every solver step:

\[
\tilde c_{t+\Delta t}
=
\mathrm{SolverStep}(c_t,V_\theta),
\]

project:

\[
c_{t+\Delta t}(u)
=
\begin{cases}
q(u)c_s(u)
+
(1-q(u))\tilde c_{t+\Delta t}(u),
&
M_s(u)=1,\\
\tilde c_{t+\Delta t}(u),
&
M_s(u)=0.
\end{cases}
\]

For clean benchmark inputs, initialize or regularize \(q\) close to 1.

During synthetic training, inject controlled LiDAR outliers and explicitly supervise \(q\) when corruption labels are known.

Ablate:

- hard anchor \(q=1\);
- no projection;
- learned confidence.

---

# 10. Geometry features

The core model should be able to operate without RGB because the relative-depth prior already contains substantial visual geometry.

Construct prior features from:

\[
R,\quad
z_0,\quad
\nabla z_0,\quad
\nabla^2 z_0.
\]

If camera intrinsics are available, optionally compute a coarse 3D surface:

\[
P_0(u)
=
D_0(u)K^{-1}\tilde u,
\]

and local normals:

\[
N_0(u)
=
\mathrm{Normal}(P_0).
\]

Recommended geometry input tensor:

\[
X_g=
[
\tilde R,\;
z_0,\;
\partial_x z_0,\;
\partial_y z_0,\;
|\nabla z_0|,\;
N_0
].
\]

Normals should be optional because valid intrinsics may not always be available in general-domain settings.

---

# 11. Anchor features

Sparse-anchor information should not be represented only by zero-filled depth.

Use:

\[
X_a=
[
M_s,\;
C_s,\;
D_s,\;
\Delta_s,\;
Q_s
],
\]

where:

- \(M_s\): binary mask;
- \(C_s\): sparse log-depth correction;
- \(D_s\): sparse metric depth;
- \(\Delta_s\): normalized distance-to-nearest-anchor map;
- \(Q_s\): optional sensor-confidence / validity channel.

The distance map helps distinguish:

- a region with no nearby metric evidence;
- a region immediately surrounding LiDAR anchors.

A masked-convolution or sparse-convolution stem is preferable to naïvely applying ordinary convolution to zero-filled sparse depth.

---

# 12. Dual encoder

Use two lightweight encoders.

## 12.1 Relative-geometry encoder

Input: \(X_g\)

Purpose:
- detect prior boundaries;
- encode large-scale scene layout;
- identify smooth surfaces;
- detect likely structural discontinuities.

## 12.2 Sparse-anchor encoder

Input: \(X_a\)

Purpose:
- encode correction signs/magnitudes;
- model anchor density;
- detect contradictory/noisy anchors;
- create multi-scale metric cues.

Fuse the two streams at 1/2, 1/4, 1/8 and 1/16 resolution.

A practical lightweight channel configuration:

```text
1/2   : 32 channels
1/4   : 48 channels
1/8   : 64 channels
1/16  : 96 channels
```

Use depthwise-separable residual blocks or efficient ConvNeXt-style blocks if edge deployment is important.

Target completion-head size:

\[
\approx 3\text{M} - 6\text{M parameters}.
\]

The upstream monocular model must be reported separately.

---

# 13. The neural deformation field

The flow should not be an unconstrained generic U-Net output.

Use a **reaction + anisotropic diffusion** formulation:

\[
\boxed{
\frac{\partial c_t}{\partial t}
=
R_\theta(c_t,F,t)
+
\lambda(t)
\nabla\cdot
\left(
G_\theta(F,t)
\odot
\nabla c_t
\right).
}
\]

This decomposition is important.

## 13.1 Reaction term

\[
R_\theta(c_t,F,t)
\]

handles:

- non-local object corrections;
- prior failures;
- corrections that cannot be obtained by smooth propagation;
- nonlinear scale distortions.

## 13.2 Diffusion term

\[
\nabla\cdot
\left(
G_\theta\odot\nabla c_t
\right)
\]

propagates sparse metric corrections spatially.

The conductance field can be represented by horizontal/vertical edge weights:

\[
G_\theta=
[g_x,g_y],
\qquad
g_x,g_y\in[0,1].
\]

A strong monocular-depth discontinuity should usually reduce conductance:

\[
|\nabla R|\uparrow
\Rightarrow
g\downarrow.
\]

Thus a correction on a car can spread within the car but is discouraged from bleeding onto the road behind it.

The conductance is **learned**, not manually fixed.

---

# 14. Why this is more defensible than a generic Neural ODE

A generic formulation

\[
\frac{dc}{dt}=f_\theta(c,F,t)
\]

risks being dismissed as a recurrent residual network written in ODE notation.

The reaction-diffusion decomposition gives the dynamics an explicit depth-completion interpretation:

\[
\text{metric propagation}
+
\text{learned geometric correction}.
\]

It also creates interpretable ablations:

- diffusion only;
- reaction only;
- isotropic diffusion;
- learned anisotropic diffusion;
- no continuous recurrence.

If the continuous model cannot beat a matched direct residual model, the ODE/flow hypothesis should be rejected.

---

# 15. Ray-constrained ShapeFlow interpretation

For pixel \(u\):

\[
D_t(u)
=
D_0(u)e^{c_t(u)}.
\]

Its 3D point is:

\[
P_t(u)
=
D_0(u)e^{c_t(u)}r_u.
\]

Differentiate:

\[
\frac{dP_t(u)}{dt}
=
D_0(u)e^{c_t(u)}
\frac{dc_t(u)}{dt}
r_u.
\]

Thus:

\[
\boxed{
\frac{dP_t}{dt}
\parallel r_u.
}
\]

The learned geometric motion is always collinear with the original camera ray.

This provides a clean connection to ShapeFlow:

- ShapeFlow advects a source 3D shape through a learned vector field.
- AnchorFlow-DC advects a dense image-derived 3D surface through a **ray-restricted vector field**.

No point births are needed because the source depth map is already dense.

---

# 16. Numerical integration

Avoid an adaptive ODE solver.

Adaptive Dormand-Prince-style solving makes runtime unpredictable and can destroy edge-deployment advantages.

Recommended main configuration:

\[
K=3\text{ or }4
\]

fixed steps.

First implementation:

\[
c_{k+1}
=
\Pi_{\mathrm{anchor}}
\left[
c_k+
hV_\theta(c_k,F,t_k)
\right],
\]

where:

\[
h=\frac1K.
\]

This is Euler integration with shared weights.

Then test fixed-step Heun:

\[
k_1=V_\theta(c_k,F,t_k),
\]

\[
k_2=
V_\theta(c_k+hk_1,F,t_k+h),
\]

\[
c_{k+1}
=
c_k+
\frac{h}{2}(k_1+k_2).
\]

For edge deployment, compare accuracy against number of function evaluations (NFE), not merely "steps".

Recommended ablation:

```text
1 NFE
2 NFE
4 NFE
6 NFE
```

The intended operating point should ideally be 2-4 NFE.

---

# 17. Training with ground truth

## 17.1 Critical rule

Ground truth must **never** be used as a network input.

It only defines supervision:

\[
D^*.
\]

For KITTI, use the ground-truth validity mask:

\[
M_{gt}.
\]

All supervised metric losses are masked by \(M_{gt}\).

---

# 18. Stage 1 training — dense synthetic pretraining

This stage is strongly recommended.

Real KITTI supervision is semi-dense. A continuous deformation model benefits from dense supervision over the whole correction surface.

Use one or more synthetic datasets with dense metric depth.

Possible options:

- Virtual KITTI 2 for outdoor driving geometry;
- synthetic driving data from existing depth-completion pipelines;
- Hypersim / SceneNet-like data for additional geometric diversity if cross-domain robustness is a target.

For each dense synthetic sample:

1. obtain RGB and dense metric GT \(D^*\);
2. run the **same frozen monocular model** used at inference to produce \(R\);
3. simulate sparse LiDAR patterns;
4. perturb sparsity, noise and outliers;
5. create \(D_0\);
6. train the complete AnchorFlow dynamics.

This avoids training only on perfect priors and ensures the model learns actual failure modes of the chosen MDE foundation model.

---

# 19. LiDAR simulation curriculum

Do not train only with one fixed 64-line pattern.

Randomize:

```text
64-beam
32-beam
16-beam
8-beam
random point dropout
non-uniform spatial dropout
0-10% additive depth noise
0-5% gross outliers
localized missing scan regions
```

This is important because robustness across sparse-depth patterns is now a major evaluation dimension, as emphasized by OGNI-DC and OMNI-DC.

The curriculum can start with denser/cleaner inputs and progressively increase sparsity/noise.

---

# 20. Stage 2 training — real KITTI fine-tuning

For real KITTI:

Input:

\[
(R,D_s,M_s)
\]

Target:

\[
(D^*,M_{gt}).
\]

Because \(D^*\) is not dense everywhere, use:

- masked metric endpoint loss;
- sparse holdout loss;
- valid-neighbor gradient loss;
- prior-structure regularization in unlabeled regions.

Do **not** hallucinate missing GT labels as if they were true supervision.

---

# 21. Sparse-anchor holdout training

A particularly useful regularizer is to randomly split sparse LiDAR into:

\[
M_{\mathrm{in}}
\quad\text{and}\quad
M_{\mathrm{hold}}.
\]

The model receives only \(M_{\mathrm{in}}\).

The hidden LiDAR samples serve as additional metric supervision:

\[
\mathcal L_{\mathrm{hold}}
=
\frac1{|M_{\mathrm{hold}}|}
\sum
M_{\mathrm{hold}}
\rho(\hat D-D_s).
\]

This forces the network to learn spatial metric propagation instead of simply copying every observed anchor.

Recommended holdout rate:

\[
10\%-30\%.
\]

Randomize it during training.

---

# 22. Ground-truth correction field

When dense or valid GT is available:

\[
c^*(u)
=
\log(D^*(u)+\epsilon)
-
\log(D_0(u)+\epsilon).
\]

This is the actual correction that the flow needs to recover.

The endpoint supervision can therefore be applied either in depth space or correction space.

---

# 23. Coarse-to-fine trajectory supervision

A central question is:

> Why use multiple flow steps if a one-shot residual network can simply predict \(c^*\)?

The model needs a non-trivial trajectory objective.

Construct a Laplacian pyramid of the target correction:

\[
c^*
=
B_0+B_1+\cdots+B_L,
\]

where coarse bands encode large-scale metric calibration and fine bands encode object/detail corrections.

Define intermediate cumulative targets:

\[
c_1^*=B_0,
\]

\[
c_2^*=B_0+B_1,
\]

\[
\ldots
\]

\[
c_K^*=c^*.
\]

Interpretation:

```text
step 1 -> global / low-frequency metric correction
step 2 -> regional correction
step 3 -> object-level correction
step 4 -> fine boundaries/details
```

Then train:

\[
\mathcal L_{\mathrm{traj}}
=
\sum_{k=1}^{K}
w_k
\left\|
M_{gt}\odot(c_k-c_k^*)
\right\|_1.
\]

On synthetic data, this supervision is dense.

On real KITTI, apply it only where valid metric GT exists.

This design makes the recurrent deformation **coarse-to-fine by construction** instead of relying on the word "continuous" alone.

---

# 24. Recommended loss function

Use:

\[
\mathcal L
=
\lambda_d\mathcal L_d
+
\lambda_{inv}\mathcal L_{inv}
+
\lambda_c\mathcal L_c
+
\lambda_{traj}\mathcal L_{traj}
+
\lambda_{grad}\mathcal L_{grad}
+
\lambda_a\mathcal L_{anchor}
+
\lambda_h\mathcal L_{\mathrm{hold}}
+
\lambda_s\mathcal L_{\mathrm{smooth}}
+
\lambda_q\mathcal L_{\mathrm{conf}}.
\]

Do not activate every term at full weight from epoch 1.

---

## 24.1 Metric-depth loss

Masked Charbonnier or Smooth-L1:

\[
\mathcal L_d
=
\frac{
\sum
M_{gt}
\rho(\hat D-D^*)
}{
\sum M_{gt}
}.
\]

---

## 24.2 Inverse-depth loss

Because KITTI evaluates iRMSE/iMAE:

\[
\mathcal L_{inv}
=
\frac{
\sum
M_{gt}
\rho
\left(
\frac1{\hat D+\epsilon}
-
\frac1{D^*+\epsilon}
\right)
}{
\sum M_{gt}
}.
\]

This balances far/near geometry differently from raw-depth loss.

---

## 24.3 Correction loss

\[
\mathcal L_c
=
\frac{
\sum
M_{gt}
\rho(c_1-c^*)
}{
\sum M_{gt}
}.
\]

This directly supervises what the flow is meant to estimate.

---

## 24.4 Multi-scale gradient loss

For scales \(s\):

\[
\mathcal L_{\mathrm{grad}}
=
\sum_s
\left\|
M_{\nabla,s}
\odot
\left(
\nabla\log\hat D_s
-
\nabla\log D_s^*
\right)
\right\|_1.
\]

This is important for edge fidelity.

Only evaluate finite differences where the required GT neighbors are valid.

---

## 24.5 Anchor loss

For observed sparse measurements:

\[
\mathcal L_{\mathrm{anchor}}
=
\frac{
\sum M_s
q
\rho(\hat D-D_s)
}{
\sum M_s
}.
\]

If hard projection is used with perfectly clean anchors, this term becomes almost zero and can be downweighted.

---

## 24.6 Edge-aware correction regularization

Do not force depth itself to be overly smooth.

Smooth the **correction field**:

\[
\mathcal L_{\mathrm{smooth}}
=
\sum_u
w_R(u)
|\nabla c_1(u)|,
\]

where:

\[
w_R(u)
=
\exp
\left(
-\beta|\nabla \tilde R(u)|
\right).
\]

Hence correction is encouraged to vary smoothly within prior surfaces but is free to jump across predicted depth boundaries.

---

# 25. Prior trust map

A monocular prior can itself be wrong.

If the network is forced to preserve every prior edge, it may lock in bad geometry.

Therefore predict a prior-trust map:

\[
p(u)\in[0,1].
\]

High \(p\):
- preserve relative geometry strongly.

Low \(p\):
- allow the reaction term to override the prior.

Inputs to the trust predictor can include:

- relative-depth gradients;
- disagreement between sparse anchors and \(D_0\);
- local anchor density;
- current flow residual.

Use \(p\) to modulate conductance and prior-structure regularization.

This is preferable to blindly copying the monocular prior.

---

# 26. Optional uncertainty head

Predict:

\[
\sigma(u)>0.
\]

Train with a robust Laplacian/Gaussian NLL on valid GT.

Potential benefits:

- confidence-aware downstream fusion;
- better handling of regions far from anchors;
- diagnostic visualization;
- uncertainty can help choose whether additional flow steps are needed.

This is optional for the first implementation.

---

# 27. Proposed architecture specification

A practical first model:

```text
Relative geometry encoder
  channels: 32 -> 48 -> 64 -> 96

Sparse-anchor encoder
  channels: 16 -> 32 -> 48 -> 64

Cross-scale fusion
  lightweight gated concatenation + 1x1 conv

Flow core
  shared 3-scale U-Net / FPN
  depthwise-separable residual blocks
  time embedding
  current correction c_t as state input

Heads
  reaction field R_theta: 1 channel
  conductance G_theta: 2 channels
  anchor confidence q: 1 sparse-aware channel
  optional prior trust p: 1 channel
  optional uncertainty sigma: 1 channel

Solver
  fixed Euler or Heun
  2-4 NFE recommended

Output
  D_hat = D0 * exp(c_1)
```

Suggested target for the completion component:

```text
Parameters: 3-6 M
Inference state: full-resolution correction + multiscale features
NFE: 2-4
No adaptive solver
No test-time optimization
```

---

# 28. Pseudocode

```python
# Inputs:
# R       dense relative depth
# S       sparse metric LiDAR
# M       sparse valid mask
# Kcam    optional camera intrinsics

# 1. robust metric bootstrap
D0 = robust_response_alignment(R, S, M)

z0 = log(D0 + eps)
zs = log(S + eps)

# sparse metric correction
Cs = M * (zs - z0)

# 2. features
Fg = geometry_encoder(
    R,
    z0,
    grad(z0),
    optional_normals(D0, Kcam)
)

Fa = sparse_encoder(
    S,
    M,
    Cs,
    distance_to_valid(M)
)

F = fuse(Fg, Fa)

# 3. initialize ShapeFlow-like correction state
c = zeros_like(D0)

for k in range(K_steps):

    t = k / K_steps

    reaction, gx, gy, q = flow_net(c, F, t)

    diffusion = anisotropic_divergence(
        c,
        gx,
        gy
    )

    velocity = reaction + lambda_t(t) * diffusion

    # fixed-step integration
    c_new = c + dt * velocity

    # confidence-aware sparse Dirichlet projection
    c = where(
        M,
        q * Cs + (1 - q) * c_new,
        c_new
    )

D_hat = D0 * exp(c)

return D_hat
```

For Heun integration, evaluate the same shared vector field twice per step.

---

# 29. Training schedule

## Phase A — synthetic dense pretraining

Goal:
- learn dense anchor propagation;
- learn coarse-to-fine correction dynamics;
- make the flow robust to sparse pattern changes.

Train:
- all flow modules;
- anchor confidence;
- prior trust map.

Freeze:
- upstream relative-depth model.

---

## Phase B — KITTI adaptation

Use:
- KITTI sparse input LiDAR;
- KITTI relative prior;
- KITTI semi-dense GT.

Lower learning rate.

Emphasize:
- metric endpoint loss;
- inverse-depth loss;
- held-out anchor loss;
- real sensor robustness.

---

## Phase C — speed distillation, optional

Train a small-NFE student from a stronger multi-step teacher.

Example:

\[
K_{\mathrm{teacher}}=6,
\qquad
K_{\mathrm{student}}=2.
\]

Student loss:

\[
\mathcal L_{\mathrm{distill}}
=
\|
D_{\mathrm{student}}
-
\mathrm{stopgrad}(D_{\mathrm{teacher}})
\|_1.
\]

This is useful if the research target includes edge deployment.

---

# 30. Essential baselines

The proposal is only credible if compared against the following conceptual baselines under the **same relative prior**.

### B0 — robust alignment only

\[
R+D_s
\rightarrow D_0.
\]

Tests whether the neural model is needed.

### B1 — one-shot residual U-Net

\[
\hat D=D_0\exp(g_\theta(R,S,M)).
\]

This is the most important falsification baseline.

### B2 — recurrent residual network

Shared iterative network, but no time variable / continuous-field formulation.

### B3 — diffusion-only propagation

No reaction term.

### B4 — reaction-only

No explicit anisotropic diffusion.

### B5 — AnchorFlow without anchor projection

Tests whether metric boundary conditions matter.

### B6 — AnchorFlow with hard anchors

### B7 — AnchorFlow with learned anchor confidence

### B8 — matched-parameter CSPN/BP-style propagation head

Tests whether the proposed flow is more than another propagation network.

If AnchorFlow cannot outperform B1/B2 at the same parameter/runtime budget, the continuous-deformation framing is not justified.

---

# 31. Solver ablation

Report:

| Solver | NFE | Accuracy | Runtime | Notes |
|---|---:|---:|---:|---|
| direct residual | 1 | | | non-flow baseline |
| Euler | 1 | | | degenerate flow |
| Euler | 2 | | | |
| Euler | 4 | | | main candidate |
| Heun | 4 | | | 2 Heun steps |
| Euler | 6 | | | diminishing-return check |

The expected paper story should be based on an **accuracy-vs-NFE Pareto curve**, not an arbitrary solver choice.

---

# 32. Response-coordinate ablation

Because response geometry matters, compare:

| State | Formula |
|---|---|
| raw-depth residual | \(D=D_0+c\) |
| log-depth residual | \(D=D_0e^c\) |
| inverse-depth residual | \(1/D=1/D_0+c\) |

The main hypothesis is that log-depth is a strong compromise for KITTI, but this should be demonstrated rather than assumed.

---

# 33. Sparse-pattern robustness tests

Beyond standard KITTI input, test:

```text
64 lines
32 lines
16 lines
8 lines
50% random dropout
75% random dropout
random sparse points
localized LiDAR holes
Gaussian depth noise
outlier corruption
```

Report performance degradation curves.

A model that is marginally better at standard density but collapses at 16/8 lines is less compelling than a robust flow model.

---

# 34. Evaluation metrics

For KITTI report official metrics:

\[
\mathrm{RMSE},
\quad
\mathrm{MAE},
\quad
\mathrm{iRMSE},
\quad
\mathrm{iMAE}.
\]

Also report:

- parameters;
- GFLOPs;
- NFE;
- latency;
- peak GPU memory;
- upstream-prior runtime;
- completion-head runtime;
- total end-to-end runtime.

For deployment work, specify actual hardware.

Do not compare reported runtimes from different hardware as if directly equivalent.

---

# 35. Qualitative diagnostics

Visualize:

1. dense relative prior;
2. coarse metric bootstrap \(D_0\);
3. sparse correction anchors \(C_s\);
4. \(c_t\) after every step;
5. conductance maps \(g_x,g_y\);
6. prior-trust map;
7. anchor confidence;
8. final depth;
9. absolute error;
10. 3D point cloud.

A particularly strong figure would show:

```text
D0
 -> step 1 global correction
 -> step 2 regional correction
 -> step 3 object correction
 -> step 4 edge/detail correction
 -> GT
```

This directly supports the continuous-deformation interpretation.

---

# 36. Self-critique: main technical risks

## Risk 1 — "This is just a recurrent residual U-Net"

This is the most serious reviewer criticism.

Mitigation:

- explicit reaction-diffusion vector-field structure;
- shared vector field;
- time conditioning;
- ray-constrained geometric derivation;
- anchor projection after each integration step;
- coarse-to-fine trajectory supervision;
- matched direct/recurrent baselines.

If those do not provide measurable gains, remove the ODE claim rather than forcing it.

---

## Risk 2 — a good relative prior may already solve most of the problem

If \(D_0\) is already excellent after sparse calibration, the learned flow may provide only tiny gains.

Mitigation:
- test increasingly sparse anchors;
- test local prior distortions;
- test cross-domain priors;
- evaluate boundary and 3D geometry metrics, not RMSE alone.

---

## Risk 3 — poor monocular prior geometry

Relative-depth models can produce:
- incorrect edges;
- missing thin structures;
- wrong ordering;
- reflective/transparent failures.

A pure diffusion process could preserve these errors.

Mitigation:
- learned reaction term;
- learned prior-trust map;
- allow correction discontinuities where sparse evidence contradicts the prior;
- synthetic corruption of the relative prior during training.

---

## Risk 4 — hard anchors can be wrong

Exact sparse constraints can hurt around:
- moving objects;
- LiDAR/camera timestamp mismatch;
- calibration errors.

Mitigation:
- learned confidence;
- synthetic outlier training;
- robust bootstrap;
- report hard vs soft anchor ablation.

---

## Risk 5 — sparse KITTI GT is insufficient for trajectory supervision

Dense correction is unknown in unlabeled pixels.

Mitigation:
- dense synthetic pretraining;
- real fine-tuning with masked losses;
- holdout LiDAR supervision;
- do not treat missing GT as zero/error-free.

---

## Risk 6 — prior inference dominates compute

A lightweight completion head does not automatically imply a lightweight full system.

Mitigation:
- report both precomputed-prior and end-to-end settings;
- use a small upstream prior when targeting edge;
- distill the relative prior or cache it in temporal pipelines;
- benchmark full latency honestly.

---

## Risk 7 — novelty overlap with 2026 methods

LDCM, Midas Touch, Any2Full and Adaptive Response Geometry already cover significant parts of:

\[
\text{relative prior}+\text{sparse metric evidence}.
\]

Therefore the paper cannot claim novelty from that combination alone.

The novelty candidate must be narrowed to:

\[
\boxed{
\text{anchor-constrained, ray-preserving learned deformation dynamics}
}
\]

plus its specific trajectory/propagation design.

---

# 37. What should *not* be claimed

Do not claim:

- first flow-based depth completion;
- first relative-to-metric depth completion;
- first use of monocular foundation priors for depth completion;
- first use of hard sparse anchors;
- first iterative propagation method;
- first geometry-inspired depth-completion method.

All of these areas already have strong prior work.

A more defensible claim is:

> We investigate depth completion as deformation of a dense monocular geometry prior under sparse metric boundary conditions, using a ray-constrained learned continuous correction field.

Whether it is actually the first exact formulation still requires a final dedicated novelty search immediately before submission.

---

# 38. Candidate contribution statement

A possible future paper contribution statement:

> We propose AnchorFlow-DC, a ShapeFlow-inspired formulation of depth completion that views a dense monocular prior as a source 2.5D surface and sparse LiDAR samples as metric boundary conditions. Rather than regressing dense depth directly, a shared neural vector field continuously deforms a log-depth correction surface along fixed camera rays. The vector field combines learned reaction dynamics with anisotropic geometry-aware diffusion, while confidence-aware Dirichlet projections preserve trustworthy metric observations throughout integration. A coarse-to-fine trajectory objective and dense synthetic pretraining enable effective learning despite semi-dense real supervision.

This wording is a **proposal**, not a verified novelty claim.

---

# 39. Minimal viable implementation

Before building the full method, implement the following in order.

### Experiment 0 — prior + calibration

Inputs:
- relative depth;
- sparse LiDAR.

Output:
- \(D_0\).

Measure how much error remains.

### Experiment 1 — direct log-residual U-Net

\[
\hat D=D_0e^{c}.
\]

This establishes whether local learned correction is useful.

### Experiment 2 — shared iterative residual

Reuse the same network 4 times.

### Experiment 3 — add hard anchors

Project sparse metric corrections after each step.

### Experiment 4 — add anisotropic diffusion head

Separate reaction and propagation.

### Experiment 5 — add trajectory supervision

Coarse-to-fine correction pyramid.

### Experiment 6 — add confidence/trust heads

Only after the core idea works.

This progression prevents spending weeks on a complex model before establishing that the basic deformation hypothesis is valid.

---

# 40. Recommended first experiment configuration

For KITTI:

```yaml
prior:
  frozen: true
  precompute: true

bootstrap:
  response: log_depth
  solver: robust_affine_irls

completion_head:
  params_target: 4M
  state: log_depth_correction
  encoder_scales: [2, 4, 8, 16]
  channels: [32, 48, 64, 96]

flow:
  type: reaction_diffusion
  solver: euler
  nfe: 4
  shared_weights: true
  time_conditioning: true

anchors:
  projection: hard_first
  holdout_rate: [0.1, 0.3]

training:
  stage1: synthetic_dense
  stage2: kitti_real
  losses:
    - metric_smooth_l1
    - inverse_depth
    - correction
    - trajectory
    - gradient
    - holdout_anchor

ablation_priority:
  - direct_residual_vs_flow
  - nfe
  - hard_anchor
  - anisotropic_diffusion
  - response_coordinate
  - synthetic_pretraining
```

---

# 41. Success criteria

The idea is worth continuing only if several conditions hold.

### Accuracy

It should clearly beat:
- bootstrap only;
- matched direct residual;
- matched recurrent residual.

### Efficiency

The optimal point should remain in a small NFE regime.

### Robustness

Performance should degrade gracefully as LiDAR density decreases.

### Geometry

The model should improve:
- object boundaries;
- thin structures;
- point-cloud consistency;
- local scale consistency.

### Interpretability

Intermediate correction fields should exhibit meaningful progression rather than arbitrary oscillation.

If these conditions are not met, the continuous flow is probably unnecessary.

---

# 42. Potential stronger second-generation version

If the basic AnchorFlow-DC succeeds, a stronger model can evolve **two coupled states**:

\[
c_t
\]

for metric correction and

\[
p_t
\]

for prior reliability.

Dynamics:

\[
\frac{\partial c_t}{\partial t}
=
V_c(c_t,p_t,F,t),
\]

\[
\frac{\partial p_t}{\partial t}
=
V_p(c_t,p_t,F,t).
\]

Interpretation:

- the model not only corrects depth;
- it dynamically decides where the monocular geometry should be trusted.

This is more novel but should not be attempted before validating the single-state model.

---

# 43. Research conclusion

The strongest version of the ShapeFlow analogy is **not**:

\[
\text{sparse LiDAR}
\rightarrow
\text{flow}
\rightarrow
\text{dense points}.
\]

A continuous deformation cannot create missing topology from a sparse point set by itself.

The better formulation is:

\[
\boxed{
\text{dense relative surface}
+
\text{sparse metric anchors}
\rightarrow
\text{continuous ray-constrained deformation}
\rightarrow
\text{dense metric surface}.
}
\]

The relative prior solves the **density and structural-prior problem**.

Sparse LiDAR solves the **metric-reference problem**.

The learned flow solves the **local metric-correction and propagation problem**.

The ground truth supplies the **training signal for the deformation field**, but is never an inference input.

The most defensible architecture is therefore:

\[
\boxed{
R
\rightarrow
D_0
\rightarrow
c_0
\xrightarrow[\text{LiDAR Dirichlet anchors}]
{\text{learned reaction-diffusion flow}}
c_1
\rightarrow
D_0e^{c_1}.
}
\]

The decisive experiment is a matched comparison against a one-shot residual network. If continuous deformation does not produce a better accuracy/robustness/efficiency tradeoff, the model should be simplified rather than retaining an ODE merely for novelty.

---

# 44. References

1. Jiang, C. M., Huang, J., Tagliasacchi, A., Guibas, L. **ShapeFlow: Learnable Deformations Among 3D Shapes.** NeurIPS 2020.  
   https://arxiv.org/abs/2006.07982

2. Zuo, Y., Deng, J. **OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations.** ECCV 2024.  
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/319_ECCV_2024_paper.php

3. Tang, J. et al. **Bilateral Propagation Network for Depth Completion.** CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Bilateral_Propagation_Network_for_Depth_Completion_CVPR_2024_paper.html

4. Liang, Y. et al. **Distilling Monocular Foundation Model for Fine-grained Depth Completion.** CVPR 2025.  
   https://arxiv.org/abs/2503.16970

5. Gui, M. et al. **DepthFM: Fast Generative Monocular Depth Estimation with Flow Matching.** AAAI 2025 Oral.  
   https://ojs.aaai.org/index.php/AAAI/article/view/32330

6. Viola, M. et al. **Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Viola_Marigold-DC_Zero-Shot_Monocular_Depth_Completion_with_Guided_Diffusion_ICCV_2025_paper.html

7. Zuo, Y. et al. **OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_OMNI-DC_Highly_Robust_Depth_Completion_with_Multiresolution_Depth_Integration_ICCV_2025_paper.html

8. Ma, Y. et al. **The Midas Touch for Metric Depth.** CVPR 2026.  
   https://arxiv.org/abs/2605.11578

9. Zhou, Z. et al. **Any to Full: Prompting Depth Anything for Depth Completion in One Stage.** 2026.  
   https://arxiv.org/abs/2603.05711

10. Yu, Z. et al. **Large Depth Completion Model from Sparse Observations.** ICLR 2026.  
    https://arxiv.org/abs/2605.30115

11. Zhang, M., Peng, J. **Boosting Metric Depth Completion via Training-Free Adaptive Response Geometry.** arXiv, 2026-09-28.  
    https://arxiv.org/abs/2609.36168

12. KITTI Depth Completion Benchmark.  
    https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion

---

## Final note

This document is deliberately written as a **falsifiable research proposal**, not as a claim that every component is novel or guaranteed to beat current SOTA. The strongest first research question is:

\[
\boxed{
\text{Does learned anchor-constrained deformation outperform direct residual correction under the same prior, parameters and compute?}
}
\]

If yes, the ShapeFlow-inspired formulation has genuine empirical value. If no, simplify the architecture and retain only the components that survive ablation.
"""

path = Path("/mnt/data/AnchorFlow_DC_ShapeFlow_Inspired_Proposal.md")
path.write_text(md, encoding="utf-8")
print(f"Created: {path}")
print(f"Characters: {len(md):,}")
