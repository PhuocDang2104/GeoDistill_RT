# GeoDistill-RT

## Conflict-Aware Geometry Distillation for Real-Time Sparse Depth Completion

> **Goal.** Build a small real-time student model for image-guided sparse depth completion. During inference, the model uses only RGB, sparse depth, validity mask, and camera intrinsics. During training, multiple geometry teachers are used to produce reliable pseudo supervision.

---

## Table of Contents

1. [Core Idea](#1-core-idea)
2. [Input and Output](#2-input-and-output)
3. [Teacher Signals](#3-teacher-signals)
4. [Conflict-Aware Teacher Fusion](#4-conflict-aware-teacher-fusion)
5. [Student Architecture](#5-student-architecture)
6. [Training Objective](#6-training-objective)
7. [Inference](#7-inference)
8. [Research Gap and Contributions](#8-research-gap-and-contributions)
9. [Conclusion](#9-conclusion)
10. [Reference Notes](#10-reference-notes)

---

# 1. Core Idea

The objective is to build a **small real-time student model** for sparse depth completion.

At inference time, the model uses:

```math
(I,S,M,K)
```

and predicts:

```math
(D_{\text{full}},C_{\text{full}})
```

where:

- $D_{\text{full}}$ is the full-resolution metric depth.
- $C_{\text{full}}$ is the full-resolution confidence / uncertainty map.
- $D_{1/4}$ and $C_{1/4}$ are internal coarse predictions used for efficient computation.

During training, the framework separates teacher supervision into two different roles:

| Signal | Role |
|---|---|
| Ground-truth depth $D_{\text{gt}}$ | strongest metric anchor where available |
| DMD3C $D_{\text{DMD}}$ | primary dense coarse metric teacher |
| Depth Anything V2 | primary relative monocular geometry teacher for layout, edge, and ordinal structure |
| Metric3D v2 | metric-aware diagnostic / fallback geometry source |
| DSINE | surface-normal prior for geometric reliability |
| Sparse LiDAR $S$ | real metric anchor available at inference and training |

The teachers are used **only during training**. At inference time, only the lightweight student is deployed.

The key idea is a separated supervision and prediction pipeline:

1. the **coarse metric teacher** is built from $D_{\text{gt}}$ and DMD3C to minimize metric RMSE;
2. the **geometry teacher** is fused from the implemented teacher set: Depth Anything V2, Metric3D v2 diagnostics, and low-prior DMD3C structure;
3. the student predicts an efficient $1/4$ coarse depth internally, then produces full-resolution dense depth with guided upsampling and tiny residual refinement;
4. the geometry teacher is used through scale-and-shift-invariant, ordinal, and structure-aware losses.

---

# 2. Input and Output

## 2.1 Input

The input consists of RGB image, sparse depth, validity mask, and camera intrinsics:

```math
I \in \mathbb{R}^{H \times W \times 3},
\qquad
S \in \mathbb{R}^{H \times W \times 1},
\qquad
M \in \{0,1\}^{H \times W \times 1},
\qquad
K \in \mathbb{R}^{3 \times 3}.
```

where:

- $I$ is the RGB image.
- $S$ is the sparse depth map.
- $M$ is the validity mask.
- $K$ is the camera intrinsic matrix.

The valid sparse-depth set is:

```math
\Omega_M
=
\{p\mid M(p)=1,\ S(p)>0,\ S(p)<D_{\max}\}.
```

From $K$, construct a ray map:

```math
\mathbf{r}(u,v)
=
\mathrm{normalize}
\left[
\frac{u-c_x}{f_x},
\frac{v-c_y}{f_y},
1
\right].
```

## 2.2 Output

The student uses a coarse internal representation at $1/4$ resolution:

```math
D_{1/4} \in \mathbb{R}^{\frac{H}{4}\times\frac{W}{4}},
\qquad
C_{1/4} \in \mathbb{R}^{\frac{H}{4}\times\frac{W}{4}}.
```

The official inference output is full-resolution:

```math
D_{\text{full}} \in \mathbb{R}^{H\times W},
\qquad
C_{\text{full}} \in \mathbb{R}^{H\times W}.
```

The model decomposition is:

```math
D_{1/4},C_{1/4}
=
f_{\theta}(I,S,M,K),
```

```math
D_{\text{full}},C_{\text{full}}
=
g_{\phi}
\left(
D_{1/4},C_{1/4},I,S,M,K
\right).
```

Final output:

```math
D_{\text{out}}=D_{\text{full}},
\qquad
C_{\text{out}}=C_{\text{full}}.
```

---

# 3. Teacher Signals

The updated design uses **two teacher branches**:

```math
\mathcal{B}_{\text{metric}}
\qquad\text{and}\qquad
\mathcal{B}_{\text{geom}}.
```

The two branches have separate responsibilities and separate losses.

## 3.1 Coarse Metric Teacher: GT + DMD3C

The coarse metric teacher is responsible for minimizing RMSE. It uses:

```math
D_{\text{gt}},
\qquad
D_{\text{DMD}}.
```

where:

- $D_{\text{gt}}$ is the KITTI ground-truth depth where available;
- $D_{\text{DMD}}$ is the DMD3C dense depth-completion prediction;
- both are metric-depth signals in meters.

Because $D_{\text{gt}}$ is the most trustworthy metric supervision, it has priority over any teacher:

```math
\Omega_{\text{gt}}
=
\{p\mid D_{\text{gt}}(p)>0,\ D_{\text{gt}}(p)<D_{\max}\}.
```

Similarly:

```math
\Omega_{\text{DMD}}
=
\{p\mid D_{\text{DMD}}(p)>0,\ D_{\text{DMD}}(p)<D_{\max}\}.
```

Optionally, DMD3C can be calibrated to ground truth on valid pixels before being used outside $\Omega_{\text{gt}}$:

```math
(\gamma^\star,\delta^\star)
=
\arg\min_{\gamma,\delta}
\sum_{p\in\Omega_{\text{gt}}\cap\Omega_{\text{DMD}}}
\rho
\left(
\gamma D_{\text{DMD}}(p)+\delta-D_{\text{gt}}(p)
\right),
```

```math
\widehat{D}_{\text{DMD}}(p)
=
\gamma^\star D_{\text{DMD}}(p)+\delta^\star.
```

Without calibration:

```math
\widehat{D}_{\text{DMD}} = D_{\text{DMD}}.
```

The final coarse metric teacher is:

```math
D_{\text{cm}}(p)
=
\begin{cases}
D_{\text{gt}}(p),
& p\in\Omega_{\text{gt}},
\\[2mm]
\widehat{D}_{\text{DMD}}(p),
& p\notin\Omega_{\text{gt}},\ p\in\Omega_{\text{DMD}},
\\[2mm]
0,
& \text{otherwise}.
\end{cases}
```

The confidence of this metric teacher is:

```math
C_{\text{cm}}(p)
=
\begin{cases}
1,
& p\in\Omega_{\text{gt}},
\\[2mm]
c_{\text{DMD}}(p),
& p\notin\Omega_{\text{gt}},\ p\in\Omega_{\text{DMD}},
\\[2mm]
0,
& \text{otherwise}.
\end{cases}
```

where:

```math
c_{\text{DMD}}(p)=C_{\text{DMD}}(p).
```

DMD3C confidence combines sparse consistency, geometry agreement, edge risk, and range risk:

```math
C_{\text{DMD}}(p)
=
\mathrm{clip}
\left(
C_S(p)\,
C_G^{\text{DMD}}(p)\,
C_E(p)\,
C_R(p),
C_{\min},
1
\right).
```

The sparse-consistency term measures local disagreement with LiDAR anchors:

```math
e_S(p)
=
\min_{q\in\mathcal{N}_K(p)\cap\Omega_M}
\frac{
\left|D_{\text{DMD}}(q)-S(q)\right|
}{
S(q)+\epsilon
},
\qquad
C_S(p)=\exp(-a e_S(p)).
```

If $\mathcal{N}_K(p)\cap\Omega_M$ is empty, set $C_S(p)=1$.

The geometry-agreement term compares DMD3C structure with the fused geometry teacher once $R_G^\ast$ is available:

```math
e_G(p)
=
\left|
\mathrm{Norm}
\left(
\log(D_{\text{DMD}}(p)+\epsilon)
\right)
-
R_G^\ast(p)
\right|,
```

```math
C_G^{\text{DMD}}(p)=\exp(-b e_G(p)).
```

The edge-risk term lowers DMD3C supervision around high-risk discontinuities:

```math
e_E(p)
=
\left|\nabla D_{\text{DMD}}(p)\right|
\left|\nabla I(p)\right|,
\qquad
C_E(p)=\exp(-c e_E(p)).
```

A lightweight alternative is:

```math
C_E(p)=\exp(-c|\nabla I(p)|).
```

The range-risk term reduces confidence in far-range regions:

```math
e_R(p)
=
\frac{D_{\text{DMD}}(p)}{D_{\max}},
\qquad
C_R(p)=\exp(-d e_R(p)).
```

Recommended default:

```math
C_{\min}=0.05.
```

This branch produces the metric target used by:

```math
\mathcal{L}_{\text{cm}},
\qquad
\mathcal{L}_{C}.
```

It is the only dense teacher branch allowed to dominate metric RMSE.

## 3.2 Geometry Teacher: Fused Monocular Layout

The geometry branch is responsible for layout, ordinal depth, boundaries, and global scene structure. Candidate teachers include:

| Teacher | Output type | Recommended role |
|---|---|---|
| Depth Anything V2 | relative depth | strong dense structure and boundaries |
| Metric3D v2 | metric depth | optional diagnostic or fallback geometry source |
| DMD3C | metric depth completion | low-prior structure candidate and metric-confidence source |

Let the geometry teacher set be:

```math
\mathcal{G}
=
\left\{
G_1,\ldots,G_n
\right\}.
```

Each teacher is converted to a canonical structure representation before fusion:

```math
R_i(p)
=
\mathrm{Norm}_{\Omega_i}
\left(
\phi_i(G_i(p))
\right),
```

where:

- $\phi_i$ maps teacher output to a comparable structure signal;
- for relative-depth teachers, $\phi_i(G_i)=G_i$ or $\log(G_i+\epsilon)$;
- for metric-depth teachers, $\phi_i(G_i)=\log(G_i+\epsilon)$ or $1/(G_i+\epsilon)$;
- $\mathrm{Norm}$ is a robust median/MAD or percentile normalization.

A useful robust normalization is:

```math
\mathrm{Norm}_{\Omega}(x)(p)
=
\frac{x(p)-\mathrm{median}_{q\in\Omega}x(q)}
{\mathrm{MAD}_{q\in\Omega}x(q)+\epsilon}.
```

The geometry teacher branch produces structure supervision:

```math
R_G^\ast,
\qquad
C_G,
```

where $R_G^\ast$ is a fused relative/structure map and $C_G$ is its reliability.

---

# 4. Conflict-Aware Teacher Fusion

Fusion is now applied mainly to the **geometry teacher branch**:

```math
R_i\in\mathcal{G}.
```

The coarse metric branch follows the priority rule in Section 3.1:

```math
D_{\text{cm}}
=
D_{\text{gt}}\ \text{where valid, otherwise calibrated DMD3C}.
```

## 4.1 Metric-Branch Rule

For metric supervision:

```math
D_T^{\text{metric}}(p)=D_{\text{cm}}(p),
\qquad
C_T^{\text{metric}}(p)=C_{\text{cm}}(p).
```

This keeps metric RMSE anchored to GT and DMD3C while the monocular branch supplies structure.

## 4.2 Geometry Teacher Reliability

For each geometry teacher $R_i$, compute reliability using three optional terms.

If a term is unavailable for a teacher, set that penalty to zero:

```math
\Delta_i^k(p)=0
\quad
\text{for unavailable } k\in\{N,S,E\}.
```

### Normal Consistency

If a teacher can be converted to a metric-like depth map $\tilde{D}_i$ by SSI fitting to $D_{\text{cm}}$, compute a robust normal from the back-projected 3D point cloud.

Back-project each pixel:

```math
X_i(p)
=
\tilde{D}_i(p)
K^{-1}
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}.
```

Use symmetric finite differences in 3D:

```math
\mathbf{v}_x(p)
=
X_i(u+1,v)-X_i(u-1,v),
\qquad
\mathbf{v}_y(p)
=
X_i(u,v+1)-X_i(u,v-1).
```

The teacher-derived normal is:

```math
N(\tilde{D}_i)(p)
=
\frac{
\mathbf{v}_x(p)\times\mathbf{v}_y(p)
}{
\left\|
\mathbf{v}_x(p)\times\mathbf{v}_y(p)
\right\|_2+\epsilon
}.
```

The normal validity mask removes unstable depth discontinuities:

```math
M_N(p)
=
\mathbb{1}
\left(
\left|
\tilde{D}_i(u+1,v)-\tilde{D}_i(u-1,v)
\right|
<\tau_D
\right)
\mathbb{1}
\left(
\left|
\tilde{D}_i(u,v+1)-\tilde{D}_i(u,v-1)
\right|
<\tau_D
\right).
```

The raw normal disagreement is:

```math
\Delta_{i,\text{raw}}^N(p)
=
1-
\left\langle
N(\tilde{D}_i)(p),
N_{\text{DSINE}}(p)
\right\rangle.
```

Normal consistency is strongest on locally planar regions. Define:

```math
W_{\text{plane}}(p)
=
\exp
\left(
-\lambda_I|\nabla I(p)|
\right)
\exp
\left(
-\lambda_D|\nabla\tilde{D}_i(p)|
\right).
```

An optional curvature-aware version is:

```math
W_{\text{plane}}(p)
=
\exp
\left(
-\lambda_I|\nabla I(p)|
\right)
\exp
\left(
-\lambda_D|\nabla\tilde{D}_i(p)|
\right)
\exp
\left(
-\lambda_R|\Delta\tilde{D}_i(p)|
\right),
```

where $\Delta$ denotes a Laplacian / curvature approximation.

The robust normal penalty is:

```math
\Delta_i^N(p)
=
M_N(p)\,
W_{\text{plane}}(p)\,
\Delta_{i,\text{raw}}^N(p).
```

### Sparse Anchor Consistency

Sparse LiDAR should be used as a metric sanity check only after the teacher has been aligned to the metric branch:

```math
\Delta_i^S(p)
=
M(p)\,
\left|
\tilde{D}_i(p)-S(p)
\right|.
```

### Edge / Structure Consistency

Geometry teachers should agree with RGB or with each other at structure boundaries:

```math
\Delta_i^E(p)
=
\left|
\nabla R_i(p)
\right|
\cdot
\exp
\left(
-\kappa
\left|
\nabla I(p)
\right|
\right).
```

This term is optional. It penalizes structure edges that are not supported by RGB edges.

## 4.3 Geometry Teacher Weight

Define:

```math
q_i(p)
=
\exp
\left(
-\alpha
M_N(p)
W_{\text{plane}}(p)
\Delta_{i,\text{raw}}^N(p)
\right),
\qquad
r_i(p)=\exp(-\beta\Delta_i^S(p)),
\qquad
e_i(p)=\exp(-\eta\Delta_i^E(p)).
```

Use a teacher prior $\pi_i$ to encode empirical trust:

```math
\pi_i>0.
```

The geometry-fusion weight is:

```math
w_i^G(p)
=
\frac{
\pi_i q_i(p)r_i(p)e_i(p)
}{
\sum_{j\in\mathcal{G}}
\pi_j q_j(p)r_j(p)e_j(p)
\ +\epsilon
}.
```

Recommended initial priors:

| Teacher | Prior |
|---|---:|
| Depth Anything V2 | $1.0$ |
| Metric3D v2 | $0.5$ |
| DMD3C structure | $0.25$ |

The priors are tuned with validation metrics.

## 4.4 Fused Geometry Target

The fused geometry target is:

```math
R_G^\ast(p)
=
\sum_{i\in\mathcal{G}}
w_i^G(p)R_i(p).
```

The geometry confidence map is:

```math
C_G(p)
=
\max_{i\in\mathcal{G}}w_i^G(p).
```

This pair is saved separately from the metric teacher:

```math
\left(D_T^{\text{metric}}, C_T^{\text{metric}}\right)
\neq
\left(R_G^\ast, C_G\right).
```

**Interpretation.** DMD3C + GT determines metric scale. Monocular teachers determine relative layout and fine structure through $R_G^\ast$.

## 4.5 Boundary Ordinal Supervision

Depth discontinuities are supervised with relative ordering. Let $\mathcal{P}$ be a set of neighboring pixel pairs sampled around RGB or geometry edges:

```math
\mathcal{P}
=
\left\{
(p,q)\mid
q\in\mathcal{N}(p),
\ |\nabla I(p)|>\tau_I
\ \text{or}\
|\nabla R_G^\ast(p)|>\tau_G
\right\}.
```

The ordinal sign is:

```math
y_{pq}
=
\mathrm{sign}
\left(
R_G^\ast(p)-R_G^\ast(q)
\right).
```

The full-resolution ordinal loss is:

```math
\mathcal{L}_{\text{ord}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(p,q)\in\mathcal{P}}
\log
\left(
1+
\exp
\left[
-y_{pq}
\left(
\log(D_{\text{full}}(p)+\epsilon)
-
\log(D_{\text{full}}(q)+\epsilon)
\right)
\right]
\right).
```

This term teaches foreground/background ordering and thin-object boundaries without requiring a stable surface normal at discontinuities.

---

# 5. Student Architecture

## 5.1 Design Goal

The student model is designed to optimize:

- accuracy;
- runtime;
- teacher-free inference.

Input:

```math
(I,S,M,K).
```

Output:

```math
(D_{\text{full}},C_{\text{full}}).
```

The model also produces internal coarse predictions:

```math
(D_{1/4},C_{1/4}).
```

No normal map is predicted during inference. The normal teacher is only used during training to evaluate teacher reliability.

## 5.2 Architecture Overview

```text
RGB I + sparse depth S + mask M + intrinsics K
        ↓
Ray / UV map
        ↓
Fast Sparse Propagation at 1/4
        ↓
D_init at 1/4
        ↓
RGBStem + DepthStem + RayStem
        ↓
MobileViTv2 Encoder
        ↓
Sparse-Ray Gated Injection
        ↓
Additive LiteFPN Decoder
        ↓
Coarse log-residual head
        ↓
D_1/4, C_1/4
        ↓
Guided Full-Resolution Upsampling
        ↓
Tiny Full-Resolution Residual Refinement
        ↓
D_full, C_full
```

Main components:

1. **Fast Sparse Propagation**: creates a metric prior before the encoder.
2. **MobileViTv2 Encoder**: balances CNN local bias and lightweight global context.
3. **Sparse-Ray Gated Injection**: preserves sparse depth and ray map as metric/geometry anchors.
4. **Additive LiteFPN Decoder**: low memory traffic, suitable for coarse $1/4$ prediction.
5. **Coarse Log-Residual Head**: predicts a residual over the sparse-propagated prior.
6. **Guided Full-Resolution Upsampling**: uses RGB, sparse depth, and mask to recover full-resolution boundaries.
7. **Tiny Full-Resolution Residual Refinement**: corrects local residual errors with a very small full-resolution head.

---

## 5.3 Fast Sparse Propagation

Sparse depth is first converted into a coarse dense prior:

```math
D_{\text{init}}
```

at $1/4$ resolution.

For target pixel $p$, take the set of $K$ nearest valid sparse-depth pixels:

```math
\mathcal{N}_K(p),
\qquad
K=3 \text{ or } 4.
```

The initial depth is:

```math
D_{\text{init}}(p)
=
\sum_{q\in\mathcal{N}_K(p)}w_{pq}S(q),
```

with:

```math
\sum_{q\in\mathcal{N}_K(p)}w_{pq}=1.
```

The lightweight bilateral weight is:

```math
w_{pq}
=
\frac{
\exp
\left(
-\alpha\lVert p-q\rVert_2
-\beta\lVert I(p)-I(q)\rVert_1
-\gamma\lVert \mathbf{r}(p)-\mathbf{r}(q)\rVert_1
\right)
}{
\sum\limits_{q'\in\mathcal{N}_K(p)}
\exp
\left(
-\alpha\lVert p-q'\rVert_2
-\beta\lVert I(p)-I(q')\rVert_1
-\gamma\lVert \mathbf{r}(p)-\mathbf{r}(q')\rVert_1
\right)
}.
```

Recommended setting:

| Item | Setting |
|---|---|
| Resolution | $1/4$ |
| $K$ | 4 |
| Default | analytic bilateral weights |
| Ablation | tiny MLP weights |

Reasons:

- creates a metric prior before the encoder;
- works in both training and inference;
- cheaper than full BP-Net;
- more stable than raw sparse-depth concatenation;
- no scale-shift is applied to $D_{\text{init}}$ because it is derived from real sparse LiDAR.

---

## 5.4 Multi-Modal Stems

Use separate stems for each modality.

Depth input:

```math
X_D=[\log(S+\epsilon),M,D_{\text{init}}].
```

Geometry input:

```math
X_G=[r_x,r_y,r_z,u_{\text{norm}},v_{\text{norm}}].
```

Feature stems:

```math
F_I^0=\psi_I(I),
```

```math
F_D^0=\psi_D(X_D),
```

```math
F_G^0=\psi_G(X_G).
```

Fusion:

```math
F^0
=
\psi_0
\left(
\mathrm{Concat}
[F_I^0,F_D^0,F_G^0]
\right).
```

Recommended channels:

| Stem | Channels |
|---|---:|
| RGBStem | $3 \rightarrow 24$ |
| DepthStem | $3 \rightarrow 16$ |
| RayStem | $5 \rightarrow 12$ |
| Fusion | $52 \rightarrow 32$ or $48$ |

---

## 5.5 MobileViTv2 Encoder

Use MobileViTv2 as the main encoder:

```math
\{E_4,E_8,E_{16}\}
=
\mathrm{MobileViTv2Encoder}(F^0),
```

where:

```math
E_s
\in
\mathbb{R}^{\frac{H}{s}\times\frac{W}{s}\times C_s}.
```

Balanced configuration:

| Stage | Resolution | Channels | Block |
|---|---:|---:|---|
| Stem | $1/2$ | 32 | CNN stem |
| $E_4$ | $1/4$ | 48 | MobileViTv2 light block |
| $E_8$ | $1/8$ | 72 | MobileViTv2 block |
| $E_{16}$ | $1/16$ | 96--128 | MobileViTv2 block |

Recommended variants:

| Variant | Usage |
|---|---|
| MobileViTv2-0.5 | runtime-first |
| MobileViTv2-0.75 | default accuracy/runtime trade-off |
| MobileViTv2-1.0 | use only if accuracy is insufficient |

Design constraints:

- no attention-like block at full resolution;
- no heavy Transformer decoder;
- no classification head.

---

## 5.6 Sparse-Ray Gated Injection

Sparse depth and ray map are injected at multiple scales to preserve metric anchors.

For each scale:

```math
s\in\{4,8,16\},
```

construct sparse-ray prior:

```math
P_s
=
\rho_s
\left(
\mathrm{Pool}_s
[\log(S+\epsilon),M,D_{\text{init}},\mathbf{r}]
\right).
```

Gate:

```math
G_s
=
\sigma
\left(
\eta_s
(
\mathrm{Concat}[E_s,P_s]
)
\right).
```

Injected feature:

```math
\tilde{E}_s
=
E_s+G_s\odot P_s.
```

where $\rho_s$ and $\eta_s$ are $1\times1$ projections.

Reasons:

- preserves sparse depth as metric anchor;
- preserves camera geometry in features;
- cheaper than concat-heavy fusion.

---

## 5.7 Additive LiteFPN Decoder

Additive LiteFPN is used because it is suitable for $1/4$ coarse output, lighter than concat-heavy U-Net, and avoids iterative cost from SPN/NLSPN/DySPN.

Top-down decoding:

```math
P_{16}=\delta_{16}(\tilde{E}_{16}),
```

```math
P_8=\delta_8(\tilde{E}_8)+\mathrm{Up}_2(P_{16}),
```

```math
P_4=\delta_4(\tilde{E}_4)+\mathrm{Up}_2(P_8).
```

Smoothing:

```math
\hat{P}_8=\omega_8(P_8),
```

```math
\hat{P}_4=\omega_4(P_4).
```

where:

- $\delta_s$ is a $1\times1$ projection;
- $\omega_s$ is a lightweight $3\times3$ convolution block.

Recommended setting:

| Item | Setting |
|---|---|
| Fusion | add, not concat |
| Upsample | nearest or bilinear |
| Conv | standard $3\times3$ on GPU; DWConv+PWConv on mobile |
| Output feature | $\hat{P}_4$ |

---

## 5.8 Coarse Head, Guided Upsampling, and Full-Resolution Refinement

### Coarse Log-Residual Head

The coarse head predicts a log-residual over the sparse-propagated prior:

```math
\Delta z_{1/4}=h_D(\hat{P}_4).
```

The coarse metric depth is:

```math
D_{1/4}
=
D_{\text{init}}\,
\exp(\Delta z_{1/4}).
```

The exponential parameterization keeps depth positive and makes the residual relative to a metric prior derived from sparse LiDAR.

Coarse uncertainty:

```math
s_{1/4}=h_C(\hat{P}_4),
\qquad
C_{1/4}=\exp(-s_{1/4}).
```

Recommended coarse heads:

```text
Depth:      Conv3x3 -> activation -> Conv1x1 -> Delta z_1/4
Confidence: Conv3x3 -> activation -> Conv1x1 -> log-variance
```

### Guided Full-Resolution Upsampling

The full-resolution path uses guided upsampling:

```math
D_{\text{up}}
=
\mathrm{GuidedUp}
\left(
D_{1/4},I,S,M
\right).
```

A concrete form is learned convex upsampling:

```math
D_{\text{up}}(p)
=
\sum_{k\in\mathcal{N}(p)}
a_k(p)\,
D_{1/4}(k),
```

with:

```math
\sum_{k\in\mathcal{N}(p)}a_k(p)=1,
\qquad
a_k(p)\ge 0.
```

The upsampling weights are generated by a small head:

```math
a(p)
=
h_{\text{up}}
\left(
I,M,S,\mathrm{BilinearUp}(D_{1/4})
\right).
```

This keeps the main encoder cheap while allowing RGB and sparse depth to guide full-resolution boundaries.

### Tiny Full-Resolution Residual Refinement

After guided upsampling, a small residual head corrects local errors:

```math
\Delta z_{\text{full}}
=
h_R
\left(
[I,S,M,D_{\text{up}},C_{\text{up}}]
\right),
```

```math
D_{\text{full}}
=
D_{\text{up}}\,
\exp(\Delta z_{\text{full}}).
```

Full-resolution confidence can be predicted directly:

```math
C_{\text{full}}
=
h_{C,\text{full}}
\left(
[I,S,M,D_{\text{up}},C_{\text{up}}]
\right),
```

or computed with the lightweight option:

```math
C_{\text{full}}
=
\mathrm{GuidedUp}(C_{1/4},I).
```

Recommended tiny residual head:

```text
Input: RGB I, sparse S, mask M, D_up, C_up -> 7 channels
Conv3x3, 7 -> 16
DWConv3x3, 16 -> 16
Conv1x1, 16 -> 8
Conv1x1, 8 -> 1
Output: Delta z_full
```

### Sparse Anchor Correction

The final full-resolution prediction is softly corrected at real sparse LiDAR pixels:

```math
D_{\text{full}}(p)
\leftarrow
D_{\text{full}}(p)
+
\lambda_M M(p)
\left(
S(p)-D_{\text{full}}(p)
\right),
```

with:

```math
\lambda_M\in[0.5,0.9].
```

The hard-correction ablation is:

```math
D_{\text{full}}(p)
\leftarrow
M(p)S(p)
+
(1-M(p))D_{\text{full}}(p).
```

Interpretation:

- $D_{1/4}$ provides efficient global metric structure;
- guided upsampling recovers object boundaries;
- tiny full-resolution refinement corrects local residuals;
- sparse anchor correction preserves real LiDAR measurements.

---

## 5.9 Final Student Configuration

```text
GeoRT-Student-S

Input:
  RGB + sparse depth + mask + ray/uv

Sparse preprocessing:
  analytic fast propagation at 1/4, K=4

Encoder:
  MobileViTv2-0.75

Fusion:
  RGBStem + DepthStem + RayStem
  Sparse-Ray Gated Injection at 1/4, 1/8, 1/16

Decoder:
  Additive LiteFPN

Heads:
  coarse log-residual head
  coarse log-variance confidence head

Full-resolution path:
  guided convex upsampling
  tiny full-resolution residual head
  sparse anchor correction

Output:
  full-resolution metric depth D_full
  full-resolution confidence C_full
  internal coarse depth D_1_4 and confidence C_1_4

No inference teacher.
No normal output.
No full-resolution heavy encoder.
No heavy Transformer decoder.
No BEV/TPV branch at inference.
```

---

# 6. Training Objective

## 6.1 Training Targets

The training targets are defined at two resolutions.

Full-resolution targets:

```math
D_{\text{gt}},
\qquad
S,
\qquad
D_{\text{cm}},
\qquad
C_{\text{cm}},
\qquad
R_G^\ast,
\qquad
C_G.
```

Coarse auxiliary targets:

```math
D_{\text{gt}}^{\downarrow},
\qquad
D_{\text{cm}}^{\downarrow},
\qquad
C_{\text{cm}}^{\downarrow}.
```

The metric teacher is:

```math
D_{\text{cm}}(p)
=
\begin{cases}
D_{\text{gt}}(p),
& p\in\Omega_{\text{gt}},
\\[2mm]
\widehat{D}_{\text{DMD}}(p),
& p\notin\Omega_{\text{gt}},\ p\in\Omega_{\text{DMD}},
\\[2mm]
0,
& \text{otherwise}.
\end{cases}
```

The metric-teacher confidence is:

```math
C_{\text{cm}}(p)
=
\begin{cases}
1,
& p\in\Omega_{\text{gt}},
\\[2mm]
C_{\text{DMD}}(p),
& p\notin\Omega_{\text{gt}},\ p\in\Omega_{\text{DMD}},
\\[2mm]
0,
& \text{otherwise}.
\end{cases}
```

The geometry teacher is:

```math
R_G^\ast(p)
=
\sum_{i\in\mathcal{G}}
w_i^G(p)R_i(p).
```

The student outputs:

```math
D_{1/4},C_{1/4},D_{\text{full}},C_{\text{full}}.
```

Final supervision is applied mainly to $D_{\text{full}}$. Coarse supervision on $D_{1/4}$ is auxiliary.

---

## 6.2 Full-Resolution Metric Losses

### Ground-Truth Depth Loss

```math
\mathcal{L}_{\text{gt}}^{\text{full}}
=
\frac{1}{|\Omega_{\text{gt}}|}
\sum_{p\in\Omega_{\text{gt}}}
\rho
\left(
D_{\text{full}}(p)-D_{\text{gt}}(p)
\right).
```

### Sparse Consistency Loss

```math
\mathcal{L}_S^{\text{full}}
=
\frac{1}{|\Omega_M|}
\sum_{p\in\Omega_M}
\left|
D_{\text{full}}(p)-S(p)
\right|.
```

Sparse loss is independent from DMD3C supervision, so real LiDAR anchors remain authoritative when DMD3C conflicts with sparse measurements.

### Coarse Metric Teacher Loss

```math
\Omega_{\text{cm}}
=
\left\{
p\mid
C_{\text{cm}}(p)>0
\right\}.
```

```math
\mathcal{L}_{\text{cm}}^{\text{full}}
=
\frac{1}{|\Omega_{\text{cm}}|}
\sum_{p\in\Omega_{\text{cm}}}
C_{\text{cm}}(p)
\rho
\left(
D_{\text{full}}(p)-D_{\text{cm}}(p)
\right).
```

This term uses DMD3C through confidence masking. High-confidence DMD3C regions provide dense metric guidance; low-confidence regions leave more authority to sparse depth, GT, and geometry structure.

---

## 6.3 Coarse Auxiliary Loss

The coarse branch is trained with auxiliary supervision at $1/4$ resolution:

```math
\mathcal{L}_{\text{gt}}^{1/4}
=
\frac{1}{|\Omega_{\text{gt}}^{\downarrow}|}
\sum_{p\in\Omega_{\text{gt}}^{\downarrow}}
\rho
\left(
D_{1/4}(p)-D_{\text{gt}}^{\downarrow}(p)
\right),
```

```math
\mathcal{L}_{\text{cm}}^{1/4}
=
\frac{1}{|\Omega_{\text{cm}}^{\downarrow}|}
\sum_{p\in\Omega_{\text{cm}}^{\downarrow}}
C_{\text{cm}}^{\downarrow}(p)
\rho
\left(
D_{1/4}(p)-D_{\text{cm}}^{\downarrow}(p)
\right).
```

The auxiliary term is:

```math
\mathcal{L}_{1/4}
=
\lambda_{\text{gt}}^{1/4}
\mathcal{L}_{\text{gt}}^{1/4}
+
\lambda_{\text{cm}}^{1/4}
\mathcal{L}_{\text{cm}}^{1/4}.
```

This stabilizes $D_{1/4}$ while the final objective remains full-resolution.

---

## 6.4 Geometry SSI and Ordinal Losses

### Geometry SSI Loss

Let:

```math
\psi(D_{\text{full}})(p)
=
\log(D_{\text{full}}(p)+\epsilon).
```

The fused geometry teacher $R_G^\ast$ is scale-shift ambiguous. Align the geometry teacher to the student's current full-resolution log-depth prediction:

```math
(\alpha_G^\star,\beta_G^\star)
=
\arg\min_{\alpha_G,\beta_G}
\sum_{p\in\Omega_G}
C_G(p)
\rho
\left(
\psi(D_{\text{full}})(p)
-
\alpha_G R_G^\ast(p)
-
\beta_G
\right),
```

where:

```math
\Omega_G
=
\left\{
p\mid
C_G(p)>0
\right\}.
```

The geometry SSI loss is:

```math
\mathcal{L}_G^{\text{SSI}}
=
\frac{1}{|\Omega_G|}
\sum_{p\in\Omega_G}
C_G(p)
\rho
\left(
\log(D_{\text{full}}(p)+\epsilon)
-
\alpha_G^\star R_G^\ast(p)
-
\beta_G^\star
\right).
```

The fitted direction is:

```math
R_G^\ast
\rightarrow
\log(D_{\text{full}}+\epsilon).
```

This keeps metric scale controlled by GT, sparse depth, and DMD3C confidence masking.

### Boundary Ordinal Loss

For boundary pair set $\mathcal{P}$ from Section 4.5:

```math
\mathcal{L}_{\text{ord}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(p,q)\in\mathcal{P}}
\log
\left(
1+
\exp
\left[
-y_{pq}
\left(
\log(D_{\text{full}}(p)+\epsilon)
-
\log(D_{\text{full}}(q)+\epsilon)
\right)
\right]
\right).
```

This term supervises near/far ordering at boundaries, thin structures, and foreground/background discontinuities.

---

## 6.5 Confidence and Smoothness Losses

Let:

```math
s_{\text{full}}(p)=-\log(C_{\text{full}}(p)+\epsilon).
```

Confidence head is trained using uncertainty-weighted regression:

```math
\mathcal{L}_C
=
\frac{1}{|\Omega_{\text{sup}}|}
\sum_{p\in\Omega_{\text{sup}}}
\left[
\exp(-s_{\text{full}}(p))
\rho
\left(
D_{\text{full}}(p)-D_{\text{sup}}(p)
\right)
+
\lambda_s s_{\text{full}}(p)
\right].
```

where:

```math
D_{\text{sup}}(p)
=
\begin{cases}
D_{\text{gt}}(p),
& p\in\Omega_{\text{gt}},
\\[2mm]
D_{\text{cm}}(p),
& p\notin\Omega_{\text{gt}},\ p\in\Omega_{\text{cm}}.
\end{cases}
```

```math
\Omega_{\text{sup}}
=
\Omega_{\text{gt}}
\cup
\Omega_{\text{cm}}.
```

Edge-aware smoothness is applied at full resolution:

```math
\mathcal{L}_E
=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\left(
|\partial_xD_{\text{full}}(p)|e^{-|\partial_xI(p)|}
+
|\partial_yD_{\text{full}}(p)|e^{-|\partial_yI(p)|}
\right).
```

---

## 6.6 Total Objective

The final training objective is:

```math
\boxed{
\mathcal{L}
=
\lambda_{\text{gt}}
\mathcal{L}_{\text{gt}}^{\text{full}}
+
\lambda_S
\mathcal{L}_S^{\text{full}}
+
\lambda_{\text{cm}}
\mathcal{L}_{\text{cm}}^{\text{full}}
+
\lambda_{\text{aux}}
\mathcal{L}_{1/4}
+
\lambda_{\text{ssi}}
\mathcal{L}_G^{\text{SSI}}
+
\lambda_{\text{ord}}
\mathcal{L}_{\text{ord}}
+
\lambda_C
\mathcal{L}_C
+
\lambda_E
\mathcal{L}_E
}
```

Recommended initial weights:

| Weight | Value | Role |
|---|---:|---|
| $\lambda_{\text{gt}}$ | 1.0 | full-resolution GT supervision |
| $\lambda_S$ | 1.0 | full-resolution sparse LiDAR anchor |
| $\lambda_{\text{cm}}$ | 0.3--0.5 | DMD3C+GT metric teacher with confidence masking |
| $\lambda_{\text{aux}}$ | 0.2 | coarse $1/4$ stabilization |
| $\lambda_{\text{ssi}}$ | 0.03--0.05 | fused geometry SSI distillation |
| $\lambda_{\text{ord}}$ | 0.03--0.05 | boundary ordinal supervision |
| $\lambda_C$ | 0.03--0.05 | full-resolution confidence |
| $\lambda_E$ | 0.005--0.01 | edge-aware smoothness |
| $\lambda_s$ | 0.01 | confidence regularizer |

Training schedule:

| Stage | Objective |
|---|---|
| Epoch 0--5 | $\mathcal{L}_{\text{gt}}^{\text{full}}+\mathcal{L}_S^{\text{full}}+\lambda_{\text{aux}}\mathcal{L}_{1/4}$ |
| Epoch 5--10 | add $\mathcal{L}_{\text{cm}}^{\text{full}}$ |
| Epoch 10--15 | add $\mathcal{L}_G^{\text{SSI}}$ and $\mathcal{L}_{\text{ord}}$ |
| Epoch 15+ | add $\mathcal{L}_C$ and confidence weighting |

Reason:

1. Learn metric scale from GT and sparse LiDAR.
2. Stabilize the coarse residual branch with $1/4$ auxiliary supervision.
3. Add dense metric completion from DMD3C using confidence masking.
4. Add fused monocular geometry through SSI and ordinal losses.
5. Learn confidence after metric and geometry targets are stable.

---

## 6.7 Practical Output Names

Recommended saved teacher and student outputs:

| File group | Key | Meaning |
|---|---|---|
| `teacher_outputs/dmd3c/{split}` | `D_dmd3c` | raw DMD3C metric depth |
| `teacher_outputs/metric_coarse/{split}` | `D_cm`, `C_cm`, `C_dmd3c`, `D_dmd3c_raw`, `D_dmd3c_calibrated` | GT + generation-time calibrated DMD3C coarse metric teacher, plus raw/calibrated DMD3C audit maps |
| `teacher_outputs/geometry_raw/{teacher}/{split}` | `R_i` or raw model key | per-model relative/metric geometry output |
| `teacher_outputs/geometry_fused/{split}` | `R_G`, `C_G`, `w_*` | fused geometry teacher |
| `teacher_outputs/fused/{split}` | `D_teacher`, `C_teacher`, `D_full`, `C_full`, `C_dmd3c` | backward-compatible aliases and full-resolution DMD3C-dominant metric target |
| `student_outputs/{split}_predictions` | `D_full`, `C_full`, `D_1_4`, `C_1_4` | final full-resolution prediction and internal coarse prediction |

Conceptually:

```math
D_{\text{teacher}} \equiv D_{\text{cm}},
\qquad
R_G \not\equiv D_{\text{teacher}},
\qquad
D_{\text{out}} \equiv D_{\text{full}}.
```

---

# 7. Inference

Inference uses only:

```math
(I,S,M,K)\rightarrow(D_{\text{full}},C_{\text{full}}).
```

Not used during inference:

- Metric3D;
- Depth Anything;
- DMD3C;
- DSINE;
- normal output;
- BEV / TPV branch;
- full-resolution iterative propagation.

Final output:

```math
D_{\text{out}}=D_{\text{full}},
\qquad
C_{\text{out}}=C_{\text{full}}.
```

Output shapes:

```math
D_{\text{full}}\in\mathbb{R}^{H\times W},
\qquad
C_{\text{full}}\in\mathbb{R}^{H\times W}.
```

The internal coarse maps are retained for debugging and auxiliary supervision:

```math
D_{1/4},C_{1/4}\in\mathbb{R}^{\frac{H}{4}\times\frac{W}{4}}.
```

---

# 8. Research Gap and Contributions

Recent depth completion methods have used foundation models for dense supervision or sparse LiDAR for metric prediction. However, five limitations remain.

## 8.1 Research Gap

1. **Single-teacher distillation is fragile.**
   Many methods distill directly from one monocular foundation model. This improves fine-grained depth, but the teacher can suffer from local errors, scale ambiguity, and instability in object boundaries, reflective surfaces, far ranges, and low-texture regions.

2. **Metric and relative teachers are often mixed incorrectly.**
   Relative monocular teachers are valuable for layout and edges, but using them as hard metric labels can degrade RMSE. Metric supervision should remain anchored to GT, sparse LiDAR, and a depth-completion teacher such as DMD3C.

3. **Surface normal is underused for teacher reliability.**
   Some methods use surface normal as an auxiliary prediction or intermediate representation, but do not explicitly use normal consistency to estimate geometry-teacher reliability per pixel.

4. **Coarse-only output loses boundary detail.**
   A $1/4$ depth output is efficient, but final prediction quality depends on full-resolution boundary recovery around cars, poles, pedestrians, object silhouettes, and foreground/background discontinuities.

5. **Real-time inference is often not central.**
   Many high-accuracy methods rely on heavy backbones, iterative refinement, or auxiliary 3D branches, which are not ideal for real-time deployment.

## 8.2 Contributions

GeoDistill-RT addresses these gaps through:

1. **Multi-teacher geometry distillation**

   Depth Anything V2, Metric3D v2 diagnostics, DMD3C structure, DSINE, and sparse LiDAR are combined for dense structure supervision across complementary sources.

2. **DMD3C-dominant metric supervision**

   GT and DMD3C form the coarse metric teacher. Monocular geometry teachers affect metric prediction through SSI and ordinal structure losses.

3. **Conflict-aware geometry pseudo supervision**

   Sparse depth checks metric sanity after alignment, while surface normal and structure cues check geometric consistency of each monocular teacher.

4. **Full-resolution residual output**

   The student keeps a cheap $1/4$ core and produces final full-resolution depth through guided upsampling, tiny residual refinement, and sparse anchor correction.

5. **Real-time student design**

   All teachers are offline. Inference uses only a lightweight encoder, gated fusion, and a compact decoder.

---

# 9. Conclusion

GeoDistill-RT is a real-time sparse depth completion framework with separated metric and geometry supervision. GT plus DMD3C defines the coarse metric teacher for low RMSE, while Depth Anything V2, Metric3D v2 diagnostics, DMD3C structure, and DSINE provide fused geometry for layout-aware distillation. The student keeps teacher-free inference, uses an efficient $1/4$ internal representation, and outputs full-resolution metric depth with full-resolution confidence.

---

# 10. Reference Notes

The theory above follows these practical observations from recent teacher models:

1. **DMD3C / CVPR 2025.** [DMD3C](https://openaccess.thecvf.com/content/CVPR2025/html/Liang_Distilling_Monocular_Foundation_Model_for_Fine-grained_Depth_Completion_CVPR_2025_paper.html) uses monocular foundation-model distillation for fine-grained depth completion and explicitly addresses scale ambiguity with scale-and-shift-invariant learning during real-world fine-tuning.

2. **Depth Anything V2.** [DA2](https://github.com/DepthAnything/Depth-Anything-V2) provides strong relative depth and has separate metric variants, but the standard released pretrained models are primarily robust relative-depth predictors. Therefore, DA2 is best used as a dense structure teacher unless a metric variant is explicitly selected and validated.

3. **Metric3D v2.** Metric3D v2 is kept as a metric-aware diagnostic and fallback geometry source. It is not allowed to override the GT+DMD3C metric branch.
