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

---

# 1. Core Idea

The objective is to build a **small real-time student model** for sparse depth completion.

At inference time, the model uses:

$$
(I,S,M,K)
$$

and predicts:

$$
(D_c,C),
$$

where:

- $D_c$ is the coarse metric depth.
- $C$ is the confidence / uncertainty map.

During training, the framework uses multiple geometry teachers:

| Signal | Role |
|---|---|
| Metric3D v2 | metric depth prior |
| Depth Anything V2 | fine relative depth structure |
| DSINE | surface normal prior |
| Sparse LiDAR | real metric anchor |

The teachers are used **only during training**. At inference time, only the lightweight student is deployed.

The key idea is **not to distill from a single teacher directly**. Instead, multiple geometry teachers are fused with per-pixel reliability estimated from:

1. sparse LiDAR metric consistency;
2. surface-normal geometric consistency.

---

# 2. Input and Output

## 2.1 Input

The input consists of RGB image, sparse depth, validity mask, and camera intrinsics:

$$
I \in \mathbb{R}^{H \times W \times 3},
\qquad
S \in \mathbb{R}^{H \times W \times 1},
\qquad
M \in \{0,1\}^{H \times W \times 1},
\qquad
K \in \mathbb{R}^{3 \times 3}.
$$

where:

- $I$ is the RGB image.
- $S$ is the sparse depth map.
- $M$ is the validity mask.
- $K$ is the camera intrinsic matrix.

From $K$, construct a ray map:

$$
\mathbf{r}(u,v)
=
\operatorname{normalize}
\left[
\frac{u-c_x}{f_x},
\frac{v-c_y}{f_y},
1
\right].
$$

## 2.2 Output

The student predicts coarse metric depth and confidence at $1/4$ resolution:

$$
D_c \in \mathbb{R}^{\frac{H}{4}\times\frac{W}{4}},
\qquad
C \in \mathbb{R}^{\frac{H}{4}\times\frac{W}{4}}.
$$

If full-resolution visualization is required:

$$
D_{\text{out}}=\operatorname{Up}(D_c),
\qquad
C_{\text{out}}=\operatorname{Up}(C).
$$

---

# 3. Teacher Signals

Depth teachers:

$$
D_{\text{M3D}},
\qquad
D_{\text{DA}}.
$$

Normal teacher:

$$
N_{\text{DSINE}}.
$$

Sparse LiDAR:

$$
S.
$$

Depth Anything V2 mainly provides relative depth. Therefore, it is aligned to metric scale using sparse LiDAR:

$$
\tilde{D}_{\text{DA}} = aD_{\text{DA}} + b.
$$

The scale-shift parameters are estimated as:

$$
(a,b)
=
\arg\min_{a,b}
\sum_{p \in \Omega_M}
\rho
\left(
 aD_{\text{DA}}(p)+b-S(p)
\right),
$$

where:

$$
\Omega_M = \{p \mid M(p)=1\}.
$$

After alignment, $\tilde{D}_{\text{DA}}$ is treated as a metric-aligned depth candidate.

The depth teacher set is:

$$
\mathcal{T}
=
\left\{
D_{\text{M3D}},
\tilde{D}_{\text{DA}}
\right\}.
$$

---

# 4. Conflict-Aware Teacher Fusion

For each teacher:

$$
D_i \in \mathcal{T},
$$

compute the surface normal derived from depth:

$$
N(D_i).
$$

## 4.1 Normal Consistency

Measure geometric consistency between the teacher-derived normal and DSINE normal:

$$
\Delta_i^N(p)
=
1-
\left\langle
N(D_i)(p),
N_{\text{DSINE}}(p)
\right\rangle.
$$

## 4.2 Sparse Metric Consistency

Measure metric consistency at valid sparse LiDAR pixels:

$$
\Delta_i^S(p)
=
M(p)
\left|
D_i(p)-S(p)
\right|.
$$

## 4.3 Teacher Confidence Terms

Normal-based confidence:

$$
q_i(p)=\exp\left(-\alpha \Delta_i^N(p)\right).
$$

Sparse-depth-based confidence:

$$
r_i(p)=\exp\left(-\beta \Delta_i^S(p)\right).
$$

## 4.4 Teacher Weight

The final per-pixel teacher weight is:

$$
w_i(p)
=
\frac{
q_i(p)r_i(p)
}{
\sum_{j\in\mathcal{T}}q_j(p)r_j(p)
}.
$$

## 4.5 Fused Pseudo-Depth Target

The fused pseudo-depth target is:

$$
D_T^\ast(p)
=
\sum_{i\in\mathcal{T}}w_i(p)D_i(p).
$$

With two teachers:

$$
D_T^\ast(p)
=
w_{\text{M3D}}(p)D_{\text{M3D}}(p)
+
w_{\text{DA}}(p)\tilde{D}_{\text{DA}}(p).
$$

## 4.6 Optional Teacher Confidence Map

A simple and stable teacher confidence map is:

$$
C_T(p)=\max_{i\in\mathcal{T}}w_i(p).
$$

For a simpler baseline:

$$
C_T(p)=1.
$$

**Interpretation.** A teacher receives higher supervision weight at pixel $p$ if it is more consistent with both sparse LiDAR and DSINE surface normal.

---

# 5. Student Architecture

## 5.1 Design Goal

The student model is designed to optimize:

- accuracy;
- runtime;
- teacher-free inference.

Input:

$$
(I,S,M,K).
$$

Output:

$$
(D_c,C).
$$

No normal map is predicted during inference. The normal teacher is only used during training to evaluate teacher reliability.

## 5.2 Architecture Overview

```text
RGB I + sparse depth S + mask M + intrinsics K
        ↓
Ray / UV map
        ↓
Fast Sparse Propagation at 1/4
        ↓
RGBStem + DepthStem + RayStem
        ↓
MobileViTv2 Encoder
        ↓
Sparse-Ray Gated Injection
        ↓
Additive LiteFPN Decoder
        ↓
Depth Head + Confidence Head
```

Main components:

1. **Fast Sparse Propagation**: creates a metric prior before the encoder.
2. **MobileViTv2 Encoder**: balances CNN local bias and lightweight global context.
3. **Sparse-Ray Gated Injection**: preserves sparse depth and ray map as metric/geometry anchors.
4. **Additive LiteFPN Decoder**: low memory traffic, suitable for $1/4$ output.
5. **Depth + Confidence Heads**: predict coarse metric depth and uncertainty.

---

## 5.3 Fast Sparse Propagation

Sparse depth should not be only concatenated directly into the encoder. First, construct a coarse dense prior:

$$
D_{\text{init}}
$$

at $1/4$ resolution.

For target pixel $p$, take the set of $K$ nearest valid sparse-depth pixels:

$$
\mathcal{N}_K(p),
\qquad
K=3 \text{ or } 4.
$$

The initial depth is:

$$
D_{\text{init}}(p)
=
\sum_{q\in\mathcal{N}_K(p)}w_{pq}S(q),
$$

with:

$$
\sum_{q\in\mathcal{N}_K(p)}w_{pq}=1.
$$

The lightweight bilateral weight is:

$$
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
$$

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

$$
X_D=[\log(S+\epsilon),M,D_{\text{init}}].
$$

Geometry input:

$$
X_G=[r_x,r_y,r_z,u_{\text{norm}},v_{\text{norm}}].
$$

Feature stems:

$$
F_I^0=\psi_I(I),
$$

$$
F_D^0=\psi_D(X_D),
$$

$$
F_G^0=\psi_G(X_G).
$$

Fusion:

$$
F^0
=
\psi_0
\left(
\operatorname{Concat}
[F_I^0,F_D^0,F_G^0]
\right).
$$

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

$$
\{E_4,E_8,E_{16}\}
=
\operatorname{MobileViTv2Encoder}(F^0),
$$

where:

$$
E_s
\in
\mathbb{R}^{\frac{H}{s}\times\frac{W}{s}\times C_s}.
$$

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

$$
s\in\{4,8,16\},
$$

construct sparse-ray prior:

$$
P_s
=
\rho_s
\left(
\operatorname{Pool}_s
[\log(S+\epsilon),M,D_{\text{init}},\mathbf{r}]
\right).
$$

Gate:

$$
G_s
=
\sigma
\left(
\eta_s
(
\operatorname{Concat}[E_s,P_s]
)
\right).
$$

Injected feature:

$$
\tilde{E}_s
=
E_s+G_s\odot P_s.
$$

where $\rho_s$ and $\eta_s$ are $1\times1$ projections.

Reasons:

- preserves sparse depth as metric anchor;
- preserves camera geometry in features;
- cheaper than concat-heavy fusion.

---

## 5.7 Additive LiteFPN Decoder

Additive LiteFPN is used because it is suitable for $1/4$ coarse output, lighter than concat-heavy U-Net, and avoids iterative cost from SPN/NLSPN/DySPN.

Top-down decoding:

$$
P_{16}=\delta_{16}(\tilde{E}_{16}),
$$

$$
P_8=\delta_8(\tilde{E}_8)+\operatorname{Up}_2(P_{16}),
$$

$$
P_4=\delta_4(\tilde{E}_4)+\operatorname{Up}_2(P_8).
$$

Smoothing:

$$
\hat{P}_8=\omega_8(P_8),
$$

$$
\hat{P}_4=\omega_4(P_4).
$$

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

## 5.8 Depth Head and Confidence Head

### Depth Head

Predict log-depth for stable training:

$$
z_D=h_D(\hat{P}_4),
$$

$$
D_c=\exp(z_D).
$$

Recommended head:

```text
Conv3×3 → activation → Conv1×1 → log-depth
```

### Confidence Head

Predict log variance:

$$
s=h_C(\hat{P}_4),
\qquad
s=\log\sigma^2.
$$

Confidence:

$$
C=\exp(-s).
$$

Recommended head:

```text
Conv3×3 → activation → Conv1×1 → log-variance
```

Interpretation:

- high $C$: reliable prediction;
- low $C$: far sparse anchor, hard boundary, low texture, or teacher conflict.

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
  log-depth head
  log-variance confidence head

Output:
  coarse metric depth D_c
  confidence C

No inference teacher.
No normal output.
No full-resolution SPN.
No heavy Transformer decoder.
No BEV/TPV branch at inference.
```

---

# 6. Training Objective

## 6.1 Training Targets

Depth teachers:

$$
D_{\text{M3D}},
\qquad
D_{\text{DA}}.
$$

Metric-aligned Depth Anything:

$$
\tilde{D}_{\text{DA}}=aD_{\text{DA}}+b.
$$

$$
(a,b)
=
\arg\min_{a,b}
\sum_{p\in\Omega_M}
\rho
\left(
 aD_{\text{DA}}(p)+b-S(p)
\right).
$$

Teacher set:

$$
\mathcal{T}=\{D_{\text{M3D}},\tilde{D}_{\text{DA}}\}.
$$

Normal teacher:

$$
N_{\text{DSINE}}.
$$

The normal teacher is only used to compute teacher reliability, not as an inference output.

---

## 6.2 Conflict-Aware Teacher Fusion

For each teacher $D_i\in\mathcal{T}$:

Normal consistency:

$$
\Delta_i^N(p)
=
1-
\left\langle
N(D_i)(p),
N_{\text{DSINE}}(p)
\right\rangle.
$$

Sparse consistency:

$$
\Delta_i^S(p)
=
M(p)
\left|
D_i(p)-S(p)
\right|.
$$

Teacher confidence:

$$
q_i(p)=\exp(-\alpha\Delta_i^N(p)),
$$

$$
r_i(p)=\exp(-\beta\Delta_i^S(p)).
$$

Teacher weight:

$$
w_i(p)
=
\frac{
q_i(p)r_i(p)
}{
\sum_{j\in\mathcal{T}}q_j(p)r_j(p)
}.
$$

Pseudo target:

$$
D_T^\ast(p)=\sum_{i\in\mathcal{T}}w_i(p)D_i(p).
$$

With two teachers:

$$
D_T^\ast(p)
=
w_{\text{M3D}}(p)D_{\text{M3D}}(p)
+
w_{\text{DA}}(p)\tilde{D}_{\text{DA}}(p).
$$

Optional teacher confidence:

$$
C_T(p)=\max_{i\in\mathcal{T}}w_i(p).
$$

Simple baseline:

$$
C_T(p)=1.
$$

---

## 6.3 Student Loss

The student outputs:

$$
D_c,
\qquad
s,
$$

where:

$$
s(p)=\log\sigma^2(p),
\qquad
C(p)=\exp(-s(p)).
$$

The total objective is:

$$
\boxed{
\mathcal{L}
=
\lambda_{\text{gt}}\mathcal{L}_{\text{gt}}
+
\lambda_T\mathcal{L}_T
+
\lambda_S\mathcal{L}_S
+
\lambda_C\mathcal{L}_C
+
\lambda_E\mathcal{L}_E
}
$$

No normal loss $\mathcal{L}_N$ is used in the main objective because DSINE is already used in teacher fusion through $\Delta_i^N$.

### Ground-Truth Depth Loss

$$
\mathcal{L}_{\text{gt}}
=
\frac{1}{|\Omega_{\text{gt}}|}
\sum_{p\in\Omega_{\text{gt}}}
\rho
\left(
D_c(p)-D_{\text{gt}}^{\downarrow}(p)
\right).
$$

Here, $D_{\text{gt}}^{\downarrow}$ is the ground-truth depth at the resolution of $D_c$.

### Teacher Distillation Loss

$$
\mathcal{L}_T
=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
C_T(p)
\rho
\left(
D_c(p)-D_T^\ast(p)
\right).
$$

If teacher confidence is not used:

$$
\mathcal{L}_T
=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\rho
\left(
D_c(p)-D_T^\ast(p)
\right).
$$

### Sparse Consistency Loss

$$
\mathcal{L}_S
=
\frac{1}{|\Omega_M|}
\sum_{p\in\Omega_M}
\left|
D_c(p)-S^{\downarrow}(p)
\right|.
$$

Here, $S^{\downarrow}$ and $M^{\downarrow}$ are downsampled to the resolution of $D_c$.

### Confidence / Uncertainty Loss

Confidence head is trained using uncertainty-weighted regression:

$$
\mathcal{L}_C
=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\left[
\exp(-s(p))
\rho
\left(
D_c(p)-D_{\text{sup}}(p)
\right)
+
\lambda_s s(p)
\right].
$$

where:

$$
D_{\text{sup}}(p)
=
\begin{cases}
D_{\text{gt}}^{\downarrow}(p), & p\in\Omega_{\text{gt}},\\
D_T^\ast(p), & \text{otherwise}.
\end{cases}
$$

Interpretation:

- confident region: low $s$, high $C$;
- hard / noisy / conflict region: high $s$, low $C$.

### Edge-Aware Smoothness Loss

$$
\mathcal{L}_E
=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
\left(
|\partial_xD_c(p)|e^{-|\partial_xI^{\downarrow}(p)|}
+
|\partial_yD_c(p)|e^{-|\partial_yI^{\downarrow}(p)|}
\right).
$$

Here, $I^{\downarrow}$ is the RGB image downsampled to the same resolution as $D_c$.

---

## 6.4 Recommended Loss Weights

Initial setting:

| Weight | Value |
|---|---:|
| $\lambda_{\text{gt}}$ | 1.0 |
| $\lambda_T$ | 0.5 |
| $\lambda_S$ | 1.0 |
| $\lambda_C$ | 0.05 |
| $\lambda_E$ | 0.01 |
| $\lambda_s$ | 0.01 |

Training schedule:

| Stage | Objective |
|---|---|
| Epoch 0--5 | $\mathcal{L}_{\text{gt}}+\mathcal{L}_S$ |
| Epoch 5--15 | add $\mathcal{L}_T$ |
| Epoch 15+ | add $\mathcal{L}_C$ and optional $C_T$ weighting |

Reason:

1. First learn metric scale from GT and sparse LiDAR.
2. Then add dense teacher supervision.
3. Finally learn confidence to avoid uncertainty collapse.

---

# 7. Inference

Inference uses only:

$$
(I,S,M,K)\rightarrow(D_c,C).
$$

Not used during inference:

- Metric3D;
- Depth Anything;
- DSINE;
- normal output;
- BEV / TPV branch;
- full-resolution iterative propagation.

Final output:

$$
D_c,C\in\mathbb{R}^{\frac{H}{4}\times\frac{W}{4}}.
$$

Optional full-resolution visualization:

$$
D_{\text{out}}=\operatorname{Up}(D_c),
\qquad
C_{\text{out}}=\operatorname{Up}(C).
$$

---

# 8. Research Gap and Contributions

Recent depth completion methods have used foundation models for dense supervision or sparse LiDAR for metric prediction. However, three limitations remain.

## 8.1 Research Gap

1. **Single-teacher distillation is fragile.**
   Many methods distill directly from one monocular foundation model. This improves fine-grained depth, but the teacher can suffer from local errors, scale ambiguity, and instability in object boundaries, reflective surfaces, far ranges, and low-texture regions.

2. **Surface normal is underused for teacher reliability.**
   Some methods use surface normal as an auxiliary prediction or intermediate representation, but do not explicitly use normal consistency to estimate teacher reliability per pixel.

3. **Real-time inference is often not central.**
   Many high-accuracy methods rely on heavy backbones, iterative refinement, or auxiliary 3D branches, which are not ideal for real-time deployment.

## 8.2 Contributions

GeoDistill-RT addresses these gaps through:

1. **Multi-teacher geometry distillation**

   Metric3D v2, Depth Anything V2, DSINE, and sparse LiDAR are combined instead of relying on a single supervision source.

2. **Conflict-aware pseudo supervision**

   Sparse depth checks metric consistency, while surface normal checks geometric consistency of each depth teacher.

3. **Real-time student design**

   All teachers are offline. Inference uses only a lightweight encoder, gated fusion, and a compact decoder.

---

# 9. Conclusion

GeoDistill-RT is a real-time sparse depth completion framework that uses multiple geometry teachers to build more reliable pseudo supervision. It resolves conflicts between metric depth, relative depth, surface normal, and sparse LiDAR during training, then distills the filtered supervision into a compact student model for fast teacher-free inference.
