# GeoDistill-RT

## Conflict-Aware Geometry Distillation with a Scale-Equivariant, Anchor-Routed Real-Time Student

> **Method definition.** GeoDistill-RT is an RGB-guided sparse depth-completion framework. Training uses separated metric and relative-geometry supervision. Deployment uses only RGB, sparse metric depth, its validity mask, and camera intrinsics. The inference student is GeoRT-SAR, built around Scale-Equivariant Multi-Scale Anchor Routing (SE-MSAR) and a Scale-Preserving Boundary Residual Pyramid (SP-BRP).

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Scale-Normalized Input Representation](#2-scale-normalized-input-representation)
3. [Separated Teacher Supervision](#3-separated-teacher-supervision)
4. [Conflict-Aware Teacher Reliability](#4-conflict-aware-teacher-reliability)
5. [GeoRT-SAR Student](#5-geort-sar-student)
6. [Training Objective](#6-training-objective)
7. [Training and Augmentation Protocol](#7-training-and-augmentation-protocol)
8. [Inference and Deployment Contract](#8-inference-and-deployment-contract)
9. [Repository Contract](#9-repository-contract)
10. [Contributions](#10-contributions)

---

# 1. Problem Definition

Given an RGB image $I$, sparse metric depth $S$, validity mask $M$, and camera intrinsics $K$, the student predicts dense metric depth and a reliability map:

$$
(I,S,M,K)\longrightarrow(D_{\mathrm{full}},C_{\mathrm{full}}).
$$

The tensors are

$$
I\in\mathbb{R}^{H\times W\times 3},\quad
S\in\mathbb{R}^{H\times W},\quad
M\in\{0,1\}^{H\times W},\quad
K\in\mathbb{R}^{3\times3},
$$

$$
D_{\mathrm{full}},C_{\mathrm{full}}\in\mathbb{R}^{H\times W}.
$$

The valid sparse set is

$$
\Omega_M=\{p\mid M(p)=1,\ d_{\min}<S(p)<d_{\max}\}.
$$

The deployed model does not use DMD3C, Depth Anything V2, Metric3D v2, DSINE, a normal predictor, an iterative propagation network, or an optimization solver. All teacher computation is offline.

The method has two independent responsibilities:

1. **Metric completion:** preserve absolute scale and recover low-frequency scene depth from real sparse anchors, ground truth, and DMD3C supervision.
2. **Structural reconstruction:** recover object boundaries, thin structures, and near/far ordering using relative geometry supervision and the SP-BRP output path.

The internal coarse outputs are

$$
D_{1/4},C_{1/4}\in
\mathbb{R}^{\frac H4\times\frac W4},
$$

while $D_{\mathrm{full}}$ and $C_{\mathrm{full}}$ are the official inference outputs.

---

# 2. Scale-Normalized Input Representation

## 2.1 Sparse-depth scale normalization

For each sample, define the detached sparse-depth median

$$
m_S=\operatorname{median}\{S(p)\mid p\in\Omega_M\}.
$$

If $\Omega_M$ is empty, the implementation uses $m_S=1$ and marks the anchor bank invalid; zero-anchor behavior must be trained and evaluated separately. The normalized sparse depth is

$$
\bar S(p)=\frac{M(p)S(p)}{m_S+\epsilon}.
$$

The network operates in normalized log-depth space:

$$
z(p)=\log(\bar D(p)+\epsilon),
\qquad
D(p)=m_S\exp z(p).
$$

This representation targets scale equivariance:

$$
F(I,\beta S,M,K)\approx\beta F(I,S,M,K),\qquad\beta>0.
$$

The mask is not scaled. Median normalization alone is not treated as an independent contribution; it is the coordinate system in which anchor routing and bounded residual correction operate.

## 2.2 Camera geometry and sparse-distance maps

For pixel $p=(u,v)$, construct a unit camera ray

$$
\mathbf r(u,v)=
\frac{
\left[(u-c_x)/f_x,\ (v-c_y)/f_y,\ 1\right]^\top
}{
\left\|\left[(u-c_x)/f_x,\ (v-c_y)/f_y,\ 1\right]^\top\right\|_2+\epsilon
}.
$$

Normalized image coordinates are

$$
u_n=2u/(W-1)-1,\qquad v_n=2v/(H-1)-1.
$$

Let $d_M(p)$ be the normalized Euclidean distance from $p$ to the nearest valid sparse point. The depth and geometry inputs are

$$
X_D=[\log(\bar S+\epsilon),M,d_M],
$$

$$
X_G=[r_x,r_y,r_z,u_n,v_n].
$$

Ray and UV maps must be regenerated after every resize, crop, or horizontal flip using the transformed intrinsics.

---

# 3. Separated Teacher Supervision

Teacher roles are separated by semantics. A stored convenience target must not redefine these roles.

| Signal | Semantic role | Metric authority |
|---|---|---:|
| $D_{\mathrm{gt}}$ | supervised ground-truth target | highest where valid |
| $S$ | real sensor anchor available at train and inference | highest at valid sensor samples |
| $D_{\mathrm{DMD}}$ | only dense metric teacher | confidence weighted |
| Depth Anything V2 | primary relative layout and boundary teacher | none |
| Metric3D v2 | optional secondary geometry teacher or diagnostic | none in the main metric loss |
| DSINE | normal-reliability signal | none |

## 3.1 Dense metric teacher

The dense metric teacher is

$$
D_T^{\mathrm{metric}}=
\operatorname{stopgrad}(D_{\mathrm{DMD}}).
$$

Ground truth and sparse LiDAR remain separate supervised signals; they are not called teachers. An optional affine calibration of DMD3C may be evaluated as an ablation:

$$
(\gamma^\star,\delta^\star)=
\arg\min_{\gamma,\delta}
\sum_{p\in\Omega_{\mathrm{gt}}\cap\Omega_{\mathrm{DMD}}}
\rho(\gamma D_{\mathrm{DMD}}(p)+\delta-D_{\mathrm{gt}}(p)),
$$

$$
\widehat D_{\mathrm{DMD}}=\gamma^\star D_{\mathrm{DMD}}+\delta^\star.
$$

The uncalibrated $D_{\mathrm{DMD}}$ is the default theoretical signal. Calibration status, fitted parameters, overlap count, and raw/calibrated maps must be retained when calibration is enabled.

## 3.2 Relative geometry teachers

The default geometry set is

$$
\mathcal G=\{\mathrm{DA2},\mathrm{Metric3D\ v2}\},
$$

with Depth Anything V2 as the primary source and Metric3D v2 as an optional secondary source. DMD3C is excluded from the default geometry fusion because it already inherits correlated knowledge from a monocular foundation model and is used as the metric teacher.

Each geometry prediction $G_i$ is converted into a canonical structure map:

$$
R_i(p)=\operatorname{RobustNorm}_{\Omega_i}(\phi_i(G_i(p))),
$$

where $\phi_i$ is the raw relative output for a relative model and log-depth for a metric-like diagnostic. A robust normalization is

$$
\operatorname{RobustNorm}_{\Omega}(x)=
\frac{x-\operatorname{median}_{\Omega}(x)}
{\operatorname{MAD}_{\Omega}(x)+\epsilon}.
$$

The geometry branch outputs a relative structure target and its reliability:

$$
(R_G^\star,C_G).
$$

Neither map is a metric-depth target.

## 3.3 No self-referential teacher confidence

DMD3C confidence is allowed to compare DMD3C with a geometry reference only when that reference excludes DMD3C:

$$
R_G^{-\mathrm{DMD}}
=
w_{\mathrm{DA2}}R_{\mathrm{DA2}}
+w_{\mathrm{M3D}}R_{\mathrm{M3D}}.
$$

The invalid dependency

$$
D_{\mathrm{DMD}}\rightarrow R_G^\star
\rightarrow C_{\mathrm{DMD}}
\rightarrow\mathcal L_{\mathrm{DMD}}
$$

is therefore absent. If a diagnostic experiment includes DMD3C structure in a larger fusion, $C_{\mathrm{DMD}}$ must still use leave-one-out agreement with $R_G^{-\mathrm{DMD}}$.

---

# 4. Conflict-Aware Teacher Reliability

## 4.1 Geometry-teacher reliability

For each $i\in\mathcal G$, define a prior $\pi_i$ and validity $V_i$. Reliability can use normal, sparse, and unsupported-edge penalties.

When a metric-like realization $\widetilde D_i$ is available after fitting the geometry output to a metric reference, back-project it as

$$
X_i(p)=\widetilde D_i(p)K^{-1}[u,v,1]^\top
$$

and derive the surface normal $N(\widetilde D_i)$. DSINE supplies a reliability reference rather than a depth label:

$$
\Delta_i^N(p)=W_{\mathrm{plane}}(p)
\left[1-\langle N(\widetilde D_i)(p),N_{\mathrm{DSINE}}(p)\rangle\right],
$$

$$
W_{\mathrm{plane}}(p)=
\exp(-\lambda_I|\nabla I(p)|)
\exp(-\lambda_D|\nabla\widetilde D_i(p)|).
$$

Sparse consistency is computed only after metric alignment:

$$
\Delta_i^S(p)=M(p)|\widetilde D_i(p)-S(p)|.
$$

Unsupported structure edges are penalized by

$$
\Delta_i^E(p)=|\nabla R_i(p)|
\exp(-\kappa|\nabla I(p)|).
$$

The unnormalized score and normalized fusion weight are

$$
q_i(p)=\pi_iV_i(p)
\exp[-\alpha\Delta_i^N(p)-\beta\Delta_i^S(p)-\eta\Delta_i^E(p)],
$$

$$
w_i^G(p)=\frac{q_i(p)}{\sum_{j\in\mathcal G}q_j(p)+\epsilon}.
$$

The fused relative geometry and confidence are

$$
R_G^\star(p)=\sum_{i\in\mathcal G}w_i^G(p)R_i(p),
$$

$$
C_G(p)=\mathbb 1\!\left[\sum_iq_i(p)>0\right]
\max_{i\in\mathcal G}w_i^G(p).
$$

A stronger confidence alternative, such as normalized score mass or inter-teacher agreement, must be calibrated and reported as a separate ablation.

## 4.2 DMD3C confidence

DMD3C reliability is factored into independent risks:

$$
C_{\mathrm{DMD}}(p)=
C_{\mathrm{sparse}}(p)
C_{\mathrm{edge}}(p)
C_{\mathrm{range}}(p)
C_{\mathrm{agree}}(p).
$$

Sparse consistency uses relative error at LiDAR anchors, propagated only within a declared radius:

$$
e_S(q)=\frac{|D_{\mathrm{DMD}}(q)-S(q)|}{S(q)+\epsilon},
\qquad
C_{\mathrm{sparse}}(p)=\exp[-a\,\widetilde e_S(p)].
$$

The edge and range terms are

$$
C_{\mathrm{edge}}(p)=\exp[-c\,e_E(p)],
\qquad
C_{\mathrm{range}}(p)=
\exp\left[-d\frac{D_{\mathrm{DMD}}(p)}{d_{\max}}\right].
$$

Agreement is structure-only and leave-one-out:

$$
e_A(p)=
\left|
\operatorname{RobustNorm}(\log(D_{\mathrm{DMD}}(p)+\epsilon))
-R_G^{-\mathrm{DMD}}(p)
\right|,
$$

$$
C_{\mathrm{agree}}(p)=\exp[-b e_A(p)].
$$

Confidence is zero outside valid DMD3C depth. A positive floor may be tested, but the main result must report whether the floor was used because it changes the effective teacher coverage.

---

# 5. GeoRT-SAR Student

## 5.1 Pipeline

```text
RGB I + sparse depth S + mask M + intrinsics K
                         |
          per-sample median depth normalization
                         |
              ray / UV / sparse-distance maps
                         |
             multi-scale analytic anchor bank
                         |
             RGB + depth + geometry stems
                         |
       lightweight local-global encoder (1/4--1/16)
                         |
             SE-MSAR updates at 1/8 and 1/4
                         |
          additive low-frequency metric decoder
                         |
             normalized log-depth at 1/4
                         |
              SP-BRP at 1/2 and full scale
                         |
           adaptive sparse correction + confidence
                         |
                D_full, C_full
```

## 5.2 Multi-scale analytic anchor bank

At scales $s\in\{8,4\}$, downsample sparse depth with a valid-only aggregation to obtain $(\bar S_s,M_s)$. For every radius or dilation $r\in\mathcal R_s$, compute

$$
A_{s,r}(p)=
\frac{
\sum_{q\in\mathcal N_r(p)}M_s(q)\bar S_s(q)
}{
\sum_{q\in\mathcal N_r(p)}M_s(q)+\epsilon
},
$$

$$
V_{s,r}(p)=
\mathbb 1\!\left[\sum_{q\in\mathcal N_r(p)}M_s(q)>0\right].
$$

The anchor bank is analytic and parameter free. Invalid proposals are masked, never represented as a valid zero-depth anchor. Recommended starting banks are

| Scale | Radius bank | Log-depth correction bound |
|---|---|---:|
| $1/8$ | $\{3,7,15,31\}$ | $\tau_8=0.30$ |
| $1/4$ | $\{1,3,7,15\}$ | $\tau_4=0.15$ |

These values are hyperparameters, not fixed claims.

## 5.3 Multi-modal stems and local-global encoder

The three shallow streams are

$$
F_I^0=\psi_I(I),\qquad
F_D^0=\psi_D(X_D),\qquad
F_G^0=\psi_G(X_G).
$$

Projected additive fusion limits feature width:

$$
F^0=W_IF_I^0+g_D\odot W_DF_D^0+g_G\odot W_GF_G^0,
$$

$$
[g_D,g_G]=\sigma\!\left(h_{\mathrm{mod}}
[F_I^0,F_D^0,F_G^0,M,d_M]\right).
$$

The encoder uses local convolution at $1/2$ and $1/4$, and lightweight global-context blocks only at $1/8$ and $1/16$.

| Variant | Typical width | Encoder role |
|---|---:|---|
| GeoRT-SAR-S | 24--32 / 40--48 / 64--80 / 96--128 | runtime-first, MobileViTv2-0.5 or width-matched mobile CNN |
| GeoRT-SAR-M | wider $1/8$ and $1/16$ stages | accuracy-oriented, MobileViTv2-0.75 |

The backbone is an efficiency choice rather than the method contribution.

## 5.4 Scale-Equivariant Multi-Scale Anchor Routing

At scale $s$, a lightweight router consumes decoder feature $F_s$, the proposal bank, proposal validity, camera rays, and distance to the nearest sparse point:

$$
\ell_{s,r}(p)=h_{\mathrm{route},s}
\left(F_s,\log(A_{s,r}+\epsilon),V_{s,r},\mathbf r_s,d_{M,s}\right).
$$

Invalid radii receive exactly zero probability:

$$
\alpha_{s,r}(p)=
\frac{V_{s,r}(p)\exp(\ell_{s,r}(p))}
{\sum_{r'\in\mathcal R_s}V_{s,r'}(p)\exp(\ell_{s,r'}(p))+\epsilon}.
$$

The routed anchor target is

$$
z_{A,s}(p)=\sum_{r\in\mathcal R_s}
\alpha_{s,r}(p)\log(A_{s,r}(p)+\epsilon).
$$

Given the current normalized log-depth state $z_s$, a per-pixel gate performs one bounded update:

$$
g_s(p)=\sigma(h_{g,s}(F_s)),
$$

$$
z_s^+(p)=z_s(p)+g_s(p)
\operatorname{clip}(z_{A,s}(p)-z_s(p),-\tau_s,\tau_s).
$$

If all proposals at a pixel are invalid, the update is defined as zero. The corrected state is projected back into the decoder feature before the next top-down stage. SE-MSAR performs exactly one update at $1/8$ and one at $1/4$ in the main configuration. It has no recurrent hidden state, learned iterative affinity propagation, or solver.

## 5.5 Low-frequency metric decoder

The additive decoder is

$$
P_{16}=\delta_{16}(E_{16}),
$$

$$
P_8=\delta_8(E_8)+\operatorname{Up}_2(P_{16}),
\qquad
(P_8^+,z_8^+)=\operatorname{Route}_8(P_8,A_8),
$$

$$
P_4=\delta_4(E_4)+\operatorname{Up}_2(P_8^+),
\qquad
(P_4^+,z_4^+)=\operatorname{Route}_4(P_4,A_4).
$$

The coarse heads predict

$$
z_{1/4}=h_D(P_4^+),
\qquad
s_{1/4}=\operatorname{softplus}(h_C(P_4^+)),
$$

$$
D_{1/4}=m_S\exp(z_{1/4}),
\qquad
C_{1/4}=\exp(-s_{1/4}).
$$

This branch is responsible for global metric structure, planar regions, range, and coarse object layout.

## 5.6 Scale-Preserving Boundary Residual Pyramid

The low-frequency metric state is upsampled in log space:

$$
z_{\mathrm{metric}}=\operatorname{Up}_4(z_{1/4}).
$$

At $s\in\{1/2,1\}$, a narrow boundary head predicts a bounded raw residual

$$
r_s^{\mathrm{raw}}=a_s\tanh\!\left(
h_{R,s}[F_{I,s},F_{\mathrm{dec},s},z_{\mathrm{up},s},C_{\mathrm{up},s}]
\right)
$$

and an edge gate

$$
E_s=\sigma\!\left(
h_{E,s}[F_{I,s},|\nabla I_s|,|\nabla z_{\mathrm{up},s}|,C_{\mathrm{up},s}]
\right).
$$

Define the local high-pass operator

$$
\mathcal H_k(x)=x-\operatorname{AvgPool}_k(x).
$$

The boundary residual is

$$
r_s=\mathcal H_k(E_s\odot r_s^{\mathrm{raw}}),
$$

and the final normalized log-depth is

$$
z_{\mathrm{full}}=
\operatorname{Up}_4(z_{1/4})+
\operatorname{Up}_2(r_{1/2})+r_1.
$$

Before sensor correction,

$$
D_{\mathrm{full}}=m_S\exp(z_{\mathrm{full}}).
$$

The high-pass projection constrains the full-resolution path to local structural correction and reduces metric-scale drift; it does not mathematically guarantee zero global bias under finite windows, padding, and learned gates.

The default starting amplitudes in log-depth space are

$$
a_{1/2}=0.10,\qquad a_1=0.05.
$$

Both residual levels use depthwise-separable heads with 8--16 channels. Full-resolution attention and wide standard convolutions are excluded from the main student.

## 5.7 Adaptive sparse anchor correction

The main correction mode predicts a bounded sensor trust coefficient:

$$
\lambda_M(p)=0.5+0.4\,
\sigma\!\left(h_M[C_{\mathrm{full}}(p),d_M(p),|\nabla I(p)|]\right),
$$

$$
D_{\mathrm{out}}(p)=D_{\mathrm{full}}(p)+
\lambda_M(p)M(p)[S(p)-D_{\mathrm{full}}(p)].
$$

Thus $\lambda_M(p)\in[0.5,0.9]$. None, fixed-soft, adaptive-soft, and hard replacement are explicit ablations. Hard replacement gives zero anchor error but may amplify sensor noise or calibration artifacts.

---

# 6. Training Objective

## 6.1 Metric losses

Ground-truth supervision is

$$
\mathcal L_{\mathrm{gt}}=
\frac{1}{|\Omega_{\mathrm{gt}}|}
\sum_{p\in\Omega_{\mathrm{gt}}}
\rho(D(p)-D_{\mathrm{gt}}(p)).
$$

Sparse consistency is

$$
\mathcal L_S=
\frac{1}{|\Omega_M|}
\sum_{p\in\Omega_M}|D(p)-S(p)|.
$$

DMD3C distillation is

$$
\mathcal L_{\mathrm{DMD}}=
\frac{
\sum_{p\in\Omega_{\mathrm{DMD}}}
C_{\mathrm{DMD}}(p)
\rho(D(p)-D_{\mathrm{DMD}}(p))
}{
\sum_{p\in\Omega_{\mathrm{DMD}}}C_{\mathrm{DMD}}(p)+\epsilon
}.
$$

GT, sparse, and DMD3C terms are computed independently. A compatibility map that stores GT where valid and DMD3C elsewhere may be used for I/O, but it is not the definition of the metric teacher and must not silently double-count GT in $\mathcal L_{\mathrm{DMD}}$.

Coarse $1/4$ supervision mirrors the metric terms after valid-only downsampling and is weighted by $\lambda_{\mathrm{aux}}$.

## 6.2 Geometry losses

The geometry target is fitted to the current student log-depth without allowing it to set metric scale:

$$
(\alpha^\star,\beta^\star)=
\arg\min_{\alpha,\beta}
\sum_{p\in\Omega_G}C_G(p)
\rho(\log(D(p)+\epsilon)-\alpha R_G^\star(p)-\beta).
$$

The fit is detached before regression:

$$
\mathcal L_{\mathrm{SSI}}=
\frac{
\sum_{p\in\Omega_G}C_G(p)
\rho(\log(D(p)+\epsilon)-\alpha^\star R_G^\star(p)-\beta^\star)
}{
\sum_{p\in\Omega_G}C_G(p)+\epsilon
}.
$$

For edge-neighbor pairs $\mathcal P$, the ordinal sign is corrected for the fitted orientation:

$$
y_{pq}=\operatorname{sign}\!\left(
\alpha^\star[R_G^\star(p)-R_G^\star(q)]
\right).
$$

The ordinal loss is

$$
\mathcal L_{\mathrm{ord}}=
\frac1{|\mathcal P|}
\sum_{(p,q)\in\mathcal P}
\log\!\left(1+\exp\{-y_{pq}[\log D(p)-\log D(q)]\}\right).
$$

## 6.3 Boundary losses

Define a dilated edge band

$$
\Omega_E=\operatorname{Dilate}
\left(|\nabla R_G^\star|>\tau_G
\ \lor\ |\nabla D_{\mathrm{gt}}|>\tau_D\right).
$$

The edge-band loss is

$$
\mathcal L_{\mathrm{edge}}=
\frac1{|\Omega_E|}
\sum_{p\in\Omega_E}\rho(D(p)-D_{\mathrm{sup}}(p)).
$$

At full and half resolution, log-depth gradients are supervised by

$$
\mathcal L_{\nabla}=
\sum_{s\in\{1,1/2\}}
\left(
|\partial_xz_s-\partial_xz_{T,s}|+
|\partial_yz_s-\partial_yz_{T,s}|
\right).
$$

$z_T$ uses GT where valid and otherwise a confidence-masked teacher target. Invalid target gradients are excluded.

## 6.4 Scale-equivariance loss

Sample $\beta\sim\operatorname{LogUniform}(0.5,2.0)$ and evaluate

$$
D_1=F(I,S,M,K),\qquad
D_\beta=F(I,\beta S,M,K).
$$

The consistency loss is

$$
\mathcal L_{\mathrm{eq}}=
\frac1{|\Omega|}
\sum_p\left|
\log(D_\beta(p)+\epsilon)-
\log(\beta D_1(p)+\epsilon)
\right|.
$$

Pixels affected by $d_{\min}$ or $d_{\max}$ clipping are excluded. This loss detects paths that accidentally consume raw, unnormalized metric depth.

## 6.5 Confidence and optional 3D consistency

Let $s(p)=-\log(C_{\mathrm{full}}(p)+\epsilon)$. A heteroscedastic regression term is

$$
\mathcal L_C=
\frac1{|\Omega_{\mathrm{sup}}|}
\sum_{p\in\Omega_{\mathrm{sup}}}
\left[
e^{-s(p)}\rho(D(p)-D_{\mathrm{sup}}(p))+lambda_s s(p)
\right].
$$

Confidence calibration is evaluated separately with risk-coverage, AUSE/AURG, NLL, and error-confidence correlation. If calibration is not demonstrated, $C_{\mathrm{full}}$ is called a reliability map.

Optional 3D consistency back-projects depth without adding an inference head:

$$
X(p)=D(p)K^{-1}[u,v,1]^\top,
$$

$$
\mathcal L_{3D}=
\frac1{|\Omega|}
\sum_p\rho(X_{\mathrm{student}}(p)-X_{\mathrm{DMD}}(p)).
$$

## 6.6 Total loss

$$
\begin{aligned}
\mathcal L={}&
\lambda_{\mathrm{gt}}\mathcal L_{\mathrm{gt}}
+\lambda_S\mathcal L_S
+\lambda_{\mathrm{DMD}}\mathcal L_{\mathrm{DMD}}
+\lambda_{\mathrm{aux}}\mathcal L_{1/4}\\
&+\lambda_{\mathrm{SSI}}\mathcal L_{\mathrm{SSI}}
+\lambda_{\mathrm{ord}}\mathcal L_{\mathrm{ord}}
+\lambda_{\mathrm{edge}}\mathcal L_{\mathrm{edge}}
+\lambda_{\nabla}\mathcal L_{\nabla}\\
&+\lambda_{\mathrm{eq}}\mathcal L_{\mathrm{eq}}
+\lambda_{3D}\mathcal L_{3D}
+\lambda_C\mathcal L_C.
\end{aligned}
$$

Recommended starting values are

| Term | Weight |
|---|---:|
| $\lambda_{\mathrm{gt}}$ | 1.0 |
| $\lambda_S$ | 1.0 |
| $\lambda_{\mathrm{DMD}}$ | 0.3 |
| $\lambda_{\mathrm{aux}}$ | 0.2 |
| $\lambda_{\mathrm{SSI}}$ | 0.03 |
| $\lambda_{\mathrm{ord}}$ | 0.03 |
| $\lambda_{\mathrm{edge}}$ | 0.10 |
| $\lambda_{\nabla}$ | 0.03 |
| $\lambda_{\mathrm{eq}}$ | 0.02 |
| $\lambda_{3D}$ | 0.00; test 0.02 |
| $\lambda_C$ | 0.03 |

These are initialization points for the ablation protocol, not final tuned values.

---

# 7. Training and Augmentation Protocol

Training follows a curriculum.

| Stage | Fraction of steps | Active components |
|---|---:|---|
| A: metric core | 20--30% | normalization, anchor bank, SE-MSAR, coarse decoder, $\mathcal L_{\mathrm{gt}}+\mathcal L_S+\mathcal L_{\mathrm{DMD}}$ |
| B: geometry and boundary | 50--60% | add SP-BRP, SSI, ordinal, edge-band, and gradient losses |
| C: confidence and robustness | final 20% | add confidence, equivariance, sparse corruption, and optional 3D loss; reduce LR |

Recommended optimization uses AdamW, AMP, gradient clipping, cosine or OneCycle decay, and EMA. Comparisons must use equal optimization steps, identical splits and pseudo labels, and at least three seeds for finalists.

Geometry-safe transforms update $K$:

$$
f_x'=s_xf_x,\quad f_y'=s_yf_y,\quad
c_x'=s_xc_x-x_0,\quad c_y'=s_yc_y-y_0.
$$

For a horizontal flip,

$$
c_x'=W-1-c_x.
$$

Sparse-pattern augmentation includes point dropout, simulated LiDAR line counts, local holes, foreground point removal, range-dependent noise, outliers, small RGB-LiDAR misalignment, and random metric scale. RGB augmentation may include photometric jitter, light blur, fog/rain simulation, and darkening. Every geometric transform is applied consistently to RGB, sparse depth, mask, GT, teachers, and intrinsics.

---

# 8. Inference and Deployment Contract

Inference is exactly

$$
(I,S,M,K)\rightarrow(D_{\mathrm{out}},C_{\mathrm{full}}).
$$

The main GeoRT-SAR-S configuration contains

- per-sample sparse median normalization;
- analytic anchor banks at $1/8$ and $1/4$;
- one bounded SE-MSAR update at each of those scales;
- a lightweight local-global encoder and additive FPN;
- SP-BRP at $1/2$ and full resolution;
- adaptive sparse correction;
- a full-resolution reliability head.

It contains zero iterative SPN steps and no teacher-side component. The design target is 5--10M parameters with only 8--16 channels in the full-resolution path. Real-time status is established only by end-to-end batch-1 measurements on declared hardware, including normalization, anchor-bank construction, correction, and data transfers used in deployment.

---

# 9. Repository Contract

This document is the normative V2 method definition. The checked-in Python code currently provides the executable reference baseline used to construct ablations; it does not yet implement every V2 equation.

| Repository component | Current executable contract | V2 experiment contract |
|---|---|---|
| `src/model_geort.py` | local/KNN sparse propagation, concat stems, sparse-ray injection, LiteFPN, guided convex upsampling, ordinary full-resolution residual, fixed soft correction | median normalization, anchor bank, SE-MSAR, SP-BRP, adaptive correction require explicit modules and flags |
| `src/losses.py` | GT, sparse, confidence-weighted composite metric target, coarse auxiliary, range, SSI, ordinal, confidence, smoothness | separate raw DMD loss plus edge-band, gradient, equivariance, and optional 3D terms must be enabled explicitly |
| `src/teacher_fusion.py` | can include DMD3C in geometry fusion | main V2 geometry target must use DA2 and optional Metric3D only; DMD3C agreement is leave-one-out |
| `configs/teacher.yaml` | `geometry_prior_dmd3c: 0.25` | use zero/no DMD3C geometry candidate for the main V2 run and regenerate geometry pseudo labels |
| `D_cm`, `C_cm` files | compatibility composite with GT priority and DMD3C elsewhere | I/O convenience only; GT, sparse, and DMD3C retain distinct loss semantics |

Consequently, a checkpoint produced by the current default configuration is named the **reference GeoRT-Student-S baseline**, not GeoRT-SAR. The GeoRT-SAR name is reserved for runs that enable both SE-MSAR and SP-BRP and satisfy the inference contract above.

The ablation protocol in `docs/GeoDistill_RT_ablation.md` defines the implementation checks, configuration matrix, metrics, and selection rule required before a configuration is called the final architecture.

---

# 10. Contributions

GeoDistill-RT is defined by five claims that must each be supported by controlled experiments:

1. **Separated teacher roles.** DMD3C is the only dense metric teacher; GT and sparse LiDAR remain direct supervision; DA2 and optional Metric3D provide relative geometry.
2. **Conflict-aware geometry distillation.** Sparse, normal, edge, range, and leave-one-out agreement signals determine pixel-wise teacher reliability without a self-referential DMD3C loop.
3. **Scale-equivariant sparse anchor routing.** SE-MSAR routes between analytic real-sensor proposals and applies one bounded correction at each of two scales.
4. **Scale-preserving boundary reconstruction.** SP-BRP uses bounded edge-gated high-pass residuals to improve local structure while limiting metric-scale drift.
5. **Teacher-free non-iterative inference.** The deployed student uses only $(I,S,M,K)$ and contains no teacher, normal model, iterative propagation, or optimization solver.

The target positioning is therefore:

> A sparse depth-completion framework that combines DMD3C-only metric distillation and conflict-aware relative-geometry supervision with a scale-equivariant anchor-routed student. The student performs one-shot multi-scale metric anchoring and high-pass boundary refinement without teachers or iterative propagation at inference.
