# AnchorFlow V10.1 — LiteMetric technical contract

**Status:** triển khai và kiểm thử local; GPU training accuracy/latency **chưa đo**. V10 gốc được giữ nguyên. Input: RGB, sparse depth, mask, intrinsics; teacher chỉ có trong training objective.

## 1. Graph

~~~text
RGB ── MobileNetV4 ImageNet ── F4 / F8 / F16 / F32
                                     │           │
Sparse + mask + K ── compact pyramid  │       Context32
                    │                │           │
                    └── gated fusion + LiteFPN ───┘
                                     │
                                  D16 → D8 → D0 at 1/4
                                     │
                          minmod quadratic inverse-depth jet
                                     │
                         shared feedback vector field ×2
                         learned h per sample, projection
                                     │
                                   j², D4
                                     │
                        phase lift 4→2 + five-jet consensus
                        uncertainty gate + sparse innovation
                                     │
                                    D2
                                     │
 P4 ─ PW48→12 ─ DW3 ─ zero-init PW12→24 ─ nearest 2×
                                     ├── final phase sampler 2→1
                                     └── final detail / sensor trust
                                                      │
                                           learned soft sensor fusion
                                                      │
                                                    Dfull
~~~

Không thêm branch encoder, adaptive ODE solver, GT ở inference, grid_sample, full-resolution learned feature map, stop policy hoặc ConvGRU.

## 2. Tensor contract

| Tensor | Channels | Spatial,352×1216 |
|---|---:|---|
| RGB features F4 /F8 /F16 /F32 | encoder-reported | 88×304 /44×152 /22×76 /11×38 |
| Sparse features,4/8/16 | 16/16/16 | corresponding scale |
| Fused decoder widths4/8/16 | 48/64/96 | corresponding scale |
| P4 | 48 | 88×304 |
| Guidance G2 /packed G4 | 8/32 | 176×608 /88×304 |
| Dynamic context /jet state | 32/6 | 88×304 |
| Phase2 query hidden | 24 | 88×304 |
| Added context Q4 /Q2 | 24/24 | 88×304 /176×608 |
| Final phase hidden /detail hidden | 16/24 | 176×608 |
| D16 /D8 /D4 /D2 /D1 /Dfull | 1 | native scale |

Images must be divisible by32 for this encoder/context contract. Depth unit **m**, model bounds **0.1–120m**, unchanged. K supplies sparse ray-coordinate features; jet transport operates in quarter-grid coordinates, not physical-time motion.

## 3. State-dependent geometric dynamics

At each quarter pixel, define inverse-depth quadratic jet:

$$
j_p=[v_p,g_{x,p},g_{y,p},h_{xx,p},h_{xy,p},h_{yy,p}],
\qquad v_p=D_p^{-1}.
$$

Initial gradients/Hessian use continuous minmod on detached inverse depth. The value component retains depth gradients.

$$
\mathrm{minmod}(a,b)=
\max(0,\min(a,b))-\max(0,\min(-a,-b)).
$$

The learned field receives current jet, initial value, sparse metric/inverse innovation, support/density/spread/reliability, RGB/sparse context and pseudo-time. Its forcing and conductance are recomputed with **shared weights at both steps**:

$$
F_\theta(j,Z,S,\tau)_p
=R_\theta(j_p,Z_p,S_p,\tau)
+\sum_{q\in\mathcal N_4(p)}
w_{pq,\theta}(j,Z,S)\,[T_{q\to p}(j_q)-j_p].
$$

$$
j^{k+1}=\Pi\bigl(j^k+h_k F_\theta(j^k,Z,S,\tau_k)\bigr),
\qquad k\in\{0,1\}.
$$

**Fixed number of steps, learned step size**: exactly two field evaluations, no stop decision or adaptive solver. The head shares the current field feature Hk (32channels) and predicts one scalar per sample, not a single global model parameter or a per-pixel timestep:

$$
h_k=\frac16+\left(\frac13-\frac16\right)
\sigma\left(\mathrm{Mean}_{x,y}(W_hH_k+b_h)\right),
\qquad \tau_0=0,\quad \tau_{k+1}=\tau_k+h_k.
$$

This is a **learned-step projected Euler unroll**, not adaptive continuous NODE/adjoint training or an error-controlled ODE solver. No normalization forces the two steps to sum to one: terminal pseudo-time is learned within [1/3,2/3]. Current accumulated time, not k/3, enters the next field. Fresh head weight/bias=0 gives h1=h2=0.25, with non-saturated sigmoid gradients.

Quadratic change of origin, displacement (x,y):

$$
T(j;x,y)=
\begin{bmatrix}
v+g_x\,x+g_y\,y+\tfrac12h_{xx}x^2+h_{xy}xy+\tfrac12h_{yy}y^2\\
g_x+h_{xx}x+h_{xy}y\\
g_y+h_{xy}x+h_{yy}y\\
h_{xx}\\h_{xy}\\h_{yy}
\end{bmatrix}.
$$

Two learned east/south edge rates produce reciprocal E/W/S/N conductance, with zero outward boundary flux. Rates≤0.6; h≤1/3 yields neighbour mass≤0.8 for the diffusion-like part. This does **not** prove contraction of the full jet/reaction system.

Projection:

$$
v\in[1/120,10],\qquad
|g_x|,|g_y|\le0.5v,\qquad
|h_{xx}|,|h_{xy}|,|h_{yy}|\le0.25v.
$$

CNNs use native BF16 when available; analytic state, transport, reciprocal and reductions use FP32. No BN is introduced inside the recurrent shared field.

## 4. Keep geometric consensus, strengthen final readout

Quarter→half keeps5 candidate jets (center/E/W/S/N),4 subpixel phases, bounded logits, depth compatibility and learned barrier cost. The softmax inverse-depth consensus supplies geometry; dispersion attenuates its blend gate. Local sparse-innovation mean/std/support helps metric correction.

New final bypass:

$$
Q_4=W_2\bigl(\mathrm{DW}_3(\mathrm{PW}_{48\to12}(P_4))\bigr),
\qquad
Q_2=\mathrm{NearestUp}_2(Q_4).
$$

PW/DW intermediate blocks include BN/SiLU as implemented. W2 is a12→24 projection whose weight **and bias are zero-init**.

$$
H_{\mathrm{sample}}'=H_{\mathrm{sample}}+Q_2[:,0:16],
\qquad
H_{\mathrm{detail}}'=H_{\mathrm{detail}}+Q_2.
$$

The shared latent24 channels therefore condition both the final3×3 phase weights/residual and detail/trust. Channel slicing is a cheap learned shared-basis constraint, not a second CNN. Only1.044 added parameters and quarter-grid convolutions; nearest resize creates a half-grid context tensor but **no full-grid feature CNN**.

Final sampler remains convex metric-depth aggregation:

$$
D_{1,\mathrm{base}}=
\mathrm{PixelShuffle}_2\!
\left[
\sum_{q=1}^{9}\pi_qD_{2,q}
+(0.5+0.02\sum_{q=1}^{9}\pi_qD_{2,q})\tanh r
\right].
$$

Detail and sensor fusion are unchanged:

$$
D_1=
\mathrm{clip}\bigl(D_{1,\mathrm{base}}
+(0.5+0.05D_{1,\mathrm{base}})\tanh h,\ 0.1,\ 120\bigr).
$$

$$
t=(0.5+0.02D_1)e^{\,\mathrm{clip}(a,-1,2)},\quad
g=M\,\sigma\bigl(\ell-\log(1+((S-D_1)/t)^2)\bigr),
\quad D_{\mathrm{full}}=(1-g)D_1+gS.
$$

Do not replace this with hard anchor: V10 hard diagnostic RMSE1.135m vs soft0.996m. Hard output is logged separately, never used to claim default accuracy.

## 5. Targeted objective

All active terms are computed directly, no chain of legacy loss wrappers. GT Huber1m scale weights remain:

$$
L_{\mathrm{metric}}=
0.025L_{16}+0.05L_8+0.15L_4+0.30L_2+0.50L_1+L_{\mathrm{full}}.
$$

Let V be the valid GT pixels and ε=10⁻⁶. Full-output metric/inverse terms:

$$
L_R=
\sqrt{\frac1{|V|}\sum_{p\in V}(D_p-D_{\mathrm{gt},p})^2+\varepsilon}
-\sqrt\varepsilon,
$$

$$
L_I=
\sqrt{\frac1{|V|}\sum_{p\in V}
\left(\frac{1000}{D_p}-\frac{1000}{D_{\mathrm{gt},p}}\right)^2+\varepsilon}
-\sqrt\varepsilon.
$$

LI unit is **km⁻¹**, exactly the evaluation residual with a smooth root. No hidden near-depth clipping, no inverse of pooled GT as the primary inverse objective.

$$
L=L_{\mathrm{metric}}
+0.6L_R+0.04\,r_I L_I
+0.1L_{\mathrm{range}}
+0.25\,r_T L_{\mathrm{tail}}
+0.05L_{\mathrm{boundary}}
+0.01L_{\mathrm{barrier}}
+0.1L_{\mathrm{dyn}}
+0.02L_{\mathrm{sparse}}
+0.05L_{\mathrm{holdout}}
+0.02L_{\mathrm{trust}}
+\lambda_{\mathrm{KD}}L_{\mathrm{KD}}
+0.015\,r_{\mathrm{rel}}L_{\mathrm{rel}}.
$$

Ramps use continuous epoch progress e:

$$
r_I=\min(1,e/4),\quad r_T=\min(1,e/2),\quad
r_{\mathrm{rel}}=\min(1,e/3),\quad
\lambda_{\mathrm{KD}}=0.012(1-0.5\min(1,e/E)).
$$

| Term | Definition |
|---|---|
| Range | Same five GT bins0–20/20–40/40–60/60–80/80–120; minimum support64, far sparse-bin vote as baseline |
| Tail | All-valid normalized2×Huber10 of max(|error|−2m,0); no top-k/sort |
| Boundary | Full depth RMSE in3px band of observed GT discontinuities, not RGB texture alone |
| Barrier | Balanced BCE on clean quarter-GT edge targets |
| Dynamics | Mean GT-Huber over actual steps1/2, not ghost3/4 |
| Sparse/holdout/trust | Same GT-priority reliability targets and10% sensor holdout |
| Metric KD | Cached D_cm/C_cm; exclude every original sensor/GT pixel, plus blocked coarse cells |
| Relative | Cached R_T/C_T, eligible unobserved local normalized inverse-depth gradients; no ordinal/normal |

Default skips tiny GT-log/GT-edge/teacher-edge computations and disabled robustMSE. **Low magnitude alone is not proof of low gradient value**; these remain configurable for control40. Disabled relative/teacher variants compute no such KD loss.

At cold initialization, phase and detail output heads are zero-init. The added zero-init context branch receives its first useful gradient **after the original phase/detail heads open**, verified by a two-update contract test. Do not require nonzero new-context gradient on the first fresh smoke forward.

## 6. Training, evaluation and selection

Fresh40: AdamW3e−4, encoder LR0.5×, new context1×, weight decay1e−5, warm1epoch/cosine to0.05×, B4, clip1, freeze encoder BN, all weights fine-tuned. Early stop min20/patience8/Δ0.001m, monitor RMSE.

Optional fine20: V10best23 model-only; optimizer/RNG/schedule new, LR1e−4, encoder0.25×, new context and step head2×, min8/patience5. Controller keys are the only allowed removed keys; context/step-head keys the only allowed new keys. New step-head bias=4 gives h≈0.33034 to limit parent drift. This is near saturation, so fresh training uses bias0 instead. Initial output is **not promised bit-identical**; notebook logs initial400val before learning.

Validation/test have **no teacher inputs**, same preprocessing/crop/GT support. Global metrics accumulate pixel sums, not mean image RMSE. Log coarse-native stage metrics,0–5/5–10/10–20 inverse diagnostics, range/boundary/tail and soft/hard outputs.

Three distinct validation-selected checkpoints are retained:

$$
\mathrm{best}_{\mathrm{joint}}=
\arg\min_e\max\left(
\frac{\mathrm{RMSE}_e}{0.9},
\frac{\mathrm{iRMSE}_e}{3.2}
\right).
$$

Do not report best-RMSE from one epoch and best-iRMSE from another as a joint result. Default evaluate/test/profile/export consistently use minimum-RMSE **best.pth**. Explicit alternate checkpoint selection is recorded.

Resume locks source/config/subset/teacher hashes and restores optimizer/scaler/global_step/RNG/early-stop/metric-selection state. Changed recipes require new run name; no silent source/precision fallback.

## 7. Sources and scope of novelty

[DFU, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Improving_Depth_Completion_via_Depth_Feature_Upsampling_CVPR_2024_paper.html) supports the principle of reusing dense context to guide depth upsampling. The lightweight shared phase-context bypass here is **an engineering adaptation**, not DFU reproduction or a proven new SOTA mechanism.

[Neural ODE, NeurIPS2018](https://arxiv.org/abs/1806.07366) motivates state evolution by a vector field. This implementation is a fixed projected Euler unroll, **not an adaptive NODE solver**. Novelty should be claimed at the geometric shared jet field/readout level only after matched ablations.

[KITTI evaluation](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) defines iRMSE in1/km and metric depth RMSE inmm. Internal reports here use **m** for metric depth and **km⁻¹** for inverse depth. Internal400val is not the public leaderboard.

Canonical files: **model.py/model_base.py/geometry.py**, **losses.py/relative_loss.py**, **run.py/data.py**, configs and the notebook. Historical V10 source is not edited.
