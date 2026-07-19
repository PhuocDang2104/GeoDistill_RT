# GeoLift-RT v2.1

## Role-Separated Geometry Distillation with Phase-Structured Affine Inverse-Depth Lifting for Real-Time Depth Completion

> **Framework:** GeoLift-RT  
> **Student:** GeoLift-S2  
> **Training strategy:** RSGD — Role-Separated Geometry Distillation  
> **Progressive decoder:** RayLift-ID — Phase-Structured Affine Inverse-Depth Lifting  
> **High-resolution guidance:** PPG — Phase-Packed Guidance

---

## 1. Mục tiêu thiết kế

GeoLift-S2 giải bài toán **RGB-guided sparse depth completion**:

\[
(I,S,M,K)\longrightarrow (D_{\mathrm{full}},C_{\mathrm{full}}),
\]

trong đó:

- \(I\in\mathbb{R}^{B\times3\times H\times W}\): ảnh RGB;
- \(S\in\mathbb{R}^{B\times1\times H\times W}\): sparse metric depth;
- \(M\in\{0,1\}^{B\times1\times H\times W}\): validity mask;
- \(K\in\mathbb{R}^{B\times3\times3}\): camera intrinsics;
- \(D_{\mathrm{full}}\): dense metric depth;
- \(C_{\mathrm{full}}\in(0,1)\): reliability map.

Bản v2.1 ưu tiên bốn yêu cầu:

1. Learned processing bắt đầu chủ yếu tại \(1/4\), không xây feature rộng ở full resolution.
2. Decoder không dùng propagation lặp, recurrent refinement hay full-resolution attention.
3. Geometry transport không materialize ray tensor hoặc 3D point tensor cho từng candidate.
4. Kiến trúc đủ đơn giản để triển khai và ablate tuần tự bằng PyTorch trước khi viết custom operator.

Thiết kế được chia thành hai phần rõ ràng:

- **RSGD:** chỉ dùng khi training để truyền metric, relative-geometry và local-surface knowledge.
- **GeoLift-S2:** student chạy độc lập khi inference.

---

## 2. Ý tưởng cốt lõi

GeoLift-S2 không tạo high-resolution depth bằng một phép resize duy nhất. Model thực hiện:

\[
\boxed{
\text{coarse metric inference}
\rightarrow
\text{progressive }2\times\text{ reconstruction}
\rightarrow
\text{sparse metric anchoring}
}
\]

Tại mỗi stage, RayLift-ID:

1. giữ riêng bốn target subpixel phases;
2. chọn một số source inverse-depth hypotheses bằng stencil có cấu trúc;
3. vận chuyển từng hypothesis sang target ray bằng local affine inverse-depth geometry;
4. tổng hợp các hypotheses;
5. trộn kết quả với bilinear baseline bằng update gate.

Lõi toán học:

\[
\boxed{
\widetilde\xi_i
=
\xi_i
+a_\phi(x_t-x_i)
+b_\phi(y_t-y_i)
}
\]

với \(\xi=1/z\). Đây là biểu diễn affine của inverse depth trên một local 3D plane trong mô hình pinhole sử dụng z-depth.

---

## 3. Kiến trúc tổng thể

Với input \(H\times W=352\times1216\):

| Scale | Resolution |
|---:|---:|
| Full | \(352\times1216\) |
| \(1/2\) | \(176\times608\) |
| \(1/4\) | \(88\times304\) |
| \(1/8\) | \(44\times152\) |
| \(1/16\) | \(22\times76\) |

```text
Input: RGB I, sparse depth S, mask M, intrinsics K
  │
  ├─ valid-aware sparse downsample + compact sparse prior at 1/4
  ├─ RGB strided stem → X_I at 1/4
  └─ analytic ray coordinates (x,y) at 1/4
          │
          ▼
  Fusion at 1/4, width 32
          │
          ▼
  Stage-adapted MobileViTv2-0.75
  F4, F8, F16
          │
          ▼
  Additive LiteFPN, width 24
  P16, P8, P4
          │
          ├──────────── PPG from raw high-resolution cues at 1/2
          │                         │
          ▼                         ├─ G2 for 4→2
  Initial metric head              └─ Q2_phase for 2→1
  D16, C16
          │
          ▼
  RayLift-ID 16→8, K=5
          │
  RayLift-ID 8→4,  K=3
          │
  RayLift-ID 4→2,  K=3
          │
  RayLift-ID 2→1,  K=2
          │
          ▼
  Sparse metric anchoring
          │
          ▼
  D_full, C_full
```

### Main test configuration

| Component | Configuration |
|---|---|
| Input | \(352\times1216\) |
| RGB stem | 24 channels at \(1/4\) |
| Sparse/prior stem | 16 channels at \(1/4\) |
| Analytic geometry | \(x,y\), 2 channels |
| Fusion | \(42\rightarrow32\) at \(1/4\) |
| Encoder | stage-adapted MobileViTv2-0.75 |
| FPN width | 24 |
| Decoder samples | \(K=(5,3,3,2)\) |
| Iterative propagation | none |
| Teacher at inference | none |

Một deployment encoder thuần convolution có thể được thử sau khi decoder đã ổn định; nó không thuộc graph kiểm chứng đầu tiên.

---

# Part I — Compact Metric Front-End

## 4. Camera-ray coordinates

Với pixel homogeneous coordinate:

\[
\widetilde p=
\begin{bmatrix}
u\\v\\1
\end{bmatrix},
\]

camera ray cho z-depth là:

\[
r(p)=K^{-1}\widetilde p
=
\begin{bmatrix}
x\\y\\1
\end{bmatrix},
\]

trong đó:

\[
x=\frac{u-c_x}{f_x},
\qquad
y=\frac{v-c_y}{f_y}.
\]

Nếu depth là camera-axis z-depth:

\[
P(p)=z(p)r(p).
\]

GeoLift-S2 chỉ materialize hai map \(x,y\) tại các scale cần thiết. Thành phần \(r_z=1\) là hằng số và không cần lưu.

> **Scale convention:** mọi công thức RayLift-ID trong tài liệu này giả sử z-depth. Nếu dataset dùng Euclidean range, phải chuyển representation nhất quán trước khi dùng affine inverse-depth transport.

---

## 5. Compact sparse prior tại \(1/4\)

### 5.1 Valid-aware downsampling

Sparse depth không được downsample bằng average pooling thông thường. Với mỗi cell \(1/4\):

\[
S_4(p)
=
\frac{
\sum_{q\in\mathcal B_4(p)}M(q)S(q)
}{
\sum_{q\in\mathcal B_4(p)}M(q)+\epsilon
},
\]

\[
M_4(p)=\mathbb 1\left[\sum_{q\in\mathcal B_4(p)}M(q)>0\right].
\]

Có thể thay mean trong cell bằng nearest valid sample nếu dataset yêu cầu bảo toàn scan pattern.

### 5.2 Local normalized prior

Main model chỉ dùng một radius đủ lớn để tránh nhiều cue trùng lặp:

\[
D^{\mathrm{init}}_4(p)
=
\frac{
\sum_{q\in\mathcal N_7(p)}k(p,q)M_4(q)S_4(q)
}{
\sum_{q\in\mathcal N_7(p)}k(p,q)M_4(q)+\epsilon
}.
\]

Validity và density:

\[
V^{\mathrm{init}}_4(p)
=
\mathbb 1
\left[
\sum_{q\in\mathcal N_7(p)}M_4(q)>0
\right],
\]

\[
\rho_4(p)=\operatorname{AvgPool}_7(M_4)(p).
\]

Depth-branch input:

\[
X_D^0=
[S_4,M_4,D^{\mathrm{init}}_4,V^{\mathrm{init}}_4,\rho_4].
\]

Recommended stem:

```text
1×1 Conv: 5→16
DWConv 3×3
PWConv 16→16
```

### 5.3 Optional ablation

Near-biased softmin prior có thể được thử riêng nếu foreground/background bleeding còn lớn:

\[
D^{\mathrm{softmin}}_4
=
-\tau\log
\frac{
\sum_q M_4(q)e^{-S_4(q)/\tau}
}{
\sum_qM_4(q)+\epsilon
}.
\]

Nó không nằm trong main graph để tránh tăng complexity và độ nhạy với outlier trước khi có bằng chứng thực nghiệm.

---

## 6. True \(1/4\) front-end

### 6.1 RGB stem

```text
Conv 3×3, stride 2, 3→12
Rep/DWConv 3×3, stride 2, 12→12
PWConv 1×1, 12→24
```

\[
X_I\in\mathbb R^{B\times24\times H/4\times W/4}.
\]

Chỉ convolution đầu tiên đọc full-resolution RGB; output lập tức được giảm xuống \(1/2\).

### 6.2 Depth stem

Toàn bộ learned depth processing chạy tại \(1/4\):

\[
X_D=\psi_D(X_D^0)
\in\mathbb R^{B\times16\times H/4\times W/4}.
\]

### 6.3 Analytic geometry channels

\[
X_R=[x_4,y_4]
\in\mathbb R^{B\times2\times H/4\times W/4}.
\]

Không dùng learned ray stem trong main graph.

### 6.4 Fusion

\[
X_0=\operatorname{Concat}[X_I,X_D,X_R]
\in\mathbb R^{B\times42\times H/4\times W/4}.
\]

\[
Y=\operatorname{Conv}_{1\times1}^{42\rightarrow32}(X_0),
\]

\[
F_0
=
Y+
\operatorname{PW}_{32}
\left(
\operatorname{RepDW}_{3\times3}(Y)
\right).
\]

Fusion residual giúp khởi tạo ổn định và có thể reparameterize khi deployment.

---

# Part II — Low-Resolution Metric Encoder

## 7. Stage-adapted encoder

Encoder nhận tensor đã ở \(1/4\):

```text
F0 at 1/4, 32ch
  → adapter/stage at 1/4 = F4
  → stride-2 stage       = F8
  → stride-2 stage       = F16
```

Output:

\[
F_4,F_8,F_{16}.
\]

Không được đưa \(F_0\) vào một `features_only` wrapper mặc định rồi lấy các output có `reduction=[4,8,16]`, vì như vậy scale thực so với ảnh gốc có thể trở thành \(1/16,1/32,1/64\).

Main test dùng stage-adapted MobileViTv2-0.75 để chỉ thay front-end và decoder trước. Việc thay backbone được giữ cho ablation sau.

### Vai trò từng scale

- \(F_4\): local sparse alignment, small objects và boundaries;
- \(F_8\): object-level structure;
- \(F_{16}\): global context và coarse metric geometry.

Không thêm một geometry-injection block riêng trong main graph. Sparse và analytic ray cues đã được fuse tại \(1/4\), còn RayLift-ID nhận geometry trực tiếp trong decoder. Affine modulation chỉ nên được thử như một ablation nếu encoder mất thông tin metric sau downsampling.

---

## 8. Additive LiteFPN

Dùng width thống nhất:

\[
C_P=24.
\]

\[
P_{16}=\delta_{16}(F_{16}),
\]

\[
P_8
=
\delta_8(F_8)+\operatorname{Up}_2(P_{16}),
\]

\[
P_4
=
\delta_4(F_4)+\operatorname{Up}_2(P_8).
\]

Chỉ smooth output cao nhất:

\[
P_4
\leftarrow
P_4+
\operatorname{RepDW}_{3\times3}(P_4).
\]

LiteFPN chỉ decode feature. Metric depth được decode bởi RayLift-ID.

---

## 9. Initial metric head

Recommended first implementation giữ metric-depth head để cô lập thay đổi decoder:

\[
H_{16}
=
\operatorname{PW}
\left(
\operatorname{DW}_{3\times3}(P_{16})
\right),
\]

\[
D_{16}
=
\operatorname{softplus}(h_D(H_{16}))+d_{\min},
\]

\[
C_{16}=\sigma(h_C(H_{16})).
\]

Trong RayLift-ID:

\[
\xi_{16}=\frac{1}{D_{16}+\epsilon}.
\]

Chỉ chuyển coarse head sang direct inverse-depth prediction sau khi normal-form và inverse-depth decoder đã được so sánh công bằng.

---

# Part III — Phase-Packed Guidance

## 10. Mục tiêu của PPG

RayLift-ID cuối cùng chạy tại source scale \(1/2\), nên không cần tạo một learned feature map \(G_1\) ở full resolution.

PPG thực hiện:

\[
\boxed{
\text{raw full-resolution cues}
\rightarrow
\text{PixelUnshuffle}
\rightarrow
\text{phase-preserving projection at }1/2
}
\]

Bốn full-resolution children được giữ tách biệt trước learned mixing.

---

## 11. PPG input và projection

Dùng raw guidance tối giản:

\[
X_H=
[I,M,M\log(S+\epsilon),E_I],
\]

trong đó \(E_I\) là fixed RGB gradient magnitude:

\[
E_I=\sqrt{\|\partial_xI\|_2^2+\|\partial_yI\|_2^2+\epsilon}.
\]

Tổng số channel:

\[
C_H=3+1+1+1=6.
\]

PixelUnshuffle:

\[
U_2
=
\operatorname{PixelUnshuffle}_2(X_H)
\in
\mathbb R^{B\times24\times H/2\times W/2}.
\]

Dùng grouped pointwise convolution với bốn groups:

\[
Q_2^{\mathrm{phase}}
=
\operatorname{GroupPW}_{4}(U_2)
\in
\mathbb R^{B\times16\times H/2\times W/2}.
\]

Mỗi phase có bốn learned channels trước khi cross-phase mixing.

Tạo guidance gọn cho stage \(1/4\rightarrow1/2\):

\[
G_2
=
\operatorname{PW}_{16\rightarrow12}
\left(
\operatorname{DW}_{3\times3}(Q_2^{\mathrm{phase}})
\right).
\]

Sử dụng:

- \(G_2\) cho RayLift-ID \(4\rightarrow2\);
- \(Q_2^{\mathrm{phase}}\) trực tiếp cho RayLift-ID \(2\rightarrow1\).

Ray coordinates không đưa vào PPG vì chúng được tính analytic trong RayLift-ID.

---

# Part IV — RayLift-ID Decoder

## 12. Block definition

Một RayLift-ID block là:

\[
(D_h,C_h)
=
\operatorname{RayLiftID}_l
(D_l,C_l,F_l,G_h,K),
\]

với target resolution gấp đôi source resolution.

| Stage | Source feature | Target guidance | \(K\) |
|---|---|---|---:|
| \(1/16\rightarrow1/8\) | \(P_{16}\) | \(P_8\) | 5 |
| \(1/8\rightarrow1/4\) | \(P_8\) | \(P_4\) | 3 |
| \(1/4\rightarrow1/2\) | \(P_4\) | \(G_2\) | 3 |
| \(1/2\rightarrow1\) | compact source state | \(Q_2^{\mathrm{phase}}\) | 2 |

Depth is sampled directly from source inverse depth. Không tạo một target depth map tạm rồi chạy refinement network riêng.

---

## 13. Subpixel phases

Mỗi source parent tạo bốn target children:

\[
\phi\in\Phi=
\{(0,0),(1,0),(0,1),(1,1)\}.
\]

Với `align_corners=False`, base coordinate của child \(\phi=(\phi_x,\phi_y)\) trên source grid:

\[
q_{p,\phi}
=
p+
\begin{bmatrix}
\phi_x/2-1/4\\
\phi_y/2-1/4
\end{bmatrix}.
\]

Bốn offsets tương đối là:

\[
(-0.25,-0.25),
(0.25,-0.25),
(-0.25,0.25),
(0.25,0.25).
\]

Mỗi child có riêng:

- sample weights;
- affine slopes \((a_\phi,b_\phi)\);
- planarity gate \(\eta_\phi\);
- update gate \(g_\phi\).

---

## 14. Shared parameter head

Target guidance được đưa về source resolution khi cần:

\[
G_h'=\operatorname{PixelUnshuffle}_2(G_h).
\]

Input head:

\[
X_l=
\operatorname{Concat}
[F_l,G_h',D_l,C_l,x_l,y_l].
\]

Ở hai stage cuối, source state có thể được project hẹp từ \([D_l,C_l,G_2]\) thay vì giữ một wide full-resolution feature.

Shared trunk:

\[
Z_l^{24}
=
\phi\left(
\operatorname{Conv}_{1\times1}^{C_{\mathrm{in}}\rightarrow24}(X_l)
\right),
\]

\[
Z_l^{\mathrm{dw}}
=
\phi\left(
\operatorname{DWConv}_{3\times3}(Z_l^{24})
\right),
\]

\[
Z_l^{16}
=
\phi\left(
\operatorname{Conv}_{1\times1}^{24\rightarrow16}(Z_l^{\mathrm{dw}})
\right).
\]

Spatial \(3\times3\) convolution chỉ chạy một lần và được chia sẻ cho mọi output head.

---

## 15. Structured source-space sampling

### 15.1 Stage \(16\rightarrow8\): cross stencil

\[
\mathcal R_5
=
\{(0,0),(-1,0),(1,0),(0,-1),(0,1)\}.
\]

Predict parent-shared:

\[
\Theta_p=(t_x,t_y,\theta,\ell_{\parallel},\ell_{\perp}).
\]

\[
t_p=t_{\max}\tanh([t_x,t_y]),
\]

\[
a_{\parallel}
=a_{\min}+(a_{\max}-a_{\min})\sigma(\ell_{\parallel}),
\]

\[
a_{\perp}
=a_{\min}+(a_{\max}-a_{\min})\sigma(\ell_{\perp}).
\]

Sample coordinate:

\[
s_{p,\phi,i}
=
q_{p,\phi}+t_p+
\mathcal R(\theta_p)
\begin{bmatrix}
a_{\parallel,p}&0\\
0&a_{\perp,p}
\end{bmatrix}r_i.
\]

### 15.2 Stages \(8\rightarrow4\) and \(4\rightarrow2\): oriented line

\[
\mathcal R_3=\{(-1,0),(0,0),(1,0)\}.
\]

Chỉ cần:

- bounded translation;
- orientation;
- one radius.

Không cần anisotropic cross ở các scale này trong main graph.

### 15.3 Stage \(2\rightarrow1\): base plus one routed neighbor

\[
K=2.
\]

Candidates:

1. phase-base coordinate \(q_{p,\phi}\);
2. one routed neighbor:

\[
s_{p,\phi,2}
=q_{p,\phi}
+r_{p,\phi}
\begin{bmatrix}
\cos\theta_{p,\phi}\\
\sin\theta_{p,\phi}
\end{bmatrix}.
\]

Stage cuối chỉ cần một alternate-surface hypothesis. \(K=1\) được giữ cho speed ablation, không phải main model.

---

## 16. Direct inverse-depth sampling

Source inverse depth:

\[
\xi_l=\frac{1}{D_l+\epsilon}.
\]

Candidate:

\[
\xi_{p,\phi,i}
=
\operatorname{Bilinear}
(\xi_l,s_{p,\phi,i}).
\]

Không bilinear-sample ray map. Với source sampling coordinate \((u_i,v_i)\), normalized ray coordinates được tính analytic:

\[
x_i=\frac{u_i-c_x^{(l)}}{f_x^{(l)}},
\qquad
y_i=\frac{v_i-c_y^{(l)}}{f_y^{(l)}}.
\]

Target child coordinates \((x_t,y_t)\) được tính tương tự ở target scale.

---

## 17. Affine inverse-depth transport

Một 3D plane:

\[
n^\top X=\delta.
\]

Với:

\[
X=z[x,y,1]^\top,
\]

inverse depth trên plane là:

\[
\xi(x,y)=ax+by+c,
\]

trong đó:

\[
a=\frac{n_x}{\delta},
\qquad
b=\frac{n_y}{\delta},
\qquad
c=\frac{n_z}{\delta}.
\]

RayLift-ID dự đoán child-specific slopes:

\[
a_\phi=s_l\tanh(\widehat a_\phi),
\qquad
b_\phi=s_l\tanh(\widehat b_\phi),
\]

với \(s_l\) là bound phụ thuộc scale.

Transport source candidate sang target ray:

\[
\boxed{
\widetilde\xi_{p,\phi,i}
=
\xi_{p,\phi,i}
+a_{p,\phi}(x_t-x_i)
+b_{p,\phi}(y_t-y_i)
}.
\]

Clamp:

\[
\widetilde\xi_{p,\phi,i}
\leftarrow
\operatorname{clamp}
(\widetilde\xi_{p,\phi,i},\xi_{\min},\xi_{\max}).
\]

Không cần:

- sampled 3D ray tensor;
- explicit \(P=dr\);
- normalized surface-normal output;
- 3D dot products cho mỗi candidate;
- division bởi \(n^\top r_t\).

---

## 18. Planarity-aware fallback

Không phải vùng nào cũng phù hợp với local plane. Dự đoán:

\[
\eta_{p,\phi}=\sigma(h_\eta(Z_l^{16})_{p,\phi}).
\]

Blend trong inverse-depth domain:

\[
\widehat\xi_{p,\phi,i}
=
(1-\eta_{p,\phi})\xi_{p,\phi,i}
+
\eta_{p,\phi}\widetilde\xi_{p,\phi,i}.
\]

- \(\eta\approx0\): standard dynamic inverse-depth sampling;
- \(\eta\approx1\): affine plane transport.

Cơ chế này tránh ép local-plane assumption tại foliage, reflective surfaces, highly curved regions và depth discontinuities.

---

## 19. Candidate aggregation và residual update

Raw logits:

\[
a_{p,\phi,i}^{\alpha}
=h_\alpha(Z_l^{16})_{p,\phi,i}.
\]

Weights:

\[
\alpha_{p,\phi,i}
=
\frac{
\exp(a_{p,\phi,i}^{\alpha})
}{
\sum_j\exp(a_{p,\phi,j}^{\alpha})
}.
\]

Aggregated candidate:

\[
\overline\xi_{p,\phi}
=
\sum_i
\alpha_{p,\phi,i}
\widehat\xi_{p,\phi,i}.
\]

Bilinear base:

\[
\xi_{p,\phi}^{\mathrm{base}}
=
\operatorname{Bilinear}(\xi_l,q_{p,\phi}).
\]

Update gate:

\[
g_{p,\phi}=\sigma(h_g(Z_l^{16})_{p,\phi}).
\]

Output:

\[
\xi_h(p,\phi)
=
(1-g_{p,\phi})\xi_{p,\phi}^{\mathrm{base}}
+
g_{p,\phi}\overline\xi_{p,\phi}.
\]

\[
D_h(p,\phi)=\frac{1}{\xi_h(p,\phi)+\epsilon}.
\]

Bốn child tensors được rearrange bằng PixelShuffle:

\[
[B,4,H_l,W_l]
\rightarrow
[B,1,2H_l,2W_l].
\]

---

## 20. Reliability from candidate consensus

Dispersion:

\[
V_{p,\phi}^{\xi}
=
\sum_i
\alpha_{p,\phi,i}
\left|
\widehat\xi_{p,\phi,i}
-
\overline\xi_{p,\phi}
\right|.
\]

Normalized dispersion:

\[
\widetilde V_{p,\phi}^{\xi}
=
\frac{
V_{p,\phi}^{\xi}
}{
|\overline\xi_{p,\phi}|+\epsilon
}.
\]

Inherited confidence:

\[
C_{p,\phi}^{\mathrm{base}}
=
\operatorname{Bilinear}(C_l,q_{p,\phi}).
\]

Intermediate-stage confidence không cần learned logit riêng:

\[
C_h(p,\phi)
=
C_{p,\phi}^{\mathrm{base}}
\exp(-\gamma_l\widetilde V_{p,\phi}^{\xi}).
\]

Tại final stage có thể thêm một learned calibration factor:

\[
C_1
\leftarrow
C_1\,\sigma(h_C(Z_2^{16})).
\]

Thiết kế này giảm output channels ở ba stage đầu và giữ confidence liên kết trực tiếp với candidate agreement.

---

## 21. Initialization

Khởi tạo gần bilinear upsampling:

\[
t=0,
\qquad
\theta=0,
\qquad
\text{radius}=1,
\]

\[
a_\phi=b_\phi=0,
\qquad
\eta_\phi\approx0,
\qquad
g_\phi\approx0.05,
\]

\[
\alpha_{\phi,i}\approx\frac{1}{K}.
\]

Khi bắt đầu training:

\[
D_h\approx\operatorname{Bilinear}(D_l).
\]

Model học routing và geometry correction dần, tránh instability do random offsets hoặc random slopes.

---

# Part V — Output Anchoring

## 22. Sparse metric anchoring

Main test giả sử sparse sensor đã được benchmark protocol xem là metric measurement hợp lệ. Dùng hard anchoring đơn giản:

\[
D_{\mathrm{full}}(p)
=
(1-M(p))D_1(p)+M(p)S(p).
\]

Reliability output:

\[
C_{\mathrm{full}}(p)
=
(1-M(p))C_1(p)+M(p).
\]

Ưu điểm:

- exact sparse consistency;
- không thêm outlier head trước khi cần;
- dễ đối chiếu với baseline;
- không che giấu lỗi pre-anchor của network.

Luôn báo thêm pre-anchor sparse error:

\[
\mathcal E_S^{\mathrm{pre}}
=
\frac{1}{|\Omega_M|}
\sum_{p\in\Omega_M}
|D_1(p)-S(p)|.
\]

Adaptive anchoring chỉ nên được thêm sau nếu dataset chứa outlier hoặc RGB–LiDAR misalignment đáng kể.

---

# Part VI — RSGD Training

## 23. Teacher roles

Main training setup dùng ba teacher roles:

| Source | Role |
|---|---|
| Strong metric depth-completion teacher (e.g. a DMD3C checkpoint) | dense metric structure |
| Depth Anything V2 | relative layout, ordinal relation, fine boundaries |
| DSINE | local plane slopes và planarity reliability |

Primary supervision:

- ground-truth metric depth;
- real sparse LiDAR.

Metric3D v2 được giữ như một **optional geometry-consensus ablation**, không nằm trong main training graph. Việc bỏ teacher thứ tư giúp giảm cache, alignment logic và teacher conflict trước khi chứng minh có gain.

---

## 24. Conflict-aware metric-teacher reliability

Metric teacher không được tin đồng đều trên mọi pixel. DMD3C ở đây chỉ một strong offline checkpoint được dùng như teacher, không phải một module chạy trong student.

Sparse consistency:

\[
E_S(p)
=
M(p)
\frac{
|D_T^{\mathrm{metric}}(p)-S(p)|
}{
S(p)+\epsilon
}.
\]

Relative-geometry agreement sau scale-shift alignment:

\[
E_G(p)
=
\left|
\log(D_T^{\mathrm{metric}}(p)+\epsilon)
-
\widetilde R_{\mathrm{DA2}}(p)
\right|.
\]

Metric-teacher weight:

\[
W_T(p)
=
\operatorname{clip}
\left[
\exp(-E_S(p)/\tau_S)
\exp(-E_G(p)/\tau_G)
W_E(p),
0,1
\right],
\]

trong đó \(W_E\) giảm weight tại vùng teacher halo hoặc RGB/depth edge conflict.

Không dùng DMD3C để tự tạo geometry consensus rồi quay lại chấm confidence cho chính nó.

---

## 25. Slope supervision from normal teacher

Nếu DSINE normal là \(n\), và metric anchor tại pixel là:

\[
P=d[x,y,1]^\top,
\]

plane offset:

\[
\delta=n^\top P.
\]

Teacher slopes:

\[
a_T=\frac{n_x}{\delta},
\qquad
b_T=\frac{n_y}{\delta}.
\]

Nếu \(n\rightarrow-n\), đồng thời \(\delta\rightarrow-\delta\), nên \(a_T,b_T\) không đổi. Slope supervision không có normal-sign ambiguity.

Chỉ tạo target khi:

- metric anchor hợp lệ;
- \(|\delta|>\epsilon_\delta\);
- DSINE reliability đủ cao;
- local normal variance không quá lớn.

Planarity target:

\[
\eta_T(p)
=
\exp
\left[
-\kappa
\operatorname{Var}_{q\in\mathcal N(p)}N_T(q)
\right].
\]

---

## 26. Training objective

Giữ objective đủ nhỏ để dễ tune:

\[
\boxed{
\begin{aligned}
\mathcal L
={}&
\lambda_{\mathrm{gt}}\mathcal L_{\mathrm{gt}}
+
\lambda_S\mathcal L_S^{\mathrm{pre}}
+
\lambda_T\mathcal L_{\mathrm{KD}}^{\mathrm{metric}}
\\
&+
\lambda_G\mathcal L_{\mathrm{rel}}
+
\lambda_E\mathcal L_{\mathrm{edge}}
+
\lambda_{ab}\mathcal L_{ab}
+
\lambda_\eta\mathcal L_\eta
\\
&+
\lambda_{\mathrm{cyc}}\mathcal L_{\mathrm{cycle}}
+
\lambda_C\mathcal L_C.
\end{aligned}
}
\]

### 26.1 Multi-scale metric loss

\[
\mathcal L_{\mathrm{gt}}
=
\sum_{s\in\{16,8,4,2,1\}}
\lambda_s
\frac{1}{|\Omega_s|}
\sum_{p\in\Omega_s}
\rho_H
\left[
\log D_s(p)-\log D_{\mathrm{gt},s}(p)
\right].
\]

### 26.2 Pre-anchor sparse loss

\[
\mathcal L_S^{\mathrm{pre}}
=
\frac{1}{|\Omega_M|}
\sum_{p\in\Omega_M}
|D_1(p)-S(p)|.
\]

### 26.3 Reliability-weighted metric KD

\[
\mathcal L_{\mathrm{KD}}^{\mathrm{metric}}
=
\sum_s
\frac{
\sum_pW_{T,s}(p)
\rho_H
[\log D_s(p)-\log D_{T,s}(p)]
}{
\sum_pW_{T,s}(p)+\epsilon
}.
\]

### 26.4 Relative geometry loss

Use one combined term:

\[
\mathcal L_{\mathrm{rel}}
=
\mathcal L_{\mathrm{SSI}}
+
\lambda_{\mathrm{ord}}^{\mathrm{in}}\mathcal L_{\mathrm{ord}}.
\]

Ordinal pairs tập trung tại teacher-agreement boundaries và thin structures.

### 26.5 Boundary loss

\[
\mathcal L_{\mathrm{edge}}
=
\mathcal L_{\mathrm{boundary\ band}}
+
\lambda_\nabla^{\mathrm{in}}
\mathcal L_\nabla.
\]

Chỉ áp dụng mạnh tại \(D_2,D_1\).

### 26.6 Slope và planarity

\[
\mathcal L_{ab}
=
\sum_{h,p,\phi}
W_{ab}(p,\phi)
\operatorname{Huber}
\left(
[a,b]_{p,\phi}-[a_T,b_T]_{p,\phi}
\right),
\]

\[
\mathcal L_\eta
=
\operatorname{BCE}(\eta,\eta_T).
\]

### 26.7 Cross-scale consistency

\[
\mathcal L_{\mathrm{cycle}}
=
\sum_{h=2l}
\left\|
\operatorname{VDown}_2(D_h)-D_l
\right\|_1.
\]

### 26.8 Confidence calibration

Chỉ calibrate final reliability:

\[
u(p)=-\log(C_1(p)+\epsilon),
\]

\[
e(p)=
|\log D_1(p)-\log D_{\mathrm{gt}}(p)|,
\]

\[
\mathcal L_C
=
\frac{1}{|\Omega|}
\sum_{p\in\Omega}
[e(p)e^{-u(p)}+u(p)].
\]

### Starting weights

| Loss | Initial weight |
|---|---:|
| \(\lambda_{\mathrm{gt}}\) | 1.0 |
| \(\lambda_S\) | 0.2–0.5 |
| \(\lambda_T\) | 0.2–0.4 |
| \(\lambda_G\) | 0.02–0.05 |
| \(\lambda_E\) | 0.05–0.10 |
| \(\lambda_{ab}\) | 0.02–0.05 |
| \(\lambda_\eta\) | 0.01–0.02 |
| \(\lambda_{\mathrm{cyc}}\) | 0.02–0.05 |
| \(\lambda_C\) | 0.01–0.03 |

Các giá trị trên là starting range, không phải benchmark claim.

---

## 27. Training schedule

### Stage A — Metric bootstrap

Bật:

- metric GT;
- pre-anchor sparse loss;
- reliability-weighted metric KD;
- multi-scale outputs.

Khởi tạo:

- update gates gần zero;
- planarity gates gần zero;
- slopes bằng zero.

Tắt hoặc giảm mạnh:

- slope loss;
- ordinal loss;
- confidence calibration.

### Stage B — Phase-geometry learning

Bật:

- PPG;
- relative geometry loss;
- boundary loss;
- slope supervision;
- planarity target;
- cross-scale consistency.

Tăng dần learning rate multiplier hoặc bias cho update gate, không mở geometry correction đột ngột.

### Stage C — Calibration and compression

Bật:

- final confidence calibration;
- sparse dropout/noise augmentation;
- optional teacher-to-student distillation từ \(K=(5,5,3,3)\) sang \(K=(5,3,3,2)\) nếu reduced-K model mất accuracy.

Không cần một deployment-teacher stage riêng nếu main reduced-K model đã đạt acceptance criteria.

---

# Part VII — Implementation and Testing

## 28. Reference implementation contract

### Correctness mode

- static scale definitions;
- vectorized four phases;
- `grid_sample` chỉ cho source inverse depth;
- analytic source/target ray coordinates;
- explicit tensors để unit test;
- FP32 trước, sau đó FP16.

### Không materialize

- sampled ray tensor;
- sampled 3D points;
- full \([B,H,W,4,K,3]\) intermediates;
- large `unfold` patches;
- full-resolution learned guidance map.

### Fused operator chỉ viết khi cần

Chỉ viết custom CUDA/TensorRT plugin nếu profile cho thấy:

\[
\frac{T_{\mathrm{RayLiftID}}}{T_{\mathrm{full}}}>25\%-30\%.
\]

Trước đó ưu tiên:

- static shapes;
- `torch.compile` nếu ổn định;
- ONNX Runtime;
- TensorRT built-in operators.

---

## 29. Unit tests bắt buộc

### 29.1 Normal-form equivalence

Sinh random valid plane:

\[
n^\top X=\delta.
\]

So sánh:

1. normal-form ray-plane intersection;
2. affine inverse-depth transport.

FP32 requirement:

\[
\max
|d_{\mathrm{normal}}-d_{\mathrm{ID}}|
<10^{-4}
\]

cho các denominator hợp lệ.

### 29.2 Phase-coordinate test

Kiểm tra bốn offsets:

\[
(-0.25,-0.25),
(0.25,-0.25),
(-0.25,0.25),
(0.25,0.25).
\]

### 29.3 PixelUnshuffle ordering

Đặt bốn constants khác nhau ở bốn phases và kiểm tra đúng thứ tự channel của framework đang dùng.

### 29.4 Sparse-prior validity

Không có valid sparse sample trong neighborhood thì:

\[
V_4^{\mathrm{init}}=0
\]

và prior không được xem là metric observation hợp lệ.

### 29.5 Bilinear initialization

Khi:

\[
a=b=0,
\quad
\eta=0,
\quad
g=0,
\]

output phải bằng bilinear inverse-depth baseline trong sai số số học.

### 29.6 Anchor test

Tại \(M=0\):

\[
D_{\mathrm{full}}=D_1.
\]

Tại \(M=1\):

\[
D_{\mathrm{full}}=S.
\]

---

## 30. Minimal ablation plan

Không thay backbone và decoder đồng thời.

### Architecture ablation

| ID | Configuration |
|---|---|
| A0 | current student/reference graph |
| A1 | true \(1/4\) front-end + analytic \(x,y\) |
| A2 | A1 + PPG, remove full-resolution \(G_1\) |
| A3 | A2 + RayLift-ID, \(K=(5,5,3,3)\) |
| A4 | A3 + reduced \(K=(5,3,3,2)\) — main GeoLift-S2 |
| A5 | A4 + convolutional deployment encoder |

### Distillation ablation

| ID | Supervision |
|---|---|
| T0 | GT + sparse only |
| T1 | + reliability-weighted DMD3C KD |
| T2 | + DA2 relative/boundary supervision |
| T3 | + DSINE slope/planarity supervision — full RSGD |
| T4 | + Metric3D v2 consensus, optional |

Main paper comparison cần tách:

1. architecture gain without distillation;
2. distillation gain on the same architecture;
3. full GeoLift-RT result.

---

## 31. Acceptance criteria

### Main graph

Chấp nhận thay đổi khi:

\[
\Delta\mathrm{RMSE}_{\mathrm{global}}
\leq0.5\%-1.0\%
\]

và:

\[
\Delta\mathrm{latency}\leq-20\%.
\]

Boundary metrics không giảm đáng kể.

### Speed graph

Với \(K=(5,3,3,1)\) hoặc deployment encoder hẹp hơn:

\[
\Delta\mathrm{RMSE}_{\mathrm{global}}
\leq1.5\%,
\]

latency giảm ít nhất \(30\%\) so với main graph.

Các threshold này là engineering gates.

---

## 32. Metrics and benchmark protocol

### Accuracy

- RMSE, MAE, iRMSE, iMAE;
- REL, \(\delta_1\) nếu protocol dùng;
- near/mid/far-range RMSE;
- pre-anchor sparse MAE.

### Boundary

- boundary-band RMSE 3 px và 5 px;
- depth-gradient MAE;
- thin-object error;
- foreground/background ordinal accuracy.

### Reliability

- error–confidence correlation;
- risk–coverage;
- AUSE/AURG.

### Runtime

- batch size 1;
- fixed \(352\times1216\);
- median và P95 latency;
- peak allocated memory;
- component timing;
- FP32 và FP16;
- full pipeline gồm sparse prior và output anchoring.

Không suy speedup từ MAC đơn thuần. Bilinear sampling, tensor layout, kernel launch và memory traffic phải được profile trên target runtime.

---

# Part VIII — Novelty Positioning

## 33. Contributions nên claim

### Contribution 1 — RSGD

**Role-Separated, Conflict-Aware Geometry Distillation**:

- metric teacher dạy dense metric structure;
- relative teacher dạy layout, ordinal relation và boundaries;
- normal teacher dạy affine inverse-depth slopes và planarity;
- teacher reliability được xử lý theo pixel.

### Contribution 2 — Phase-Structured RayLift-ID

Một progressive reconstruction operator kết hợp:

\[
\boxed{
\text{phase-packed target guidance}
+
\text{structured source sampling}
+
\text{child-specific affine inverse-depth transport}
+
\text{planarity fallback}
+
\text{one-pass gated reconstruction}
}
\]

Mọi routing parameters được dự đoán tại lower source resolution.

### Deployment realization

True \(1/4\) fusion, analytic ray coordinates, reduced stage-wise \(K\), no sampled-ray tensor và no full-resolution learned feature là các **deployment design choices**, không nên tách thành novelty claims riêng.

---

## 34. Những gì không claim là mới riêng lẻ

Không claim novelty từ:

- PixelShuffle/PixelUnshuffle;
- dynamic sampling;
- inverse depth của một plane là affine theo ray coordinates;
- ray-plane intersection;
- local planar guidance;
- knowledge distillation;
- sparse anchoring;
- MobileViTv2 hoặc Rep/UIB-style blocks.

Tính mới cần được bảo vệ ở **tổ hợp operator và teacher–student correspondence**, không phải từng primitive.

---

## 35. Positioning với related directions

### So với Local Planar Guidance

GeoLift-S2 không chỉ predict một plane rồi render toàn bộ local block. Nó:

- sample nhiều source metric hypotheses;
- route samples bằng structured stencil;
- dùng child-specific slopes;
- có scalar/plane fallback;
- tái dựng progressive trong depth-completion setting.

### So với DySample và dynamic upsampling

RayLift-ID không chỉ học sampling coordinates. Mỗi sampled inverse-depth hypothesis còn được transport từ source ray sang target ray bằng camera-aware local geometry trước khi aggregation.

### So với DMD3C

DMD3C là một depth-completion distillation framework có thể cung cấp strong offline checkpoint. GeoLift-RT không xem DMD3C là một primitive mới trong student; nó dùng một strong metric checkpoint như một role của RSGD và tập trung novelty kiến trúc vào phase-aware geometric reconstruction.

### So với DSINE

DSINE cung cấp ray-aware normal knowledge offline. GeoLift-S2 chuyển normal teacher output thành sign-invariant affine slope targets; DSINE không chạy khi inference.

---

## 36. Central claim

> **GeoLift-RT progressively reconstructs metric depth by routing source inverse-depth hypotheses to individual target subpixels and transporting them through child-specific affine ray geometry, while all learned routing is performed at the lower source resolution.**

Không dùng từ “first” trước systematic literature và patent search.

---

# Part IX — Final Test Specification

## 37. GeoLift-S2 main configuration

```text
Input
  RGB I                       [B,3,352,1216]
  sparse depth S              [B,1,352,1216]
  mask M                      [B,1,352,1216]
  intrinsics K                [B,3,3]

Compact sparse prior at 1/4
  valid-aware S4, M4
  normalized local prior D_init4, radius 7
  validity V_init4
  density rho4

Front-end at 1/4
  RGB stem                    24ch
  sparse/prior stem           16ch
  analytic ray coordinates     2ch
  concat                      42ch
  residual fusion             42→32

Encoder
  stage-adapted MobileViTv2-0.75
  F4, F8, F16

Feature decoder
  additive LiteFPN            24ch
  P16, P8, P4

Phase-Packed Guidance
  raw cues                    RGB, M, MlogS, RGB edge
  PixelUnshuffle at 1/2
  grouped phase projection    Q2_phase, 16ch
  compact guidance            G2, 12ch
  no learned full-resolution G1

Initial depth
  D16, C16

RayLift-ID
  16→8                        K=5, structured cross
  8→4                         K=3, oriented line
  4→2                         K=3, oriented line
  2→1                         K=2, base + routed neighbor

RayLift-ID trunk
  1×1 projection              24ch
  DWConv 3×3                  24ch
  1×1 projection              16ch

Per-child predictions
  sample weights
  inverse-depth slopes a,b
  planarity gate
  update gate
  final-stage confidence calibration

Output
  hard sparse metric anchoring
  D_full, C_full

Inference exclusions
  no teacher
  no foundation model
  no recurrent propagation
  no full-resolution attention
  no full-resolution learned guidance
  no sampled ray map
  no explicit sampled 3D points
  no large unfold
```

---

## 38. Recommended implementation order

1. Fix the true \(1/4\) scale contract.
2. Implement compact sparse prior and analytic \(x,y\).
3. Train front-end + current decoder as A1.
4. Implement PPG and remove \(G_1\).
5. Unit-test phase ordering and coordinates.
6. Implement RayLift-ID with \(K=(5,5,3,3)\).
7. Verify analytic equivalence on synthetic planes.
8. Train normal-form versus inverse-depth form under the same protocol.
9. Reduce to \(K=(5,3,3,2)\).
10. Add full RSGD only after architecture-only gains are known.
11. Replace encoder only after decoder v2 is stable.
12. Write a fused plugin only if profiling confirms RayLift-ID is the remaining bottleneck.

---

## 39. One-sentence summary

> **GeoLift-S2 removes unnecessary full-resolution learned processing and reconstructs depth progressively by phase-separated, structured sampling and child-specific affine inverse-depth transport, while RSGD supplies metric, relative and local-surface knowledge only during training.**

---

## 40. Primary related works to verify during paper writing

- *Separable Self-attention for Mobile Vision Transformers* — MobileViTv2, arXiv:2206.02680.
- *Learning to Upsample by Learning to Sample* — DySample, ICCV 2023, arXiv:2308.15085.
- *From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation* — BTS/LPG, arXiv:1907.10326.
- *Rethinking Inductive Biases for Surface Normal Estimation* — DSINE, CVPR 2024, arXiv:2403.00712.
- *Depth Anything V2* — NeurIPS 2024, arXiv:2406.09414.
- *Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation* — arXiv:2404.15506.
- *Distilling Monocular Foundation Model for Fine-grained Depth Completion* — DMD3C, CVPR 2025, arXiv:2503.16970.
- *MobileNetV4: Universal Models for the Mobile Ecosystem* — ECCV 2024, arXiv:2404.10518.
