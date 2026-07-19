# GeoLift-RT

## Role-Separated Geometry Distillation with Phase-Aware Ray-Plane Lifting for Real-Time Depth Completion

> **Framework name:** GeoLift-RT  
> **Student:** GeoLift-S  
> **Offline training strategy:** RSGD — Role-Separated Geometry Distillation  
> **Progressive decoder block:** RayLift — Phase-Aware Ray-Plane Residual Upsampling

---

## 1. Mục tiêu

GeoLift-RT giải bài toán **RGB-guided sparse depth completion** với hai yêu cầu đồng thời:

1. Dự đoán dense metric depth chính xác, đặc biệt tại biên vật thể, vật thể mỏng và vùng thiếu LiDAR.
2. Giữ inference nhỏ, không lặp, không dùng teacher, không dùng foundation model và không dùng propagation network nhiều bước.

Tại inference, student chỉ nhận:

$$
(I,S,M,K) \longrightarrow (D_{\mathrm{full}},C_{\mathrm{full}}),
$$

trong đó:

- $I\in\mathbb{R}^{B\times3\times H\times W}$: ảnh RGB.
- $S\in\mathbb{R}^{B\times1\times H\times W}$: sparse metric depth.
- $M\in\{0,1\}^{B\times1\times H\times W}$: validity mask.
- $K\in\mathbb{R}^{B\times3\times3}$: camera intrinsics.
- $D_{\mathrm{full}}$: dense metric depth full-resolution.
- $C_{\mathrm{full}}\in(0,1)$: confidence hoặc reliability map.

Teacher chỉ được sử dụng **offline khi training**. Deployment không cần DMD3C, Depth Anything V2, Metric3D v2 hoặc DSINE.

---

## 2. Ý tưởng tổng quát

GeoLift-RT gồm hai phần tương ứng rõ ràng:

### 2.1 RSGD — Role-Separated Geometry Distillation

Teacher được phân vai theo loại tri thức:

- **DMD3C:** dense metric-depth teacher.
- **Depth Anything V2:** relative layout, ordinal structure và high-frequency boundary.
- **Metric3D v2:** optional secondary geometry teacher hoặc geometry diagnostic.
- **DSINE:** surface-normal supervision và normal reliability; không phải depth teacher.

Các teacher không được trộn thành một pseudo-ground-truth duy nhất. Mỗi teacher giám sát đúng thành phần student mà nó có thế mạnh.

### 2.2 GeoLift-S với RayLift decoder

Student thực hiện:

$$
\text{low-resolution metric inference}
\rightarrow
\text{progressive }2\times\text{ reconstruction}
\rightarrow
\text{adaptive sparse anchoring}.
$$

Mỗi RayLift block hợp nhất:

$$
2\times\text{ upsampling}
+
\text{structured adaptive sampling}
+
\text{ray-plane geometry transport}
+
\text{residual refinement}.
$$

Không tồn tại chuỗi tách rời “upsample depth trước rồi chạy ReDC sau”. Depth target-resolution được tái dựng trực tiếp từ source depth trong một stage.

---

## 3. Ký hiệu và kích thước

Với input $H\times W=352\times1216$:

| Scale | Kích thước |
|---:|---:|
| Full | $352\times1216$ |
| $1/2$ | $176\times608$ |
| $1/4$ | $88\times304$ |
| $1/8$ | $44\times152$ |
| $1/16$ | $22\times76$ |

Ký hiệu:

- $F_s$: encoder feature tại scale $1/s$.
- $P_s$: LiteFPN feature tại scale $1/s$.
- $G_s$: shallow high-resolution guidance tại scale $1/s$.
- $D_s,C_s$: metric depth và confidence tại scale $1/s$.
- $R_s$: camera ray map tại scale $1/s$.
- $M_s,\rho_s$: validity mask và local sparse-density map.
- $D^{\mathrm{init}}_s$: analytic sparse-derived metric prior.

---

# Phần I — Offline Teacher Distillation

## 4. Phân vai teacher

### 4.1 Metric teacher

Dense metric teacher duy nhất là:

$$
D_T^{\mathrm{metric}}
=
\operatorname{stopgrad}\!\left(D_{\mathrm{DMD3C}}\right).
$$

Cần phân biệt:

- $D_{\mathrm{gt}}$: supervised ground-truth.
- $S$: real sensor metric anchor.
- $D_T^{\mathrm{metric}}$: dense teacher prediction.

Ground truth và sparse LiDAR không được gọi là teacher.

### 4.2 Relative geometry teachers

Đặt tập geometry teachers:

$$
\mathcal{T}_G
=
\left\{
T_{\mathrm{DA2}},
T_{\mathrm{M3D}}
\right\}.
$$

Mỗi teacher tạo relative geometry representation $R_t$, có thể là relative inverse depth hoặc normalized log-depth. DMD3C không tham gia geometry consensus mặc định để tránh teacher tự đánh giá chính nó.

### 4.3 Surface-normal teacher

DSINE tạo:

$$
N_T\in\mathbb{R}^{B\times3\times H\times W},
\qquad
\|N_T(p)\|_2=1,
$$

và reliability $W_N(p)$. Normal teacher chủ yếu giám sát phase-specific plane prediction trong RayLift.

---

## 5. Geometry alignment và consensus

Relative teacher $R_t$ không được dùng như metric label. Trước tiên, teacher được scale–shift align trong log-depth space trên tập anchor đáng tin cậy $\Omega_A$:

$$
(a_t^\star,b_t^\star)
=
\arg\min_{a,b}
\sum_{p\in\Omega_A}
\omega_A(p)
\left|
aR_t(p)+b-\log D_A(p)
\right|^2,
$$

trong đó $D_A$ có thể lấy từ ground truth hợp lệ, sparse anchors hoặc metric teacher đã qua reliability filtering.

Aligned geometry map:

$$
\widetilde R_t(p)=a_t^\star R_t(p)+b_t^\star.
$$

Geometry consensus:

$$
R_G^\star(p)
=
\sum_{t\in\mathcal{T}_G}
\pi_t(p)\widetilde R_t(p),
\qquad
\sum_t\pi_t(p)=1.
$$

Teacher mixing weights $\pi_t$ được suy ra từ teacher agreement, edge quality và optional normal consistency.

---

## 6. Conflict-aware teacher reliability

### 6.1 Sparse consistency

Tại các sparse points:

$$
E_S(p)=M(p)\left|D_T^{\mathrm{metric}}(p)-S(p)\right|.
$$

Reliability:

$$
W_S(p)
=
\exp\left(
-\frac{E_S(p)}{\tau_S(S(p)+\epsilon)}
\right).
$$

Ngoài sparse positions, $W_S$ có thể được lan cục bộ bằng fixed masked pooling; không sử dụng iterative propagation.

### 6.2 Geometry agreement

Metric teacher được so với geometry consensus không chứa DMD3C:

$$
E_G(p)
=
\left|
\log(D_T^{\mathrm{metric}}(p)+\epsilon)
-
R_G^\star(p)
\right|.
$$

$$
W_G(p)=\exp\left(-\frac{E_G(p)}{\tau_G}\right).
$$

### 6.3 Edge and range reliability

$$
W_E(p)=\exp\left(-\frac{E_{\mathrm{edge}}(p)}{\tau_E}\right),
\qquad
W_R(p)=\exp\left(-\frac{E_{\mathrm{range}}(p)}{\tau_R}\right).
$$

$E_{\mathrm{edge}}$ tăng khi teacher tạo halo hoặc foreground–background bleeding. $E_{\mathrm{range}}$ mô hình hóa vùng far-range hoặc vùng ngoài miền tin cậy của teacher.

### 6.4 Metric-teacher confidence

$$
W_T^{\mathrm{metric}}(p)
=
\operatorname{clip}
\left(
W_S(p)W_G(p)W_E(p)W_R(p),
0,
1
\right).
$$

Công thức này loại bỏ vòng lặp tự tham chiếu: DMD3C không được dùng trong $R_G^\star$ rồi quay lại tự tạo confidence cho chính nó.

---

## 7. Teacher–student correspondence

| Teacher information | Student component được giám sát |
|---|---|
| DMD3C metric depth | $D_{16},D_8,D_4,D_2,D_1$ và global metric structure |
| DA2 relative structure | ordinal relations, boundary layout, RayLift child routing |
| Metric3D v2 | optional secondary geometry consensus |
| DSINE normals | RayLift phase-specific plane normals và planarity gate |
| Sparse LiDAR | metric anchoring và scale preservation |
| Ground truth | primary supervised target |

Sự tương ứng này là nguyên tắc thiết kế chính: teacher supervision phải gắn trực tiếp với subproblem cụ thể của student, thay vì chỉ cộng nhiều dense losses vào output cuối.

---

# Phần II — GeoLift-S Student

## 8. Tổng quan kiến trúc

```text
RGB I ── RGB stem, 24ch ────────────────┐
Sparse S, mask M, D_init ─ depth stem 16ch ─┼─ concat, 52ch
Ray/UV geometry ─ ray stem, 12ch ───────┘
                         │
                         ▼
Efficient fusion
1×1 Conv 52→32 → DWConv 3×3 → PWConv 1×1
                         │
                         ▼
MobileViTv2-0.75 encoder
F4, F8, F16
                         │
                         ▼
Sparse/ray-gated injection
                         │
                         ▼
Additive LiteFPN
P16, P8, P4
                         │
                         ▼
Initial metric head at 1/16
D16, C16
                         │
                         ▼
RayLift 16→8, K=5, guide=P8
                         │
RayLift 8→4, K=5, guide=P4
                         │
RayLift 4→2, K=3, guide=G2
                         │
RayLift 2→1, K=3, guide=G1
                         │
                         ▼
Adaptive sparse anchoring
                         │
                         ▼
D_full, C_full
```

### Main configuration

| Thành phần | Cấu hình |
|---|---|
| Input resolution | $352\times1216$ |
| RGB stem | 24 channels |
| Depth stem | 16 channels |
| Ray/UV stem | 12 channels |
| Fusion width | 32 channels |
| Encoder | MobileViTv2-0.75 |
| Encoder scales | $1/4,1/8,1/16$ |
| Decoder | additive LiteFPN + four RayLift stages |
| RayLift samples | $K=5,5,3,3$ |
| RayLift trunk | $24\rightarrow16$ channels |
| Iterative propagation | 0 iterations |
| Teacher at inference | none |

---

## 9. Camera-ray representation

Với pixel homogeneous coordinate:

$$
\widetilde p=
\begin{bmatrix}
u\\v\\1
\end{bmatrix},
$$

camera ray cho z-depth được định nghĩa:

$$
r(p)=K^{-1}\widetilde p
=
\begin{bmatrix}
x\\y\\1
\end{bmatrix}.
$$

Nếu depth $d$ là camera-axis z-depth:

$$
P(p)=d(p)r(p).
$$

Nếu dataset dùng Euclidean range, phải thay bằng unit ray và chuyển đổi representation nhất quán. Không được trộn z-depth với Euclidean range trong ray-plane transport.

Ray input gồm:

$$
X_R=[r_x,r_y,r_z,u_{\mathrm{norm}},v_{\mathrm{norm}}].
$$

---

## 10. Analytic initialization $D^{\mathrm{init}}$

$D^{\mathrm{init}}_4$ được tạo độc lập từ sparse depth và mask bằng local normalized propagation tại $1/4$:

$$
D^{\mathrm{init}}_4(p)
=
\frac{
\sum_{q\in\mathcal N(p)}
k(p,q)M_4(q)S_4(q)
}{
\sum_{q\in\mathcal N(p)}
k(p,q)M_4(q)+\epsilon
}.
$$

Validity:

$$
V^{\mathrm{init}}_4(p)
=
\mathbb 1
\left[
\sum_{q\in\mathcal N(p)}M_4(q)>0
\right].
$$

Tạo pyramid bằng valid-aware downsampling:

$$
D^{\mathrm{init}}_8
=
\operatorname{VDown}_2
\left(D^{\mathrm{init}}_4,V^{\mathrm{init}}_4\right),
$$

$$
D^{\mathrm{init}}_{16}
=
\operatorname{VDown}_2
\left(D^{\mathrm{init}}_8,V^{\mathrm{init}}_8\right).
$$

Không dùng $D^{\mathrm{init}}_4$ để upsample mạnh lên $1/2$ hoặc full resolution, vì local propagation error có thể trở thành boundary artifact.

---

## 11. Tri-modal stems và efficient fusion

Ba stem đưa modality về cùng scale $1/4$:

$$
X_I=\psi_I(I)\in\mathbb{R}^{B\times24\times H/4\times W/4},
$$

$$
X_D=\psi_D(S,M,D^{\mathrm{init}}_4,V^{\mathrm{init}}_4)
\in\mathbb{R}^{B\times16\times H/4\times W/4},
$$

$$
X_R=\psi_R(R_4,UV_4)
\in\mathbb{R}^{B\times12\times H/4\times W/4}.
$$

Concatenation:

$$
X_0=\operatorname{Concat}(X_I,X_D,X_R)
\in\mathbb{R}^{B\times52\times H/4\times W/4}.
$$

Efficient fusion:

$$
F_0
=
\operatorname{PW}_{32}
\left(
\operatorname{DW}_{3\times3}
\left(
\operatorname{Conv}_{1\times1}^{52\rightarrow32}(X_0)
\right)
\right).
$$

Depthwise convolution xử lý spatial context với chi phí thấp; pointwise convolution trộn channel sau đó.

### Scale contract bắt buộc khi triển khai

Các tensor $X_I,X_D,X_R,X_0$ ở đặc tả GeoLift-S là tensor **$1/4$ thật so với ảnh gốc**. Vì vậy không được đưa $X_0$ vào nguyên trạng một `timm features_only` encoder rồi tiếp tục lấy các output có `reduction=[4,8,16]`; cách đó sẽ tạo nhầm $F_{16},F_{32},F_{64}$ so với ảnh gốc.

Cấu hình mục tiêu phải dùng MobileViTv2 stage adapter:

```text
X0 at 1/4, 32ch
  → adapter 1×1, 32→48 = F4
  → MobileViTv2 stage down 2× = F8
  → MobileViTv2 stage down 2× + low-resolution transformer = F16
```

Nếu dùng wrapper MobileViTv2 chưa được stage-adapt như code A0 hiện tại, ba stem buộc phải giữ full resolution để output encoder vẫn đúng $1/4,1/8,1/16$; khi đó MAC/runtime phải được báo theo graph A0, không được dùng con số ước lượng của GeoLift-S mục tiêu.

---

## 12. MobileViTv2 encoder và gated injection

Encoder tạo:

$$
F_4,F_8,F_{16}.
$$

Tại scale $s\in\{4,8,16\}$, sparse/ray injection:

$$
g_s^D
=
\sigma\left(h_s^D(F_s,M_s,\rho_s)\right),
$$

$$
g_s^R
=
\sigma\left(h_s^R(F_s,R_s)\right),
$$

$$
\widetilde F_s
=
F_s
+
g_s^D\odot\phi_s^D(X_s^D)
+
g_s^R\odot\phi_s^R(R_s).
$$

Mục đích:

- Sparse feature không lấn át RGB trong vùng không có measurements.
- Ray geometry chỉ được đưa vào mức cần thiết.
- Global context vẫn do encoder học ở resolution thấp.

---

## 13. Additive LiteFPN

LiteFPN decode **feature**, không trực tiếp decode depth:

$$
P_{16}=\delta_{16}(\widetilde F_{16}),
$$

$$
P_8
=
\delta_8(\widetilde F_8)
+
\operatorname{Up}_2(P_{16}),
$$

$$
P_4
=
\delta_4(\widetilde F_4)
+
\operatorname{Up}_2(P_8).
$$

Output:

$$
\boxed{\{P_{16},P_8,P_4\}}.
$$

Vai trò:

- $P_{16}$: scene-level geometry và large structures.
- $P_8$: object-level structure.
- $P_4$: localized boundaries và small objects.

LiteFPN và RayLift là hai quá trình khác nhau:

- LiteFPN: progressive **feature decoding**.
- RayLift: progressive **metric-depth decoding**.

---

## 14. Shallow high-resolution guidance

LiteFPN dừng ở $1/4$. Hai scale cao dùng guidance hẹp:

$$
G_1\in\mathbb{R}^{B\times C_1\times H\times W},
\qquad C_1\in[8,12],
$$

$$
G_2\in\mathbb{R}^{B\times C_2\times H/2\times W/2},
\qquad C_2\in[12,16].
$$

Một cấu hình nhẹ:

```text
RGB shallow feature ───────────────┐
Sparse/mask shallow projection ────┼─ concat → 1×1 projection → G1
Ray/UV shallow projection ─────────┘

G1 → DWConv 3×3 stride 2 → PWConv 1×1 → G2
```

Không dùng attention hoặc wide standard convolution ở $1/2$ và full resolution.

---

## 15. Initial metric head tại $1/16$

Từ $P_{16}$:

$$
H_{16}
=
\operatorname{PW}
\left(
\operatorname{DW}_{3\times3}(P_{16})
\right).
$$

Depth:

$$
D_{16}
=
\operatorname{softplus}\left(h_D(H_{16})\right)+d_{\min}.
$$

Confidence:

$$
C_{16}=\sigma\left(h_C(H_{16})\right).
$$

$D_{16}$ là coarse dense metric depth đầu tiên. Nó có global structure nhưng chưa có high-resolution boundary.

---

# Phần III — RayLift Progressive Decoder

## 16. Định nghĩa RayLift

Một RayLift block là hàm:

$$
(D_h,C_h)
=
\operatorname{RayLift}_l
\left(
D_l,C_l,F_l,G_h,R_l,R_h,M_h,\rho_h,D^{\mathrm{init}}_l
\right),
$$

với target scale $h=2l$.

Các stage:

| Stage | Source feature | Target guidance | Samples |
|---|---|---|---:|
| $1/16\rightarrow1/8$ | $P_{16}$ | $P_8$ | $K=5$ |
| $1/8\rightarrow1/4$ | $P_8$ | $P_4$ | $K=5$ |
| $1/4\rightarrow1/2$ | $P_4$ | $G_2$ | $K=3$ |
| $1/2\rightarrow1$ | $G_2$ | $G_1$ | $K=3$ |

RayLift không materialize một high-resolution depth map rồi đọc lại để refinement. Sampling được thực hiện trực tiếp trên source depth $D_l$.

---

## 17. Phase-aware target guidance

Mỗi source pixel tạo bốn target children:

$$
\phi\in\Phi=
\{(0,0),(1,0),(0,1),(1,1)\}.
$$

Target guidance được đưa về source resolution:

$$
G_h'=\operatorname{PixelUnshuffle}_2(G_h).
$$

Do đó head chạy tại $H_l\times W_l$, nhưng vẫn thấy riêng feature của bốn subpixel phases.

Tọa độ source tương ứng với child $\phi=(\phi_x,\phi_y)$ dưới quy ước `align_corners=False`:

$$
q_{p,\phi}
=
p+
\begin{bmatrix}
\phi_x/2-1/4\\
\phi_y/2-1/4
\end{bmatrix}.
$$

Bốn child có base locations lệch $\pm0.25$ source pixel, thay vì cùng dùng một parent coordinate.

---

## 18. Shared lightweight parameter head

Input tại source resolution:

$$
X_l=
\operatorname{Concat}
\left[
F_l,
G_h',
D_l,
C_l,
D^{\mathrm{init}}_l,
M_l,
\rho_l,
R_l,
\operatorname{PixelUnshuffle}_2(R_h)
\right].
$$

Với các stage $1/4\rightarrow1/2$ và $1/2\rightarrow1$, bỏ $D^{\mathrm{init}}_l$ nếu không có prior tương ứng.

Shared trunk:

$$
Z_l^{24}
=
\phi\left(
\operatorname{Conv}_{1\times1}^{C_{\mathrm{in}}\rightarrow24}(X_l)
\right),
$$

$$
Z_l^{\mathrm{dw}}
=
\phi\left(
\operatorname{DWConv}_{3\times3}(Z_l^{24})
\right),
$$

$$
Z_l^{16}
=
\phi\left(
\operatorname{Conv}_{1\times1}^{24\rightarrow16}(Z_l^{\mathrm{dw}})
\right).
$$

Các output heads đều là $1\times1$ convolution:

1. Parent-shared structured geometry.
2. Phase-specific sample weights.
3. Phase-specific update gates.
4. Phase-specific planarity gates.
5. Phase-specific surface normals.
6. Phase-specific confidence logits.

Phần spatial $3\times3$ chỉ chạy một lần và được chia sẻ giữa mọi prediction head.

---

## 19. Structured sampling stencil

### 19.1 Base stencil

Với $K=5$:

$$
\mathcal R_5
=
\{(0,0),(-1,0),(1,0),(0,-1),(0,1)\}.
$$

Với $K=3$:

$$
\mathcal R_3
=
\{(-1,0),(0,0),(1,0)\}.
$$

### 19.2 Parent-shared transform

Geometry head dự đoán:

$$
\Theta_p
=
(t_x,t_y,\theta,\ell_{\parallel},\ell_{\perp}).
$$

Bounded translation:

$$
t_p
=
t_{\max}\tanh([t_x,t_y]).
$$

Positive anisotropic scales:

$$
a_{\parallel}
=
a_{\min}+(a_{\max}-a_{\min})\sigma(\ell_{\parallel}),
$$

$$
a_{\perp}
=
a_{\min}+(a_{\max}-a_{\min})\sigma(\ell_{\perp}).
$$

Rotation matrix:

$$
\mathcal R(\theta)
=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

Sample location cho child $\phi$ và stencil point $i$:

$$
s_{p,\phi,i}
=
q_{p,\phi}
+t_p
+
\mathcal R(\theta_p)
\begin{bmatrix}
a_{\parallel,p}&0\\
0&a_{\perp,p}
\end{bmatrix}
r_i.
$$

Structured transform thay $2K$ offset tự do bằng năm geometry parameters. Nó:

- Giảm output channels.
- Bảo toàn neighborhood structure.
- Hạn chế sample collapse.
- Cho phép orientation và anisotropic search dọc theo bề mặt.

---

## 20. Direct source-space sampling

Depth sample:

$$
d_{p,\phi,i}
=
\operatorname{Bilinear}
\left(D_l,s_{p,\phi,i}\right).
$$

Ray sample:

$$
r_{p,\phi,i}^{s}
=
\operatorname{Bilinear}
\left(R_l,s_{p,\phi,i}\right).
$$

3D point hypothesis:

$$
P_{p,\phi,i}
=
d_{p,\phi,i}r_{p,\phi,i}^{s}.
$$

Không sử dụng `unfold` patch lớn. Implementation mong muốn là một fused source-space sampler.

---

## 21. Phase-specific ray-plane lifting

### 21.1 Child-specific normal

Normal head dự đoán bốn normals cho bốn children:

$$
\widehat n_{p,\phi}\in\mathbb{R}^{3},
$$

$$
n_{p,\phi}
=
\frac{\widehat n_{p,\phi}}
{\|\widehat n_{p,\phi}\|_2+\epsilon}.
$$

Child-specific normals quan trọng ở object boundaries: bốn children của cùng một parent có thể nằm trên hai bề mặt khác nhau.

### 21.2 Plane transport

Mặt phẳng qua sample point $P_{p,\phi,i}$ với normal $n_{p,\phi}$ thỏa:

$$
n_{p,\phi}^{\top}X
=
n_{p,\phi}^{\top}P_{p,\phi,i}.
$$

Target point nằm trên ray $r_h(p,\phi)$:

$$
X=\widetilde d_{p,\phi,i}r_h(p,\phi).
$$

Do đó transported depth candidate:

$$
\widetilde d_{p,\phi,i}
=
\frac{
n_{p,\phi}^{\top}P_{p,\phi,i}
}{
n_{p,\phi}^{\top}r_h(p,\phi)+\epsilon
}.
$$

Cần clamp denominator và candidate depth để tránh numerical instability:

$$
\left|n_{p,\phi}^{\top}r_h(p,\phi)\right|
\geq \epsilon_n.
$$

### 21.3 Planarity-aware fallback

Không phải vùng nào cũng phù hợp với một local plane. Dự đoán planarity gate:

$$
\eta_{p,\phi}
=
\sigma\left(h_{\eta}(Z_l^{16})_{p,\phi}\right).
$$

Blended candidate:

$$
\widehat d_{p,\phi,i}
=
(1-\eta_{p,\phi})d_{p,\phi,i}
+
\eta_{p,\phi}\widetilde d_{p,\phi,i}.
$$

- $\eta\approx0$: fallback về standard dynamic depth sampling.
- $\eta\approx1$: dùng ray-plane transport.

Cơ chế này giúp RayLift ổn định ở non-planar, reflective hoặc highly discontinuous regions mà vẫn khai thác 3D geometry ở road, walls và object surfaces.

---

## 22. Phase-specific aggregation

Raw child logits:

$$
a_{p,\phi,i}=h_{\alpha}(Z_l^{16})_{p,\phi,i}.
$$

Normalized weights:

$$
\alpha_{p,\phi,i}
=
\frac{\exp(a_{p,\phi,i})}
{\sum_{j=1}^{K}\exp(a_{p,\phi,j})},
$$

$$
\sum_i\alpha_{p,\phi,i}=1.
$$

Geometry-routed candidate:

$$
\overline D_{p,\phi}
=
\sum_{i=1}^{K}
\alpha_{p,\phi,i}
\widehat d_{p,\phi,i}.
$$

Softmax weights đóng vai trò chọn surface-consistent samples; structured geometry quyết định các sample được đặt ở đâu.

---

## 23. Gated residual upsampling

Bilinear base:

$$
B_{p,\phi}
=
\operatorname{Bilinear}
\left(D_l,q_{p,\phi}\right).
$$

Update gate:

$$
g_{p,\phi}
=
\sigma\left(h_g(Z_l^{16})_{p,\phi}\right).
$$

Target child depth:

$$
D_h(p,\phi)
=
B_{p,\phi}
+
g_{p,\phi}
\left(
\overline D_{p,\phi}-B_{p,\phi}
\right).
$$

Tương đương:

$$
D_h(p,\phi)
=
(1-g_{p,\phi})B_{p,\phi}
+
g_{p,\phi}\overline D_{p,\phi}.
$$

- $g\approx0$: giữ bilinear base ổn định.
- $g\approx1$: dùng fully geometry-routed reconstruction.
- Gate được dự đoán riêng cho từng child.

Bốn child outputs tại source grid được rearrange bằng PixelShuffle:

$$
[B,4,H_l,W_l]
\rightarrow
[B,1,2H_l,2W_l].
$$

---

## 24. Confidence từ transported-sample consensus

Dùng dispersion trong log-depth space:

$$
V_{p,\phi}
=
\sum_i
\alpha_{p,\phi,i}
\left|
\log(\widehat d_{p,\phi,i}+\epsilon)
-
\log(\overline D_{p,\phi}+\epsilon)
\right|.
$$

Learned confidence factor:

$$
\widehat C_{p,\phi}
=
\sigma\left(h_C(Z_l^{16})_{p,\phi}\right).
$$

Inherited confidence:

$$
C^{\mathrm{base}}_{p,\phi}
=
\operatorname{Bilinear}(C_l,q_{p,\phi}).
$$

Updated confidence:

$$
C_h(p,\phi)
=
\widehat C_{p,\phi}
\,C^{\mathrm{base}}_{p,\phi}
\exp(-\gamma V_{p,\phi}).
$$

Ý nghĩa:

- Sample đồng thuận $\Rightarrow V$ nhỏ $\Rightarrow$ confidence cao.
- Foreground/background conflict $\Rightarrow V$ lớn $\Rightarrow$ confidence giảm.
- Confidence không phải một output hoàn toàn tự do; nó liên kết với geometry consensus.

---

## 25. RayLift output-channel budget

Parent-shared geometry:

$$
5\text{ channels}.
$$

Phase-specific outputs:

- Weights: $4K$.
- Update gates: $4$.
- Planarity gates: $4$.
- Normals: $4\times3=12$.
- Confidence logits: $4$.

Tổng:

$$
C_{\mathrm{out}}^{\mathrm{RayLift}}
=
5+4K+4+4+12+4
=
29+4K.
$$

- $K=5$: $49$ channels.
- $K=3$: $41$ channels.

Các channels này được dự đoán ở **source resolution**, không phải target resolution. Hai stage đắt nhất chỉ dùng $K=3$.

---

## 26. RayLift initialization

Để bắt đầu gần bilinear upsampling:

$$
t_p=0,
\qquad
\theta_p=0,
\qquad
a_{\parallel}=a_{\perp}=1,
$$

$$
g_{p,\phi}\approx0.05,
\qquad
\eta_{p,\phi}\approx0,
$$

$$
\alpha_{p,\phi,i}\approx\frac{1}{K}.
$$

Khi bắt đầu training:

$$
D_h\approx\operatorname{Bilinear}(D_l).
$$

Sau đó model học structured routing và plane transport dần, tránh random-offset instability.

---

## 27. Progressive reconstruction schedule

### Stage 1: $1/16\rightarrow1/8$

```text
Inputs: D16, C16, P16, P8, D_init16/8, R16/R8, M8
K=5 cross stencil
Search radius lớn nhất
Output: D8, C8
```

Mục tiêu: recover large structures và large-hole regions.

### Stage 2: $1/8\rightarrow1/4$

```text
Inputs: D8, C8, P8, P4, D_init8/4, R8/R4, M4
K=5 cross stencil
Output: D4, C4
```

Mục tiêu: object-level structure và metric alignment với $D^{\mathrm{init}}_4$.

### Stage 3: $1/4\rightarrow1/2$

```text
Inputs: D4, C4, P4, G2, R4/R2, M2
K=3 oriented line
Local search
Output: D2, C2
```

Mục tiêu: boundary localization và thin structures.

### Stage 4: $1/2\rightarrow1$

```text
Inputs: D2, C2, G2, G1, R2/R1, M1
K=3 oriented line
Smallest search radius
Output: D1, C1
```

Mục tiêu: pixel-level boundary sharpening mà không dùng full-resolution wide features.

---

## 28. Adaptive sparse anchoring

Tại full resolution, local sparse density:

$$
\rho(p)=\operatorname{AvgPool}_k(M)(p).
$$

Anchor discrepancy:

$$
\Delta_S(p)
=
\frac{|S(p)-D_1(p)|}{S(p)+\epsilon}.
$$

Learned anchoring coefficient:

$$
\lambda_p
=
M(p)
\left[
\lambda_{\min}
+
(1-\lambda_{\min})
\sigma\left(
h_{\lambda}
[C_1(p),\Delta_S(p),\rho(p),|\nabla I(p)|]
\right)
\right].
$$

Default clean-sensor setting:

$$
\lambda_{\min}=0.5.
$$

Final output:

$$
D_{\mathrm{full}}(p)
=
D_1(p)
+
\lambda_p
\left(S(p)-D_1(p)\right).
$$

Với $M(p)=0$, $\lambda_p=0$ và output giữ nguyên prediction. Có thể giảm $\lambda_{\min}$ hoặc thêm outlier classifier khi sensor noise cao.

Confidence output:

$$
C_{\mathrm{full}}(p)
=
(1-M(p))C_1(p)
+
M(p)\max(C_1(p),C_S(p)),
$$

trong đó $C_S$ là sensor confidence nếu có.

---

# Phần IV — Training Objectives

## 29. Multi-scale supervised metric loss

Các output:

$$
\mathcal S=\{16,8,4,2,1\}.
$$

Ground truth được downsample valid-aware thành $D_{\mathrm{gt},s}$.

$$
\mathcal L_{\mathrm{gt}}^{s}
=
\frac{1}{|\Omega_s|}
\sum_{p\in\Omega_s}
\rho_H
\left(
\log(D_s(p)+\epsilon)
-
\log(D_{\mathrm{gt},s}(p)+\epsilon)
\right),
$$

trong đó $\rho_H$ là Huber hoặc BerHu.

Tổng:

$$
\mathcal L_{\mathrm{gt}}
=
\sum_{s\in\mathcal S}
\lambda_s^{\mathrm{gt}}
\mathcal L_{\mathrm{gt}}^s.
$$

Output scales cao hơn nhận trọng số lớn hơn.

---

## 30. Sparse-anchor loss

$$
\mathcal L_S
=
\frac{1}{|\Omega_M|}
\sum_{p\in\Omega_M}
\left|D_{\mathrm{full}}(p)-S(p)\right|.
$$

Có thể thêm pre-correction loss trên $D_1$ để đảm bảo student không phụ thuộc hoàn toàn vào post anchoring:

$$
\mathcal L_S^{\mathrm{pre}}
=
\frac{1}{|\Omega_M|}
\sum_{p\in\Omega_M}
\left|D_1(p)-S(p)\right|.
$$

---

## 31. Metric-teacher distillation

Teacher được downsample về từng scale:

$$
D_{T,s}^{\mathrm{metric}}
=
\operatorname{VDown}_s(D_T^{\mathrm{metric}}).
$$

$$
W_{T,s}^{\mathrm{metric}}
=
\operatorname{VDown}_s(W_T^{\mathrm{metric}}).
$$

Loss:

$$
\mathcal L_{\mathrm{KD}}^{\mathrm{metric}}
=
\sum_{s\in\mathcal S}
\lambda_s^T
\frac{
\sum_p W_{T,s}^{\mathrm{metric}}(p)
\rho_H
\left[
\log(D_s(p)+\epsilon)
-
\log(D_{T,s}^{\mathrm{metric}}(p)+\epsilon)
\right]
}{
\sum_p W_{T,s}^{\mathrm{metric}}(p)+\epsilon
}.
$$

DMD3C chủ yếu giám sát global metric structure và dense regions không có ground truth.

---

## 32. Relative geometry distillation

### 32.1 SSI loss

Tìm scale–shift alignment giữa student log-depth $Z=\log(D+\epsilon)$ và geometry consensus:

$$
(a^\star,b^\star)
=
\arg\min_{a,b}
\sum_p W_G(p)
\left|
aZ(p)+b-R_G^\star(p)
\right|^2.
$$

$$
\mathcal L_{\mathrm{SSI}}
=
\frac{
\sum_p W_G(p)
\rho_H
\left[
a^\star Z(p)+b^\star-R_G^\star(p)
\right]
}{
\sum_p W_G(p)+\epsilon
}.
$$

### 32.2 Ordinal loss

Với pair $(p,q)$ và teacher relation $y_{pq}\in\{-1,+1\}$:

$$
\mathcal L_{\mathrm{ord}}
=
\frac{1}{|\mathcal P|}
\sum_{(p,q)\in\mathcal P}
\log
\left[
1+
\exp
\left(
-y_{pq}
[\log D(p)-\log D(q)]
\right)
\right].
$$

Ordinal pairs nên tập trung ở foreground/background boundaries, thin structures và regions có teacher agreement cao.

---

## 33. Boundary and gradient distillation

Teacher edge band:

$$
\Omega_E
=
\operatorname{Dilate}
\left(
|\nabla R_G^\star|>\tau_G
\;\lor\;
|\nabla D_{\mathrm{gt}}|>\tau_D
\right).
$$

Boundary loss:

$$
\mathcal L_{\mathrm{edge}}
=
\frac{1}{|\Omega_E|}
\sum_{p\in\Omega_E}
\rho_H
\left(
D_{\mathrm{full}}(p)-D_{\mathrm{sup}}(p)
\right).
$$

Gradient matching:

$$
\mathcal L_{\nabla}
=
\sum_{s\in\{2,1\}}
\left(
\|\partial_x\log D_s-\partial_x Z_T\|_1
+
\|\partial_y\log D_s-\partial_y Z_T\|_1
\right).
$$

$Z_T$ ưu tiên ground truth; nếu ground truth thiếu, dùng reliability-weighted geometry consensus.

---

## 34. RayLift normal distillation

Downsample DSINE normal và reliability về từng target scale:

$$
N_{T,h}=\operatorname{NDown}_h(N_T),
\qquad
W_{N,h}=\operatorname{VDown}_h(W_N).
$$

Normal loss cho bốn children:

$$
\mathcal L_N
=
\sum_h
\frac{
\sum_{p,\phi}
W_{N,h}(p,\phi)
\left[
1-
\left|
n_{p,\phi}^{\top}N_{T,h}(p,\phi)
\right|
\right]
}{
\sum_{p,\phi}W_{N,h}(p,\phi)+\epsilon
}.
$$

Dùng absolute cosine nếu normal orientation có thể đảo dấu; dùng signed cosine nếu convention đã được thống nhất.

Planarity target có thể được suy ra từ local normal variance:

$$
\eta_T(p)
=
\exp\left(-\kappa\operatorname{Var}_{q\in\mathcal N(p)}N_T(q)\right).
$$

$$
\mathcal L_{\eta}
=
\operatorname{BCE}(\eta,\eta_T).
$$

---

## 35. Routing regularization

### 35.1 Translation regularization

$$
\mathcal L_t
=
\sum_{p}\left\|t_p\right\|_1.
$$

Có thể edge-weight:

$$
\mathcal L_t^{\mathrm{edge}}
=
\sum_p
(1-E_p)\|t_p\|_1,
$$

để cho phép routing linh hoạt hơn tại boundaries.

### 35.2 Anisotropy regularization

$$
\mathcal L_a
=
\sum_p
\left|
\log a_{\parallel,p}
-
\log a_{\perp,p}
\right|
(1-E_p).
$$

Không phạt anisotropy mạnh ở edge regions.

### 35.3 Gate sparsity

$$
\mathcal L_g
=
\sum_{p,\phi}
(1-U_{p,\phi})g_{p,\phi},
$$

trong đó $U$ là teacher/student uncertainty hoặc boundary likelihood. Gate được khuyến khích nhỏ ở vùng coarse depth đã đúng.

---

## 36. Confidence calibration loss

Đặt predicted uncertainty:

$$
u_s(p)=-\log(C_s(p)+\epsilon).
$$

Depth residual:

$$
e_s(p)
=
\left|
\log(D_s(p)+\epsilon)
-
\log(D_{\mathrm{gt},s}(p)+\epsilon)
\right|.
$$

Heteroscedastic Laplace objective:

$$
\mathcal L_C
=
\sum_s
\frac{1}{|\Omega_s|}
\sum_{p\in\Omega_s}
\left[
e_s(p)e^{-u_s(p)}+u_s(p)
\right].
$$

Confidence cần được đánh giá bằng risk–coverage, AUSE/AURG hoặc error–confidence correlation; nếu không, nên gọi là reliability map.

---

## 37. Total objective

$$
\boxed{
\begin{aligned}
\mathcal L
={}&
\lambda_{\mathrm{gt}}\mathcal L_{\mathrm{gt}}
+
\lambda_S\mathcal L_S
+
\lambda_{S,\mathrm{pre}}\mathcal L_S^{\mathrm{pre}}
\\
&+
\lambda_T\mathcal L_{\mathrm{KD}}^{\mathrm{metric}}
+
\lambda_{\mathrm{SSI}}\mathcal L_{\mathrm{SSI}}
+
\lambda_{\mathrm{ord}}\mathcal L_{\mathrm{ord}}
\\
&+
\lambda_{\mathrm{edge}}\mathcal L_{\mathrm{edge}}
+
\lambda_{\nabla}\mathcal L_{\nabla}
+
\lambda_N\mathcal L_N
+
\lambda_{\eta}\mathcal L_{\eta}
\\
&+
\lambda_R
(\mathcal L_t+\mathcal L_a+\mathcal L_g)
+
\lambda_C\mathcal L_C.
\end{aligned}
}
$$

Starting weights cần được tune theo dataset; một cấu hình khởi đầu:

| Loss | Weight |
|---|---:|
| $\lambda_{\mathrm{gt}}$ | 1.0 |
| $\lambda_S$ | 0.5–1.0 |
| $\lambda_{S,\mathrm{pre}}$ | 0.1 |
| $\lambda_T$ | 0.2–0.4 |
| $\lambda_{\mathrm{SSI}}$ | 0.02–0.05 |
| $\lambda_{\mathrm{ord}}$ | 0.02–0.05 |
| $\lambda_{\mathrm{edge}}$ | 0.05–0.15 |
| $\lambda_{\nabla}$ | 0.02–0.05 |
| $\lambda_N$ | 0.02–0.05 |
| $\lambda_{\eta}$ | 0.01–0.03 |
| $\lambda_R$ | $10^{-4}$–$10^{-3}$ |
| $\lambda_C$ | 0.02–0.05 |

---

# Phần V — Training and Deployment

## 38. Training schedule

### Stage A — Metric bootstrap

Bật:

- Ground-truth metric loss.
- Sparse-anchor loss.
- DMD3C metric distillation.
- Initial head và bốn RayLift stages.

Tắt hoặc giảm mạnh:

- Geometry SSI/ordinal.
- Normal distillation.
- Confidence calibration phức tạp.

Mục tiêu: học metric scale và stable progressive reconstruction.

### Stage B — Geometry lifting

Bật:

- DA2/Metric3D geometry consensus.
- Boundary và ordinal losses.
- DSINE normal supervision.
- Planarity target.

Tăng dần RayLift update gates và planarity gates từ near-zero initialization.

### Stage C — Conflict-aware calibration

Bật đầy đủ:

- Teacher reliability masks.
- Confidence calibration.
- Sparse outlier/noise augmentation.
- RGB–LiDAR misalignment augmentation.
- Lower learning rate và EMA.

---

## 39. Data augmentation

Geometry transforms phải cập nhật $K$, sparse depth, mask và cached teacher maps nhất quán.

### Recommended

- Horizontal flip với cập nhật principal point.
- Resize/crop với cập nhật intrinsics.
- Sparse point dropout.
- Simulated 8/16/32/64-line LiDAR.
- Local holes và foreground point removal.
- Range-dependent depth noise.
- Sparse outlier injection.
- RGB–LiDAR shift 1–3 pixels.
- Brightness, contrast, gamma và blur nhẹ.

Ray maps phải được tái tạo sau geometry augmentation.

---

## 40. Deployment-oriented implementation

### Required

1. Không dùng `unfold` patch lớn.
2. Sample trực tiếp trên $D_l$ và $R_l$.
3. Vectorize bốn phases và $K$ samples.
4. Fuse bilinear sampling, ray-plane transport, weighted sum và residual update.
5. Dùng FP16 hoặc BF16 nếu phần cứng hỗ trợ.
6. Fuse Conv–Norm–Activation.
7. Cố định input resolution khi benchmark production.

### Fused RayLift operator

```text
Read source depth/ray
Read structured geometry and child parameters
Generate K coordinates for four phases
Bilinear sample depth and ray
Ray-plane transport
Planarity blend
Softmax accumulation
Residual gate
Confidence consensus
Write D_h and C_h
```

### Không dùng

- Iterative CSPN/NLSPN/DySPN.
- Full-resolution Transformer.
- ConvGRU refinement.
- Optimization solver.
- Teacher/foundation model khi inference.
- Wide standard convolution ở full resolution.

---

## 41. Complexity rationale

RayLift parameter head chạy ở source resolution. Tổng target sample count cho $K=5,5,3,3$:

$$
N_{\mathrm{samples}}
=
5N_8+5N_4+3N_2+3N_1.
$$

Với $352\times1216$:

- Hai stage đầu dùng nhiều samples nhưng chạy ở kích thước nhỏ.
- Hai stage cuối chỉ dùng ba samples.
- Không có iterative passes.
- Ray-plane transport chỉ thêm dot products trên các samples đã lấy.

Chi phí thực tế vẫn phải được đo bằng batch-1 latency và peak memory; FLOPs không đủ phản ánh random memory access của bilinear sampling.

---

# Phần VI — Novelty and Experimental Validation

## 42. Đóng góp chính

### Contribution 1 — Role-Separated Geometry Distillation

Metric, relative geometry và surface-normal teachers được phân vai; teacher conflict được xử lý theo pixel thay vì hợp nhất thành một pseudo-label duy nhất.

### Contribution 2 — Phase-Aware Low-Resolution Reconstruction

RayLift dùng PixelUnshuffle target guidance để dự đoán riêng bốn children trong khi parameter head vẫn chạy ở source resolution.

### Contribution 3 — Structured Geometry Routing

Sampling positions được sinh từ translation, orientation và anisotropic scale của một stencil có cấu trúc, thay vì $2K$ offsets độc lập.

### Contribution 4 — Phase-Specific Ray-Plane Lifting

Mỗi sampled depth được back-project thành 3D point hypothesis và transported đến target ray qua child-specific local plane trước khi aggregation.

### Contribution 5 — Planarity-Aware Dual-Domain Fallback

Mỗi child tự blend giữa conventional scalar-depth sampling và ray-plane transported depth, giúp giữ ổn định ngoài planar regions.

### Contribution 6 — Teacher-Free, Non-Iterative Real-Time Student

Inference chỉ dùng lightweight encoder, LiteFPN và bốn one-pass RayLift stages.

---

## 43. Novelty positioning

Các ý tưởng riêng lẻ như progressive decoding, dynamic sampling, convex upsampling, teacher distillation và ray encoding đều có prior art. Novelty không nên được tuyên bố dựa trên từng thành phần độc lập.

Điểm khác biệt cần bảo vệ là tổ hợp toán học và kiến trúc:

$$
\boxed{
\text{role-separated distillation}
+
\text{phase-aware structured routing}
+
\text{child-specific ray-plane lifting}
+
\text{one-pass progressive reconstruction}
}
$$

Claim trung tâm phù hợp:

> GeoLift-RT reconstructs each high-resolution depth child by transporting adaptively sampled metric hypotheses through a phase-specific local 3D plane, while all routing parameters are predicted at the lower source resolution.

Không nên tuyên bố tuyệt đối “first” trước khi hoàn thành systematic literature và patent search.

---

## 44. Main architecture ablation

| ID | Cấu hình |
|---|---|
| A0 | Current GeoRT student: coarse $1/4$ + guided convex $4\times$ + fixed correction |
| A1 | Progressive bilinear $2\times$ decoder |
| A2 | Progressive dynamic weights, fixed sampling |
| A3 | Free-offset ReDC after each upsample |
| A4 | Joint upsampling–refinement, free offsets |
| A5 | Structured stencil routing |
| A6 | + phase-aware PixelUnshuffle guidance |
| A7 | + ray-plane lifting |
| A8 | + phase-specific normals |
| A9 | + planarity fallback |
| A10 | + consensus confidence |
| A11 | + adaptive sparse anchoring — full GeoLift-S |

---

## 45. RayLift ablation

### Sampling

- $K=9,9,5,3$.
- $K=5,5,5,3$.
- $K=5,5,3,3$ — main.
- $K=3,3,3,3$.

### Geometry

- Fixed stencil.
- Free independent offsets.
- Translation only.
- Translation + rotation.
- Translation + rotation + anisotropic scaling — main.

### Plane transport

- Scalar depth only.
- Parent-shared normal.
- Child-specific normal.
- Child-specific normal + planarity fallback — main.

### Guidance

- No target guidance.
- Bilinear-downsampled target guidance.
- PixelUnshuffle phase-aware guidance — main.

### Confidence

- Learned confidence only.
- Sample-consensus confidence.
- Inherited + learned + sample-consensus — main.

---

## 46. Distillation ablation

| ID | Supervision |
|---|---|
| T0 | GT + sparse only |
| T1 | + DMD3C uniform metric KD |
| T2 | + conflict-aware DMD3C reliability |
| T3 | + DA2 SSI |
| T4 | + ordinal boundary supervision |
| T5 | + Metric3D geometry consensus |
| T6 | + DSINE normal supervision |
| T7 | Full RSGD |

Bảng kết quả chính phải tách:

$$
\text{architecture gain}
\quad\text{vs.}\quad
\text{distillation gain}.
$$

Tối thiểu cần báo:

1. Current student.
2. GeoLift-S không distillation.
3. Current student + RSGD.
4. Full GeoLift-RT.

---

## 47. Metrics

### Accuracy

- RMSE, MAE, iRMSE, iMAE.
- REL và $\delta_1$ nếu protocol sử dụng.
- Near/mid/far-range RMSE.
- Sparse-anchor error.

RMSE tuyệt đối chỉ có nghĩa cho tensor metric depth như $D^{\mathrm{init}},D_{16},D_8,D_4,D_2,D_1,D_{\mathrm{full}}$. Stem, fusion, encoder, injection và feature FPN không có target depth trực tiếp; ảnh hưởng accuracy của chúng phải được báo bằng $\Delta$RMSE từ ablation có cùng training protocol, không gán một RMSE giả cho feature tensor.

### Boundary

- Boundary-band RMSE 3 px và 5 px.
- Depth-gradient MAE.
- Foreground/background ordinal accuracy.
- Thin-object region error.

### Confidence

- Error–confidence correlation.
- Risk–coverage curve.
- AUSE, AURG.
- RMSE theo confidence quantile.

### Efficiency

- Parameters.
- MACs/FLOPs.
- Median và P95 latency.
- FPS.
- Peak allocated memory.
- FP32 và FP16.
- PyTorch, ONNX Runtime và TensorRT khi có.

---

## 48. Benchmark protocol

- Batch size 1.
- Input $352\times1216$.
- Warm-up tối thiểu 100 iterations.
- Đo tối thiểu 500 iterations.
- CUDA synchronize trước và sau timing.
- Báo median và P95.
- Đo toàn pipeline gồm preprocessing, $D^{\mathrm{init}}$, network và sparse anchoring.
- RMSE/MAE chính phải aggregate trên toàn bộ valid pixels của split; báo thêm macro per-image nếu cần, nhưng không thay thế global aggregation.
- Không so trực tiếp latency từ paper khác nếu GPU, resolution hoặc runtime khác nhau.

Bảng profile và trạng thái triển khai hiện tại nằm tại [`GeoLift-RT_Component_Metrics.md`](GeoLift-RT_Component_Metrics.md).

---

## 49. Rủi ro và biện pháp

### Ray-plane instability

**Rủi ro:** denominator nhỏ hoặc predicted normal sai.

**Giải pháp:** denominator clamp, depth clamp, planarity fallback, DSINE supervision và near-bilinear initialization.

### Single-parent geometry tại boundaries

**Rủi ro:** bốn children thuộc các bề mặt khác nhau.

**Giải pháp:** child-specific normals, weights, update gates và planarity gates; chỉ stencil transform được parent-shared.

### Geometry branch copy RGB texture

**Rủi ro:** target guidance tạo false depth edges.

**Giải pháp:** sparse/ray cues, metric teacher supervision, confidence gate và edge reliability.

### Adaptive anchoring tin outlier

**Rủi ro:** sparse sensor chứa outlier hoặc misalignment.

**Giải pháp:** discrepancy input, outlier augmentation, configurable $\lambda_{\min}$ và optional outlier classifier.

### Dynamic sampler chậm trên deployment

**Rủi ro:** generic `grid_sample` bị memory-bound.

**Giải pháp:** direct fused CUDA/TensorRT operator, không `unfold`, static shape và FP16.

---

# Phần VII — Final Technical Specification

## 50. GeoLift-S main configuration

```text
Input:
  RGB I                      [B,3,352,1216]
  Sparse depth S             [B,1,352,1216]
  Validity mask M            [B,1,352,1216]
  Camera intrinsics K        [B,3,3]

Analytic prior:
  D_init at 1/4
  valid-aware D_init pyramid at 1/8 and 1/16

Stems at 1/4:
  RGB                         24ch
  sparse/mask/D_init          16ch
  ray/UV                      12ch
  concat                      52ch

Fusion:
  Conv 1×1, 52→32
  DWConv 3×3
  PWConv 1×1

Encoder:
  stage-adapted MobileViTv2-0.75
  X0 at 1/4 projected to F4; only stages producing F8/F16 are executed
  F4, F8, F16
  sparse/ray gated injection

Feature decoder:
  additive LiteFPN
  P16, P8, P4

Depth decoder:
  initial metric head         D16,C16
  RayLift 16→8                K=5
  RayLift 8→4                 K=5
  RayLift 4→2                 K=3
  RayLift 2→1                 K=3

RayLift trunk:
  projection                  24ch
  DWConv 3×3                  24ch
  PWConv 1×1                  16ch

RayLift geometry:
  parent-shared translation
  parent-shared orientation
  parent-shared anisotropic scale
  child-specific weights
  child-specific normal
  child-specific planarity gate
  child-specific update gate
  child-specific confidence

Output:
  adaptive sparse anchoring
  D_full                     [B,1,352,1216]
  C_full                     [B,1,352,1216]

Inference exclusions:
  no teacher
  no foundation model
  no iterative propagation
  no recurrent refinement
  no full-resolution attention
  no optimization solver
```

---

## 51. Compact forward flow

$$
D^{\mathrm{init}}_4
=
\operatorname{LocalNormalizedPropagation}(S,M),
$$

$$
F_4,F_8,F_{16}
=
\operatorname{Encoder}
\left(
\operatorname{Fusion}
[
\psi_I(I),
\psi_D(S,M,D^{\mathrm{init}}_4),
\psi_R(K)
]
\right),
$$

$$
P_{16},P_8,P_4
=
\operatorname{LiteFPN}(F_{16},F_8,F_4),
$$

$$
(D_{16},C_{16})
=
H_{\mathrm{init}}(P_{16}),
$$

$$
(D_8,C_8)
=
\operatorname{RayLift}_{16\rightarrow8}
(D_{16},C_{16},P_{16},P_8),
$$

$$
(D_4,C_4)
=
\operatorname{RayLift}_{8\rightarrow4}
(D_8,C_8,P_8,P_4),
$$

$$
(D_2,C_2)
=
\operatorname{RayLift}_{4\rightarrow2}
(D_4,C_4,P_4,G_2),
$$

$$
(D_1,C_1)
=
\operatorname{RayLift}_{2\rightarrow1}
(D_2,C_2,G_2,G_1),
$$

$$
(D_{\mathrm{full}},C_{\mathrm{full}})
=
\operatorname{SparseAnchor}(D_1,C_1,S,M).
$$

---

## 52. Tóm tắt một câu

> **GeoLift-RT distills metric, relative-geometry and normal knowledge into a teacher-free student whose RayLift decoder reconstructs each high-resolution child by phase-aware structured sampling and local ray-plane depth transport, without iterative propagation.**

---

## 53. Tên và cách viết trong paper

### Full title

**GeoLift-RT: Role-Separated Geometry Distillation with Phase-Aware Ray-Plane Lifting for Real-Time Depth Completion**

### Framework

**GeoLift-RT**

### Student

**GeoLift-S**

### Training strategy

**RSGD: Role-Separated Geometry Distillation**

### Decoder block

**RayLift: Phase-Aware Ray-Plane Residual Upsampling**

### Core contributions

1. Role-separated, conflict-aware offline distillation.
2. Phase-aware low-resolution parameter prediction.
3. Structured anisotropic sampling.
4. Child-specific ray-plane metric-depth lifting.
5. Planarity-aware scalar/geometry fallback.
6. Non-iterative teacher-free real-time inference.
