Có, và nếu làm đúng thì hướng này có tiềm năng mạnh hơn V5 Piecewise hiện tại vì contribution chuyển từ “propagation/interpolation tốt hơn” thành một low-dimensional metric shape dynamics model cho depth completion.
Nhưng có một điều phải sửa ngay trong formulation: không nên PCA trực tiếp toàn bộ raw point cloud rồi kỳ vọng giữ được edge. BrainODE dùng PCA hiệu quả vì các mesh có correspondence cố định và cùng topology; PCA 150 chiều giữ >98% shape variability trong setting đó. Nó sau đó dùng self-attention để sinh velocity 150 chiều trong PCA shape space. OpenReview
Còn paper 2026 mà đang nhắc tới đúng là Attention-based PCA, được Claire Boyer liệt kê là NeurIPS 2026 Spotlight. Paper chứng minh trong setting lý thuyết Gaussian rằng softmax/linear attention học các hướng align với principal eigenvectors của covariance và có thể recover leading spectral direction. Nó là lý thuyết về attention–PCA, không phải sẵn một architecture 3D để copy. Imo Université Paris-Saclay
Tôi sẽ lấy nó làm motivation, rồi xây architecture riêng.
1. Ý tưởng tổng quát sẽ trở thành
Thay vì V5 hiện tại:
\[
D_4
\rightarrow
\text{local transport}
\rightarrow
D_4'
\]
thì chuyển sang:
\[
\boxed{
D_4
\rightarrow
\text{metric surface}
\rightarrow
\text{PCA latent shape}
\rightarrow
\text{attention vector field}
\rightarrow
\text{latent flow}
\rightarrow
\text{metric surface}
\rightarrow
D
}
\]
Tức gần BrainODE hơn rất nhiều.
BrainODE:
\[
\Lambda_t
\xrightarrow{
f_\theta(\Lambda_t,c,t)
}
\frac{d\Lambda_t}{dt}
\]
với:
undefined
Trong đó \(\tau\) không phải thời gian vật lý.
Nó là pseudo-time:
\[
\tau:0\rightarrow1
\]
biểu diễn:
\[
\boxed{
\text{coarse surface}
\rightarrow
\text{completed surface}
}
\]
2. Điểm rất hay: D4 giải quyết vấn đề correspondence
Raw LiDAR point cloud:
\[
P=\{X_i\}
\]
không có fixed correspondence giữa hai frame/cảnh.
Do đó PCA:
\[
P\rightarrow z
\]
trực tiếp là nguy hiểm.
Nhưng AnchorFlow đã có dense \(D_4\).
Ví dụ:
\[
D_4\in\mathbb R^{88\times304}.
\]
Mỗi pixel luôn có index cố định:
\[
p=(u,v).
\]
Back-project:
\[
X_p
=
D_4(p)
K_4^{-1}
\begin{bmatrix}
u\\v\\1
\end{bmatrix}.
\]
Vậy:
\[
X=
[X_1,\ldots,X_N]
\]
đã có camera-grid correspondence.
Đây chính là thứ cho phép mình xây một shape vector:
\[
x
=
[
X_1^T,\ldots,X_N^T
]^T
\in\mathbb R^{3N}.
\]
Rồi:
\[
x\approx\mu+Ez.
\]
Với:
\[
E=[e_1,\ldots,e_K]
\]
là PCA basis.
3. Nhưng global PCA đơn thuần sẽ làm mất thứ mình đang cần nhất: edge
Đây là vấn đề lớn.
PCA ưu tiên phương sai lớn và low-rank reconstruction.
Do đó:
\[
x\approx\mu+Ez
\]
thường giữ:
- road geometry;
- global camera geometry;
- large objects;
- horizon;
- broad depth layout.
Nhưng dễ bỏ:
- pole;
- thin structures;
- car silhouette;
- occlusion discontinuity;
- sharp depth jump.
BrainODE thậm chí báo PCA làm surface mượt hơn bằng cách filtering high-frequency noise. Điều đó tốt với hippocampus, nhưng với depth completion thì high-frequency không phải lúc nào cũng noise — depth edge chính là signal cần giữ. OpenReview
Vì vậy không nên:
\[
\boxed{D = \text{PCA decoder only}}
\]
mà phải:
\[
\boxed{
D
=
D_{\text{spectral/global}}
+
D_{\text{edge/high-frequency}}
}
\]
Đây là key của architecture.
4. Tôi sẽ gọi concept này là
Edge-Preserved Spectral Surface Flow
EPSF
hoặc paper title kiểu:
Learning Edge-Preserved Spectral Surface Dynamics for Sparse Metric Depth Completion

Research story rõ hơn hẳn “Piecewise Surface Flow”.
5. Architecture tôi khuyên triển khai
RGB + sparse + mask + K
          │
          ▼
      V3 backbone
          │
          ▼
          D4
          │
          ▼
   Back-project by K
          │
          ▼
 Metric Surface X0
      /          \
     /            \
    ▼              ▼
Spectral Branch   Edge Branch
PCA coefficients  RGB/depth/sparse edge
    │              │
    ▼              │
Attention           │
vector field        │
    │               │
Latent ODE          │
z0 → z1             │
    │               │
PCA decode          │
    │               │
global surface      high-freq correction
     \              /
      \            /
       ▼          ▼
       edge-gated fusion
              │
              ▼
           D4 refined
              │
              ▼
       existing D2 → D1

Đây là architecture tôi thấy hợp lý nhất.
6. Spectral representation
Từ point surface:
\[
x_0=\operatorname{vec}(X_0).
\]
Thay vì PCA absolute geometry, tôi thích residual PCA hơn.
Định nghĩa correction training target:
\[
\Delta x
=
x_{GT}-x_0.
\]
Thu thập các correction trên training set:
\[
\Delta x_1,\Delta x_2,\ldots,\Delta x_M.
\]
PCA:
\[
\Delta x
\approx
Ez.
\]
Điều này cực kỳ hợp bài toán.
PCA không phải học:
“tất cả cảnh KITTI trông như thế nào?”

mà học:
“V3 thường phải biến dạng theo những mode nào để tiến về GT?”

Đây mới là shape/deformation space giống ShapeFlow và BrainODE.
7. PCA basis lúc này chính là các “deformation modes”
Ví dụ:
\[
e_1
\]
có thể encode:
- road plane correction.
\[
e_2
\]
có thể encode:
- depth scale/far geometry.
\[
e_3
\]
có thể encode:
- left-right perspective correction.
...
Mỗi scene có coefficient:
\[
z=
[z_1,\ldots,z_K].
\]
Coarse correction:
\[
\Delta X_{low}
=
\sum_{i=1}^{K}z_ie_i.
\]
Đây thực sự giống BrainODE hơn V5 hiện tại rất nhiều.
8. Attention không nên chỉ là Transformer đặt sau PCA
Nếu chỉ:
\[
z\rightarrow Transformer\rightarrow z'
\]
thì novelty không mạnh.
Tôi sẽ biến mỗi PCA mode thành một token.
Token \(i\):
\[
h_i=
[
z_i,
\lambda_i,
g_i,
s_i,
e_i^{edge}
].
\]
Trong đó:
- \(z_i\): current PCA coefficient;
- \(\lambda_i\): explained variance/eigenvalue;
- \(g_i\): projection của global RGB/depth feature lên mode đó;
- \(s_i\): sparse-sensor evidence;
- \(e_i^{edge}\): mode đó ảnh hưởng edge mạnh đến mức nào.
Sau đó:
\[
Q=HW_Q,
\quad
K=HW_K,
\quad
V=HW_V.
\]
Attention:
\[
A
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right).
\]
Đây chính là nơi Attention-based PCA 2026 trở thành inspiration hợp lý: paper cho thấy attention có quan hệ trực tiếp với principal spectral directions trong canonical PCA settings. Imo Université Paris-Saclay
Nhưng manuscript chỉ nên nói:
“motivated by the demonstrated connection between attention and principal spectral directions”

chứ không claim theorem đó đảm bảo cho point clouds.
9. Vector field sẽ nằm trong PCA space
Đây mới là phần BrainODE thật sự.
Cho:
\[
z(\tau)\in\mathbb R^K.
\]
Network sinh:
\[
\boxed{
v_z(\tau)
=
f_\theta
(
z(\tau),
H,
\tau
)
}
\]
với:
\[
v_z
=
[
\dot z_1,
\dot z_2,\ldots,\dot z_K
].
\]
Tức mỗi PCA mode có một “velocity”:
\[
\frac{dz_i}{d\tau}.
\]
Sau đó:
\[
\frac{dz}{d\tau}
=
f_\theta(z,C,\tau).
\]
Và:
\[
z(1)
=
z(0)+
\int_0^1f_\theta(z(\tau),C,\tau)d\tau.
\]
Đây chính xác là analogous với BrainODE:
undefined
Ý nghĩa:
lúc đầu chưa correction.

Rồi:
\[
z(0)
\rightarrow
z_1
\rightarrow
z_2
\rightarrow
z(1).
\]
Decode:
\[
\Delta x_{spec}
=
Ez(1).
\]
Surface:
\[
X_{spec}
=
X_0+\Delta X_{spec}.
\]
Rất clean.
11. Không cần adaptive ODE solver
Cho Edge AI:
\[
z^{k+1}
=
z^k+\Delta\tau f_\theta(z^k,C).
\]
Chỉ:
\[
k=0,1
\]
hoặc:
\[
k=0,1,2.
\]
Tức 2–3 fixed Euler steps.
Dimension:
\[
K=32,64,128
\]
rất nhỏ so với:
\[
88\times304\times3.
\]
Attention cost:
\[
O(K^2)
\]
với \(K=64\):
\[
4096
\]
pair interactions.
Gần như không đáng kể so với CNN backbone.
Đây là ưu điểm lớn.
12. Đây mới là nơi Attention-based PCA rất hợp Edge AI
Full image self-attention:
\[
N=26752
\]
tốn:
\[
O(N^2).
\]
Không khả thi.
Spectral attention:
\[
K=64
\]
chỉ:
\[
64^2=4096.
\]
Tức:
\[
\boxed{
\text{global reasoning}
\text{ in a tiny spectral space}
}
\]
Đây là một selling point mạnh:
global geometric interaction without spatial full-resolution attention.

13. Nhưng edge phải đi đường khác
Tôi sẽ không bắt PCA học edge.
Cho edge branch:
\[
B
=
g_\phi(
F_4,
\nabla D_4,
S_4,
M_4
).
\]
Trong đó:
\[
B_p\in[0,1]
\]
là probability pixel thuộc depth discontinuity.
Edge detection nên sử dụng đồng thời:
RGB
\[
|\nabla I|
\]
depth
\[
|\nabla D_4|
\]
sparse LiDAR discontinuity
\[
|\Delta S|
\]
geometric normal change
\[
1-n_p^Tn_q.
\]
Như vậy không bị lỗi:
màu đổi nhưng surface không đổi.

14. Edge branch dự đoán high-frequency residual
\[
R_{edge}
=
r_\phi(F_4,D_4,B).
\]
Nhưng giới hạn:
\[
R_{HF}
=
B\odot R_{edge}.
\]
Final:
\[
\boxed{
D_4'
=
D_4
+
\Delta D_{PCA}
+
B\odot R_{edge}
}
\]
Đây là decomposition rất rõ:
\[
\boxed{
\text{low-frequency/global}
+
\text{high-frequency/edge}
}
\]
15. Tôi còn muốn PCA branch biết edge tồn tại
Không để hai branch hoàn toàn độc lập.
Project edge map vào PCA modes:
\[
b_i
=
\sum_p
|e_i(p)|B_p.
\]
Nếu mode \(i\) có support lớn trên edge:
\[
b_i\uparrow.
\]
Ta đưa:
\[
b_i
\]
vào PCA token.
Attention lúc đó biết:
mode này rất liên quan tới boundary → đừng update quá aggressive bằng global dynamics.

Có thể gate velocity:
\[
\dot z_i
=
(1-\rho b_i)
f_i(z,C).
\]
Tức PCA không được phá những mode tập trung ở high-frequency boundary.
16. Một formulation mạnh hơn: spectral split
Thay vì chỉ global PCA:
\[
E=
[E_L,E_H].
\]
Low-frequency modes:
\[
E_L
\]
được xử lý bằng attention ODE.
High-frequency modes:
\[
E_H
\]
chỉ được active gần edge.
\[
\Delta x
=
E_Lz_L+
B\odot E_Hz_H.
\]
Đây là một concept rất đẹp:
\[
\boxed{
\text{spectral dynamics}
+
\text{spatial discontinuity gating}
}
\]
Tuy nhiên tôi sẽ để nó là phase 2; bản đầu chỉ cần PCA global + explicit edge residual.
17. Câu hỏi: PCA có thực sự đủ representation không?
Không được assume.
BrainODE làm hẳn explained-variance analysis và chọn \(K=150\); họ đạt khoảng 98% compactness cho các brain structures. OpenReview
Depth paper của mình cũng phải làm:
\[
EVR(K)
=
\frac{
\sum_{i=1}^{K}\lambda_i
}{
\sum_i\lambda_i
}.
\]
Test:
\[
K=
16,32,64,128,256.
\]
Và quan trọng hơn:
\[
RMSE_{PCA-recon}
\]
trên GT residual.
Ví dụ nếu:
\[
K=64
\]
mà oracle PCA reconstruction floor vẫn:
\[
0.95m,
\]
thì architecture chết từ đầu.
Nếu oracle:
\[
0.4-0.5m,
\]
thì latent space có tiềm năng.
Đây phải là experiment đầu tiên trước khi train ODE.
18. Tôi thậm chí sẽ dùng PCA trên correction thay vì absolute depth
Điểm này tôi khá chắc.
Absolute scene:
\[
X_{GT}
\]
variance quá lớn:
- xe ở vị trí khác;
- building khác;
- object khác;
- road scene khác.
Residual:
\[
X_{GT}-X_{V3}
\]
có distribution hẹp hơn nhiều.
PCA sẽ cần ít mode hơn.
Có thể:
\[
K_{residual}\ll K_{absolute}.
\]
Đây vừa giúp compute vừa giúp modeling.
19. Attention conditioning nên chứa gì?
Tôi sẽ cho:
\[
C=
[
z,
F_{global},
S_{stats},
E_{stats},
R_{range}
].
\]
Trong đó:
Global RGB/depth context
\[
F_{global}
=
GAP(F_{16}).
\]
Sparse sensor statistics
Ví dụ:
\[
S_{stats}
=
[
density,
mean,
variance,
coverage_{near},
coverage_{far}
].
\]
Edge statistics
\[
E_{stats}
=
[
edge\ density,
orientation,
mean\ jump
].
\]
Range
Có thể lấy histogram:
\[
R_{range}
=
[r_{0-20},r_{20-40},...].
\]
Attention vector field lúc đó scene-conditioned.
20. Attention ODE block cụ thể
Cho \(K\) PCA tokens:
\[
H^0\in\mathbb R^{K\times d}.
\]
Mỗi pseudo-step:
\[
A^k
=
\operatorname{softmax}
\left(
\frac{
Q^k(K^k)^T
}{
\sqrt d
}
+
G_{edge}
\right).
\]
Sau đó:
\[
V^k=A^kV_h^k.
\]
Velocity:
\[
\dot z^k
=
W_oV^k.
\]
Update:
\[
z^{k+1}
=
z^k+
\Delta\tau\dot z^k.
\]
Đây gần BrainODE nhưng mode-wise attention rõ ràng hơn.
21. Điểm novelty có thể mạnh hơn PT-PSF ở đâu?
PT-PSF trước đó giải:
\[
\text{local geometric transport}.
\]
Hướng PCA này giải:
\[
\boxed{
\text{global scene deformation dynamics}
}
\]
ở rất ít dimension.
Depth SOTA thường mạnh ở:
- propagation;
- local affinity;
- coarse-to-fine;
- high-res refinement.
Nhưng formulation:
\[
\text{coarse metric depth}
\rightarrow
\text{PCA deformation space}
\rightarrow
\text{attention-conditioned latent vector field}
\]
không phải pattern phổ biến trong depth completion mà tôi thấy qua literature search hiện tại.
Đặc biệt nó tách:
\[
\text{global correction}
\]
khỏi:
\[
\text{edge correction}.
\]
Đó là research story mạnh.
22. Nhưng đừng bỏ hoàn toàn Piecewise Surface idea
Tôi sẽ kết hợp cả hai, nhưng không làm architecture quá phức tạp.
Tên tổng thể:
Piecewise Spectral Surface Flow — PSSF
Có hai dynamics:
Global spectral flow
\[
\frac{dz}{d\tau}
=
f_\theta(z,C,\tau)
\]
→ sửa low-frequency/global geometry.
Local edge flow
\[
R_E
=
g_\phi(F,D,B)
\]
→ sửa discontinuities.
Final:
\[
\boxed{
X_1
=
X_0+
E z(1)+
B\odot R_E
}
\]
Tôi thấy đây là formulation tốt nhất hiện tại.
23. Nó kế thừa ShapeFlow + BrainODE + Attention-PCA rất sạch
ShapeFlow
Lấy:
\[
\boxed{\text{deformation instead of regeneration}}
\]
BrainODE
Lấy:
\[
\boxed{
\text{PCA shape space}
+
\text{velocity in PCA coefficients}
+
\text{attention-conditioned ODE}
}
\]
BrainODE thực sự output một 150-D velocity vector trong shape space. OpenReview
Attention-based PCA 2026
Lấy motivation:
\[
\boxed{
\text{attention naturally interacts with principal spectral directions}
}
\]
Trong canonical Gaussian PCA setting, paper chứng minh parameters của attention align với principal eigenvectors. Imo Université Paris-Saclay
AnchorFlow
Giữ:
\[
\boxed{
\text{metric sensor anchor}
+
\text{lightweight backbone}
+
\text{fixed-step inference}
}
\]
V5 Piecewise
Giữ:
\[
\boxed{
\text{edge/discontinuity must remain piecewise}
}
\]
24. Đây là chỗ tôi thấy novelty Q1 tiềm năng nhất
Không nên claim novelty là:
PCA + attention.

Quá yếu.
Không:
Neural ODE for depth completion.

Cũng chưa đủ.
Contribution phải là toàn formulation:
A low-dimensional spectral deformation formulation for sparse metric depth completion, where an attention-conditioned vector field evolves a coarse camera-centric surface in a PCA correction space, while a complementary piecewise high-frequency branch explicitly preserves visibility discontinuities.

Ba contribution có thể là:
1. PCA deformation space for depth completion
\[
\Delta X\approx Ez.
\]
Không model full scene, model correction manifold.
2. Attention-conditioned latent surface dynamics
\[
\frac{dz}{d\tau}
=
f_\theta(z,C,\tau).
\]
Global interaction chỉ trong low-rank spectral space.
3. Edge-preserving spectral/residual decomposition
\[
X=
X_0+
\underbrace{Ez}_{global}
+
\underbrace{B\odot R_E}_{boundary}.
\]
PCA không được phép wash out occlusions.