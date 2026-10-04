Có. Sau khi mở rộng ra 3D deformation, scene flow, neural fields, differential geometry, medical registration, dynamic NeRF/Gaussian và depth completion, tôi nghĩ hướng V5 hiện tại vẫn còn hơi “mỏng”: transport weights + barrier dễ bị reviewer xem là một biến thể SPN/anisotropic propagation.
Muốn nâng thành một contribution đủ sâu để nhắm journal Q1, tôi sẽ đổi trọng tâm từ:
\[
\boxed{\text{learned directional interpolation}}
\]
sang:
\[
\boxed{\textbf{geometry-correct vector transport on a piecewise metric surface}}
\]
Cụ thể tôi đề xuất Parallel-Transport Piecewise Surface Flow — PT-PSF.
1. Trước hết: các dòng paper nền tảng thật sự liên quan
Không chỉ ShapeFlow. Ý tưởng của bạn nằm ở giao điểm của 5 dòng nghiên cứu.
Dòng	Công trình quan trọng	Insight cần lấy
Continuous flow / deformation	Neural ODE, FFJORD, ShapeFlow, CanFields	Geometry được evolve bằng vector field, không decode lại từ đầu
Shape deformation / registration	ARAP, Neural Cages, VoxelMorph, Deep Implicit Templates, DIF-Net, NeuroMorph	Vector deformation cần geometric constraints, correspondence, rigidity
3D scene flow	FlowNet3D, PointPWC-Net, RAFT-3D, HCRF-Flow, RigidFlow	Vector không nên independent per-point; cần local structure / rigid groups / multiscale
Surface differential geometry	Parallel Transport CNN, Field Convolution, Vector Diffusion, Vector Neurons	Vector trên hai tangent planes không thể cộng trực tiếp
Depth completion	NLSPN, TWISE, DySPN, BP-Net, TPVD, GBPN	Sparse evidence phải được propagate, nhưng không được đi qua wrong surface


Đây là literature map quan trọng nhất.
2. Neural ODE → ShapeFlow: nguồn gốc của “velocity field”
Neural ODE thay một chuỗi layer rời rạc bằng:
\[
\frac{d\mathbf x(t)}{dt}
=
f_\theta(\mathbf x(t),t).
\]
Tức neural network không trực tiếp output trạng thái cuối, mà output đạo hàm / velocity của trạng thái. Proceedings NeurIPS
FFJORD tiếp tục ý tưởng này sang continuous normalizing flow, cho thấy một learned continuous dynamics có thể tạo mapping invertible mà không phải bó architecture thành các transform rời rạc. OpenReview
ShapeFlow lấy đúng abstraction đó sang 3D:
\[
\mathbf x(1)
=
\mathbf x(0)
+
\int_0^1
\mathbf v_\theta(\mathbf x(t),t)\,dt.
\]
Điểm hay là source geometry được advect thay vì reconstruct lại. ShapeFlow còn phân tích invertibility, self-intersection và divergence-free volume preservation. NeurIPS
Bài học cho AnchorFlow
Không phải:
\[
D_4\rightarrow CNN\rightarrow\Delta D.
\]
Mà:
\[
D_4
\rightarrow
\boxed{\text{estimate geometric vector field}}
\rightarrow
\boxed{\text{evolve surface}}
\rightarrow
D_4'.
\]
Đây mới là connection thật sự với ShapeFlow.
3. CanFields 2025 cho một lesson còn quan trọng hơn
CanFields ICCV 2025 vẫn dùng smooth velocity field + diffeomorphic flow, nhưng thêm một dynamic consolidator và confidence để xử lý:
- missing regions;
- noise;
- sparse point clouds;
- varying sequence length. Open Access CVF
Điều này rất gần bài toán của bạn về mặt philosophy:
\[
\text{sparse/noisy geometry}
+
\text{confidence}
+
\text{velocity field}
\rightarrow
\text{coherent geometry}.
\]
Nhưng CanFields giải bài toán 4D interpolation theo thời gian.
AnchorFlow của bạn sẽ giải một bài khác:
\[
\boxed{
\text{single-frame sparse metric completion}
}
\]
và “time” trong flow của bạn chỉ là pseudo-time của optimization/refinement, không phải thời gian vật lý.
Đây là distinction rất quan trọng cho paper.
4. Scene Flow nói cho chúng ta biết: vector độc lập từng pixel là chưa đủ
FlowNet3D CVPR 2019 học trực tiếp:
\[
\mathbf v_i\in\mathbb R^3
\]
cho mỗi point cloud point. Open Access CVF
PointPWC-Net cải thiện bằng coarse-to-fine:
\[
V^{coarse}
\rightarrow
\text{warp}
\rightarrow
V^{fine}.
\]
Nó dùng 3D cost volume và upsample flow từ coarse sang fine. ECVA
RAFT-3D đi xa hơn: không chỉ predict XYZ translation mà iteratively update một dense SE(3) field, đồng thời dùng rigid-motion embeddings để các pixel thuộc cùng object có transformation tương thích. Open Access CVF
RigidFlow cũng chỉ ra local regions của real scenes thường được mô tả tốt hơn bằng piecewise rigid transforms thay vì point-wise arbitrary flow. Open Access CVF
Lesson
Nếu V5 hiện tại chỉ output:
\[
v_p=(v_x,v_y,v_z)
\]
independently cho mỗi pixel rồi smooth chúng:
\[
v_p\leftarrow\sum_qw_{pq}v_q,
\]
thì vẫn còn yếu.
Phải hỏi:
Hai vector \(v_p\) và \(v_q\) thực sự có nằm trong cùng hệ tọa độ hình học để được average không?

Và đây dẫn tới phần thú vị nhất.
5. Đây là insight tôi thấy mạnh nhất: parallel transport
Giả sử surface cong:
               q
              /
             /
------------p

Pixel \(p\) có tangent plane:
\[
T_p\mathcal M
\]
còn \(q\):
\[
T_q\mathcal M.
\]
Một vector:
\[
v_p\in T_p\mathcal M
\]
và:
\[
v_q\in T_q\mathcal M
\]
không thực sự nằm trong cùng vector space địa phương.
Nên phép:
\[
v_p+v_q
\]
một cách naïve là không đúng về differential geometry.
Đây chính là vấn đề mà các surface-network papers xử lý bằng parallel transport.
PFCNN CVPR 2020 xây local tangent frames và transport giữa tangent planes trên surface. Open Access CVF
Field Convolutions ICCV 2021 kết hợp surface convolution trực tiếp với parallel transport của vector fields. Open Access CVF
Vector Diffusion Maps còn xây hẳn connection Laplacian để diffusion/interpolate vector fields trên manifold thay vì scalar fields. PMC
Đây chính là thứ tôi sẽ đưa vào AnchorFlow.
Tôi chưa thấy trong các depth-completion papers tôi kiểm tra một formulation dùng explicit parallel transport của learned surface velocity field trên reconstructed metric depth manifold.
Đây là gap đáng theo.
6. Vấn đề của interpolation V5 hiện tại
Giả sử V5 hiện có:
\[
D_p^{k+1}
=
D_p^k+
\sum_qT_{pq}
(D_q^k-D_p^k).
\]
Nó tốt hơn CNN residual, nhưng toán học của nó gần:
\[
\boxed{\text{anisotropic scalar diffusion}}
\]
hơn là ShapeFlow.
Tức \(T_{pq}\) chỉ trả lời:
Lấy bao nhiêu depth từ neighbor \(q\)?

NLSPN đã học non-local neighbors + affinities từ 2020. ML Anthology
DySPN đã làm dynamic affinity theo iteration và diffusion suppression. AAAI
BP-Net đã học propagation coefficients conditioned trên spatial + radiometric relations. Open Access CVF
GBPN 2026 thậm chí học scene-specific graph, adaptive non-local edges rồi chạy Gaussian belief propagation. arXiv
Nên nếu paper của bạn chỉ là:
\[
T_{pq}
+
\text{edge barrier}
\]
thì reviewer hoàn toàn có thể nói:
“This is another affinity propagation variant.”

7. Nâng cấp tôi chốt: PT-PSF
Parallel-Transport Piecewise Surface Flow
Đây là pipeline tôi thấy vừa có mathematical story, vừa giữ được mục tiêu edge deployment.
Bước 1 — Lift \(D_4\) thành metric surface thật sự
Tại pixel:
\[
p=(u,v)
\]
có:
\[
D_p.
\]
Back-project:
\[
\boxed{
X_p
=
D_pK_4^{-1}
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
}
\]
với:
\[
X_p\in\mathbb R^3.
\]
Bây giờ network không còn xử lý một ma trận scalar:
\[
D_4
\]
mà có một point-sampled surface:
\[
\mathcal S_4=\{X_p\}.
\]
Đây là khác biệt lớn.
8. Bước 2 — xây local tangent frame
Từ neighborhood 3D:
\[
X_p,X_{p+x},X_{p+y}
\]
tính surface normal:
\[
n_p
=
\frac{
(X_{p+x}-X_p)
\times
(X_{p+y}-X_p)
}{
\|
(X_{p+x}-X_p)
\times
(X_{p+y}-X_p)
\|
}.
\]
Hoặc dùng weighted local plane fit.
Sau đó lấy hai tangent directions:
\[
t_p^1,\quad t_p^2
\]
sao cho:
\[
t_p^1\perp n_p,
\qquad
t_p^2=n_p\times t_p^1.
\]
Local frame:
\[
F_p=
[
t_p^1,t_p^2,n_p
].
\]
9. Bước 3 — network không output XYZ arbitrary nữa
Head dự đoán:
\[
\xi_p=
(a_p,b_p,c_p).
\]
Sau đó:
\[
\boxed{
v_p=
a_pt_p^1+
b_pt_p^2+
c_pn_p
}
\]
Ý nghĩa rất rõ:
\[
a,b
\]
= transport dọc surface.
\[
c
\]
= correction theo surface normal.
Đây rất hay vì bạn đã disentangle:
\[
\boxed{
\text{surface transport}
}
\]
và:
\[
\boxed{
\text{surface correction}
}
\]
thay vì để network tự học XYZ vô nghĩa.
10. Đây là chỗ interpolation vector hiện tại được thay hoàn toàn
Giả sử \(q\) là neighbor của \(p\).
Nó có:
\[
v_q\in T_q\mathcal M.
\]
Không average trực tiếp:
\[
v_p+\color{red}{v_q}.
\]
Trước tiên tìm minimal rotation:
\[
R_{q\rightarrow p}
\]
align:
\[
n_q\rightarrow n_p.
\]
Sau đó:
\[
v_{q\rightarrow p}
=
R_{q\rightarrow p}v_q.
\]
Bây giờ mới interpolate:
\[
\boxed{
\bar v_p
=
\frac{
\sum_{q\in\mathcal N(p)}
w_{pq}B_{pq}
R_{q\rightarrow p}v_q
}{
\sum_qw_{pq}B_{pq}+\epsilon
}
}
\]
Đây chính là parallel-transport vector interpolation.
Nó khác fundamentally với:
\[
\sum_qw_{pq}v_q.
\]
11. Piecewise barrier vẫn giữ — nhưng bây giờ có nghĩa hình học hơn
TWISE đã chứng minh interpolation qua foreground/background là nguyên nhân trực tiếp của depth smearing. Open Access CVF
Vì vậy surface không thể coi là một manifold toàn cục:
car             background
████████ │
████████ │            wall
  12m    │            45m

Tại edge:
\[
\mathcal M_1
\;\not\leftrightarrow\;
\mathcal M_2.
\]
Do đó \(B_{pq}\) không chỉ dựa RGB edge.
Tôi sẽ dùng metric plane compatibility:
\[
r_{pq}^{plane}
=
|n_p^T(X_q-X_p)|.
\]
Nếu \(q\) nằm trên cùng local surface:
\[
r_{pq}^{plane}\approx0.
\]
Nếu khác surface:
\[
r_{pq}^{plane}\gg0.
\]
Symmetric version:
\[
r_{pq}
=
|n_p^T(X_q-X_p)|
+
|n_q^T(X_p-X_q)|.
\]
Rồi:
\[
B_{pq}
=
\sigma
\left[
g_\theta(
r_{pq},
|D_p-D_q|,
\Delta F_{pq},
\Delta I_{pq}
)
\right].
\]
Đây mạnh hơn rất nhiều so với:
\[
B=f(\text{RGB edge}).
\]
12. Tại sao cái này hơn V5 hiện tại?
V5 hiện tại gần:
pixel p
 ↓
predict affinities
 ↓
weighted neighboring depths

PT-PSF:
pixel p
 ↓
metric 3D point Xp
 ↓
local surface frame
 ↓
3D tangent/normal velocity
 ↓
parallel transport neighbor vectors
 ↓
visibility barrier
 ↓
surface-consistent transport
 ↓
depth update

Tức chuyển từ:
\[
\boxed{\text{learned interpolation}}
\]
sang:
\[
\boxed{\text{learned differential-geometric transport}}.
\]
Đây mới là novelty level khác.
13. Nhưng tôi KHÔNG khuyên warp point cloud thật trong inference
Một implementation đúng ShapeFlow tuyệt đối sẽ làm:
\[
X^{k+1}=X^k+\Delta t\,v(X^k)
\]
rồi reproject:
\[
X\rightarrow(u',v').
\]
Điều này kéo theo:
- scatter;
- grid sampling;
- holes;
- collision;
- z-buffer;
- irregular memory.
Không hợp Edge AI.
Thay vào đó vector chỉ điều khiển fixed local stencil.
Cho:
\[
e_{pq}=X_q-X_p.
\]
Projection lên tangent plane:
\[
e_{pq}^{T}
=
(I-n_pn_p^T)e_{pq}.
\]
Affinity directional:
\[
s_{pq}
=
\frac{
\bar v_p^Te_{pq}^T
}{
\tau(\|\bar v_p\|\|e_{pq}^T\|+\epsilon)
}.
\]
Sau đó:
\[
w_{pq}
=
\operatorname{softmax}_q
\left(
s_{pq}+\log(B_{pq}+\epsilon)
\right).
\]
Depth update vẫn cực rẻ:
\[
\boxed{
D_p^{k+1}
=
D_p^k+
\eta_p
\sum_q
w_{pq}
(D_q^k-D_p^k)
}
\]
nhưng direction của interpolation bây giờ đến từ 3D surface vector geometry.
Không phải arbitrary CNN affinity nữa.
14. Sparse LiDAR trở thành “source term” của flow
Đây là một điểm tôi rất thích.
Sparse measurement:
\[
S_p
\]
không phải hard overwrite.
Existing sensor trust của AnchorFlow:
\[
g_p\in[0,1]
\]
có thể trở thành forcing term:
\[
\boxed{
D_p^{k+1}
=
D_p^k
+
\eta_p
\sum_qw_{pq}(D_q^k-D_p^k)
+
\gamma g_pM_p(S_p-D_p^k)
}
\]
Interpretation:
\[
\text{surface dynamics}
=
\text{internal transport}
+
\text{sensor forcing}.
\]
Rất giống một PDE.
Đây là research story đẹp hơn rất nhiều.
15. Nó nối với ShapeFlow như thế nào?
Bạn có thể viết continuum form:
\[
\frac{\partial D_p(t)}{\partial t}
=
F_\theta
(
D(t),
v(t),
B,
S
).
\]
Current model dùng:
\[
D^{k+1}
=
D^k+\Delta tF(D^k)
\]
→ Euler discretization của flow.
Như vậy connection ShapeFlow/Neural ODE hoàn toàn rõ:
\[
\boxed{
\text{continuous surface dynamics}
\rightarrow
\text{fixed-step edge-friendly discretization}
}
\]
Nhưng không cần ODE solver khi deploy.
16. Có thể nâng Euler thành RK2/Heun
Một cải tiến nhỏ nhưng rất hợp story.
Euler:
\[
k_1=F(D^k)
\]
\[
D^{k+1}=D^k+\Delta tk_1.
\]
Thay bằng Heun:
\[
k_1=F(D^k)
\]
\[
\tilde D=D^k+\Delta tk_1
\]
\[
k_2=F(\tilde D)
\]
\[
\boxed{
D^{k+1}
=
D^k+
\frac{\Delta t}{2}(k_1+k_2)
}
\]
Divergence-Free Shape Correspondence cũng từng sử dụng second-order Runge–Kutta để integrate deformation field. Wiley Online Library
Nhưng lưu ý:
RK2 không phải novelty.

Nó chỉ là một numerically better realization của contribution chính.
Ablation:
\[
Euler1,\;
Euler2,\;
RK2
\]
là đủ.
17. Connection Laplacian cho bạn một loss rất đẹp
Nếu cùng surface, vector của hai điểm sau parallel transport nên tương thích:
\[
\boxed{
L_{conn}
=
\sum_{(p,q)}
B_{pq}
\|
v_p-
R_{q\rightarrow p}v_q
\|_1
}
\]
Không phải conventional:
\[
\|v_p-v_q\|.
\]
Vì \(v_p,v_q\) thuộc tangent frames khác nhau.
Đây chính là tư duy của connection Laplacian/vector diffusion. Vector Diffusion Maps được xây để diffuse và interpolate vector fields trên manifold bằng connection-aware operators. PMC
Field Convolution cũng dùng parallel transport vì lý do tương tự. Open Access CVF
18. Đừng dùng divergence-free blindly
ShapeFlow dùng:
\[
\nabla\cdot v=0
\]
để bảo toàn volume.
Đưa loss đó sang depth completion sẽ khó biện minh.
Depth map không phải incompressible fluid.
Tôi không khuyên:
\[
L_{div}=|\nabla\cdot v|.
\]
Tương tự:
- volume preservation: không hợp;
- global bijectivity: không hợp vì occlusion thật sự tạo discontinuity.
Thứ phù hợp hơn là:
\[
\boxed{\text{within-patch connection smoothness}}
\]
-
\[
\boxed{\text{cross-patch discontinuity}}.
\]
Tức:
\[
\text{smooth inside}
+
\text{discontinuous at boundary}.
\]
19. Đây cũng chính là điểm khác CanFields / ShapeFlow
ShapeFlow:
\[
\text{globally smooth deformation}.
\]
CanFields:
\[
\text{diffeomorphic smooth motion}.
\]
Nhưng HyperNeRF đã chỉ ra một vấn đề fundamental: một continuous deformation field khó biểu diễn topology changes, vì topology change cần discontinuity. SIGGRAPH Asia 2021
Depth image có đúng phenomenon đó tại occlusion:
\[
12m\;|\;45m.
\]
Do đó contribution của bạn có thể được đặt là:
\[
\boxed{
\textbf{piecewise continuous flow rather than globally continuous flow}
}
\]
Mỗi visible surface patch:
\[
\mathcal M_i
\]
có smooth field.
Nhưng qua:
\[
\partial\mathcal M_i
\]
thì connection bị cắt.
Đây là một lập luận khá mạnh về mặt geometry.