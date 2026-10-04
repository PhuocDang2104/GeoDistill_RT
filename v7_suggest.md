Tôi đã đọc kỹ cả spec V6 lẫn train_log (5).csv, rồi đối chiếu với các hướng depth completion mạnh tới 03/10/2026. Kết luận chính của tôi là:
Không nên tiếp tục nâng V6 bằng cách làm “parallel transport” của vector 3D phức tạp hơn.
Điểm yếu hiện tại không phải thiếu capacity, mà là representation của ConnectionMetric4 không khớp trực tiếp với đại lượng cuối cùng cần sửa: scalar depth dọc camera ray.
Hướng V7 tôi thấy mạnh hơn rõ rệt là Projective 2-Jet Transport + Jet Phase Rendering: biểu diễn một patch depth bằng giá trị inverse-depth + gradient + curvature bậc hai, transport trạng thái này analytically giữa các pixel, rồi dùng cùng trạng thái đó để render chính xác 4 phase con khi upsample D4→D2.
Nó vừa khác V6 về bản chất, vừa tránh overlap quá mạnh với BP-Net/OGNI/GBPN/FieldConv, đồng thời còn nhẹ hơn V6 về Conv MAC.

1. Trước hết: log V6 đang nói điều gì?
Đây là phần quan trọng nhất, vì nó cho biết module nào thật sự đáng giữ.
Metric	đầu fine-tune	epoch 14	thay đổi
Global RMSE	1.02328 m	1.00929 m	−13.99 mm
MAE	0.26243	0.25802	−4.41 mm
RGB-edge RMSE	1.44989	1.43044	−19.45 mm
GT boundary 1 px	1.89045	1.87496	−15.49 mm
D4 base → surface	1.80627 → 1.78222		−24.05 mm
D4 surface → connection	1.78222 → 1.78164		−0.58 mm
D2 base → phase	1.28538 → 1.25636		−29.02 mm


Đây là bằng chứng khá mạnh:
V5 piecewise surface transport đang có tác dụng. PhaseMetric2 cũng đang có tác dụng rõ. Nhưng ConnectionMetric4 gần như không cải thiện native D4.
Điều đáng chú ý hơn nữa là ConnectionMetric4 không hề bị dead branch:
Internal quantity	đầu	cuối
connection abs delta	0.00497 m	0.03737 m
tangent magnitude	0.00583	0.06921
normal magnitude	0.01522	0.05826
neighbour mass	0.6876	0.7308


Tức là branch đã mở, vector correction đã lớn lên, neighbour mixing cũng rất mạnh, nhưng cuối cùng D4 chỉ tốt thêm ~0.58 mm.
Đây không còn giống vấn đề “train thêm epoch là được”. Nó giống representation/conditioning problem hơn.
2. Vấn đề toán học quan trọng nhất của V6
V6 dự đoán
\[
v_p=c_a t_1+c_b t_2+c_n n
\]
sau đó cuối cùng lại chỉ lấy
\[
\Delta D_p
=
\frac{r_p^T \bar v_p}
{r_p^T r_p}.
\]
Điều này tạo ra một vấn đề rất cơ bản.
Hai bậc tự do của vector gần như không quan sát được tại pixel hiện tại
Output thực sự chỉ phụ thuộc vào projection của \(v\) lên \(r\).
Mọi vector \(z\) thỏa
\[
r_p^Tz=0
\]
đều cho
\[
\Delta D_p=0.
\]
Trong \(\mathbb R^3\), null space đó có dimension 2.
Nói đơn giản:
Head dự đoán 3 DOF, nhưng tại pixel hiện tại task chỉ cần 1 scalar DOF.

Hai DOF còn lại chỉ có thể trở nên hữu ích nếu sau neighbor transport chúng xoay thành thành phần có projection lên camera ray.
Đó là một supervision path khá yếu và gián tiếp.
Điều này khớp rất đẹp với log:
- tangent magnitude tăng lên ~69 mm,
- normal magnitude ~58 mm,
- neighbour mass ~0.73,
- nhưng native D4 gain vẫn <1 mm.
Đây là lý do tôi không khuyên tăng ConnectionMetric từ 4 → 8 neighbors hay thêm attention.
Nó sẽ tăng computation nhưng không giải quyết under-determined representation.
3. Rủi ro novelty của V6 hiện tại
Có một vấn đề khác quan trọng nếu mục tiêu là journal mạnh.
Việc transport tangent vectors trên surfaces không phải vùng trắng. PFCNN dùng parallel frames dựa trên locally flat connections, còn Field Convolutions kết hợp surface convolution với parallel transport. arXiv
Ngay trong depth completion, ICCV 2019 đã dùng depth-normal constraints và anisotropic diffusion của plane-origin distance để truyền geometry giữa các pixel. Open Access CVF
Vì vậy claim kiểu:
“chúng tôi đưa discrete connection / normal transport vào depth completion”

sẽ rất dễ bị reviewer hỏi:
Nó khác gì một simplification của surface connection + geometry-aware propagation?

V6 có khác, nhưng novelty story sẽ khó bảo vệ hơn mức cần thiết.
4. Landscape depth completion hiện tại
Các hướng mạnh gần đây đang chiếm khá rõ từng territory.
Work	Ý tưởng chính	Khoảng trống còn lại
BP-Net, CVPR 2024	bilateral propagation depth ngay từ early stage dựa radiometric + spatial distance	propagation vẫn chủ yếu learned scalar affinity, không mô hình local differential surface state theo camera projection Open Access CVF
TPVD, CVPR 2024 Oral	explicit 3D geometry qua tri-perspective views + recurrent 2D–3D–2D fusion	geometry mạnh nhưng architecture phức tạp hơn cho edge deployment Open Access CVF
DFU, CVPR 2024	giữ dense decoder information khi upsample; adaptive guidance	cho thấy upsampling stage rất quan trọng, nhưng không render geometry analytically Open Access CVF
OGNI-DC, ECCV 2024	refine depth-gradient field rồi differentiable integration	optimization/gradient-field territory đã khá đông; iterative ECVA
Flexible DC / MSPN, CVPR 2024	robustness với varying sparse-density	practical sparsity robustness đã thành tiêu chí quan trọng Open Access CVF
DMD³C, CVPR 2025	foundation-model distillation	KD nên là training aid, không nên là novelty chính của V7 Open Access CVF
FANet, TNNLS 2025	training-only multimodal alignment, không tăng inference cost	rất đáng học về philosophy edge deployment IEEE Xplore
DSRD, RA-L 2025	hardware-efficient dual branch + structural gating	edge territory hiện đã yêu cầu accuracy–runtime tradeoff thực tế ResearchGate
GBPN, ECCV 2026	dynamically learned MRF + adaptive non-local edges + Gaussian belief propagation	khiến hướng “thêm learned graph/message-passing solver” trở nên khó claim novelty arXiv


Đặc biệt GBPN rất quan trọng đối với hướng của bạn: đừng biến V7 thành một MRF/uncertainty/message-passing model khác. Territory đó vừa có một paper rất mạnh năm 2026. Cvlibs
Official KITTI leaderboard hiện có các published methods quanh khoảng 0.676–0.685 m test RMSE, chẳng hạn DMD3C++, DMD³C, GBPN và BP-Net. Nhưng runtime trên bảng KITTI được đo trên các environment khác nhau nên không nên dùng trực tiếp làm latency Pareto. Cvlibs
5. Research gap tôi sẽ chọn cho AnchorFlow
Tôi sẽ chuyển gap từ:
“Existing methods lack geometric vector transport.”

sang một statement chặt hơn:
Existing depth-completion methods predominantly propagate scalar depth/affinity, optimize first-order depth gradients, diffuse planar quantities, or construct learned probabilistic graphs. Meanwhile, surface-vector methods transport generic tangent fields whose degrees of freedom are not directly aligned with the scalar camera-ray depth correction required by depth completion. An underexplored alternative is to propagate a compact, directly observable local differential model of the projective depth surface itself, and to reuse the same representation for metric refinement and sub-pixel reconstruction.

Đây là gap mạnh hơn.
Nó dẫn thẳng tới V7.
6. AnchorFlow V7 — Projective Jet Transport
Tên tôi khuyên dùng:
AnchorFlow V7 — Projective Jet Transport and Phase Rendering
hoặc paper-oriented hơn:
JetDC: Edge-Efficient Depth Completion via Projective Jet Transport
Core flow:
RGB ── MobileNetV4 ── F4/F8/F16/F32
Sparse + mask ── sparse pyramid
                    │
                    ▼
              existing decoder
                    │
             D16 → D8 → D4
                    │
      AnchorFlow + metric refinement
                    │
       V5 piecewise plane transport
                    │
                D4_surface
                    │
        ┌──────────────────────────┐
        │ Projective 2-Jet Head    │
        │ ξ + gradient + Hessian   │
        │ analytic jet transport   │
        │ barrier/confidence       │
        └──────────────────────────┘
                    │
                  D4_jet
                    │
        ┌──────────────────────────┐
        │ Jet Phase Rendering 4→2 │
        │ analytic subpixel query │
        │ + tiny learned residual │
        └──────────────────────────┘
                    │
                   D2
                    │
           existing D2 → D1
                    │
           phase detail / trust
                    │
          soft sensor fusion
                    ▼
                  Dfull

Điểm quan trọng:
bỏ hoàn toàn tangent frame \(t_1,t_2,n\), Rodrigues và ambient vector residual khỏi branch mới.
7. Tại sao dùng inverse depth 2-jet?
Đặt
\[
\xi=\frac{1}{D}.
\]
Một plane 3D perspective camera có một tính chất rất đẹp:
\[
\xi(u,v)=au+bv+c.
\]
Tức là:
inverse depth của một 3D plane là affine trên image coordinates.

Các local-plane methods đã khai thác họ ý tưởng này, nên bản thân affine inverse-depth không phải novelty. MDPI
Nhưng ta có thể đi một bước khác:
thay vì giả định patch luôn là plane, biểu diễn local surface bằng Taylor polynomial bậc hai.
Tại D4:
\[
\xi_p(s,t)
=
\xi_p
+
g_p^T
\begin{bmatrix}s\\t\end{bmatrix}
+
\frac12
\begin{bmatrix}s&t\end{bmatrix}
H_p
\begin{bmatrix}s\\t\end{bmatrix}.
\]
Trong đó
\[
g_p=
\begin{bmatrix}
g_x\\g_y
\end{bmatrix},
\qquad
H_p=
\begin{bmatrix}
h_{xx}&h_{xy}\\
h_{xy}&h_{yy}
\end{bmatrix}.
\]
Do đó mỗi pixel chỉ cần state:
\[
j_p=
[
\xi,\,
g_x,\,
g_y,\,
h_{xx},\,
h_{xy},\,
h_{yy}
].
\]
Tổng cộng 6 scalars.
Không có local tangent gauge.
Không có arbitrary vector direction.
Không có cross product.
Không có 3×3 rotation.
8. Ý nghĩa của 2-jet
Nếu
\[
H=0,
\]
V7 trở về một local projective plane.
Nếu
\[
H\neq0,
\]
nó có thể mô tả local bending:
- road curvature,
- vehicle body,
- poles,
- vegetation,
- rounded surfaces,
- transition smooth giữa planes.
Đừng gọi \(H\) là exact intrinsic curvature.
Tên chính xác hơn là:
second-order projective surface variation
hoặc
projective curvature jet.

Đây là điểm khác quan trọng với V5 piecewise plane transport.
9. Phần hay nhất: transport jet không cần Rodrigues
Giả sử jet tại neighbor \(q\), muốn biểu diễn polynomial của \(q\) tại center của \(p\).
Gọi
\[
\Delta=
\begin{bmatrix}
\Delta s\\
\Delta t
\end{bmatrix}.
\]
Translation của quadratic polynomial là analytic:
\[
\xi_{q\rightarrow p}
=
\xi_q
+
g_q^T\Delta
+
\frac12\Delta^TH_q\Delta.
\]
Gradient transform:
\[
g_{q\rightarrow p}
=
g_q+H_q\Delta.
\]
Và Hessian:
\[
H_{q\rightarrow p}=H_q.
\]
Đây là phần tôi đánh giá cao nhất của V7.
Không phải geodesic parallel transport.
Không cần gọi nó là connection.
Nó đơn giản là:
analytic translation of a local second-order projective surface model.

Điều đó tránh rất nhiều novelty baggage từ FieldConv/PFCNN. Open Access CVF
10. Dùng residual jet, không thay toàn bộ depth
Đừng để network trực tiếp predict toàn bộ jet.
Giữ V5 làm baseline tốt:
\[
j_p^{base}
=
(\xi,g,H)
\]
được estimate từ \(D4_{surface}\).
Head chỉ sinh
\[
\Delta j_p
=
[
\Delta\xi,
\Delta g_x,
\Delta g_y,
\Delta h_{xx},
\Delta h_{xy},
\Delta h_{yy}
].
\]
Sau transport:
\[
j'_p
=
j_p^{base}
+
\overline{\Delta j}_p.
\]
Như vậy:
- zero-init vẫn exact no-op;
- pretrained V5 không bị phá;
- head học refinement thay vì recreate surface;
- ablation rất sạch.
Đây phù hợp trực tiếp với cách bạn đang fine-tune V5 → V6.
11. Barrier-guided Projective Jet Transport
Tôi sẽ giữ V5 symmetric barriers.
Với 4-neighbor \(q\in N(p)\):
\[
w_{pq}
=
b_{pq}
\kappa_{pq}
c_{pq}.
\]
Trong đó:
- \(b_{pq}\): V5 learned barrier;
- \(\kappa_{pq}\): base geometric compatibility;
- \(c_{pq}\): lightweight predicted reliability.
Sau đó:
\[
\overline{\Delta j}_p
=
\left(1-\sum_qw_{pq}\right)\Delta j_p
+
\sum_q
w_{pq}
T_{q\rightarrow p}(\Delta j_q).
\]
Vẫn giữ
\[
\sum_qw_{pq}\le m_{\max}<1.
\]
Tôi sẽ thử
\[
m_{\max}\approx0.8.
\]
Không softmax neighbor.
Điểm này của V5/V6 hiện tại là đúng và nên giữ.
12. Chạy 2 transport steps, không cần 8-neighbor
Một lợi thế lớn của representation mới là transport gần như toàn elementwise.
Tôi sẽ dùng:
\[
T=2
\]
cross-neighbor iterations.
Một 4-neighbor stencil sau hai iteration đã có effective support rộng hơn mà không cần:
- attention,
- dynamic neighbor search,
- cost volume,
- graph construction,
- KNN.
Ablation:
\[
T=0,1,2,3
\]
sẽ rất đẹp.
Nếu \(T=2\) đạt saturation thì paper còn có một edge-efficiency story tốt.
13. Projective Jet Phase Rendering — contribution thứ hai
Đây là chỗ V7 có thể mạnh hơn hẳn V6.
Log của bạn nói PhaseMetric2 hiện tại rất đáng giá:
\[
1.28538
\rightarrow
1.25636\ \text{m}
\]
tức ~29 mm native D2 gain.
Thay vì để CNN tự học hoàn toàn 4 phase, ta có sẵn local 2-jet.
Với child location
\[
\delta_c=
\begin{bmatrix}
\delta s_c\\
\delta t_c
\end{bmatrix},
\]
evaluate:
\[
\xi_{p,c}
=
\xi'_p
+
g_p'^T\delta_c
+
\frac12
\delta_c^TH'_p\delta_c.
\]
Sau đó
\[
D_{p,c}^{jet}
=
\frac{1}{\xi_{p,c}}.
\]
Đó chính là continuous subpixel query.
Không resize một scalar D4 rồi đoán correction.
Ta hỏi local surface:
“Nếu ray đi qua phase con này, depth của surface local sẽ là bao nhiêu?”

14. Các phase offsets còn cực kỳ đẹp
D4 center cách nhau 4 full-resolution pixels.
Khi D4 → D2, bốn child centers chỉ lệch parent center khoảng:
\[
(\pm1,\pm1)
\]
full-resolution pixels.
Nếu dùng D4-cell coordinates thì chỉ là
\[
\delta_c
\in
\left\{
\left(-\frac14,-\frac14\right),
\left(\frac14,-\frac14\right),
\left(-\frac14,\frac14\right),
\left(\frac14,\frac14\right)
\right\}.
\]
Do đó Jet Phase Rendering cực rẻ và numerically stable.
Bốn polynomial evaluations.
Không grid_sample.
Không deformable convolution.
Không splatting.
Không point-cloud reproject.
15. Nhưng vẫn cần learned residual
Không hard-replace D2_base bằng \(D2_{jet}\).
Local quadratic model vẫn sai tại:
- occlusion boundaries,
- thin objects,
- repeated surfaces,
- vegetation,
- very noisy coarse geometry.
Do đó dùng jet như một analytic proposal, còn RGB/phase feature vẫn xử lý residual.
Một formulation phù hợp:
\[
e_{jet,c}
=
D^{jet}_{2,c}
-
D^{base}_{2,c}.
\]
Phase head nhận:
\[
F_{phase}
=
[
G_2,\,
D2_{base},\,
e_{jet},\,
sparse,\,
density,\,
innovation,\,
barrier
].
\]
Rồi sinh
\[
\Delta D_{2,c}
=
A(D_2)\tanh(h_c(F_{phase})).
\]
Như vậy final PW zero-init vẫn exact no-op.
Quan trọng là analytic proposal trở thành input rất informative, chứ không cưỡng ép output phải planar/quadratic.
Đây chính là kiểu hybrid tôi nghĩ hợp cho real-world nhất.
16. Tại sao 2-jet mạnh hơn vector V6?
V6 connection vector	V7 projective jet
3D residual vector	scalar surface field + derivatives
3 DOF → 1 ray-depth output	every coefficient has explicit spatial meaning
2D local null-space	no arbitrary ambient null-space
tangent frame required	no tangent frame
gauge/fallback axis issue	common image/projective coordinates
Rodrigues normal alignment	polynomial translation
mainly first-order tangent geometry	second-order local surface
3D tensor \(4\times3\)	6 scalar channels
phase head separate concept	same jet drives D4 and D2
geometry sophisticated but weakly observable	representation directly tied to depth function


Đây là lý do tôi thấy V7 này thuyết phục hơn về cả engineering lẫn paper story.
17. Một insight nữa: V6 đang mix neighbor quá mạnh
Cuối training:
\[
\text{neighbour mass}\approx0.731.
\]
Tức center chỉ còn khoảng 27% mass trong average connection.
Nhưng gain native D4 chỉ khoảng:
\[
0.58\text{ mm}.
\]
Nếu neighbour transport thực sự đang truyền useful correction field, với mass ~0.73 ta kỳ vọng tác động rõ hơn.
Một interpretation khá hợp lý là:
correction vectors đang bị transport/mix, nhưng khi cuối cùng project về camera rays, nhiều component triệt tiêu hoặc rơi vào weakly supervised directions.

Jet transport loại vấn đề đó vì neighbor state được translate trực tiếp vào predicted inverse depth tại center.
18. Một vấn đề khác của V6: tail loss
Log cuối:
\[
\text{SSE fraction}_{|e|>5m}
:
0.7842\rightarrow0.7817
\]
gần như không đổi.
Với \(>20m\):
\[
0.32304\rightarrow0.32320.
\]
Thậm chí nhích xấu đi.
Trong khi:
- 0–20 m tốt lên,
- 20–40 m tốt lên,
- 40–60 m hơi xấu,
- 60–80 m hơi xấu,
- 80–120 m tốt hơn nhưng vẫn khoảng 9.09 m RMSE.
Điều này cũng phù hợp với loss hiện tại.
Bạn gọi
\[
2\rho_{20}(e)
\]
là robust-MSE.
Nhưng khi
\[
|e|>20
\]
Huber chuyển thành linear tail, không còn squared tail.
Do đó với các extreme outlier đang thống trị RMSE, gradient không tăng tiếp theo \(e\).
Nếu mục tiêu chính thức là KITTI RMSE, thì câu:
“robust-MSE trực tiếp đánh SSE/outlier”

chỉ đúng ở vùng
\[
|e|\le20m.
\]
Ở vùng lớn hơn, nó cố tình downweight extreme errors.
Tôi sẽ không bỏ Huber hoàn toàn, nhưng sẽ không đặt nó làm “tail RMSE loss”.
19. Training objective V7 tôi sẽ sửa thế nào?
Không cần thêm 10 loss.
Giữ phần lớn V5 objective.
Thay new robust-MSE bằng một small reliable tail-SSE term chỉ trên valid GT không conflict:
\[
L_{tail}
=
\frac{1}{|T|}
\sum_{p\in T}
e_p^2.
\]
Trong đó \(T\) có thể là top 5–10% error pixels của batch hoặc pixels vượt một moving threshold.
Weight nhỏ:
\[
0.02\sim0.05.
\]
Mục tiêu không phải để architecture “cheat” RMSE, mà để tránh việc loss hiện tại có gradient gần constant chính ở nơi official metric phạt mạnh nhất.
Tôi sẽ ablate:
\[
L_{robust}
\quad \text{vs}\quad
L_{tail}
\quad \text{vs}\quad
\text{none}.
\]