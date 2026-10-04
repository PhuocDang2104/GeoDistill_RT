Kết quả V4 hiện tại giúp thu hẹp hướng nghiên cứu khá rõ: ContextWide có tác dụng, nhưng tăng capacity tiếp không còn là hướng đáng đầu tư nhất. Hướng đáng làm tiếp là một module geometric refinement chuyên xử lý discontinuity/edge, thay vì tiếp tục mở rộng backbone/context.
1. V4 có học thật, không phải warm-start “ảo”
CSV hiện có 13 epoch, từ 0 đến 12; chưa có epoch 13–14 trong file. Projection norm của branch mới tăng từ 0.244 ở epoch 0 lên 1.875 ở epoch 12, nên zero-init branch đã mở và thực sự được sử dụng. train_log (2)
Best global validation nằm ở epoch 7 / cumulative 37:
Metric	V3 baseline	V4 best e7	Thay đổi
RMSE	1.04537 m	1.03007 m	−1.46%
MAE	0.26874	0.26410	−1.73%
iRMSE	3.7651	3.6553	−2.92%
D0 native	2.0840 m	2.0294 m	−2.62%
D4 native	1.9066 m	1.8519 m	−2.87%
Edge RMSE	1.48945 m	1.45918 m	−2.03%


Best V4 row is directly visible ở epoch 7. train_log (2)
Điều này củng cố đúng hypothesis ban đầu của ContextWide:
\[
\boxed{\text{better coarse latent representation}
\rightarrow
\text{better D4/D0}
\rightarrow
\text{better final depth}}
\]
Không phải chỉ final head học lại sensor fusion.
2. Nhưng gain của V4 rất không đồng đều theo depth
Đây mới là phần quan trọng.
Ở epoch 7:
\[
RMSE_{0-20}:
0.4990\rightarrow0.4681
\]
tức khoảng −6.19%.
\[
RMSE_{20-40}:
1.4738\rightarrow1.4318
\]
tức khoảng −2.85%.
Nhưng:
\[
RMSE_{40-60}:
2.5686\rightarrow2.6067
\]
\[
RMSE_{60-80}:
3.9332\rightarrow3.9544
\]
\[
RMSE_{80-120}:
9.6219\rightarrow9.7570.
\]
train_log (2)
Tức V4 đang học rất tốt phần near/mid-range chiếm nhiều pixel, trong khi long-range geometry chưa được giải quyết.
Thú vị hơn nữa, ở các epoch khác long-range đôi lúc tốt hơn rất nhiều. Ví dụ epoch 3 đạt 80–120 m = 9.21284 m, thấp hơn V3, nhưng global RMSE khi đó lại 1.03712 m. train_log (2)
Điều này nói lên rằng model có khả năng sửa far geometry, nhưng current objective không tìm được một operating point đồng thời tối ưu near + far + edge.
3. Edge cũng cho cùng một tín hiệu
V3:
\[
RMSE_{\rm edge}=1.48945\,m
\]
V4:
\[
RMSE_{\rm edge}=1.45918\,m.
\]
Gain ≈ 2.03%, lớn hơn gain global 1.46%.
Tức context tốt hơn đã giúp boundary một chút.
Nhưng nó không giải quyết edge trực tiếp.
Đáng chú ý, tại best epoch 7 log của bạn có:
\[
L_{\rm edge}^{weighted}
\approx1.34\times10^{-5}
\]
và:
\[
L_{\rm teacher-edge}^{weighted}
\approx8.75\times10^{-6},
\]
trong khi:
\[
L_{\rm metric}\approx0.300,
\quad
L_{\rm RMSE}\approx0.217,
\quad
L_{\rm range}\approx0.208.
\]
train_log (2)
Không thể kết luận gradient edge yếu chỉ dựa vào scalar loss — muốn chắc chắn phải log gradient norm — nhưng về contribution value thì edge objective hiện cực nhỏ.
Đây là lý do tôi không kỳ vọng ContextWide tự học một representation boundary rất sắc dù tăng thêm hơn một triệu parameters.
4. Và V4 đang khá đắt cho gain này
Đây là điểm tôi sẽ cân nhắc nghiêm túc nếu paper của bạn muốn claim Edge AI / efficient depth completion.
V3:
\[
572\,017\ params,\quad 2.974G\ MAC
\]
V4:
\[
1\,857\,721\ params,\quad 6.344G\ MAC.
\]
Tức:
\[
Params \approx 3.25\times
\]
và:
\[
MAC \approx 2.13\times.
\]
Đổi lại hiện tại:
\[
RMSE:
1.0454\rightarrow1.0301
\]
≈ 1.46%.
Đây không phải kết quả xấu về accuracy, nhưng về efficiency thì rất khó biến câu chuyện thành:
“more computation gives a little more accuracy”

Đặc biệt DFU đã làm rất rõ câu chuyện giữ dense depth features qua coarse-to-fine với limited computational overhead. Open Access CVF
LP-Net cũng đã đi thẳng vào global-context → progressive high-frequency refinement bằng Laplacian pyramid. arXiv
OMNI-DC đã khai thác multiresolution depth integration. Open Access CVF
Vì vậy “bigger multiscale context pyramid” không phải nơi novelty mạnh nhất nữa.
5. Literature hiện tại cũng chỉ đúng hướng phải chuyển sang boundary geometry
Có bốn nhóm cạnh tranh trực tiếp với hướng tiếp theo.
Propagation. BP-Net propagates sparse measurements dựa trên spatial/radiometric relations từ rất sớm. Open Access CVF DSPN đã học deformable receptive field và affinity cho từng pixel từ 2020. arXiv GBPN 2026 còn đi xa hơn với learned non-local graph + Gaussian belief propagation. arXiv
Vì vậy chỉ thêm:
\[
(\Delta u,\Delta v)+\text{adaptive propagation}
\]
không đủ novelty.
Boundary-specific completion. Twin Surface CVPR 2021 đã chỉ ra rất rõ foreground/background interpolation gây depth smearing và dùng hai surface hypotheses tại occlusion boundary. Open Access CVF
Vì vậy chỉ thêm:
foreground depth + background depth

cũng không đủ.
High-frequency/detail. HFD-Teacher ICCV 2025 đã distill high-frequency geometry từ foundation depth model bằng adaptive local wavelets và topological constraint. Open Access CVF
Nên chỉ thêm edge loss/high-frequency teacher cũng không phải contribution đủ mạnh.
Explicit 3D geometry. TPVD CVPR 2024 đã explicitly model 3D geometry rồi geometric propagation; DeCoTR cũng uplift feature thành 3D point cloud. Open Access CVF
Vì vậy chỉ nói “project depth to 3D” cũng chưa đủ.
6. Hướng tôi chốt: Piecewise Metric Surface Flow
Tôi sẽ không tiếp tục ContextWide v5.
Tôi sẽ làm:
AnchorFlow v5 — Piecewise Metric Surface Flow (PMSF)
Ý tưởng:
At depth discontinuities, instead of refining depth by unconstrained image-space propagation, explicitly transport metric surface geometry along the local surface while blocking transport across occlusion boundaries.

Đây mới là phần thực sự lấy ý tưởng từ ShapeFlow nhưng biến nó thành đúng bài toán depth completion.
ShapeFlow:
\[
\frac{d\mathbf{x}}{dt}
=
f_\theta(\mathbf{x},t)
\]
với geometry liên tục.
PMSF:
\[
\frac{d\mathbf{x}}{dt}
=
f_\theta(\mathbf{x},F,S,K),
\]
nhưng chỉ trong local piecewise-visible surface.
Quan trọng:
\[
\mathbf{x}=(u,v,z)
\]
hoặc tốt hơn, lift bằng intrinsic:
\[
\mathbf X_p
=
D(p)K^{-1}
\begin{bmatrix}
u\\v\\1
\end{bmatrix}.
\]
Như vậy flow không đơn thuần là feature/image offset nữa.
Nó là metric surface correction.
7. Kiến trúc cụ thể tôi khuyên triển khai
Không thêm branch lớn kiểu ContextWide nữa.
Lấy feature ở 1/4 resolution, ngay sau D4 hiện tại:
P4 / F4 / D4 / sparse4 / mask4 / K
                │
                ▼
       Piecewise Surface Flow
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
   edge e     flow v    confidence c
       │        │         │
       └────────┼─────────┘
                ▼
      2 fixed flow steps
                │
                ▼
             D4+
                │
        existing D2 → D1

Tiny head thôi.
Không cần 192/96/48 channels.
Ví dụ width:
\[
32\rightarrow32\rightarrow
\{e,\mathbf v,c,r\}.
\]
Target nên là:
\[
<100k\ params
\]
và lý tưởng:
\[
<0.3-0.5G\ additional\ MAC.
\]
8. Flow phải là 3D metric flow, không phải DSPN thứ hai
Đây là distinction quan trọng nhất cho paper.
Predict:
\[
\mathbf v_p=
(v_u,v_v,v_z).
\]
Một fixed Euler update:
\[
\mathbf x_p^{k+1}
=
\mathbf x_p^k+
\alpha c_p
\mathbf v_p^k.
\]
với:
\[
k=0,1.
\]
Chỉ 2 steps, không adaptive ODE solver.
Như vậy vẫn lấy continuous-flow interpretation:
\[
\mathbf x(1)
=
\mathbf x(0)+
\int_0^1 f_\theta(\mathbf x(t),t)dt
\]
nhưng inference được discretize thành:
\[
\mathbf x^2
\approx
\mathbf x^0+
\alpha f_\theta(\mathbf x^0)
+
\alpha f_\theta(\mathbf x^1).
\]
Điều này hợp edge deployment hơn ShapeFlow gốc rất nhiều.
9. Nhưng điểm novelty thật sự là piecewise barrier
Depth scene không phải một manifold continuous duy nhất.
Ví dụ:
\[
D_{fg}=8m,
\qquad
D_{bg}=35m.
\]
Không được phép flow:
\[
8\rightarrow14\rightarrow22\rightarrow35.
\]
Do đó predict edge/barrier:
\[
e_p=\sigma(g_\theta(F_p,D_p)).
\]
Cho hai pixel \(p,q\):
\[
b_{pq}
=
\exp(
-\beta_D |D_p-D_q|
-\beta_E(e_p+e_q)
).
\]
Surface transport weight:
\[
a_{pq}
=
\frac{
b_{pq}\exp(s_{pq})
}{
\sum_{r\in\mathcal N(p)}
b_{pr}\exp(s_{pr})
}.
\]
Nếu hai pixel nằm khác side của occlusion:
\[
b_{pq}\rightarrow0.
\]
Nghĩa là:
\[
\boxed{
\text{flow along surface}
\neq
\text{flow across surface}
}
\]
Đây là distinction mà ShapeFlow nguyên bản không cần giải quyết vì nó giả định continuous deformation field; depth camera thì lại có visibility discontinuity.
10. Tôi còn khuyên không dùng full twin-surface representation
Twin Surface đã có paper rất mạnh từ CVPR 2021. Open Access CVF
Nếu làm:
\[
D_f,D_b
\]
trên toàn ảnh, reviewer rất dễ nói đây là TWISE + modern backbone.
Thay vào đó chỉ cần two-sided transport semantics.
Ví dụ edge normal:
\[
\mathbf n_e.
\]
Flow được decomposed:
\[
\mathbf v
=
v_{\parallel}\mathbf t_e
+
v_{\perp}\mathbf n_e.
\]
Và penalize normal crossing:
\[
L_{\rm cross}
=
\sum_p
e_p
\max(0,|v_{\perp,p}|-\tau).
\]
Trong khi tangent transport vẫn tự do:
\[
v_\parallel.
\]
Interpretation rất đẹp:
Geometry may propagate along an object boundary but should not freely propagate through an occlusion boundary.

Cái này khác rõ Twin Surface.
11. Loss nên thay đổi cùng module
Không nên giữ edge supervision yếu như hiện tại.
Giữ objective hiện tại làm backbone objective, nhưng thêm PMSF terms:
\[
L_{\rm surf}
=
\frac{
\sum_p e_p
(\hat D_p-D_p)^2
}{
\sum_p e_p+\epsilon
}.
\]
Depth gradient:
\[
L_{\nabla}
=
\frac{
\sum_p e_p
|\nabla\hat D_p-\nabla D_p|
}{
\sum_p e_p+\epsilon
}.
\]
Barrier:
\[
L_{\rm cross}
=
\sum_{(p,q)}
e_{pq}a_{pq}.
\]
Nó trực tiếp punish probability mass truyền qua edge.
Flow regularization chỉ trong same-surface region:
\[
L_{\rm smooth}
=
\sum_{p,q}
(1-e_{pq})
|\mathbf v_p-\mathbf v_q|.
\]
Tổng thêm:
\[
L_{\rm PMSF}
=
\lambda_sL_{\rm surf}
+
\lambda_gL_\nabla
+
\lambda_cL_{\rm cross}
+
\lambda_vL_{\rm smooth}.
\]
Tôi sẽ không tăng mạnh global edge MSE đơn thuần, vì nó dễ làm model sharpen RGB texture thành false depth edges. HFD-Teacher cũng cho thấy high-frequency recovery phải đồng thời giữ structural/topological consistency, không chỉ sharpen. Open Access CVF
12. Cách dùng sparse sensor cũng rất hợp với AnchorFlow
Bạn đã có learned sensor trust.
Giữ nguyên nó.
PMSF nhận thêm innovation:
\[
r_p
=
M_p(S_p-D_p).
\]
Flow head condition trên:
\[
[F_p,D_p,M_p,r_p,C_{\rm sensor},K].
\]
Ý nghĩa:
- không có sensor → surface context quyết định flow;
- sensor đồng ý prediction → confidence cao;
- sensor disagree → correction mạnh;
- nhưng nếu sensor nằm ngay occlusion edge → barrier/trust quyết định không propagate nó sang opposite surface.
Đây kết nối rất đẹp với philosophy AnchorFlow.
13. Nên đặt PMSF ở đâu?
Tôi chọn đúng một chỗ:
\[
\boxed{D4\rightarrow PMSF\rightarrow D2}
\]
Không D1.
Không full resolution.
Không F16.
Lý do chính là diagnostic của bạn:
\[
D4:
1.9066\rightarrow1.8519
\]
khi thêm ContextWide.
train_log (2)
Tức 1/4 đang là vị trí mà representation vẫn còn nhiều room nhưng đã có spatial resolution đủ để nhìn thấy boundary.
Sau đó existing phase upsampling chịu trách nhiệm đưa detail lên full resolution.
14. Vậy V4 ContextWide nên bỏ hay giữ?
Về research experiment, giữ.
Nó rất có giá trị như một control:
\[
\text{V3}
\]
vs.
\[
\text{V3 + generic capacity}
\]
vs.
\[
\text{V3 + targeted PMSF}.
\]
Đây thậm chí còn tạo ablation đẹp hơn.
Nếu PMSF:
\[
1.045\rightarrow1.01
\]
với +0.2 G MAC,
trong khi ContextWide:
\[
1.045\rightarrow1.030
\]
với +3.37 G MAC,
thì story sẽ rất mạnh:
improvement does not arise merely from increased capacity; explicitly modeling piecewise geometric transport provides a more efficient inductive bias.

Đó là paper story tốt hơn rất nhiều.
Tôi không khuyên stack PMSF lên full V4 ngay từ đầu.
Prototype đầu tiên nên:
\[
\boxed{\text{V3 + PMSF}}
\]
để isolate contribution.
Sau đó mới test:
\[
\text{V4 + PMSF}
\]
như upper-bound accuracy experiment.
15. Ablation tối thiểu nên chạy
Model	Capacity	Flow	Barrier	Mục đích
V3	–	–	–	baseline
V4 ContextWide	✓	–	–	generic-capacity control
PMSF-A	–	✓	–	flow alone
PMSF-B	–	✓	✓	full proposal


PMSF-A rất quan trọng.
Nếu:
\[
PMSF-A
\]
làm edge tệ đi nhưng:
\[
PMSF-B
\]
giảm edge RMSE rõ ràng, thì bạn có bằng chứng thực nghiệm cho chính hypothesis:
continuous transport alone is insufficient at depth discontinuities; piecewise visibility constraints are necessary.

Đây là result rất publishable nếu mạnh.
16. Metrics cần bổ sung cho paper
Overall RMSE vẫn primary.
Nhưng contribution này bắt buộc report:
\[
RMSE_{\rm edge}
\]
và thêm các band:
\[
RMSE_{\text{edge},1px},
\quad
RMSE_{\text{edge},3px},
\quad
RMSE_{\text{edge},5px}.
\]
Tôi còn muốn một metric:
\[
BER(k)
=
RMSE
\left(
\{p:d(p,\mathcal E)\le k\}
\right).
\]
Tức boundary error theo khoảng cách tới GT depth edge.
Sau đó plot:
\[
k=1,2,3,5,10.
\]
Nếu PMSF đúng:
error
│\
│ \
│  \ PMSF
│   \________
│
│    baseline
└──────────── distance from edge

Gain phải tập trung gần edge rồi hội tụ về baseline ở vùng phẳng.
Đây gần như signature chứng minh module làm đúng thứ nó claim.
Mind The Edge cũng nhấn mạnh rằng depth discontinuity cần được đánh giá riêng thay vì chỉ nhìn per-pixel depth metric. Open Access CVF
17. Claim novelty nên viết thế nào
Không claim:
“first flow-based depth completion.”

Nguy hiểm vì propagation/deformable propagation đã có.
Không claim:
“first boundary-aware depth completion.”

Sai vì Twin Surface.
Không claim:
“first 3D depth completion.”

Sai vì TPVD/DeCoTR.
Claim hợp lý hơn là:
We formulate boundary refinement in sparse-to-dense metric depth completion as piecewise surface transport. Unlike image-space propagation that freely exchanges depth information across learned neighborhoods, the proposed flow operates on metric depth geometry and explicitly suppresses transport across predicted visibility discontinuities. A fixed-step realization preserves the geometric interpretation of continuous deformation while remaining suitable for edge deployment.

Theo literature tôi vừa kiểm tra, tôi chưa thấy formulation trùng đúng tổ hợp này. Nhưng trước khi paper dùng chữ “first”, vẫn nên làm systematic literature sweep thêm.
Chốt hướng triển khai
Tôi sẽ dừng hướng:
\[
\boxed{\text{more ContextWide capacity}}
\]
ở V4.
V4 đã chứng minh một điều hữu ích:
\[
\text{better context helps}
\]
nhưng với cost:
\[
2.97G\rightarrow6.34G
\]
cho khoảng:
\[
1.46\%
\]
global RMSE gain, nó không phải hướng phù hợp nhất để đẩy tiếp một paper efficient/edge depth completion.
Hướng tiếp theo nên là:
\[
\boxed{
\textbf{AnchorFlow v5: Piecewise Metric Surface Flow}
}
\]
với ba nguyên tắc rất rõ:
\[
\boxed{\text{metric 3D-aware flow}}
\]
\[
+
\boxed{\text{explicit no-cross-occlusion barrier}}
\]
\[
+
\boxed{\text{2 fixed steps only at }1/4\text{ resolution}}
\]
và prototype đầu tiên nên build trên V3, không phải V4, để chứng minh rằng geometry-aware inductive bias thắng brute-force capacity.
Nếu experiment này giảm được edge RMSE từ khoảng 1.489 m xuống cỡ 1.3x m đồng thời overall RMSE xuống dưới ~1.02–1.03 m với compute tăng nhỏ, lúc đó research story của AnchorFlow sẽ mạnh lên đáng kể: không còn là “một lightweight decoder được tune tốt”, mà trở thành một phương pháp geometric depth completion có contribution riêng rõ ràng.