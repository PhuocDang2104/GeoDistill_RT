ọc kỹ V8, tôi nghĩ baseline này đã đủ tốt để ngừng thêm module kiểu “cho mạnh hơn”. Vấn đề hiện tại không phải thiếu capacity: chỉ 578k params, nhưng pipeline đã có RGB/sparse fusion, inverse-depth jet 6D, feedback dynamics, analytic transport, phase query, detail head và learned sensor fusion.    Markdown đã dán (1) Điểm nghẽn bây giờ khá rõ từ số liệu, nên V9 nên là một surgical upgrade chứ không phải architecture mới hoàn toàn.
1. Tôi đọc kết quả V8 như thế nào?
Điểm quan trọng nhất là RMSE 0.99755 m nhìn có vẻ còn xa 0.9, nhưng thực ra bạn chỉ cần giảm khoảng 18.6% total SSE để xuống 0.90 m.
Trong khi:
- >5 m: chỉ 0.58% pixel, nhưng chiếm 77.14% SSE.
- >10 m: chỉ 0.18% pixel, nhưng chiếm 57.67% SSE.
- GT boundary 3 px chiếm 38.12% SSE.
- 40–60 m + 60–80 m chiếm khoảng 50.5% SSE, dù số pixel rất nhỏ.
- Edge RMSE = 1.409 m, non-edge chỉ 0.879 m.    Markdown đã dán (1)    Markdown đã dán (1)
Một phép tính khá đáng chú ý: nếu toàn bộ phần ngoài nhóm >5m giữ nguyên, bạn chỉ cần giảm khoảng 24% SSE bên trong nhóm >5m là đủ để đạt RMSE ≈ 0.9.
Nói cách khác:
V8 không có “general accuracy problem”; V8 có “rare catastrophic geometry failure problem”.

Đây nên là research question của V9.
2. Weak point lớn nhất của architecture hiện tại: Jet query
Đây là tín hiệu mà tôi đánh giá quan trọng thứ hai:
stage	RMSE
D0	1.773
D4 step 1	1.662
D4 step 2	1.639
D4 step 3	1.642
D2 base	1.271
D2 pure jet query	1.350
D2 fused	1.222
final	0.998


Dynamics rõ ràng đang học geometry có ích: D0 → D4 giảm ~131 mm. Nhưng geometric readout của bạn lại không tận dụng state đó đủ tốt: pure jet query tệ hơn learned D2 base tới ~78 mm.    Markdown đã dán (1)
Tôi nghi nguyên nhân chính nằm trong giả định này:
\[
\xi(s,t)=v+g_xs+g_yt+
\frac12h_{xx}s^2+h_{xy}st+\frac12h_{yy}t^2
\]
Bạn lấy một local quadratic surface ở pixel trung tâm, rồi query subpixel.    Markdown đã dán (1)
Điều này rất đẹp ở vùng smooth surface, nhưng ở:
- object boundary,
- occlusion,
- thin objects,
- foreground/background mixed cells,
một Taylor polynomial duy nhất không biết phase đó thuộc surface nào.
Đây chính xác là nơi V8 đang mất rất nhiều SSE.
Các hướng SOTA cũng cho thấy đây là vấn đề cốt lõi. NLSPN dùng non-local relevant neighbors để tránh mixed-depth boundary; BP-Net cho thấy propagation phù hợp ngay từ sparse measurements rất quan trọng; DFU 2024 dùng confidence-aware adaptive guidance; GBPN 2026 thậm chí học scene-specific non-local graph để tránh propagation sai. arXiv
3. Upgrade tôi ưu tiên nhất: Occlusion-Aware Multi-Jet Query
Đây là hướng tôi đánh giá mạnh nhất cả về metric lẫn novelty.
Hiện tại V8 query phase từ jet của center:
\[
j_p \rightarrow D(x).
\]
Thay vì vậy, tại một target phase \(x\), lấy một tập rất nhỏ các candidate từ center + 4 hoặc 8 neighbor:
\[
\hat D_{q\rightarrow x}
=
\mathcal Q(T_{q\rightarrow x}(j_q)),
\qquad q\in\mathcal N(p).
\]
Tức là mỗi neighboring jet đều được analytic transport tới đúng tọa độ target, rồi mới query depth.
Sau đó dùng chính những thứ V8 đã có:
- learned barrier,
- current depth compatibility,
- RGB phase guidance,
- sparse reliability,
để tính:
\[
a_{q,x}
=
\operatorname{softmax}
\left(
-\alpha\,\Delta_{\rm geom}
-\beta\,B_{pq}
-\gamma\,\Delta_{\rm RGB}
\right)
\]
và:
\[
Q(x)
=
\sum_q a_{q,x}\hat D_{q\rightarrow x}.
\]
Không transformer. Không Mamba. Không global attention. Không grid_sample.
Chỉ là multi-source geometric querying.
Điểm hay hơn nữa: uncertainty gần như miễn phí
Các candidate jet tự tạo ra disagreement:
\[
U(x)
=
\sum_q
a_{q,x}
\left(
\hat D_{q\rightarrow x}-Q(x)
\right)^2.
\]
Nếu các neighboring jets đều mô tả cùng một surface, \(U\) thấp.
Nếu pixel nằm ở occlusion/boundary và các jets đại diện cho foreground/background khác nhau, \(U\) cao.
Bạn vừa có một geometrically-derived uncertainty, không cần thêm một uncertainty CNN lớn.
Sau đó dùng \(U\) để điều khiển existing gate:
\[
g_Q
=
g_Q(Z,G,S,U).
\]
Ở geometry chắc chắn → trust analytic jet.
Ở boundary/occlusion → giảm trust query, để learned detail/RGB/sparse correction xử lý.
Đây rất phù hợp với observation của DFU rằng confidence-aware guidance cải thiện depth upsampling, nhưng representation của bạn khác hẳn: confidence ở đây được sinh trực tiếp từ agreement của transported Taylor jets, không chỉ predict một confidence map bằng CNN. Open Access CVF
Tôi sẽ pitch novelty như thế này
Occlusion-aware multi-jet geometric readout that reconstructs subpixel depth from multiple transported local differential states and derives uncertainty directly from cross-jet geometric disagreement.

Tôi đánh giá novelty này cao hơn đáng kể so với chỉ thêm CSPN, attention hay uncertainty head.
4. Upgrade thứ hai: sửa objective để đánh đúng RMSE tail
Loss hiện tại đang có:
- global RMSE,
- range-balanced RMSE,
- boundary RMSE,
- squared excess tail >2m.
Nhưng tail term tại epoch 21 chỉ đóng góp khoảng 0.0415, trong khi global RMSE và range RMSE mỗi term khoảng 0.21.    Markdown đã dán (1)
Vấn đề nữa là fixed threshold >2m vẫn gom gần 2% pixel; trong khi phần thật sự giết RMSE lại là top ~0.5%.
Tôi sẽ không thêm nhiều loss mới. Chỉ thay tail loss hiện tại bằng một dạng top-q risk loss.
Ví dụ với \(q=1\%\):
\[
\mathcal H_q
=
\operatorname{TopQ}_{q}
\{|D-G|\},
\]
\[
\mathcal L_{\rm risk}
=
\frac1{|\mathcal H_q|}
\sum_{p\in \mathcal H_q}
\min
\left[
(D_p-G_p)^2,\,
c^2
\right].
\]
Tôi sẽ thử:
\[
q \in \{0.5\%,1\%,2\%\}
\]
nhưng 1% là run đầu tiên.
Có warm-up 3–5 epochs rồi ramp lên.
Điều này khác global MSE ở chỗ gradient budget không bị 99% easy pixels làm loãng.
Tôi không coi đây là headline novelty — hard mining/CVaR/risk-sensitive optimization không mới — nhưng nó là performance intervention hợp lý nhất với thống kê hiện tại.
Một caveat lớn: trước khi dùng nó, hãy inspect 50–100 worst pixels. Nếu một phần tail là KITTI GT aggregation artefact, projection misalignment hoặc moving-object ghosting thì hard-mining MSE có thể ép model học noise.
5. iRMSE: loss hiện tại gần như chưa optimize đúng thứ bạn muốn
V8:
\[
\mathrm{iRMSE}=3.395
\]
và đặc biệt:
- 0–20 m: 3.677
- edge: 4.197
- non-edge: 3.186.    Markdown đã dán (1)
Trong khi log-depth Huber và log-gradient loss ở epoch 21 chỉ đóng góp:
\[
6.8\times10^{-5},
\qquad
1.3\times10^{-5}.
\]
Nói cách khác chúng gần như không có vai trò trong total objective.    Markdown đã dán (1)
Mà state của AnchorFlow lại chính là:
\[
v=\frac1D.
\]
Đây là một sự lệch objective khá rõ.
Tôi sẽ bỏ hoặc giảm vai trò hai log loss đó và thêm direct inverse-depth supervision:
\[
\mathcal L_{\rm inv}
=
\operatorname{Huber}
\left(
\frac1D-\frac1G
\right).
\]
Tốt hơn nữa là supervise ở:
\[
D_4,\quad D_2,\quad D_{\rm full}
\]
nhưng trọng số nhỏ.
Ví dụ:
\[
0.02L_{\rm inv}^{D4}
+
0.03L_{\rm inv}^{D2}
+
0.05L_{\rm inv}^{full}.
\]
Không cần tăng weight lớn ngay.
Lợi ích là nó align trực tiếp optimization với representation của JetDynamics và với iRMSE.
Tôi kỳ vọng thay đổi này ảnh hưởng iRMSE nhiều hơn RMSE, đặc biệt near-range.
Đây cũng không phải novelty chính; nó là một objective alignment ablation rất cần thiết.
6. Dynamics: chưa nên thêm step, hãy thử adaptive horizon
Step 2:
\[
1.6387\text{ m}
\]
Step 3:
\[
1.6418\text{ m}.
\]
Step 3 cải thiện MAE nhưng làm RMSE hơi xấu đi.    Markdown đã dán (1)
Tôi không nghĩ nên bỏ step 3 ngay, nhưng đây là dấu hiệu có một số pixel bị over-correction.
JetDynamics hiện cũng là module chậm nhất:
\[
3.42\text{ ms}/10.22\text{ ms}.
\]
   Markdown đã dán (1)
Thay vì thêm step 4/5, thử một gate cực nhỏ:
\[
j^\star
=
\alpha j_2+(1-\alpha)j_3.
\]
Trong đó \(\alpha\) phụ thuộc:
\[
\alpha=f(U,\;r_{\rm sparse},\;\text{barrier},\;Z).
\]
Nếu multi-jet disagreement cao hoặc step 3 đang gây bất ổn → ưu tiên step 2.
Ở smooth surface → dùng step 3.
Câu chuyện lúc đó rất đẹp:
The model does not assume that a fixed number of geometric feedback updates is optimal at every pixel.

Nhưng tôi xếp cái này sau multi-jet query. Không làm cùng lúc ở run đầu tiên.
7. Relative-depth / foundation teacher: có nên thêm không?
Có, nhưng chỉ dùng để dạy structure, không dùng nó làm metric teacher thứ hai.
DMD³C tại CVPR 2025 đã cho thấy distilling monocular foundation models có thể cải thiện fine-grained depth completion; HFD-Teacher tại ICCV 2025 đi còn rõ hơn: distill high-frequency depth information để khôi phục fine structures/boundaries. Open Access CVF
Vì vậy nếu bạn chỉ thêm một relative-depth teacher toàn ảnh thì:
- khả năng cải thiện tốt,
- nhưng novelty không cao nữa.
Tôi sẽ dùng relative model chỉ để tạo scale-free structural target ở difficult regions.
Ví dụ pairwise ordinal:
\[
\mathcal L_{\rm ord}
=
\operatorname{BCE}
\left[
\operatorname{sign}
(\tilde D_i-\tilde D_j),
\operatorname{sign}
(D_i-D_j)
\right]
\]
hoặc normalized local gradients:
\[
\nabla
\frac{D-\mu_D}{\sigma_D}.
\]
Và chỉ áp dụng tại:
- RGB/depth boundaries,
- high multi-jet uncertainty,
- thin structures.
Như vậy teacher dạy model:
“surface nào nằm trước/sau và boundary ở đâu”

chứ không ép metric scale của relative model vào student.
Đây là supporting technique tốt cho headline multi-jet idea.
8. Tôi sẽ xây V9 theo thứ tự nào?
Priority	Thay đổi	Mục tiêu	Novelty	Cost
P0	Tail × boundary × range audit	Xác định outlier thật	—	rất thấp
P1	Top-1% risk loss	Giảm catastrophic SSE	thấp	~0 inference
P1	Direct inverse-depth loss	Giảm iRMSE	thấp	0 inference
P2	Occlusion-aware multi-jet query	Boundary + tail	cao	thấp
P2	Jet disagreement uncertainty	Query/trust gating	cao khi gắn với multi-jet	rất thấp
P3	Adaptive j2/j3 horizon	Chặn over-correction	trung bình–cao	rất thấp
P4	Structural relative-depth distillation	Fine boundary	trung bình	train-only


Nếu chỉ được chọn một architecture novelty, tôi chọn:
Occlusion-Aware Multi-Jet Consensus Readout
Nếu được chọn thêm một optimization idea:
Tail-risk + direct inverse-depth supervision
Tôi không thêm Transformer, Mamba, diffusion, full-resolution feature CNN hay nhiều JetDynamics steps ở V9.
9. Ablation tôi nghĩ đủ mạnh cho paper nhưng không over-engineering
Tôi sẽ chỉ chạy sequence sau:
ID	Model	Câu hỏi cần trả lời
B0	V8	baseline
A1	V8 + direct inverse loss	iRMSE có thực sự giảm?
A2	V8 + top-1% risk loss	catastrophic tail có giảm?
A3	V8 + A1+A2	loss-only ceiling
A4	A3 + multi-jet query	geometry query có vượt D2_base?
A5	A4 + jet disagreement gate	uncertainty có giảm boundary/tail?
A6	A5 + adaptive j2/j3	step3 overshoot có được giải quyết?


Không cần ablate 20 thứ.
Success criterion cho A4 rất rõ: hiện tại
\[
D2_{\rm query}=1.350
\]
trong khi
\[
D2_{\rm base}=1.271.
\]
Nếu multi-jet geometric query vẫn không xuống được <1.27, tôi sẽ không giữ nó chỉ vì novelty.
Nếu xuống khoảng 1.20–1.24, lúc đó bạn đã có bằng chứng rằng differential jet representation thực sự useful ở readout, chứ không chỉ dynamics.
10. Metric cần log thêm ở mọi ablation
Đừng chỉ xem final RMSE.
Quan trọng nhất là log chung một table với:
\[
RMSE,\ iRMSE,
\ RMSE_{0-20},
\ RMSE_{20-40},
\ RMSE_{40-60},
\ RMSE_{60-80}
\]
cộng thêm:
\[
RMSE_{\rm boundary1/3/5px},
\]
\[
P(|e|>2),\;
P(|e|>5),\;
P(|e|>10),
\]
và quan trọng hơn:
\[
\frac{SSE_{|e|>5}}{SSE_{\rm total}}.
\]
Nếu final RMSE giảm nhưng >5m SSE share không giảm, bạn chưa thực sự chữa căn bệnh V8.
11. Target tôi đặt cho V9
Tôi không đặt mục tiêu đầu tiên là 0.7. Với 2k training samples, như vậy dễ dẫn tới overengineering.
Tôi sẽ đặt milestone:
Metric	V8	V9 target đầu tiên
RMSE	0.9975	<0.90
iRMSE	3.395	<3.0–3.1
D2 query	1.350	<1.27, tốt nhất ≈1.20–1.24
Edge RMSE	1.409	<1.25
>5m SSE share	77.14%	giảm mạnh, ideally <65%
Latency	10.22 ms	≈ giữ nguyên / <11 ms


Đặc biệt, 0.9 không phải một bước nhảy phi thực tế. Nếu magnitude trung bình của các pixel hiện thuộc nhóm >5m giảm chỉ khoảng 15% và phần khác không đổi, phép tính lý tưởng cho RMSE đã xuống khoảng 0.884 m. Đây chỉ là sanity calculation chứ không phải dự báo training, nhưng nó cho thấy tại sao tập trung tail hợp lý hơn tăng accuracy đồng đều toàn ảnh.
12. Novelty story mà tôi thấy mạnh nhất
Nếu V9 train ra đúng như hypothesis, bài có thể không cần kể 6 contribution khác nhau.
Core story có thể là:
Existing depth-completion refinement treats local propagation and high-resolution readout largely as learned feature operations. We instead maintain a compact differential inverse-depth state and reconstruct each high-resolution point from multiple analytically transported local surface hypotheses. Their cross-hypothesis agreement provides an intrinsic geometric uncertainty signal, allowing the model to suppress cross-boundary transport and selectively trust learned correction only where the local surface model becomes ambiguous.

Điểm này khác khá rõ với:
- NLSPN: learned non-local propagation. arXiv
- BP-Net: bilateral sparse propagation. Open Access CVF
- GBPN: learned MRF + Gaussian belief propagation/non-local graph. arXiv
- DFU: confidence-aware learned feature upsampling. Open Access CVF
- Midas Touch 2026: discontinuity-aware geodesic refinement of metric depth. Open Access CVF
AnchorFlow khác ở chỗ propagated object là local differential surface state \((v,g,H)\), và uncertainty có thể xuất hiện từ disagreement giữa các transported surface hypotheses. Đây là phần tôi thấy có chất nghiên cứu nhất.
Cuối cùng, cần nhớ benchmark context: KITTI chính thức có khoảng 93k training images + 1k validation, trong khi run hiện tại chỉ dùng 2,000 IDs; vì vậy các con số leaderboard ~0.68 m hiện nay không so trực tiếp được với 0.998 m của V8. CV Libs Với paper, tôi sẽ chốt architecture trên 2k bằng các ablation trên, rồi mới full-train một lần sau khi design đã ổn. Đây vừa tiết kiệm compute vừa tránh “architecture search bằng full KITTI”.