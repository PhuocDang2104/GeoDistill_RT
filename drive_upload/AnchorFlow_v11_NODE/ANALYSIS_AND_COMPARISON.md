# Chốt V11 từ evidence V10 và góp ý NODE

## 1. Evidence đã có, không thay kết quả lịch sử

Nguồn: results/dual_teacher-20261004T093136Z-1-001.zip; internal400val,
25,424,992GTpixels. V10 fresh40 early-stop sau32epochs, bestRMSE epoch23.

| Metric | V8 | V9 | V9.1 | V10 |
|---|---:|---:|---:|---:|
| RMSE m | 0.99755 | 1.00235 | 0.99616 | 0.99626 |
| iRMSE km⁻¹ | 3.39459 | 3.50293 | 3.49429 | 3.40021 |
| MAE m | 0.26023 | 0.26151 | 0.26306 | 0.26076 |
| GT-boundary3px RMSE m | 1.69774 | 1.68680 | 1.68746 | 1.68272 |

V10 minimum-iRMSE epoch29 =3.33797 nhưng RMSE1.00186; không ghép với bestRMSE epoch23.
Lịch sử khác initialization/budget, không chứng minh superiority nhân quả.
V10.1 LiteMetric chưa có trained run trong evidence hiện tại.

## 2. Phản biện góp ý

**Đúng:** NN nên học vector field; numerical solver quyết định h theo error estimate nếu muốn
gần NODE adaptive chuẩn. Fixed T cần giữ chung để thay h không đồng thời đổi thời gian kết thúc.
GT benefit là task-policy target, không phải numerical local truncation error.

**Cần sửa:** “3–4NFE + solver adaptive chuẩn + h≤1/3 + luôn tớiT1” không phải một budget
dễ đáp ứng cho field bất kỳ. Euler lagged-field-change chỉ là proxy; không đủ cho true accept/reject.
Nếu budget buộc tăng h dù error quá lớn thì phải báo tolerance violation, không gọi solver chuẩn.
Một RK4 macrostep có4NFE nhưng không tương đương adaptive RK với tolerance control.

**V11 quyết định:** bỏ neural h/stop và chấp nhận variable NFE của official RK3(2).
Bỏ projection giữa RK stages bằng smooth bounded chart; giữ default T1.
Không cố “9/10 NODE” bằng rating chủ quan. Đặc tả chỉ rõ IVP, solver, precision và gradient.

## 3. Insight không được bỏ quên

V10 all400val exit2, h sát1/3; stop accuracy bằng majority-target rate.
Same-checkpoint forced3/4 không cải thiện globalRMSE rõ.
Điều đó bác bỏ claim policy có lợi trong run này, **không** chứng minh dynamics vô ích.

D4 step1/2/3/4 RMSE =1.634/1.631/1.649/1.670; final≈.996.
Coarse gain không chuyển hết thành final gain: readout có thể là bottleneck.
Giữ uncertainty-gated five-jet readout, metric innovation, context-guided final phase lift và soft sensor fusion.
Không tăng backbone/width/hypothesis/teacher để giảm confounding.

0.5724% pixels có error>5m gây77.06% metricSSE. CPU audit:
20–80m gây≈81% metricSSE; 0–10m gây≈71.5% inverseSSE.
Outlier nearGT≈2.2m/pred≈32m là foreground/background mixing, không phải sát-zero artifact.
Do đó giữ direct iRMSE, range/tail/boundary objectives từ LiteMetric; không clamp để “fix” metric.

## 4. Cái gì thay / không thay?

| Thành phần | Quyết định |
|---|---|
| Learned h / GT-stop / controller BCE | Bỏ hoàn toàn |
| Variable terminal horizon | Thay bằng fixed T=1 |
| Projected physical jet | Smooth chart, no projection trong ODE |
| Shared reaction–transport neural field | Giữ geometric conditioning; định nghĩa normalized RHS trong chart |
| Numerical integrator | Official embedded RK3(2), accepted/rejected trial thật |
| Backbone / sparse fusion / readout | Giữ latest LiteMetric |
| Two teachers / train-only caches / split | Giữ đúng data hiện có |
| Loss | Giữ LiteMetric; trajectory được sample tại fixed t=.5/1 |
| Edge inference | Separate fixed midpoint8, audit accuracy cùng checkpoint |

State chart và solver thay cùng nhau: nếu V11 tốt hơn, chưa thể quy gain chỉ cho solver.
Một research ablation sạch sau run là cùng V11 field/chart/weights, đổi solver/tolerance.
Notebook đã có same-checkpoint solver audit, nhưng nó chưa thay thế matched-training/multiple-seed ablation.

## 5. Trade-off được chấp nhận

V10 real100 adaptive2 đã đo≈12.380ms trên RTX PRO6000 Blackwell BF16;
masked4≈15.348ms. Không dùng số này làm latencyV11.
V11 thường≥13NFE; FP32 RHS và solver host decisions có thể chậm hơn đáng kể.
Edge candidate8NFE không tự đảm bảo faster hoặc sameaccuracy; cần đo cả hai.

Target RMSE<.9 và iRMSE<3.2 chưa đạt trong evidence V10.
Từ .99626→.9 cần≈18.39% metricSSE reduction; 3.40021→3.2 cần≈11.43% inverseSSE reduction.
NODE principle đúng không tự tạo ra phần giảm này trên1600trainingimages.
Nếu V11 plateau hoặc thêm compute mà không gain, quay về lighter baseline là kết luận hợp lệ.

**Acceptance:** cùng checkpoint đạt targets hoặc cải thiện rõ Pareto accuracy/runtime;
near/far/boundary không xấu hơn đáng kể, sameGTsupport, full solver/NFE audit,
samehardwareprecision benchmark. Không chọn tighter tolerance trên400val rồi gọi đó là independent test.
