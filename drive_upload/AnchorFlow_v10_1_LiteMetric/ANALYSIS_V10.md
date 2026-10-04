# V10 completed run — phân tích và quyết định V10.1

Nguồn: **dual_teacher-20261004T093136Z-1-001.zip**, V10 fresh40; dừng sớm sau32epoch, best epoch23. B4, native BF16, RTX PRO6000 Blackwell; internal400val với **25.424.992 GT pixels**. Anonymous1000test không có public GT.

## 1. Kết quả thực tế

| Metric | V8 | V9 | V9.1 | V10 best23 |
|---|---:|---:|---:|---:|
| RMSE m | 0.99755 | 1.00235 | 0.99616 | **0.99626** |
| MAE m | 0.26023 | 0.26151 | 0.26306 | 0.26076 |
| iRMSE km⁻¹ | 3.39459 | 3.50293 | 3.49429 | **3.40021** |
| iMAE km⁻¹ | 1.08401 | 1.09834 | 1.11053 | 1.09569 |
| AbsRel | 0.013660 | 0.013831 | 0.013951 | 0.013778 |
| δ1 | 0.996436 | 0.996327 | 0.996382 | 0.996455 |
| 0–20 m RMSE | 0.45775 | 0.47610 | 0.46312 | 0.45801 |
| 20–40 m | 1.35176 | 1.38277 | 1.36733 | **1.38771** |
| 40–60 m | 2.56482 | 2.50340 | 2.52936 | 2.52191 |
| 60–80 m | 3.89574 | 3.84548 | 3.84419 | **3.79105** |
| 80–120 m | 8.85541 | 9.08989 | 8.83067 | 9.14928 |
| RGB-edge RMSE | 1.40875 | 1.41726 | 1.41021 | 1.41036 |
| GT-boundary3px RMSE | 1.69774 | 1.68680 | 1.68746 | **1.68272** |

Các run khác budget/init: bảng là lịch sử quan sát, **không phải causal ablation**. V10 gần như ngang V8/V9.1, chưa chứng minh adaptive integrator cải thiện global metric.

Minimum-iRMSE ở **epoch29:3.33797**, nhưng RMSE1.00186m, kém best23. V10.1 lưu riêng best-RMSE, best-inverse và best-joint để không bỏ mất checkpoint tốt theo mục tiêu thứ hai.

## 2. Controller collapse, không phải dynamics vô ích

| Quan sát của V10 | Giá trị / kết luận |
|---|---|
| Exit2 / exit3 / exit4,400val | **100% /0% /0%** |
| h1 /h2 trung bình | 0.333263 /0.333322, sát maximum1/3 |
| h1 /h2 std | khoảng9.7e−6 /2.0e−6 |
| Stop2 accuracy | 56.75%, đúng bằng tỷ lệ target “stop” |
| Stop3 accuracy | 87.50%, đúng bằng tỷ lệ target “stop” |
| Early epochs | 0–4 toàn exit4; từ5 toàn exit2; không có mixed-exit policy |
| Validation/train rollout | vẫn tính4field calls |

Accuracy ở đây bằng majority-class accuracy: không chứng minh controller phân biệt được scene khó/dễ. Tuy nhiên **reaction và conductance vẫn phụ thuộc state hiện tại**, nên giữ fixed-step feedback vẫn là learned geometric dynamics.

Same-checkpoint policy audit:

| Output selection | RMSE m | iRMSE km⁻¹ | Nhận xét |
|---|---:|---:|---|
| Learned policy / forced2 | 0.996262 | 3.400205 | giống nhau |
| Forced3 | 0.997010 | 3.383494 | iRMSE/MAE nhỉnh hơn, RMSE kém nhẹ |
| Forced4 | 0.999018 | 3.411856 | không lợi global |

**Quyết định cập nhật theo yêu cầu:** default **fixed2 nhưng learned h**, bounds1/6–1/3; chỉ bỏ stop MLP, stop BCE, exit exploration và states3/4. Log cũ chỉ chứng minh learned h của run đó bão hòa, **không chứng minh learned step-size luôn thừa**. Thay expensive controller bằng33parameter shared state-conditioned head, không thay số bước. Hơi giảm lợi ích inverse của step3 là trade-off được ghi rõ; flow_steps3 là control riêng nếu cần.

## 3. Coarse tốt hơn không chuyển hết thành final accuracy

Native stage RMSE của V10; mỗi scale dùng **valid-area-mean GT riêng**, không so pixel support các scale như cùng target:

| Stage | RMSE m |
|---|---:|
| D16 /D8 /D0 | 2.724 /2.260 /1.742 |
| D4 step1 /step2 | 1.634 /**1.631** |
| D4 step3 /step4 | 1.649 /1.670 |
| D2 base /query /fused | 1.292 /1.261 /**1.216** |
| D1 base /detail /final | 1.040 /1.010 /**0.996** |
| Hard anchor diagnostic | **1.135**, kém soft fusion |

D4/D2 tốt hơn lịch sử, final gần như không đổi. Đây là **dấu hiệu readout bottleneck**, không chứng minh nguyên nhân duy nhất. Giữ five-jet consensus, uncertainty gate và sparse-innovation head: bỏ chúng chưa có thí nghiệm chứng minh tốt hơn.

## 4. Pixel audit: metric và inverse bị chi phối bởi hai nhóm khác nhau

Re-evaluate toàn400 bằng **CPU FP32**, nguyên trained weights; global0.996558 /iRMSE3.398354, hơi khác GPU BF16 reference. Không dùng số CPU thay official run report.

| GT range | Share metric SSE | Share inverse SSE |
|---|---:|---:|
| 0–5 m | 0.70% | **37.29%** |
| 5–10 m | 3.64% | **34.23%** |
| 10–20 m | 12.10% | 18.98% |
| 20–40 m | **32.55%** | 8.05% |
| 40–60 m | **28.23%** | 1.17% |
| 60–80 m | **20.15%** | 0.26% |
| 80–120 m | 2.63% | 0.014% |

Vùng0–5m chỉ1.50% valid pixels nhưng gây37% inverse SSE. 20–80m gây khoảng81% metric SSE. Vì vậy chỉ tăng far weighting không đảm bảo iRMSE tốt lên.

Theo GPU report: **145.538 pixels**, chỉ0.5724%, có |error|>5m nhưng gây **77.06% SSE**; |error|>10m gây57.88%. Vùng không có sensor trực tiếp gây93.17% metric SSE và92.80% inverse SSE theo CPU audit.

Ví dụ outlier thực: GT≈2.20m, prediction≈32.05m, **không có sparse tại pixel đó**. Đây là near foreground/background mixing, không phải NaN hoặc sát-zero artifact. Saturation >90% bound chỉ≈0.0188% D1 và0.0111% D2 trên toàn frame: chưa có cơ sở nới bound toàn cục.

**Quyết định:** đưa richer P4 context vào final phase sampling **và** detail/trust head; không chỉ thêm một residual rất hạn chế sau khi sampling đã chọn sai surface.

## 5. Runtime và phần tốn tài nguyên

| V10, RTX PRO6000, B1 BF16 | Median | P95 | Compute |
|---|---:|---:|---|
| Synthetic masked graph | 14.794 ms | 14.864 ms | 4NFE |
| Real100 masked graph | 15.348 ms | 15.407 ms | 4NFE |
| Real100 adaptive B1 | 12.380 ms | 12.437 ms | 2NFE, có host sync |

Params582.350; registered Conv/Linear MAC3.369G **không đếm analytic transport/memory traffic**. Dynamics component≈7.067ms trong masked synthetic pass; phase2≈1.374ms. Controller MLP nhỏ nhưng policy còn nhiều reduction/branch/kernel launches, không chỉ482parameters.

V10.1 bỏ old controller statistics và depth telemetry khỏi deploy/profile; training/validation vẫn log trajectory và h1/h2/terminal time. Added context projection **1.044parameters**, new step head33; bỏ482controller; tổng **582.945**, +0.102%. Mỗi bước chỉ thêm1×1head + một spatial mean. Không có learned full-res feature hoặc transformer mới.

Structural Conv/Linear MAC,352×1216: **3.286924608G** so masked4 V10 3.368789312G. Added quarter-context projection≈0.0260G, step head≈0.00171G; phần tiết kiệm runtime tiềm năng còn đến từ analytic operations/kernel launches không nằm trong MAC. Không so MAC mới fixed2 với MAC cũ4NFE rồi gọi đó là gain so adaptive2 nếu chưa đếm cùng graph.

Local CPU FP32 re-evaluation400val của **V10-trained weights sau pruning**, context zero-init, new learned step-head bias4 (warm migration), được lưu trong local_verification.json. Đây chỉ là initialization/migration audit, **không phải kết quả đã train V10.1**. Các số fixed-h của bản trước không được dùng làm proof cho bản learned-h mới.

**Không gán latency GPU cho V10.1 trước khi đo.** 2NFE thay4 giảm số lần field evaluation, không đồng nghĩa latency toàn model giảm50%. Real100 protocol được giữ để so lại cùng hardware/backend.

## 6. Loss nào bỏ / giữ / nâng

| Thành phần | Hành động | Cơ sở / giới hạn suy luận |
|---|---|---|
| Controller BCE, trajectory3/4 | Bỏ | Policy collapse và extra rollout không lợi theo same-checkpoint audit |
| Inverse Huber100×, nhiều scale | Thay direct full-output inverse RMSE | Mục tiêu iRMSE; inverse của mean GT coarse không phải mean inverse GT |
| GT log/edge, teacher-edge | Default0, không tính; control giữ hệ số cũ | Weighted magnitudes rất nhỏ; **không phải bằng chứng gradient vô ích** |
| Disabled robust-MSE/legacy debug stats | Bỏ code/computation | Không đóng góp vào objective đang chạy |
| Global RMSE weight | 0.4→**0.6** | Target metric SSE, giữ signal gradient trên tail |
| Direct inverse RMSE | **0.04**, ramp4epochs | ỞiRMSE≈3.4, weighted term≈0.136, không còn quá nhỏ |
| Range0.1, tail0.25, GT-boundary0.05 | Giữ | Far/rare-error/boundary là bottleneck thực |
| Trust/sparse holdout/barrier | Giữ | Hard anchor tệ hơn rõ; tránh làm sensor conflict nặng hơn |
| Metric/relative KD | Default giữ2teacher | Chưa có matched ablation chứng minh relative nên bỏ; nó không thêm inference cost |

Architecture và objective thay cùng nhau trong bundle. Muốn kết luận từng nguyên nhân, dùng **control40**, rồi bật context/loss từng phần vào run/tag khác. Không cần chạy nhiều experiment để dùng default, nhưng không được gọi default gain là chứng minh causal riêng một module.

## 7. Mục tiêu và rủi ro

Từ0.99626→0.9 cần giảm **18.39% total SSE**; 3.40021→3.2 cần giảm **11.43% inverse SSE**. Không phải thay đổi nhỏ chắc chắn đạt được với1.600train.

Tiêu chí giữ: đạt cả2 trên **cùng checkpoint**, không làm40–80m hoặc boundary xấu rõ, runtime real100 không vượt V10 adaptive12.38ms khi đo matched. Nếu chỉ RMSE tốt nhưng inverse kém, báo Pareto trade-off; không clamp hay đổi GT mask.

Nếu plateau≈1m tiếp: inspect 0–5m foreground errors, teacher conflict/quality và sample-level tail trước khi tăng width/backbone. Tiny context bypass vẫn local: không tự giải quyết toàn bộ occlusion hoặc thiếu sparse.
