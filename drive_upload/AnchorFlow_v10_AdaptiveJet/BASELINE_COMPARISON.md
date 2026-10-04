# Chốt baseline từ kết quả V8 / V9 / V9.1

Nguồn V9.1: dual_teacher-20261004T054209Z-1-001.zip; validation400 ảnh,
25.424.992 pixel GT hợp lệ, cùng subset và bounds (0.1,120) m.
Đây là validation nội bộ, không phải KITTI leaderboard/test RMSE.

## 1. Kết quả thực đo

| Metric ↓ trừ delta1 | V8 | V9 | V9.1 |
|---|---:|---:|---:|
| Global RMSE (m) | 0.99755 | 1.00235 | **0.99616** |
| MAE (m) | **0.26023** | 0.26151 | 0.26306 |
| iRMSE (km^-1) | **3.39459** | 3.50293 | 3.49429 |
| iMAE (km^-1) | **1.08401** | 1.09834 | 1.11053 |
| AbsRel | **0.013660** | 0.013831 | 0.013951 |
| delta1 ↑ | **0.996436** | 0.996327 | 0.996382 |
| RMSE0–20 m | **0.45775** | 0.47610 | 0.46312 |
| RMSE20–40 m | **1.35176** | 1.38277 | 1.36733 |
| RMSE40–60 m | 2.56482 | **2.50340** | 2.52936 |
| RMSE60–80 m | 3.89574 | 3.84548 | **3.84419** |
| RMSE80–120 m | 8.85541 | 9.08989 | **8.83067** |
| RGB-edge RMSE | **1.40875** | 1.41726 | 1.41021 |
| Non-edge RMSE | 0.87890 | 0.88251 | **0.87644** |
| GT-boundary3px RMSE | 1.69774 | **1.68681** | 1.68746 |
| Pixel error>5 m | 0.57980% | 0.58496% | **0.57688%** |
| Absolute SSE error>5 m | 19,516,088 | 19,762,112 | **19,398,279** |
| Parameters | 578,064 | 578,568 | 581,868 |
| Conv/Linear MAC | 3.21469G | 3.22754G | 3.31400G |
| Observed median/P95 latency(ms) | 10.221/10.265 | 10.716/10.797 | 11.062/11.098 |

GPU đều ghi RTX PRO6000 Blackwell, BF16, batch1, 352×1216, Torch2.11cu130.
Tuy nhiên V9.1 khóa cuDNN benchmark=false, historical profile không có đầy đủ backend
fingerprint. Chênh +8.23% latency so V8 là **quan sát**, không quy hoàn toàn cho model.
Params/MAC tăng0.66%/3.09% so V8 là chênh graph xác định được. MAC không tính transport,
pooling/softmax/elementwise/memory traffic. Không suy latency bằng MAC đơn thuần.

Full precision bảng số: [comparison_v8_v9_v9_1.csv](comparison_v8_v9_v9_1.csv).

## 2. V9.1 có thật sự lợi hơn?

**Có lợi kỹ thuật, chưa có lợi accuracy rõ ràng.**

- RMSE giảm0.00619 m (0.62%) so V9, chỉ giảm0.00139 m (0.14%) so V8.
- MAE/iMAE/AbsRel kém cả V8 và V9; iRMSE vẫn kém V8. Không thắng toàn diện.
- V9.1 khởi đầu1.001918 m; best0.996158 tại local epoch1. Sau8epoch early-stop,
  last1.007426 m. Train RMSE proxy giảm0.834887→0.815753 từ epoch1→7, nhưng val xấu đi.
  Đây là tín hiệu generalization/optimization mismatch, không đủ để kết luận thiếu capacity.
  Proxy là mean batch RMSE, không được so trực tiếp với global val RMSE.
- V9.1 warm-start **V9 bestepoch23**, không fresh: đã có24epoch cập nhật trong parent
  state và thêm2epoch tới best, thử tổng8epoch. Parent experiment đã chạy30epoch.
  V8/V9/V9.1 không cùng seed/init/precision history/budget; chưa có multiple-seed CI.
  Mức gain0.14% chưa chứng minh statistically significant.
- V9.1 trained ONNX đã PASS7case; max sai khác0.000429 m. V9 bản trước failed export.
  Đây là lý do giữ minmod/readout ổn định, không phải bằng chứng RMSE<0.9.

## 3. Điểm mạnh cần giữ

| Block | Evidence / quyết định |
|---|---|
| MobileNetV4 small + F32 context + sparse branch | Giữ width/encoder như hiện có; chưa có evidence cần widen |
| Reaction + analytic jet transport | Giữ field có feedback, bounds và3step baseline |
| V9 five-jet consensus | D2 query1.34957(V8)→1.28539(V9)→1.27691(V9.1); giữ |
| Continuous minmod initialization | ONNX trained parity thực tế; chỉ limiter seed, không claim TVD cả NN |
| Tiny sparse-innovation metric head | V9.1 D2fused1.21894 tốt hơn V8(1.22179)/V9(1.22739); giữ trong technical baseline |
| Learned sensor reliability fusion | V9.1 soft0.99616 vs legacy hard1.13500; không dùng hard-anchor làm output chính |
| GT-first + weak metric/relative-gradient KD | Giữ V9.1 loss cho V10; chưa tách riêng causal benefit từng teacher/head |

**Baseline kỹ thuật canonical được chốt: V9.1 graph + V9.1 GT-first loss + fixed3 h=1/3.**
Accuracy reference vẫn giữ riêng V8 vì MAE/iRMSE/near tốt nhất. Không thể ghép các scalar
metric tốt nhất của từng run thành một checkpoint “giữ hết điểm mạnh”.
B0 fresh40 trong V10 folder cho phép kiểm chứng baseline này ở cùng budget, không dùng
checkpoint fine-tune V9.1 làm initialization.

## 4. Vì sao thử dynamic integrator

| Quarter-grid stage RMSE(m) | V8 | V9 | V9.1 |
|---|---:|---:|---:|
| D0 | 1.77279 | 1.76832 | 1.76784 |
| Step1 | 1.66154 | 1.65509 | 1.65283 |
| Step2 | **1.63875** | **1.64269** | **1.63831** |
| Step3 | 1.64179 | 1.65508 | 1.64927 |

Step3 không giảm RMSE native so step2 trong cả3run. Điều này hỗ trợ hypothesis chọn
terminal state thích hợp, **không chứng minh** output Dfull dừng2 sẽ tốt hơn:
readout đã được train với state3. V10 phải train/readout được cả2/3/4 và audit final output.

Prior V9 pixel audit:97.35% pixel error>5m có sparse trong8px, nhưng chỉ0.79% tail saturate
D2 residual bound. Không có bằng chứng “chỉ tăng amplitude là đủ”; sparse ở gần chưa chắc cùng
surface, valid GT cũng có thể conflict sensor/occlusion.
Từ0.99616→0.9 cần giảm **18.37% total SSE**; →0.8 cần khoảng35.50%.
Adaptive integrator là một thí nghiệm có động cơ rõ ràng, không một bảo đảm đạt target.

Loss budget ở best V9.1: metric26.60%, RMSE31.85%, range19.81%, boundary8.00%,
tail6.00%, trajectory2.22%, metricKD1.28%, relative0.0565%, inverse0.0653%.
Log/edge rất nhỏ. Đây là đóng góp magnitude đo được, không phải phần trăm tác động lên gradient;
không suy “teacher vô dụng” chỉ vì coefficient/budget nhỏ. Cần matched ablation.

## 5. Acceptance

Ưu tiên global RMSE giảm đáng kể mà MAE/iRMSE, near/far và tail không tệ đi; báo đồng thời
GT-boundary3px, absolute tail SSE, real conditional median/P95 trên cùng backend.
Desired RMSE<0.9; desired meanNFE≤3; hard maximum4.
Nếu controller luôn4 hoặc host-sync xóa lợi runtime, gọi đúng learned4/static4, không claim
adaptive-compute improvement. Muốn khẳng định novelty/gain cần B0/A1/A2/A3 và nhiều seed,
không chỉ historical scalar comparison.
