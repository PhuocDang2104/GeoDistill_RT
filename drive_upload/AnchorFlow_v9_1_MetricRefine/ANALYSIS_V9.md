# V9 result audit → V9.1 decision

Nguồn: results/dual_teacher-20261004T040538Z-1-001.zip. V9 fresh BF16 train 30 epoch,
best 23; historical V8 best 21, run 29 epoch và chuyển FP16→BF16 sau 5 epoch.
Same 400 val, 25.424.992 valid pixels; **không matched initialization/precision/budget**.

## 1. Kết quả đã đo

| Metric | V8 | V9 | Đánh giá V9 |
|---|---:|---:|---|
| Global RMSE m | 0.997549 | 1.002352 | +0.48% |
| MAE m | 0.260227 | 0.261511 | +0.49% |
| iRMSE km⁻¹ | 3.394592 | 3.502926 | +3.19% |
| iMAE km⁻¹ | 1.084010 | 1.098343 | Kém nhẹ |
| AbsRel | 0.013660 | 0.013831 | Kém nhẹ |
| δ1 | 0.996436 | 0.996327 | Kém nhẹ |
| RMSE 0–20 m | 0.457755 | 0.476097 | +4.01% |
| RMSE 20–40 m | 1.351756 | 1.382774 | +2.29% |
| RMSE 40–60 m | 2.564824 | 2.503402 | −2.39% |
| RMSE 60–80 m | 3.895736 | 3.845479 | −1.29% |
| RMSE 80–120 m | 8.855406 | 9.089888 | Ít pixel, kém nhẹ |
| RGB-edge RMSE m | 1.408754 | 1.417262 | Kém nhẹ |
| Non-edge RMSE m | 0.878901 | 0.882511 | Kém nhẹ |
| GT-boundary 3px RMSE m | 1.697738 | 1.686805 | −0.64% |
| Tail >5 m pixel % | 0.579804 | 0.584960 | Không giảm |
| Tail >5 m absolute SSE | 19,516,088 | 19,762,112 | +1.26% |
| Parameters | 578,064 | 578,568 | +504 |
| Conv/Linear MAC | 3.21469 G | 3.22754 G | +0.40% |
| BF16 median latency ms | 10.2212 | 10.7165 | +4.85% |
| P95 latency ms | 10.2653 | 10.7975 | +5.18% |
| Peak allocated VRAM MiB | 98.3364 | 98.7495 | Gần như không đổi |

Latency cùng GPU RTX PRO 6000 Blackwell/PyTorch 2.11+cu130, batch 1, 352×1216,
không I/O/H2D. Đây không phải Jetson/edge latency.
RGB-edge và GT-boundary khác mask, không dùng thay thế nhau.

## 2. Stage localization

| V9 native stage | RMSE m |
|---|---:|
| D16 | 2.83190 |
| D8 | 2.25294 |
| D0 coarse 1/4 | 1.76832 |
| Dynamics step 1 | 1.65509 |
| Dynamics step 2 | 1.64269 |
| Dynamics step 3 / D4 | 1.65508 |
| D2 base | 1.29048 |
| D2 query | 1.28539 |
| D2 fused | 1.22739 |
| D1 base | 1.04131 |
| D1 pre-sensor | 1.01557 |
| Dfull soft sensor | 1.00235 |
| Hard sensor diagnostic | 1.14026 |

V9 query tốt hơn V8 query 1.34957; nhưng D2 base/fused kém hơn V8 1.27129/1.22179.
**Consensus đã có tín hiệu tốt; chưa chuyển thành gain final**. Không hợp lý bỏ ngay toàn bộ query
hay thêm nhiều recurrent steps khi step 3 còn kém step 2.
Các native scales có GT valid-area pooling riêng, không so trực tiếp như cùng một support.

Training best 23→last 29: global RMSE gần như đứng ở 1.002–1.003.
Train batch-mean RMSE proxy best≈.842, không phải global train RMSE để tính generalization gap chính xác.
Loss budget best: metric 29.97%, RMSE 22.53%, range 22.38%, boundary 8.99%, tail 4.39%,
metric KD 3.79%, relative 0.786%, inverse 0.073%.
Magnitude nhỏ **không chứng minh** gradient không ảnh hưởng. Ordinal≈.684 không thể kết luận random
bằng ln2: reduction có confidence weights và sampling riêng.

## 3. Audit tất cả 400 val, parent checkpoint FP32 CPU

FP32 RMSE 1.00249 gần BF16 uploaded 1.00235; không trộn hai precision thành cùng phép đo.

- Tail >5 m: 148.803 pixels, chiếm khoảng 0,585%, nhưng chiếm≈77,4% total SSE.
- **97,35% tail >5 m có sparse point trong radius 8 pixels**; chỉ 6,30% đúng tại observed sensor pixel.
- D2 metric residual >90% bound: **0,79%** tail >5 m; D1: 0,45%.
- Tail >20 m tương tự: 97,08% gần sparse, D2 saturated 2,45%.
- Mean disagreement U trên tail >5 m≈.00774 vs all valid≈.000226: có correlation với lỗi.
  Tuy nhiên chỉ 1,16% tail vượt U>.1; **không coi U là detector đủ để mask/cắt lỗi**.

Do đó **saturation không phải bottleneck chính**; không nâng correction amplitude mù.
Thay vào đó cho metric head thấy donor residual lân cận và dense decoder context, có dispersion để
phân biệt mixed surfaces. Sparse-near statistic chỉ là khả năng khai thác, không chứng minh điểm đó
luôn cùng surface hay đúng GT; head phải học selection bằng feature.

Muốn RMSE 1.00235→0.9 cần giảm **19,38% total SSE** trên cùng valid pixels.
Nếu phần non-tail giữ nguyên, tương đương giảm≈25,05% SSE nhóm tail >5 hiện tại.
Không đạt bằng thay color scale, mask GT, clip outliers khi evaluate hay hard anchor toàn bộ.

## 4. Export vấn đề cần sửa

Uploaded export_console.log: ONNX parity fail, max difference≈.12646 m.
CPU seed 42 audit không tái hiện chính xác final error này; final đạt tolerance nhưng empty-sparse
D2 query sai≈.02455 m. Sai số bắt đầu lớn trong feedback geometry.
Probe dùng minmod làm tất cả stage seed 42 gần khớp (~1e−4 m).

Đây là **bằng chứng hỗ trợ** slope tie instability và remedy, không chứng minh mọi lỗi GPU/export đều
do cùng nguyên nhân. V9.1 giữ tolerance atol=.01 m, rtol=1e−4, kiểm nhiều seed + real samples,
không nới threshold hay bật NaN masking. Sau train mới vẫn chạy parity lại.

## 5. Chốt bundle V9.1

| Quyết định | Lý do / đánh đổi |
|---|---|
| Giữ encoder width, 3 dynamics steps, 5 query | Không thêm compute lớn để xử lý bottleneck chưa rõ |
| Minmod initial jet | Slope continuous ở extrema; giữ geometric state học được |
| Quarter-grid metric innovation head | Tail thường gần sparse, nhưng decoder chưa đọc local error đủ tốt |
| Giữ residual bound | Saturation hiếm; mở bound không nhắm đúng bằng chứng |
| RMSE .4 + robust excess tail .25 | Hướng gradient vào metric errors, tránh cực đại gây domination |
| Giảm metric KD .05→.012 | Dense pseudo labels là prior, không lấn át GT refinement |
| Relative gradient .015, bỏ ordinal | Giữ local shape; giảm compute/constraint trong metric refinement |
| Fine-tune parent model-only 15 epoch | Test nhanh hướng mới; không giả vờ equal-budget gain |

Liên hệ nền tảng:
[DFU CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Improving_Depth_Completion_via_Depth_Feature_Upsampling_CVPR_2024_paper.html)
ủng hộ khai thác dense decoder features khi upsample;
[BP-Net CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Bilateral_Propagation_Network_for_Depth_Completion_CVPR_2024_paper.html)
nhấn mạnh reliable sparse propagation.
Đây là motivation, không claim copy đúng module hoặc chắc chắn tái hiện paper score với 2k images.
[Monocular foundation distillation CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liang_Distilling_Monocular_Foundation_Model_for_Fine-grained_Depth_Completion_CVPR_2025_paper.html)
cho cơ sở giữ shape prior nhưng không đồng nhất relative teacher với metric truth.

## 6. Acceptance

- Strong: global RMSE<.9 m, near 0–20 không đổi xấu đáng kể, iRMSE/GT boundary/tail tuyệt đối không regress.
- Partial: giảm initial/final tail SSE và RMSE rõ, latency tăng nhỏ; chưa claim đạt mục tiêu.
- Reject: chỉ đẹp color, far gain nhưng near/global regress, hoặc export unstable.
- Luôn báo initial/best/final, source/config/data hashes, thêm epoch budget, GPU median/P95/VRAM.

Anonymous test PNG/preview **không có public GT**. Metrics trên 400 val, không KITTI leaderboard.
