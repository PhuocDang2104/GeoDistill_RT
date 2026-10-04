# Từ V8 đến V9 — phân tích và quyết định

Nguồn: `results/metric_kd-20261003T091559Z-1-001.zip`, audit V8 và `v9_suggest.md`. Gói giữ nguyên ba JSON V8 để so sánh; không chứa V8 checkpoint. Accuracy từ `val_metrics.json`, không trộn sai khác BF16 evaluate và train-log rounded values.

## 1. V8 đã đo được gì?

Best epoch21, RMSE **0,997549 m**, MAE **0,260227 m**, iRMSE **3,394592 km⁻¹**, δ1 **0,996436**. Early stop sau 29 epoch hoàn tất; 25.424.992 valid GT pixels trên 400 validation nội bộ. Đây không phải official KITTI leaderboard score.

| Quarter-grid dynamics / half-grid readout | Native GT RMSE m |
|---|---:|
| D0 → D4 step1 → step2 → step3 | 1,772792 → 1,661539 → 1,638747 → 1,641792 |
| D2 learned base | 1,271287 |
| D2 center-only jet query | 1,349566 |
| D2 blend + metric detail | 1,221794 |
| D1 base → D1 pre-fusion → Dfull | 1,035537 → 1,010101 → 0,997549 |
| Hard-anchor diagnostic | 1,136453 |

| V8 valid-pixel error | Tỷ lệ pixels | Tỷ lệ total SSE | Absolute SSE m² |
|---|---:|---:|---:|
| >2 m | 1,9448% | 90,5871% | 22.918.991,55 |
| >5 m | 0,5798% | 77,1372% | 19.516.087,56 |
| >10 m | 0,1833% | 57,6690% | 14.590.550,44 |
| >20 m | 0,0375% | 30,0596% | 7.605.225,32 |

GT-boundary 3px RMSE **1,697738 m**, chiếm **38,1207% SSE**. Far 40–60/60–80 m RMSE **2,564824 / 3,895736 m**. Near 0–20 m **0,457755 m**. Sparse hard anchor tệ hơn learned fusion, nên không quay về anchor cứng.

## 2. Đồng ý và phản biện đề xuất

1. **Đồng ý:** cải thiện readout thay vì tăng model capacity là run tiếp hợp lý. Single quadratic jet có thể trộn surface tại occlusion; năm candidate giúp mỗi phase chọn origin phù hợp hơn.
2. **Cần thận trọng:** pure query kém base không chứng minh jet vô ích; blend đang tốt hơn base. Không có direct per-pixel causal audit để khẳng định mọi tail outlier do query. V9 kiểm tra hypothesis này, không trình bày như nguyên nhân đã được chứng minh.
3. **Không dùng hard cap** `min(error²,c²)`: gradient bằng 0 khi error vượt c, trái mục tiêu sửa catastrophic tail. GT completion có thể có label artifacts; chưa xem outlier GT không được mặc định tất cả là lỗi model hoặc tất cả là lỗi GT.
4. **Chưa thêm top-1%/CVaR:** V8 đã có tail/range/boundary loss. Top-k thêm ranking và sensitivity trên batch4; trộn quá nhiều thay đổi sẽ khó giải thích gain. Giữ squared-excess >2m, log absolute SSE và error rates thay vì chỉ SSE share.
5. **Chưa dùng adaptive horizon:** step3 RMSE hơi tệ hơn step2 nhưng MAE tốt hơn; không đủ kết luận step3 vô ích. Giữ fixed3 tránh variable latency, auxiliary-loss/halting complexity và phá đối chứng.
6. **Chọn inverse consensus và dimensionless U:** thống nhất với jet state; tránh variance metric-depth phình theo distance. Nhưng U không là calibrated uncertainty, và vẫn cần measured latency/memory vì candidate transport không miễn phí.
7. **Relative teacher chỉ structural:** não metric teacher và relative teacher khác role; tránh dense SSI toàn ảnh hoặc target metric giả. GT/original sparse luôn ưu tiên, val/test không dùng teacher.

## 3. Kiến trúc chốt cho run đầu

Giữ V8 encoder, sparse/FPN, ba feedback updates, learned sensor fusion. Thay center-only query bằng **Barrier-Aware Multi-Jet Consensus**: center+4 neighbors, exact phase queries, learned local logits, reused barrier/depth compatibility, inverse consensus, variance-controlled existing blend gate. Chỉ thêm **504 parameters**.

Train fresh30/BF16/earlystop với metric cache cũ + offline **DA3MONO-LARGE relative cache**. Giữ V8 losses, thêm inverse-GT .01 và relative local gradient/ordinal .05 ramp3. Không normals, DSINE, plane teacher, transformer decoder, extra dynamics steps, adaptive ODE hoặc global matrices.

Novelty nên mô tả: **phase-specific reconstruction từ nhiều transported differential states, với cross-jet disagreement điều khiển geometric readout trong feedback depth-completion decoder**. Không claim first-ever without systematic literature review; confidence/propagation/distillation đều có prior art. Khác biệt cần chứng minh bằng experiment, không phải tên module.

## 4. Paper nền tảng và ảnh hưởng thực tế

- [Depth Anything 3 — official paper](https://arxiv.org/abs/2511.10647), [official monocular weights](https://huggingface.co/depth-anything/DA3MONO-LARGE): chọn dedicated RGB monocular teacher, cache offline, pinned revision. Không tự đồng nhất DA3 any-view leaderboard với chất lượng KITTI monocular.
- [DFU, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Improving_Depth_Completion_via_Depth_Feature_Upsampling_CVPR_2024_paper.html): motivation confidence-aware adaptive guidance khi upsample. V9 dùng cross-jet disagreement thay vì copy decoder paper.
- [HFD-Teacher, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_HFD-Teacher_High-Frequency_Depth_Distillation_from_Depth_Foundation_Models_for_Enhanced_ICCV_2025_paper.html): motivation lấy structural/high-frequency knowledge từ foundation depth teacher. V9 dùng local normalized pairs và không thêm foundation inference vào deploy.

Đây là kiến trúc tiềm năng có cơ sở, không tuyên bố là tối ưu toàn cục hoặc chắc chắn <0,8 m trên 1.600 train.

## 5. Tiêu chí sau run

| Kiểm tra | V8 đã đo | V9 mong muốn / điều kiện |
|---|---:|---|
| Global RMSE | 0,99755 m | <0,90 m là first target; <0,8 chưa bảo đảm |
| iRMSE | 3,39459 km⁻¹ | <3,1 hoặc ít nhất không xấu đi |
| Pure D2 query | 1,34957 m | <1,27129 m learned-base reference, và fused D2 tiếp tục tốt hơn |
| >5m absolute SSE | 19,516 triệu m² | Giảm rõ cùng pixel rate; không chỉ giảm share |
| Boundary3 / 40–80m | 1,69774 / 2,56482 / 3,89574 m | Cải thiện mà near không đổi lấy quá nhiều |
| GPU median/P95 | 10,221 / 10,265 ms historical | Đo cùng GPU/torch/BF16/shape; mục tiêu median ≤11ms, chưa đo |

RMSE .99755→.90 cần giảm **18,6% total SSE**; .99755→.80 cần khoảng **35,7%**. Không lấy việc tail chiếm77% để suy ra giảm tail chắc chắn dễ hoặc tất cả tail nằm ở boundary.

Historical V8 epoch0–4 FP16, sau đó explicit BF16 recovery; V9 BF16 từ đầu và student mới. Vì vậy historical compare là **reference**, không causal ablation. Option `v8_control` train V8 gốc fresh/BF16 cùng recipe/split/cache, objective V8 chính xác, tắt inverse/relative mới. Bundle V9 vẫn đồng thời đổi readout và supervision; muốn attribution sau có tín hiệu cần thêm model-only / teacher-only ablation, không claim từ run này.
