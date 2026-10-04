# V6 → V7: bằng chứng, phản biện và quyết định

Nguồn chính: `results/metric_kd-20261002T124131Z-1-001.zip`, full 15 epoch V6 và checkpoint best epoch 14. `v7_suggest.md` là đề xuất tham khảo, không phải chứng minh thực nghiệm. Các CSV/JSON gốc cần thiết đã đưa vào gói này.

## 1. V6 đạt gì?

| Metric | Trước fine-tune, epoch −1 | Best epoch 14 / fresh evaluation |
|---|---:|---:|
| Global RMSE | 1,023401 m | 1,009297 m |
| MAE | 0,263689 m | 0,258020 m |
| iRMSE | 3,64394 km⁻¹ | 3,53287 km⁻¹ |
| RMSE 0–20 m | — | 0,463071 m |
| RMSE 20–40 m | — | 1,401602 m |
| RMSE 40–60 m | — | 2,562314 m |
| RMSE 60–80 m | — | 3,859394 m |
| RGB-edge RMSE | — | 1,430437 m |
| GT boundary band 1 px | — | ~1,875 m |

CSV epoch 14 RMSE=1,009290 m; fresh evaluation=1,009297 m, khác rất nhỏ do numerical execution. **Epoch 0 không phải initial**: đã thực hiện một epoch training. Chỉ giảm ~1,38% RMSE trong 15 epoch; không ngoại suy thành <0,8 sau thêm 15 epoch.

Run V6 dùng A100-SXM4-40GB, batch 4. Median sau warm-up: **60,17 s train / 91,68 s cả epoch gồm validation**. Khoảng 26,6 train samples/s, đã gồm loader/forward/backward/optimizer. Đây không phải inference latency; không áp thời gian này cho Colab T4 hoặc suy V7 chắc chắn nhanh hơn. Archive không có candidate V6 `profile.json`, chỉ có profile của parent V5.

## 2. Đuôi lỗi mới là mục tiêu chính

| V6 best, absolute error | Pixels / tổng valid pixels | Tỷ lệ SSE global |
|---|---:|---:|
| >2 m | 1,9044% | 90,98% |
| >5 m | 0,5779% | 78,18% |
| >10 m | 0,1864% | 59,40% |
| >20 m | 0,0414% | 32,32% |

400 validation có 25.424.992 valid GT pixels. Để đạt 0,8 m từ 1,0093 m cần giảm **37,17% MSE**, không phải chỉ giảm vài mm ở vùng gần.

Tỷ lệ SSE không phải SSE tuyệt đối. Sau 15 epoch, đóng góp MSE của |e|>5 giảm 0,81980→0,79636 m²; |e|>20 giảm 0,33595→0,32926 m² (~1,99%). Vì vậy nói “đuôi không cải thiện” là quá mạnh. Kết luận đúng: đuôi cải thiện **chậm**, vẫn thống trị objective RMSE.

0–20 m chiếm 77,43% pixel nhưng chỉ **16,30% SSE**; 20–80 m chiếm ~22,54% pixel và ~81,15% SSE. Không được suy từ số pixel rằng global RMSE chỉ bị vùng gần chi phối. Baseline đã có range-RMSE loss; V7 giữ nó, không thêm một range loss khác.

Hard-anchor oracle floor=0,60243 m nếu giả định model hoàn hảo ở mọi pixel không có sparse. Đây là lower bound **có điều kiện cho hard-anchor**, không phải noise floor của output soft fusion. V7 giữ soft trust fusion, không quay lại hard anchor để giới hạn metric.

## 3. Causal audit trên checkpoint thật

Native D4 `surface→connection` chỉ tốt thêm ~0,58 mm; D2 `base→phase` tốt thêm ~29 mm. Hai stage delta **không thay thế ablation**: phase còn phụ thuộc learned connection context.

Đã làm thêm inference-only ablation trên đủ 400 ảnh, CPU FP32, cùng checkpoint/dataset/GT mask:

| Output | Global RMSE | MAE | iRMSE |
|---|---:|---:|---:|
| Full V6 | 1,009274 m | 0,258030 m | 3,53217 km⁻¹ |
| Bỏ vector correction, giữ learned context + phase | 1,009439 m | 0,258508 m | 3,53040 km⁻¹ |

RMSE chỉ xấu **0,165 mm**. Đây là căn cứ thực nghiệm để thay vector correction. Không phải kết quả retraining của reduced V6; không kết luận vector không có tác dụng trong mọi model/dataset. Báo cáo đầy đủ: [v6_causal_audit.json](v6_causal_audit.json).

## 4. Phản biện `v7_suggest.md`

| Insight | Đánh giá / quyết định V7 |
|---|---|
| Vector 3D → scalar ray depth có null-space cục bộ 2D | Đúng tại một readout; neighbor transport có thể làm các hướng này hữu ích. Không chứng minh toàn branch vô dụng, nhưng audit cho thấy ít gain hiện tại. |
| Sáu jet coefficients “đều observable” | Không đúng nếu chỉ đo center: chỉ value observable. Bốn query ±1/4 còn không phân biệt riêng hxx/hyy. Translation qua neighbors và multi-scale loss tạo thêm đường supervision; không bảo đảm identifiability toàn cục. |
| H là curvature của surface | Chỉ là Hessian inverse-depth trong image/D4-cell coordinates, không intrinsic curvature. |
| Zero-init là exact V6 no-op | Không nếu bỏ learned vector head. V7 exact reduced-V6 no-op; full-parent metric đo riêng trước train. |
| Bỏ cả connection CNN để nhẹ | Làm mất context đã huấn luyện cho PhaseMetric2. Giữ body62→64 và phase weights; chỉ bỏ vector head195params/3D rotation. |
| Jet chắc chắn ít MAC và nhanh hơn | Không bảo đảm. Six-output head + adapter tăng nhẹ Conv MAC; giảm 3D geometry ops nhưng thêm jet tensors. Latency/VRAM đo cùng GPU và edge device. |
| Top5–10% tail-SSE / bỏ GT conflict | Tail normalization có thể khuếch đại gradient mạnh; sorting tốn thêm work. Bỏ GT khó có nguy cơ lệch mục tiêu. Dùng squared-excess, chuẩn hóa **mọi valid GT**, không thay eval mask. |
| Tăng neighbor mass ~0,8 | Chưa có bằng chứng tốt; dùng ≤0,5 và ít nhất50% center **mỗi step** để bảo vệ local correction. |
| GBPN đã là ECCV2026 | Nguồn kiểm được là preprint2026; không tự xác nhận venue từ đề xuất. |

## 5. Loss budget V6 cuối run

| Weighted term | Contribution | % total=0,912492 |
|---|---:|---:|
| Multi-scale GT metric | 0,273561 | 29,98% |
| Output RMSE | 0,210037 | 23,02% |
| Range-RMSE | 0,208785 | 22,88% |
| GT boundary-RMSE | 0,084367 | 9,25% |
| Robust-MSE, 2Huber(δ=20) | 0,071726 | 7,86% |
| Metric teacher KD | 0,022199 | 2,43% |
| Sparse + holdout + trust + barrier + others | ~0,041815 | ~4,58% |

Không có sparse loss chiếm76% như các run cũ; tăng/giảm sparse mạnh không còn là ưu tiên. Log/edge magnitude rất nhỏ, nhưng đổi đồng thời nhiều coefficients sẽ khó quy gain. V7 chỉ thay robust tail term: squared-excess giữ gradient tăng cho |e|>20 m.

## 6. Papers → quyết định có chọn lọc

| Nguồn chính | Nguyên lý dùng / tránh |
|---|---|
| [Depth-Normal Constraints, ICCV2019](https://arxiv.org/abs/1910.06727) | Ràng buộc depth-normal và sensor confidence đã có precedent; không claim lần đầu geometric depth diffusion. |
| [BTS / Local Planar Guidance](https://arxiv.org/abs/1907.10326) | Plane-guided depth reconstruction trong decoder đã có precedent; chỉ nâng bậc polynomial không đủ làm novelty. Claim V7 phải nằm ở shared residual jet translation + phase correction và empirical trade-off. |
| [LIIF, CVPR2021](https://arxiv.org/abs/2012.09161) | Continuous local query đã có nền tảng. V7 dùng polynomial evaluation cố định thay learned MLP query toàn ảnh. |
| [Field Convolutions, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Mitchel_Field_Convolutions_for_Surface_CNNs_ICCV_2021_paper.html) | Parallel transport trên surfaces không mới. V7 dùng recentering polynomial, không gọi geodesic transport. |
| [BP-Net, CVPR2024](https://arxiv.org/abs/2403.11270) | Sparse guidance và early propagation đáng giữ; V7 không thêm graph/search nonlocal. |
| [DFU, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Improving_Depth_Completion_via_Depth_Feature_Upsampling_CVPR_2024_paper.html) | Dense decoder guidance quan trọng cho upsampling: giữ context64 và phase CNN đã học. |
| [OGNI-DC, ECCV2024](https://arxiv.org/abs/2406.11711) | Derivative representation có ích; không dùng ConvGRU + global differentiable integration trong gói edge nhỏ. |
| [OMNI-DC, ICCV2025](https://arxiv.org/abs/2411.19278) | Gradient integration có nguy cơ tích lũy lỗi ở sparse holes; V7 chỉ local residual, không tuyên bố global integration. |
| [DMD³C, CVPR2025](https://arxiv.org/abs/2503.16970) | Dense teacher supervision là training aid; tái sử dụng D_cm/C_cm, không đưa teacher vào inference hoặc claim sao chép nguyên training DMD³C. |
| [InfiniDepth, 2026](https://arxiv.org/abs/2601.03252) | Continuous query cho geometry detail là hướng có cơ sở. Không mang DINOv3-L/large synthetic pretraining/MLP decoder sang2kedge run. |
| [GBPN, preprint2026](https://arxiv.org/abs/2601.21291) | Learned MRF/nonlocal Gaussian belief propagation là hướng khác; không gán “solver mới” cho stencil jet local. |
| [Deep Basis Fitting, WACV2020](https://arxiv.org/abs/1912.10336), [Bayesian DBF, ICCV2021](https://arxiv.org/abs/2103.15254) | Learned basis + differentiable least-squares/Bayesian fitting đã có prior art. V7 có basis monomial cố định local, không solve scene-level coefficients hoặc calibrated uncertainty. |
| [Radar-Guided Polynomial Fitting](https://arxiv.org/abs/2503.17182) | Polynomial calibration trên scaleless depth values đã có. V7 dùng polynomial **theo spatial offsets s,t**, không global polynomial mapping theo monocular depth z/radar. |

Đây là **suy luận thiết kế từ các nguồn**, không phải các paper đã chứng minh V7 sẽ <0,8 m. Không thêm PCA scene template (phải xác định basis/missing-depth/domain prior) hoặc PINN (không có PDE vật lý phù hợp được xác lập). Taylor + perspective projection đã là inductive bias có toán học rõ, không cần gắn thêm tên lý thuyết.

## 7. Chốt một run

Default V7 giữ tất cả V5/V6 shared weights và trained phase head, chỉ bỏ2vector tensors; thêm518params jet/adapter; T=2,4neighbors ở1/4; learned RGB detail/soft trust output giữ nguyên. Fine-tune15epoch từ V6cum59; fresh AdamW, batch4, same KD và GT protocol.

Pass nghiên cứu tối thiểu: tốt hơn **full V6 same-runtime** và không tăng latency median>10% trên cùng device/settings; đồng thời xem far/edge/iRMSE/absolute tail SSE. Đạt<0,8 là pass mục tiêu mạnh, **không phải hứa hẹn**. Nếu chỉ giảm vài mm, không tuyên bố breakthrough; chạy reduced-V6 same recipe để phân biệt tail-loss gain với jet gain.
