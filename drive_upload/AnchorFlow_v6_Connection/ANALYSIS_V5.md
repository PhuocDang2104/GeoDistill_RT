# Full V5 audit và quyết định V6

Nguồn: người dùng cung cấp `results/train_log (4).csv` đủ15epoch0–14, `gt_boundary_metrics.csv`, `kitti_test_predictions.zip`. Bản copy log/metrics nằm ngay trong bundle. CSV cũ `(3)` chỉ có11epoch, không còn là nguồn kết luận cuối. ZIP có1.000 prediction PNG, **không chứa checkpoint hoặc public test GT**.

## 1. Kết quả thực tế

Best global là **epoch14**, cumulative44, tức V3 đã30epoch + V5 thêm15epoch. Epoch10 tốt nhất trong log cũ không còn là best.

| Metric | V5 epoch10 | V5 best epoch14 |
|---|---:|---:|
| Global RMSE m | 1,03005 | **1,02341** |
| MAE m | 0,26452 | 0,26369 |
| iRMSE km⁻¹ | 3,68544 | 3,64396 |
| Pre-fusion RMSE m | 1,04009 | 1,03383 |
| Legacy hard-anchor RMSE m | 1,16143 | 1,15617 |
| Near0–20m RMSE | 0,49710 | 0,48146 |
| 20–40m RMSE | 1,45870 | 1,43401 |
| 40–60m RMSE | 2,52918 | 2,54782 |
| 60–80m RMSE | 3,81915 | 3,87064 |
| 80–120m RMSE | 9,34400 | 9,41989 |
| RGB-edge RMSE | 1,46350 | 1,45211 |
| Observed-GT band3px RMSE | 1,71531 | 1,71509 |

Global improvement cuối chủ yếu ở near/mid; 40–80m hơi xấu lại so epoch10. Không được kết luận mọi range đều cải thiện. V4 historical best1,03008m; V5 best giảm~0,65% RMSE, nhưng initialization/budget khác nên không gọi matched architecture gain. V3 parent1,04537→V5 1,02341 là~2,10% với thêm15epoch và loss mới.

Median train time epoch1–14 **102,84s**, tổng epoch quanh130s. CSV không ghi GPU nên không tự gán A100 cho run này. Parameter norm không cho biết gradient importance.

## 2. Bottleneck có bằng chứng

- D4_base1,83969→D4surface1,82430: nhánh surface giúp nhưng native RMSE chỉ giảm~0,84%; không phải bước sửa metric lớn.
- Mean surface correction **0,04526m**, transport0,03614m/reaction0,01870m. Hai thành phần có thể triệt nhau nên tổng magnitude không bằng tổng hai mean.
- Surface amplitude tăng0,00237→0,14194, transport mass0,434→0,802. Branch **đã học**, không có bằng chứng toàn bộ gate chết. Average4,5cm không chứng minh mọi outlier đều chỉ sửa4,5cm; cần distribution/tail audit.
- Barrier BCE0,820→0,226; barrier probability mean không giảm đơn điệu. Không gọi barrier collapse chỉ vì average gate nhỏ.
- Weighted budget epoch14: metric33,82%; RMSE24,95%; range23,89%; boundary9,83%; KD2,72%; holdout2,77%; barrier0,26%. Sparse không còn chiếm76% như S2 cũ; tăng teacher weight mù không được bằng chứng hiện tại ủng hộ.

## 3. Boundary không phải toàn bộ lỗi

GT band3 gồm3.346.179/25.424.992pixels =13,16%, chiếm36,96% global SSE; band10 chiếm66,28%. Band là vùng quanh discontinuity **quan sát được trên GT thưa**, không phải semantic edge hoàn hảo.

Nếu giả sử chỉ sửa band3 hoàn hảo và giữ mọi pixel ngoài band nguyên trạng, RMSE còn khoảng **0,81254m**, vẫn >0,7. Đây là phép tính **conditional**, không phải lower bound cho kiến trúc có thể sửa cả interior. Vì vậy V6 cần metric residual ngoài boundary, không chỉ barrier/edge head.

Audit V4 CPU trước đó: lỗi>10m chỉ~0,198%pixels nhưng~59,9%SSE, phần lớn tail nằm ngoài band3. **Không chuyển tỷ lệ đó thành số của V5**: user không đưa V5 validation predictions/checkpoint local. V6 evaluator đã thêm error-tail audit để đo lại thật.

Từ1,02341 xuống0,7 cần giảm **53,22% MSE**; xuống0,6 cần~65,63%. Không có lý do hợp lệ để hứa chỉ thêm một block là đạt. Test ZIP minh họa có đường/xe khá rõ, thin structures và horizon/sky vẫn không nên suy độ chính xác từ màu; anonymous test không có public GT.

## 4. Phản biện `v6_suggest.md`: PT-PSF

| Ý tưởng | Kết luận kỹ thuật |
|---|---|
| Vector ở hai pixel không cộng được | XYZ vectors cùng camera R3 **cộng được**; local frame coefficients mới cần alignment. Giữ ambient-vector control để kiểm tác dụng rotation. |
| Minimal normal rotation = exact parallel transport | Chỉ discrete approximation; exact manifold transport phụ thuộc path/connection. Không claim intrinsic guarantees. |
| Transport mọi coefficient cùng nhau | Tangent vector rotate; normal coefficient là scalar rồi gắn vào destination normal. |
| Arbitrary3D vector rồi lấy z | Fixed-ray depth phải dùng deltaD=(r·v)/(r·r), nếu không không khớp hình học camera. |
| Heun/RK2/ODE ngay | Nhân chi phí field evaluation; chưa có bằng chứng error numerical integration là bottleneck. Bỏ. |
| Connection loss nhân learned barrier | Có thể giảm loss bằng đóng mask. Chưa có nhu cầu/ablation; bỏ regularizer mới. |
| Diffeomorphism/divergence-free | Không phù hợp occlusion/discontinuous depth, không thêm constraint volume. |

V5 đã có camera-aware inverse-plane transport, không chỉ scalar averaging. V6 phải bổ sung correction metric thật và kiểm soát novelty claims, không đổi tên propagation thành flow rồi coi đó là đóng góp.

## 5. Phản biện `v6_addition.md`: PCA + latent attention/ODE

Hướng residual low-rank/global + local detail **đáng nghiên cứu**, nhưng chưa là lựa chọn đáng tin nhất cho một run15epoch hiện tại:

1. Pixel-grid index cố định là correspondence trong **image coordinates**, không phải cùng physical/anatomical point qua các cảnh. PCA trên grid vẫn có thể làm statistical prior; không được dùng đó để suy low-rank surface deformation mạnh như template brain meshes.
2. KITTI GT thưa: `X_GT-X_parent` không dense. Zero-fill residual ngoài valid mask tạo mask/camera-layout modes giả. Cần masked PCA hoặc dense targets có provenance, và fit basis **chỉ1.600train**. Tuyệt đối không fit vào400val.
3. Coarse parent được fine-tune nên residual distribution thay đổi. Basis còn phải xử lý horizontal flip, intrinsics, plane/ray constraint và unit/scale. Learned spectral conditioning cần oracle test trước, không copy EVR từ bài y khoa.
4. Attention K² nhỏ không phải toàn bộ cost: basis encoder/decoder, buffers/readback và chuẩn bị residual dataset cũng tốn. K64 ×3×88×304 =5.136.384 basis entries,~20,55MB FP32, chưa có mean/attention. V6 toàn bộ params612.896,~2,45MB FP32 weights; so memory phải tính cả non-trainable buffers.
5. Vector-PCA correction cũng phải project về camera ray; decode XYZ rồi đọc z là sai fixed-grid constraint. Explained variance không đảm bảo depth RMSE/edge accuracy.
6. Oracle PCA reconstruction0,95m **không chứng minh cả hybrid chết** nếu local branch sửa được thêm; cũng không đủ để hứa0,4m. Cần masked-validation oracle RMSE, train-only basis và same parent/resolution.

[BrainODE primary paper](https://openreview.net/pdf?id=7yOl9qiLWd) và [author code](https://github.com/PWonjung/BrainODE) nghiên cứu longitudinal brain shapes/PCA parameterization, không chứng minh KITTI depth completion. Không chuyển số compactness của bài đó thành guarantee cho outdoor scenes. [Attention-based PCA](https://arxiv.org/abs/2605.18315) chứng minh quan hệ attention–spectral directions trong Gaussian PCA setting; không phải theorem đảm bảo depth completion. Venue/Spotlight/Q1 không được dùng thay thực nghiệm.

**Quyết định:** chưa đưa PCA/attention/ODE vào default V6; giữ ý tưởng separation geometry/detail bằng connection metric@D4 + phase metric@D2. Đây là lựa chọn có scope rõ, no-op warm-start kiểm được, ít thay graph và phù hợp kiểm chứng15epoch. Không claim đã tìm kiến trúc tối ưu tuyệt đối.

## 6. Kế hoạch đánh giá run

- Mặc định từ V5 best thật trên Drive; initial400val phải tái tạo baseline trong tolerance của GPU/precision.
- Train15epoch, tất cả GT/KD/trust/boundary loss bật ngay; thêm robust-MSE đánh SSE tail.
- Chọn best global RMSE; xem near/far/boundary/native stages/error-tail và profile cùng GPU, không chỉ scalar best.
- <0,7 là stretch target. Nếu <1,00 và tail/far/edge tốt hơn mà latency overhead chấp nhận được: mới có tín hiệu giữ. Nếu không cải thiện parent: epoch-1 vẫn giữ best, không tuyên bố thành công.
- Muốn claim architecture: chạy optional V5 cùng parent/recipe/15epoch; muốn claim connection: chạy V6 ambient cùng capacity. Đây là control tùy chọn, notebook Run all không tự nhân run.

Không có V6 trained accuracy, GPU/Jetson latency hoặc tensorRT engine trong bundle. Khả năng deploy mới ở mức static ONNX parity; target hardware phải benchmark riêng.
