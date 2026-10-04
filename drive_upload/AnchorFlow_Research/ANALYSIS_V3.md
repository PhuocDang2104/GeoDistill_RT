# Audit v3: 15 epoch mới và hướng tới <0,7 m

Nguồn: **metric_kd-20261002T074647Z-1-001.zip**. Run v3 metric_kd, thêm 15 epoch index0–14 trên v2 đã học15epoch; cumulative0–29. Best mới epoch14. **1.600 train / 400 validation**, cùng 25.424.992 pixel GT. Test anonymous1.000 không có public GT. Không phải điểm KITTI leaderboard.

## 1. V3 có cải thiện thật, nhưng chủ yếu ở gần

| Metric | Trước 15 epoch mới | Best sau 15 epoch | Thay đổi |
|---|---:|---:|---:|
| Global RMSE | 1,15514 m | **1,04537 m** | −9,50% |
| MAE | 0,29395 m | **0,26874 m** | −8,57% |
| iRMSE | 4,2433 km⁻¹ | **3,7651 km⁻¹** | −11,27% |
| Edge RMSE | 1,68448 m | **1,48945 m** | −11,58% |
| 0–20 m RMSE | 0,63732 m | **0,49900 m** | −21,70% |
| 20–40 m RMSE | 1,69076 m | **1,47382 m** | −12,83% |
| 40–60 m RMSE | 2,59137 m | 2,56864 m | −0,88% |
| 60–80 m RMSE | 3,90757 m | 3,93323 m | **+0,66%** |
| 80–120 m RMSE | 10,04657 m | 9,62191 m | −4,23% |

Không nên tiếp tục chỉ tối ưu sensor trust/biên gần. Vùng40–80m còn chiếm **46,44% squared error**, trong khi chỉ khoảng5,81% GT pixels.

Sensor policy cuối cùng:
- D1_base:1,06527m → detail D1:1,05503m → soft D_full:**1,04537m**.
- Ép hard-anchor trên D1 mới lại xấu thành **1,17406m**.
- Giữ soft fusion; không quay về ép sensor tuyệt đối.

## 2. Full prediction audit 400 ảnh, không chỉ đọc CSV

Đã chạy lại checkpoint best thật trên CPU FP32: RMSE1,045427m, rất gần A100 FP16 trong artifact1,045367m. Không fit thêm parameter/threshold hoặc dùng GT trong forward.

| Nhóm pixel theo absolute error | Tỷ lệ valid GT pixels | Tỷ lệ toàn bộ squared error |
|---|---:|---:|
| >1 m | 4,509% | 95,81% |
| >2 m | 2,020% | 91,27% |
| >5 m | **0,624%** | **78,70%** |
| >10 m | **0,204%** | **59,84%** |
| >20 m | 0,044% | 32,15% |

Median absolute error chỉ0,0797m; p99=3,518m và p99.9=14,269m. Điểm nghẽn là **đuôi lỗi lớn**, không phải toàn bộ depth map đều sai một scale cố định.

Signed bias:
- 0–20m:+0,0316m;20–40m:+0,0377m.
- 40–60m:−0,1411m;60–80m:**−0,9300m**;80–120m:−5,9370m.

Có xu hướng underestimate ở xa, nhưng không nên chữa bằng nhân depth theo GT range ở evaluation. Bias mean cũng không giải thích hết các outlier hai phía. File **parent_prediction_audit.json** có worst20 sample theo SSE để inspect tiếp; không loại chúng khỏi validation.

## 3. Đóng góp stage và convergence

| Stage, native-scale target | Initial | Best |
|---|---:|---:|
| D16 | 3,4862 | 3,0087 |
| D8 | 2,4104 | 2,2955 |
| D0 @1/4 | 2,2608 | 2,0840 |
| D4_flow @1/4 | 1,9805 | 1,9472 |
| D4 sau metric refine @1/4 | 1,9805 | 1,9066 |
| D2 | 1,3886 | 1,3120 |
| D1_base | 1,1575 | 1,0653 |
| D1 | 1,1575 | 1,0550 |
| D_full | 1,1551 | 1,0454 |

Chỉ so stage **cùng resolution/GT target** để tách flow/refinement. D16 và D1 không cùng benchmark; không diễn giải chênh lệch RMSE thành contribution giữa scale.

Epoch0/4/9/14 lần lượt1,1101 /1,0675 /1,0593 /1,0454m. Cuối run vẫn đạt best nhưng tốc độ cải thiện chậm. Chưa đủ dữ kiện để kết luận tuyệt đối “thiếu params” hoặc “overfit”; validation-loss gap không thể lấy trực tiếp từ batch train RMSE có holdout/augmentation.

## 4. Loss budget đã hợp lý hơn trước

| Term weighted epoch14 | Giá trị | % total0,832418 |
|---|---:|---:|
| GT multi-scale | 0,316654 | 38,04% |
| Global batch RMSE | 0,230116 | 27,64% |
| Range RMSE | 0,220728 | 26,52% |
| Metric KD | 0,025446 | 3,06% |
| Holdout | 0,024134 | 2,90% |
| Trust | 0,007750 | 0,93% |
| Sparse | 0,007487 | 0,90% |

KD không còn chi phối loss. Coverage đủ điều kiện khoảng71,23%, mean confidence0,7995.
Mean |delta4|0,1282m, |delta1|0,0515m; các nhánh thực sự đã học. Sensor gate mean khoảng0,823.

Log/edge weighted vẫn rất nhỏ, nhưng tăng hàng loạt loss cùng kiến trúc sẽ làm mất khả năng so sánh. **Run v4 giữ objective v3**, chỉ tăng capacity model. Nếu sau control không có gain, mới tách một thí nghiệm objective/outlier-training riêng.

## 5. Lựa chọn v4

Tăng capacity trong **context/decoder ở1/16 và1/8**, nơi nhìn được vùng rộng và ảnh hưởng sparse thiếu support; giữ processing high-res nhẹ.

- Thêm dense latent pyramid192→96→48, depthwise5×5 dilation1/2/3 và global context.
- Truyền latent từ F32 qua1/16→1/8→1/4; inject vào decoder và phase context.
- Giữ3flow steps, sensor policy, teacher recipe và toàn bộ weights v3.
- Không thay encoder sang model lớn khác làm mất checkpoint; không chồng thêm full-resolution propagation.
- Projection zero-init: toàn bộ output ban đầu khớp v3. Học thêm từ mốc tốt hiện tại, không bắt đầu lại.

V4 **1,8577M params /6,3436G counted MAC** so v3 **0,5720M /2,9741G**. Đây là tăng đáng kể (3,25×params;2,13×MAC), không mô tả như chi phí gần miễn phí. GPU latency mới phải đo; v3 đã đo A10025,404ms median/P9526,261ms/91,95MiB, không so ngang với T4 ở run v2.

## 6. Mục tiêu <0,7m cần gì?

Từ1,04537→0,7m cần giảm **55,16% MSE** nữa. Đây là bước lớn, chưa có căn cứ hứa một lần tăng width sẽ đủ.

Một phân bổ RMSE theo bin **minh họa mục tiêu**, không phải dự báo:

| Bin | Hiện tại | Mốc minh họa |
|---|---:|---:|
| 0–20m | 0,499 | 0,40 |
| 20–40m | 1,474 | 0,90 |
| 40–60m | 2,569 | 1,50 |
| 60–80m | 3,933 | 2,50 |
| 80–120m | 9,622 | 6,00 |

Với cùng pixel distribution, bảng này tương ứng global khoảng0,676m. Nó cho thấy phải tiến bộ nhiều ở trung–xa, không chỉ near.

Ưu tiên sau run:
1. So v4 với **v3-control train thêm cùng15epoch** trước khi nhận định gain do kiến trúc.
2. Nếu train fit tốt hơn mà val xa không tiến bộ: ưu tiên thêm scene/raw-drive training độc lập (có thể mở từ1.600 lên phần train của cache10k), giữ400val không leakage.
3. Inspect worst20 sample, occlusion/boundary/resize/GT mismatch và teacher sai tại vùng không GT. Không “chữa metric” bằng bỏ sample/mask.
4. Thử loss nhạy tail hoặc dense-feature teacher trong **thí nghiệm riêng** nếu capacity control không đủ; metric teacher fused với GT không phải oracle độc lập.
5. Target edge cần profile thật, rồi mới cân nhắc giảm block/INT8/TensorRT. Không suy latency từ param count.

Việc thường xuyên chọn kiến trúc theo cùng400val có thể overfit validation. Khi chốt thiết kế cần held-out drive hoặc KITTI official submission, không coi điểm subset là SOTA.
