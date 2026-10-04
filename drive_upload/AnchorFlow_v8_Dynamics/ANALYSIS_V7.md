# V7 → V8: kết luận từ run đã hoàn tất

Nguồn duy nhất cho số liệu V7 ở đây: [`v7_train_log.csv`](v7_train_log.csv), bản sao nguyên byte của `train_log_v7.csv`. 15 epoch mới, 1.600 train / 400 val, 400 optimizer updates/epoch, cumulative index 60–74. Không có prediction/checkpoint V7 mới trong đầu vào của lượt này: **không giả vờ đã audit pixel hoặc chạy causal ablation trên checkpoint V7**.

## 1. Kết quả chính

| Metric | Epoch 0 | Best RMSE — epoch 7 | Epoch 14 |
|---|---:|---:|---:|
| Global RMSE (m) | 1,016791 | **1,006078** | 1,008006 |
| MAE (m) | 0,258801 | 0,257185 | 0,256997 |
| iRMSE (km⁻¹) | 3,567558 | 3,513848 | 3,474600 |
| Pre-sensor-fusion RMSE (m) | 1,027944 | 1,016564 | 1,019131 |
| Legacy hard-anchor RMSE (m) | 1,151420 | 1,141605 | 1,143575 |
| Training total loss | 0,907775 | 0,882137 | 0,854676 |

Best cũ V6 được ghi ở bundle V7: khoảng 1,009290 m. V7 giảm khoảng **3,21 mm, 0,32% RMSE** so với reference đó. Đây là so sánh lịch sử, không phải matched-runtime ablation chứng minh jet gây ra toàn bộ cải thiện: V7 đổi cả tail loss và đã fine-tune thêm.

Training loss giảm 5,85% từ epoch 0 đến 14, nhưng validation chỉ giảm 0,86%. Best ở giữa run; chưa có evidence rằng train lâu hơn cùng recipe sẽ tự xuống 0,8 m. Hơn nữa weighted KD giảm do lịch giảm hệ số, nên total loss đi xuống không đồng nghĩa tất cả task đều học tốt hơn.

## 2. Jet không chết, nhưng tác động đo được rất nhỏ

So sánh **cùng grid và cùng target** tại epoch 7:

| Chuỗi | RMSE trước → sau (m) | Cải thiện |
|---|---:|---:|
| D4 surface → D4 jet | 1,762518 → 1,759421 | **3,10 mm** |
| D2 base → D2 analytic jet | 1,282139 → 1,281880 | **0,26 mm** |
| D2 analytic jet → learned phase correction | 1,281880 → 1,252955 | **28,93 mm** |

Jet correction magnitude ở train khoảng 0,049 m tại epoch 7, 0,058 m cuối run; gradient/Hessian và parameter norm tăng. Điều này bác bỏ diễn giải đơn giản “head không học gì”, nhưng không chứng minh correction đúng hướng hoặc generalize. Correction magnitude không phải RMSE gain. Cũng không được so trực tiếp D4 RMSE với D2 RMSE rồi quy toàn bộ chênh lệch cho upsampling: GT pooling/valid support khác nhau.

**Giữ learned phase readout**: đây là phần có gain native-grid rõ hơn. **Thay static jet predictor + fixed transport** bằng state-feedback dynamics, không thêm một stack correction nữa lên trên mọi module cũ.

## 3. Bottleneck còn lại là tail và geometry/boundary vùng xa

| GT range / region | Best V7 RMSE (m) |
|---|---:|
| 0–20 m | 0,453026 |
| 20–40 m | 1,395564 |
| 40–60 m | 2,568589 |
| 60–80 m | 3,871171 |
| 80–120 m | 9,225996 |
| RGB edge | 1,425998 |
| GT boundary band 3 px | 1,702928 |

Tại best V7, pixel có absolute error >5 m đóng góp **78,12% SSE**; >20 m đóng góp **32,14% SSE**. Đây là **phần sai số bình phương**, không phải tỷ lệ pixel. CSV không có tail pixel counts của validation nên không suy diễn tỷ lệ pixel từ bảng này.

Từ 1,006078 xuống 0,8 m cần giảm khoảng **36,77% MSE**; xuống 0,7 m cần khoảng **51,59% MSE**. Chỉ một correction nhỏ hoặc đổi tên thành neural PDE không thể bảo đảm mức giảm đó.

Soft sensor fusion tốt hơn hard anchor khoảng 0,136 m tại best, còn trust conflict fraction ở train khoảng 5,39%. Vì vậy giữ learned sensor reliability và báo hard-anchor riêng; không dùng GT để lọc sensor lúc validation.

## 4. Phản biện đề xuất dynamics

- Đúng: **jet block V7** predict residual một lần; hai transport steps dùng guidance cached. Chưa phải learned feedback evolution của jet.
- Cần bổ sung: **toàn model V7 đã có QuarterFlow recurrent** cập nhật scalar correction theo state hiện tại. “V7 hoàn toàn không có dynamics” là không chính xác.
- Neural ODE/TNRD/SPN/recurrent gradient refinement đều có prior art. Không claim phát minh vector field hay Euler method.
- [OMNI-DC, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zuo_OMNI-DC_Highly_Robust_Depth_Completion_with_Multiresolution_Depth_Integration_ICCV_2025_paper.pdf) thậm chí bỏ ConvGRU recurrent updates trong large-scale training vì không thấy ích lợi. Vì vậy “dynamic hơn” là đặc tính implementation cần kiểm chứng, không phải bằng chứng sẽ tốt hơn về metric.
- Jet 6 chiều là geometric-state field, **không phải optical-flow displacement field** hay velocity 3D. Pseudo-time không phải thời gian video.
- Ba steps chỉ có local transport radius ba quarter-grid cells. Global context đến từ encoder/F32; đây không thay thế được global integration solver của OGNI/OMNI.
- Không có guarantee RMSE giảm từng step hoặc hội tụ. Conductance bound chỉ kiểm soát hệ số mixing; không chứng minh stability toàn bộ jet khi có Hessian transport và learned reaction.

## 5. Quyết định V8

Một bundle nghiên cứu rõ ràng: **shared feedback jet dynamics** thay QuarterFlow + MetricRefine4 + V5 surface + V7 static jet; giữ CNN backbone, gated sparse fusion, phase readout và soft sensor trust. Forcing và conductance được recompute từ jet/sensor residual tại từng step. Bounded projected Euler, fixed horizon T=3, matrix-free stencil tại 1/4 resolution.

Train mới tối đa epoch 0–29, early-stop patience7/min_delta0,001m/minimum15 completed epochs; ImageNet RGB pretrained, không chuyển student cũ. Giữ GT-priority metric teacher, range/boundary/tail objective; thêm intermediate-step GT supervision và ramp squared-tail 2 epoch đầu để recipe phù hợp fresh initialization. Vì thay initialization và objective, đây **không phải architecture-only continuation ablation**.

Control có sẵn nhưng không tự chạy: `v8_frozen_feedback`, cùng backbone, parameters, T=3, teacher, recipe, seed, max budget và early-stop policy; chỉ CNN/compatibility đọc state ban đầu thay state hiện tại. Phải train control riêng để đánh giá lợi ích feedback và báo actual epochs. Không so fresh30+ES với V7 cumulative75 rồi kết luận dynamics thắng/thua.

Mục tiêu <0,8 m chưa được chứng minh. Nếu chưa đạt: inspect step RMSE, tail SSE, GT boundaries và train/val gap trước khi tăng steps/width hoặc đặt thêm teacher role.
