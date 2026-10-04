# AnchorFlow V10.1 LiteMetric

V10 mới hoàn tất: **best epoch23, RMSE0.99626m, iRMSE3.40021km⁻¹**; early stop sau32epoch. Epoch29 có iRMSE tốt hơn3.33797 nhưng RMSE1.00186, không được ghép hai epoch thành cùng kết quả.

Phân tích đầy đủ: [ANALYSIS_V10.md](../drive_upload/AnchorFlow_v10_1_LiteMetric/ANALYSIS_V10.md).

## Quyết định

- Giữ encoder, sparse pyramid, feedback jet field, five-jet consensus, sparse innovation và soft sensor fusion.
- Bỏ stop controller/stop BCE/exit exploration/rollout3–4; default dùng2 Euler steps, **h vẫn learned từ current state/context**, bounds1/6–1/3, fresh init0.25.
- Thêm quarter-grid zero-init context bypass dùng chung cho final phase sampler và final detail/trust.
- Loss trực tiếp full-output iRMSE,0.04/ramp4epoch; global RMSE0.4→0.6. Default bỏ tiny log/edge/teacher-edge nhưng giữ control config.
- Giữ2teacher như V10; metric-only là lựa chọn ablation riêng.

Nguyên lý, công thức, tensor và objective: [ARCHITECTURE.md](../drive_upload/AnchorFlow_v10_1_LiteMetric/ARCHITECTURE.md).

## Chạy

Upload folder **AnchorFlow_v10_1_LiteMetric** vào MyDrive, mở [notebook](../drive_upload/AnchorFlow_v10_1_LiteMetric/Train_V10_1_LiteMetric_TAR2000.ipynb), chọn GPU, chạy từ trên xuống. Data giữ nguyên **GeoLift_Data/teacher_subset_2000**,1600train/400val/1000test.

Default fresh40 + early stop; optional model-only fine20 từ V10best23. [README](../drive_upload/AnchorFlow_v10_1_LiteMetric/README.md) ghi rõ đường dẫn, resume và output.

Tổng582.945params, registered Conv/Linear3.287G; kiểm thử CPU contracts, full-image backward,9ONNX cases và actual runner save/resume được ghi tại local_verification.json. **Chưa có accuracy/GPU runtime sau train V10.1; RMSE<0.9, iRMSE<3.2 vẫn là mục tiêu.** V10 gốc và historical bundles không bị sửa.
