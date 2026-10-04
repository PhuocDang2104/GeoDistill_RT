# AnchorFlow V7 — Projective Jet candidate

V7 tiếp nối V5/V6, không thay baseline đã train. Gói hoàn chỉnh:

- [Drive folder và cách chạy](../drive_upload/AnchorFlow_v7_Jet/README.md)
- [Notebook FT15](../drive_upload/AnchorFlow_v7_Jet/AnchorFlow_v7_Jet_TAR2000_FT15.ipynb)
- [Kiến trúc và công thức](../drive_upload/AnchorFlow_v7_Jet/ARCHITECTURE.md)
- [Phân tích log V6, phản biện và paper references](../drive_upload/AnchorFlow_v7_Jet/ANALYSIS_V6.md)
- [Kiểm chứng local và giới hạn](../drive_upload/AnchorFlow_v7_Jet/VERIFICATION.md)

## Tóm tắt

Giữ MobileNetV4, sparse pyramid, quarter AnchorFlow, V5 plane/barriers và V6 learned context/phase. Thay vector correction3D bằng residual inverse-depth2-jet6coefficients; analytic translation4neighbors/T=2; reusejet cho4phaseD4→D2. Không global solver/attention/teacher inference.

| Thuộc tính | V6 | V7 |
|---|---:|---:|
| Parameters | 612.896 | 613.219 |
| Conv/Linear MAC,352×1216 | 4,023675 G | 4,032235 G |
| Validation RMSE đã train | 1,009297 m | **Chưa train** |
| Full V6 CPU400val | 1,009274 m | — |
| Bỏ vector, giữ context/phase, CPU400val | — | 1,009439 m tại zero initialization |
| GPU/edge latency | Phải đo cùng setup | Notebook đo, chưa có local CUDA |

ZeroV7khớp **reduced V6**, không exact fullV6. Parent fullV6best epoch14 được bundle, SHA pin. 15epoch bổ sung, cùng1.600train/400val; metric teacherD_cm/C_cm chỉ train. Robust-Huber tail được thay bằng squared-excess risk0,1trên tất cả validGT; không bỏ GTconflict khỏi metric.

<0,8m là mục tiêu; từ V6 cần giảm~37,17%MSE. Đây là candidate có lý thuyết/local checks, **không** mô hình đã được chứng minh đạt mục tiêu hoặc first-ever novelty. Có optional reduced-V6 same-recipe vàT=0controls, không tự chạy nhiều experiments.

Chỉ upload folder `AnchorFlow_v7_Jet`; data vẫn ở `MyDrive/GeoLift_Data/teacher_subset_2000/`. Output riêng ở `MyDrive/GeoLift_RT_Runs/AnchorFlow_v7_Jet_FromV6_FT15/metric_kd/`.
