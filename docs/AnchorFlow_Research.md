# AnchorFlow Research · V3 baseline / V4 ContextWide

Current canonical folder: [AnchorFlow_Research](../drive_upload/AnchorFlow_Research/README.md).

## Run vừa hoàn thành

V3 sau thêm 15 epoch: **RMSE 1,04537 m**, cải thiện 9,50% so mốc khởi tạo1,15514m. Vùng 0–20m tốt hơn21,70%, 20–40m tốt hơn12,83%, nhưng 40–80m gần như chưa tiến bộ. Full400val prediction audit cho thấy0,204% pixel lỗi>10m đóng góp59,84% squared error.

## Candidate tiếp theo

**V4 ContextWide**, tăng0,572M→1,858M parameters và2,974G→6,344G countedMAC. Thêm latent pyramid192/96/48 ở1/16→1/8→1/4 để nâng khả năng học context; giữ flow/sensor fusion/loss. Khởi tạo output đúng v3 nhờ zero projection. **Chưa có kết quả train v4 hoặc bảo đảm <0,7m.**

| Mục | File |
|---|---|
| Upload ZIP | [AnchorFlow_Research.zip](../drive_upload/AnchorFlow_Research.zip) |
| Notebook | [AnchorFlow_Research_TAR2000_15ep.ipynb](../drive_upload/AnchorFlow_Research/AnchorFlow_Research_TAR2000_15ep.ipynb) |
| Hướng dẫn Drive | [README](../drive_upload/AnchorFlow_Research/README.md) |
| Phân tích log/tail/bias | [ANALYSIS_V3](../drive_upload/AnchorFlow_Research/ANALYSIS_V3.md) |
| Kiến trúc/công thức/paper | [ARCHITECTURE](../drive_upload/AnchorFlow_Research/ARCHITECTURE.md) |
| Kiểm chứng | [VERIFICATION](../drive_upload/AnchorFlow_Research/VERIFICATION.md) |

Đổi MODEL_NAME=v4 hoặc v3 trong một notebook. Hai model dùng chung code train/data/loss/evaluation. V3 control cần train tiếp cùng15epoch nếu muốn kết luận gain do kiến trúc, không chỉ do thêm training budget.

## Git và cleanup ngày 2026-10-02

- Đã fetch và pull --ff-only origin/main: **3e574ee → c7db611**, không conflict.
- Remote bổ sung S3 augmentation và server/Docker runtime. Giữ nguyên các phần này; không tự áp dụng augmentation vào AnchorFlow benchmark.
- Bundle AnchorFlow-DC/v2 và các standalone bundle/script cũ được chuyển ra **C:/Users/ADMIN/Desktop/GeoDistill_RT_archive/pre_v4_20261002**, có thể khôi phục. Active upload chỉ cần AnchorFlow_Research; baseline v3 vẫn có trong model_v3.py.
- Giữ data, teacher cache, results, checkpoint source ZIP và paper. Không dùng git reset/clean để xóa worktree.
- Checkpoint/ONNX được ignore trong Git nhưng checkpoint parent vẫn đóng trong ZIP upload Drive. Không push/commit tự động.

Input không đổi: MyDrive/GeoLift_Data/teacher_subset_2000. Output candidate: MyDrive/GeoLift_RT_Runs/AnchorFlow_v4_ContextWide_FT15/metric_kd.
