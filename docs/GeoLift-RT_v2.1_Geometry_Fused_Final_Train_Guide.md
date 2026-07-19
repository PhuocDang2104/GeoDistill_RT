# GeoLift-RT v2.1 — fused geometry final train

Notebook: [`GeoLift_RT_v2_1_Geometry_Inspect_Final_Train.ipynb`](../notebooks/GeoLift_RT_v2_1_Geometry_Inspect_Final_Train.ipynb).

## Chạy

1. Đưa repo hiện tại lên `MyDrive/GeoDistill_RT`.
2. Mở notebook trên Colab, chọn GPU và kiểm tra `DRIVE_REPO_DIR`/`RUN_DRIVE`.
3. Cần tối thiểu khoảng 85 GiB local SSD trống khi ba teacher tar cùng tồn tại.
4. Chạy lần lượt từ trên xuống. Không bỏ qua Gate A hoặc Gate B.

Nếu public ID báo `Too many users have viewed or downloaded`, hãy tạo bản copy trong Drive cá nhân, share `Anyone with the link`, rồi thay `GEOMETRY_FUSED_ID` (và ID teacher nào bị quota) trong cell cấu hình. Gate A dùng cùng cơ chế xác nhận download nên sẽ dừng rõ ràng trước khi train.

## Hai gate bắt buộc

Gate A stream-inspect archive trước khi tải toàn bộ:

```text
layout = geometry_fused/train/*.npz
keys   = R_G, C_G
R_G/C_G cùng shape, float32, finite
C_G trong [0,1]
```

Gate B chạy sau khi đã tách train/val:

- metric, DA raw và fused geometry coverage đều 100%;
- 800 train và 200 val không trùng ID hoặc raw drive;
- filename cũ được canonicalize về ID dataset;
- loader đọc trực tiếp fused `R_G/C_G`, không normalize lần hai;
- `geometry_fallback=false`, nên DA/DMD không thể làm preflight pass giả.

Notebook hiển thị riêng RGB, sparse, GT, metric teacher, DA raw, `R_G` và `C_G` của một mẫu train và một mẫu val trước khi cho phép train.

## Teacher role trong final loss

| Cache | Dùng trong loss cuối |
|---|---|
| `metric_coarse` | Reliability-weighted metric KD |
| `geometry_fused` | SSI relative geometry + ordinal/boundary supervision |
| `depth_anything/train_raw` | Inspect/audit và ablation; không fallback trong final run |

Archive fused đã kiểm tra chỉ có `R_G/C_G`. Nó không chứa normal, inverse-depth slope `a_T,b_T` hoặc planarity target `eta_T`. Vì vậy final run dùng đầy đủ fused relative geometry nhưng vẫn để `lambda_plane=0` và ghi `slope_supervision=false`; không gọi đây là full slope-T3.

## Kết quả trên Drive

```text
RUN_DRIVE/
├── geometry_remote_preview.json
├── teacher_train_val_report.json
├── teacher_subset_manifest.json
├── run_manifest.json
├── checkpoints/{last.pth,best.pth}
├── logs/infer_val_metrics_global.json
├── logs/geolift_component_profile.json
└── kitti_test_1000.zip
```
