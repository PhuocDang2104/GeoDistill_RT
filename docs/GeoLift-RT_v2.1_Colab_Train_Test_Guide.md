# GeoLift-RT v2.1 — hướng dẫn Colab

Notebook chạy end-to-end: [`GeoLift_RT_v2_1_Colab_Train_Test.ipynb`](../notebooks/GeoLift_RT_v2_1_Colab_Train_Test.ipynb).

## Chạy nhanh

1. Upload toàn bộ repo vào `MyDrive/GeoDistill_RT`, hoặc đặt `REPO_GIT_URL` ở cell cấu hình.
2. Mở notebook bằng Google Colab và chọn GPU runtime.
3. Kiểm tra `DRIVE_REPO_DIR`, `RUN_DRIVE`, `BATCH_SIZE`; sau đó chọn **Runtime → Run all**.
4. Không đổi hai Drive ID teacher nếu muốn dùng đúng dữ liệu trong `Install_dataset.md`.

Notebook tự thực hiện:

```text
hai teacher tar (metric + DA)
  → lấy giao sample ID
  → chọn cố định seed=42: 800 train + 200 val
  → lấy RGB/sparse/GT/calibration tương ứng từ KITTI chính thức
  → lấy KITTI test anonymous chính thức: 1.000 ảnh
  → unit test + teacher coverage + hard-anchor preflight
  → train/resume GeoLift-S2
  → val metric + stage RMSE
  → xuất 1.000 benchmark PNG + component runtime profile
```

Teacher tar khoảng 27 GB và 14 GB. Cần tối thiểu khoảng 55 GB SSD Colab trống trước lúc tải. Notebook chỉ giải nén 1.000 NPZ cần dùng rồi xóa tar. KITTI depth ZIP và raw-drive ZIP cũng được xóa sau khi trích các frame đã chọn.

## Protocol chính xác

| Phần | Số mẫu | Có GT? | Mục đích |
|---|---:|---:|---|
| `train_800.txt` | 800 | Có | Train student |
| `val_200.txt` | 200 | Có | Chọn checkpoint và tính metric; drive-disjoint với train |
| `test_1000.txt` | 1.000 | Không công khai | Xuất PNG để nộp KITTI |

Không báo “test RMSE” từ 1.000 ảnh anonymous. RMSE/MAE/iRMSE/iMAE local chỉ có thể tính trên `val_200` có ground truth.

Mặc định `SELECTION_STRATEGY='min_drives'` để giảm số KITTI raw drive phải tải. Train và validation được cấp từ các raw drive tách rời để tránh leakage theo sequence. Đây là subset kiểm tra kiến trúc và ablation, không phải protocol train đầy đủ 93k của benchmark. Nếu đã có đủ storage/bandwidth và muốn chọn drive theo thứ tự ngẫu nhiên, đổi thành `uniform`.

## Kiến trúc và teacher profile

Code thực thi là `GeoLiftStudentS2`, được chọn bởi:

```yaml
model:
  architecture: geolift_s2
```

Graph gồm compact sparse prior, learned stems ở 1/4, fusion `42→32`, stage-adapted MobileViTv2, FPN-24, PPG và bốn RayLift-ID `5→3→3→2`, kết thúc bằng hard sparse anchoring.

Hai archive hiện có tạo profile **T2**:

- `metric_coarse_train.tar`: metric KD;
- `depth_anything_train_raw.tar`: SSI/relative geometry.

Không có DSINE slope/planarity trong hai archive, nên config để `lambda_plane: 0.0`. Kết quả không được ghi là full T3/RSGD.

## Resume và kết quả

Sau mỗi epoch, trainer backup `last.pth`, `best.pth` và logs vào `RUN_DRIVE`. Khi chạy lại notebook, `last.pth` tự được copy về SSD và resume cả model, optimizer, scaler và scheduler.

Các file chính trên Drive:

```text
RUN_DRIVE/
├── checkpoints/{last.pth,best.pth}
├── logs/infer_val_metrics_global.json
├── logs/geolift_component_profile.json
├── run_manifest.json
├── teacher_subset_manifest.json
├── val_predictions/
├── test_predictions/benchmark_png/       # đúng 1.000 PNG
└── kitti_test_1000.zip
```

`infer_val_metrics_global.json` chứa metric global-pixel và `stage_rmse_m` cho `D_init`, `D16`, `D8`, `D4`, `D2`, `D1`, `D_full`. Lưu ý `D_init` là kết quả sau local sparse propagation; `D_full` đã hard-anchor nên phải đọc thêm `D1`/`D_pre_anchor` để đánh giá phần học của student mà không bị sparse points làm đẹp nhân tạo.

`geolift_component_profile.json` đo batch 1, FP16, `352×1216`, CUDA synchronize, gồm median/P95, FPS, peak VRAM và runtime từng component. Runtime không gồm đọc file/DataLoader.

## Lỗi thường gặp

- `Drive quota`: tạo bản copy hai tar trong Drive cá nhân, share lại và thay ID.
- `CUDA OOM`: đổi `BATCH_SIZE=2` xuống `1`; không giảm resolution nếu muốn so runtime/metric cùng protocol.
- `teacher coverage < 95%`: dataset và teacher không cùng sample ID; không bỏ qua preflight.
- `Only ... valid common teacher IDs`: archive hỏng, tải thiếu, hoặc hai archive không cùng split.
- Không thấy `best.pth`: xem `RUN_DRIVE/logs/train_student.log`; không dùng checkpoint A0 cho GeoLift-S2.
