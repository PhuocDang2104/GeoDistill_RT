# AnchorFlow V7 — Projective Jet

**Gói độc lập để upload Drive và fine-tune 15 epoch từ V6 best thật.** Mục tiêu validation RMSE <0,8 m; chưa có kết quả train V7 để bảo đảm đạt mục tiêu.

## Chạy ngay

1. Upload nguyên folder `AnchorFlow_v7_Jet` vào `MyDrive/AnchorFlow_v7_Jet`.
2. Mở [AnchorFlow_v7_Jet_TAR2000_FT15.ipynb](AnchorFlow_v7_Jet_TAR2000_FT15.ipynb) bằng Colab, chọn GPU.
3. Giữ cấu hình mặc định, chạy từ trên xuống. Notebook đã kèm `init_v6_best.pth`; **không cần tự tìm checkpoint V6 trên Drive**.
4. Nếu GPU OOM: trước khi bắt đầu run, đổi `BATCH_SIZE=2` và `RUN_TAG="_B2"`. Không đổi giữa lần resume.

## Dữ liệu được dùng

```text
MyDrive/GeoLift_Data/
├── teacher_subset_2000/
│   ├── selected_2000_ids.json
│   ├── kitti_trainval_2000.tar       # RGB + sparse + GT + intrinsics + splits
│   └── metric_coarse_train_2000.tar # D_cm + C_cm; KD chỉ trên 1.600 train
└── test_1000/
    └── kitti_test_1000.tar          # tự lấy official KITTI/cache nếu chưa có
```

Không cần `geometry_fused`, `depth_anything`, DSINE hay weights của teacher. Cache metric teacher có 2.000 records nhưng loader chỉ cache **1.600 train**, không nạp teacher khi validation/test. Teacher là privileged supervision; không lấy teacher-vs-GT overlap làm accuracy độc lập.

Dataset/split/protocol giữ nguyên V6: 1.600 train, 400 val drive-disjoint, 1.000 anonymous test; 352×1216; depth PNG/256; valid GT 0,1–120 m. Data extract và training chạy trên SSD Colab `/content`, **có dùng local disk Colab**, không dùng ổ đĩa máy cá nhân.

## Output và resume

```text
MyDrive/GeoLift_RT_Runs/AnchorFlow_v7_Jet_FromV6_FT15/
├── source_bundle/                  # frozen code/config/parent
├── data_contract.json
├── smoke_report.json
└── metric_kd/
    ├── best.pth / last.pth
    ├── train_log.csv / train_log.jsonl / train.log
    ├── parent_same_runtime_val_metrics.json / initial_val_metrics.json
    ├── val_metrics.json / gt_boundary_metrics.csv / loss_budget.csv
    ├── comparison_accuracy.csv / comparison_efficiency.csv
    ├── profile.json / parent_profile_same_device.json
    ├── kitti_test_predictions.zip
    └── anchorflow_edge_fp32.onnx / experiment_summary.json
```

Checkpoint/log được backup về Drive **sau mỗi epoch**. Console cập nhật mỗi 50 batch; `train.log` trên Drive cập nhật cuối epoch. Chạy lại cùng notebook/config tự resume `last.pth` và RNG; không thêm 15 epoch ngoài tổng budget. Không overwrite run V6.

## Điều quan trọng khi so sánh

- Full V6 được evaluate trước train trên chính GPU/precision hiện tại.
- Zero head V7 khớp **reduced V6**, không khớp full V6: bỏ vector correction, giữ context/phase CNN đã học. 400-val CPU audit: 1,009274 → 1,009439 m.
- V7 là bundle **architecture + tail-loss**. Muốn tách gain architecture, chạy thêm `MODEL_NAME="v6_reduced"` với cùng recipe; không tự chạy control mặc định.
- `v7_no_transport` là control T=0. Mỗi model có run folder riêng và recipe hash để chặn so sánh sai parent/data/loss/budget.
- Best epoch có thể là −1 nếu 15 epoch không tốt hơn initialization; notebook báo rõ. Anonymous test không có public GT nên không có test RMSE.
- Profiler đo full V6/V7 cùng thiết bị. ONNX checker/parity không chứng minh TensorRT/Jetson runtime.

Đọc [ARCHITECTURE.md](ARCHITECTURE.md), [ANALYSIS_V6.md](ANALYSIS_V6.md) và [VERIFICATION.md](VERIFICATION.md) để xem nguyên lý, phản biện và phạm vi kiểm chứng.
