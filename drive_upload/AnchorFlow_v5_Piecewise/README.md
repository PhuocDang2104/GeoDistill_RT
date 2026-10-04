# AnchorFlow V5 · Piecewise Surface · 15 epochs

Gói độc lập để upload Drive: **V3 trained parent + khối hiệu chỉnh bề mặt tại D4**, metric teacher khi train, không có ContextWide V4. V4 và dữ liệu cũ được giữ nguyên ngoài folder này.

## Chạy trên Colab

1. Upload **nguyên folder `AnchorFlow_v5_Piecewise`** vào `MyDrive/`. Nếu dùng ZIP, giải nén trước; notebook không chạy trực tiếp bên trong ZIP.
2. Mở `AnchorFlow_v5_TAR2000_15ep.ipynb`, chọn GPU, Run all từ trên xuống.
3. Mặc định `MODEL_NAME="v5_piecewise"`, **15 epoch thêm**, batch2. Không tự chạy nhiều experiment.
4. Xem `train_log.csv`, `train_log.jsonl`, `best_val_metrics.json` và checkpoint trong đường dẫn output dưới đây. Đồng bộ sau mỗi epoch.

```text
MyDrive/
├── AnchorFlow_v5_Piecewise/                  # upload gói này
├── GeoLift_Data/
│   ├── teacher_subset_2000/
│   │   ├── selected_2000_ids.json
│   │   ├── kitti_trainval_2000.tar           # RGB + sparse + GT + K + split
│   │   └── metric_coarse_train_2000.tar      # D_cm, C_cm
│   └── test_1000/
│       └── kitti_test_1000.tar              # nếu chưa có: prepare tải official test
└── GeoLift_RT_Runs/
    └── AnchorFlow_v5_Piecewise_FT15/
        ├── source_bundle/
        ├── data_contract.json / smoke_report.json
        └── metric_kd/
            ├── best.pth / last.pth
            ├── train_log.csv / train_log.jsonl / train.log
            ├── initial_val_metrics.json / best_val_metrics.json / val_metrics.json
            ├── gt_boundary_metrics.csv / gt_boundary_curve.png
            ├── comparison_accuracy.csv / comparison_efficiency.csv
            ├── profile.json / parent_profile_same_device.json
            ├── kitti_test_predictions.zip / test_report.json
            └── experiment_summary.json / export_report.json
```

`GeoLift_RT_Runs` viết hoa **Runs** theo config này. Có thể sửa `DRIVE_RUNS` ở cell1 nếu thư mục cá nhân dùng tên khác.

Data được copy/extract/cache vào **SSD Colab `/content`**; Drive chứa input lâu dài và output backup. Vẫn cần dung lượng SSD Colab; không sử dụng ổ cứng máy Windows. Gói không chứa dataset, teacher model weights, geometry TAR, DA hay DSINE.

## Run contract

| Mục | Giá trị |
|---|---|
| Parent | V3 best epoch14, đã học tổng30epoch; có `init_v3_best.pth` trong gói |
| Train thêm | 15 epoch, index0–14; cumulative30–44 |
| Data | 1.600 train / 400 val, cùng ID và raw-drive split cũ |
| Test | 1.000 anonymous KITTI; **không public GT, không tính test RMSE** |
| Metric teacher | Inspect2.000 record, chỉ cache/load1.600 train |
| Inference input | RGB, sparse, mask, intrinsics K; không teacher/GT |
| Initialization | Đủ697 parent tensors; nhánh mới no-op, không train từ random init |
| Resume | Khôi phục optimizer/scaler/RNG; source/config khác thì dừng |

`encoder_pretrained=false` chỉ tránh tải ImageNet lần nữa: encoder nhận trọng số đã pretrained và fine-tune từ parent. Fresh AdamW khi chuyển V3→V5 là có chủ ý; resume V5 đang chạy khôi phục đầy đủ.

## Control tùy chọn

| MODEL_NAME | Run name | Khác biệt |
|---|---|---|
| `v5_piecewise` | `AnchorFlow_v5_Piecewise_FT15` | Transport + directional barriers + barrier BCE |
| `v5_transport` | `AnchorFlow_v5_Transport_FT15` | Cùng module, không áp barrier và không barrier BCE |
| `v3` | `AnchorFlow_v3_BoundaryControl_FT15` | Không module mới; cùng fixed-GT boundary loss |

Tất cả cùng parent, data, LR, budget15epoch và boundary weight. **Full V5 có thêm barrier BCE**, nên không gọi đây là comparison chỉ kiến trúc. Đặt `BARRIER_WEIGHT=0` và dùng `RUN_TAG` mới để kiểm architecture-only; đặt cả hai weight mới bằng0 cho control loss V3 nguyên bản. Mỗi control là một run riêng do bạn chọn, không tự chạy.

So V5 với checkpoint V3 cũ/V4 lịch sử là so experiment bundle, còn lẫn ảnh hưởng train thêm và supervision. `recipe_sha256` kiểm data/hyperparameters/source chung, không chứng minh các term active hoàn toàn giống nhau.

## Đọc trước khi kết luận

- [ANALYSIS_V4.md](ANALYSIS_V4.md): phản biện insight, số liệu15epoch, hướng nghiên cứu và rủi ro.
- [ARCHITECTURE.md](ARCHITECTURE.md): công thức khớp code, loss, compute và novelty hypothesis.
- [VERIFICATION.md](VERIFICATION.md): phạm vi test local, chưa có GPU accuracy/latency V5.

Không bỏ qua `prepare`, tests hoặc smoke. Không đổi clamp/mask/split để làm RMSE thấp hơn. Mục tiêu0,6–0,7m chưa được chứng minh bằng gói này.
