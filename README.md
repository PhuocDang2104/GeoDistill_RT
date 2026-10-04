# GeoDistill-RT / AnchorFlow

Sparse depth completion trên KITTI, hướng nghiên cứu hiện tại: **AnchorFlow V11 — Solver-Controlled Jet NODE**.
V11 học neural vector field trên geometric jet; numerical solver tích phân đến `T=1`, không học step size/GT-stop.

## Bắt đầu từ đây

| Nội dung | File |
|---|---|
| Nguyên lý V11, training, metric và insight mới nhất | [AnchorFlow_V11.md](docs/AnchorFlow_V11.md) |
| Folder V11 upload nguyên lên Drive | [AnchorFlow_v11_NODE](drive_upload/AnchorFlow_v11_NODE/README.md) |
| Train riêng V11 | [Notebook fresh40 + early stop](drive_upload/AnchorFlow_v11_NODE/Train_AnchorFlow_V11_NODE_TAR2000_Fresh40.ipynb) |
| Train V10.1/V11 đồng thời rồi benchmark tuần tự | [Notebook paired benchmark](notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb) · [Hướng dẫn](docs/AnchorFlow_V10_1_V11_Benchmark.md) |
| Kết quả paired run đã hoàn thành, kèm raw metric nhỏ | [Benchmark pair01](results/benchmarks/v10_1_v11_pair01/README.md) |
| Baseline/control V10.1 learned-h Euler2 | [V10.1](docs/AnchorFlow_V10_1.md) |

### Kết quả hiện tại

Validation nội bộ **400 ảnh / 25.424.992 GT pixels**; cùng 1.600 train, seed42, B4, 34 epoch/model.
Inference đo tuần tự: **A100-SXM4-80GB**, native BF16 CNN + FP32 geometry, PyTorch `2.11.0+cu130`, B1, 352×1216, real100 scenes.

| Model / solver | RMSE m ↓ | iRMSE km⁻¹ ↓ | Wall median / P95 ms ↓ |
|---|---:|---:|---:|
| V10.1 learned-h Euler2 | 0.993694 | 3.180862 | 35.12 / 38.80 |
| **V11 adaptive RK3(2)** | **0.986021** | 3.215515 | 74.15 / 76.47 |
| V11 static midpoint8, cùng checkpoint | 0.985972 | 3.215281 | 50.33 / 51.80 |

V11 là **baseline nghiên cứu chính**, không phải model đã thắng về mọi metric: RMSE tốt hơn 0,77%, iRMSE primary hơi kém, adaptive runtime chậm 2,11×. Static midpoint là ứng viên triển khai đã audit accuracy riêng, chưa benchmark TensorRT/edge device. Chưa đạt đồng thời RMSE <0.9 và iRMSE <3.2; đây không phải KITTI leaderboard.

## Data trên Drive

```text
MyDrive/
├── AnchorFlow_v11_NODE/           # folder code đầy đủ, không ZIP
├── AnchorFlow_v10_1_LiteMetric/   # thêm nếu chạy paired notebook
└── GeoLift_Data/
    ├── teacher_subset_2000/
    │   ├── selected_2000_ids.json
    │   ├── kitti_trainval_2000.tar
    │   ├── metric_coarse_train_2000.tar
    │   └── relative_teacher_2000_DA3MONO_LARGE.tar
    └── test_1000/kitti_test_1000.tar
```

RGB/sparse/GT/K, 1.600 train / 400 validation, sample ID và raw drive disjoint. Teacher cache dùng **train-only**; không teacher trong validation/inference. Anonymous test 1.000 không có GT công khai. Không cần DSINE, geometry-fused TAR hoặc online teacher weights cho V11.

## Code và kiểm chứng

V11 là package tự chứa trong `drive_upload/AnchorFlow_v11_NODE/`. Không dùng `src/train_student.py` của GeoLift cũ để train V11. Các model bundle là **snapshot khóa checksum**; không normalize line endings hoặc sửa model giữa chừng để resume. Tạo version/run mới khi đổi kiến trúc/config.

```bash
python -m pip install -r drive_upload/AnchorFlow_v11_NODE/requirements.txt
python -m unittest discover -s drive_upload/AnchorFlow_v11_NODE -v
python -m unittest discover -s tests -p test_anchorflow_pair_benchmark.py -v
```

Notebook chạy CPU contracts, data gate và real-data GPU smoke trước train. Script audit kết quả: `scripts/audit_anchorflow_pair_completed.py` (cần ZIP gốc local). ZIP/checkpoint/data không được commit; SHA + CSV/JSON nhỏ được giữ ở `results/benchmarks/`.

## Lịch sử và vận hành

- [Danh mục tài liệu](docs/README.md) và [danh mục model bundle](drive_upload/README.md) phân biệt baseline hiện tại với snapshot lịch sử V3–V10.
- GeoLift [S2 v2.1](docs/GeoLift-RT_v2.1_Baseline.md), [S3 Lite](docs/GeoLift-S3_Lite_Baseline.md) và notebook augmentation được giữ để tái lập, không phải entrypoint V11.
- Server/Docker hiện có: [operator runbook](docs/Server_Training_Runbook.md); adapter GeoLift/server không tự động hỗ trợ V11.
- Papers tại `docs/papers/` và các `v*_suggest.md` là tài liệu/ý tưởng nghiên cứu, không phải implementation contract.

Không xóa dataset, checkpoint hoặc kết quả lịch sử khi dọn Git. Xem [quy ước lưu results](results/README.md).
