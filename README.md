# GeoDistill-RT / GeoLift

Repo nghiên cứu sparse depth completion thời gian thực trên KITTI. Hai baseline được duy trì:

| Baseline | Mục đích | Teacher khi train | Tài liệu chuẩn |
|---|---|---|---|
| GeoLift-S2 v2.1 | Mốc teacher-distillation để so sánh | `D_cm/C_cm` và `R_G/C_G` | [V2.1 baseline](docs/GeoLift-RT_v2.1_Baseline.md) |
| GeoLift-S3 Lite | Hướng phát triển chính, gọn và teacher-free | Không | [S3 Lite baseline](docs/GeoLift-S3_Lite_Baseline.md) |

Ở inference, cả hai model chỉ dùng RGB, sparse depth, mask và camera intrinsics; không chạy Metric3D, Depth Anything, DSINE hoặc DMD3C.

## Benchmark cố định

- Input: `352×1216`, depth theo mét, miền hợp lệ `(0.001, 120)`.
- TAR2000: 1.600 train và 400 validation, không trùng sample ID hoặc raw drive.
- KITTI anonymous test: 1.000 ảnh, không có GT công khai.
- Mọi run so sánh phải dùng cùng `selected_2000_ids.json` và protocol lock.

Drive input:

```text
MyDrive/GeoLift_Data/teacher_subset_2000/
├── selected_2000_ids.json
├── kitti_trainval_2000.tar
├── metric_coarse_train_2000.tar      # S2 only
└── geometry_fused_train_2000.tar     # S2 only
```

Notebook chuẩn:

- S2: `notebooks/GeoLift_RT_v2_1_TAR2000_Train1600_Val400_Test_OneRun.ipynb`
- S3: `notebooks/GeoLift_S3_Lite_TAR2000_Train1600_Val400_Test_OneRun.ipynb`
- Decoder-V2 quick ablation: `notebooks/GeoLift_Decoder_V2_TAR2000_Finetune_OneRun.ipynb`

## Source of truth

```text
src/model_geolift_s2.py              # S2 forward graph
src/model_geolift_s3.py              # S3 forward graph
src/model_geolift_decoder_v2.py      # warm-start decoder ablation
src/losses.py                        # geolift_loss / geolift_s3_loss
src/train_student.py                 # train, validation, checkpoint, log
configs/geolift_s2_v2_1_balanced_ablation.yaml
configs/geolift_s3_lite_tar2000.yaml
configs/geolift_decoder_v2_finetune_tar2000.yaml
scripts/run_standard_experiment.py   # protocol-locked train/infer/profile
```

Chạy test contract trước khi benchmark:

```bash
python -m unittest discover -s tests -v
```

Mỗi run chuẩn lưu `experiment_record.json`, `checkpoints/`, `logs/train_log.csv`, metric validation, profile FP16 và 1.000 PNG test trong thư mục run trên Drive.

Các paper tham khảo được giữ nguyên tại `docs/papers/`; chúng không phải đặc tả implementation.
