# Results storage

Git chứa **curated evidence nhỏ**, không dùng làm kho dataset hoặc checkpoint.

- [V10.1 / V11 pair01](benchmarks/v10_1_v11_pair01/README.md): benchmark GPU đã hoàn thành, CSV/JSON/plot và SHA nguồn/checkpoint.
- `anchorflow_pair_local_verification.json`: QA hai runner trên dữ liệu giả CPU; **không phải metric GPU/KITTI**.
- `GeoLift-S3-Lite_TAR2000_train_log_epoch0_19.csv`: historical tracked S3 log.

Raw ZIP/TAR, prediction ZIP/PNG, checkpoints, teacher/source audit clones và preview local vẫn được giữ trên máy/Drive nhưng `.gitignore` loại khỏi commit. Không file dữ liệu/checkpoint nào bị xóa trong lần dọn repo này. Khi publish run mới, tạo `results/benchmarks/<run>/` chỉ chứa metadata/metric cần tái kiểm chứng; ghi rõ seed/protocol/device/budget và nguồn evidence.
