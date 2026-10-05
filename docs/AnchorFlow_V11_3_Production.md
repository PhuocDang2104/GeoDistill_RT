# Chạy V11.3 full KITTI trên GPU member

Package độc lập: [production/AnchorFlow_V11_3_FullKITTI](../production/AnchorFlow_V11_3_FullKITTI/README.md). ZIP được tạo bằng `python scripts/package_anchorflow_v11_3_production.py`.

Model V11.3 giữ nguyên; thêm production adapter, full dataset loader, offline DMD3C++ + DA3MONO-LARGE generation, coverage gate và batch-level recovery dựa trên runtime S3. Entry point mới là `pipeline.py`, **không phải** `src.train_student` hay DockerHub S3 image cũ.

User đã chốt fresh student + ImageNet RGB encoder. Flow tự tải dataset (sau khi xác nhận KITTI terms), generate toàn bộ train-only teacher, rồi fresh train tối đa40epoch và evaluate/test. Default85.898train/6.852val/1.000anonymous test; không gộp val để gọi90ktrain. Teacher float32 NPY ~548GiB trước cộng KITTI/weights; khuyên persistent SSD≥1TB.

Member nhận nguyên folder và làm theo README: cấu hình license/SSD → `bash run_full_flow.sh` → xem `docker compose logs -f trainer`. Không phụ thuộc Colab/Drive. CUDA extension của teacher cần build trong đúng image/SM; BF16 native GPU bắt buộc.

Member có thể pull repo và chạy thẳng, không sửa tracked config:

```bash
git pull --ff-only origin main
bash production/AnchorFlow_V11_3_FullKITTI/run_full_flow.sh \
  --storage-root /mnt/ssd/geolift --accept-kitti-license
```

Thay `/mnt/ssd/geolift` bằng SSD persistent của member. Flag license là xác nhận sau khi đọc KITTI terms. Launcher tự detect GPU0 SM, lưu runtime config ngoài source và in lệnh log/stop. Không cần notebook hoặc host pip dependencies. Docker/NVIDIA Container Toolkit phải được cài trước.

Output: checkpoints last + four best policies; CSV/JSONL loss và global/range/boundary/inverse metrics; ZIP1.000testdepth;10RGB/depth previewPNG. Test không publicGT, không có testRMSE. Full validation không phải400ảnh historical protocol.

Đây là implementation đã kiểm tra contract CPU, **không phải GPU-qualified full90k run**. Không claim teacher hoặc student đã chạy full data tại máy local, không bảo đảm đạt target. GPU real-data forward/backward gate sẽ chạy ở máy member trước train. Không sửa baseline research đã train; production full-data là protocol mới.
