# AnchorFlow V6 — Drive package

**Run chính:** V5 best → Ray-Connection + Phase Metric → fine-tune **15 epoch**. Cùng data/split/protocol V5, dùng metric teacher **chỉ khi train**. Mục tiêu RMSE <0,7 m **chưa được chứng minh**.

## Chạy trên Colab

1. Giải nén ZIP trên máy rồi upload **nguyên folder `AnchorFlow_v6_Connection`** vào `MyDrive/`. Không cần upload repo cũ.
2. Mở `AnchorFlow_v6_TAR2000_15ep.ipynb`, chọn GPU, sửa `PARENT_CHECKPOINT` nếu V5 best nằm nơi khác.
3. Kiểm input bên dưới rồi chạy lần lượt từ trên xuống. Không bỏ qua SHA, data gate hoặc smoke.
4. Mặc định batch4. Nếu OOM, dùng batch2 **và RUN_TAG mới trước khi train**. Không thay batch/loss/source khi resume.

```text
MyDrive/
├── AnchorFlow_v6_Connection/                 ← folder này
├── GeoLift_Data/
│   ├── teacher_subset_2000/
│   │   ├── selected_2000_ids.json
│   │   ├── kitti_trainval_2000.tar
│   │   └── metric_coarse_train_2000.tar
│   └── test_1000/kitti_test_1000.tar
└── GeoLift_RT_Runs/
    ├── AnchorFlow_v5_Piecewise_FT15/metric_kd/best.pth  ← parent mặc định
    └── AnchorFlow_v6_Connection_FromV5_FT15/            ← output mới
```

Không cần `geometry_fused`, DA, DSINE hoặc weights teacher. Nếu test TAR chưa có, `prepare` tải official KITTI test một lần và cache vào `GeoLift_Data/test_1000/`. Extract/cache ở SSD Colab `/content`, không phải ổ Windows. Drive chứa input và bản backup.

## Parent và resume

`best.pth` V5 thực tế **không có trong các file kết quả người dùng vừa cung cấp**; ZIP predictions không chứa trọng số. Vì vậy notebook đọc checkpoint đã có trên Drive, kiểm architecture/SHA/data và không tự fallback. Nó lấy epoch từ checkpoint, không đoán theo tên log.

`init_v3_best.pth` chỉ là fallback **chủ động** bằng `INIT_SOURCE="bundled_v3"`; run có hậu tố `FromV3`, không được coi là cùng initialization với V5. `encoder_pretrained=false` nghĩa là đã load toàn bộ parent pretrained/fine-tuned, không phải encoder random.

Checkpoint/log đồng bộ sau mỗi epoch vào:

```text
GeoLift_RT_Runs/AnchorFlow_v6_Connection_FromV5_FT15/metric_kd/
```

`best.pth`, `last.pth`, `train_log.csv`, `train_log.jsonl`, `train.log`, `val_metrics.json`, native-stage/range/boundary/error-tail metrics, profile median/P95/VRAM, ONNX và test ZIP đều nằm tại đây. `source_bundle/` ở folder cha giữ nguồn, actual parent và resolved config để tái lập.

## Đọc trước khi diễn giải kết quả

- [Phản biện ý tưởng và phân tích V5](ANALYSIS_V5.md).
- [Kiến trúc, công thức, loss và footprint](ARCHITECTURE.md).
- [Phạm vi kiểm chứng local](VERIFICATION.md).

Notebook mặc định chỉ train một run. Optional `v5_piecewise` là control cùng parent/batch/loss/15epoch; `v6_ambient` chỉ bỏ normal-alignment rotation, giữ nguyên capacity. So với log V5 cũ là **architecture + recipe + thêm training**, không đủ để kết luận gain riêng từ novelty. Anonymous test không có public GT/RMSE.
