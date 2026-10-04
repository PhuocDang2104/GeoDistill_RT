# AnchorFlow Research · một pipeline, hai model

**Default: V4 ContextWide, fine-tune thêm 15 epoch từ v3 best đã train của bạn.**

## Chạy nhanh

1. Giải nén ZIP, upload nguyên folder **AnchorFlow_Research** vào **MyDrive/AnchorFlow_Research**.
2. Mở **AnchorFlow_Research_TAR2000_15ep.ipynb**, chọn GPU, giữ MODEL_NAME="v4", Run all.
3. Không bỏ qua verify/data gate/smoke. Parent checkpoint **init_v3_best.pth** có sẵn (7,53 MB).
4. Khi Colab ngắt, chạy lại với cùng MODEL_NAME/config để resume last.pth. Tổng chỉ 15 epoch mới, không cộng thêm 15 mỗi lần chạy.

Input giữ nguyên:

```text
MyDrive/GeoLift_Data/
├── teacher_subset_2000/
│   ├── selected_2000_ids.json
│   ├── kitti_trainval_2000.tar
│   └── metric_coarse_train_2000.tar
└── test_1000/kitti_test_1000.tar
```

Không cần geometry TAR, DA, DSINE hay teacher weights. Nếu test TAR chưa có, helper tải KITTI official một lần và lưu vào Drive. Cache/train trên SSD Colab /content, không phải máy cá nhân. Nên có khoảng 25–30 GiB Colab SSD trống cho data/cache/test/output.

## Chỉ đổi model để benchmark

| MODEL_NAME | Model | Run folder | Mục đích |
|---|---|---|---|
| **v4** (mặc định) | 1.857.721 params, ContextWide | AnchorFlow_v4_ContextWide_FT15 | Candidate tăng capacity |
| v3 | 572.017 params, RefineKD | AnchorFlow_v3_Control_FT15 | Control train thêm cùng 15 epoch |

Cả hai dùng **cùng checkpoint v3**, data/loss/teacher, LR, augmentation, seed và evaluator.
Muốn control: đổi MODEL_NAME="v3" ở cell1 rồi Run all; tốn thêm một run 15 epoch riêng. Notebook **không tự chạy hai run**.

So v4 với v3 parent cũ phản ánh **kiến trúc + train thêm**. Chỉ khi có control v3 train tiếp cùng ngân sách mới phân biệt được architecture gain. Hai run có recipe_sha256 để chặn so sánh sai setting.

## Output

```text
MyDrive/GeoLift_RT_Runs/AnchorFlow_v4_ContextWide_FT15/
├── source_bundle/
├── data_contract.json
├── smoke_report.json
└── metric_kd/
    ├── initial_val_metrics.json / migration_report.json
    ├── best.pth / last.pth
    ├── train_log.csv / train_log.jsonl / train.log
    ├── run_manifest.json / resolved_config.json
    ├── val_metrics.json / best_val_metrics.json
    ├── comparison_accuracy.csv / comparison_efficiency.csv
    ├── profile.json / parent_profile_same_device.json
    ├── component_profile.csv / loss_budget.csv
    ├── training_diagnostics.png
    ├── kitti_test_predictions.zip / test_report.json
    ├── anchorflow_edge_fp32.onnx / export_report.json
    └── experiment_summary.json
```

Checkpoint/log sync Drive **sau mỗi epoch**. Best được so cả initial epoch -1; best=-1 nghĩa là fine-tune chưa cải thiện. Test 1.000 ảnh anonymous không public GT, ZIP không đồng nghĩa có test RMSE local.

## Cấu trúc gọn

```text
model.py          # factory: v3/v4 + strict checkpoint migration
model_v3.py       # baseline đã đạt 1,04537 m
model_v4.py       # chỉ phần kiến trúc mới
core.py           # primitive convolution / sparse / flow / phase dùng chung
data.py           # cùng loader, split, teacher gate
losses.py         # objective v3 giữ nguyên
loss_helpers.py   # reduction/KD helpers
metrics.py        # evaluator giữ nguyên
run.py            # chung train / resume / eval / test / profile / export
config.json       # shared recipe; notebook chỉ đổi model và run_name
```

Đọc [phân tích v3](ANALYSIS_V3.md), [kiến trúc v4](ARCHITECTURE.md), [kiểm chứng](VERIFICATION.md).

**Chưa có kết quả 15 epoch của v4; không bảo đảm đạt <0,7 m.** Encoder_pretrained=false chỉ vì đã nạp trọng số trained v3, không phải train random-init.
