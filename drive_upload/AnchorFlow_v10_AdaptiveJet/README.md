# AnchorFlow V10 — Fresh40 / Dual teacher / Early stop

**Upload folder này vào MyDrive/AnchorFlow_v10_AdaptiveJet.** Mở
[02_Train_V10_Fresh40_EarlyStop.ipynb](02_Train_V10_Fresh40_EarlyStop.ipynb),
chọn GPU và chạy từ trên xuống. Không cần upload toàn repo hoặc checkpoint cũ.

## Dữ liệu dùng lại

~~~text
MyDrive/
├── AnchorFlow_v10_AdaptiveJet/            ← folder code + notebook này
├── GeoLift_Data/
│   ├── teacher_subset_2000/
│   │   ├── selected_2000_ids.json
│   │   ├── kitti_trainval_2000.tar        ← RGB / sparse / GT / K / split
│   │   ├── metric_coarse_train_2000.tar   ← D_cm / C_cm
│   │   └── relative_teacher_2000_DA3MONO_LARGE.tar ← R_T / C_T, dùng lại V9
│   └── test_1000/
│       └── kitti_test_1000.tar           ← dùng lại; nếu thiếu tải official KITTI
└── GeoLift_RT_Runs/
    └── AnchorFlow_v10_adaptive_Fresh40_ES_bf16/
        ├── source_bundle/
        ├── data_contract.json
        └── dual_teacher/
            ├── initial_val_metrics.json
            ├── best.pth / last.pth
            ├── train_log.csv / train_log.jsonl / train.log
            ├── training_status.json / resolved_config.json / run_manifest.json
            ├── val_metrics.json / policy_audit.json
            ├── profile.json / historical_comparison.csv
            ├── export_report.json / static4_export_report.json
            ├── anchorflow_edge_fp32.onnx / anchorflow_static4_fp32.onnx
            └── kitti_test_predictions.zip
~~~

Tên GeoLift_RT_Runs phân biệt chữ hoa/thường. Run FP32 có hậu tố fp32 thay bf16.
Data được extract/cache vào SSD **Colab /content**, cần >25 GiB trống; không dùng SSD máy cá nhân.
Checkpoint/log/result được backup về Drive sau mỗi epoch.

## Hai notebook

| Notebook | Khi dùng |
|---|---|
| 01_Optional_Relative_Teacher_Resume | **Bỏ qua nếu đã có relative TAR từ V9.** Cùng pinned teacher recipe; resume từng ảnh trên Drive. Cần native BF16 GPU. |
| 02_Train_V10_Fresh40_EarlyStop | Default train fresh epoch0; không load student V8/V9/V9.1. Train1600 / val400 / anonymous test1000. |

Fresh không có nghĩa RGB random: MobileNetV4 dùng ImageNet pretrained, tất cả các nhánh
depth/controller/readout khởi tạo mới. Hai teacher chỉ là training targets, không có trong forward.

## Thiết lập mặc định

| Thuộc tính | Giá trị |
|---|---|
| POLICY | adaptive: learned h, exit2/3/4 |
| Epoch | Tối đa40, index0…39 |
| Early stop | Min20 epoch, patience8, min_delta0.001 m |
| Batch / accumulation | 4 / 1 |
| LR | Decoder3e-4, encoder1.5e-4, new head/controller3e-4 |
| Optimizer | AdamW; warmup1 → cosine, minLR5% |
| Precision | Native BF16 nếu SM≥8 + native-support check; T4/V100 **explicit FP32** |
| Resume | Tự đọc last.pth cùng source/config/data, khôi phục optimizer/RNG/global step/early-stop |
| Checkpoint evaluate/test | best.pth, không phải last.pth |

Không tự fallback FP16, skip batch hoặc che NaN. Đổi GPU precision, source hoặc recipe phải
đổi RUN_TAG để tạo run mới. Notebook UI bị Colab sửa không gây lỗi checksum; code/config và
immutable notebook contracts vẫn được kiểm tra.

## So sánh công bằng

Trong cell cấu hình, đổi POLICY và chạy vào folder riêng:

| Policy | Vai trò | Field calls |
|---|---|---|
| fixed3 | B0: baseline kỹ thuật V9.1, exact old objective | 3, h=1/3 |
| fixed4 | A1: thêm một bước nhưng không học h/stop | 4, h=1/3 |
| learned4 | A2: học h nhưng không dừng sớm | 4 |
| adaptive | A3: V10 chính | Train/val4; deploy batch1 thực2–4 |

Default chỉ train **một run adaptive**, không tự chạy4 experiment.
Baseline config được giữ riêng ở baseline_fixed3_config.json; trên Colab chọn POLICY='fixed3'
để resolve đúng Drive paths/precision và chạy baseline fresh40, không dùng raw Colab config trên máy local.
B0 có cùng fresh40/data/seed/loss gốc, nhưng giữ auxiliary bước1/2 cũ; A1–A3 dùng
auxiliary4 trạng thái chuẩn hóa cùng tổng trọng số. B0 giữ482 parameter controller không hoạt động
để cùng code/initialization; không tốn controller MAC, không nhận gradient.
Do đó B0↔A3 là bundle integration/training-policy ablation, không phải chỉ một scalar h.

policy_audit sau train là so exit2/3/4 của **cùng checkpoint**, không thay thế4 run độc lập.
Historical V8/V9/V9.1 khác initialization, budget và runtime backend; không dùng để claim
causal gain của V10.

## Kết quả và giới hạn

Đọc [BASELINE_COMPARISON.md](BASELINE_COMPARISON.md) và [ARCHITECTURE.md](ARCHITECTURE.md).
V9.1 chỉ nhỉnh RMSE V8 rất nhỏ; không có bằng chứng bước nhảy lớn.
Mục tiêu RMSE<0.9 m là hypothesis; chưa có V10 train trên GPU.
Anonymous test không có public GT, không được gán validation RMSE cho từng ảnh test.

ONNX masked-adaptive và static4 đều chạy4 calls. Chỉ profile Python adaptive_batch1
mới đo real early exit, gồm host synchronization. CPU ORT parity không phải chứng nhận
TensorRT hoặc tốc độ edge. Kiểm chứng local được ghi trong local_verification.json.
