# AnchorFlow V10.1 LiteMetric

**Bản V10 tinh gọn, không thay backbone hoặc dataset.** Fixed two-step feedback jet dynamics, **learned step size** + context-guided final phase lift + metric/inverse-depth objective.

Đúng2 bước cho mọi ảnh; mỗi bước học h từ state/context hiện tại, giới hạn1/6–1/3. Không stop controller, stop BCE hoặc host branching. Fresh h khởi tạo0.25, không bão hòa sát maximum. Data-dependent step head chỉ thêm33parameters.

Để ablate riêng step-size: giữ nguyên các cấu hình khác, đặt **learned_step_size=false**, h=1/3, và dùng run/tag mới. control40 còn thay context/loss nên không phải ablation riêng step-size.

Mục tiêu: **RMSE <0.9 m và iRMSE <3.2 km⁻¹ trên cùng checkpoint**. Đây là mục tiêu nghiên cứu, **chưa có kết quả GPU train cho V10.1**.

## Chạy trên Drive/Colab

1. Upload nguyên folder này vào **MyDrive/AnchorFlow_v10_1_LiteMetric**.
2. Mở [Train_V10_1_LiteMetric_TAR2000.ipynb](Train_V10_1_LiteMetric_TAR2000.ipynb), chọn GPU, chạy từ trên xuống.
3. Giữ mặc định **TRAIN_MODE='fresh40'**, **VARIANT='dual_teacher'** để train từ epoch0, max40, early stop.
4. Bị ngắt thì chạy lại cùng mode/tag/config. Notebook tự resume **last.pth**; đổi recipe phải đổi **RUN_TAG**.

Không cần upload repo, checkpoint hay teacher model weights cho fresh40. Notebook dùng thẳng dữ liệu đang có:

~~~text
MyDrive/GeoLift_Data/
├── teacher_subset_2000/
│   ├── selected_2000_ids.json
│   ├── kitti_trainval_2000.tar
│   ├── metric_coarse_train_2000.tar
│   └── relative_teacher_2000_DA3MONO_LARGE.tar
└── test_1000/
    └── kitti_test_1000.tar
~~~

Test TAR chưa có: workflow tải official KITTI như trước. Relative TAR đã sinh thì **không generate lại**. Không cần geometry_fused, DA raw, normal hay DSINE.

Data/cache được extract vào SSD **/content**; checkpoint/log đồng bộ Drive mỗi epoch. Không sử dụng SSD của PC. Chuẩn bị >25 GiB local Colab SSD trống.

## Các chế độ

| Mode | Khởi tạo | Budget / early stop | Dùng để |
|---|---|---|---|
| fresh40 — mặc định | ImageNet RGB only, student mới | max40; min20 / patience8 / Δ0.001 m | Đánh giá bản nâng cấp từ epoch0 |
| finetune20 | MODEL-only V10 best epoch23; optimizer mới | max20; min8 / patience5 / Δ0.001 m | Thử nhanh, cần đúng best.pth cũ |
| control40 | Fresh, fixed2, bỏ context bypass, hệ số loss V10 cũ | max40; min20 / patience8 | Control của bundle, không phải exact adaptive-V10 |

fine20 đọc mặc định **GeoLift_RT_Runs/AnchorFlow_v10_adaptive_Fresh40_ES_bf16/dual_teacher/best.pth**. Chỉnh **PARENT_RUN** nếu run cũ nằm nơi khác. Không dùng last.pth thay best. Cả model migration lẫn data/source/teacher SHA được kiểm tra.

Default dual_teacher giữ metric + relative như V10 để hạn chế thay teacher role. Có thể chọn **metric_kd** để bỏ relative loading/loss hoặc **gt_only** để bỏ KD; phải báo riêng, không coi đó là cùng experiment.

## Output

~~~text
MyDrive/GeoLift_RT_Runs/
└── AnchorFlow_v10_1_fresh40_bf16_learnedH2/dual_teacher/
    ├── best.pth                 # minimum RMSE
    ├── best_inverse.pth         # minimum iRMSE
    ├── best_joint.pth           # minimum max(RMSE/0.9, iRMSE/3.2)
    ├── last.pth                 # exact resume
    ├── initial_val_metrics.json
    ├── val_metrics.json
    ├── best_*_val_metrics.json
    ├── train_log.csv / train_log.jsonl / train.log
    ├── training_status.json / run_manifest.json
    ├── checkpoint_selection_metrics.csv / loss_budget.csv
    ├── historical_comparison.csv / learning_curves.png
    ├── profile.json / export_report.json
    ├── anchorflow_edge_fp32.onnx
    └── kitti_test_predictions.zip
~~~

Run name thay đổi theo mode, precision và tag. Native BF16 GPU dùng BF16; T4/V100 dùng **FP32**, không tự fallback FP16. Giữ heuristic cuDNN, **benchmark=false**, như run V10 đã hoàn tất.

Notebook dùng tag **_learnedH2** để không resume nhầm checkpoint của bản fixed-h trước. Nếu đã upload bản cũ, thay toàn bộ folder bằng bản ZIP mới; không trộn source/manifest của hai bản.

Test/profile/export mặc định dùng minimum-RMSE checkpoint. Không ghép RMSE và iRMSE của hai checkpoint để tuyên bố đạt cả hai.

## Tài liệu

- [ANALYSIS_V10.md](ANALYSIS_V10.md): metric, bottleneck, bằng chứng giữ/bỏ.
- [ARCHITECTURE.md](ARCHITECTURE.md): công thức, tensor, loss, training contract.
- [local_verification.json](local_verification.json): CPU tests/backward/ONNX proof; không phải GPU benchmark.

Bundle không chứa data, teacher weights hoặc student checkpoint. Source/config/immutable notebook snapshot được seal checksum; .ipynb UI được phép Colab đổi metadata/output.
