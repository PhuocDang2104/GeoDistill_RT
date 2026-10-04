# AnchorFlow V11 — upload và train ngay

V11 là **Jet NODE**: mạng học vector field, numerical solver chọn step-size, luôn tích phân đến T=1.
Không learned h/GT-stop/projected Euler. Đọc [ARCHITECTURE.md](ARCHITECTURE.md) và
[ANALYSIS_AND_COMPARISON.md](ANALYSIS_AND_COMPARISON.md) trước khi diễn giải gain.

## 1. Upload

Giải nén ZIP rồi upload **nguyên folder AnchorFlow_v11_NODE** vào MyDrive.
Không copy repo/weights/data vào folder code; bundle hoàn toàn độc lập.

~~~text
MyDrive/
├── AnchorFlow_v11_NODE/
│   ├── Train_AnchorFlow_V11_NODE_TAR2000_Fresh40.ipynb
│   ├── config.json
│   ├── bundle_manifest.json
│   └── ... source, tests, docs
├── GeoLift_Data/
│   ├── teacher_subset_2000/
│   │   ├── selected_2000_ids.json
│   │   ├── kitti_trainval_2000.tar
│   │   ├── metric_coarse_train_2000.tar
│   │   └── relative_teacher_2000_DA3MONO_LARGE.tar
│   └── test_1000/kitti_test_1000.tar
└── GeoLift_RT_Runs/
~~~

Test TAR được dùng nếu có; nếu chưa có, flow cũ tải official KITTI anonymous test.
Không cần geometry_fused, DSINE, DA raw hoặc teacher model weights.
**Không chạy lại notebook generate relative teacher** nếu đã có audited DA3 TAR đúng subset.

## 2. Chạy

1. Mở notebook trên Colab, chọn GPU.
2. Kiểm tra BUNDLE/DRIVE_DATA/DRIVE_RUNS ở cell đầu; chạy lần lượt từ trên xuống.
3. Giữ default dual_teacher, batch4, fresh40; không bỏ qua data gate hoặc smoke.
4. Notebook tự chọn native BF16 main CNN nếu GPU hỗ trợ; T4/V100 dùng FP32.
   Shared ODE RHS/state/solver/loss luôn FP32; không dùng FP16.
5. Kết thúc có full400 validation, historical comparison, solver audit, test1000 ZIP,
   profile adaptive/fixed và optional ONNX fixed-midpoint parity.

Cần >25GiB **SSD local Colab /content**, không phải SSD máy Windows.
Adaptive solver có nhiều calls hơn V10; A100/L4/Blackwell có lợi hơn T4 cho tốc độ train.
Không có số GPU latency V11 trước khi bạn chạy profiler.

## 3. Train / resume / log

Fresh student epoch0, ImageNet pretrained RGB encoder; **không warm-start student**.
Tối đa40epochs (0…39); early stop từ khi đủ20epochs, patience8, meaningful gain0.001m.
Ngắt Colab: chạy lại notebook cùng config/VARIANT/RUN_TAG, tự resume last.pth + optimizer/RNG.
Đổi batch/precision/solver/source/loss phải tạo RUN_TAG mới; không bỏ checksum/protocol guard.

Default output khi GPU native BF16:

~~~text
MyDrive/GeoLift_RT_Runs/AnchorFlow_v11_NODE_Fresh40_ES_bf16_bosh3_T1/
├── source_bundle/                    # immutable code + resolved recipe
├── data_contract.json
├── data_preview.png
└── dual_teacher/
    ├── train_log.csv / train_log.jsonl
    ├── train_console.log / *_console.log
    ├── last.pth / best.pth / best_inverse.pth / best_joint.pth
    ├── training_status.json / run_manifest.json / resolved_config.json
    ├── initial_val_metrics.json / val_metrics.json
    ├── best*_val_metrics.json / checkpoint_selection_metrics.csv
    ├── historical_comparison.csv / learning_curves.png / loss_budget.csv
    ├── solver_comparison.json / val_metrics_*solver*.json
    ├── kitti_test_predictions.zip
    ├── profile.json / profile_fixed_midpoint8.json
    └── anchorflow_edge_fp32.onnx / export_report.json
~~~

FP32 run đổi suffix bf16 thành fp32. Logs/checkpoints sync mỗi epoch;
console streaming hiển thị ngay ở cell và copy lên Drive khi subprocess kết thúc/fail.

## 4. Không hiểu nhầm “2 observation times”

flow_steps=2 chỉ là t=.5 và t=1 cho trajectory auxiliary.
Không phải 2 Euler steps/NFE. Adaptive RK3(2) thuận lợi thường **13 RHS calls**; state khó có thể hơn.
193NFE là guard chống runaway, không buộc solver accept khi chưa đạt tolerance.
Hết budget sẽ báo lỗi thật; không tự skip sample hay đổi solver.

## 5. Edge export khác research solver

Main train/val/test dùng adaptive RK3(2). ONNX là **four-step midpoint, 8NFE**, cùng weights/T/chart.
Đây là solver approximation khác, không adaptive ONNX. Notebook đánh giá lại đầy đủ400val
trên cùng checkpoint trước khi quyết định dùng static graph.
Chỉ báo edge gain nếu accuracy/runtime của fixed solver thật sự chấp nhận được.

## 6. Đã kiểm tra gì?

Đọc local_verification.json: CPU contracts, analytic ODE accuracy/gradient,
real KITTI full-resolution backward, actual runner save/resume và static ONNX parity.
Local backward dùng real RGB/sparse/GT/K nhưng **synthetic teacher targets** vì cache Drive không
được mount tại đây; Colab smoke sẽ kiểm tra **teacher thật** trước train.
Không có V11 trained metric/GPU performance được đo ở máy này.
