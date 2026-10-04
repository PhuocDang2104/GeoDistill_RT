# AnchorFlow V8 · Feedback Jet Dynamics · Fresh30 + Early Stop

Upload **folder này**, không cần toàn repo/data/weights. Kiến trúc: [`ARCHITECTURE.md`](ARCHITECTURE.md). Kết luận V7: [`ANALYSIS_V7.md`](ANALYSIS_V7.md).

## Chạy trên Colab

1. Upload `AnchorFlow_v8_Dynamics` vào `MyDrive/AnchorFlow_v8_Dynamics`.
2. Mở [`AnchorFlow_v8_Dynamics_TAR2000_Fresh30_EarlyStop.ipynb`](AnchorFlow_v8_Dynamics_TAR2000_Fresh30_EarlyStop.ipynb), chọn GPU; kiểm tra ba đường dẫn ở cell đầu.
3. Run all: bundle tests → data gates → preview → AMP smoke → train → metrics → test ZIP → profile → ONNX. Không bỏ qua gates.

Notebook `.ipynb` đang mở có thể bị Colab sửa metadata/output; checksum runtime bỏ qua riêng artifact này. `notebook_contract.json` là bản canonical immutable, được verify/copy và dùng cho notebook tests; code/config vẫn kiểm tra SHA256 nghiêm ngặt. Nếu tests lỗi, cell in toàn bộ traceback và lưu `/content/anchorflow_v8_code/unit_test_output.txt`, không chỉ hiện `CalledProcessError`.

**Cập nhật từ bundle cũ bị lỗi Colab:** upload lại nguyên folder đồng bộ gồm cả `notebook_contract.json` và `bundle_manifest.json`, mở notebook mới, Run all. Nếu đã tạo frozen `source_bundle` từ bản cũ, đặt `RUN_TAG="_ColabFix"` để không ghi đè source run đó. Không cần upload lại data TAR.

```text
MyDrive/
├── AnchorFlow_v8_Dynamics/                  ← upload folder code này
├── GeoLift_Data/
│   ├── teacher_subset_2000/
│   │   ├── selected_2000_ids.json
│   │   ├── kitti_trainval_2000.tar          ← RGB + sparse + GT + K + splits
│   │   └── metric_coarse_train_2000.tar     ← D_cm/C_cm; chỉ train loader đọc
│   └── test_1000/kitti_test_1000.tar        ← reuse test hiện có
└── GeoLift_RT_Runs/
    └── AnchorFlow_v8_Dynamics_Fresh30_ES/
        ├── source_bundle/                 ← frozen source + resolved config
        ├── data_contract.json
        ├── data_preview.png
        ├── smoke_report.json
        └── metric_kd/
            ├── best.pth / last.pth
            ├── train_log.csv / train_log.jsonl / train.log
            ├── initial_val_metrics.json / best_val_metrics.json / val_metrics.json
            ├── resolved_config.json / run_manifest.json
            ├── training_status.json        ← actual epochs, stop reason/counter
            ├── comparison_accuracy.csv / loss_budget.csv / dynamics_evolution.csv
            ├── dynamics_learning.png / profile.json
            ├── kitti_test_predictions.zip / test_report.json
            └── anchorflow_edge_fp32.onnx / export_report.json
```

Chỉ cần hai TAR train/teacher + manifest. **Không cần geometry_fused, DA raw, DSINE, teacher model weights hoặc checkpoint student cũ.** Nếu thiếu test TAR, notebook tải official KITTI và lưu test TAR vào Drive một lần. Anonymous test không có GT công khai; ZIP chưa phải một bảng RMSE leaderboard.

## Fresh initialization và resume

- Default **tối đa 30 epoch 0–29**, 1.600 train / 400 validation / 1.000 test. ImageNet pretrained RGB; phần student còn lại mới khởi tạo. Encoder pretrained vẫn được fine-tune, chỉ BN statistics freeze.
- **Early stop**: global validation RMSE, patience 7, meaningful improvement >0,001 m, không dừng trước khi hoàn tất 15 epoch. Những cải thiện nhỏ được cộng dồn so với meaningful best. Best checkpoint vẫn lưu mọi raw RMSE thấp hơn; min_delta chỉ quyết định counter dừng, không loại bỏ checkpoint tốt nhất.
- Counter/monitor best/stop flag nằm trong checkpoint; resume không reset patience. Run đã early-stop không tự train tiếp; evaluate/test/profile/export vẫn chạy bằng `best.pth`. LR và KD schedules tính theo horizon tối đa30, không nén lại khi dừng sớm.
- Notebook/trainer không tải best V7/V6. Gói không chứa `.pth`. Thiếu ImageNet download thì dừng rõ ràng, không silent fallback random encoder. HF_TOKEN là optional nếu cần rate limit cao hơn.
- Mỗi epoch đồng bộ log/checkpoint về `metric_kd` ở Drive. GPU process log cũng hiện trực tiếp trong cell train. Backup theo epoch, không mỗi batch; nếu đứt giữa epoch sẽ chạy lại epoch chưa lưu.
- Chạy lại notebook với cùng run/config sẽ resume `last.pth` V8; khôi phục optimizer, scaler, LR/global step và RNG. Muốn fresh lần nữa hoặc đổi batch/epochs/source, đặt `RUN_TAG` mới, không sửa/ghi đè run đang resume.
- Train/eval lấy dữ liệu đã extract vào **SSD Colab `/content`**, không đọc PNG/NPZ nhỏ qua Drive mỗi iteration. Drive chứa input TAR và backup; không dùng dung lượng ổ PC cá nhân. Cần khoảng 12 GiB SSD Colab trống tối thiểu, khuyến nghị ≥20 GiB để có headroom/download.
- T4 OOM: đặt `BATCH_SIZE=2`, `RUN_TAG="_B2"` trước khi bắt đầu. Thay effective batch có thể đổi kết quả; ghi rõ khi so sánh.

## V8 thực sự dynamic ở đâu?

```text
D0 → initial 6D jet j0
       ↓ F_theta(j0, sensor residual0, Z)
      j1
       ↓ SAME weights, NEW F_theta(j1, sensor residual1, Z)
      j2
       ↓ SAME weights, NEW F_theta(j2, sensor residual2, Z)
      j3 → D4 + continuous phase query → D2 → D1 → sensor reliability fusion
```

T=3 projected Euler; learned forcing + state-dependent conductance + analytic quadratic transport.
Không adaptive ODE solver, không optical flow, không guarantee “đến hội tụ”. Old QuarterFlow và các quarter-resolution correction branches được **thay**, không chồng thêm.

Optional control: đổi `MODEL_NAME="v8_frozen_feedback"` trong cell config; notebook tạo run riêng, recipe/compute/parameters giống nhưng field đọc j0 ở cả ba bước. Không tự chạy thêm experiment. Muốn so sánh teacher, CLI hỗ trợ `--variant gt_only` ở output riêng; notebook default chỉ `metric_kd`.

## Giới hạn cần đọc trước khi kết luận

V7 best 1,0061 m; target V8 <0,8 m là mục tiêu **chưa train/chưa chứng minh**. Fresh30+ES không là continuation từ checkpoint accumulated75 epoch. Local verification chỉ CPU/BF16, ảnh KITTI thật và ONNX parity, không xác nhận CUDA FP16/TensorRT/latency edge hay khả năng đạt target. Khi so control, báo cả số epoch thực tế sau early stop.

Khi đọc log: global + far bins + boundary + iRMSE + tail SSE, D4_step1/2/3 native RMSE, forcing/transport/state change. Metric teacher chỉ train, exclude GT/original sparse; GT không bị lọc khỏi evaluator vì conflict/outlier.
