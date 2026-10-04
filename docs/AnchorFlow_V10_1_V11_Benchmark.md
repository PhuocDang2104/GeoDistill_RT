# V10.1 / V11 — one-notebook benchmark

[Mở notebook](../notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb). Notebook tự chứa script điều phối; không cần upload repo hoặc folder helper.

## 1. Upload và chạy

Upload **hai folder đầy đủ đã giải nén**, giữ nguyên tên và file bên trong. Đặt notebook ở vị trí nào trong Drive cũng được:

```text
MyDrive/
├── AnchorFlow_v10_1_LiteMetric/
├── AnchorFlow_v11_NODE/
├── AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb
└── GeoLift_Data/
    ├── teacher_subset_2000/
    │   ├── selected_2000_ids.json
    │   ├── kitti_trainval_2000.tar
    │   ├── metric_coarse_train_2000.tar
    │   └── relative_teacher_2000_DA3MONO_LARGE.tar
    └── test_1000/kitti_test_1000.tar
```

Nếu test TAR chưa có, `prepare` tải official KITTI anonymous test. Hai teacher TAR đã có thì không cần notebook generate teacher. Không cần geometry fused, DSINE hoặc weights teacher.

Mở Colab, chọn GPU, kiểm tra đường dẫn trong cell cấu hình rồi chạy từ trên xuống. Cần hơn 25 GiB **SSD Colab `/content`**, không dùng SSD máy cá nhân. Dữ liệu được extract/audit một lần cho mỗi cặp experiment, sau đó cả hai model đọc chung cache local. Local cache/log/checkpoint cũng được tách theo precision và `BENCH_TAG`, tránh CSV cũ lẫn vào run fresh mới.

| Thiết lập | Mặc định |
|---|---|
| Train / validation / anonymous test | 1.600 / 400 / 1.000 |
| Khởi tạo student | Fresh epoch 0; RGB ImageNet pretrained; không checkpoint depth cũ |
| Teacher | Metric + relative, cache chỉ 1.600 ảnh train; không teacher trong forward/evaluation |
| Budget | Tối đa 40 epoch/model |
| Early stop | Ít nhất 20 epoch; patience 8; min improvement 0.001 m |
| Batch / accumulation / seed | 4 / 1 / 42, giống nhau |
| Precision | Native BF16 trên tất cả GPU dùng; nếu không hỗ trợ thì FP32; không FP16 |
| Train scheduling | Hai subprocess song song; mỗi GPU một model nếu có hai GPU, nếu chỉ một GPU thì chia sẻ |
| Final evaluate / profile | Tuần tự, cùng một `PROFILE_GPU_INDEX` |

V10.1 giữ **learned step size, Euler 2 bước**. V11 giữ **NODE với adaptive RK3(2), numerical error control, fixed terminal time 1**. Không chỉnh source/config trong sealed model bundle.

## 2. VRAM và resume

Một GPU chạy hai model không bảo đảm nhanh hơn; cả VRAM và compute đều được chia sẻ. Notebook không tự bật MPS, thay CUDA/Torch, giảm batch hoặc bỏ batch lỗi.

`SHARED_GPU_MIN_FREE_GIB=20` chỉ là **guard bảo thủ chưa được đo**, không phải minimum VRAM đã chứng minh. Nếu ít VRAM, dùng `PARALLEL_TRAIN=False`. Nếu muốn thử batch nhỏ, **trước khi bắt đầu experiment mới** đặt `BATCH_SIZE=2`, `ACCUMULATION=2` cho cả hai và tự chọn guard phù hợp; vẫn có thể OOM. Cùng effective batch không làm hai microbatch recipe tương đương hoàn toàn, vì vậy phải ghi rõ thay đổi.

Colab ngắt: chạy lại cùng notebook, `BENCH_TAG`, precision, source và config. Mỗi runner resume `last.pth` riêng. Model đã early-stop sẽ không bị train lại. Thay recipe cần `BENCH_TAG` mới. Nếu một child lỗi, peer tiếp tục hoàn thành; xem console của child lỗi, sửa nguyên nhân rồi resume. Có thể chuyển `PARALLEL_TRAIN=False` khi resume mà không đổi model recipe. Không mở hai phiên cùng ghi vào một output root.

## 3. Output trên Drive

```text
GeoLift_RT_Runs/Compare_V10_1_V11_Fresh40_{bf16|fp32}_{BENCH_TAG}/
├── source_bundle/{V10_1,V11}/    # immutable source + resolved config
├── execution_*.json             # GPU/precision/scheduling history
├── data_contract.json
├── train_pair_status.json
├── V10_1/dual_teacher/
├── V11/dual_teacher/
├── comparison_accuracy.csv
├── comparison_checkpoint_selections.csv
├── comparison_efficiency.csv
├── comparison_training_budget.csv
├── comparison_common_epoch_budget.csv
├── comparison_stages.csv
├── comparison_tails.csv
├── comparison_GT_boundaries.csv
├── comparison_sensor_policies.csv
├── comparison_v11_solvers.csv    # nếu bật RUN_SOLVER_AUDIT
├── comparison_summary.json
├── comparison_dashboard.png
└── COMPARISON.md
```

Trong mỗi `dual_teacher/`: `train_console.log` mirror khoảng 30 giây; `train_log.csv`, JSONL và checkpoint backup mỗi epoch; `best.pth`, `best_inverse.pth`, `best_joint.pth`, `last.pth`; các JSON validation/profile và `kitti_test_predictions.zip`. Console Colab có prefix model, dễ phân biệt hai luồng.

## 4. Đọc bảng đúng cách

- Bảng chính dùng **checkpoint minimum RMSE**, iRMSE đi kèm là của **cùng checkpoint**. Hai lựa chọn inverse/joint báo riêng. SHA checkpoint được kiểm tra trước/sau evaluate/profile.
- Runtime chính là **real100 wall median/P95**, batch 1, 352×1216, cùng GPU/precision/backend, sau khi hai tiến trình train đã thoát. Epoch time khi train đồng thời chỉ phản ánh contention, không phải isolated model speed.
- V11 static midpoint 8 NFE là **solver approximation riêng**, có accuracy riêng cùng checkpoint; không ghép adaptive accuracy với static latency.
- Early stop có thể khác số epoch: báo actual optimizer updates/epochs và best trong common logged epoch prefix. Common-prefix lấy từ epoch log, không giả định checkpoint epoch đó còn giữ.
- Native-stage metrics ở scale khác có support khác, không so trực tiếp như final RMSE. Anonymous test không có public GT; ZIP dùng submit, không có local test RMSE chính thức.

Đây là một-seed **architecture bundle comparison**, không phải pure solver-only ablation: V11 khác cả state chart/solver. Cùng seed không bảo đảm mọi student weight khởi tạo giống nhau; cùng 400 val được dùng early stop và chọn checkpoint, không phải independent held-out test.

## 5. Kiểm tra local

Coordinator có contract tests và QA chạy đồng thời **hai train runner thật**, mô phỏng ngắt sau epoch 0 rồi resume. Các phép này dùng dữ liệu giả 64×128 trên CPU; không phải bằng chứng GPU latency/VRAM/accuracy. Hai model bundle cũ vẫn giữ nguyên checksum. Notebook chạy lại contract tests + smoke với data/GPU thật trước train.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_anchorflow_pair_benchmark.py -v
.\venv\Scripts\python.exe scripts/check_anchorflow_pair_runner.py
```
