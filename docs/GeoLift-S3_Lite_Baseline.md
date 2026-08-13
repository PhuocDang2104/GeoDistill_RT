# GeoLift-S3 Lite — baseline kỹ thuật và hướng phát triển chính

Tài liệu này mô tả đúng `GeoLiftStudentS3Lite`, objective teacher-free và benchmark TAR2000 hiện tại. S3 là hướng phát triển chính; S2 v2.1 được giữ làm mốc teacher-distillation.

## 1. Thiết kế cốt lõi

S3 tách RGB và sparse depth, dùng RayLift chỉ tại hai stage có giá trị lớn nhất cho boundary/subpixel reconstruction, đồng thời thêm residual inverse-depth correction để sửa metric bias.

```text
RGB ── MobileNetV4-Conv-Small-0.5 pretrained ─ F4 ─ F8 ─ F16
                                                        │
S,M ─ compact sparse pyramid ─────────────── S4 ─ S8 ─ S16
                                     gated fusion tại từng scale
                                                        │
                                           LiteFPN 32/24/24
                                                        │
                                                      D16
                                                        │ residual metric 16→8
                                                       D8
                                                        │ residual metric 8→4
                                                       D4
                                                        │ Residual RayLift 4→2
                                                       D2
                                                        │ Residual RayLift 2→1
                                                       D1
                                                        │ hard sparse anchor
                                                     D_full
```

### Contract

| Mục | Giá trị |
|---|---|
| Input | RGB, sparse depth `S`, mask `M`, intrinsics `K` |
| Kích thước | `B×3×352×1216`; H/W chia hết cho 16 |
| Miền depth | `(0.001, 120)` m |
| Encoder | `mobilenetv4_conv_small_050.e3000_r224_in1k` |
| Pretrained | ImageNet, `true`; toàn encoder được fine-tune |
| Output học được | `D16, D8, D4, D2, D1` |
| Output deploy | `D_full=(1-M)D1+MS` |
| Teacher | Không đọc khi train hoặc inference |

Pretrained chỉ là initialization; không thêm parameter, MAC hoặc inference branch.

## 2. Các block

### Sparse pyramid và gated fusion

Sparse branch tạo local normalized prior ở `1/4`, rồi valid-pool thành `S4/S8/S16`. Mỗi state có năm kênh: sparse, mask, local prior, prior validity và density; learned width chỉ 8 kênh mỗi scale.

Tại `s∈{4,8,16}`:

\[
F_s=\operatorname{DWRes}(P_I(F^I_s)+g_sP_S(F^S_s)),
\quad g_s=\sigma(G_s([P_I,P_S])).
\]

Fusion gate khởi tạo gần `0.10`; RGB và depth không concat tại input. FPN xuất width `32/24/24` cho `P4/P8/P16`.

### Low-resolution residual metric lift

Hai stage `16→8` và `8→4` bỏ sampling hình học đắt tiền:

\[
\xi_{out}=\operatorname{up}(\xi_{in})+g\,\Delta\xi,
\quad \xi=1/D,
\]

với residual limit `0.02` và gate khởi tạo `0.05`.

### Learned phase guidance và Residual RayLift

RGB được `PixelUnshuffle/phase_pack` thành bốn phase tại `1/2`, sau đó grouped `1×1` + DWConv tạo guidance 12 kênh. Không tạo learned feature map full-resolution.

Hai stage cao dùng:

| Stage | Sampling | Residual limit |
|---|---|---:|
| `4→2` | line, 3 hypotheses | 0.02 |
| `2→1` | center + learned neighbor, 2 hypotheses | 0.01 |

RayLift sample parent inverse depth, affine-transport theo camera ray, softmax-aggregate hypothesis rồi blend với bilinear. S3 thêm correction rõ ràng:

\[
\xi_{out}=\xi_{RayLift}+g_r\Delta\xi,
\qquad g_{r,0}=0.05.
\]

`C_full` chỉ là confidence analytic truyền qua decoder để giữ infer/export contract; S3 hiện không có confidence head hoặc confidence calibration loss.

## 3. Objective teacher-free

Tất cả loss bật từ epoch 0, không có Stage A/B/C:

\[
L=L_{metric}+0.2L_{log}+0.1L_{sparse}+0.05L_{edge}.
\]

\[
L_{metric}=0.05L_{D16}+0.10L_{D8}+0.25L_{D4}+0.50L_{D2}+1.00L_{D1}.
\]

| Loss | Định nghĩa |
|---|---|
| Metric | Huber depth theo mét, `δ=1`, trên năm scale |
| Log | Log-depth Huber tại `D1` |
| Sparse | L1 giữa `D_pre_anchor` và sparse sensor |
| Edge | Sai khác log-depth gradient tại cặp GT hợp lệ |

Hard anchor chỉ được áp dụng sau `D_pre_anchor`; vì thế sparse loss không bị bằng 0 giả. Gate tự học qua gradient, không được mở theo epoch.

Các bin `[0,20,40,60,80,120]` hiện **chỉ dùng để log validation**. S3 chưa có range-balanced loss và mọi GT pixel hợp lệ có trọng số như nhau trong từng scale.

## 4. Train và protocol

| Mục | Giá trị canonical |
|---|---|
| Data Drive | `selected_2000_ids.json` + `kitti_trainval_2000.tar` |
| Split | 1.600 train / 400 val, raw-drive disjoint |
| Test | 1.000 KITTI anonymous |
| Epoch | 30 (`0…29`) |
| Batch | 2 |
| Optimizer | AdamW, LR `2e-4`, weight decay `1e-5` |
| Scheduler | 1 epoch warm-up, cosine, min LR ratio `0.05` |
| Precision | FP16 AMP, channels-last, grad clip `1.0` |

Notebook: `notebooks/GeoLift_S3_Lite_TAR2000_Train1600_Val400_Test_OneRun.ipynb`.

Output Drive: `MyDrive/GeoLift_RT_runs/v3_s3_lite_pretrained_train1600_val400/`.

Notebook không mở hoặc extract teacher TAR. Nó kiểm tra manifest/bundle, dataset contract, contract tests, one-batch forward/backward và protocol lock trước khi chạy train → val → anonymous test → FP16 profile. Resume chỉ dùng checkpoint S3 cùng run.

## 5. Baseline quan sát được — epoch 0…19

Nguồn: `results/GeoLift-S3_Lite_TAR2000_train_log_epoch0_19.csv`. Đây là validation nội bộ 400 ảnh, chưa phải KITTI leaderboard và run 30 epoch chưa kết thúc.

| Metric | Epoch 0 | Best hiện tại, epoch 19 |
|---|---:|---:|
| RMSE | 9.029 m | **1.510 m** |
| MAE | 4.694 m | **0.458 m** |
| AbsRel | 0.3391 | **0.0261** |
| δ1 | 0.5632 | **0.9889** |

RMSE giảm `83.3%`. Sau epoch đầu, train thường khoảng `138–147 s/epoch` và validation khoảng `34–40 s/epoch` trên runtime đã ghi log; đây không phải inference latency.

Range tại epoch 19:

| GT range | RMSE | Tỷ lệ valid pixels |
|---|---:|---:|
| 0–20 m | 0.903 m | 77.43% |
| 20–40 m | 1.926 m | 16.73% |
| 40–60 m | 3.358 m | 4.41% |
| 60–80 m | 5.702 m | 1.40% |
| 80–120 m | 15.496 m | 0.031% |

Hai lưu ý bắt buộc khi đọc kết quả:

1. Global RMSE bị chi phối mạnh bởi vùng gần; bin xa nhất có rất ít GT pixel.
2. iRMSE tăng bất thường từ epoch 5 và đạt `310.4` ở epoch 19 dù RMSE giảm. Điều này gợi ý một số prediction rất gần zero; cần inspect percentile/min-depth và ảnh outlier trước khi promote checkpoint.

Loss budget epoch 19 gần như chỉ còn metric và sparse:

| Weighted term | Giá trị | Tỷ lệ total |
|---|---:|---:|
| Metric | 0.88657 | 90.74% |
| Sparse | 0.09019 | 9.23% |
| Log | 0.000268 | 0.027% |
| Edge | 0.000012 | 0.001% |

Vì vậy `lambda_log=0.2` và `lambda_edge=0.05` không đồng nghĩa chúng đóng góp 20% và 5% gradient/loss thực tế.

## 6. Hướng phát triển có kiểm soát

Ưu tiên giữ nguyên S3 làm control và thay đúng một yếu tố mỗi run:

1. Hoàn thành đủ 30 epoch và inspect outlier gây iRMSE trước.
2. Thêm range-balanced term như một ablation riêng; không thay bin, teacher và kiến trúc cùng lúc.
3. Kiểm tra gradient/scale của edge loss vì weighted contribution hiện gần zero.
4. Ablate pretrained `true/false`, residual correction on/off và learned guidance so với raw PPG.
5. Chỉ thêm một metric teacher KD nếu validation thật sự plateau; không đưa lại multi-teacher curriculum ngay.
6. Dùng CUDA FP16 component profile để xác định bottleneck; MAC không bao gồm `grid_sample` và memory/layout cost.

## 7. Source of truth

- Model: `src/model_geolift_s3.py`
- RayLift primitive: `src/model_geolift_s2.py::RayLiftIDBlock`
- Loss: `src/losses.py::geolift_s3_loss`
- Config: `configs/geolift_s3_lite_tar2000.yaml`
- Train/eval: `src/train_student.py`
- Profile: `scripts/profile_geolift_s3.py`
- Contract tests: `tests/test_geolift_s3_contracts.py`

Mọi claim mới phải báo ít nhất global metrics, range metrics, iRMSE, edge/non-edge metrics, parameter, FP16 latency/P95/VRAM và protocol hash.

## 8. Decoder V2 — ablation kế tiếp

Decoder V2 là một run fine-tune tách biệt; S3 baseline và loss không bị sửa.

```text
S3 epoch_019.pth
      │
      ├── direct metric-depth residual tại 8→4
      ├── sparse error E8=M8(S8-D8) tại 8→4
      └── encoder F2 reuse tại 4→2 và 2→1
      │
10 epoch fine-tune, cùng TAR2000/protocol/loss
```

Direct correction:

\[
D_4=D_{4,base}+g_m\Delta D,
\quad \Delta D=5\tanh(h_D(Z_8)),
\quad g_m=\sigma(h_g(Z_8)).
\]

`metric_delta` khởi tạo zero, `metric_gate` khởi tạo `0.05`, sparse projection khởi tạo rất nhỏ (`σ=1e-3`) và hệ số F2 `alpha=0`. Khi load S3 epoch 19, mọi output `D16…Dfull` ban đầu trùng baseline chính xác. Thay đổi thêm 1.473 parameters, từ 369.209 lên 370.682 (`+0.40%`), khoảng `0.019 G` Conv MAC ở `352×1216`; runtime thật vẫn phải lấy từ CUDA profiler. Sparse chỉ làm guidance; hard anchor vẫn chỉ nằm tại output cuối.

Fine-tune dùng LR base `5e-5`, encoder `1.25e-5`, decoder cũ `5e-5`, head mới `1e-4`, warm-up một epoch rồi cosine decay. Objective giữ nguyên S3; không thêm range loss.

Depth audit chạy mỗi validation và final inference:

- min và quantile `0.01%/0.1%/1%/10%` của `D8/D4/D2/D1/Dfull` trên valid GT grid;
- count prediction `<0.01/0.05/0.1/0.5 m` theo stage;
- `iRMSE_raw` và `iRMSE_protocol` cho `D1/Dfull`;
- top-100 D1 inverse-depth outlier với sample ID, `(u,v)`, GT và prediction của mọi stage.

Notebook: `notebooks/GeoLift_Decoder_V2_TAR2000_Finetune_OneRun.ipynb`. Config: `configs/geolift_decoder_v2_finetune_tar2000.yaml`. Output tách biệt: `MyDrive/GeoLift_RT_runs/v3_decoder_v2_finetune10_from_s3_epoch19/`.

Acceptance giữ nguyên: pass mạnh nếu RMSE `≤1.45 m` và latency tăng không quá `5%`; pass vừa nếu RMSE `≤1.48 m` cùng edge và `40–80 m` tốt hơn rõ, không làm `0–20 m` xấu đáng kể.
