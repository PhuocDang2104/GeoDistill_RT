# GeoLift-RT v2.1 — baseline kỹ thuật

Tài liệu này mô tả đúng `GeoLiftStudentS2` và profile train TAR2000 hiện tại. Đây là baseline teacher-distillation để so sánh với S3, không phải graph GeoRT/A0 cũ.

## 1. Contract

| Mục | Giá trị |
|---|---|
| Input | RGB, sparse depth `S`, mask `M`, intrinsics `K` |
| Kích thước | `B×3×352×1216`; H/W phải chia hết cho 16 |
| Miền depth | `(0.001, 120)` m |
| Output học được | `D16, D8, D4, D2, D1` và confidence tương ứng |
| Output deploy | `D_full=(1-M)D1+MS`; `C_full=(1-M)C1+M` |
| Teacher inference | Không có; teacher chỉ được đọc khi train |

Hard anchor bảo đảm depth tại điểm LiDAR bằng sensor chính xác. Vì vậy sparse loss phải đo trên `D_pre_anchor=D1`, không đo trên `D_full`.

## 2. Inference graph

```text
RGB ── RGBQuarterStem ─────────────── 24 ch ┐
S,M ─ local sparse prior ─ DepthStem 16 ch ├─ concat + ray xy: 42 ch
K   ─ ray coordinates ───────────────  2 ch ┘
                                             │
                              1×1 42→32 + residual DWConv
                                             │
                       adapted MobileViTv2-0.75: F4/F8/F16
                                             │
                                  LiteFPN, width 24
                                             │
                                      D16, C16
                                             │
                         RayLift-ID 16→8, cross, K=5
                                             │
                         RayLift-ID  8→4, line,  K=3
                                             │
                 PPG ─── RayLift-ID  4→2, line,  K=3
                                             │
                 PPG ─── RayLift-ID  2→1, neighbor, K=2
                                             │
                                      hard sparse anchor
                                             │
                                      D_full, C_full
```

### Sparse prior và fusion

Sparse depth được valid-aware downsample về `1/4`, sau đó local-normalized trong cửa sổ bán kính 7. Prior trả về `S4, M4, D_init, V_init, rho4`; vùng không có support giữ invalid, không global-fill.

Learned processing chỉ bắt đầu ở `1/4`. Fusion hiện tại là bottleneck 1×1 + depthwise residual, không còn hai standard convolution rộng full-resolution:

\[
F_4=\operatorname{DWRes}(\operatorname{Conv}_{1\times1}([F_I,F_S,r_x,r_y])).
\]

### RayLift-ID

Mỗi stage làm việc trong inverse depth `ξ=1/D`:

1. phase-pack bốn vị trí con theo thứ tự `q00,q10,q01,q11`;
2. dự đoán offset và sample các hypothesis từ parent bằng bilinear `grid_sample`;
3. dự đoán slope `(a,b)` và vận chuyển affine theo camera ray;
4. softmax aggregation các candidate;
5. gate giữa inverse-depth bilinear và RayLift;
6. phase-unpack để tạo depth ở độ phân giải gấp đôi.

Stage cuối có confidence calibration nhẹ. Sau `D1`, hard anchor tạo output deploy.

## 3. Teacher train profile

Run TAR2000 canonical dùng đúng hai cache:

| Cache | Tensor | Vai trò |
|---|---|---|
| `metric_coarse_train_2000.tar` | `D_cm, C_cm` | Reliability-weighted metric KD |
| `geometry_fused_train_2000.tar` | `R_G, C_G` | SSI relative geometry và ordinal boundary |

Depth Anything raw và DSINE không được loader đọc trong run này. `geometry_fallback=false`; thiếu fused geometry phải fail preflight. `lambda_plane=0`, nên đây không phải slope/normal supervision.

## 4. Loss và curriculum canonical

Source: `configs/geolift_s2_v2_1_balanced_ablation.yaml` và `geolift_loss()`.

\[
\begin{aligned}
L={}&L_{GT}^{multi}+w_S L_S+w_{cm}L_{KD}+w_RL_{range}\\
&+L_{boundary}+0.02L_{cycle}
+w_G(L_{SSI}+L_{ord})+w_CL_C.
\end{aligned}
\]

GT multiscale dùng log-Huber:

| Scale | `D16` | `D8` | `D4` | `D2` | `D1` |
|---|---:|---:|---:|---:|---:|
| Weight | 0.20 | 0.35 | 0.50 | 0.70 | 1.00 |

Lịch theo epoch zero-based:

| Epoch | Thành phần mới/thay đổi |
|---|---|
| 0–2 | GT multiscale + sparse `0.20` + boundary `1.0` + cycle `0.02` |
| 3–5 | Metric KD ramp `0.133→0.267→0.40`; range ramp bắt đầu |
| 3–7 | Sparse giảm `0.17→0.14→0.11→0.08→0.05`; range tăng đến `0.005` |
| 5–7 | Fused geometry SSI + ordinal ramp `0.01→0.02→0.03` |
| 8–9 | Confidence loss ramp `0.01→0.02` |
| ≥9 | Giữ toàn bộ trọng số cuối |

Metric KD theo bin `[0,20,40,60,80,120]` chỉ được up-weight nếu cell audit xác nhận teacher đủ coverage và không kém baseline quá ngưỡng. Plane loss luôn tắt.

## 5. Protocol tái lập

| Mục | Giá trị canonical |
|---|---|
| Split | 1.600 train / 400 val, raw-drive disjoint |
| Test | 1.000 KITTI anonymous, không có local GT |
| Epoch | 30 (`0…29`) |
| Batch notebook | 2 |
| Optimizer | AdamW, LR `2e-4`, weight decay `1e-5` |
| Scheduler | 1 epoch warm-up, cosine, min LR ratio `0.05` |
| Precision | FP16 AMP, channels-last, grad clip `1.0` |

Notebook chuẩn: `notebooks/GeoLift_RT_v2_1_TAR2000_Train1600_Val400_Test_OneRun.ipynb`.

Notebook bắt buộc kiểm tra manifest, schema/coverage teacher theo split, raw-drive disjointness, loader/model/loss smoke test và immutable protocol lock trước khi train. Resume chỉ từ checkpoint cùng run.

## 6. Đánh giá đúng

- Chọn `best.pth` theo global pixel-weighted validation RMSE.
- Báo global RMSE/MAE/iRMSE/iMAE/AbsRel/δ và RMSE theo năm bin range.
- `D_init`, `D16…D1`, `D_full` có thể đo stage RMSE; feature stem/fusion không có RMSE độc lập.
- Runtime phải lấy từ CUDA FP16 profiler; MAC Conv/Linear không tính `grid_sample`, interpolation, phase transform và memory traffic.
- Anonymous test chỉ xuất 1.000 PNG để submit; không được gọi đó là test RMSE local.

## 7. Source of truth

- Model: `src/model_geolift_s2.py`
- Loss: `src/losses.py::geolift_loss`
- Config canonical: `configs/geolift_s2_v2_1_balanced_ablation.yaml`
- Train/eval: `src/train_student.py`
- Profile: `scripts/profile_geolift_s2.py`
- Contract tests: `tests/test_geolift_s2_contracts.py`

S2 được giữ làm mốc có teacher và bốn RayLift stage. Mọi phát triển mới nên so với S3 trước; chỉ quay lại S2 khi cần cô lập giá trị của teacher geometry/KD.
