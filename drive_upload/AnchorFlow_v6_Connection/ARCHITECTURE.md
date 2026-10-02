# AnchorFlow V6 — Ray-Connection Phase Metric

**Implementation:** `model_v6.py` · **Training:** 15 epoch fine-tune từ V5 · **Input:** RGB, sparse depth, mask, intrinsics. Không GT/teacher trong inference. Đây là kiến trúc nghiên cứu đã implement, **không phải model đã đạt <0,7 m**.

## 1. Flow cuối cùng

```text
 RGB ── MobileNetV4-Conv small 0.50 pretrained ── F4/F8/F16/F32
 Sparse + mask ── compact sparse pyramid ──────────────┐
                                                      ▼
                      F32 context + lightweight decoder
                                      │
                        D16 → D8 → coarse D4
                                      │
                    AnchorFlow 3 bước + metric refine  [giữ V3]
                                      │
               Piecewise plane transport + barriers   [giữ V5]
                                      │ D4_surface
                 ╔══════════════════════════════════╗
                 ║ Ray-Connection Metric @1/4       ║
                 ║ predicted normal + tangent frame ║
                 ║ tangent/normal residual vector   ║
                 ║ 4-neighbor normal alignment      ║
                 ║ ray-projected scalar correction  ║
                 ╚══════════════════════════════════╝
                                      │ D4
                  existing phase upsample 4→2
                                      │ D2_base
                   Phase Metric @1/4 → PixelShuffle2
                                      │ D2
                  existing phase upsample 2→1
                                      │
                    phase detail + sensor trust       [giữ V3]
                                      │
                     learned soft sensor fusion
                                      ▼
                                    Dfull
```

Không tăng encoder, không reset decoder V5, không thêm CNN feature full-resolution. Không PCA/latent attention, ODE solver, cost volume, point-cloud splat/warp, normal teacher hay loss connection mới. Context toàn cảnh vẫn đến từ F32/context hiện hữu.

## 2. Tensor/compute contract

RGB352×1216; các grid D16/D8/D4/D2/D1 lần lượt22×76/44×152/88×304/176×608/352×1216. P4 có48 channels; connection hidden64; phase hidden32 tại1/4. H,W chia hết32. Geometry/depth arithmetic FP32; CNN dùng AMP FP16 khi train/profile trên CUDA.

| Component | Parameters | Conv/Linear MAC ở batch1 |
|---|---:|---:|
| V5 giữ nguyên | 579.993 | 3,176 G |
| ConnectionMetric4 mới | 24.131 | 0,623 G |
| PhaseMetric2 mới | 8.772 | 0,224 G |
| **V6 tổng** | **612.896** | **4,024 G** |

Tăng5,67% params và26,68% Conv/Linear MAC. Số trên đếm bằng hook forward352×1216, **không phải latency ước lượng GPU**. Không đếm đầy đủ stencil shifts, pooling, interpolation, geometry, PixelShuffle và memory traffic. Normal alignment không tạo ma trận3×3 từng pixel: dùng cross-product Rodrigues vectorized trên bốn láng giềng. Không NxN attention hoặc Python pixel loop. Tensor geometry4×3 tại1/4 vẫn có memory cost, phải đo median/P95 và VRAM.

## 3. Geometry-aware metric residual

Camera ray `r_p` có thành phần z=1; quarter pixel center ánh xạ về full grid bằng `(u+0.5)*4-0.5`. Điểm camera:

$$
X_p = D_p r_p.
$$

Normal lấy từ đạo hàm inverse depth theo normalized ray coordinates. Với $\xi=1/D$, $a=\partial\xi/\partial x$, $b=\partial\xi/\partial y$:

$$
n_p = \frac{[a,b,\xi-ax-by]}{\|[a,b,\xi-ax-by]\|_2+\epsilon}.
$$

Code dùng one-sided derivative có magnitude nhỏ hơn để giảm cross-edge contamination; image borders dùng đạo hàm một phía thật. Normal/frame **detach** từ D4 để tránh gradient qua normal estimation, nhưng feature/depth trong head và residual update vẫn có gradient. Không có GT normals. Frame $t_1,t_2,n$ orthonormal với fallback axis khi n gần trục x; không claim gauge-equivariance.

Input head: P4(48), RGB pooled4(3), D/S innovation/valid/density/spread(5), ray(3), normal(3), tổng62 channels. Head PW62→64 → inverted-residual DW3 → DW5 dilation2 → PW64→3 sinh coefficients:

$$
(c_a,c_b,c_n)=(1+0.08D)\tanh(h(F)),\qquad
v^T=c_a t_1+c_b t_2,\qquad v=v^T+c_n n.
$$

PW output weights/bias bằng0. Không thêm một amplitude gate khởi tạo0 khiến branch khó mở. Residual vector là correction trong mét, không phải chuyển động vật lý của cảnh.

### Connection xấp xỉ, không phải geodesic parallel transport

Với normal q→p, đặt $k=n_q\times n_p$, $c=n_q^T n_p$:

$$
R_{q\to p}v^T_q
=v^T_q+k\times v^T_q+
\frac{k\times(k\times v^T_q)}{1+c}.
$$

Chỉ transport tangent vector; normal coefficient là scalar được gắn vào $n_p$. Pair gần antipodal có c≤−0,95 bị bỏ và denominator được bảo vệ. Đây là **minimal normal-alignment discrete connection approximation**, không exact Levi-Civita transport trên path của manifold.

Độ tương thích hình học sử dụng symmetric point-to-plane error:

$$
e_{pq}=|n_p^T(X_q-X_p)|+|n_q^T(X_q-X_p)|.
$$

$$
\kappa_{pq}=0.05+0.95\exp\left(-\frac{e_{pq}}{0.25+0.02D_p}\right).
$$

Transport conductance kế thừa V5 learned symmetric barriers, nhân compatibility và antipodal mask. Tổng mass≤1, phần dư ở center. Không softmax lại chỉ trên neighbor, tránh normalize mất tác dụng barrier. Compatibility floor0,05 giảm tự khóa do coarse geometry sai; không phải bảo đảm tuyệt đối qua mọi boundary.

$$
\bar v_p = \left(1-\sum_q w_{pq}\right)v_p
+\sum_qw_{pq}\left(R_{q\to p}v_q^T+c_{n,q}n_p\right).
$$

Fixed camera grid không nhận arbitrary XYZ deformation. Least-squares projection giữ đúng ray:

$$
\Delta D_p=\frac{r_p^T\bar v_p}{r_p^Tr_p},\qquad
D'_p=\mathrm{clip}(D_p+\Delta D_p,0.1,120).
$$

Không lấy đơn giản v_z, không reproject point cloud rồi sinh holes/z-buffer. Mode `v6_ambient` average vectors trực tiếp trong cùng camera frame: control hợp lệ vì vectors cùng R3 vẫn cộng được.

## 4. Phase Metric tại D2

Reuse connection hidden64, phase-packed learned G2(32), phase-packed D2/sparse validity/density/innovation(16):112 channels tại1/4. PW112→32 + LiteBlock + PW32→4, PixelShuffle2 sinh correction tại1/2:

$$
D'_2=\mathrm{clip}\left(D_2+(1+0.05D_2)\tanh(\mathrm{Shuffle}_2(h_2(F))),0.1,120\right).
$$

Head cuối zero/no-op. Không hard-gate branch chỉ tại predicted edge: tránh bỏ qua interior outliers do edge map sai. Learned RGB phase, sparse evidence và boundary supervision vẫn hỗ trợ detail. Sau đó reuse D2→D1 và full-resolution scalar sensor fusion đã train.

## 5. Sensor và teacher

`Dfull=(1-g)D1+gS` với g là mask×learned trust×disagreement prior. Giữ soft fusion V5; hard anchor chỉ diagnostic `D_hard`. No GT in forward; GT–sensor consistency chỉ làm label/weight trong loss.

Metric KD dùng `D_cm/C_cm`, C≥0,5, tại D4/D2/D1. Loại **mọi pixel có GT hoặc original sparse sensor**, kể cả holdout; coarse cell chứa forbidden pixel cũng bị loại. Chỉ cache1.600 train teachers. Val/test không load teacher. KD coefficient0,05 giảm tuyến tính về khoảng0,025 qua15epoch; không dùng geometry/normal teacher.

## 6. Objective và optimization

Giữ objective V5 và thêm robust-MSE:

$$
\mathcal L_{V6}=\mathcal L_{V5}+0.1\mathcal L_{\mathrm{robustMSE}}.
$$

$$
\mathcal L_{\mathrm{robustMSE}}
=\frac{1}{|\mathcal V|}\sum_{p\in\mathcal V}2\rho_{20}(D_{\mathrm{full},p}-D_{\mathrm{GT},p}).
$$

Huber rho_delta bằng0,5e² khi |e|≤delta, và delta(|e|−0,5delta) khi lớn hơn. Vì vậy 2rho20 = MSE ở lỗi≤20m, linear-tail robust ở lỗi>20m. Chỉ valid GT, FP32. Đánh trực tiếp SSE/outliers; không guarantee giảm RMSE hoặc gradient budget.

| Term V5 giữ lại | Weight |
|---|---:|
| Multi-scale GT Huber D16/D8/D4/D2/D1/Dfull | 0,025 /0,05 /0,15 /0,30 /0,50 /1,00 |
| Full-output batch RMSE / range RMSE | 0,25 /0,10 |
| Log / gradient-log | 0,20 /0,05 |
| Reliability-weighted sparse / sensor holdout / trust BCE | 0,02 /0,05 /0,02 |
| Metric KD / teacher edge | 0,05→~0,025 /0,02 |
| Fixed observed-GT boundary band3 RMSE / V5 directional barrier BCE | 0,05 /0,01 |
| **New robust-MSE** | **0,10** |

Tất cả bật từ epoch0; GT priority và sensor holdout10%, hflip giữ nguyên. Batch4, AdamW fused, channels-last, AMP FP16, clip1. Peak LR encoder1e-5/old decoder1e-4/new heads2e-4; warm-up1epoch rồi cosine minimum0,05×. Freeze encoder BN, không freeze encoder weights. New BN được train. Không gradient checkpointing hoặc compile mặc định.

15epoch với1600train/batch4 =6.000 optimizer updates; không so như batch2/12.000updates. Resume strict model/source/protocol, restore optimizer/scaler/RNG. Init V5 giữ mọi tensor cũ, new heads no-op; fresh optimizer/schedule. Epoch-1 parent validation vẫn eligible best để không mất baseline.

## 7. Đánh giá và novelty boundary

Đủ400val mỗi epoch: global pixel RMSE/MAE/iRMSE, ranges0–20/20–40/40–60/60–80/80–120, RGB-edge, GT-boundary1/2/3/5/10px, native D4_surface→D4/D2_base→D2, stage minima/low-depth count, error-tail1/2/5/10/20m với pixel fraction và SSE fraction. Không giảm subset validation để có số đẹp. Native stages so trên cùng grid/valid-area-mean GT, không so trực tiếp các scale khác nhau.

**Hypothesis contribution:** camera-ray-projected piecewise metric correction, tách tangent/normal coefficients và normal-alignment neighborhood transport, đi cùng phase-resolved metric refinement. [Field Convolutions](https://github.com/twmitchel/FieldConv) là related work về vector-field surface learning, không proof gain của V6. [PFCNN](https://arxiv.org/abs/1808.04952) dùng parallel frames trên surfaces; V6 không nhận toàn bộ intrinsic/gauge guarantees của phương pháp đó.

Không claim first, exact geodesic PT, physical flow, diffeomorphism, ODE benefit hoặc Q1 novelty đã chứng minh. Cần matched V5/V6-ambient controls và accuracy–latency Pareto. Local checks trong [VERIFICATION](VERIFICATION.md); chưa có V6 trained RMSE/CUDA/edge-device timing.
