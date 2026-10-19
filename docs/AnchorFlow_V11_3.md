# AnchorFlow V11_3 — official research baseline

> Historical accuracy reference; superseded for deployment on19/10/2026 by
> [S4-U-PIR NoRelative](AnchorFlow_V11_Baseline.md). The dated statements below
> describe the05/10 run, not the current solver. Linked historical payloads may
> require the external D archive; they are not needed by current G inference.

**Baseline chính từ 05/10/2026 · checkpoint `best_policy`, fine-tune epoch 9 / cumulative epoch index 32.**

V11_3 giữ neural vector field và numerical integration của V11, thêm fine NODE tại D2 và phase-aware sparse-innovation readout (PIR). Student không nhận teacher/GT khi inference. Đây là baseline được chọn để phát triển tiếp, **không phải model đã đạt mọi target hoặc KITTI leaderboard**.

Nguồn: [run FT15 đã audit](../results/benchmarks/v11_3_ft15/README.md), ZIP `dual_teacher-20261005T071424Z-1-001.zip`. [Spec trong bundle](../drive_upload/AnchorFlow_v11_3_PhaseInnovation/ARCHITECTURE.md) là frozen snapshot trước train; không sửa source/config/checksum đã train.

## 1. Flow và nhiệm vụ từng khối

```text
RGB → MobileNetV4 small pretrained → F4/F8/F16 + F32 context
Sparse S, M, intrinsics K → compact sparse pyramid
                         ↓ gated fusion / decoder
                      D16 → D8 → D0 (1/4)
                         ↓ seed inverse-depth jet → smooth chart z0
                   Coarse NODE: Bosh3, T=1
                         ↓ D4 geometric (1/4)
                   Phase lift → D2_base
                         ↓ five-jet inverse consensus
                       D2_query
                         ↓ sparse-innovation metric correction
                       D2_pre_fine
                         ↓ Fine NODE: one midpoint RK2, exactly 2 NFE
                         D2 (1/2)
                         ↓ phase lift + learned quarter-grid context
                       D1_base
                         ↓ learned detail, half-grid CNN / PixelShuffle
                       D1_pre_pir
                         ↓ phase-aware sparse-innovation residual (PIR)
                         D1 (full)
                         ↓ learned sensor reliability fusion
                        Dfull
```

| Contract | Implementation |
|---|---|
| Forward inputs | RGB, sparse S, mask M, K; không GT/teacher |
| Image / D4 / D2 grid | 352×1216 / 88×304 / 176×608; H,W chia hết 32 |
| Model | `AnchorFlowEdge` trong frozen V11_3 bundle |
| Encoder | `mobilenetv4_conv_small_050.e3000_r224_in1k`; weights đã pretrained và train ở parent |
| Parameters | **584,845**, gồm PIR 952 và fine NODE 981 |
| Precision | CNN native BF16; chart/RHS/solver/loss FP32 |
| Full-resolution path | Scalar depth, phase packing/shuffle và analytic operations; không full-res learned feature CNN |
| Output policy | Soft sensor fusion; hard anchor chỉ diagnostic |

## 2. Jet và coarse/fine dynamics

Mỗi cell có local quadratic inverse-depth surface:

$$
j=[v,g_x,g_y,h_{xx},h_{xy},h_{yy}],\qquad v=1/D,
$$

$$
v(s,t)=v+g_xs+g_yt+\tfrac12h_{xx}s^2+h_{xy}st+\tfrac12h_{yy}t^2.
$$

s,t là grid-cell coordinates, không world XY; τ là pseudo-time refinement, không physical motion. Minmod seed derivative được detach; value còn gradient. Solver tích phân chart z với smooth bounded decode, **không** hard-project từng RK stage:

$$
v=\exp[\log(1/120)+\log(1200)\sigma(z_0)],\qquad
j_{1:5}=v\kappa_{1:5}\odot\tanh(z_{1:5}),
$$

κ = [1,0.5,0.5,0.25,0.25,0.25]. Với finite z: center depth 0.1–120 m, slopes ≤0.5v, Hessian ≤0.25v. Bounds không chứng minh accuracy, global integrability hoặc physics/PINN constraints.

Shared RHS refresh current jet, sparse innovation và conductance mỗi call. Analytic transport T đổi gốc polynomial neighbor trước khi trao đổi:

$$
A_p=R_{\theta,p}+Q_{\theta,p}+\sum_qw_{pq}[T_{q\to p}(j_q)-j_p],
\qquad \frac{dz_p}{d\tau}=2\tanh\left(\frac{A_p}{2v_p\kappa}\right).
$$

ODE định nghĩa trong **chart space**; A không bằng dj/dτ và normalization không là exact Jacobian pullback của một physical-jet ODE.

- **Coarse D4:** official `torchdiffeq==0.2.5`, Bosh3 RK3(2), rtol0.01/atol0.001, first step0.25, max step1, T=1. Dense output lấy z(0.5), không ép endpoint0.5. Full400: **113 scenes 7 NFE, 287 scenes 10 NFE**, mean9.1525; zero rejects, T=1. Không learned step/stop; guard báo lỗi, không forced accept.
- **Fine D2:** seed lại jet trên half-grid, không copy derivatives D4. Fine field có weights riêng (981 parameters), shared giữa hai calls, không shared với coarse field. Một fixed midpoint step: k1=f_fine(0,z0), k2=f_fine(0.5,z0+0.5k1), z1=z0+k2. **Đúng 2 NFE**; midpoint là internal RK stage, không exact half-time solution. Context dùng existing p2/g2/sensor, reused coarse barrier; không adaptive error control.

## 3. Readout và PIR

Five-jet readout hỏi center/E/W/S/N tại cùng target phase, tổng hợp **inverse depth** rồi reciprocal. Metric innovation sửa D2 query; fine NODE sửa geometry tiếp ở half-grid.

PIR dùng donor-relative sparse error e=M(S−D1_pre_pir). PixelUnshuffle giữ 4phase; 3×3 half-grid pooling theo từng phase tạo mean/std/density. Thêm phase log-depth contrast và existing24ch detail feature: PW40→16, DW3×3, PW16→8 (delta4 + gate4). Không GT/teacher trong head.

$$
\Delta D=(0.5+0.05D_1^0)\odot\mathrm{PixelShuffle}[\tanh(a)\odot\sigma(b)],
\qquad D_1=\mathrm{clip}(D_1^0+\Delta D,0.1,120).
$$

Delta zero-init bảo toàn parent lúc bắt đầu. Sensor fusion tính lại gate từ refined depth và sensor conflict; **không hard-anchor bắt buộc**.

## 4. Training đã chạy

1,600 train / 400 val / 1,000 anonymous test; ID/raw-drive disjoint. GT/sparse uint16 /256 theo mét. Metric teacher D_cm/C_cm và relative DA3MONO-LARGE R_T/C_T cache **1,600 train / 0 val**, loại GT/original-sensor support khỏi teacher loss. Relative confidence heuristic, không calibrated certainty.

Warm-start model-only V11_2 **epoch22**; optimizer/schedule mới. Stage0…14 là **15 extra epochs**, cumulative23…37. BF16, B4, seed42, fused AdamW; LR5e−5, encoder×0.25/olddecoder×1/PIR×2, warmup1→cosine, clipping1, freezeencoderBN. Không early stop trong run này.

| Objective | Weight / schedule |
|---|---|
| GT multiscale Huber D16/D8/D4/D2/D1/Dfull | 0.025/0.05/0.15/0.30/0.50/1 |
| Final RMSE / range RMSE | 0.6 / 0.1 |
| Direct iRMSE / excess-tail Huber | 0.04→0.05 / 0.25→0.30, ramp2epochs |
| Coarse half/final trajectory | total0.10→0.05, ramp2; equal half/final |
| Sparse / holdout / trust / boundary / barrier | 0.02/0.05/0.02/0.05/0.01 |
| Metric KD | 0.006→0.004 |
| Relative gradient KD at D2 | 0→0.015, ramp2epochs, confidence≥0.35 |
| Log / edge / teacher-edge / legacy inverse | Off |

`best`, `best_inverse`, `best_joint`, `best_policy` đều chọn **cùng model state epoch9**, không ghép metric hai checkpoints. `last` epoch14 phục vụ exact child-resume, không là baseline accuracy chính. Training wall trong epoch log: ~39.6 phút cả train/val/checkpoint; riêng train ~27.0 phút, không gồm prepare/smoke/test/profile.

| FT epoch / cumulative index | Train total | Val RMSE m | Val iRMSE km⁻¹ | Mốc |
|---|---:|---:|---:|---|
| 0 / 23 | 1.33620 | 0.990193 | 3.221522 | Bắt đầu child run |
| 3 / 26 | 1.34255 | 0.987548 | 3.198553 | Lần đầu iRMSE≤3.2 |
| **9 / 32** | **1.32090** | **0.986517** | **3.169735** | **Tất cả best cùng checkpoint** |
| 14 / 37 | 1.30728 | 0.986815 | 3.180581 | Last, không tốt hơn best |

Train total tăng nhẹ đầu run khi các hệ số loss đang ramp, không tự chứng minh model xấu đi. Sau epoch9, total tiếp tục giảm nhưng validation không cải thiện tương ứng: ưu tiên phân tích outliers/generalization, không đơn thuần kéo dài training.

## 5. Metric chính thức của baseline nội bộ

Global pixel accumulation trên **25,424,992 valid GT pixels**, không mean per-image RMSE. iRMSE/iMAE đơn vị **km⁻¹**; anonymous test không public GT.

| Output / checkpoint | RMSE m ↓ | iRMSE km⁻¹ ↓ | MAE m ↓ |
|---|---:|---:|---:|
| Parent V11_2 inverse, epoch22 | 0.999491 | 3.207087 | 0.259726 |
| **V11_3 baseline, stage9 / cumulative32** | **0.986517** | **3.169735** | **0.255663** |
| Same weights, disable PIR diagnostic | 0.987965 | 3.173899 | 0.257384 |
| Same weights, pre-sensor fusion | 0.999321 | — | 0.267073 |
| Same weights, hard-anchor diagnostic | 1.127354 | — | 0.265322 |

iMAE **1.028780**, AbsRel **0.0131823**, δ1 **0.9967231**. Target iRMSE<3.2 đã đạt; **RMSE<0.98 chưa đạt**. Original V11 best-RMSE là0.986021/3.215515: V11_3 RMSE kém0.50mm nhưng inverse tốt hơn0.04578; không claim thắng mọi metric/budget.

| GT region | Parent RMSE m | V11_3 RMSE m |
|---|---:|---:|
| 0–20 | 0.462871 | 0.442788 |
| 20–40 | 1.400359 | 1.370168 |
| 40–60 | 2.504023 | 2.499864 |
| 60–80 | 3.817678 | 3.839715 |
| 80–120 | 8.831096 | 8.986066 |
| RGB edge | 1.409427 | 1.392866 |
| GT boundary3px | 1.684436 | 1.680538 |

## 6. Tác động từng stage / runtime

| Same-grid transition | Native RMSE m before → after | Diễn giải |
|---|---|---|
| Coarse D0 → D4 | 1.85103 → 1.67155 | Geometry coarse có tác dụng |
| D2_base → D2_query → metric readout | 1.33821 → 1.27047 → 1.22663 | Consensus + metric innovation quan trọng |
| Fine NODE D2 | 1.22663 → 1.20692 | Gain19.7mm RMSE; MAE0.34842→0.31815 |
| D1_base → detail → PIR | 1.03655 → 1.00202 → 0.99932 | Detail lớn hơn PIR |
| D1 → soft sensor Dfull | 0.99932 → 0.98652 | Giữ soft fusion; hard diagnostic tệ hơn |

Mỗi dòng cùng resolution/pooled-GT support. **Không** lấy RMSE D4 và D2 trừ nhau để claim component gain. Counterfactual bỏ PIR cùng checkpoint không thay thế matched retraining ablation.

A100-SXM4-**40GB**, Torch2.11+cu130, BF16 CNN/FP32 geometry, B1, real100 after train: adaptive **75.18/79.17ms median/P95**, static coarse-midpoint8 + fine2 **61.51/62.62ms**. Static accuracy full400 **chưa có trong ZIP**, không suy parity từ run V11 cũ. Profile MAC adaptive synthetic10NFE≈3.9752G, static≈3.8656G; excludes analytic/pooling/memory. Instrumented dynamics36.92ms và fine7.33ms là bottleneck; separate component pass, không cộng để thay wall timing.

## 7. Insight và hướng khai thác tiếp

1. **Chốt epoch9, không last:** tất cả raw best đồng thời; sau đó train loss giảm nhưng val plateau/oscillate. Train thêm đơn thuần chưa giải quyết tail.
2. **Gain chủ yếu near/mid:** 60–120m hơi xấu đi; không nói far-range đã được sửa. Giữ range và boundary audit khi thay objective.
3. **PIR hữu ích nhưng nhỏ:** cùng checkpoint giảm1.449mm final RMSE, +952params. Mean correction fullframe38mm; gate0.116, phase support24.95%. Nó còn tăng SSE |e|>10m ~41,814m² dù total tốt hơn: cần inspect donor/mixed-surface outlier trước khi tăng amplitude.
4. **Tail vẫn quyết định:** 0.1757% pixels |e|>10m đóng57.29% SSE; boundary3px đóng38.19%. Nếu phần còn lại giữ nguyên, giảm ~2.30% SSE của tail này đủ đạt0.98; đây là phép tính điều kiện, không dự báo training.
5. **Inverse bảo vệ near:** 0–5m chỉ~1.50% GT pixels nhưng29.81% inverse SSE; không có final prediction<0.1m. Không clamp metric để tạo gain.
6. **Relative chưa có ablation nhân quả:** best weighted contribution ~0.099% total, coverage31.17%; metric KD~0.710%. Không chứng minh relative vô dụng hoặc nó gây gain. Model/loss/schedule/teacher đều đổi trong FT bundle.

400 validation đã dùng lặp lại để chọn model/checkpoint; cần independent holdout và nhiều seed trước khi claim generalization hoặc causal architectural gain.

## 8. Source / chạy / visualization

[Folder train V11_3](../drive_upload/AnchorFlow_v11_3_PhaseInnovation/README.md) · [Notebook FT15](../drive_upload/AnchorFlow_v11_3_PhaseInnovation/Train_AnchorFlow_V11_3_Epoch22_DualTeacher_FT15.ipynb) · [Audit/provenance](../results/benchmarks/v11_3_ft15/audit.json).

Entrypoint: frozen `run.py`; model: `model.py`, `model_parent.py`, `model_base.py`; geometry: `geometry.py`, `fine_geometry.py`, `ode_solver.py`; PIR: `phase_readout.py`. Không dùng legacy `src/train_student.py` để train baseline này. Promote không thay weights/source của run đã train; đổi model/config tạo version mới.

[Pipeline viewer](../visualizations/jet_dynamics_lab/index.html) dùng **cùng best-policy checkpoint V11_3**, 10 real validation scenes, CPU FP32 replay. Khung full-width chạy progress D0→D4→D2→D1→Dfull, có cùng pixel-depth trajectory và before/after/Δdepth/Δ|lỗi GT|. Coarse animation dùng dense observations; fine dùng continuous extension RK2 bậc2 từ đúng hai calls, không exact solution. Internal RK-mid inspect riêng, thường khác dense half. Stage CNN/readout chỉ blend endpoints để trình bày, không giả intermediate network predictions. Full-image six-channel jets được giữ lossless trong sidecar, cùng slider/play 3D và vùng cell đồng bộ; khung ảnh lớn có RGB/depth hoặc previous/current wipe. Native scene audit không phải BF16 global400 benchmark. [Gói Vercel](../visualizations/AnchorFlow_V11_3_Vercel/README_DEPLOY.md) có zero-dependency static build, source full-jets ~288MB (không dùng CLI Hobby100MB); không thay model/checkpoint.
