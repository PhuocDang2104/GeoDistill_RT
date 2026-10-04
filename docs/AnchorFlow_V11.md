# AnchorFlow V11 — Solver-Controlled Jet NODE

**Baseline nghiên cứu chính · implementation + kết quả đã train · cập nhật 04/10/2026.**

V11 học **đạo hàm của geometric state hiện tại**, numerical solver tích phân state đến **cùng horizon T=1**. Không learned step size, learned stop hay GT quyết định solver. Novelty hướng tới **sparse-conditioned neural dynamics trong bounded second-order inverse-depth jet chart**, kết hợp analytic transport và phase-consensus readout; chưa claim “first-ever”.

Kết quả/source đối chiếu với [paired run V10.1/V11](../results/benchmarks/v10_1_v11_pair01/README.md). Package [ARCHITECTURE.md](../drive_upload/AnchorFlow_v11_NODE/ARCHITECTURE.md) là snapshot đặc tả **trước train**; tài liệu này bổ sung metric/insight sau train, không sửa frozen model.

## 1. Kiến trúc: mỗi khối làm gì?

~~~text
RGB ── MobileNetV4 small, ImageNet pretrained ── F4/F8/F16 + F32 context
Sparse depth S, mask M, intrinsics K ── compact sparse pyramid ───┘
                             │ gated fusion + lightweight decoder
                             ▼
                     D16 → D8 → D0 at 1/4
                             │ minmod seed + smooth chart
                             ▼
                  z(0): six-channel geometric state
                             │
        shared neural RHS fθ(t,z; RGB/sparse context)
                             │ numerical RK3(2), fixed T=1
                  z(0.5) ────┴──── z(1)
                    aux GT            │ decode final jet
                                      ▼
                          D4 + five-jet consensus query
                                      │ sparse-innovation metric correction
                                      ▼
                             D2 → phase lift/detail → D1
                                      │ learned sensor reliability fusion
                                      ▼
                                    Dfull
~~~

| Contract | Implementation |
|---|---|
| Input forward | RGB, S, M, K; không GT/teacher |
| Image / geometric grid | 352×1216 / 88×304; H,W chia hết 32 |
| Encoder | mobilenetv4_conv_small_050.e3000_r224_in1k |
| P4 / dynamics context / jet | 48 / 32 / 6 channels |
| Params | **582.912**, toàn model |
| Precision | Main CNN native BF16; RHS/state/solver/loss FP32; T4 dùng explicit FP32 |
| High-resolution path | Phase packing/shuffle; không learned full-resolution feature CNN |

Coarse decoder học absolute metric depth; dynamics phát triển local geometry; readout dựng boundary/subpixel depth. Final output **soft sensor fusion**, không hard-anchor bắt buộc.

## 2. Geometric state và neural vector field

Mỗi quarter-grid cell có jet inverse-depth, không phải một vector 6D chung cho toàn ảnh:

$$
j_p=[v_p,g_{x,p},g_{y,p},h_{xx,p},h_{xy,p},h_{yy,p}],\qquad v_p=1/D_p.
$$

Jet biểu diễn local quadratic surface trong **grid coordinates**, không phải vận tốc 3D hay physical-time motion:

$$
v_p(s,t)=v_p+g_{x,p}s+g_{y,p}t+\tfrac12h_{xx,p}s^2+h_{xy,p}st+\tfrac12h_{yy,p}t^2.
$$

Minmod khởi tạo derivative từ coarse inverse depth; derivative seed dùng detach, value channel vẫn có gradient. Solver tích phân latent chart z thay vì trực tiếp jet bị clip:

$$
v=\exp\big[\log(1/120)+\log(1200)\,\sigma(z_0)\big],\qquad
j_{1:5}=v\,\kappa_{1:5}\odot\tanh(z_{1:5}),
$$

với κ = [1, 0.5, 0.5, 0.25, 0.25, 0.25]. Decode giữ v trong [1/120,10], slope ≤0.5v và Hessian ≤0.25v với finite state; không project/clip từng RK stage. Output depth giới hạn 0.1–120 m.

Context Z tính một lần. Ở **mỗi RHS evaluation**, model decode current jet, refresh sparse innovation, conductance và depth compatibility:

$$
A_p=R_{\theta,p}+Q_{\theta,p}+\sum_{q\in\mathcal N_4(p)}w_{pq}\,[T_{q\to p}(j_q)-j_p],
$$

$$
\frac{dz_p}{d\tau}=f_{\theta,p}(\tau,z;Z,S_4,M_4)
=2\tanh\left(\frac{A_p}{2v_p\kappa}\right).
$$

R là learned six-channel reaction; Q là reliability-weighted sparse inverse innovation, tác động vào value; T đổi gốc polynomial bậc 2 chính xác; w hạn chế trao đổi qua discontinuity. Shared field dùng current state và actual pseudo-time, không BatchNorm trong repeated RHS.

Chia theo từng channel trong công thức RHS. Các channel là hệ số local polynomial; model không enforce global spatial integrability của slope/Hessian giữa mọi cell.

**Hiểu đúng:** ODE được định nghĩa **trong chart space**. Normalization không phải exact Jacobian pullback của V10 physical-jet ODE. Bounds giúp domain hợp lệ, không chứng minh depth accuracy, stability toàn NN hay physics/PINN constraints. Neural derivative + independent solver theo nguyên lý [Neural ODE](https://arxiv.org/abs/1806.07366).

## 3. Solver: học field, không học h

$$
z(1)=z(0)+\int_0^1 f_\theta(\tau,z(\tau))\,d\tau.
$$

Official torchdiffeq==0.2.5, **bosh3 embedded RK3(2)**: rtol=0.01, atol=0.001, first/max step=0.25; sample trajectory tại 0.5 và 1. Error norm lấy **max sample/channel spatial RMS** của normalized chart error. Sai số vượt tolerance thì reject/retry; guard 193 NFE/64 steps báo lỗi, không forced accept, đổi tolerance hay Euler fallback. Direct autograd qua executed solve; không adjoint/constant-memory claim. Xem [official solver documentation](https://github.com/rtqichen/torchdiffeq).

**flow_steps=2 là hai observation times, không phải 2 RHS calls.** Trong run này, train và full400 validation đều log **13 NFE, 4 accepted steps h=0.25, zero rejects, T=1**. Adaptive solver tồn tại thật, nhưng **chưa kích hoạt variable-step/compute trên data này**: mesh đang bị max-step cap chi phối. Tighter rtol=0.002/atol=0.0002 cho cùng NFE và cùng metric; chưa có bằng chứng cần solver chính xác hơn để giảm RMSE.

Static deployment candidate dùng four midpoint steps, **8 NFE**, cùng field/chart/T nhưng là solver approximation riêng. ONNX17 parity so với PyTorch **static**, không phải adaptive graph; chưa đo TensorRT hoặc edge GPU.

## 4. Training và hai teacher

Fresh student từ epoch 0; chỉ encoder RGB ImageNet pretrained. 1.600 train / 400 val, ID và raw drive disjoint, 1.000 anonymous test không public GT. GT/sparse PNG uint16 /256, depth theo mét, valid GT **0.1 < D < 120**. Metric là global pixel accumulation, không average per-image RMSE.

Drive dùng selected_2000_ids.json, kitti_trainval_2000.tar, metric_coarse_train_2000.tar (**D_cm/C_cm**) và relative_teacher_2000_DA3MONO_LARGE.tar (**R_T/C_T**, RGB-only relative teacher). Cache train **1.600**, cache val **0** cho cả hai; teacher loss loại trừ GT và original sensor support. Không online teacher/DSINE/geometry-fused branch trong inference.

| Objective component | Weight / rule |
|---|---|
| GT multi-scale Huber | D16/D8/D4/D2/D1/Dfull: 0.025/0.05/0.15/0.30/0.50/1 |
| Final metric RMSE / range RMSE | 0.6 / 0.1 |
| Direct final iRMSE, km⁻¹ | 0.04, ramp 4 epochs |
| Sparse / held-out sparse / sensor trust BCE | 0.02 / 0.05 / 0.02 |
| GT-boundary 3px RMSE / barrier BCE | 0.05 / 0.01 |
| Excess-tail Huber | 0.25; threshold 2 m, delta 10 m; ramp 2 epochs |
| Trajectory Huber at t=0.5 / t=1 | 0.05 / 0.05 |
| Metric KD | Nominal 0.012→0.006 over 40 epochs; confidence ≥0.5 |
| Relative normalized-gradient KD | 0.015; ramp 3 epochs; confidence ≥0.35 |
| Old inverse-Huber/log/edge/teacher-edge | Off; không h/stop loss |

AdamW LR 3e−4, encoder×0.5, weight decay1e−5, warmup1→cosine, clipping1; B4, seed42, train flip + sparse holdout10%, freeze encoder BN. Max40; early-stop min20/patience8/min_delta0.001 m. **Cả hai run thực tế dừng sau 34 epochs / 13.600 updates.** Best V11 epoch29 được giữ dù gain sau significant-best epoch25 nhỏ hơn min_delta và không reset patience; đây là hành vi đúng của early stop, không mất checkpoint tốt nhất.

## 5. Kết quả đã đo

Nguồn: [curated evidence](../results/benchmarks/v10_1_v11_pair01/README.md), source/checkpoint SHA đã kiểm tra. A100-SXM4-80GB, Torch2.11+cu130, B1, native BF16 CNN + FP32 geometry, 352×1216; **real100 wall timing, tuần tự sau train**. Epoch là index từ 0.

| Model / solver | RMSE m ↓ | iRMSE km⁻¹ ↓ | Median / P95 ms ↓ | NFE |
|---|---:|---:|---:|---:|
| V10.1 best-RMSE, epoch25 | 0.993694 | 3.180862 | 35.12 / 38.80 | 2 |
| **V11 best-RMSE, epoch29** | **0.986021** | 3.215515 | 74.15 / 76.47 | 13 |
| V11 same epoch29, static midpoint | 0.985972 | 3.215281 | 50.33 / 51.80 | 8 |

V11 primary MAE=0.256946 m, iMAE=1.041083 km⁻¹, AbsRel=0.013359, δ1=0.996696. V11 best-iRMSE epoch33: **RMSE=0.987694 / iRMSE=3.182093**, không ghép với RMSE epoch29. Target RMSE<0.9/iRMSE<3.2 đồng thời **chưa đạt**.

| GT region | V10.1 RMSE m | V11 RMSE m | V11 nhận xét |
|---|---:|---:|---|
| 0–20 m | 0.4478 | 0.4664 | Kém hơn |
| 20–40 m | 1.3718 | 1.3614 | Tốt hơn nhẹ |
| 40–60 m | 2.5269 | 2.4609 | Tốt hơn |
| 60–80 m | 3.8652 | 3.8045 | Tốt hơn |
| 80–120 m | 9.2028 | 8.7575 | Tốt hơn, chỉ 7.986 GT pixels |
| RGB edge | 1.3978 | 1.3898 | Tốt hơn nhẹ |
| GT boundary 3px | 1.6867 | 1.6678 | Tốt hơn |

Gain RMSE **7.67 mm / 0.77%**, metric SSE giảm1.54%, nhưng primary iRMSE tăng0.03465 km⁻¹ và adaptive latency tăng **2.11×**. Registered MAC: V10.1 3.2869G, V11 adaptive3.8879G, static3.6139G; chưa tính analytic/memory traffic. Instrumented synthetic dynamics timing: **7.55→45.51 ms**, là bottleneck tăng lớn nhất; đây là separate component pass, không cộng với real100 latency. Peak PyTorch allocated primary ~108.5 MiB, không phải toàn GPU memory footprint.

## 6. Insight và việc nên làm tiếp

1. **Giữ V11 làm research baseline, V10.1 làm runtime control.** Far range/boundary tốt hơn là tín hiệu hữu ích, nhưng một seed, fresh init khác thứ tự, chart + solver + horizon khác nhau: không kết luận NODE/adaptivity một mình tạo gain. Không tăng width/NFE chỉ vì metric đã plateau.
2. **Static midpoint là win triển khai rõ nhất hiện có:** latency giảm **32.12%**, global RMSE lệch adaptive chỉ −0.0000495 m. Không có nghĩa output pixel-wise giống nhau. Audit ít NFE hơn trên cùng checkpoint trước khi train lại; giữ adaptive research path riêng.
3. **RMSE bị rare large-error tail chi phối:** chỉ **0.176%** pixels có |error|>10 m nhưng đóng **57.20% SSE**. Nếu phần còn lại giữ nguyên, giảm ~29.2% SSE của tail này đủ đưa RMSE về0.9 — phép tính điều kiện, không dự báo training. GT-boundary3px: 13.16% pixels/37.65% SSE; ưu tiên inspect occlusion, mixed-surface query và sensor conflict. Các tail threshold lồng nhau, không cộng tỷ lệ.
4. **iRMSE cần bảo vệ near range:** 0–5 m chỉ1.50% GT pixels nhưng29.80% inverse SSE; primary không có prediction<0.1 m, không còn bằng chứng near-zero collapse. Inspect near-boundary/outliers thay vì clamp metric hoặc tăng inverse loss mù.
5. **Dynamics không đảm bảo D4 monotone:** V11 D0→half→final native RMSE=1.8727→1.6685→1.6784 m; cùng support quarter-grid, final MAE vẫn giảm0.4724→0.4440. D2 consensus giảm RMSE1.3224→1.2617 m, mạnh hơn V10.1 query gain. Đây là dấu hiệu geometry/readout phối hợp, không chứng minh bỏ t=1 tốt hơn final output. Metric giữa các scale có support khác.
6. **Giữ soft sensor reliability:** V11 pre-fusion0.9995→soft0.9860, hard diagnostic1.1264 m. Không quay lại hard anchor. Teacher metric/relative weighted loss tại best epoch lần lượt **1.12% / 0.10% total**; magnitude nhỏ chưa đủ kết luận teacher vô dụng — cần matched ablation trước khi xóa.

Numerical tolerance không phải bound của metric depth error; solver-compute trade-off phải đo thực nghiệm, phù hợp cảnh báo trong [Opening the Blackbox, ICML2021](https://proceedings.mlr.press/v139/pal21a.html). Chưa có independent test GT, multiple-seed evidence hoặc edge hardware benchmark.

## 7. Chạy lại và source of truth

[Notebook riêng V11](../drive_upload/AnchorFlow_v11_NODE/Train_AnchorFlow_V11_NODE_TAR2000_Fresh40.ipynb) hoặc [paired notebook](../notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb). Upload nguyên folder AnchorFlow_v11_NODE vào MyDrive, dùng data trong GeoLift_Data; output về GeoLift_RT_Runs. Đổi source/config cần version/run mới; resume chỉ cùng protocol.

| Source trong drive_upload/AnchorFlow_v11_NODE/ | Vai trò |
|---|---|
| geometry.py, ode_solver.py | Bounded chart, shared field, genuine numerical integration |
| geometry_primitives.py | Minmod, analytic translation, five-jet query |
| model.py, model_base.py, core.py, support.py | Encoder/decoder/phase readout/soft sensor fusion |
| losses.py, relative_loss.py, loss_helpers.py | GT-first dual-teacher objective |
| data.py, relative_data.py, run.py | Split/cache gates, train/resume/val/test/profile/export |
| config.json, bundle_manifest.json | Frozen default và byte checksum |

Không thay đổi model/config trong lần cập nhật tài liệu này. CSV/JSON, solver audit và SHA của paired run được publish nhỏ gọn cùng code; checkpoint/data/ZIP lớn tiếp tục giữ local/Drive.
