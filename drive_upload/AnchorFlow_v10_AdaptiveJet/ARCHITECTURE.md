# AnchorFlow V10 — Budgeted Residual-Controlled Adaptive Jet Integrator

Implemented specification · fresh student · max40epochs · two cached train-only teachers.
Đọc [BASELINE_COMPARISON.md](BASELINE_COMPARISON.md) trước khi diễn giải novelty/gain.

## 1. Flow và tensor contract

~~~text
RGB ── MobileNetV4 small pretrained ── F4/F8/F16 + F32 context
Sparse S,M,K ── compact sparse pyramid ───────┘
                     │ gated fusion / decoder
                     ▼
               D16 → D8 → D0 (1/4)
                     │
      minmod seed j0 = [v,gx,gy,hxx,hxy,hyy]
                     │
       Shared reaction–transport field Fθ
       ┌─────────────────────────────────────────────┐
       │ F(j0,τ0) → learned h1 → project → j1       │
       │ F(j1,τ1) → learned h2 → project → j2       │
       │                         stop? ─ yes ─┐     │
       │                          no          │     │
       │ F(j2,τ2) → learned h3 → project → j3 │     │
       │                         stop? ─ yes ─┤     │
       │                          no          │     │
       │ F(j3,τ3) → learned h4 → project → j4 ┤     │
       └──────────────────────────────────────┼─────┘
                                              ▼
                                  terminal geometric state j*
                                              │
                       D4 / phase upsample → D2_base
                                              │
       five-jet inverse consensus + uncertainty gate
              + tiny sparse-innovation metric correction
                                              │
                         D2 → phase upsample → D1/detail
                                              │
                      learned sensor reliability fusion → Dfull
~~~

| Tensor | Channels | Resolution at352×1216 |
|---|---:|---|
| RGB F4/F8/F16 | 16/32/48 | 88×304 / 44×152 / 22×76 |
| Sparse feature scales | 16/16/16 | 88×304 / 44×152 / 22×76 |
| P4 / dynamics context / G4 | 48/32/32 | 88×304 |
| Geometric state j | 6 | 88×304 |
| Controller statistics | 10/sample | scalar reductions |
| Learned h / stop probability | 1/sample each | no per-pixel step map |
| D2 / D1 / Dfull | 1 | 176×608 / 352×1216 / 352×1216 |

H,W chia hết32 theo encoder/F32 path. Full tensor state thuộc
R^(6×H4×W4), không phải chỉ một vector6D toàn ảnh. Mỗi pixel có một jet6D.
Input forward chỉ RGB/S/M/K, không teacher/GT. Geometric operations/loss/controller
FP32; CNN dùng native BF16 hoặc all-FP32 trên T4.

## 2. State và shared field — đúng nghĩa dynamics

Inverse depth v=1/D và jet tại một cell:

$$
j_p=[v_p,g_{x,p},g_{y,p},h_{xx,p},h_{xy,p},h_{yy,p}].
$$

Quadratic query quanh cell (s,t):

$$
v_p(s,t)=v_p+g_{x,p}s+g_{y,p}t
+\tfrac12h_{xx,p}s^2+h_{xy,p}st+\tfrac12h_{yy,p}t^2.
$$

Translation T đổi gốc chính xác cho polynomial bậc2. Field được tính **lại từ state
hiện tại và pseudo-time thật**, với cùng neural weights ở mọi call:

$$
\frac{dj_p}{d\tau}
=F_{\theta,p}(j,Z,S_4,M_4,\tau)
=R_{\theta,p}
+\sum_{q\in\mathcal N_4(p)}
w_{pq}\big[T_{q\to p}(j_q)-j_p\big].
$$

Reaction gồm bounded learned6-channel forcing + reliability-weighted sparse source.
Conductance và depth compatibility refresh theo j; barrier dùng context cố định.
State-input có14channels: normalized jet6, value displacement1, metric innovation1,
inverse innovation1, valid/density/spread/reliability4, actual time1.
Shared field không có BN; context CNN tính một lần.

Projection giữ nguyên V9.1:

$$
1/120\le v\le10,\qquad
|g_x|,|g_y|\le0.5v,\qquad
|h_{xx}|,|h_{xy}|,|h_{yy}|\le0.25v.
$$

Giữ forcing limits[0.9,0.15,0.15,0.06,0.06,0.06] và conductance cap0.6;
không tăng amplitude/width/teacher trong V10.

## 3. Budgeted learned Euler integrator

$$
j^{k+1}=P\big(j^k+h_{k+1}F_\theta(j^k,Z,S_4,M_4,\tau_k)\big),
\qquad \tau_{k+1}=\tau_k+h_{k+1},\quad\tau_0=0.
$$

$$
h_{k+1}=\frac16+\frac16\sigma(a_k),\qquad
N\in\{2,3,4\},\qquad T=\sum_{k=1}^{N}h_k\le\frac43.
$$

Không ép T=1. Controller shared MLP10→16→16→2, SiLU, **482params**.
Bias step4 → h≈0.330336; bias stop−2 → initially conservative4calls.
Không warm-start student hay phase-disable branch; controller/field học từ epoch0.

Mười statistics per-sample:

| # | Signal |
|---|---|
| 1/2 | Sparse inverse innovation absmean/RMS, e=M4(S4^-1−v)/v |
| 3 | Sparse support fraction |
| 4/5 | RMS value/derivative change normalized by previous v; initial zero |
| 6/7 | RMS current reaction / transport contribution normalized by v |
| 8 | Mean neighbor conductance |
| 9 | Mean barrier |
| 10 | Actual accumulated pseudo-time |

Statistics detached khỏi geometry graph; h vẫn nhận gradient qua Euler update.
Controller và state/reductions FP32. RMS dùng sqrt(mean-square+1e-8)−1e-4 để finite derivative
tại zero, empty sparse trả zero innovation. Không percentile/sort/pixel controller.

**Causality:** tính field→pre-state stats→h→update; sau update dùng post-state innovation/change/time,
nhưng **reuse** reaction/transport diagnostics từ call vừa thực hiện để dự đoán stop.
Không nhìn j(k+1) tương lai trước khi quyết định, không thêm field evaluation cho stop.
Step1/2 bắt buộc; dừng sau2 nếu p2≥0.5, nếu không dừng sau3 nếu p3≥0.5, còn lại4.

## 4. Train objective và readout exposure

Loss nền giữ V9.1:

| Term | Coefficient / rule |
|---|---|
| GT metric multi-scale Huber | D16/D8/D4/D2/D1/Dfull:0.025/0.05/0.15/0.30/0.50/1 |
| Final batch RMSE / range RMSE | 0.4 / 0.1 |
| Log / log-gradient edge | 0.2 / 0.05 |
| Sparse / held-out sparse / reliability BCE | 0.02 / 0.05 / 0.02 |
| GT-boundary3px RMSE / barrier BCE | 0.05 / 0.01 |
| Robust excess-tail Huber | 0.25, threshold2m, delta10m, ramp2epochs; normalize all valid GT |
| Metric KD / teacher log-edge | 0.012→0.006 over40epochs / 0.01; confidence≥0.5, exclude GT/sparse cells |
| GT inverse auxiliary | 0.01; D4/D2/Dfull weights0.25/0.5/1; Huber of100×inverse error |
| Relative normalized-gradient KD | 0.015, ramp3epochs; confidence≥0.35, offsets1/4; no ordinal |
| Dynamics trajectory auxiliary | 0.1 total; weights[0.15,0.25,0.30,0.30] sum1 |
| New stop-controller BCE | 0.02 |
| Compute penalty / h-smoothness | Off by default |

Train luôn evaluate4field calls, nhưng high-resolution readout chạy **một lần**:
chọn hard terminal j2/j3/j4, không convex-blend jets/pseudo-times.
20% sample train chọn uniform2/3/4 để readout được exposure cả3exit; còn80% dùng detached hard policy.
Validation/test không exploration. RNG được checkpoint để resume đúng.
Auxiliary4state giữ tổng0.1, không nhân loss4lần. B0 fixed3 giữ auxiliary cũ0.5Lstep1+0.5Lstep2.

Cho mỗi sample train, Jk là valid-area-mean GT Huber tại D4_stepk:

$$
B_k=\frac{J_k-J_{k+1}}{\max(J_k,10^{-6})},\qquad
y_k^{\rm stop}=\mathbf1[B_k\le0.003],\qquad k\in\{2,3\}.
$$

$$
L_{\rm stop}=\tfrac12\sum_{k=2}^{3}{\rm BCEWithLogits}(s_k,y_k^{\rm stop}).
$$

Targets/benefit detached, GT chỉ dùng trong loss. Sample không GT support không góp BCE.
Jk chỉ dùng metric Huber; không thêm inverse loss thứ hai (inverse auxiliary nền đã có).
Future benefit quarter-grid là **proxy**, không đảm bảo final RMSE monotone; vì vậy phải policy-audit
final2/3/4 cùng checkpoint, tuyệt đối không dùng GT oracle stop ở inference.

## 5. NODE: giữ nguyên lý nào, không claim gì?

V10 giữ neural derivative field, state feedback, actual time, shared weights, Euler integration
và task-conditioned variable terminal horizon. Gần NODE về cấu trúc dynamics, không chỉ CNN
predict một correction rồi fixed transport.

Nhưng đây là **projected, budgeted, learned task integrator**, không black-box adaptive ODE solver:
không RK45/local truncation-error estimator, reject step, adjoint hoặc torchdiffeq.
Stop được train theo GT task benefit, không numerical tolerance. NFE hard cap4 và BPTT unroll4,
không claim constant-memory adjoint, exactODE, numerical convergence order của toàn NN,
asymptotic convergence hay “stability envelope được bảo đảm”. h≤1/3 chỉ hạn chế single-step
magnitude; nonlinear field/projection/variable horizon vẫn phải kiểm tra thực nghiệm.

Nguồn nguyên lý: [Neural ODE, NeurIPS2018](https://arxiv.org/abs/1806.07366) cho neural vector field
và integration; [PonderNet2021](https://arxiv.org/abs/2107.05407) cho ý tưởng learned computation budget
(V10 không implement PonderNet's probabilistic objective); [Opening the Blackbox, ICML2021](https://proceedings.mlr.press/v139/pal21a.html)
cho tầm quan trọng của solver heuristics/compute trade-off, không xem solver như detail vô hại.

Research claim dự kiến: **sparse-innovation/jet-evolution controlled bounded integration of a
learned reaction–transport surface field**. Không claim first-ever adaptiveNN/NODE/PDE.
Novelty và gain cần matched ablation, multiple-seed và edge-device benchmark.

## 6. Deployment và audit contract

| Mode | Actual calls | Output | Có early-exit saving? |
|---|---:|---|---|
| Train | 4 | Selected/explored j2/3/4 | Không |
| Validation masked_adaptive | 4 | Hard-selected j2/3/4 | Không |
| adaptive_batch1 eager | 2–4 | Same hard-selected state | Có thể; đo cả host sync |
| ONNX masked_adaptive | 4 | Same selected output, no Python branch | Không |
| static4 fallback | 4 | j4 output | Không; khác policy output |

Không gọi static4 graph “adaptive compute export”. ONNX17 checker + CPU parity tolerance0.01m/
rtol1e-4. Chưa build TensorRT; branch-native execution là công việc deployment tiếp theo.
Profile main total median/P95 + component eager; policy_profiles dùng real validation scenes,
wall median/P95 gồm host sync, executedNFE distribution, peakVRAM, instrumented field/controller
timing/branch-host-wait. Controller timing là nested trong dynamics, không cộng lần nữa.

Model **582,350params**, thêm482 so V9.1; all-four learned rollout **3.368789312G Conv/Linear MAC**
(+1.653% so V9.1), chưa tính analytic operations/reductions/memory traffic. Không gán fixed GPU
latency khi chưa đo V10. NFE thấp không tự đảm bảo tổng runtime thấp.

Mỗi val log h1…h4 mean/std (cả những call đã execute trong masked rollout), terminal T,
selected/executedNFE mean, p50/p95, exit2/3/4, stop accuracy/target fraction/future benefit,
D0/step1…4 native RMSE/MAE/iRMSE/tails, final global/range/RGB-edge/GT-boundary/tail,
query/innovation diagnostics. Stage targets ở khác scale không cùng support.

Stop threshold0.5 được freeze trước train. Post-hoc calibration nếu làm phải báo riêng và
không xem cùng400val như independent test. Anonymous test1000 không có public GT.

## 7. Source of truth

| File | Responsibility |
|---|---|
| model.py | Controller, shared time-conditioned field, projective Euler, hard budget/selection |
| model_baseline.py / model_v9_reference.py / model_v8.py | Chosen V9.1 readout / V9 consensus / V8 primitives |
| losses.py / losses_baseline.py / losses_v8.py | New trajectory/stop loss + unchanged baseline objectives |
| run.py | Fresh40, strict resume, data/smoke/train/val/test/profile/export/policy_audit |
| adaptive_report.py | Integration/trajectory diagnostics |
| config.json | Frozen default fresh40 parameters |
| test_contracts.py / local_verification.json | Verification evidence and limitations |

Teacher generation/cache recipe, raw archive schema, depth units and split không đổi.
Một bug implementation ở bản copy loss nền được sửa: inverse accumulation dùng out-of-place,
không mutate zero placeholder của teacher-disabled/ramp statistics. Objective dual-teacher
không đổi về toán học; test thêm xác nhận các term disabled bằng0 và ramp đúng1.
