# AnchorFlow V11 — Solver-Controlled Jet NODE

**Implemented specification · fresh student · max 40 epochs + early stop · two cached teachers**

V11 học **đạo hàm của một geometric state**, không học step-size hoặc quyết định dừng theo GT.
Một numerical solver độc lập tích phân state đến cùng thời điểm **T = 1**.
Đây là bản research để kiểm tra NODE thực sự; không mặc định nhanh hơn V10.

## 1. Chốt kiến trúc

~~~text
RGB ── ImageNet MobileNetV4 small ── F4/F8/F16 + F32 context
Sparse S, M, K ── compact sparse pyramid ─────────────┘
                              │
                      gated fusion + decoder
                              │
                         D16 → D8 → D0
                              │
                  minmod geometric jet at 1/4
                              │
                   encode bounded chart → z(0)
                              │
     ┌───────────────────────────────────────────────────┐
     │  dz/dt = Fθ(t, z, cached RGB context, sparse)      │
     │                                                   │
     │  torchdiffeq bosh3 — embedded RK3(2)               │
     │  estimate error → accept/reject → select next h    │
     │  recompute SAME field at each RK stage            │
     │  integrate until fixed T = 1                      │
     └───────────────────────────────────────────────────┘
                              │
                    smooth decode j(1) → D4
                              │
           phase lift + five-jet inverse consensus
            uncertainty gate + sparse innovation → D2
                              │
          context-guided final phase lift + detail → D1
                              │
                 learned sensor reliability → Dfull
~~~

| Contract | Canonical value |
|---|---|
| Forward inputs | RGB, sparse depth, validity mask, intrinsics; no GT/teacher |
| Input shape | RGB: B×3×352×1216; S/M: B×1×352×1216 |
| Encoder | mobilenetv4_conv_small_050.e3000_r224_in1k, ImageNet initialization |
| RGB features | 16/32/48 channels at 1/4, 1/8, 1/16; F32 context retained |
| Sparse features | 16/16/16 channels; valid mean/density/local prior/spread/camera rays |
| Decoder P4 / ODE context / G4 | 48 / 32 / 32 channels, 88×304 |
| ODE state | B×6×88×304: one six-coordinate state per cell |
| Observation times | t = 0.5 and t = 1, independent of accepted solver-step count |
| Trainable parameters | **582,912**; no learned h/stop head |
| Main solver | torchdiffeq 0.2.5, bosh3; direct autograd through executed solve |
| Precision | Native BF16 main CNN if supported; **FP32 shared RHS/state/solver/loss** |
| Depth domain | 0.1–120 m, unchanged evaluation/output convention |

Backbone, fusion, readout, teacher roles và objective giữ từ V10.1 LiteMetric.
H/W chia hết cho 32. Không thêm transformer, full-resolution learned feature hoặc grid_sample.

## 2. Vì sao không tiếp tục learned h?

Learned h không vô nghĩa nói chung: nó có thể là learned computation policy.
Nhưng ở V10 nó đồng thời thay đổi thời gian kết thúc, nên không phải bộ điều khiển sai số ODE chuẩn.
Run V10 đã quan sát h gần 1/3 và toàn bộ 400 validation chọn exit2; chưa chứng minh policy phân biệt scene.

| V10 / V10.1 | V11 |
|---|---|
| NN sinh h từ feature/statistics | h do numerical solver chọn; **không có tham số trainable** |
| V10 stop theo future GT task benefit | Không GT-stop, không stop loss |
| Terminal horizon phụ thuộc h/exit | Mọi sample đều đến **T = 1** |
| Projected explicit Euler | Embedded RK3(2), không projection trong rollout |
| “3–4 NFE” budget thấp | Không ép 3–4 NFE rồi giả vờ đáp ứng tolerance |
| Task-conditioned integration policy | Neural field + independent numerical integration |

Ước lượng Euler dùng chênh lệch field của hai bước liên tiếp chỉ là **lagged proxy**.
Nó không tương đương embedded RK với accept/reject. Giới hạn budget rồi buộc một bước lớn
để đạt T cũng không bảo đảm tolerance. V11 không dùng hai shortcut đó.

## 3. Geometric jet và smooth bounded chart

Jet inverse-depth trên quarter-grid:

$$
j=[v,g_x,g_y,h_{xx},h_{xy},h_{yy}], \qquad v=1/D.
$$

Query hình học tại offset (s,t):

$$
v(s,t)=v+g_xs+g_yt+\frac12h_{xx}s^2+h_{xy}st+\frac12h_{yy}t^2.
$$

ODE không tích phân jet rồi clip sau từng bước. Nó tích phân tọa độ **z** không bị projection:

$$
v(z)=\exp\left(\log(1/120)+\log(1200)\,\sigma(z_0)\right),
$$

$$
g_x=0.5v\tanh(z_1), \qquad g_y=0.5v\tanh(z_2),
$$

$$
h_{xx}=0.25v\tanh(z_3), \qquad
h_{xy}=0.25v\tanh(z_4), \qquad
h_{yy}=0.25v\tanh(z_5).
$$

Vì vậy mọi finite z giải mã thành positive bounded inverse-depth, gradient/Hessian có giới hạn.
Không cần gián đoạn state bằng projection tại các RK stages.

Khởi tạo: v từ D0; minmod seed gradient/Hessian; map ngược bằng logit/atanh.
Chỉ lúc khởi tạo dùng epsilon 1e−6 ở inverse-depth chart và ratio ±0.9999 ở derivative chart
để tránh tọa độ vô hạn. Đây không phải clamp dùng để “sửa” metric iRMSE.

**Giới hạn:** sigmoid/tanh có thể saturate; bounded representation không chứng minh dynamics
không stiff hoặc gradient luôn tốt. Log NFE, error tails và chất lượng near-range để kiểm tra.

## 4. Shared neural vector field

Context CNN tính một lần. Mỗi RHS call giải mã **state hiện tại**, dùng **thời gian thực hiện tại**,
refresh reaction, conductance và sparse innovation với cùng neural weights.

Physical-jet proposal:

$$
A_p=R_{\theta,p}+
\sum_{q\in\mathcal N_4(p)}
w_{pq,\theta}(j,Z,S,M)
\left[T_{q\to p}(j_q)-j_p\right].
$$

T là phép đổi gốc chính xác của quadratic polynomial. Neighbor edge rates đối xứng,
zero-flux tại biên; depth compatibility chặn vận chuyển qua surface không tương thích.
Reaction gồm bounded six-channel forcing và soft sparse source; không hard-anchor giữa rollout.

ODE **được định nghĩa trong chart space**:

$$
\frac{dz_{p,c}}{dt}
=F_{\theta,p,c}(t,z,Z,S,M)
=2\tanh\left(\frac{A_{p,c}}{2v_p\kappa_c}\right),
$$

$$
\kappa=[1,0.5,0.5,0.25,0.25,0.25].
$$

Đây là normalized bounded reaction–transport forcing cho z. **Không** tuyên bố nó là exact
Jacobian pullback của projected physical-jet field cũ. Physical trajectory thực sự là
j(t)=decode(z(t)); derivative của j phải tính qua chain rule của decoder.
Ví dụ:

$$
\frac{dv}{dt}
=v\,\log(1200)\,\sigma(z_0)\big(1-\sigma(z_0)\big)\frac{dz_0}{dt}.
$$

State-input head vẫn 14 channels: normalized jet6, value ratio1, metric innovation1,
inverse innovation1, valid/density/spread/reliability4, time1.
RHS không có BN; chỉ context CNN có BN. FP32 RHS tránh BF16 quantization noise chi phối error estimator.

## 5. Solver gần NODE theo đúng nghĩa số học

Initial-value problem:

$$
\frac{dz}{dt}=F_\theta(t,z,Z,S,M), \qquad
z(0)=z_{\mathrm{seed}}, \qquad
z(1)=\mathrm{ODESolve}(F_\theta,z(0),0,1).
$$

Default dùng official **Bogacki–Shampine RK3(2)**. Tại một trial:

$$
k_1=F(t,z), \quad
k_2=F(t+h/2,z+hk_1/2), \quad
k_3=F(t+3h/4,z+3hk_2/4),
$$

$$
z^{(3)}=z+h\left(\frac29k_1+\frac13k_2+\frac49k_3\right),
\qquad k_4=F(t+h,z^{(3)}),
$$

$$
z^{(2)}=z+h\left(\frac7{24}k_1+\frac14k_2+\frac13k_3+\frac18k_4\right),
\qquad e=z^{(3)}-z^{(2)}.
$$

Sai số được chuẩn hóa bằng atol + rtol·max(abs(z),abs(z³)).
Norm là **maximum sample/channel spatial RMS**, không để sample khó bị dilute bởi sample dễ.
Solver accept khi normalized error ≤ 1; nếu reject thì state/time không advance.
Rule safety 0.9 và error exponent 1/3 thuộc numerical solver, không phải neural head.

| Numerical parameter | Default |
|---|---:|
| rtol / atol | 0.01 / 0.001, trên dimensionless chart coordinates |
| First / maximum h | 0.25 / 0.25 |
| Terminal T | 1, không learned |
| Prescribed endpoints | 0.5 và 1, không overshoot terminal horizon |
| Solver trial guard | 64 mỗi observation interval |
| Global RHS guard | 193 actual NFE mỗi forward, tính cả rejected trial |
| Guard exceeded | **Raise rõ ràng**; không forced acceptance/fallback/tolerance change |

FSAL reuse k4 làm k1 kế tiếp: NFE = 1 + 3 × số trial nếu không phát sinh jump.
Rollout thuận lợi với bốn bước h=0.25 thường là **13 NFE**, không phải “4 calls”.
Field khó hơn có thể cần nhiều calls/rejects. Training B4 dùng một grid chung theo worst norm;
validation B1 có thể dùng grid khác. NFE/latency vì vậy phụ thuộc state, batch và tolerance.

Tightened tolerance kiểm tra **numerical sensitivity**, không bảo đảm GT RMSE tốt hơn.
Tolerance không phải đơn vị m và không tương đương bound cho final depth error.

## 6. Train, teachers và objective

Fresh decoder/field/readout từ epoch0; chỉ RGB encoder có ImageNet initialization.
Không load checkpoint V8/V9/V10. Resume chỉ nhận checkpoint **V11 cùng source/protocol**.
Max40 epochs, early stop min20/patience8/min_delta0.001 m.
AdamW 3e−4, encoder LR×0.5, weight decay1e−5, warmup1→cosine, gradient clip1.
B4, accumulation1, train horizontal flip, sparse holdout10%, freeze encoder BN.

| Loss / signal | Weight |
|---|---|
| GT Huber D16/D8/D4/D2/D1/Dfull | 0.025 / 0.05 / 0.15 / 0.30 / 0.50 / 1 |
| Full-output metric RMSE / range RMSE | 0.6 / 0.1 |
| Direct full-output iRMSE, km⁻¹ | 0.04, ramp4epochs |
| Sparse / held-out sparse / sensor reliability | 0.02 / 0.05 / 0.02 |
| GT-boundary3px RMSE / barrier BCE | 0.05 / 0.01 |
| Excess-tail robust Huber | 0.25, threshold2m, delta10m, ramp2epochs |
| Fixed-time trajectory GT Huber | 0.1 total = 0.05 at t=.5 + 0.05 at t=1 |
| Metric KD | 0.012→0.006 across40epochs, confidence≥0.5 |
| Relative normalized-gradient KD | 0.015, ramp3epochs, confidence≥0.35 |
| Old inverse-Huber / log / edge / teacher-edge | 0, không computed mặc định |
| Learned h/stop/controller loss | **Không tồn tại** |

Metric KD và relative KD loại trừ public GT/sensor support theo flow hiện có.
Metric TAR chứa D_cm/C_cm; relative TAR chứa R_T/C_T, RGB-only DA3MONO-LARGE cache đã audit.
Teacher chỉ là privileged training targets; không đi vào field hoặc inference.
Trajectory targets là pooled valid-area-mean GT; không phải GT-guided solver decisions.

## 7. NODE claim và giới hạn

V11 có learned derivative, current-state feedback, time conditioning, shared weights,
fixed-horizon IVP và genuine embedded numerical error control.
Nó là **Neural ODE trong bounded geometric chart**, thay vì learned residual scheduler.

Không dùng adjoint: direct differentiation qua solver để kiểm chứng gradient đơn giản hơn.
Không claim constant-memory, physical PDE constraints/PINN, exact physics, monotone GT improvement,
independent-test success hoặc first-ever NODE depth completion.
Sigmoid/tanh bounds không chứng minh numerical convergence của toàn model ở tolerance đã chọn.
V11 thay **state chart + solver**, nên không phải pure solver-only ablation với V10.1.

Nguồn:
[Neural ODE, NeurIPS 2018](https://arxiv.org/abs/1806.07366),
[torchdiffeq official solvers](https://github.com/rtqichen/torchdiffeq),
[RK3(2) tableau](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/bosh3.py),
[package pinned 0.2.5](https://pypi.org/project/torchdiffeq/0.2.5/),
[Opening the Blackbox, ICML 2021](https://proceedings.mlr.press/v139/pal21a.html).

## 8. Edge path: tách research accuracy khỏi deployment

| Mode | Numerical method | Calls | Export |
|---|---|---|---|
| Train / main val / anonymous test | Adaptive RK3(2), same T=1 | Variable, thường ≥13 | Eager Python solver |
| Same-checkpoint tighter reference | Adaptive RK3(2), rtol.002/atol.0002 | Thường nhiều hơn; guard769 | Numerical audit |
| Static edge candidate | Four midpoint steps h=.25 | **8 NFE** | ONNX17, no adaptive branches |

Fixed midpoint dùng **cùng field weights/chart/horizon**, nhưng là approximate solver khác:
phải đánh giá full400 val, không mặc định output bằng adaptive solver.
ONNX parity chỉ so với PyTorch **fixed midpoint**, không chứng minh parity với adaptive RK3(2).
Notebook chạy solver_audit cả3 trên cùng best checkpoint trước khi dùng fixed graph.

Không trace adaptive Python decisions rồi gọi đó là adaptive ONNX.
Không ép accuracy-first solver vào 3–4NFE nếu muốn error control thật.
NFE guard là safety guard, **không phải hard edge latency budget**.

## 9. Đánh giá và acceptance

Structural profiling trên CPU, fresh synthetic input 352×1216:

| Graph thực sự được đếm | Actual NFE của sample | Registered Conv/Linear MAC |
|---|---:|---:|
| Adaptive RK3(2) | 13 | 3.887881536 G |
| Fixed midpoint4steps | 8 | 3.613941056 G |

Adaptive MAC thay đổi theo actual NFE, không phải hằng số của mọi scene/checkpoint.
Hai graph có cùng582,912parameters. Counter không đếm analytic transport/chart,
softmax/interpolation/pooling hoặc memory traffic; **không** quy MAC này thành milliseconds.
Ghi nhận trong local_verification.json; GPU latency và trained accuracy chưa được đo.

Primary: global valid-pixel RMSE, không mean per-image RMSE.
iRMSE = sqrt(mean([1000/Dpred − 1000/Dgt]²)) trên cùng valid support.
Validation400 dự kiến **25,424,992 GT pixels**, protocol không đổi.
Anonymous1000 không có public GT; ZIP prediction dùng để submit benchmark.

Log global/range0–20/20–40/40–60/60–80/80–120, RGB-edge, GT-boundary1/2/3/5/10px,
near-inverse0–5/5–10/10–20, tails1/2/5/10/20m và native-stage metrics.
Solver log actual NFE mean/P50/P95/max/histogram, accepted/rejected steps, accepted h,
terminal-time error. D4_step1/2 là t=.5/1, **không** accepted step1/2.

Chọn best-RMSE, best-iRMSE và best-joint riêng; phải báo metric cùng checkpoint.
Best-joint score = max(RMSE/0.9, iRMSE/3.2); score<1 mới đồng thời đạt hai target.
GPU profiler báo adaptive/fixed synthetic + real100 wall median/P95, peak VRAM, precision,
NFE và components; wall timing gồm solver host decisions. MAC không thay thế latency.

**V11 chưa train**. Không cam kết RMSE<0.9 hoặc iRMSE<3.2.
Nếu NODE không cải thiện metric mà latency tăng, giữ bản simpler model cho deployment;
nguyên lý gần NODE không tự chứng minh lợi ích depth completion.

## 10. Source of truth

| File | Responsibility |
|---|---|
| geometry.py | Smooth chart, shared reaction–transport RHS, fixed-time observation |
| ode_solver.py | Official adaptive solve, counters/guards, static midpoint |
| geometry_primitives.py | Minmod, polynomial transport, fixed neighborhood query |
| model.py / model_base.py / core.py / support.py | Encoder, decoder, unchanged LiteMetric readout |
| losses.py / relative_loss.py / loss_helpers.py | GT-first dual-teacher training, no solver-target loss |
| data.py / relative_data.py | Audited Drive TAR, train-only teacher cache, locked split |
| run.py | Prepare/smoke/train/evaluate/solver_audit/test/profile/export, strict resume |
| test_contracts.py / local_verification.json | Numerical/gradient/runtime/export verification evidence |
| Train_AnchorFlow_V11_NODE_TAR2000_Fresh40.ipynb | Single Colab workflow |
