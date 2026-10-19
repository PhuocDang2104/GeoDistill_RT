# AnchorFlow V11 — Jet NODE · RK3 6 + RK2 2 NFE

> Historical contract archived19/10/2026, NOT current deployment baseline.
> Current: [S4-U-PIR NoRelative, coarse4/fine0](../AnchorFlow_V11_Baseline.md).
> Metric attribution below retains its historical scope.

**Đặc tả canonical: V11.3 + coarse RK3 6 NFE + fine RK2 2 NFE.**
Cập nhật 14/10/2026 · đối chiếu production source · không thay model/config.

V11 học cách **phát triển hình học từ state hiện tại**, không chỉ dự đoán một residual duy nhất.
RGB cung cấp ngữ cảnh/boundary, sparse depth cung cấp metric scale; hai neural vector fields
refine inverse-depth surface tại D4 và D2. **Model học vector field; solver tích phân field**
đến horizon cố định, không learned step size hoặc learned stop.

> **Trạng thái:** kiến trúc Fixed6 đã có code và contract tests. Accuracy/runtime đã đo
> bên dưới thuộc adaptive parent hoặc S6 midpoint; **chưa phải benchmark RK3/TensorRT**.

## 1. Flow và input/output

```text
RGB ── MobileNetV4 small pretrained ── F4/F8/F16 + F32 context
Sparse S, mask M, intrinsics K ── compact sparse pyramid ─────┘
                         │ gated fusion + lightweight decoder
                         ▼
                    D16 → D8 → D0 (1/4)
                         │ minmod seed → bounded chart z(0)
                         ▼
            Coarse Jet NODE: RK3 ×2, h=0.5, 6 NFE
                  z(0) → z(0.5) → z(1)
                         │ decode → D4 geometric
                         ▼
             Phase lift → D2_base → five-jet consensus
                         │ sparse metric innovation
                         ▼
                     D2_pre_fine
                         │ fine seed/context, separate learned field
                         ▼
              Fine Jet NODE: RK2 ×1, h=1, 2 NFE
                         │
                         D2
                         │ phase lift + phase context
                         ▼
                D1_base → detail → PIR → D1
                         │ learned sensor reliability fusion
                         ▼
                       Dfull
```

| Contract | Giá trị |
|---|---|
| Model class / parameters | `AnchorFlowEdge` / **584,845** |
| RGB encoder | `mobilenetv4_conv_small_050.e3000_r224_in1k` |
| Forward API | `model(rgb, sparse, mask, K)`; không GT/teacher |
| RGB / sparse / mask / K | `[B,3,H,W]` / `[B,1,H,W]` / `[B,1,H,W]` / `[B,3,3]` |
| RGB / depth units | RGB trong [0,1], normalize bằng ImageNet mean/std; depth theo mét |
| Mask / intrinsics | M=1 tại sparse sensor hợp lệ; K theo pixel coordinates sau crop/resize |
| Chuẩn KITTI | H×W = **352×1216**, H/W chia hết 32 |
| D4 / D2 / output grids | 88×304 / 176×608 / 352×1216 |
| RGB F4/F8/F16 | 16 / 32 / 48 channels |
| Sparse branches / fused decoder | 16/16/16 channels / 48/64/96 tại F4/F8/F16 |
| P4 / coarse context / fine context | 48 / 32 / 12 channels |
| Deployment output | `D_full`, `[B,1,H,W]`; giá trị depth, không ảnh màu |
| Precision | Main CNN BF16; chart, RHS, solver, sparse statistics FP32; fine NODE FP32 |

**NFE chỉ đếm lần gọi vector field.** Encoder/context/readout không chạy lại mỗi NFE.
Coarse và fine có weights riêng; trong từng NODE, weights được dùng lại ở mọi call.
Tổng **6+2=8 NFE** tại hai grids khác nhau, không phải 8 layers cùng chi phí.
Train forward trả dictionary multi-stage; deployment wrapper chỉ trả tensor depth.
Forward loại sparse không finite hoặc ngoài (0.1,120) m khỏi sensor mask.

## 2. Encoder và coarse metric depth

Sparse depth không concat vào RGB input. Pyramid pool numerator/support cùng nhau:

$$
\mu_s=\frac{\mathrm{Pool}_s(SM)}{\max(\mathrm{Pool}_s(M),\epsilon)},
\qquad \rho_s=\mathrm{Pool}_s(M).
$$

Sparse state 8 kênh gồm mean depth, valid mask, density, local 7×7 prior,
prior support, relative standard deviation và camera coordinates x/y từ K.
Các projection nhỏ sinh sparse features; gated fusion cộng vào RGB features:

$$
F_s=\mathrm{LiteBlock}\left(A_s+\sigma(G_s([A_s,B_s]))\odot B_s\right).
$$

Decoder dùng positive coarse D16, rồi bounded log-depth residual ở D8/D0.
D0 blend thêm local sparse prior theo support/gate; không hard-anchor ở từng scale.

## 3. Jet: một local surface cho mỗi cell

Mỗi cell giữ sáu hệ số của polynomial inverse-depth bậc hai:

$$
j_p=[v_p,g_{x,p},g_{y,p},h_{xx,p},h_{xy,p},h_{yy,p}],\qquad v_p=1/D_p,
$$

$$
v_p(s,t)=v_p+g_{x,p}s+g_{y,p}t
+\frac{1}{2}h_{xx,p}s^2+h_{xy,p}st+\frac{1}{2}h_{yy,p}t^2.
$$

s/t là offsets theo **grid cell**, không world coordinates. Pseudo-time τ là tiến trình
refinement, không thời gian chuyển động vật lý. Jet không phải displacement 3D.
Minmod tạo seed slope/Hessian từ D0; derivative seed detach, value vẫn nhận gradient.

Solver tích phân latent chart z, không clip/project jet sau từng bước:

$$
v_p=\exp\left[\log(1/120)+\log(1200)\sigma(z_{p,0})\right],
\qquad j_{p,i}=v_p\kappa_i\tanh(z_{p,i}),\quad i=1,\ldots,5,
$$

với κ = [1, 0.5, 0.5, 0.25, 0.25, 0.25]; chỉ số 0 là **value channel**,
không phải pseudo-time. Finite z giữ center depth trong 0.1–120 m,
|slope|≤0.5v và |Hessian|≤0.25v. Model không enforce global integrability
giữa mọi cell; bounds không chứng minh accuracy/stability.

## 4. Dynamic field: reaction + sparse source + transport

Ở **mỗi RHS call**, model decode current state, refresh sparse innovation và
neighbor compatibility; context RGB/sparse được tính một lần:

$$
A_p=R_{\theta,p}+Q_{\theta,p}
+\sum_{q\in\mathcal N_4(p)}w_{pq}\left[T_{q\to p}(j_q)-j_p\right],
$$

$$
\frac{dz_p}{d\tau}=f_{\theta,p}(\tau,z;Z,S,M)
=2\tanh\left(\frac{A_p}{2v_p\kappa}\right).
$$

- **R:** bounded learned reaction, điều chỉnh geometry theo context/state.
- **Q:** reliability-weighted sparse inverse-depth innovation, tác động value channel.
- **Transport:** hấp thụ geometry từ bốn neighbor sau khi đổi gốc polynomial.
- **w:** learned conductance × boundary barrier × depth compatibility; zero flux ở biên grid.

Nếu đổi gốc một khoảng (dx,dy), analytic translation giữ nguyên local quadratic surface:

$$
v'=v+g_xdx+g_ydy+\frac{1}{2}h_{xx}dx^2+h_{xy}dxdy+\frac{1}{2}h_{yy}dy^2,
$$

$$
g_x'=g_x+h_{xx}dx+h_{xy}dy,\qquad
g_y'=g_y+h_{xy}dx+h_{yy}dy,\qquad H'=H.
$$

RHS được định nghĩa **trong chart space**. A không được claim là exact physical
dj/dτ; normalization không phải Jacobian pullback chính xác. Đây là learned
reaction–transport dynamics, không PINN có physics residual loss.

## 5. Coarse RK3: đúng 6 NFE

Hai bước classic Kutta RK3, h=0.5, từ τ=0→0.5→1:

$$
k_1=f_\theta(\tau_n,z_n),\qquad
k_2=f_\theta(\tau_n+h/2,z_n+(h/2)k_1),
$$

$$
k_3=f_\theta(\tau_n+h,z_n-hk_1+2hk_2),\qquad
z_{n+1}=z_n+\frac{h}{6}(k_1+4k_2+k_3).
$$

RHS calls tại τ = **0, 0.25, 0.5, 0.5, 0.75, 1**. Hai call tại 0.5 dùng
**states khác nhau**; không được reuse chỉ vì time giống nhau. z(0.5) là state
sau bước RK3 đầu, không dense interpolation. Fixed mesh, không reject/adaptive h,
không learned horizon; direct autograd khi train, unroll tĩnh khi deploy.

**Không nhầm các bản cùng tên “6 NFE”:**

| Variant | Coarse solve | Fine | Vai trò |
|---|---|---:|---|
| **Fixed6 canonical** | RK3 ×2, h=[0.5,0.5] | 2 NFE | Production contract tài liệu này |
| S6 benchmark | Midpoint RK2 ×3, h=[0.25,0.25,0.5] | 2 NFE | Accuracy/runtime audit cùng checkpoint |
| Historical adaptive V11.3 | Bosh3 RK3(2), tolerance-controlled | 2 NFE | Parent accuracy/reference |

Cùng NFE không có nghĩa cùng output. Không đổi solver rồi resume optimizer dưới
protocol cũ; tạo run/version mới và ghi rõ solver.

## 6. D4→D2→Dfull

Five-jet consensus query center/E/W/S/N tại bốn subpixel phases
(s,t)∈{−0.25,+0.25}²; tổng hợp **inverse depth trước**, sau đó reciprocal:

$$
\widehat v_p(s,t)=\sum_{q\in\mathcal N_5(p)}
\alpha_{pq}(s,t)\,\mathrm{clip}\left([T_{q\to p}(j_q)](s,t),1/120,10\right),
\qquad \alpha_{pq}\ge0,\quad \sum_q\alpha_{pq}=1.
$$

α là softmax learned logits trừ depth/boundary penalty; mask loại neighbor ngoài grid.
Readout blend query depth với phase-lift depth qua uncertainty gate và bounded
correction, rồi thêm sparse metric innovation tại D2; không average metric depth.

Fine NODE seed lại jet từ D2_pre_fine trên half-grid, dùng context 12 channels
và field riêng **981 parameters**. Một bước midpoint RK2 h=1:

$$
k_1^f=f_f(0,z_2^0),\qquad
k_2^f=f_f(0.5,z_2^0+0.5k_1^f),\qquad z_2^1=z_2^0+k_2^f.
$$

Đúng **2 NFE**; midpoint internal stage không phải exact solution tại t=0.5.
Fine NODE sửa depth center; slope/Hessian tham gia dynamics/transport, không được
claim là một fine polynomial query mới dùng để dựng D1.

Phase lift/PixelShuffle dựng D1 từ half-grid features. PIR (**952 parameters**)
phase-pack sparse donor error, mean/std/density và log-depth contrast; tiny PW/DW
head sinh bounded metric correction. Không thêm full-resolution learned feature CNN.

$$
\Delta D=(0.5+0.05D_1^0)\odot\mathrm{PixelShuffle}[\tanh(a)\odot\sigma(b)],
\qquad D_1=\mathrm{clip}(D_1^0+\Delta D,0.1,120).
$$

Output dùng learned sensor reliability, không hard-anchor bắt buộc:

$$
\begin{aligned}
\lambda&=(0.5+0.02D_1)\exp(\mathrm{clip}(a_\lambda,-1,2)),\\
g&=M\odot\sigma\left(\ell-\log\left[1+((S-D_1)/\lambda)^2\right]\right),\\
D_{\mathrm{full}}&=(1-g)D_1+gS.
\end{aligned}
$$

Tolerance λ theo mét; aλ là learned scalar. Sensor conflict lớn làm giảm gate;
hard-anchor chỉ là diagnostic.

## 7. Training và đánh giá

Production Fixed6: fresh student, encoder ImageNet pretrained; full KITTI
**85,898 train / 6,852 val / 1,000 anonymous test**. Cached DMD3C++ metric
`D_cm/C_cm` và DA3MONO-LARGE relative `R_T/C_T` chỉ dùng train. Không online
teacher khi inference; teacher KD loại trừ GT và original sparse support.

| Recipe full KITTI hiện tại | Giá trị |
|---|---|
| Optimizer / LR | AdamW; 3×10⁻⁴, encoder ×0.5; weight decay 10⁻⁵ |
| Schedule / stabilization | Warm-up 1 epoch → cosine, minimum LR ×0.05; clipping 1; encoder BN frozen |
| Data / execution | B2, seed42, flip + sparse holdout10%; BF16, channels-last, fused AdamW |
| Budget / selection | Max40; dual-policy early stop, min20/patience8/min_delta0.0002 |

GT-first objective; trọng số **thực tế sau adapter + loss wrapper**, không lấy
trực tiếp nominal fields của historical config:

| Term | Weight / rule |
|---|---|
| GT metric Huber, δ=1 m | D16/D8/D4/D2/D1/Dfull: 0.025/0.05/0.15/0.30/0.50/1 |
| Final RMSE / range RMSE | 0.6 / 0.1 |
| Final inverse RMSE, km⁻¹ | 0.04→0.05 trong 2 epoch đầu |
| Sparse / held-out / trust BCE | 0.02 / 0.05 / 0.02 |
| GT-boundary 3px / barrier BCE | 0.05 / 0.01 |
| Excess-tail | 0.25→0.30 trong 2 epoch; threshold2 m, Huber δ=10 m |
| Coarse trajectory at τ=0.5/1 | Tổng 0.10→0.05 trong 2 epoch; chia đều hai observations |
| Metric KD | 0.006→0.004 trong 40 epoch; confidence≥0.5 |
| Relative normalized-gradient KD | 0.015; ramp 2 epoch, confidence≥0.35 |
| Legacy inverse-Huber/log/edge/teacher-edge | Tắt |

GT downsample bằng valid-area mean; các scale loss cộng trực tiếp, không chia
tổng weights. Training RMSE dùng smooth square-root để gradient finite tại zero;
evaluation dùng global pixel sums chuẩn dưới đây.

**Policy:** khi iRMSE≤3.2, minimize RMSE; trước đó minimize
max(RMSE/0.98, iRMSE/3.2). Best checkpoint lưu theo raw policy improvement;
patience dùng significant improvement riêng. Recipe fresh full KITTI này kế thừa
loss schedules V11.3, **không phải warm-start/resume FT15 subset**.

Global pixel metrics, depth theo mét, inverse metric theo km⁻¹:

$$
\mathrm{RMSE}=\sqrt{\frac1N\sum_{i\in\mathcal V}(D_i-D_i^{gt})^2},\qquad
\mathrm{iRMSE}=\sqrt{\frac1N\sum_{i\in\mathcal V}
\left(\frac{1000}{D_i}-\frac{1000}{D_i^{gt}}\right)^2}.
$$

N=|V|; GT valid: mask hợp lệ và 0.1 < Dgt < 120. Anonymous test không public GT. Báo cùng checkpoint:
global/range/edge/GT-boundary/error-tail, không chỉ một RMSE.

### Kết quả đã có — không gộp khác solver/protocol

| Evidence đã có | RMSE m | iRMSE km⁻¹ | Latency | Phạm vi |
|---|---:|---:|---|---|
| V11.3 adaptive best_policy, subset | 0.986517 | 3.169735 | Tùy run/solver | FT epoch9, cumulative32; internal400 |
| S6 **midpoint**, cached A100 | 0.986601 | 3.165399 | median51.44 ms | Log người dùng; GPU-resident B1 |
| S6 **midpoint**, CUDA Graph A100 | 0.986601 | 3.165399 | median8.84/P95 8.86 ms | Cùng log; không loading/H2D |
| **RK3 Fixed6 + fine2** | — | — | **Chưa đo** | Chưa có full400 audit tương ứng |
| TensorRT + CUDA Graph Fixed6 | Chưa đo | Chưa đo | Chưa đo | Deployment path cần qualification |

S6 graph giảm latency82.8% vs cached trong log, không thay learned parameters.
Đây không phải TensorRT benchmark, không phải end-to-end hoặc Jetson result.
Accuracy parent/S6 **chưa đạt RMSE<0.98** dù iRMSE<3.2.
Không dùng validation subset để claim full-KITTI/leaderboard accuracy.

## 8. TensorRT + CUDA Graph: hai tầng tối ưu khác nhau

```text
Frozen RK3 + fine2 PyTorch deployment graph
       ↓ ONNX/static tensor graph + correctness gate
TensorRT engine: tactic selection, supported-op/layer fusion, precision policy
       ↓ fixed shape + persistent device buffers + warm-up
CUDA Graph capture enqueueV3
       ↓ copy new inputs → replay → consume current output
```

**TensorRT** tối ưu cách thực thi operators; **CUDA Graph** giảm CPU dispatch/
launch overhead. Graph không tự giảm NFE/MAC và không mặc định fuse kernels.
Tổ hợp hữu ích cho graph tĩnh, nhưng gain không được nhân hai speedup một cách máy móc.
[NVIDIA TensorRT performance guide](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/performance/optimization.html).

Qualification tối thiểu:

1. Strict-load checkpoint; configure `rk3`, fixed_steps=2, coarse6/fine2, T=1.
   Export B1/H352/W1216, chỉ Dfull, unroll RK stages, tắt diagnostics; kiểm ONNX parity.
2. Giữ chart/RHS/solver/fine NODE FP32, CNN BF16 khi được hỗ trợ;
   dùng casts/precision policy đúng TensorRT version và kiểm layer types sau build.
3. Buffers có địa chỉ cố định, warm-up sau shape/profile change; capture `enqueueV3`
   trên non-default stream. Không CPU sync/copy/allocation từ host bên trong capture.
4. Replay inputs mới; kiểm A→B→A, đổi sparse support và K; consume/clone output
   trước replay tiếp. Parity stage + full400 với **PyTorch RK3 cùng precision/TF32**.
5. Đo eager / TensorRT / TensorRT+graph cùng GPU/B1/input set/warmup.
   Báo median/P95, memory; tách loading, preprocessing, H2D và end-to-end.
6. Qualify lại trên edge device; ghi checkpoint SHA, GPU/CUDA/TRT version.
   Không giả định engine A100 portable hoặc speedup giữ nguyên trên 5060/Jetson.

PyTorch graph runner hiện có trong inference package; **repo chưa có bằng chứng
TensorRT engine Fixed6 đã build và pass parity**. Đây là specification triển khai,
không phải cam kết runtime. Production source còn `new_tensor` trong forward;
cần capture-safe copy riêng, không sửa byte-locked training snapshot để resume.
[PyTorch CUDA Graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

## 9. Source of truth và chạy production

| Phần | File trong `production/AnchorFlow_V11_3_Fixed6_FullKITTI/` |
|---|---|
| Recipe/adapter/NFE gates | `config.json`, `adapter.py` |
| Coarse RK3 | `v11_model/ode_solver.py` |
| Chart/RHS/translation | `v11_model/geometry.py`, `geometry_primitives.py` |
| Fine RK2 | `v11_model/fine_geometry.py` |
| Decoder/phase/detail/PIR | `model_base.py`, `model_parent.py`, `phase_readout.py`, `model.py` trong v11_model |
| Data/teachers/train/test | `full_data.py`, `teachers.py`, `teacher_cache.py`, `trainer.py`, `pipeline.py` |
| Packaging/evidence | `bundle_manifest.json`, `verification.json` |

```bash
bash production/AnchorFlow_V11_3_Fixed6_FullKITTI/run_full_flow.sh \
  --storage-root /mnt/ssd/geolift \
  --accept-kitti-license \
  --cuda-arch 12.0
```

Đọc KITTI terms trước flag license; GPU/member prerequisites trong
[production runbook](../AnchorFlow_V11_3_Fixed6_Production.md).
Historical accuracy: [V11.3 research baseline](../AnchorFlow_V11_3.md).
Storage archive không thay đổi weights/config/source của frozen V11 packages.

Kiểm chứng 14/10/2026: `pipeline.py verify` PASS; **16/16 CPU production contract
tests** PASS, gồm RK3 schedule/NFE/gradient, model adapter, cache và resume.
Không thay thế GPU/teacher smoke, TensorRT parity hoặc edge-device qualification.

```bash
cd production/AnchorFlow_V11_3_Fixed6_FullKITTI
python pipeline.py verify
python -m unittest test_production -v
```
