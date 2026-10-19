# AnchorFlow V11 baseline — S4 · U-PIR · NoRelative

**Chốt deployment baseline 19/10/2026: `G_s4_upir_norel`, `best_policy.pth`, epoch27
(đếm từ0; epoch28/30).** Đây là V11-derived S4, **không phải** RK3 coarse6 + fine2.
Giữ nguyên snapshot đã train; chọn theo accuracy/runtime trade-off, không claim
thắng mọi metric hoặc đã chứng minh riêng đóng góp U-PIR.

## 1. Kiến trúc và input/output

```text
RGB ─ MobileNetV4-small ImageNet ─┐
Sparse S, mask M, K ─ pyramid ───┴─ gated decoder → D16 → D8 → D0 (1/4)
  → inverse-depth jet → bounded chart → coarse Jet NODE S4, T=1
  → D4_geo → phase lift + five-jet consensus + sparse innovation → D2
                         └ relative variance U2 ──────────────────────┐
  → phase lift/detail → existing PIR + bounded U2 cue → sensor fusion → D_full
```

**583.880 parameters; coarse4 / fine0 NFE; không fine NODE/CNN bổ sung.**
D2=`D2_pre_fine`. U-PIR mở rộng đúng một projection40→41 inputs,
hidden16/output8 không đổi, thêm16 weights. Encoder:
`mobilenetv4_conv_small_050.e3000_r224_in1k`.

| Tensor | Inference contract |
|---|---|
| RGB | FP32 B×3×352×1216, RGB [0,1], không ImageNet-normalize thêm ở caller |
| S, M | FP32 B×1×352×1216; sparse depth m, mask0/1, missing depth0 |
| K | FP32 B×3×3; intrinsics đã điều chỉnh theo crop/resize |
| D_full | FP32 B×1×352×1216; metric depth m, domain [0,1;120] |

Export static **B=1**. D4:88×304; D2:176×608. Không GT/teacher/cache trong forward.
Sparse input là sensor, không ground truth.

## 2. Jet dynamics: model học field, solver tích phân

Jet biểu diễn Taylor surface inverse-depth cục bộ, không phải6 depth maps độc lập:

\[
j=[v,g_x,g_y,h_{xx},h_{xy},h_{yy}],\quad
q_j(\delta)=v+g_x\delta_x+g_y\delta_y+
\tfrac12h_{xx}\delta_x^2+h_{xy}\delta_x\delta_y+\tfrac12h_{yy}\delta_y^2.
\]

Offset tính theo ô D4. Seed từ inverse-depth D0 và minmod finite differences.
Bounded chart giữ state hợp lệ, không hard-project mỗi solver step:

\[
v=\exp[\log(1/120)+\log(1200)\sigma(z_0)],\quad
j_{1:5}=v\,(.5,.5,.25,.25,.25)\odot\tanh(z_{1:5}).
\]

RHS gồm learned reaction, sparse inverse-depth innovation và analytic Taylor
transport từ4 ô lân cận, với boundary/depth-aware conductance:

\[
\dot z=f_\theta(\tau,z;\mathrm{RGB},S,M,K)
=2\tanh\!\left[\frac{A_{\rm reaction}+A_{\rm sensor}+A_{\rm transport}}
{2v\kappa}\right],\quad\kappa=(1,.5,.5,.25,.25,.25).
\]

Đây là ODE **định nghĩa trong chart**, không exact Jacobian pullback của field cũ.
Coefficients local học được không có bảo đảm integrability toàn cục; không phải physical-time PINN.

**S4 = hai explicit-midpoint RK2 steps**, h=0,5, horizon1:

\[
k_1=f(\tau_n,z_n),\quad k_2=f(\tau_n+h/2,z_n+hk_1/2),\quad
z_{n+1}=z_n+hk_2,\quad n=0,1.
\]

RHS gọi tại τ=0;0,25;0,5;0,75 → **4 NFE**, cùng field weights; không RK4,
không adaptive stop, learned step size hay fine solver.

## 3. U-PIR thêm gì?

Five-jet consensus đọc center/east/west/south/north tại4 subpixel phases.
Với inverse-depth proposals q và normalized learned weights α:

\[
\bar q=\sum_{a=1}^{5}\alpha_aq_a,\quad
U=\frac{\sum_a\alpha_a(q_a-\bar q)^2}{\max(\bar q^2,10^{-6})},\quad
c(U)=\frac{\max(U,0)}{\max(U,0)+0.002}.
\]

U pixel-shuffle thành1 channel D2: **mức bất đồng của local jets**, không phải
calibrated xác suất sai. Không thêm field/network để tính U. PIR cũ dùng24 hidden
+16 same-phase sparse-innovation/contrast features; U-PIR thêm c(U) vào projection hiện có:

\[
h=\operatorname{SiLU}\!\left(\operatorname{DWConv}_{3\times3}
\left[\operatorname{SiLU}(W_xx+w_uc(U)+b)\right]\right),\quad
[a,b_g]=W_oh+b_o,
\]
\[
\Delta D=(0.5+0.05D)\odot\operatorname{PixelShuffle}_2
\left[\tanh(a)\odot\sigma(b_g)\right].
\]

Residual cộng lên detail depth rồi clamp và soft sensor fusion. U có thể ảnh
hưởng sign/amplitude lẫn gate; **không** chỉ nhân beta vào depth.
Không có JPIR `beta` trong model này. Cột w_u khởi tạo0 giữ parity ban đầu.

## 4. Train và kết quả đã đo

Fresh30, seed42/B4;1.600train/400val; ImageNet encoder; AdamW3e−4,
encoder LR×0,5, warmup1+cosine. GT-first multi-scale + metric teacher KD
(.012→.006); inverse/boundary/tail/dynamics/sensor objectives giữ nguyên.
**Relative OFF** (`relative_enabled=false`, `relative_weight=0`), không đọc relative archive.
Train native BF16 CNN + FP32 geometry/loss, **không train bằng TensorRT**.
Không early stop; `last.pth` resume epoch-boundary với optimizer/RNG và strict
source/config/data contract. Deploy dùng `best_policy`, không `last`.

**A100-SXM4-40GB · TensorRT10.13.3 FP32 + CUDA Graph · full400 /25.424.992 GT pixels.**
Wall gồm D2D, không đọc ảnh/preprocess/H2D. A–E relative ON; E chỉ26epochs,
nên không phải matched-loss controls của G.

| Model | RMSE m ↓ | iRMSE km⁻¹ ↓ | MAE m ↓ | Boundary3 RMSE m ↓ | Median/P95 ms ↓ |
|---|---:|---:|---:|---:|---:|
| A: S4 + fine RK2 |0.991838|3.185364|0.258894|**1.683244**|4.843/4.860|
| B: S4 + residual CNN |0.990902|3.174048|**0.258379**|1.684816|4.030/4.050|
| C: S4 + phase CNN |0.992158|3.198326|0.259127|1.688851|3.947/3.964|
| D: S4 no-fine |0.992871|3.188647|0.258913|1.686430|3.830/3.884|
| E: JPIR26 |0.994212|3.243492|0.260845|1.684237|**3.815**/3.859|
| **G: U-PIR NoRelative30** |**0.990151**|**3.172074**|0.258984|1.685668|3.827/**3.845**|

G: **261,3 FPS**, AbsRel0,013387; δ₁99,6648%; iMAE1,047601km⁻¹.
Versus D: RMSE giảm2,72mm (0,274%), iRMSE giảm0,520%; MAE tăng0,071mm;
latency chênh0,003ms, **chưa có bằng chứng speedup đáng kể**. Versus A latency
giảm20,99%. RMSE<0,98 chưa đạt; iRMSE<3,2 đã đạt.

**U giúp checkpoint này; chưa isolate lợi ích kiến trúc khi retrain.** Audit full400
đã sửa eval-mode được user báo cáo: original0,990151/3,172073/0,258984;
tắt riêng U-input0,990889/3,191544/0,261689 (RMSE/iRMSE/MAE). Mean prediction
change3,67cm là khác biệt hai dự đoán, không phải giảm lỗi3,67cm. D→G vẫn đổi
cả U-conditioning lẫn relative loss. Cần **D-NoRelative fresh30** cùng seed/recipe
để isolate architecture. Một seed, validation lặp nhiều lần là development evidence,
không independent confirmation/leaderboard; không kết luận relative gây hại.

Best epoch27, first iRMSE-feasible epoch24; relative terms=0 toàn30epochs.
Epoch29 RMSE0,990432 không hơn best. Train≈35,7phút, tổng epoch≈50,1phút trong
run này, không phải estimate Orin. Chỉ0,178% pixels có |error|>10m nhưng chiếm
57,12% SSE: bottleneck còn ở rare large errors, không chỉ average smoothness.

Strict latency target cũ≤3,815386ms **FAIL** (+0,298%);
`all_targets_met=false` giữ nguyên. Chốt G không sửa gate cho pass.
MSG-CHN pretrained fullKITTI khác training/overlap chưa audit, không dùng để
quy architectural advantage. [Raw evidence/analysis](../benchmarks/s4_upir_norel_2026_10_19/README.md).

## 5. TensorRT + CUDA Graph và Orin

TensorRT compile/fuse graph tĩnh, chọn kernels; CUDA Graph replay dependency
graph, giảm CPU/launch overhead. **Không giảm NFE/đổi phương trình**. Warm enqueue
trước capture, non-default stream, fixed device addresses; D2D staging đổi ảnh.
A/B/A chống stale output; finite/pixel/global-metric gates phải pass.

A100 runtime100real images, warmup100/iterations300, TF32 OFF. GPU-event,
host-enqueue và synchronized-wall báo riêng. Host enqueue0,121ms **không** phải
model latency. Torch memory908MiB trong log chứa nhiều backends/inputs, chưa gồm
direct TRT allocations, **không** isolated model memory.

Orin NX/AGX dùng [inference-only package](../production/AnchorFlow_S4_UPIR_Orin/README.md):
FP32 ONNX17 + real inputs/references từ Colab, build engine native TensorRT10,
parity rồi benchmark eager/graph. **Không mang A100 engine sang Orin**, không suy
latency A100 sang Jetson; phải ghi RAM/power mode/thermal. Orin chưa được đo.
[JetPack6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62) có CUDA12.6/TRT10.3;
không cài CUDA13/TRT10.13 A100 lên Jetson.

## 6. Tái lập

- [Frozen train folder/notebook](../drive_upload/AnchorFlow_S4_UPIR_NoRelative_30ep_A100/README.md): upload nguyên folder, không đổi `.py` khi resume.
- Drive: `GeoLift_RT_Runs/S4_UPIR_NoRelative_Fresh30_pair01/G_s4_upir_norel/best_policy.pth`.
- Orin chỉ cần ONNX + manifest + samples +2 runner scripts, không checkpoint/teacher/trainer.
- [Fixed6 RK3+fine2 cũ](history/AnchorFlow_V11_Fixed6_2026_10_14.md),
  [V11.3 adaptive](AnchorFlow_V11_3.md) là historical controls; không đổi nhãn metric.
