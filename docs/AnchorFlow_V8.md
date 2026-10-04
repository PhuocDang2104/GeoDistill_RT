# AnchorFlow V8 — Feedback Jet Dynamics

> **Implementation và kết quả đã train · cập nhật 2026-10-04.**  
> RGB + sparse depth → depth metric. Teacher chỉ dùng khi train, không có teacher inference.  
> Best checkpoint epoch 21 · validation RMSE **0.997549 m** · 578.064 parameters.

## 1. Ý tưởng và kiến trúc

V8 giữ một **state hình học 6 chiều**. Sau mỗi bước, neural block đọc lại state và sai lệch sparse hiện tại rồi cập nhật bằng cùng bộ trọng số. Không chỉ predict correction một lần rồi propagate cố định.

~~~mermaid
flowchart TD
    RGB[RGB] --> ENC[MobileNetV4 pretrained<br/>F4 / F8 / F16 / F32]
    SM[Sparse S + mask M + K] --> SP[Compact sparse pyramid]
    ENC --> FPN[F32 context + gated fusion + LiteFPN<br/>P4 48 / P8 64 / P16 96 channels]
    SP --> FPN
    FPN --> BASE[Coarse decoder<br/>D16 → D8 → D0 at 1/4]
    RGB --> GUIDE[Learned phase guidance<br/>G2 8ch → phase-pack G4 32ch]
    FPN --> CTX[Cached context Z32]
    GUIDE --> CTX
    SP --> CTX
    BASE --> J0[Base inverse-depth jet j0]
    J0 --> STEP1[Shared JetDynamics<br/>j0 → j1]
    CTX --> STEP1
    STEP1 --> STEP2[Same weights + new state feedback<br/>j1 → j2]
    CTX --> STEP2
    STEP2 --> STEP3[Same weights + new state feedback<br/>j2 → j3]
    CTX --> STEP3
    STEP3 --> D4[D4 + continuous quadratic query]
    D4 --> D2[Phase upsample + gated geometry + metric detail<br/>D2]
    GUIDE --> D2
    D2 --> D1[Phase upsample + detail / trust head<br/>D1]
    D1 --> FUSION[Learned sensor reliability fusion]
    SM --> FUSION
    FUSION --> OUT[Dfull — output chính]
~~~

| Khối | Chức năng |
|---|---|
| RGB encoder | MobileNetV4 Conv Small 0.50 ImageNet pretrained; fine-tune weights, freeze encoder BatchNorm |
| Sparse pyramid | Valid pooling, local prior, density/spread, ray channels từ K; không concat sparse ở input RGB |
| Fusion + FPN | Gated RGB/sparse đa scale; F32 bổ sung context rộng |
| Coarse decoder | D16 dương qua softplus; bounded log-depth residual lifts tạo D8 và D0 |
| Phase guidance | PixelUnshuffle RGB → CNN nhẹ tại 1/2 → phase-pack xuống 1/4 |
| JetDynamics | 3 shared updates tại 1/4; reaction và conductance phụ thuộc state hiện tại |
| Readout / detail | Query hình học + CNN phase upsample tạo D2, rồi D1 |
| Sensor fusion | Học mức tin sparse từng pixel; không hard-anchor output chính |

Input: RGB **[B,3,352,1216]**, S/M **[B,1,352,1216]**, K **[B,3,3]**. Depth mét, model bound 0.1–120 m. Quarter-grid 88×304; half-grid 176×608. Không learned feature CNN full-resolution, adaptive ODE solver, ConvGRU, attention, grid_sample, matrix NxN hay global linear solver.

## 2. Công thức cốt lõi

### 2.1. State: quadratic inverse-depth jet

Tại quarter-grid pixel p, inverse depth v = 1/D:

$$
j_p=[v_p,g_{x,p},g_{y,p},h_{xx,p},h_{xy,p},h_{yy,p}]^T.
$$

v là inverse depth; g mô tả độ nghiêng; H mô tả độ cong. Derivative theo **tọa độ pixel quarter-grid**, không phải vận tốc 3D hoặc displacement ảnh. j0 lấy v từ D0 differentiable; gradient/Hessian khởi tạo bằng finite differences từ v detached, hạn chế derivative qua boundary. Sau update, derivative channels là state học được; chưa ràng buộc integrability toàn ảnh.

### 2.2. Dynamic update: reaction + transport

$$
j_p^{k+1}=P\bigl(j_p^k+\Delta\tau F_{\theta,p}(j^k,Z,S_4,M_4,\tau_k)\bigr),
\qquad \Delta\tau=\frac13,\quad \tau_k=\frac{k}{3},\quad k=0,1,2.
$$

$$
F_{\theta,p}=R_{\theta,p}^k+
\sum_{q\in\mathcal N_4(p)}w_{pq}^k\bigl(T_{q\to p}(j_q^k)-j_p^k\bigr).
$$

- **Reaction R:** sửa state local theo RGB/context và lỗi sparse hiện tại.
- **Transport:** nhận geometry từ bốn neighbor tương thích, sau khi đổi gốc tọa độ.
- **P:** chiếu về miền bounded. Z cache một lần; reaction/conductance/compatibility **tính lại mỗi bước từ current state**.

Neural field: state 14ch → PW32 + Z32 → DW3×3 → SiLU → PW32 → SiLU; head 1×1 sinh reaction 6ch, conductance 2ch và sensor gate 1ch. Không BatchNorm trong loop. Shared weights vẫn chạy đủ ba lần.

### 2.3. Sparse feedback và learned reaction

S4 là valid-area mean sparse; M4 là có support. Spread4 cao giảm độ tin mixed foreground/background measurements:

$$
r=M_4\exp(-4\,\mathrm{spread}_4),\qquad
e^k=M_4\frac{S_4^{-1}-v^k}{v^k},
$$

$$
R^k=v^k b\odot\tanh(h_R^k)
+[\sigma(h_S^k)\,r\,v^k\,\mathrm{clip}(e^k,-1,1),0,0,0,0,0]^T,
$$

$$
b=[0.9,0.15,0.15,0.06,0.06,0.06]^T.
$$

Code clamp S4 tối thiểu 0.1 trước reciprocal, mask vùng không support về zero. Sensor forcing chỉ thêm vào value component; không hard-anchor các bước trung gian.

### 2.4. Analytic transport: đổi gốc Taylor polynomial

Với a = x_p − x_q, c = y_p − y_q trên quarter-grid:

$$
\begin{aligned}
v'&=v+g_xa+g_yc+\tfrac12h_{xx}a^2+h_{xy}ac+\tfrac12h_{yy}c^2,\\
g_x'&=g_x+h_{xx}a+h_{xy}c,\\
g_y'&=g_y+h_{xy}a+h_{yy}c,\qquad H'=H.
\end{aligned}
$$

Hai learned undirected rates tạo bốn hướng reciprocal, zero flux ở biên. Rate kết hợp learned barrier với compatibility từ **depth mismatch hiện tại**. Mỗi weight ≤0.6; dt × tổng bốn weight ≤0.8.

Projection giới hạn v trong [1/120,10], mỗi gradient trong ±0.5v, mỗi Hessian entry trong ±0.25v. Geometry FP32, learned convolution AMP. Đây là **fixed-step projected geometric dynamics**; scalar mixing bound không chứng minh stability toàn jet, hội tụ hoặc accuracy tăng mỗi bước. Không gọi PINN, optical flow hay adaptive Neural ODE.

### 2.5. Continuous phase query và metric readout

State j3 dựng inverse depth tại bốn phase (s,t), mỗi tọa độ ±1/4 quarter-cell:

$$
\xi(s,t)=v+g_xs+g_yt+\tfrac12h_{xx}s^2+h_{xy}st+\tfrac12h_{yy}t^2.
$$

Reciprocal sau clamp và PixelShuffle tạo Q2 (D2_query). B2 (D2_base) từ learned neighborhood phase upsample. Readout:

$$
D_2=\mathrm{clip}\bigl(
B_2+g_Q\odot\mathrm{clip}(Q_2-B_2,-L,L)
+(1+0.05B_2)\odot\tanh(h_\Delta),\ 0.1,120\bigr),
\qquad L=1+0.1B_2.
$$

Gate/metric head dùng context + guidance + sensor phases tại quarter-grid, rồi PixelShuffle. D2→D1 tiếp tục phase upsample và bounded learned detail tại half-grid; chỉ scalar depth/fusion ở full-resolution.

### 2.6. Learned sensor fusion — không hard-anchor

$$
g_S=M\odot\sigma\bigl(\ell_S-\log(1+((S-D_1)/t)^2)\bigr),
\qquad
D_{\mathrm{full}}=(1-g_S)\odot D_1+g_S\odot S.
$$

t = (0.5 + 0.02D1) × exp(clipped learned log-tolerance). Trust head học confidence; disagreement giảm độ tin sparse. GT chỉ giám sát trust khi train, không vào forward. Dhard thay sparse trực tiếp chỉ là diagnostic.

## 3. Training và teacher

| Thiết lập | Run đã thực hiện |
|---|---|
| Data | Fixed 2.000 IDs: 1.600 train / 400 val, disjoint ID và raw drive; 1.000 anonymous test |
| Student initialization | Fresh từ epoch 0; chỉ RGB encoder ImageNet pretrained, không load V5/V6/V7 |
| Teacher | Cache D_cm/C_cm trong metric_coarse_train_2000.tar, train only |
| KD mask | Confidence ≥0.5; exclude GT + original sparse, kể cả holdout; coarse cell có forbidden pixel cũng bị loại |
| Optimizer | Fused AdamW; batch 4, accumulation 1; weight decay 1e-5; clip gradient 1 |
| LR | Decoder/dynamics 3e-4, encoder 1.5e-4; warm-up 1 epoch → cosine, min ratio 0.05 |
| Augmentation | Horizontal flip đồng bộ K; sparse holdout 10% |
| Precision thực tế | Epoch 0–4 FP16; từ epoch 5 BF16 recovery; geometry/reductions FP32 |
| Early stop | Max 30; patience 7, min_delta 0.001 m, minimum 15 completed epochs |
| Kết thúc | 29 epoch, index 0–28; best epoch 21, last epoch 28 |

Recovery giữ weights/optimizer/RNG/LR/early-stop, fork folder riêng; không phải run BF16 thuần từ đầu. FP16 encoder convolution từng overflow; cùng batch/state BF16 và FP32 hữu hạn. Không suy ra inference FP16 an toàn.

### Objective

$$
\mathcal L=\mathcal L_{\mathrm{GT}}+\sum_i\lambda_i\mathcal L_i,
\qquad
\mathcal L_{\mathrm{GT}}=
\sum_s w_s\,\mathrm{mean}_{p\in\mathcal V_s}\rho_1(D_s(p)-D_{\mathrm{GT},s}(p)).
$$

D16/D8/D4/D2/D1/Dfull weights: **0.025 / 0.05 / 0.15 / 0.30 / 0.50 / 1.00**. GT scale thấp là valid-area mean; không normalize tổng scale weights. Huber rho-delta: r²/2 khi |r|≤delta; delta × (|r|−delta/2) ngoài miền đó.

| Term | Weight | Weighted contribution epoch 21 |
| --- | --- | --- |
| Multi-scale GT Huber | Như trên | 0.278913 |
| Global RMSE | 0.25 | 0.210804 |
| Range-balanced RMSE | 0.10 | 0.210379 |
| Log-depth Huber | 0.20 | 0.000068 |
| Log-gradient edge | 0.05 | 0.000013 |
| Observed sparse Huber | 0.02 | 0.008156 |
| Held-out sparse Huber | 0.05 | 0.024450 |
| Sensor trust BCE | 0.02 | 0.007557 |
| Confidence-weighted metric KD | 0.05 → 0.025 | 0.036911 |
| Teacher log-gradient edge | 0.02 | 0.000009 |
| GT-boundary 3px RMSE | 0.05 | 0.084338 |
| Balanced barrier BCE | 0.01 | 0.002719 |
| Squared excess-tail >2 m | 0.10 × ramp | 0.041468 |
| D4 step1/2 auxiliary Huber mean | 0.10 | 0.023247 |
| Robust MSE diagnostic | 0, không dùng trong total | 0.000000 |
| **Total** |  | 0.929033 |

Tail = mean trên **tất cả valid GT pixel** của max(|error|−2,0)²; ramp 0→1 trong hai epoch đầu. Range loss weighted mean RMSE của bin đủ ≥64 pixel trong batch; weights các bin là **1 / 1 / 1 / 1 / 0.25**, nên bin 80–120 m chỉ có 1/4 vote. KD giảm tuyến tính theo horizon 30; D4/D2/D1 KD weights 0.25/0.50/0.25. Sensor loss downweight GT-conflicting measurements. RMSE trong objective dùng sqrt(MSE + 1e-6) − 0.001 để ổn định; evaluation dùng RMSE trực tiếp.

Tất cả modules bật từ epoch 0, không curriculum bật/tắt architecture. Hệ số loss không phải % contribution thực tế. Tại epoch 21: KD eligible coverage **71.23% ảnh train**, mean confidence eligible **0.7999**, sensor GT-conflict fraction **5.39%** trên observed sensor có GT. Đây là train diagnostics, không phải metric accuracy teacher hoặc validation. Các cột legacy connection/surface/static-jet bằng zero không có nghĩa V8 dynamics bị tắt.

D_cm/C_cm không đủ xác định cache chỉ do teacher nào sinh; cần metadata provenance.

## 4. Metric hiện tại — best checkpoint epoch 21

Nguồn: [val_metrics.json](../results/anchorflow_v8_completed_audit/val_metrics.json), [train_log.csv](../results/anchorflow_v8_completed_audit/train_log.csv). Đây là **validation nội bộ**, không phải KITTI leaderboard. Anonymous test không có public GT.

| Global metric | Giá trị |
| --- | --- |
| RMSE evaluate lại | **0.997549 m** |
| Best RMSE trong train log | 0.997516 m |
| MAE | 0.260227 m |
| iRMSE / iMAE | 3.394592 / 1.084010 km^-1 |
| AbsRel | 0.013660 |
| delta1 | 0.996436 = 99.6436% |
| GT support | 25.424.992 pixel |
| Valid GT predictions <0.1 m | 0 |

Primary RMSE là global valid-pixel metric, không average RMSE từng ảnh:

$$
\mathrm{RMSE}=\sqrt{\frac1N\sum_{i\in\mathcal V}(D_i-G_i)^2},
\qquad
\mathrm{iRMSE}=\sqrt{\frac1N\sum_{i\in\mathcal V}(1000/D_i-1000/G_i)^2}.
$$

MAE/iMAE thay root mean square bằng mean absolute error; AbsRel = mean(|D−G|/G). delta1 là tỷ lệ max(D/G,G/D)<1.25. Code hiện không log delta2/delta3. Chênh lệch train/evaluate chỉ 0.000032 m, cùng checkpoint.

### Range và RGB edge

| GT subset | Pixels | RMSE m | MAE m | iRMSE km^-1 | % tổng SSE |
| --- | --- | --- | --- | --- | --- |
| 0-20 | 19.685.953 | 0.457755 | 0.128806 | 3.676595 | 16.30 |
| 20-40 | 4.254.287 | 1.351756 | 0.509690 | 2.317528 | 30.73 |
| 40-60 | 1.120.771 | 2.564824 | 1.128843 | 1.700752 | 29.14 |
| 60-80 | 355.995 | 3.895736 | 1.692552 | 1.446711 | 21.35 |
| 80-120 | 7.986 | 8.855406 | 5.575342 | 2.245158 | 2.48 |
| edge | 4.669.922 | 1.408754 | 0.396695 | 4.196753 | 36.63 |
| non_edge | 20.755.070 | 0.878901 | 0.229522 | 3.186395 | 63.37 |

Edge là gradient RGB grayscale >0.05, khác GT-depth boundary. Range partition GT; edge/non-edge cũng partition GT nhưng chồng với range, không cộng hai nhóm bảng.

### Stage metrics và output policy

| Output | Grid | RMSE m | MAE m |
| --- | --- | --- | --- |
| D16 | 1/16 | 2.799709 | 1.021257 |
| D8 | 1/8 | 2.268219 | 0.820902 |
| D0 | 1/4 | 1.772792 | 0.594268 |
| D4_step1 | 1/4 | 1.661539 | 0.464266 |
| D4_step2 | 1/4 | 1.638747 | 0.430124 |
| D4_step3 | 1/4 | 1.641792 | 0.421826 |
| D2_base | 1/2 | 1.271287 | 0.359200 |
| D2_query | 1/2 | 1.349566 | 0.394865 |
| D2 | 1/2 | 1.221794 | 0.306328 |
| D1_base | Full | 1.035537 | 0.292081 |
| D1 | Full | 1.010101 | 0.272226 |
| D_full | Full | 0.997549 | 0.260227 |
| D_hard | Full | 1.136453 | 0.269570 |

D4 = D4_step3; D0 trước dynamics; D2_query geometry thuần; D2 blend + learned detail; D1 trước sensor fusion. **Chỉ so gain trực tiếp trong cùng grid/target**; GT native scale khác nhau. Step3 MAE tốt hơn nhưng RMSE hơi kém step2; chưa đủ kết luận bỏ step3 cải thiện final.

D1 iRMSE 3.445678, Dhard iRMSE 3.866669 km^-1. Final minimum toàn ảnh validation 0.290762 m; 236 pixel toàn ảnh <0.5 m, không đồng nghĩa có GT hoặc đều lỗi.

### GT boundary và error tail

| GT-boundary band | Pixels | RMSE m | MAE m | % tổng SSE |
| --- | --- | --- | --- | --- |
| 1px | 1.725.092 | 1.880340 | 0.962584 | 24.11 |
| 2px | 2.598.642 | 1.752524 | 0.846506 | 31.55 |
| 3px | 3.346.179 | 1.697738 | 0.779305 | 38.12 |
| 5px | 4.563.265 | 1.648158 | 0.702198 | 48.99 |
| 10px | 6.665.785 | 1.587197 | 0.608508 | 66.37 |

Boundary theo GT discontinuity absolute 1 m / relative 0.05, band dilation Chebyshev. Bands lồng nhau, không cộng %.

| Absolute error > | Pixels | % valid pixels | % tổng SSE |
| --- | --- | --- | --- |
| 1 m | 1.117.182 | 4.3940 | 95.49 |
| 2 m | 494.471 | 1.9448 | 90.59 |
| 5 m | 147.415 | 0.5798 | 77.14 |
| 10 m | 46.594 | 0.1833 | 57.67 |
| 20 m | 9.534 | 0.0375 | 30.06 |

Tail thresholds cũng lồng nhau. Full inverse/relative metrics theo subset, boundary bad-pixel rates/rings và near-zero stage counts xem JSON nguồn; không lặp toàn bộ trong bản tóm tắt.

## 5. Runtime và deployment

Nguồn [profile.json](../results/anchorflow_v8_completed_audit/profile.json): trained model, RTX PRO 6000 Blackwell Server Edition, PyTorch 2.11.0+cu130, BF16 batch1, channels-last, eager, 352×1216, 100 measured runs.

| Component | Parameters | Conv/Linear MAC G | Component median ms |
| --- | --- | --- | --- |
| RGB encoder | 340.992 | 0.536 | 1.800 |
| Sparse pyramid | 4.896 | 0.051 | 1.174 |
| F32 context | 52.144 | 0.025 | 0.225 |
| Gated fusion + decoder | 149.748 | 1.153 | 1.324 |
| Learned guidance | 200 | 0.018 | 0.106 |
| JetDynamics, 3 calls | 10.891 | 0.387 | 3.420 |
| Phase upsample 4→2 | 6.016 | 0.152 | 0.383 |
| Jet phase readout | 5.144 | 0.130 | 0.676 |
| Context2 | 400 | 0.010 | 0.056 |
| Phase upsample 2→1 | 2.488 | 0.241 | 0.356 |
| Detail + sensor trust | 5.145 | 0.511 | 0.503 |
| **Total** | **578.064** | 3.214694 | **10.221 median / 10.265 P95** |

Peak CUDA allocated **98.34 MiB**. Component medians đo pass instrument riêng, không cộng để thay total. MAC bỏ qua fixed neighborhood convolutions, transport/shift, pooling, interpolation, softmax, finite differences và memory traffic. Dynamics latency lớn nhất dù ít parameters. Không tính disk I/O / host→device transfer; **chưa phải latency edge GPU**.

Theo log epoch5–28, BF16 train median khoảng 19.92 s/epoch, train + validation 29.71 s/epoch, không tính mọi notebook/download/backup overhead. Anonymous 1.000 test export mất 22.73 s gồm data I/O và PNG; không có test RMSE công khai.

[ONNX report](../results/anchorflow_v8_completed_audit/export_report.json): FP32 opset17, static B1 352×1216, CPU parity max error 0.000237 m trên input thử; không teacher inputs. Chưa build TensorRT, chưa đo target edge, chưa verify trained checkpoint FP16 inference.

## 6. Kết luận có thể khẳng định

- D0→D4 giảm khoảng 0.131 m RMSE trên cùng quarter-grid: dynamics có ích trong graph đã train, chưa isolate bằng matched ablation.
- Pure geometry query chưa vượt learned base; learned blend/detail quan trọng. Chưa đủ kết luận query vô ích.
- Learned sensor fusion tốt hơn hard anchor. Outlier hiếm chi phối SSE; cần audit boundary × range × tail trước khi tăng width/steps/KD.
- Historical V7 1.006078 → V8 0.997549 m: tốt hơn 8.53 mm; khác initialization/budget, chưa có multi-seed significance hoặc frozen-feedback control đã train.
- Mục tiêu <0.8 chưa đạt; cần giảm 35.69% MSE để đạt 0.8, 50.76% để đạt 0.7. Novelty/parameter count không bảo đảm metric.

## 7. Source of truth và ảnh

- Model: [model.py](../drive_upload/AnchorFlow_v8_Dynamics/model.py), [core.py](../drive_upload/AnchorFlow_v8_Dynamics/core.py), [support.py](../drive_upload/AnchorFlow_v8_Dynamics/support.py).
- Train: [losses.py](../drive_upload/AnchorFlow_v8_Dynamics/losses.py), [run.py](../drive_upload/AnchorFlow_v8_Dynamics/run.py), [resolved config đã chạy](../results/anchorflow_v8_completed_audit/resolved_config.json).
- Evaluate: [metrics.py](../drive_upload/AnchorFlow_v8_Dynamics/metrics.py), [boundaries.py](../drive_upload/AnchorFlow_v8_Dynamics/boundaries.py); [raw reports](../results/anchorflow_v8_completed_audit).
- [Notebook](../drive_upload/AnchorFlow_v8_Dynamics/AnchorFlow_v8_Dynamics_TAR2000_Fresh30_EarlyStop.ipynb), [BF16 recovery](../scripts/migrate_anchorflow_v8_bf16.py).
- [10 cặp RGB/depth](../results/v8_depth_preview/README.md), [PNG RGB/V6/V8 — 5 sample](../results/v8_depth_preview/v6_vs_v8_5_rgb_depth.png).

Đặc tả trong frozen upload bundle là snapshot **trước train**, giữ nguyên checksum để reproducibility. Tài liệu này cập nhật nguyên lý + metric sau train; không sửa model, loss, config hoặc notebook đã seal.
