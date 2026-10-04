# AnchorFlow V9.1 — Sparse-Innovation Metric Readout

Implementation contract: RGB+sparse+mask+K → metric depth; teachers chỉ xuất hiện trong training objective.
Input352×1216, train1.600/val400/test1.000, depth0,1–120m. Baseline tham chiếu: V9 best epoch23.

## 1. Graph

~~~text
RGB ── MobileNetV4 pretrained ── F4 / F8 / F16 / F32
                                   │             │
Sparse + mask + K ── sparse pyramid │      context F32→F16
                                   ▼
                       gated fusion + lite decoder
                          D16 → D8 → D0(1/4)
                                   │
                       minmod base jet initialization       NEW
                                   │
                shared learned reaction + transport ×3
                                   │
                            final jet j3 → D4
                                   │
                     phase metric upsample → B2
                                   │
            ┌──────────────────────┼────────────────────────┐
            │                      │                        │
      five-jet query         old metric head       sparse innovation
      consensus + U                │            + P4 + hidden + U
            │                      │                        │
      geometric gate               │            tiny PW/DW/PW head NEW
            │                      └────────── + ───────────┘
            │                             same bounded tanh
            └──────────────────────┬────────────────────────┘
                                   ▼
                                  D2
                                   │
                    phase metric upsample → B1
                                   │
                    RGB detail + learned sensor trust
                                   ▼
                                 Dfull
~~~

Không thêm teacher network, transformer, adaptive ODE, recurrent full-resolution propagation,
hay full-resolution learned feature map.

## 2. Jet dynamics giữ nguyên

Jet inverse-depth tại grid1/4:

$$
j=[v,g_x,g_y,h_{xx},h_{xy},h_{yy}],\qquad v=D^{-1}.
$$

Ba bước dùng chung trọng số, đọc state hiện tại và sensor innovation; step size α=1/3,
pseudo-time τk=k/3:

$$
j_p^{k+1}=\Pi\!\left(j_p^k+\alpha
\left[R_\theta(Z_p,j_p^k,e_p^k,\tau_k)+
\sum_q w_{pq}^k\bigl(T_{q\to p}(j_q^k)-j_p^k\bigr)\right]\right),
\qquad k=0,1,2.
$$

T là Taylor jet transport analytic; R là forcing học được; conductance/barrier giới hạn trao đổi;
Π là projection bound. Đây là fixed-step learned geometry dynamics, không adaptive solver.
Chi tiết kernel/projection giữ nguyên [model_v8.py](model_v8.py).

## 3. Minmod initialization

Cho hai finite differences một phía a,b:

$$
\mathrm{mm}(a,b)=\max(0,\min(a,b))-\max(0,\min(-a,-b)).
$$

Nếu cùng dấu: lấy slope có độ lớn nhỏ hơn; nếu trái dấu: zero. Biên ảnh giữ one-sided difference.
Áp dụng cho slope và Hessian của **base jet ban đầu**, không áp lại như ràng buộc lên mọi learned update.
Các derivative seed detach giống baseline; inverse depth center vẫn có gradient.

Limiter cổ điển tránh thêm extrema trong piecewise-linear reconstruction; **không suy ra toàn bộ
network này TVD, monotone, hay có bảo đảm ổn định PDE**.
Tham khảo [LeVeque — finite-volume/minmod](https://depts.washington.edu/clawpack/links/an11/an11draft13feb2011.pdf).

## 4. Sparse innovation và metric readout

B2 là depth base trước query; S2,M2 là valid-average pooled sparse và binary support grid1/2.
Tại donor anchor:

$$
E_2=M_2\odot(S_2-B_2).
$$

Với cửa sổ7×7 trên grid1/2 và A là average pooling:

$$
\rho=A(M_2),\qquad
\mu=\frac{A(E_2)}{\max(\rho,10^{-6})},\qquad
\sigma^2=\max\!\left(0,\frac{A(E_2^2)}{\max(\rho,10^{-6})}-\mu^2\right).
$$

Head nhận [μ/20, σ/20, ρ]. Code dùng sqrt(σ²+1e−6)−0.001 để gradient finite khi variance zero,
và zero vùng không support. Pooled sparse có thể trộn surfaces; dispersion là guidance học được,
**không** coi pooled mean là dense GT/hard anchor.

~~~text
hidden24 + P4(48) + phase-pack innovation(12) + phase disagreement U(4)
                                  = 88 channels @1/4
PW 88→32 → SiLU → DW3×3 dilation2 → SiLU → PW32→4
~~~

Bốn output phases theo đúng PixelShuffle PyTorch. PW cuối zero weights/bias; lần backward đầu
chỉ lớp cuối có gradient khác zero, các lớp trước bắt đầu nhận gradient sau khi lớp cuối mở.

Five-jet query giữ center/E/W/S/N, barrier penalty, inverse-depth consensus và disagreement U.
U là dispersion heuristic, không calibrated uncertainty. Metric enhancement:

$$
r_{\mathrm{new}}=h_{\mathrm{old}}(H)+h_{\mathrm{innov}}(H,P_4,\mathrm{pack}(\mu/20,\sigma/20,\rho),U),
$$

$$
\Delta D_2=(1+0.05B_2)\odot\mathrm{shuffle}(\tanh r_{\mathrm{new}}),
$$

$$
D_2=\mathrm{clip}\!\left(B_2+g_q\odot\mathrm{clip}(Q_2-B_2,-(1+0.1B_2),1+0.1B_2)+\Delta D_2,\ 0.1,\ 120\right).
$$

Residual bound **không tăng** so với V9. Tắt minmod thì zero-head tái tạo output parent bit-exact;
default bật minmod nên vẫn cần initial validation.

## 5. Objective

Giữ GT multi-scale Huber weights D16/D8/D4/D2/D1/Dfull = .025/.05/.15/.30/.50/1,
log.2, edge.05, sensor.02, holdout.05, trust.02, range.1, boundary.05, barrier.01, dynamics auxiliary.1.
Thay global RMSE weight từ.25→.4.

Cho r=Dfull−GT, valid GT mask V, ρδ là Huber:

$$
\mathcal L_{\mathrm{tail}}=
\frac{1}{|V|}\sum_{p\in V}2\rho_{10}\!\left(\max(0,|r_p|-2)\right).
$$

Weight.25, ramp2 epoch. Quadratic tới error12m, linear sau đó; **không cắt gradient extreme outliers**.
Normalize trên toàn valid GT, không sort/top-k, không normalize bằng số tail pixels.
Đây là surrogate tập trung excess risk; không đồng nhất với benchmark global SSE.

Metric KD weight.012→.006 trong15 epoch; confidence threshold.5, teacher-edge.01.
GT/sparse có ưu tiên, teacher chỉ ở vùng chưa quan sát. Inverse-GT penalty giữ.01.
Relative KD chỉ giữ locally normalized inverse-depth gradient, weight.015, ramp3 epoch,
offsets1/4 trên hai trục, confidence≥.35. **Ordinal không tính**; log ordinal=0 để tương thích schema.
Relative pair coverage mới đếm gradient pairs, không trực tiếp so scalar này với ordinal-pair coverage V9.

## 6. Training/deploy contract

- Default model-only migration từ parentV9 epoch23, optimizer/scheduler/RNG mới, 15epoch additional.
- AdamW; encoderLR.25×, olddecoder1×, newhead2×; warmup1→cosine minratio.05; clipgrad1.
- BF16 Conv, FP32 analytic geometry/loss; channels-last; không AMP FP16 fallback.
- New head trên grid1/4; pooling ở1/2; không sort, scatter theo pixel, adaptive iteration.
- Output dùng soft sensor reliability fusion; hard anchor chỉ diagnostic, không thay protocol.
- Export fixedbatch1/352×1216 ONNX17, CPUORT strict tolerance. TensorRT/INT8 chưa chứng nhận.

Novelty phù hợp để nghiên cứu: **feedback jet dynamics + disagreement-aware geometric consensus,
được bổ sung sparse-innovation conditioned metric readout**. Minmod là nền tảng có sẵn,
không claim phát minh limiter hay “first-ever” method.

## 7. Source

| Phần | File |
|---|---|
| V9.1 / minmod / innovation | [model.py](model.py) |
| V9 frozen reference / five-jet | [model_v9_reference.py](model_v9_reference.py) |
| Encoder/decoder/dynamics | [model_v8.py](model_v8.py), [core.py](core.py) |
| Objective mới / objective nền | [losses.py](losses.py), [losses_v8.py](losses_v8.py) |
| Paired dataset / teachers | [data.py](data.py), [relative_data.py](relative_data.py) |
| Train/eval/test/profile/export | [run.py](run.py) |
| Defaults / contracts | [config.json](config.json), [test_contracts.py](test_contracts.py) |
