# GeoLift-S3 Lite — executable architecture and TAR2000 training

Đây là mô tả khớp với implementation `GeoLiftStudentS3Lite` và notebook benchmark S3 hiện tại.

## 1. Inference graph

```text
RGB ── MobileNetV4-Conv-Small-0.5 pretrained
       ├── F4
       ├── F8
       └── F16

Sparse depth + mask ── compact sparse pyramid
                       ├── S4
                       ├── S8
                       └── S16

(F4,S4), (F8,S8), (F16,S16)
          ↓ gated scale fusion
       LiteFPN 32/24/24
          ↓
         D16
          ↓ Residual Metric Lift
         D8
          ↓ Residual Metric Lift
         D4
          ↓ Residual RayLift, learned phase guidance
         D2
          ↓ Residual RayLift, phase-preserving F2
         D1
          ↓ hard sparse anchor
       D_full, C_full
```

RGB và sparse không được concat tại input. MobileNetV4 chỉ nhận RGB. Sparse branch tạo local normalized prior tại `1/4`, sau đó tạo pyramid `S4/S8/S16` và fuse bằng gate riêng ở mỗi scale.

Encoder mặc định:

```yaml
encoder: mobilenetv4_conv_small_050.e3000_r224_in1k
encoder_pretrained: true
```

Toàn encoder được fine-tune; không freeze stage. Pretrained chỉ thay initialization, không tăng parameter, MAC hoặc inference latency.

## 2. Decoder

Hai stage thấp `16→8` và `8→4` dùng residual inverse-depth upsampling:

\[
\xi_{out}=\operatorname{up}(\xi_{in})+g\,\Delta\xi,
\qquad g_0=0.05.
\]

Chỉ hai stage có độ phân giải cao `4→2` và `2→1` dùng geometry-aware RayLift. Sau bước sampling, affine ray transport và aggregation, mỗi block có explicit metric correction:

\[
\xi_{out}=\xi_{RayLift}+g_r\,\Delta\xi,
\qquad g_{r,0}=0.05.
\]

`LearnedPhaseGuidance` áp dụng learned grouped/DW convolution trên RGB pixel-unshuffle tại `1/2`. Không tạo full-resolution learned feature map.

S3 vẫn xuất `C_full`, nhưng không có confidence calibration head hoặc confidence loss. Confidence chỉ được truyền analytic qua decoder để giữ contract infer/export hiện có.

## 3. Loss — bật toàn bộ từ epoch đầu

Không có Stage A/B/C và không có teacher trong run đầu:

\[
\mathcal L=
\mathcal L_{metric}
+0.2\mathcal L_{log}
+0.1\mathcal L_{sparse}
+0.05\mathcal L_{edge}.
\]

Metric loss được tính trên toàn pyramid:

\[
\mathcal L_{metric}
=\sum_{s\in\{16,8,4,2,1\}}w_s
\operatorname{Huber}(D_s-D_{gt,s}),
\]

với:

| Output | Weight |
|---|---:|
| `D16` | 0.05 |
| `D8` | 0.10 |
| `D4` | 0.25 |
| `D2` | 0.50 |
| `D1` | 1.00 |

`L_log` là log-depth Huber tại `D1`; `L_sparse` dùng `D_pre_anchor` để hard anchor không làm loss bằng 0 giả; `L_edge` so khớp log-depth gradient tại các cặp GT hợp lệ.

Mọi module và loss đều nhận gradient từ epoch 0. Gate không được mở bằng schedule: giá trị ban đầu gần `0.05`, sau đó tự thay đổi qua gradient. Notebook log:

```text
train_mean_residual_gate_8
train_mean_residual_gate_4
train_mean_residual_gate_2
train_mean_residual_gate_1
```

## 4. TAR2000 protocol

S3 dùng đúng:

- 1.600 train;
- 400 validation, raw-drive disjoint;
- 1.000 KITTI anonymous test;
- cùng manifest, KITTI bundle và immutable protocol lock với S2.

Input Drive:

```text
MyDrive/GeoLift_Data/teacher_subset_2000/
├── selected_2000_ids.json
└── kitti_trainval_2000.tar
```

`metric_coarse_train_2000.tar` và `geometry_fused_train_2000.tar` có thể vẫn nằm cùng folder nhưng notebook S3 không đọc hoặc extract chúng.

Notebook:

```text
notebooks/GeoLift_S3_Lite_TAR2000_Train1600_Val400_Test_OneRun.ipynb
```

Output:

```text
MyDrive/GeoLift_RT_runs/v3_s3_lite_pretrained_train1600_val400/
```

Notebook xuất checkpoint/log chuẩn, global/range/edge metrics, profile FP16, loss budget, gate dynamics và bảng so sánh S3 với S2.

## 5. Benchmark interpretation

S3 thay đồng thời encoder, decoder, pretrained initialization và training objective; vì vậy đây là architecture-system benchmark, không phải single-factor ablation. Nếu S3 tốt hơn, các run tiếp theo cần tách ít nhất:

1. pretrained `true/false`;
2. residual correction on/off;
3. learned phase guidance so với raw PPG;
4. RayLift bốn stage so với chỉ hai high-resolution stage;
5. teacher-free so với thêm một metric KD sau khi validation plateau.

Source of truth:

- `src/model_geolift_s3.py` — forward graph;
- `src/losses.py::geolift_s3_loss` — objective;
- `configs/geolift_s3_lite_tar2000.yaml` — hyperparameters;
- notebook TAR2000 S3 — data, resume, logging và benchmark protocol.
