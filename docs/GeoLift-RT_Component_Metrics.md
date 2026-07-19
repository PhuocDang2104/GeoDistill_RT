# GeoLift-RT v2.1 — component metrics

Đây là số của implementation `GeoLiftStudentS2` hiện tại tại input `1×3×352×1216`, không còn là ước lượng của graph A0.

## Static inference cost

| Thành phần | Conv/Linear MAC | Params | Nhận định trước khi đo GPU |
|---|---:|---:|---|
| Compact sparse prior `D_init4` | ngoài MAC | 0 | Pooling nhẹ; không global fill |
| RGB stem | 0.049 G | 984 | Nhẹ, chỉ conv đầu đọc full resolution |
| Sparse/prior stem | 0.013 G | 576 | Nhẹ, toàn bộ learned processing ở 1/4 |
| Fusion `42→32` | 0.071 G | 2,848 | Đã giảm rất mạnh so với fusion full-resolution cũ |
| Stage-adapted MobileViTv2 | **0.958 G** | **285,314** | MAC/parameter lớn nhất |
| LiteFPN | 0.075 G | 9,024 | Nhẹ |
| Initial `D16/C16` heads | 0.003 G | 1,826 | Rất nhẹ |
| PPG | 0.046 G | 520 | Nhẹ về MAC; có phase packing |
| Compact source state ở 1/2 | 0.045 G | 492 | Full-ish resolution, memory-sensitive |
| RayLift-ID `16→8`, K=5 | 0.007 G | 4,401 | Ít sample |
| RayLift-ID `8→4`, K=3 | 0.027 G | 4,248 | Nhẹ–vừa |
| RayLift-ID `4→2`, K=3 | 0.079 G | 3,096 | Memory/sample bắt đầu đáng kể |
| RayLift-ID `2→1`, K=2 | **0.208 G** | 2,108 | RayLift có khả năng chậm nhất do full-resolution layout/grid sampling |
| **Tổng model** | **1.581 G** | **315,437** | Không teacher ở inference |

MAC chỉ đếm `Conv2d` và `Linear`. Nó không tính `grid_sample`, pooling, softmax, PixelShuffle/Unshuffle, interpolation, analytic ray transport hay memory traffic. Vì vậy không kết luận bottleneck chỉ từ MAC: encoder có MAC lớn nhất, nhưng `RayLift 2→1` vẫn có thể có runtime lớn nhất.

Đo GPU thật bằng:

```bash
python scripts/profile_geolift_s2.py \
  --config /path/to/geolift_s2_colab.yaml \
  --warmup 30 --runs 100 \
  --output /path/to/geolift_component_profile.json
```

Profiler báo median, P95, FPS, peak VRAM, parameter, MAC và runtime từng component ở batch 1/FP16/`352×1216`. Phải dùng số JSON này để kết luận bottleneck trên T4/L4/A100; không chuyển runtime CPU thành FPS GPU.

## RMSE đúng điểm đo

| Output | Ý nghĩa | Cách đọc |
|---|---|---|
| `D_init` | Sau valid-aware downsample + local normalized prior 7×7 | RMSE chỉ trên `V_init=1`; báo cả số valid pixels |
| `D16` | Coarse metric prediction | Trước RayLift |
| `D8` | Sau RayLift `16→8` | Đo ảnh hưởng stage 1 |
| `D4` | Sau RayLift `8→4` | Đo ảnh hưởng stage 2 |
| `D2` | Sau RayLift `4→2` | High-resolution boundary stage |
| `D1` | Sau RayLift `2→1`, trước anchor | Metric chính phản ánh student học được gì |
| `D_full` | Sau hard sparse anchoring | Output deployment; sparse pixels được thay đúng sensor |

Các stem/fusion/encoder/FPN là feature, không có RMSE standalone. Muốn đo giá trị của chúng phải train ablation cùng seed/budget rồi báo `ΔRMSE`.

Không tái sử dụng số `D_init=1.403 m` của graph cũ: prior cũ global-fill toàn bản đồ, còn v2.1 giữ validity và chỉ dùng một cửa sổ 7×7 nên miền chấm khác. Notebook ghi đúng các giá trị mới vào:

```text
logs/infer_val_metrics_global.json
  ├── rmse, mae, irmse, imae, abs_rel
  ├── stage_rmse_m
  └── stage_valid_pixels
```

Để tìm điểm nên nâng cấp:

1. Nếu `D16` đã kém: ưu tiên encoder/metric teacher/training, không sửa RayLift trước.
2. Nếu RMSE tăng tại một stage `D8/D4/D2/D1`: kiểm tra phase order, sampling offsets, gate/slopes và boundary supervision của chính stage đó.
3. Nếu `D1` tốt nhưng `D_full` chỉ cải thiện rất ít: bình thường khi sparse density thấp; báo cả hai, không dùng hard anchor che chất lượng pre-anchor.
4. Nếu `RayLift 2→1` vượt 25–30% runtime: ưu tiên fused phase sampler/layout optimization; không tăng channel/K.
5. Nếu encoder chậm nhất nhưng RMSE còn tốt khi giảm width: ablate backbone width trước khi viết custom CUDA.
