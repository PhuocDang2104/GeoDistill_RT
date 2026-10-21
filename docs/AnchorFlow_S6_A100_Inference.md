# A100 inference benchmark — V11.3 baseline

> Historical workflow, superseded for deployment on19/10/2026 by
> [S4-U-PIR NoRelative](AnchorFlow_V11_Baseline.md). Keep the solver/metric scope
> below separate from G. Historical source/weights may require the D archive.

Baseline hiện tại giữ **V11.3 best_policy**: 584.845 params; historical validation RMSE0.986517m / iRMSE3.169735km⁻¹. [Nguyên lý và metric](AnchorFlow_V11_3.md). Đây là kết quả adaptive Bosh3 đã train, **không phải** số đo S6 mới.

Folder upload: [AnchorFlow_V11_3_S6_A100_Inference](../drive_upload/AnchorFlow_V11_3_S6_A100_Inference/README.md).
Mở [notebook inference-only](../drive_upload/AnchorFlow_V11_3_S6_A100_Inference/Benchmark_S6_LRRU_A100_INFERENCE_ONLY.ipynb), chọn A100 → Run all.

- Upload nguyên folder; checkpoint V11.3 best_policy và LRRU-Mini đã đóng trong folder/ZIP local.
- Data: `GeoLift_Data/teacher_subset_2000/{selected_2000_ids.json,kitti_trainval_2000.tar}`. Chỉ tách400val, không teacher/train.
- S6 đúng yêu cầu: **midpoint RK2 [.25,.25,.5]**, coarse6+fine2, state half-time tích phân thật. Không phải production Fixed6 classicRK3 trước đó.
- Đo eager/unrolled/cache/compile/CUDA Graph riêng, B1/352×1216,100warmup/300timed, kiểm numerical parity + full400metrics trước khi chọn winner.
- LRRU-Mini official source/checkpoint được strict-load; modern **torchvision DCNv2 adapter** có gate bilinear. Backend này không đồng nghĩa extension legacy official; CPU IP-Basic preprocessing báo riêng. [Official upstream](https://github.com/YufeiWang777/LRRU).
- Result về `MyDrive/GeoLift_RT_Runs/S6_A100_Inference_<timestamp>/`: CSV/JSON + performance report + traces.

Local hiện chỉ có CPU. Đã test reference/cache trên ảnh KITTI thật; **chưa có runtime A100 mới**. Không suy ra speedup từ MAC, không lấy latency lịch sử thay số mới. Custom kernel/TensorRT chỉ nên triển khai sau khi thấy profiler bottleneck thật; notebook không có code train hay chuyển weights.
