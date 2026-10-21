# GeoDistill-RT / AnchorFlow

**Current baseline: `Q1_s4_n1_fast` — S4 JetODE + U-PIR, Fusion512, Jet-context128.**
Adopted21/10/2026 at the user's request. **582,888 parameters; coarse4 NFE/fine0**.
Neural architecture is identical to N1; Q1 combines a fresh30 checkpoint with a
deployment-only fastpath. This is the **subset/development baseline**, not a
full-KITTI-trained model, official-test result or measured Orin implementation.

| A100 pipeline · TensorRT FP32 + CUDA Graph · full400 | Q1 reported |
|---|---:|
| RMSE / iRMSE | **0.978691 m / 3.170880 km⁻¹** |
| MAE / boundary3 RMSE | 0.255826 m / 1.667166 m |
| Synchronized wall mean / median / P95 | **3.8467 / 3.8436 / 3.8612 ms** |
| FPS = 1000 / mean | **259.97** |
| Reversed-order repeat mean | 3.8454 ms |

| Nội dung | File |
|---|---|
| Nguyên lý V11, training, metric và insight mới nhất | [AnchorFlow_V11.md](docs/AnchorFlow_V11.md) |
| Folder V11 upload nguyên lên Drive | [AnchorFlow_v11_NODE](drive_upload/AnchorFlow_v11_NODE/README.md) |
| Train riêng V11 | [Notebook fresh40 + early stop](drive_upload/AnchorFlow_v11_NODE/Train_AnchorFlow_V11_NODE_TAR2000_Fresh40.ipynb) |
| Sinh riêng teacher DMD3C++ cho toàn bộ 85.898 train, lưu ZIP KITTI và teacher shards lên Drive | [FullKITTI Colab notebook](notebooks/DMD3Cpp_FullKITTI_Train_To_Drive.ipynb) — đã sinh và kiểm tra đủ 85.898 mẫu trên A100; [data handoff](data/DMD3Cpp_FullKITTI_HANDOFF.md) |
| Train V10.1/V11 đồng thời rồi benchmark tuần tự | [Notebook paired benchmark](notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb) · [Hướng dẫn](docs/AnchorFlow_V10_1_V11_Benchmark.md) |
| Kết quả paired run đã hoàn thành, kèm raw metric nhỏ | [Benchmark pair01](results/benchmarks/v10_1_v11_pair01/README.md) |
| Baseline/control V10.1 learned-h Euler2 | [V10.1](docs/AnchorFlow_V10_1.md) |
Same Q1 weights: original engine4.2294ms → fast3.8467ms (**9.05% lower wall**).
Metrics differ by numerical noise. Accuracy differences versus old N1 are between
training runs, **not caused by the deployment rewrite**.

B1×352×1216;400 development frames/25,424,992 valid GT pixels;100 distinct GPU
timing images,100warmup/300iterations. D2D included; decode/preprocess/H2D/build/
capture excluded. **Not end-to-end throughput or Orin latency.** GPU evidence is
the user's console snapshot; full report/checkpoint hashes are not included
locally and are not fabricated from the log.

## Start here

- [Baseline specification and rigorous equations](docs/AnchorFlow_V11_Baseline.md).
- [Adopted model / exact private artifact checklist](production/AnchorFlow_S4_Q1_Fast/README.md).
- [Frozen train/benchmark package](drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100/README.md):
  upload WHOLE folder; do not modify source/manifest when resuming.
- [Same-checkpoint attribution, CSV and raw evidence](benchmarks/q1_n1fast_2026_10_21/README.md).
- [GeoLift_Data metric-only checklist](docs/GeoLift_Data_MetricOnly_Transfer.md).

Train: seed42/B4,ImageNet encoder,fresh30,1600train/400val,AdamW fused,
BF16CNN/FP32geometry+loss. Metric teacher train-only, **relative OFF**.
PyTorch trains; TensorRT compiles inference. No early stop; epoch-boundary resume.

```bash
python scripts/package_current_baseline.py
python scripts/verify_s4_n1fast.py
```

Packaging verifies/repackages frozen bytes; reverification writes outside them.
Weights, KITTI/teacher data and exported ONNX/engines stay local/Drive, not Git.
The [old G Orin helper](production/AnchorFlow_S4_UPIR_Orin/README.md) is historical,
**not a Q1 exporter**. Q1 Orin and full-KITTI generalization remain unmeasured.
Do not reuse A100 engines on another device.

[Docs](docs/README.md) · [Drive package](drive_upload/README.md) ·
[This cleanup / recovery](benchmarks/q1_n1fast_2026_10_21/CLEANUP.md).
