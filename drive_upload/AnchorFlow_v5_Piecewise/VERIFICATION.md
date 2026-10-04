# Verification scope

Đã kiểm trên Windows local, CPU PyTorch, không có CUDA GPU local:

- **21 contract tests pass**: model forward/backward, KD exclusion, confidence weighting, GT mask, global/inverse metrics, archive safety/cache train-only, scheduler, train/checkpoint/resume fixture, no-op migration cả hai mode, affine inverse-depth plane preservation, symmetric directional barrier/non-canceling center mass, boundary bands/rings, BF16 gradients và compute budget.
- Nạp checkpoint V3 thật: đủ697tensors/572.017params, không drop key; new7.976params.
- Full352×1216 KITTI samples0/199/399: Dfull no-op max error **0m** trên cả ba.
- BF16 forward/backward trên một ảnh KITTI thật: finite loss/gradients. Teacher target ở check này **synthetic**, không gọi là audit teacher cache thực tế. Colab prepare/smoke kiểm cache thật trước train.
- Count Conv/Linear: **579.993params /3.176.171.328MAC**; new202.031.104MAC. Không đếm functional fixed-neighbor convolutions, shifts, pooling, elementwise, memory traffic.
- ONNX17 static352×1216 checker + CPU ORT parity với **nonzero new heads**: max error0,000427m khi có sparse và0,000069m khi empty sparse. Trọng số new heads là structural fixture, chưa train; report ghi `untrained_weights=true`.

Raw proof: `local_verification.json`. Audit completed V4 trên400val: `v4_gt_boundary_audit.json`, có CPU/precision/checkpoint SHA/protocol; không train hoặc fit threshold bằng validation.

**Chưa kiểm:** 15epoch V5, CUDA FP16 gradient/data cache trên Drive người dùng, real GPU latency/P95/VRAM V5, TensorRT/INT8/edge hardware deployment, official test score. Notebook thực hiện train/eval/profile/export thực tế; không thay thế bằng số ước lượng.

```bash
python -m unittest discover -s . -p 'test*.py' -v
python run.py prepare --config resolved_config.json
python run.py smoke --config resolved_config.json
python run.py train --config resolved_config.json --variant metric_kd
python run.py evaluate --config resolved_config.json --variant metric_kd
python run.py test --config resolved_config.json --variant metric_kd
python run.py profile --config resolved_config.json --variant metric_kd --parent
python run.py profile --config resolved_config.json --variant metric_kd
python run.py export --config resolved_config.json --variant metric_kd
```

Notebook chạy các lệnh này với cwd/local resolved paths, không yêu cầu chỉnh thủ công source sau upload. Các test train/resume là CPU fixture1epoch nhỏ, không phải accuracy experiment.
