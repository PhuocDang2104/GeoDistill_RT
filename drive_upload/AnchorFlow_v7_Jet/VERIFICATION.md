# V7 verification — đã đo gì, chưa đo gì?

## Đã chạy local

- 30 contract tests: strict V6 migration, zero-head reduced-V6 parity, translation composition/inverse, quadratic reproduction, PixelShuffle ordering, tail gradient/no invalid pixels, GT/KD masks, empty sparse, train/checkpoint/resume, notebook parent/snapshot/rerun.
- Actual trained V6 best restored từ archive user cung cấp, epoch14; SHA256 `7ea785092601da3f64aebc1350c7d327aa4f6ce44591988d3bccc67cccd2f809`.
- Inference-only causal ablation trên **400 validation** với đúng25.424.992GT pixels, CPU FP32: fullV6RMSE1,009274m; reducedV6RMSE1,009439m.
- V7 zero heads khớp **exact** reduced V6 trên3ảnh KITTI thật352×1216, indices0/199/399, max error0m.
- Full-size real KITTI BF16 forward/loss/backward finite; new jet và proposal adapter đều có gradient khác zero. Teacher của gradient check này là synthetic; không gọi đó là actual teacher-cache accuracy/coverage audit.
- Parameters613.219; Conv/Linear MAC4.032.235.328 tại batch1/352×1216. V6=4.023.674.688MAC; tăng~0,213%, **không phải giảm93% hay đảm bảo giảm latency**.
- ONNXopset17 checker + ONNXRuntimeCPU parity với **nonzero jet/adapter heads**, retained actual V6 weights; sparse/empty sparse max absolute difference khoảng0,000885m/0,0000725m. Không phải checkpoint V7 đã train.
- Notebook schema/AST/empty outputs, default one FT15 run, frozen source/config/parent snapshot, safe rerun và recipe mismatch gate.

Raw proof: [local_verification.json](local_verification.json), [v6_causal_audit.json](v6_causal_audit.json).

## Chưa thể xác nhận ở máy này

Không có CUDA GPU local. Chưa train V7 15epoch, chưa đo FP16CUDA/TensorRT/Jetson latency/VRAM và chưa biết V7 có đạt RMSE<0,8. ONNX graph còn ops như `If`, `IsNaN/IsInf`, `Pad`, `Resize`; target engine cần kiểm/build riêng, không suy rằng mọi operator đều nhanh/hỗ trợ.

Notebook sẽ kiểm actual TAR/data/teacher coverage, AMP trên GPU thật, full-parent same-runtime validation, 15epoch train/400val mỗi epoch, test1000PNG, CUDAmedian/P95/VRAM/component profile và trained-best ONNX parity.

## Reproduce local checks

Từ repo root, dùng environment đã có PyTorch/timm/ONNX:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/audit_anchorflow_v6_for_v7.py
.\venv\Scripts\python.exe -X utf8 scripts/build_anchorflow_v7_notebook.py
.\venv\Scripts\python.exe -X utf8 scripts/verify_anchorflow_v7.py
.\venv\Scripts\python.exe -X utf8 scripts/package_anchorflow_v7.py
```

Audit cần actual result archive + local KITTI bundle. Các file này **không** nằm trong upload folder; folder chỉ chứa parent student, source, notebook, proof và small reference metrics. Không cần local audit để chạy Colab đã đóng gói.
