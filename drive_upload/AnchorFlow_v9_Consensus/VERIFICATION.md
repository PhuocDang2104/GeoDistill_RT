# Verification — phạm vi và giới hạn

Kết quả máy đọc: [local_verification.json](local_verification.json). V9 không làm thay đổi source/checksum của frozen V8 bundle.

## Đã kiểm tra trước khi seal

- **17 contract tests**, chạy cả folder gốc và layout Colab chỉ copy Python/JSON, không UI `.ipynb`.
- Analytic origin translation của center/E/W/S/N và bốn phases, invalid-neighbor mask; constant planar surface có disagreement≈0.
- CPU BF16 forward/backward channels-last finite; candidate/readout/dynamics có nonzero gradients; empty sparse an toàn.
- Training fixture fresh + resume, không duplicate epoch; protocol mismatch fail rõ, early-stop floor/patience/cumulative min_delta.
- Nonfinite loss fail **trước backward**; nonfinite gradient fail **trước optimizer.step**, không batch skipping/nan_to_num.
- Relative archive streaming, per-record hash, coverage/order/model-revision/subset matching; float32 schema, train-only cache, val audit-only.
- Teacher inverse convention larger=near, TTA horizontal alignment, positive affine invariance; GT/sensor forbidden masks, missing cache không fallback.
- Hai clean notebook snapshots parse mọi code cell; UI output edits không phá runtime checksum nhưng immutable source vẫn strict.
- V8 control dùng original `model_v8.py` / `losses_v8.py`, objective đúng V8 khi inverse=0 và relative disabled.
- Ba RGB/sparse/GT KITTI thật ở val indices 0,199,399, 352×1216, output finite. CPU BF16 backward trên sample thật dùng **synthetic teachers** chỉ để test gradient path, không đánh giá teacher quality.
- FP32 static B1 ONNX17 checker + ONNXRuntime CPU parity cho sparse hợp lệ/empty sparse, nonzero candidate/reaction/detail heads; không teacher input trong graph.
- Official DA3MONO-LARGE network từ pinned Git: instantiate/random-weight forward và đối chiếu **toàn bộ 406 safetensors keys/shapes** với header checkpoint pinned. Không silent partial checkpoint load.

## Chưa kiểm tra / không được suy diễn

- Không có CUDA local: chưa chạy full pretrained DA3 generation 2.000 ảnh, fresh30 GPU training, trained V9 accuracy hoặc GPU latency.
- Teacher header/name/shape match không chứng minh pretrained tensor values download thành công hoặc teacher phù hợp mọi KITTI sample. Notebook 01 strict-load actual weights; notebook 02 audit actual generated cache và smoke real targets.
- Local synthetic teachers không được xem là output DA3/metric teacher đã cung cấp.
- ONNX CPU parity không là TensorRT engine validation, BF16/INT8 calibration hoặc measured edge FPS.
- Không bảo đảm target RMSE <0,9/<0,8 m. Median latency không suy từ MAC hoặc parameter increase nhỏ.

## Rebuild cho developer

Từ repo root, Python environment có torch/timm/onnx/onnxruntime/nbformat và DA3 minimal dependencies:

```powershell
python scripts/build_anchorflow_v9_notebooks.py
python scripts/verify_anchorflow_v9.py
python scripts/package_anchorflow_v9.py
```

Verifier teacher adapter cần pinned official source ở `results/anchorflow_v9_audit/teacher_source` và Internet đọc config/safetensors **header only**. Teacher full weights không được đưa vào bundle. Package script từ chối source hash khác proof, test fail, notebook lỗi, checkpoint/data hoặc stale verification.
