# V9.1 — Local verification

Đã chạy CPU, PyTorch 2.11, không có CUDA local. Proof:
[local_verification.json](local_verification.json).

| Gate | Kết quả |
|---|---|
| Unit/contract tests | 26/26 pass |
| Colab-style staging không có .ipynb UI | 26/26 pass |
| Parent checksum + source + epoch | V9 best 23, strict allowlist chỉ new metric head |
| Teacher generator recipe | Byte-identical với V9, cache tái sử dụng |
| Real KITTI input/output | Val samples 0/199/399, 352×1216, finite |
| BF16 backward CPU | Finite gradients; dynamics/query/new metric head có signal |
| Zero-init migration | Tắt minmod → output bit-exact với V9 reference |
| Empty sparse / one anchor | Finite output và gradients |
| Tail extreme errors | Gradient khác zero, không hard truncation |
| Fresh trainer / resume / protocol rejection | Synthetic one-step fixtures pass |
| Nonfinite loss / gradient | Abort trước backward hoặc optimizer; không ghi bad checkpoint |
| Backend compatibility controls | Mock CUDA test: benchmark false thực sự áp dụng; BF16/layout không đổi |
| Failure diagnostics | Conv metadata + exception preserved; observer hooks removed |
| Backend provenance | Backend settings locked in checkpoint protocol |
| ONNX17 all 10 stages | 11 cases: 4 seeds×sparse/empty + 3 KITTI samples |
| ONNX max absolute difference | 0.00111008 m; atol .01 m, rtol 1e−4 |

ONNX test dùng **parent V9 trained thật**, limiter mới và new head được mở bằng random small weights.
Không phải parity của model V9.1 đã fine-tune; notebook bắt buộc export lại sau train.
BF16 backward dùng RGB/sparse/GT thật nhưng **hai teacher tensors là synthetic fixtures**.
Không giả vờ đã chạy CUDA training hoặc đo teacher accuracy trong phép kiểm này.

## Footprint

| Model | Parameters | Conv/Linear MAC |
|---|---:|---:|
| V9 frozen | 578,568 | 3,227,535,168 |
| V9.1 | 581,868 | 3,313,997,632 |
| Tăng | 3,300 (+0.5704%) | 86,462,464 (+2.6790%) |

Conv/Linear MAC không đếm pooling, phase transforms, Taylor transport, activation, tensor layout,
memory traffic. **Không suy latency tăng chính xác 2,679%** từ MAC.

## Initialization-only validation

Load parent weights, zero new head, bật minmod; **không optimizer step**.
Full 400 val CPU FP32, valid pixels 25.424.992:

- V9 original CPU FP32 audit: RMSE 1.002495 m.
- V9.1 initialization CPU FP32: RMSE **1.002047 m**.

Chênh lệch nhỏ này chỉ đo limiter migration, không phải gain học mới; không so như phép đo đồng nhất
với uploaded BF16 1.002352 m. GPU notebook lưu initial metrics riêng trước fine-tune.

Chưa kiểm chứng: RMSE V9.1 sau train, GPU median/P95/VRAM, TensorRT, INT8/FP16 export,
anonymous test accuracy, unseen subset/generalization.

CUDA compatibility revision: default cuDNN autotuning tắt; chưa tái hiện hay xác nhận fix
lỗi FIND trên GPU của người dùng. Tests backend/observer dùng mocks hoặc CPU fixtures.
