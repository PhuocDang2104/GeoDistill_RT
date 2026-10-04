# Verification evidence

Đã chạy bằng Torch2.11 CPU trên máy local; bằng chứng machine-readable:
[local_verification.json](local_verification.json).

| Check | Result / scope |
|---|---|
| Unit/contract tests | 22 PASS |
| Colab-staged layout without notebook UI | 22 PASS |
| Synthetic two-epoch training loop | Checkpoint/log, completed resume, changed-protocol rejection PASS; không GPU benchmark |
| Real KITTI RGB/sparse/GT/K backward | Full352×1216, FP32 và CPU-BF16 gradients finite |
| Teacher fixture in local backward | **Synthetic metric/relative targets**, không giả claim đã kiểm actual teacher TAR local |
| New controller / reaction / innovation head gradients | Nonzero finite |
| Fixed3 vs loaded baseline | Same output/nominal objective within numerical tolerance |
| Conditional vs masked adaptive | Identical output at exit2/3/4; real field-call counter validates early exit |
| Time/h/bounds | Actual accumulated time; h∈[1/6,1/3], horizon≤4/3; unchanged jet projection |
| T4 BF16 guard | Reject emulated BF16; auto resolves explicit FP32 before freezing config |
| ONNX17 masked policy | 9cases×11outputs PASS, max difference0.0014191 m |
| ONNX17 static4 fallback | 9cases×11outputs PASS, max difference0.0013809 m |
| Fresh V10 GPU training/accuracy | **Chưa chạy** |
| V10 GPU latency / TensorRT engine | **Chưa đo/chưa build** |

ONNX test dùng actual trained **V9.1 blocks** (best local epoch1) + time-conditioned synthetic
controller để có depth/geometry không constant, forced exit3. Đây chỉ là stress-test numerical
export, không V10 checkpoint được train, không accuracy comparison. Không có checkpoint đó
trong folder upload. Fresh training load duy nhất ImageNet RGB encoder.

9cases/graph: seeds0/7/42 observed/empty sparse + actual KITTI val indices0/199/399.
11outputs gồm4trajectory states, selectedD4/D2query/D2/D1/Dfull, selectedNFE và terminaltime.
Tolerance0.01m/rtol1e-4 không được nới; exit choices khớp giữa PyTorch và ORT.
Production notebook export còn chạy lại trên chính checkpoint V10 train được.

Sau upload, notebook vẫn bắt buộc actual data gate (1600/400/1000, unit/layout/split, cả2teacher
cache content), rồi GPU smoke/backward với **actual teacher targets** trước train.
Đây là gate không thể thay bằng local synthetic-teacher proof. Nếu lỗi CUDA/nonfinite sẽ dừng
và log traceback/diagnostic về Drive, không skip/mask/fallback ngầm.

Frozen V8/V9 sources không sửa; V9.1 result archive và best checkpoint không sửa.
ZIP seal kiểm code/config/immutable notebook contracts; runtime bỏ qua mutable .ipynb UI.
