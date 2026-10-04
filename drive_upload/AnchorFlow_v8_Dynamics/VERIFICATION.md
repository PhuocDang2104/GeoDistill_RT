# V8 verification — scope chính xác

Chi tiết máy đọc được: [`local_verification.json`](local_verification.json).

| Kiểm tra | Kết quả |
|---|---|
| Contract tests | 22 pass: dynamics feedback, 6-channel gradients, data/KD, metrics, trainer/resume, early stop, notebook |
| Colab staging | Toàn bộ tests pass khi copy code/JSON vào folder mới **không có .ipynb**; snapshot canonical thay UI notebook |
| Early stop | Patience/min_delta/min_epochs, small cumulative gains, disabled policy; actual trainer stops and resumes without extra epochs |
| Shared vector field | Được gọi 3 lần với state khác nhau; perturb current state thay forcing/conductance |
| Model implementation | Không tạo QuarterFlow/MetricRefine4/V5 surface/V6 connection stack |
| Actual KITTI input | 3 mẫu validation ID indices 0/199/399, 352×1216, output finite |
| Actual KITTI BF16 backward | Finite loss/gradients; state/reaction/conductance/query gate có gradient |
| Teacher trong local backward | Target tổng hợp để test plumbing; **không là evidence chất lượng teacher cache** |
| Params / ConvLinear MAC | 578.064 / 3.214.694.208; đủ 3 shared field calls |
| ONNX | Opset17, static batch1 352×1216; checker + ORT CPU parity pass |
| Opened-head export test | Forcing/phase/detail heads nonzero; max error sparse ≈0,000031 m, empty sparse ≈0,000132 m |
| Fresh student | Không chuyển checkpoint cũ; fresh trainer init flag và resume branch kiểm tra bằng tests |

Local environment: Python3.11, PyTorch2.11 CPU, timm1.0.26. Tests BF16 là **CPU**, không CUDA FP16.

Chưa thực hiện: tải ImageNet weights trong local verification; Colab CUDA FP16 smoke/train; full400 V8 trained accuracy; TensorRT engine; edge-device latency. Notebook sẽ tải encoder pretrained đúng flag rồi kiểm tra real-data AMP smoke, train/eval/profile/export.

V7 reference chỉ là CSV log cung cấp, không có trained V7 checkpoint mới để chạy matched-runtime causal audit. Không dùng số loss BF16 smoke để quảng bá RMSE V8. Các source changes sau verification buộc chạy lại verify script trước đóng bundle.
