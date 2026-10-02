# V6 verification — local CPU, không phải accuracy benchmark

Ngày02-10-2026. Machine local không có CUDA; PyTorch2.11.0+cpu, Python3.11.9, timm1.0.26. Chi tiết/raw output ở `local_verification.json`.

| Check | Kết quả |
|---|---|
| Unit/contract tests | **30/30 pass** |
| Actual notebook parent/config/snapshot cell | Pass: V5 epoch dynamic, explicit V3 fallback, safe rerun, changed-recipe rejection |
| V6 và V5-control train/val/save/resume fixture | Pass, CPU synthetic2samples; không dùng như metric KITTI |
| Plane normal/frame, tangent rotation/inverse/tangency, antipodal fallback | Pass |
| Ray projection khác v_z, missing parent key rejection | Pass |
| V5 simulated nonzero-parent → V6 connection/ambient | Exact no-op all old depth stages |
| Trained V3 fallback → V6, KITTI val samples0/199/399,352×1216 | Dfull max error0,0,0m |
| Actual KITTI sample BF16 objective/backward | Finite; teacher ở gradient check là synthetic, không phải cache gate |
| New-head gradients và body gradients sau mở head | Finite/nonzero |
| Empty sparse/full-size output/compute | Pass;612.896params/4.023.674.688Conv–LinearMAC |
| ONNX checker + ORT CPU, sparse/empty sparse, nonzero new heads | Pass; max abs errors0,00071335m /0,00006294m |
| Actual trained V5 checkpoint local | **Chưa có**; Colab kiểm từ Drive |
| Actual teacher coverage và15epoch CUDA V6 | **Chưa chạy**; prepare/smoke/train kiểm trên Colab |
| Target edge latency / TensorRT / INT8 | **Chưa đo/build** |

ONNX verification dùng trained V3 + V5/V6 heads mở bằng trọng số thử; `untrained_weights=true` đánh dấu **structural parity**, không phải kết quả accuracy V6. Không có chứng nhận TensorRT chỉ vì ONNX pass.

## Reproduce local

Từ root repo, dùng Python env có requirements và data RGB/sparse/GT local:

```powershell
.\venv\Scripts\python.exe scripts/verify_anchorflow_v6.py
.\venv\Scripts\python.exe scripts/package_anchorflow_v6.py
```

Verifier chạy unit tests, actual V3 no-op trên3KITTI samples, BF16 backward, compute hooks và ONNX nonzero-head parity. Packager kiểm source-proof SHA, parent fallback SHA, notebook schema/Python syntax/defaults, markdown delimiters và ZIP CRC/hash. Không train15epoch hoặc download teacher khi kiểm local.

## Required Colab gates

1. Verify frozen bundle SHA; không dùng lẫn source từ repo/version khác.
2. Load actual Drive V5 best; kiểm architecture/epoch/SHA và exact old tensor shapes/keys.
3. Verify1600train/400val disjoint ID/rawdrive,1000test; kiểm đủ2000metric records và non-GT/non-sensor KD support train.
4. Real-data CUDA AMP smoke: finite loss/gradients/KD coverage, Dfull no-op parent.
5. Full400val epoch-1 và mỗi epoch; preserve best kể cả epoch-1, không lấy last thay best.
6. Profile parent/candidate cùng GPU/settings; final ONNX parity kiểm lại với trained V6.

Config base cố ý để parent epoch/SHA=null vì parent lấy từ Drive; **chạy notebook để resolve**, không gọi train trực tiếp bằng base config. Đổi recipe/source/parent phải dùng run folder mới. Không giữ lời hứa <0,7m trước kết quả thực nghiệm.
