# Verification và trạng thái bàn giao

## Đã kiểm

- ZIP15epoch v3: log đủ epoch0–14, best14, migration/config/profile/metrics/checkpoint.
- Full400val parent CPU FP32: RMSE1,045427m, cùng25.424.992pixel. Bias/tail/worst20 lưu parent_prediction_audit.json.
- V4 warm-start:697 tensors v3,572.017params giữ nguyên,1.285.704params mới, không dropped key.
- V3/V4 no-op: output khớp; checkpoint thật và3KITTI samples352×1216 có max difference0.
- **19 unit/integration tests PASS**: migration, AMP gradients, KD exclusion, teacher cache train-only, global metric, archive safety, train/checkpoint/resume cho cả v3-control lẫn v4 và recipe comparison. Synthetic test metrics không phải KITTI accuracy.
- Params/MAC hooks:1.857.721params /6.343.626.560countedMAC.
- ONNX static batch1/352×1216, opset17 checker và ORT CPU parity sparse/empty-sparse đã PASS với graph structural chưa train v4. Notebook sẽ export lại best của run thực.
- Notebook AST/nbformat và SHA/CRC ZIP kiểm bằng scripts/package_anchorflow_research.py.
- Sau pull remote, **6 S3 augmentation contract tests PASS**. Không bật augmentation đó vào AnchorFlow recipe; chưa chạy toàn bộ server/Docker integration suite.

## Chưa kiểm

**Chưa train15epoch v4 trên GPU, chưa biết validation gain của v4, chưa đo v4 GPU latency/VRAM, chưa build TensorRT.** Không gọi no-op1,045m là kết quả v4 đã học tốt hơn.
Không có đủ teacherTAR2k local để thay thế data gate Colab; notebook kiểm lại toàn bộ nguồn Drive.

## Reproduce

```powershell
python -m unittest discover -s drive_upload/AnchorFlow_Research -p "test*.py" -v
python scripts/audit_anchorflow_predictions.py
python scripts/package_anchorflow_research.py
```

Checkpoint init_v3_best.pth là file trusted của người dùng, SHA256:

```text
1c767a871cb57336ccc49460f84f87afb81b30d95f7e746f34e7e908280a3d07
```

Notebook xác minh hash trước torch.load(weights_only=False). Không thay bằng checkpoint lạ. File weight/ONNX/ZIP được exclude khỏi Git, nhưng parent checkpoint có trong upload bundle.
