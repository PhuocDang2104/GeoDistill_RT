# AnchorFlow V10

Baseline kỹ thuật: V9.1 fixed3; V8 vẫn là historical accuracy/MAE/iRMSE reference.
V10 fresh student, ImageNet RGB-only initialization, tối đa40epoch/early-stop min20patience8.
Thay fixed3 integration bằng shared reaction–transport field + scalar learned step + exit2/3/4.

- [Kiến trúc, công thức và NODE limitations](../drive_upload/AnchorFlow_v10_AdaptiveJet/ARCHITECTURE.md)
- [So sánh thực đo V8/V9/V9.1 và chốt baseline](../drive_upload/AnchorFlow_v10_AdaptiveJet/BASELINE_COMPARISON.md)
- [Folder upload / data / notebook / resume](../drive_upload/AnchorFlow_v10_AdaptiveJet/README.md)
- [Notebook train fresh40](../drive_upload/AnchorFlow_v10_AdaptiveJet/02_Train_V10_Fresh40_EarlyStop.ipynb)

V10 chưa được train GPU local; không bảo đảm RMSE<0.9. Không có teacher trong deploy.
