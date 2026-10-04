# AnchorFlow V9 — Multi-Jet Consensus + Dual Teacher

> Implementation mới để thử nghiệm, chưa có metric V9 đã train. V8 giữ nguyên baseline.

V9 giữ MobileNetV4 + sparse/FPN + ba shared JetDynamics updates của V8. Điểm mới nằm ở **4→2 readout**: center/E/W/S/N jets cùng analytic query đúng phase, hợp nhất inverse depth theo compatibility/barrier và dùng disagreement để điều chỉnh blend gate. Chỉ thêm 504 parameters; deploy không cần teacher.

Train lại epoch0, ImageNet RGB initialization, metric cache cũ + DA3MONO-LARGE relative structure cache, BF16, max30 + early-stop. GT/sparse priority; 1.600 train / 400 val / 1.000 anonymous test giữ nguyên. Historical V8 RMSE0,997549m chỉ là reference, không matched causal ablation.

Tài liệu và gói canonical:

- [Folder upload Drive](../drive_upload/AnchorFlow_v9_Consensus/README.md): hai notebook, đường dẫn, dung lượng và output.
- [Kiến trúc/công thức](../drive_upload/AnchorFlow_v9_Consensus/ARCHITECTURE.md).
- [Phân tích V8 và phản biện v9_suggest](../drive_upload/AnchorFlow_v9_Consensus/ANALYSIS_V8.md).
- [Phạm vi kiểm tra](../drive_upload/AnchorFlow_v9_Consensus/VERIFICATION.md).
- [ZIP upload](../drive_upload/AnchorFlow_v9_Consensus.zip).

Chạy notebook01 generate relative teacher, rồi notebook02 train/val/test/compare. Optional `EXPERIMENT="v8_control"` ở notebook02 tạo đối chứng V8 fresh/BF16; không dùng V8 student checkpoint để khởi tạo V9.
