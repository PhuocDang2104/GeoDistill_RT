# AnchorFlow V9 — Multi-Jet Consensus · Dual Teacher

> Gói train mới từ epoch 0, tối đa 30 epoch + early stop.  
> Đúng 2 notebook; dùng lại data và metric teacher trên Drive. Không chứa data/weights/checkpoint.  
> Đã kiểm tra implementation; **chưa có accuracy hoặc GPU latency của V9 đã train**.

## Chạy ngay

1. Upload **cả folder** này thành `MyDrive/AnchorFlow_v9_Consensus/` (giải nén ZIP trước). Không cần upload repo đầy đủ.
2. Mở [notebook 01](01_Generate_Relative_Teacher_DA3MONO_TAR2000.ipynb), chọn GPU, chạy từ trên xuống để generate relative teacher cho đúng 2.000 RGB.
3. Mở [notebook 02](02_Train_V9_DualTeacher_Fresh30_CompareV8.ipynb) trong runtime GPU mới, chạy từ trên xuống để train/val/test/profile/compare.
4. Notebook 02 mặc định `EXPERIMENT="v9"`. Muốn đối chứng cùng recipe: chạy lại notebook này với `EXPERIMENT="v8_control"`; không cần notebook thứ ba.

Nếu folder/data/output của bạn đặt khác tên, sửa `BUNDLE`, `DRIVE_DATA`, `DRIVE_RUNS` trong cell mount. **`GeoLift_RT_Runs` phân biệt hoa/thường**; không trộn với folder `GeoLift_RT_runs` cũ.

## Drive cần có gì?

```text
MyDrive/
├── AnchorFlow_v9_Consensus/                 ← upload folder code này
└── GeoLift_Data/
    ├── teacher_subset_2000/
    │   ├── selected_2000_ids.json           ← có sẵn
    │   ├── kitti_trainval_2000.tar          ← RGB + sparse + GT + K + split
    │   ├── metric_coarse_train_2000.tar     ← D_cm/C_cm có sẵn
    │   └── relative_teacher_2000_DA3MONO_LARGE.tar  ← notebook 01 tạo
    └── test_1000/
        └── kitti_test_1000.tar              ← anonymous test RGB/sparse/K
```

Test TAR thiếu thì code tải official KITTI depth-selection ZIP. Nếu đã có test TAR, code dùng trực tiếp; không lấy 1.000 test từ 2.000 train/val.

| Thành phần | Notebook 01 | Notebook 02 |
|---|---|---|
| KITTI RGB/sparse/GT/K, split 1.600/400 | Đọc RGB; kiểm tra layout, không đưa GT/sparse/K vào teacher | Dùng RGB/sparse/K làm input; GT làm supervision |
| Metric `D_cm,C_cm` | Không dùng | Metric KD, chỉ 1.600 train |
| Relative `R_T,C_T` | Generate 2.000 từ RGB bằng DA3MONO-LARGE | Chỉ load 1.600 train; 400 val audit completeness, không supervision |
| Anonymous KITTI test 1.000 | Không dùng | Inference không teacher; không public GT |

Không cần geometry_fused, DA2 raw, DSINE, normal teacher hoặc teacher checkpoint cũ. Metric cache được giữ nguyên nguồn bạn cung cấp; gói này không khẳng định cache đó chỉ đến từ DMD3C.

## Runtime và dung lượng

- Train dùng **BF16-capable GPU: L4/A100/Blackwell**; recipe không hỗ trợ T4/FP16 fallback. Chọn GPU đúng trước khi chạy.
- Notebook 01: tối thiểu 12 GiB **SSD của Colab `/content`** trống; download teacher khoảng 1,3 GB. Default full-frame long-side 1.232, RGB flip TTA, batch 1.
- Notebook 02: tối thiểu 20 GiB SSD, khuyến nghị 25 GiB trở lên. Hai cache train float32 khoảng **5,1 GiB/cache**, cộng KITTI và checkpoint. Không train trực tiếp từ Drive mount.
- Drive lưu per-image relative `.npz` để resume và final TAR: **hai bản relative cùng tồn tại**. Dung lượng TAR được báo sau generate, không ước lượng cứng.
- Đây là dung lượng Colab/Drive, không phải SSD máy Windows. Gói upload không chứa teacher model ~1,3 GB.

Teacher chỉ generate offline. Hai model teacher **không chạy trong student forward/train step/inference**.

## Output ở đâu?

```text
MyDrive/GeoLift_Data/teacher_subset_2000/
├── relative_DA3MONO_LARGE_generation/{recipe.json,progress.json,<ID>.npz}
├── relative_teacher_2000_DA3MONO_LARGE.tar
├── relative_teacher_2000_report.json
└── relative_teacher_preview.png

MyDrive/GeoLift_RT_Runs/AnchorFlow_v9_Consensus_DualTeacher_Fresh30_ES/
├── source_bundle/                          ← source/config frozen
├── data_contract.json
├── data_preview.png
└── dual_teacher/
    ├── {best.pth,last.pth}
    ├── {run_manifest.json,resolved_config.json,training_status.json}
    ├── {train_log.csv,train_log.jsonl,train_console.log}
    ├── {initial_val_metrics.json,best_val_metrics.json,val_metrics.json}
    ├── {v9_learning_curves.png,loss_budget.csv}
    ├── {v9_vs_v8_accuracy.csv,v9_vs_v8_efficiency.csv}
    ├── {profile.json,export_report.json,anchorflow_edge_fp32.onnx}
    └── kitti_test_predictions.zip
```

Checkpoint/log được backup từng epoch; progress từng batch hiện trong cell train. Resume bằng chạy lại các cell với **cùng bundle/config/run name**. Muốn experiment khác dùng `RUN_TAG` mới; không ép load checkpoint khác source/precision/data.

Colab tự sửa metadata/output của `.ipynb`, nên runtime không checksum UI notebook. Source, config và bản notebook `_contract.json` vẫn checksum nghiêm ngặt. Không sửa Python trong folder sealed rồi bỏ qua gate; cần build/verify/seal lại hoặc experiment mới.

## Đọc gì để hiểu và quyết định?

- [ARCHITECTURE.md](ARCHITECTURE.md): flow, công thức chính xác, teacher roles, objective và compute.
- [ANALYSIS_V8.md](ANALYSIS_V8.md): số liệu V8, phản biện đề xuất V9, hypothesis và acceptance.
- [VERIFICATION.md](VERIFICATION.md): những gì đã test và chưa test.

Baseline V8 best: **RMSE 0,997549 m / iRMSE 3,394592 km⁻¹**. Mục tiêu thử nghiệm V9 là RMSE <0,90 m trước; <0,8 m là mục tiêu nghiên cứu tiếp, **không bảo đảm trên 2.000 ảnh**. So sánh historical V8 được ghi rõ không matched; V8 control fresh/BF16 dùng cùng notebook để có đối chứng recipe tốt hơn.
