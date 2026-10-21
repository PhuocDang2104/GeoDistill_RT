# N1-Fast — một mạng 582.888 params, fresh30, A100

**Chốt N1 làm gốc chất lượng, không chọn P1–P4.** Đây là nâng cấp triển khai, KHÔNG
phải một mạng neural mới hoặc kết quả đã đạt3,9ms. Không thêm packed/local branch.
Xem [phân tích kết quả](ANALYSIS.md). Chưa tự thay baseline production/full KITTI.

## Kiến trúc và thay đổi

Giữ nguyên N1: Fusion512 ở1/4, Jet-context128, U-PIR, S4=hai midpoint steps=4NFE,
fine0, encoder/readout/loss/initialization/params. Training chạy code gốc, không
chạy fast operators. Không giảm resolution, không dùng FP16/TF32 để tạo speedup.

Chỉ bản sao deployment FP32 đổi:

1. Fixed one-hot3×3 neighbour Conv → replicate-pad + chín slice cùng thứ tự.
2. Sparse innovation7×7 và phase innovation3×3: gộp ba moment trong một pooling,
   GIỮ pooling2D và thứ tự tích lũy cũ. Đã loại separable pooling vì cancellation
   variance ở vùng ít sparse support, không nới gate để giữ tối ưu đó.
3. JetODE: chuyên biệt hóa đổi gốc jet theo bốn hướng E/W/S/N, loại các tích
   với hằng0; giữ đạo hàm, trạng thái, solver và4NFE, không đổi phương trình.
4. TensorRT builder optimization level5, so với original level3; cả hai auxstreams0.

Tương đương trong số thực, **không gọi là bit-exact FP32**. Không chắc giảm runtime:
concat/layout/nhiều kernel có thể làm chậm; level5 cũng không bảo đảm nhanh hơn.
[NVIDIA mô tả trade-off builder search](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/builder-performance.html).
Mọi error gate đối chiếu với **PyTorch FP32 GỐC**, không chỉ với fastpath tự nó.

## Upload và chạy

Upload NGUYÊN folder vào `MyDrive/` hoặc `MyDrive/GeoLift_Data/`, giải nén nếu dùng
ZIP. Mở `Train_N1Fast_Fresh30_A100.ipynb`, chọn A100, chạy từ trên xuống.
Notebook tự tìm một bundle đúng đường dẫn; không upload helper vào frozen code.

Giữ run N1 cũ:

```text
MyDrive/GeoLift_RT_Runs/S4_UPIR_M3_F512_Fresh30_pair01/
  frozen_bundle/bundle_manifest.json + toàn bộ source cũ
  N1_s4_m3_f512/
    best_policy.pth
    trt_benchmark.json
    training_status.json
    train_log.csv
```

Giữ data:

```text
GeoLift_Data/teacher_subset_2000/selected_2000_ids.json
GeoLift_Data/teacher_subset_2000/kitti_trainval_2000.tar
GeoLift_Data/teacher_subset_2000/metric_coarse_train_2000.tar
GeoLift_Data/test_1000/kitti_test_1000.tar
```

Không cần P/M3 checkpoints để chạy gói này; log P chỉ là bằng chứng phân tích.
Source/config/data/checkpoint N1 được audit, không bypass hash. Gói KHÔNG chứa
data hoặc checkpoint, không tự download KITTI/teacher mới.

## Preflight trước khi tốn30 epoch

Hai engine original và fast chạy **cùng checkpoint N1 cũ**. Đo đủ400 ảnh, same
session, compare đủRMSE/iRMSE/MAE/boundary và runtime. Nếu RMSE không<0,98,
iRMSE không<3,2 hoặc mean>3,9ms (kể cả reverse-order repeat), **dừng trước fresh30**.
Đây là báo fail thật, không auto-fallback/nới pixel gate/chọn fastest repeat.

Nếu preflight đạt, chạy một fresh30 xác nhận đúng yêu cầu. Mạng không đổi nên
retraining KHÔNG bắt buộc để dùng operator upgrade với checkpoint cũ. Không claim
fresh30 sẽ tăng metric. Không warm-start trained N1; ImageNet encoder/seed42/B4,
1600train/400val, PyTorch BF16+FP32geometry/loss, AdamW fused, metric KD ON train-only,
**relative OFF hoàn toàn**, loss/teacher/augmentation/selection như N1.

Output mới: `GeoLift_RT_Runs/S4_UPIR_N1Fast_Fresh30_pair01/`. Giữ RUN_TAG để resume.
Notebook lấy source từ `frozen_bundle` của run khi có, khôi phục last.pth và
optimizer/RNG/loader tại epoch boundary; ngắt giữa epoch có thể lặp epoch đó.
Không early stop. Đã có last.pth thì không chặn resume chỉ vì preflight tốc độ
phiên mới lệch; final benchmark vẫn phải đo/qualify lại. Không sửa source khi resume.

## Benchmark final chính xác

Rebench old N1 và Q1 mới trong **một current session**. Mỗi checkpoint có original
engine và fast engine: tách hiệu ứng deployment khỏi khác biệt retraining.

- TensorRT10.13.3 FP32 strongly-typed, TF32 OFF; dedicated stream + CUDA Graph.
- Actual400 predictions/25.424.992 validGT pixels; input freshness A/B/A.
- Pixel mean≤1mm/max≤6cm so với originalFP32; nativeBF16 metric drift RMSE≤2mm,
  iRMSE≤0,02. Original6cm protocol, không giả là gate5cm đã đạt.
- B1,100distinctGPUimages,100warmup/300timediterations. Synchronized wall gồmD2D,
  loại decode/preprocessing/H2D/build/capture; FPS=1000/mean, median/P95 riêng.
- Đo reverse order bổ sung, báo riêng, không chọn lượt nhanh nhất. Cả hai lượt
  mean fast≤3,9ms để shortlist. Không đồng nhất Torch peak với fullTRT footprint.
- Q1 xuất1000test predictions+10RGB/depth pairs. Test không có publicGT.

`comparison.csv/json` gồm4rows (old/fresh × original/fast). `baseline_candidate.json`
chỉ chọn nếu số CHƯA LÀM TRÒN RMSE<0,98/iRMSE<3,2/mean≤3,9ms; nếu fail thìnull.
Mốc3,9ms là cap, nhanh hơn cũng đạt. Chưa đo Orin. Trước full KITTI cần confirm
seed/validation độc lập, không đưa400developmentframes vào training rồi báo chúng
như held-out. Các con số P/N1 chỉ là development evidence, không phải officialtest.

Kiểm chứng local: `local_verification.json`. Không có A100 tại máy local; không
claim đã train, build TensorRT hay đạt latency mục tiêu trên GPU.
