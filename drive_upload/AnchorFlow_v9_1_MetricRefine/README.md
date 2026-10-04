# AnchorFlow V9.1 — Metric Refine

Bundle độc lập để cải thiện V9 đã train; giữ nguyên V8/V9 cũ. **Mục tiêu RMSE <0,9 m chưa được chứng minh**.

## Chạy ngay trên Colab

1. Giải nén/upload **cả folder AnchorFlow_v9_1_MetricRefine** vào MyDrive.
2. Mở **02_Train_V9_1_Refine15_or_Fresh30.ipynb**, chọn GPU BF16: L4/A100/Blackwell.
3. Giữ MODE="finetune"; chạy từ trên xuống. Notebook tự load model-only V9 best epoch23 kèm bundle,
   tạo optimizer/scheduler mới và train thêm tối đa **15 epoch**, early-stop patience5/min8.
4. Muốn student mới từ epoch0: đổi MODE="fresh" trước khi chạy; ImageNet encoder, tối đa30 epoch,
   early-stop patience7/min15. Folder run khác, không ghi đè fine-tune.

Bạn đã có relative teacher từ V9 nên **không cần chạy notebook 01**.
Notebook 01 chỉ dành cho tạo/cache-resume DA3MONO-LARGE khi thiếu relative TAR; code/revision/recipe giữ nguyên.

~~~text
MyDrive/
├── AnchorFlow_v9_1_MetricRefine/          # folder này, khoảng vài MiB
├── GeoLift_Data/
│   ├── teacher_subset_2000/
│   │   ├── selected_2000_ids.json
│   │   ├── kitti_trainval_2000.tar       # RGB / sparse / GT / K / split
│   │   ├── metric_coarse_train_2000.tar # D_cm / C_cm
│   │   └── relative_teacher_2000_DA3MONO_LARGE.tar
│   └── test_1000/kitti_test_1000.tar
└── GeoLift_RT_Runs/
    └── AnchorFlow_v9_1_MetricRefine_FromV9Best15_ES_cudnn_safe/
        ├── source_bundle/
        ├── data_contract.json
        └── dual_teacher/
            ├── best.pth / last.pth
            ├── initial_val_metrics.json / val_metrics.json
            ├── train_log.csv / train_log.jsonl / training_status.json
            ├── v9_1_vs_v8_v9_accuracy.csv / v9_1_vs_v8_v9_efficiency.csv
            ├── profile.json / export_report.json
            └── kitti_test_predictions.zip
~~~

GeoLift_RT_Runs phân biệt hoa/thường trên Colab. Test thiếu thì tải từ KITTI chính thức.
Data được extract vào SSD /content; cần **>25 GiB SSD trống**. Hai teacher chỉ cache 1.600 train,
không load teacher cho val/test. Không cần teacher weights hoặc repo đầy đủ.

## Có gì thay đổi?

- Giữ encoder, ba bước learned jet dynamics và five-jet consensus.
- Khởi tạo slope bằng **minmod liên tục**, giảm tính nhạy khi hai đạo hàm một phía trái dấu/gần bằng nhau.
- Head metric nhỏ trên grid1/4: đọc feature decoder + sparse innovation lân cận + disagreement.
  Output mới zero-init và cộng **bên trong tanh cũ**; không tăng depth correction bound.
- GT ưu tiên hơn: tăng RMSE/tail supervision; tail Huberized không hard-cap gradient.
- Metric KD yếu hơn; relative chỉ giữ gradient cấu trúc, bỏ ordinal compute. Không teacher trong inference.

Xem [phân tích V9](ANALYSIS_V9.md), [kiến trúc/công thức](ARCHITECTURE.md) và
[kiểm chứng](VERIFICATION.md).

## Resume và so sánh đúng

Run bị ngắt: chạy lại notebook từ đầu, giữ nguyên MODE/RUN_TAG/config/data; tự resume last.pth.
Thay code hoặc hyperparameter: dùng RUN_TAG mới, không sửa manifest để bypass.
Colab có thể sửa .ipynb UI; runtime không hash UI nhưng vẫn hash source, checkpoint và _contract.json.

Checkpoint parent chỉ chứa **model + provenance**, không optimizer/scaler/RNG. Initial validation đo lại
sau minmod; vì minmod thay geometry nên không giả định initial output bằng V9.
Nếu mọi epoch mới kém hơn, best.pth có thể là initial epoch−1: notebook báo rõ **chưa có gain học mới**.

V9 cũ dùng30 epoch; default V9.1 thêm tối đa15, tổng ngân sách45.
Parent best epoch23 đã nhận24 epoch update; state sau15 epoch thêm tương ứng39 epoch update.
So sánh này **không phải ablation architecture cùng ngân sách**; muốn nghiên cứu causal cần control cùng recipe.

Không có GPU local để xác nhận RMSE mới hay latency mới. Không bảo đảm kết quả <0,9; không dùng
GT/test labels để chỉnh output. Export parity fail vẫn giữ accuracy/test reports, không nới tolerance.

## CUDA/cuDNN smoke engine failure — compatibility update

Lỗi FIND was unable to find an engine trong MobileNet Conv xảy ra trước decoder.
Data gate PASS không chứng nhận kernel/backend; checkpoint finite cũng không đủ để bảo đảm cuDNN chạy.
Traceback này không xác định chắc chắn nguyên nhân sâu: autotuning, workspace/VRAM, runtime/library,
hoặc convolution layout/kernel support đều cần kiểm tra.

Bản cập nhật giữ BF16 và channels-last nhưng đặt **cudnn_benchmark=false** trong device_setup của
chính subprocess train/smoke. Đặt flag ở notebook parent là không đủ, vì train chạy process khác.
CuDNN vẫn bật; không tự tắt backend, đổi FP16, retry/skip batch, hoặc bỏ qua smoke.
Tham khảo [PyTorch 2.11 — cuDNN benchmark](https://docs.pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.benchmark).

Sau upload thay bundle cũ bằng bản mới, mở lại notebook02 mới và chạy các cell từ đầu.
Giữ default RUN_TAG=_cudnn_safe để không ghi đè frozen snapshot/config cũ.
WORK/data cache giữ nguyên, không cần generate teacher lại; không xóa data hay run cũ.
Nếu chưa train, không có training progress để resume ở run cũ.

Nếu smoke vẫn fail, gửi **smoke_gpu_failure_report.json** ở folder dual_teacher của run mới:
report chứa last Conv input/weight shape, dtype/stride/groups/dilation, free/reserved VRAM,
Torch/CUDA/cuDNN và backend flags. Report chỉ quan sát, không tự thay tensors/precision.
Một lỗi engine vẫn có thể cần sửa runtime; local CPU tests không xác nhận GPU đã được khắc phục.
Profile mới ghi backend flags; không so latency trực tiếp với profile historical thiếu các flags đó.
