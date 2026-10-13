# AnchorFlow V11.3 Fixed6 — full KITTI / 5060 Ti 16 GB

Gói riêng để member **pull và chạy bằng lệnh**, không cần Drive/notebook.
Coarse NODE fixed **6 NFE**; fine NODE giữ **2 NFE**; tổng hai field **8 NFE**.
Student fresh từ epoch 0, RGB encoder ImageNet pretrained; max **40 epoch**,
early stop dual-metric. Không thay bản adaptive tại `../AnchorFlow_V11_3_FullKITTI`.

## Chạy trên server

Linux + Python3 + Docker Compose + NVIDIA Container Toolkit; driver tương thích
Blackwell/CUDA 12.8; SSD persistent **≥1 TB**, RAM khuyến nghị ≥32 GB.
Đọc KITTI terms trước khi dùng flag xác nhận. Không tự cấp giấy phép thương mại.

```bash
git pull --ff-only origin main
bash production/AnchorFlow_V11_3_Fixed6_FullKITTI/run_full_flow.sh \
  --storage-root /mnt/ssd/geolift \
  --accept-kitti-license \
  --cuda-arch 12.0
```

Đổi `/mnt/ssd/geolift` thành SSD thật của member. Launcher mặc định **batch 2**,
BF16 CNN + FP32 RHS/state/solver/loss, channels-last và fused AdamW. Không tự đổi
sang FP16, skip batch, giảm batch hoặc giả lập BF16 khi lỗi.

Kiểm tra kế hoạch mà không ghi file/build/download:

```bash
bash production/AnchorFlow_V11_3_Fixed6_FullKITTI/run_full_flow.sh \
  --storage-root /mnt/ssd/geolift --accept-kitti-license --cuda-arch 12.0 --dry-run
```

Job chạy detached; launcher in container ID cùng lệnh `docker logs -f` và
`docker stop -t 600`. Chạy lại **đúng cùng lệnh** để resume sau pause/lỗi đã sửa.
Không pull source đè vào run đang chạy. Không chạy đồng thời bản adaptive và
fixed6 trên cùng GPU0: package riêng không phải cơ chế chia GPU.

Nếu student B2 OOM, dừng container rồi **chủ động tạo run mới** với `--batch-size 1`.
Launcher tách work/config/run theo batch, không ép resume optimizer B2 sang B1.
B4 có thể chọn trước run bằng `--batch-size 4`. Không có gradient accumulation;
B2 thực sự cập nhật mỗi 2 ảnh, không phải effective B4. Với 85.898 ảnh, B2 có
42.949 updates/epoch. Đây không phải matched ablation với adaptive B4.

**16 GB chưa được qualification:** giảm NFE/student batch không giảm VRAM của
teacher. DMD3C++ (có DA3METRIC-LARGE bên trong) và DA3MONO-LARGE có GPU smoke bắt
buộc trước full generation. OOM tại teacher không được giải quyết bằng giảm
batch student. Chưa cam kết throughput, VRAM tối thiểu hoặc metric full KITTI.

## Solver và kiến trúc

```text
RGB/S/M/K → V11.3 encoder + sparse pyramid + decoder → D0 at 1/4
  → minmod jet + bounded chart z0
  → shared learned RHS, classic RK3 × 2 half-steps (6 NFE, T=1)
      z(0.5) → trajectory GT loss
      z(1)   → D4 + five-jet consensus / metric innovation
  → D2 → fine RK2 (2 NFE) → phase lift / detail / PIR
  → learned soft sensor fusion → Dfull
```

Một bước classic explicit RK3, h=0.5:

$$
k_1=f(t,z),\quad k_2=f(t+h/2,z+hk_1/2),\quad
k_3=f(t+h,z-hk_1+2hk_2),
$$

$$
z_{next}=z+\frac{h}{6}(k_1+4k_2+k_3).
$$

Hai bước tại t=0 và t=0.5; sáu RHS evaluations tại
**0, 0.25, 0.5, 0.5, 0.75, 1**. Hai calls tại 0.5 có state khác nhau;
không được gộp như cùng field input. Shared neural weights, state feedback,
actual pseudo-time và cùng horizon T=1. Autograd trực tiếp qua cả sáu calls.
Giữ half-time supervision, không cần dense interpolation.

Đây là fixed numerical integration của learned NODE, **không phải adaptive
Bosh3 capped tại 6 NFE**. Không error estimator/reject/learned h/learned stop;
không claim bảo đảm accuracy hoặc stability nonlinear chỉ từ RK3. Cấu trúc
encoder/jet/readout/loss và **584.845 params** giữ nguyên V11.3; chỉ solver đổi.
Loss teacher và trajectory không bỏ. Solver khác có thể đổi prediction/metric;
không gọi output tương đương adaptive khi chưa benchmark.

## Flow full data

```text
native BF16 + student forward + disk preflight
→ download KITTI sparse/GT/raw RGB/calibration/test
→ paired index + ID/raw-drive disjoint + SHA
→ GPU smoke DMD3C++ và DA3MONO-LARGE (process riêng)
→ generate train-only metric D_cm/C_cm và relative R_T/C_T
→ full cache audit → real-batch student backward smoke, coarse6/fine2
→ fresh max40 train + validation mỗi epoch + checkpoint/early stop
→ best-policy metrics + test1000 ZIP + 10 PNG RGB/depth color
```

**85.898 train / 6.852 val / 1.000 anonymous test**. Không phải 90k train.
Depth uint16/256 theo mét; preprocess 352×1216 giữ protocol V11.3.
Cache teacher chỉ train, không val/test; anonymous test không public GT nên
không báo test RMSE. Teacher chạy tuần tự, không online trong student inference.
Official teacher code/checkpoint revisions, confidence recipes, atomic cache
và S3-style transactional recovery giữ nguyên gói adaptive production.

Hai teacher cache float32 cần khoảng **548 GiB** chưa tính RGB/GT/sparse/weights.
Cùng storage root có thể reuse cache cũ **nếu** recipe/environment/input SHA
khớp; không giả định cache bất kỳ đều dùng được, không lược bỏ checksum gate.

## Output / resume

```text
STORAGE_ROOT/
├── data/kitti/                         # shared dataset
├── data/teachers_v11_3_full/            # shared only with compatible cache recipe
├── data/v11_3_fixed6_work_b2/           # independent pipeline/index/STOP state
├── config/v11_3_fixed6_b2.json          # immutable host runtime recipe
└── runs/v11_3_fixed6_fresh40_b2/
    ├── smoke_report.json               # coarse6 + fine2 + GPU peak reserved
    ├── resolved_config.json, run_manifest.json
    ├── checkpoints/                    # last/best/inverse/joint/policy pointers
    ├── train_log.csv, train_log.jsonl, training_status.json
    ├── val_metrics_policy.json, final_summary.json
    ├── kitti_test_predictions.zip
    └── test_preview/                   # 10 RGB/depth pairs
```

Checkpoint chứa optimizer, RNG, consumed-batch cursor, loss totals và selection;
config/source/data/environment đổi thì resume bị từ chối. Không resume checkpoint
adaptive/2k/Drive cũ; learned field/readout compatible không đồng nghĩa optimizer
protocol compatible. Dừng SIGTERM/STOP commit checkpoint; lỗi không bị mask.

## Verification

`bundle_manifest.json` kiểm tra raw-byte SHA, gồm provenance file giữ nguyên.
`verification.json` ghi CPU tests, solver/gradient/model contracts và giới hạn.
GPU teacher/student smoke, Docker build và full training **chưa đo tại máy đóng
gói**. Member phải pass các gate trên GPU thật; CPU tests không thay qualification.

```bash
cd production/AnchorFlow_V11_3_Fixed6_FullKITTI
python pipeline.py verify
python -m unittest test_production -v
```
