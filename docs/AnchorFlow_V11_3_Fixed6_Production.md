# V11.3 Fixed6 — GPU-member production

[Standalone package and full instructions](../production/AnchorFlow_V11_3_Fixed6_FullKITTI/README.md).

Run riêng, không sửa gói adaptive đã push trước đó. Coarse **classic RK3 × 2
half-steps**, exactly **6 NFE**, T=1, observations t=0.5/1. Fine RK2 giữ **2 NFE**;
**tổng 8 NFE**. Giữ neural field, bounded chart, readout, objective và 584.845
params của V11.3. Không learned h/stop, không adaptive solver capped NFE.

## Member pull + run

```bash
git pull --ff-only origin main
bash production/AnchorFlow_V11_3_Fixed6_FullKITTI/run_full_flow.sh \
  --storage-root /mnt/ssd/geolift \
  --accept-kitti-license \
  --cuda-arch 12.0
```

Cho **5060 Ti 16 GB**: mặc định B2, native BF16 CNN + FP32 geometry/loss, fresh
student + encoder ImageNet, max40/early stop. Linux/Docker/NVIDIA Toolkit và SSD
persistent ≥1 TB cần có sẵn. Chưa đo VRAM/throughput trên 5060 Ti; cả DMD3C++,
DA3MONO-LARGE và real-batch backward phải pass GPU gate. B2 không bảo đảm teacher
vừa 16 GB. Không tự hạ resolution, skip batch, đổi FP16 hoặc đổi loss.

Data/download/generate/cache/recovery giữ flow cũ: 85.898 train / 6.852 val /
1.000 anonymous test; train-only metric+relative cache. Flow kết thúc bằng
best-policy validation, test ZIP và 10 cặp RGB/depth màu. Output mặc định:
`STORAGE_ROOT/runs/v11_3_fixed6_fresh40_b2/`.

Student B2 OOM: chủ động dừng, chạy thêm `--batch-size 1` cho **run mới**, không
resume optimizer B2. B2 là batch thật, accumulation=1; không matched với baseline
B4. Không train adaptive và fixed6 đồng thời trên cùng GPU. Resume chỉ cùng
source/config/data/environment; không chuyển checkpoint adaptive cũ vào đây.

CPU solver/time/gradient/model/cache/resume contracts và manifest provenance
được ghi trong `verification.json`. GPU build/teacher/student/full train chưa
được qualification tại máy đóng gói; không cam kết gain metric hay runtime.
