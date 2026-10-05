# AnchorFlow V11.3 — full KITTI GPU-server bundle

Standalone training package, dùng pattern adapter + persistent volume của S3 server. **Không cần notebook, Google Drive hay repo gốc để chạy.** Model V11.3 được copy nguyên byte; không thay solver, jet, fine RK2, phase readout hay soft sensor fusion.

Trạng thái: contract tests CPU đã chạy; **chưa qualification trên GPU member, chưa chạy teacher full KITTI hoặc train 90k tại máy này**. GPU smoke trên dữ liệu thật là gate bắt buộc, không phải tuyên bố “production-tested”.

## Member: pull và chạy một lệnh

GPU server cần có Linux, Python3, Docker Compose và NVIDIA Container Toolkit. Đọc KITTI terms trước khi dùng flag xác nhận. Không cần host `pip install`, không sửa file tracked hoặc tải data thủ công:

```bash
git clone --depth 1 --branch main https://github.com/PhuocDang2104/GeoDistill_RT.git
cd GeoDistill_RT
# Nếu đã clone: git pull --ff-only origin main
bash production/AnchorFlow_V11_3_FullKITTI/run_full_flow.sh \
  --storage-root /mnt/ssd/geolift \
  --accept-kitti-license
```

`/mnt/ssd/geolift` phải là SSD persistent đủ dung lượng của member, không phải một path bắt buộc. Launcher tự phát hiện compute capability GPU0; nếu driver không hỗ trợ query, thêm `--cuda-arch 8.0` cho A100 (hoặc đúng SM của GPU thực tế). Runtime config được ghi riêng tại `STORAGE_ROOT/config/`, không làm repo dirty. Job chạy detached; launcher in lệnh xem log và stop theo đúng container ID. Kết quả tại `STORAGE_ROOT/runs/v11_3_full_fresh40/`.

Kiểm tra kế hoạch mà chưa ghi file/build/download:

```bash
bash production/AnchorFlow_V11_3_FullKITTI/run_full_flow.sh \
  --storage-root /mnt/ssd/geolift --accept-kitti-license --cuda-arch 8.0 --dry-run
```

Resume: chạy lại **đúng cùng lệnh/cùng storage root** sau khi giải quyết lỗi hoặc pause. Không `git pull` đè source của một run đang chạy rồi ép resume; source/protocol thay đổi cần run mới. Nếu chỉ lấy source, không cần ZIP/notebook/Drive.

## 1. Flow tự động

```text
GPU + native BF16/student-forward + disk preflight
  → download annotated / projected LiDAR / selection ZIP
  → download raw RGB theo drive + calibration
  → index và kiểm tra RGB/S/GT/K, SHA, split
  → real-input GPU smoke cả hai teacher, trước khi generate full
  → DMD3C++ generate metric teacher cho toàn bộ TRAIN
  → DA3MONO-LARGE generate relative teacher cho toàn bộ TRAIN
  → full checksum + valid-pixel coverage audit
  → real batch full-objective forward/backward smoke
  → fresh V11.3, ImageNet RGB encoder, max 40 epochs
  → evaluate best_policy + anonymous test + 10 RGB/depth PNG
```

“90k KITTI” **không phải 90k ảnh train**. Config kỳ vọng **85.898 train + 6.852 validation** (92.750 ảnh có GT), và **1.000 anonymous test** riêng. Không lấy validation/test vào train, không dùng teacher cho validation/test, không dùng subset 2.000 ảnh. Index dừng nếu số lượng/path/K không khớp; không tự bỏ ảnh lỗi để đủ chạy. [Official KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

Metric trên full validation mới không được so trực tiếp với RMSE trên 400 ảnh subset cũ như một matched ablation.

## 2. Member chạy bằng Docker — khuyên dùng

Yêu cầu: Linux, Docker Compose + NVIDIA Container Toolkit, driver tương thích CUDA 12.8, native BF16 GPU SM≥80. Khuyên GPU **24 GiB VRAM trở lên**, RAM 32 GiB và **SSD persistent ≥1 TB**; chưa đo minimum VRAM của DMD3C++ trên GPU của bạn. CPU/T4 không tự fallback. Một GPU cho pipeline này; không claim DDP/multi-GPU.

1. Gửi nguyên folder này hoặc ZIP cùng tên cho member. Không gửi dataset/checkpoint vào Git.
2. Đọc [KITTI data terms](https://www.cvlibs.net/datasets/kitti/raw_data.php), rồi sửa `config.json`: `accept_kitti_license=true`. Không mặc định coi đóng gói kỹ thuật là giấy phép thương mại.
3. Chọn host SSD bằng biến môi trường; `/data`, `/runs`, `/cache` trong config giữ nguyên:

```bash
cd AnchorFlow_V11_3_FullKITTI
export DATA_DIR=/mnt/ssd/geolift-data
export RUNS_DIR=/mnt/ssd/geolift-runs
export CACHE_DIR=/mnt/ssd/geolift-cache
# A100: 8.0; RTX 30xx: 8.6; RTX 40xx: 8.9; Hopper: 9.0.
export CUDA_ARCH_LIST=8.0
bash run_full_flow.sh --accept-kitti-license
docker compose logs -f trainer
```

GPU Blackwell cần kiến trúc đúng GPU, ví dụ SM120 đặt `CUDA_ARCH_LIST=12.0`; chọn CUDA/driver hỗ trợ thực tế. Không mặc định tất cả RTX PRO Blackwell cùng SM. Build biên dịch `BpOps` cho **đúng torch/CUDA trong image**. Không copy `.so` từ máy S3 cũ. Teacher và student chạy thành process riêng để không xung đột module `models/data/depth_anything_3`.

Compose chạy detached; đóng SSH không dừng job. `restart: no` là chủ ý: lỗi dữ liệu/GPU không được lặp mù. Sau khi sửa nguyên nhân hoặc pause, chạy lại cùng lệnh để resume. Đổi code/objective/batch/teacher thì dùng work/cache/run mới, không ép resume.

## 3. Nếu GPU member không được dùng Docker

Trong environment CUDA PyTorch có `nvcc` và compiler tương thích:

```bash
cd AnchorFlow_V11_3_FullKITTI
python -m pip install -r requirements.txt
# Sửa các path config sang persistent disk của member;
# third_party_root phải là folder được phép ghi, không bắt buộc /opt.
python bootstrap_teachers.py --config config.json
python -m unittest test_production -v
python -u pipeline.py all --config config.json
```

Có thể chạy lệnh cuối trong `tmux`/`screen`/job scheduler. Không chạy Docker-in-Docker trong pod cũ. Nếu job scheduler cấp một GPU/MIG, `CUDA_VISIBLE_DEVICES` phải trỏ GPU đã cấp. Các GPU khác không được script sử dụng tự động.

## 4. Teacher thực sự là gì?

| Target | Teacher và input | Cache cho student |
|---|---|---|
| Metric | **Official Sharpiless/DMD3Cpp**, `Pre_MF_Post_Residual_Norm_fast`, official `result_ema.pth`; RGB + sparse + K. Trong teacher này còn có DA3METRIC-LARGE frozen. | D_cm tương đương `teacher` theo mét; C_cm tương đương `confidence` |
| Relative | **DA3MONO-LARGE**, RGB-only, single-view, horizontal-flip TTA; không sparse/GT/K/metric input. | R_T: standardized relative inverse depth, near-high; C_T: TTA agreement |

Sources/checkpoints được pin revision; checkpoint completion còn pin **SHA-256**. Không dùng DMD3C/BP-Net wrapper cũ rồi đổi tên thành DMD3C++. [DMD3Cpp official](https://github.com/Sharpiless/DMD3Cpp), [official checkpoint](https://huggingface.co/datasets/Liangyingping/DMD3Cpp-checkpoints).

Docker base được pin image digest, không chỉ tag. Cache recipe và resume contract ghi cả numerical library versions/OpenCV build; thay environment không được âm thầm trộn target hoặc resume optimizer.

DA3MONO-LARGE là lựa chọn foundation relative mạnh, chuyên monocular, Apache-2.0, tương thích target V11.3. **Không có benchmark trong package chứng minh nó đứng đầu mọi model hiện tại.** Chọn nó để giữ đúng semantics và cache đã nghiên cứu, không chọn giant multi-view chỉ vì nhiều parameter. [Official model card](https://huggingface.co/depth-anything/DA3MONO-LARGE).

DMD3C++ **không trả một confidence calibrated C_cm**. Package tạo C_cm heuristic từ local agreement với train GT trong cửa sổ 15×15; vùng không có evidence dùng 0.5. Nó không trộn GT vào D_cm, không affine-fit depth theo GT, không gọi C_cm là probability calibrated. KD vẫn loại trừ GT và original sparse support. Đây là **teacher recipe mới**, không được gọi identical với metric-fusion cache subset cũ.

Headless facade chỉ bỏ optional API exporters, giữ official network/weights/feature returns và official RGB normalization của DMD3Cpp. CUDA completion FP32; internal DA3 BF16. Relative cũng BF16 và lưu target float32. Strict checkpoint loading, không fallback sang model khác hoặc giả target bằng GT.

`extension_compat.py` sửa cơ học tensor API deprecated trong **build copy riêng**: 3 `.type()`→`.scalar_type()` và 8 `.data<T>()`→`.data_ptr<T>()`, khóa SHA source gốc. Không sửa kernel math hoặc official Git checkout; patch hash được ghi vào teacher recipe. Cần real GPU smoke để xác nhận extension, không xem sửa API là CUDA qualification.

Generation mặc định 1 sample/call cho teacher nặng, để dễ audit/không OOM và không trộn các ảnh độc lập thành multi-view input. Tăng teacher throughput chỉ sau khi smoke/measurement trên GPU thực tế; không đồng nghĩa student batch=1.

## 5. Storage, download và recovery

- Hai cache NPY gồm tổng bốn float32 maps 352×1216: **~547.86 GiB** cho 85.898 train, cộng NPY headers và SQLite. Student đọc mmap, không unzip TAR hoặc decompress NPZ mỗi batch. Chọn disk capacity để đổi lấy training I/O nhanh, không giả định compression ratio.
- KITTI chỉ extract RGB frames có paired depth, không extract toàn bộ raw LiDAR/stereo assets. Giữ một raw-drive ZIP tại một thời điểm. Curl resume partial download; zip CRC kiểm tra khi đọc; lưu download SHA sau tải. Các URL cũ vẫn có thể bị nhà cung cấp đổi quyền truy cập: lỗi HTTP phải được xử lý, không bypass login/license.
- ZIP do pipeline download được xóa **sau** extraction thành công để tiết kiệm SSD; PNG/cache/checkpoint giữ nguyên. `keep_download_archives=true` nếu muốn giữ archive. Full scan/index có SHA nội dung, không chỉ đếm tên file.
- Cache sharded 256 directories mỗi role. Ghi NPY `.partial` → fsync → atomic replace → SQLite commit. Đứt giữa hai bước không làm record incomplete được tính coverage. Resume kiểm tra recipe/checksum từng record, không generate lại record đã commit tốt.
- Audit luôn kiểm tra đầy đủ trước train; rerun `all` có thể tốn thời gian đọc/checksum SSD nhưng không chạy lại teacher đã hợp lệ. Không tin marker completion để bỏ qua data gate.
- Student checkpoint mỗi **100 updates hoặc 300 giây**, cuối epoch, SIGTERM/SIGINT, hoặc STOP file. Khôi phục model/AdamW/RNG/cursor/totals/selection; deterministic epoch order gồm cả batch cuối ngắn. Augmentation/holdout chạy trên GPU, không phụ thuộc worker-prefetch RNG. Replay chỉ phần sau checkpoint cuối, không hứa zero-loss recovery khi SIGKILL.
- Stop: `docker compose stop -t 600 trainer`. STOP file ở `/data/v11_3_full_work/STOP` giữ pause qua restart; bỏ riêng file STOP khi muốn tiếp tục. Không tự xóa dataset để giải phóng dung lượng.

## 6. Training giữ nguyên model, recipe full-data riêng

Input API: `forward(rgb, sparse, mask, K)`; deploy trả `D_full` [B,1,352,1216] theo mét. RGB 0..1; sparse/GT uint16 PNG /256; K scale theo full-frame resize. Bounds/valid GT: **0.1 < D <120 m**, giống V11.3; không claim giống mọi KITTI submission crop/range protocol khác.

Model **584.845 params**: MobileNetV4 → D4 Bosh3 bounded chart NODE (T=1, max_step=1, dense t=.5) → five-jet query/metric innovation → D2 fine midpoint **2 NFE** → phase detail/PIR → soft sensor fusion. Không thêm online teacher vào inference.

Fresh từ epoch0, chỉ RGB encoder ImageNet pretrained, không load checkpoint/optimizer subset. Default B4, BF16 CNN + FP32 geometry/loss, pinned torchdiffeq0.2.5, frozen encoder BN, AdamW fused LR3e−4/encoder×.5, warmup1→cosine, clipping1. Adaptive Bosh3 không compile/CUDA-graph cả solver. Giữ channels-last, pinned memory, bounded persistent workers/prefetch. Không quảng cáo “GPU nhanh nhất” khi chưa benchmark máy member.

Fresh run dùng LR×1 cho **mọi non-encoder parameter**, không giữ ưu tiên ×2 cho PIR head của recipe fine-tune cũ. Model-config hiệu lực loại bỏ đường dẫn Drive, subset SHA và parent checkpoint metadata; config reference vẫn giữ nguyên để đối chiếu.

Objective dùng **nguyên V11.3** từ `v11_model/losses.py` và các dependencies: GT multiscale, final RMSE/range/iRMSE, tail/boundary, trajectory, sparse/holdout/trust, metric KD và relative KD. Schedules V11.3 được kéo theo budget40; optimizer full-data không phải optimizer fine-tune subset. Xem `v11_model/reference_config.json` và `resolved_config.json` để audit toàn bộ weights.

Best checkpoints riêng: RMSE, iRMSE, joint, policy. Policy ưu tiên iRMSE≤3.2 rồi minimize RMSE; early stop min20/patience8/min_delta0.0002. Target không phải guarantee. Full-data và teacher mới cần được đo trước khi promote baseline.

## 7. Output cho member/team

```text
/data/v11_3_full_work/
  dataset_contract.json, {train,val,test}_index.json
  data_gate.json, pipeline_status.json, hardware.json
  metric_gpu_smoke.json, relative_gpu_smoke.json
/data/teachers_v11_3_full/{metric,relative}/
  recipe.json, records.sqlite3, progress.json, 00..ff/*.npy
/runs/v11_3_full_fresh40/
  checkpoints/{last,best,best_inverse,best_joint,best_policy}.pth
  resolved_config.json, run_manifest.json, smoke_report.json
  train_log.csv, train_log.jsonl, training_status.json
  val_metrics_policy.json, final_summary.json
  kitti_test_predictions.zip
  test_preview/*.png                 # 10 full-size RGB/depth pairs
```

Metric: global pixel RMSE/MAE, iRMSE/iMAE km⁻¹, AbsRel/δ, near/far bins, RGB edge, GT boundary, error tail, stage/solver/fine-NFE diagnostics. Test ZIP là 1.000 uint16 depth PNG; test không public GT nên **không có test RMSE**. Preview dùng cùng scale0–80m TURBO (warm=near); không normalize màu riêng từng ảnh.

Persistent volume là yêu cầu bắt buộc. Package này **chưa gồm automatic remote backup/rclone daemon** hoặc REST serving; checkpoint trên host SSD không chống mất cả disk/node. Có thể dùng snapshot/backup riêng của member. Không tự gọi image S3 DockerHub cũ là image V11.3.

## 8. Source và kiểm chứng

| File | Responsibility |
|---|---|
| pipeline.py | Một lệnh, stage isolation, fail-fast, STOP/signal propagation |
| launch.py / run_full_flow.sh | Host stdlib launcher, GPU arch detection, separate runtime config, detached Compose |
| acquire.py / full_data.py | Official acquisition, pairing/calibration, full split, content contracts |
| teachers.py / teacher_cache.py | Official pinned teachers, atomic resume, no-leak coverage gate |
| adapter.py / trainer.py | V11.3 I/O/loss wrapper, GPU smoke, fresh40, midbatch resume, metrics/test |
| v11_model/*.py | Byte-identical frozen V11.3 architecture/objective |
| server_runtime/ | S3 atomic checkpoint, RNG, SIGTERM, lock helpers reused unchanged |
| Dockerfile / compose.yaml | Portable CUDA execution + persistent storage |
| test_production.py / verification.json | CPU evidence and explicit untested GPU limitations |

```bash
python pipeline.py verify
python -m unittest test_production -v
```

`bundle_manifest.json` protects source bytes; `config.json` is intentionally editable for paths/license. Running recipe/teacher recipes/checkpoint contract are frozen. Do not edit running source/config and relabel resume as reproducible.

Hai teacher GPU probes chạy tách process ngay sau index để phát hiện CUDA/weights/API incompatibility **trước** generation hàng chục nghìn ảnh. Các phase dùng chung GPU lock trong work root; không chạy đồng thời teacher và student của cùng job. CPU tests không tải checkpoint hoặc chứng minh BpOps chạy trên GPU thực.
