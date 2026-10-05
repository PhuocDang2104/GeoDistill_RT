# AnchorFlow V11_3 — Pipeline Inspector

Mở **[index.html](index.html)** bằng Chrome/Edge mới. Giữ nguyên folder đi kèm; chạy offline, không cần GPU/CDN. Bản 10 cảnh có full-width playback và Fine NODE continuous progress. Gói deploy: `visualizations/AnchorFlow_V11_3_Vercel.zip`; folder `visualizations/AnchorFlow_V11_3_Vercel/`. ZIP 5 cảnh cũ được giữ nguyên như snapshot lịch sử.

Baseline nội bộ: **best-policy stage epoch 9 / cumulative index 32**, full400 BF16 RMSE **0.986517 m**, iRMSE **3.169735 km⁻¹**. Đây không phải KITTI leaderboard. [Baseline summary offline](BASELINE.md) có sẵn trong ZIP; [technical baseline](../../docs/AnchorFlow_V11_3.md) / [curated evidence](../../results/benchmarks/v11_3_ft15/README.md) nằm trong repo. Metric/provenance: `pipeline_evidence.json`.

## Xem nhanh

1. Chọn một trong **10 RGB**. Mỗi scene có RGB, depth và jet trace riêng, cùng trained checkpoint V11_3.
2. Chọn bước ở sidebar hoặc thanh **D4 → D2 → D1 → Dfull**. Hai ảnh là input/output thật của bước đó, cùng colormap **0–120 m**. Mở mục RGB nếu cần đối chiếu ảnh lớn.
3. **Δdepth**: xanh = gần hơn, đỏ = xa hơn. Saturation hiển thị ở ±0.2…10 m; giá trị đọc không clip. Đây không phải bản đồ accuracy.
4. **Δ|lỗi GT|**: xanh lá = absolute error giảm, tím = tăng, xám = không có GT target. Cho thấy cả correction làm tệ hơn, không tô toàn bộ correction thành “improvement”.
5. Zoom **2×/4×/8×/12×**, bấm lên ảnh để chọn pixel. Trước/sau/map giữ cùng vùng; arrow keys trên canvas di chuyển một cell ở grid sau. Depth, native scene RMSE/MAE và bảng jet cùng cập nhật.
6. **Toàn pipeline** chạy D0 → Dfull trên ảnh toàn chiều rộng, giữ cùng pixel. Đồ thị dưới ảnh cho depth tại 10 snapshots; đường nối chỉ là guide. CNN/readout chuyển ảnh bằng blend hiển thị, không giả intermediate network predictions.
7. **Coarse NODE** có timeline/play τ=0…1. **Fine NODE** có slider/play và cùng ảnh before/after/Δdepth/Δ|lỗi|: continuous extension RK2 bậc 2 từ hai calls thật, không exact ODE. Nút **RK-mid interne** inspect riêng numerical state, khác dense half.
8. **Jet 3D** khôi phục sáu hệ số thật trên toàn ảnh. Slider/play đồng bộ với Coarse/Fine; range height cố định suốt trajectory. Ô vàng bao vùng **1×1/3×3/5×5 native cells** đúng như các patches ở dưới, có số full-image pixels. HTTP/Vercel tự tải full coefficients; file:// bấm **Nạp jets toàn ảnh (offline)** và chọn web_traces/scene_XX.jets.bin.gz, hoặc dùng npm start. Không fabricate slopes/Hessian từ depth.
9. Lý thuyết dùng **Times New Roman + native MathML**. **PNG 4K** xuất scene/stage/zoom/state hiện tại, có provenance/giới hạn.

## 9 bước

| # | Recorded transition | Vai trò |
|---|---|---|
| 1 | D0 → D4 | Coarse bounded-chart NODE, Bosh3 đến T=1 |
| 2 | D4 → D2_base | Learned phase lift 4→2 |
| 3 | D2_base → D2_query | Five-jet inverse-depth consensus |
| 4 | D2_query → D2_pre_fine | Blend/gate + sparse-innovation metric readout |
| 5 | D2_pre_fine → D2 | Fine chart NODE, một midpoint RK2 / đúng 2 NFE |
| 6 | D2 → D1_base | Learned phase lift 2→1 |
| 7 | D1_base → D1_pre_pir | Existing learned detail |
| 8 | D1_pre_pir → D1 | Phase-aware sparse-innovation residual (PIR) |
| 9 | D1 → Dfull | Learned soft sensor reliability fusion |

Không có D1/full-resolution jet field. 3D ở stage D1/Dfull vẫn là **upstream terminal D2 jet reference**, không sáu hệ số mới ở full resolution. Bước4 gồm toàn post-query readout, không chỉ một residual head.

## Contract / giới hạn

| Thành phần | Nguồn / semantics |
|---|---|
| Checkpoint | Actual trained V11_3 best-policy, stage9/cumulative32 |
| Ten scenes | Val indices **0/80/160/240/320/40/120/200/280/360**, không chọn theo metric; không anonymous test |
| Replay | **CPU FP32**, không BF16 full400 evaluation |
| Forward inputs | RGB/S/M/K; không GT hoặc teacher |
| Native image / coarse / fine grid | 352×1216 / 88×304 / 176×608 |
| Coarse observations | 41 official Bosh3 dense samples; **7 hoặc 10 production NFE**; chỉ force endpoint T=1, không ép land0.5 |
| Giữa observations | Linear interpolation decoded jet **chỉ để display**, không solve mới |
| Fine states | Actual initial / internal RK midpoint / terminal chart, **2 field calls**; continuous display extension bậc 2, không solve mới |
| Full-pixel data | Native stage depth Float32; coarse 41 value observations, fine initial/mid/end value chart; lossless byteplane + XOR + gzip |
| Six-channel jets / 3D | Primary pack ROI nhỏ; sidecar .jets.bin.gz chứa coefficients **toàn ảnh**, Float32 lossless, 41 coarse observations +3 fine charts |
| Coarse R/Q/transport telemetry | ROI16×12; 41 observer RHS calls **ngoài production NFE** |
| GT targets | Valid-area pooled GT ở scales4/2/1, chỉ scene audit/map; không feed model |
| Full400 numbers | Original completed BF16 JSON, **25,424,992 GT pixels** |
| Across-resolution delta | Nearest-expand previous depth vào target grid sau **để inspect**, không độc lập component gain |
| Sensor gate | Chỉ sparse support; learned reliability, không GT confidence |

Checkpoint byte SHA256:

```text
c0914220e2c900eebf8fe918fdba21fde1faf2dbb3c73608d2a3efecaf7a5aad
```

Replay **13 depth outputs**, field decomposition và five-jet query đạt exact FP32 parity với unmodified model. [pipeline_evidence.json](pipeline_evidence.json) và `web_traces/*.proof.json` ghi chi tiết. Full400 native numbers không đổi theo slider; current-scene numbers đổi theo state đang hiển thị.

### Fine continuous extension

Với h=1, actual recorded charts z₀, z_mid và z_end cho k₁=2(z_mid−z₀), k₂=z_end−z₀. Thanh progress hiển thị:

```text
z(τ) = z₀ + (τ − τ²)k₁ + τ²k₂,  0 ≤ τ ≤ 1
```

Đây là continuous extension bậc 2 của explicit midpoint: endpoints đúng, không thêm RHS. Tại τ=0.5, state là z₀+0.25k₁+0.25k₂, thường khác internal z_mid=z₀+0.5k₁. Không gán intermediate curve này là exact solution hoặc recorded extra solver states. Decode chart trước khi tô depth; không linear blend metric depth trong NODE.

Same-grid native before/after dùng cùng pooled-GT support. D4→D2 và D2→D1 có **target/support khác**: không trừ native RMSE để kết luận gain upsampling. Map Δ|e| dùng cùng GT target sau và nearest-expanded before, không thay benchmark protocol. Stage effect trong cùng forward không chứng minh matched retraining ablation.

Jet là local quadratic inverse-depth polynomial trong **grid coordinates**, không optical flow/velocity/point cloud. ODE nằm trong chart z; A=R+Q+transport không bằng dj/dτ. Fine field có weights riêng, shared giữa k1/k2. Không global integrability, PINN/physics guarantee hoặc monotone GT improvement. 4K figure không tăng model resolution.

## Hình có sẵn

- [Fine NODE — PNG4K](exports/V11_3_Fine_NODE_4K.png), [Dfull — PNG4K](exports/V11_3_Dfull_4K.png).
- [Desktop](exports/pipeline_desktop.png), [PIR](exports/pipeline_pir.png), [fine jet3D](exports/pipeline_fine_jet.png), [lý thuyết](exports/pipeline_theory.png), [mobile](exports/pipeline_mobile_viewport.png).

ZIP/trace/export cũ của original V11 giữ như snapshot lịch sử, không evidence của viewer này. New ZIP chỉ gồm canonical V11_3 viewer assets, không teacher/data/checkpoint hoặc trace V11 cũ.

## Tái tạo / verification

Từ **repo root**, không phải folder ZIP standalone:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/audit_v11_3_completed.py
.\venv\Scripts\python.exe -X utf8 scripts/build_v11_3_pipeline_traces.py --web
node visualizations/jet_dynamics_lab/test_math.cjs
.\venv\Scripts\python.exe -X utf8 scripts/verify_v11_3_pipeline_viewer.py
.\venv\Scripts\python.exe -X utf8 scripts/package_v11_3_vercel.py
```

Recorder cần local original ZIP, KITTI data/checkpoint đúng SHA; viewer **không cần** chúng. Frozen training source không đổi. Browser verifier cần Playwright + Chrome.

[pipeline_verification.json](pipeline_verification.json): **90 transitions / 10 scenes**, RGB–trace pairing/reload, lossless native metric parity, fine continuous extension/internal state, pipeline playback, coarse interpolation, maps, zoom/pixel, keyboard, 3D orbit, MathML, PNG4K và mobile.

Lazy loading chỉ decode scene hiện tại; mỗi scene gzip khoảng 8 MB, không nạp cả 10 vào RAM. Browser cần DecompressionStream (Chrome/Edge mới). Đổi scene lần đầu có thể mất vài giây. Không gửi riêng HTML vì cần assets/trace đi kèm.

## Deploy Vercel

Folder `../AnchorFlow_V11_3_Vercel/` không chứa model/teacher/dataset. Xem README_DEPLOY.md. Chọn **Other**, **npm run build**, output **dist**. Không deploy folder lab đầy đủ. Bản full jets vượt CLI Hobby100MB; khuyến nghị GitHub import, hoặc CLI Pro nếu đã có. Xem [Vercel limits](https://vercel.com/docs/limits). Local: npm run build rồi npm start, mở localhost4173 để full jets tự tải.

Khung FULL-WIDTH có thanh chia kéo trực tiếp: **RGB/current depth** hoặc **previous stage/current depth**. Click ngoài nút kéo để chọn cell; giữ cùng crop, thời gian và vùng mô phỏng. Các mặt 3D là local quadratic riêng mỗi cell; không ghép giả một global surface qua boundary.
