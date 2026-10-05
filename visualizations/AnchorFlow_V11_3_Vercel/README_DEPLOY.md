# AnchorFlow V11.3 — Frontend production

## Deploy nhanh từ GitHub

Import repository **PhuocDang2104/GeoDistill_RT** trên Vercel và chọn:

| Setting | Giá trị |
|---|---|
| Root Directory | `visualizations/AnchorFlow_V11_3_Vercel` |
| Framework Preset | Other |
| Node.js | 22.x |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Environment Variables | Không cần |

Bấm **Deploy**. Frontend hoạt động độc lập, không backend/Python/GPU/token. Toàn bộ 10 ảnh và recorded jets đã nằm trong `site/`; trên website chúng tự tải theo ảnh đang chọn. Không cần nạp file thủ công. `vercel.json` đã khai báo build/output. Sau khi kết nối GitHub, commit mới trên branch production được Vercel tự build lại.

Chạy local bằng `npm run dev`; kiểm tra bằng `npm test`. Không cần `npm install` vì không có dependency bên ngoài. Không commit `dist/`, `node_modules/`, dataset/checkpoint hay ZIP.

10 real validation scenes, cùng checkpoint best-policy stage9/cumulative32. Static web, không GPU/Python, model/teacher online hoặc token. Full400 BF16 baseline: **RMSE0.98651661m / iRMSE3.16973465km⁻¹**. Viewer là CPU FP32 replay, không anonymous KITTI test.

## Chạy local — khuyến nghị

Từ folder đã giải nén:

```text
npm run build
npm start
```

Mở **http://127.0.0.1:4173**. Node22+, không dependencies nên không cần npm install. Full jets tự tải đúng scene. Không cần Internet sau build.

Nếu mở dist/index.html bằng file://: Chrome chặn fetch binary local. Bấm **Nạp jets toàn ảnh (offline)**, chọn dist/web_traces/scene_XX.jets.bin.gz tương ứng ảnh. SHA256 từ chối file sai scene. Không tắt browser security. Primary depth/ROI vẫn có sẵn; full coefficients cần thao tác này hoặc local server.

## Deploy Vercel

1. Import repo GitHub hiện tại; không cần tạo repo riêng hay upload ZIP.
2. Framework **Other**, Build **npm run build**, Output **dist**, Node **22.x**. vercel.json đã đặt build/output.
3. Nếu import repo GeoDistill_RT, Root Directory phải là visualizations/AnchorFlow_V11_3_Vercel, không root chứa weights/data/ZIP lịch sử.
4. Chưa deploy lên tài khoản tự động. Build kiểm tra SHA ảnh/binary recorded assets; HTML/CSS/JS có thể chỉnh rồi build lại bình thường, không cần regenerate manifest. Source_manifest.json giữ provenance source gốc.

**Lưu ý dung lượng:** full-image jets thêm ~198MB, tổng source ~288MB. **Không dùng Vercel CLI Hobby** vì source limit100MB; nếu đã có Pro thì dưới limit1GB và có thể dùng npx vercel. Khuyến nghị GitHub import (không sử dụng CLI source upload). Không cần tự nâng gói trả phí để chạy local. [Official limits](https://vercel.com/docs/limits), [build configuration](https://vercel.com/docs/builds/configure-a-build).

## Cách xem

- Thanh chia **trên ảnh**: chọn **RGB ↔ current depth** hoặc **previous stage ↔ current depth**. Kéo nút ↔, dùng slider phụ hoặc ArrowLeft/ArrowRight. Bấm ngoài handle để chọn pixel; zoom giữ cùng crop.
- Jet 3D có slider/play **đồng bộ hai chiều** với timeline ảnh. Range height cố định theo cả trajectory để không tự co giãn mỗi frame. Gain chỉ phóng đại hiển thị, không sửa model.
- Vùng **1×1/3×3/5×5 native cells**: ô vàng đúng diện tích trên ảnh gốc, ô trắng là cell tâm. 3×3 coarse cells =12×12 pixels; 3×3 fine cells =6×6 pixels. Mỗi cell có local quadratic riêng; không ép nối mặt qua boundary. Kéo mặt 3D để xoay.
- Whole pipeline D0→Dfull, before/after, Δdepth/Δ|lỗi GT|, scene native metrics, full400 evidence và PNG3840×2160.

## Chính xác

Full coefficients **Float32 lossless**, không finite-difference reconstruction hay quantization để thay trained slopes/Hessian. Coarse 41 actual dense observations; fine 3 actual chart states initial/internal-RK-mid/terminal. Fine progress dùng order2 RK2 continuous extension từ đúng2calls, không exact ODE và không thêm NFE. Dense half thường khác internalRK-mid. CNN/readout transitions chỉ blend endpoints để trình bày, không intermediate neural predictions. Jet là local inverse-depth polynomial trong grid coordinates, không optical flow hoặc vật thể chuyển động.

Primary pack ~8MB gzip /11MBJS; supplemental full jets ~20MB gzip/scene, chỉ tải scene đang chọn, SHA256 checked. Browser cần DecompressionStream (Chrome/Edge mới). 4K không tăng model resolution. Source/checkpoint/loss/solver train không thay đổi. Byte/SHA: source_manifest.json; evidence: site/pipeline_verification.json và site/vercel_verification.json.
