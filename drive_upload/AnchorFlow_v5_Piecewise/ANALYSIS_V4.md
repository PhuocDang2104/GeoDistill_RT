# V4 complete-run analysis và phản biện V5 suggestion

Nguồn: `results/metric_kd-20261002T091341Z-1-001.zip`, CSV **15 epoch0–14**, best checkpoint epoch7/cumulative37; final evaluate FP16 có sai khác vài micro-mét so CSV. `v5_suggest.md` là phân tích13epoch, hướng chính vẫn phù hợp sau khi đọc thêm epoch13–14.

## Kết quả thực tế

| Metric | V3 parent | V4 best e7 | Kết luận |
|---|---:|---:|---|
| RMSE | 1,04537m | **1,03008m** | Giảm1,46%, MSE giảm khoảng2,90% |
| MAE | 0,26874m | 0,26410m | Giảm1,73% |
| iRMSE | 3,7651km⁻¹ | 3,6552km⁻¹ | Tốt lên; không near-zero collapse |
| RGB-edge RMSE | 1,48945m | 1,45920m | Tốt hơn2,03%; **không phải GT-depth edge** |
| 0–20m | 0,4990m | 0,4681m | Near tốt rõ |
| 20–40m | 1,4738m | 1,4318m | Mid tốt |
| 40–60m | 2,5687m | 2,6067m | Xấu hơn1,48% |
| 60–80m | 3,9332m | 3,9544m | Xấu hơn0,54% |
| 80–120m | 9,6223m | 9,7571m | Chỉ7.986pixel, dễ nhiễu |
| D4 native pooled GT | 1,90660m | 1,85195m | Coarse có học; khác target full-size |
| Params | 572.017 | 1.857.721 | 3,25× |
| Conv/Linear MAC | 2,974G | 6,344G | 2,13× |
| Same-runtime A100 median | 25,932ms | 31,010ms | **Chậm hơn19,58%** |

Global valid support không đổi25.424.992pixel. Các số không phải official KITTI leaderboard; anonymous test không public GT. Profile so cùng A100/torch/FP16, không dùng T4 log cũ.

Epoch7 train loss khoảng0,802; epoch14 xuống0,741, nhưng val1,03703 > best1,03007. D4 native e14 tiếp tục giảm1,83776 mà Dfull không tốt thêm: **coarse pooled accuracy không đảm bảo high-resolution accuracy**. Không có dấu hiệu NaN/near-zero collapse; train/val gap mở sau best phù hợp plateau/overfit nhẹ, chưa đủ kết luận nguyên nhân duy nhất là loss.

V4 far40–120m chiếm khoảng51,7%SSE dù chỉ5,84%GTpixels. Near gain không giải quyết far tail. 80–120m quá ít support để dùng một epoch tốt ở bin này làm bằng chứng model có thể đồng thời đạt near/far optimum.

## Điểm đồng ý với đề xuất

### Audit pixel-level mới: boundary quan trọng nhưng không phải toàn bộ lỗi

Đã chạy checkpoint V4 best thật trên **400val CPU FP32**, không fit threshold và không train. RMSE1,030084m tái lập run FP16≈1,030076m trên cùng25.424.992pixel.

| Observed GT boundary band | Pixels | RMSE | Tỷ lệ global SSE |
|---|---:|---:|---:|
| 1px | 1.725.092 | 1,9184m | 23,53% |
| 3px | 3.346.179 | 1,7438m | 37,72% |
| 5px | 4.563.265 | 1,6970m | 48,71% |
| 10px | 6.665.785 | 1,6439m | 66,77% |

Lỗi>10m chỉ0,198%pixel nhưng chiếm59,91%SSE; **72,83%SSE của nhóm này nằm ngoài band3px**. Lỗi>20m chiếm32,05%globalSSE. Quarter barrier labels có30,55%valid edge-pair support và5,32%positive trong support; nhãn không phủ toàn ảnh.

Nếu một phương pháp chỉ sửa hoàn hảo prediction trong fixed GT band3 và không đổi ngoài band, RMSE còn khoảng **0,813m**. Đây là counterfactual lower bound cho **giả định đó**, không phải lower bound của V5/model nói chung. V5 xử lý toàn D4, không giới hạn correction vào band.

Vì vậy phản biện quan trọng nhất: hypothesis “edge-only sẽ đưa tới0,7m” không được audit ủng hộ. GT semi-dense có thể bỏ sót biên thật; dù vậy cần báo far-tail và non-boundary errors, không chỉ sharpen boundary. Ray-plane correction + metric reaction toàn grid còn cơ hội sửa local surface/bias ngoài band, nhưng phải chứng minh qua run.

- Generic capacity đã học thật, nhưng accuracy/compute exchange chưa hấp dẫn; giữ V4 làm control, không tăng width tiếp trong V5 đầu tiên.
- Một module quarter-resolution cho local surfaces/boundaries là hypothesis hợp lý, nhỏ và dễ kiểm chứng.
- Không thêm full twin-surface decoder, ODE solver, transformer, nhiều teacher mới hoặc4loss mới trong một run.
- Cần GT-depth boundary metrics, matched-budget control và kiểm cost GPU thật.

## Điểm cần sửa trước triển khai

| Đề xuất / suy luận | Phản biện | Quyết định V5 |
|---|---|---|
| ContextWide giúp nên phần gain do architecture | Thiếu V3 cùng15epoch continuation; gain còn lẫn train thêm | Control selector riêng; không gọi historical parent comparison là architecture-only |
| Objective không tìm operating point near/far | Có thể là data/GT noise, low-res smoothing, capacity hoặc overfit; log chưa phân biệt | Không tăng range weight mù; giữ V3 range objective |
| RGB-edge gain chứng minh boundary bottleneck | RGB texture edge ≠ depth discontinuity | Audit fixed GT bands trên400val |
| Flow của `(u,v,z)` là metric3D | u/v pixel không cùng đơn vị z; dời ray phải xử lý reprojection/visibility | Giữ pixel ray, transport inverse-depth local plane; không free3D motion |
| Barrier từ `e_p+e_q` | Có thể chặn cả along-boundary; normalize neighbor-only triệt tiêu phần common gate | Predict directional pair logits, symmetric shared edges, center mass không normalize đi |
| Dựa vào current depth jump để hard-block | Predicted edge sai có thể tự khóa correction | Không fixed hard depth-jump gate trên prediction |
| Learned `e_p` tự cân pixel loss | Model có thể giảm mask/chỉ chọn vùng dễ; BCE/cross cũng có nghiệm center-only | Fixed GT mask cho depth term, separate balanced barrier BCE |
| Reuse final sensor confidence ở quarter | Final confidence chưa tồn tại tại thời điểm xử lý D4 | Quarter dùng sparse support/spread/innovation; final trust giữ nguyên phía sau |
| Hai Euler steps là ShapeFlow/ODE novelty | Fixed iterations không tự chứng minh continuous3D flow novelty | Gọi bounded ray-constrained correction; không claim first flow/ODE/geometry |
| Tiny weighted edge loss = yếu gradient | Scalar units/magnitude khác nhau, cần gradient analysis để khẳng định | Log term/contribution nhưng không kết luận gradient từ magnitude |

## V5 đã quyết định

**V3 + Ray-Constrained Piecewise Surface Correction tại D4**. Width32,7.976new params/0,202Gnew ConvMAC; aggregate inverse-depth plane hypotheses từ4 fixed cardinal neighbors, directional symmetric barrier, bounded metric reaction, hai update cố định. Không cần DSINE/geometry_fused/DA teacher.

Default thêm fixed-GT band3 RMSE0,05 và balanced barrier BCE0,01. Các supervision khác giữ V3. Không claim đây là architecture-only gain; control V3 có cùng boundary loss, transport-only không BCE, và có thể tắt BCE bằng config cho kiểm kiến trúc riêng.

**Rủi ro cần theo dõi:** quarter grid không đủ giải mọi thin structure; GT semi-dense/multi-frame có edge noise; reaction có thể làm hết gain trong khi transport không được dùng; learned barriers có thể đóng quá nhiều; teacher cache có thể đã chứa GT-informed target nên không coi teacher gain là đánh giá độc lập. Log surface amplitude/delta/conductance, barrier support/positive fraction và D4_base→D4; không kết luận novelty nếu module geometric gần như không được dùng.

## Tiêu chí nghiên cứu

1. Run principal15epoch so initial parent và V4 historical; báo global,40–80m, GT-band1/3/5px cùngruntime/VRAM.
2. Giữ proposal nếu gain không chỉ do thêm training/loss: transport/full/V3-boundary controls cùng budget cần tách hypothesis.
3. Tiến tới<1,02m với overhead thấp là mốc hữu ích; không tuyên bố đạt trước khi train.
4. TừV4≈1,030m xuống0,7m cần giảm khoảng53,8%MSE. Một tiny quarter module không có bằng chứng sẽ tạo bước nhảy đó. Nếu errors chủ yếu ngoài observed boundary, cần chuyển trọng tâm sang far-range/data/generalization, không thêm barrier phức tạp.

Xem `v4_gt_boundary_audit.json` để biết bằng chứng pixel-level400val và `ARCHITECTURE.md` để đối chiếu prior art. Các số CPU FP32 audit không thay số FP16 chính của completed run; khác precision được ghi rõ.
