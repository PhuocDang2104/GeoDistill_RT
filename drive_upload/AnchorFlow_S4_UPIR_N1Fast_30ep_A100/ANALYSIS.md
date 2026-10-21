# Quyết định sau N1 và P1–P4

Nguồn: hai log nguyên văn trong `evidence/`. Cùng400validation development,
TensorRT FP32+Graph, runtime synchronized wall gồmD2D. N1 khác phiên P/M3:
không gọi chênh lệch latency N1↔P là controlled speedup.

| Model | Params | RMSE m | iRMSE km⁻¹ | MAE m | Boundary RMSE m | Mean ms | FPS=1000/mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | 582888 | 0.979622 | 3.180601 | 0.257036 | 1.678923 | 4.220933 | 236.91 |
| M3 rebench | 610792 | 0.974078 | 3.134729 | 0.256577 | 1.682626 | 4.439177 | 225.27 |
| P1 pack128 | 578184 | 0.986350 | 3.248654 | 0.258656 | 1.684338 | 3.805111 | 262.80 |
| P2 pack160 | 590888 | 0.996513 | 3.261708 | 0.258287 | 1.700019 | 3.828941 | 261.17 |
| P3 pack128+local96 | 588744 | 0.992368 | 3.261699 | 0.258675 | 1.698108 | 3.936285 | 254.05 |
| P4 pack160+local96 | 601448 | 0.988232 | 3.256349 | 0.258489 | 1.686154 | 3.933644 | 254.22 |

## Insight hẹp, không suy diễn quá bằng chứng

1. Packing thực sự giảm latency trên cùng session M3/P; P1 giảm khoảng14,28%.
   Nhưng tất cảP trượt RMSE<0,98 và iRMSE<3,2. P1 là lựa chọn Pareto nhanh nhất,
   không phải lựa chọn đạt yêu cầu chất lượng.
2. P1→P2 tăng hidden/params nhưng RMSE xấu thêm0,010162m; image-bootstrap95%
   [0,003073;0,016843]. Không tiếp tục tăng packed width để kỳ vọng tự sửa metric.
3. Local96 không có hiệu quả độc lập ổn định: P3−P1 +0,006018m (CI chứa0);
   P4−P2 −0,008281m (CI[-0,015056;-0,000940]). Factorial interaction MSE âm
   −0,028344m², nhưng không chứng minh local luôn hữu ích. P4 vẫn tệ hơn P1 và
   không đạt speedcap3,9. Không mang nhánhlocal sang N1 chỉ vì tương tác này.
4. PixelUnshuffle là hoán vị, nhưng learned bottleneck192→128/160 không invertible;
   cả cấu trúc, initialization và trajectorytraining đều khácN1/M3. Không thể
   kết luận riêng “mất phase gây fail” hoặc “full-resolution fusion luôn cần”.
5. N1 đạt hai quality threshold nhưng sát ngưỡng: RMSE margin chỉ0,000378m,
   iRMSE margin0,019399. Cắt tiếp neural hidden từ512 có rủi ro mấtquality;
   không có own evidence để cam kết một width mới sẽ giữ metric.
6. N1 cần giảm khoảng7,60% wall từ4,221 xuống3,9ms. Trước khi thay mạng nữa,
   thử function-preserving deployment engineering, measuredagainstoriginal.
   Pooling/operator/tactic search có thể không đạt tốc độ; preflight quyết định.

Các CI là image-bootstrap của một seed/dev split được xem nhiều lần, bỏ qua
sequence dependence/multiple comparison; không phải independent confirmation.
Log không chứa trainCSV từngepoch củaP nên không khẳng định overfit/underfit hay
gradient failure. HF/ONNX Slice/logger warnings không làm invalid benchmark đã
qua qualification; không phải bằng chứng nguyên nhân P mất chất lượng.

## Chốt một mạng

N1-Fast giữ đúng N1 neural architecture/weights/params; nâng cấp deployment chứ
không claim contribution kiến trúc mới. Không thêm relative teacher, packed/local,
fineNODE, loss mới hoặc M3 distillation. Không đổi recipe giữa các variants để cứu
headline. Fresh30 chỉ là rerun xác nhận; operator upgrade không đòi retrain.

Đo cặp ORIGINAL/FAST cùngcheckpoint đủ400 là bằng chứng quan trọng nhất: tách
latency và numerical drift khỏi khác biệt fresh training. Bản FAST phải so với
ORIGINAL FP32, không chỉ so với chính nó. Builderlevel3→5 và operator rewrite là
một deployment bundle; chưa cô lập đóng góp từng rewrite. Nếu speedfail, không
tốn30epoch rồi hứa metric tốt sẽ chữa được latency. Nếu đạt subset, confirm seed
và locked officialvalidation trước fullKITTI; không tự promote production.
