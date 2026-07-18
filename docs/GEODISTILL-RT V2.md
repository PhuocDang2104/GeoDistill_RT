# **GEODISTILL-RT V2 | Conflict-Aware Geometry Distillation with a Scale-Equivariant, Anchor-Routed Real-Time Student**

---

## **1\. Mục tiêu và định vị đề tài**

GeoDistill-RT hướng đến một mô hình **RGB-guided sparse depth completion nhỏ, chính xác và chạy thời gian thực**. Khi inference, mô hình chỉ sử dụng:

\[  
(I,S,M,K)\\rightarrow(D\_{\\mathrm{full}},C\_{\\mathrm{full}})  
\]

trong đó:

* (I): ảnh RGB.  
* (S): sparse metric depth.  
* (M): validity mask.  
* (K): camera intrinsics.  
* (D\_{\\mathrm{full}}): dense metric depth.  
* (C\_{\\mathrm{full}}): confidence hoặc uncertainty map.

Tất cả teacher đều được sử dụng offline khi train; inference không cần DMD3C, Depth Anything V2, Metric3D v2 hay DSINE. Đây cũng là mục tiêu ban đầu của bản GeoDistill-RT hiện tại.

Bản hiện tại sử dụng sparse propagation, ba stem RGB–depth–ray, MobileViTv2, gated injection, LiteFPN, guided upsampling và tiny full-resolution refinement. Các thành phần này hợp lý về engineering nhưng phần lớn đã có tiền lệ trong depth completion và efficient vision; vì vậy novelty inference hiện tại chưa đủ mạnh nếu chỉ giữ nguyên cách ghép module.

Phiên bản V2 nên có hai contribution cân bằng:

1. **Training-side novelty:** phân tách metric teacher và relative geometry teacher, sau đó xử lý teacher conflict theo pixel.  
2. **Inference-side novelty:** một student có cơ chế metric anchoring và boundary reconstruction riêng, không phụ thuộc iterative SPN hoặc heavy Transformer.

---

# **2\. Cập nhật chính xác vai trò teacher**

## **2.1 DMD3C là dense metric teacher duy nhất**

Trong nhánh metric depth, teacher duy nhất phải là:

# **\[**

# **D\_T^{\\mathrm{metric}}**

\\operatorname{stopgrad}  
\\left(  
D\_{\\mathrm{DMD3C}}  
\\right)  
\]

Ground truth và sparse LiDAR không nên được gọi là teacher:

* (D\_{\\mathrm{gt}}): supervised ground-truth target.  
* (S): real sensor metric anchor.  
* (D\_{\\mathrm{DMD3C}}): dense metric teacher.

Loss metric được viết thành:

# **\[**

# **\\mathcal{L}\_{\\mathrm{metric}}**

\\lambda\_{\\mathrm{gt}}\\mathcal{L}*{\\mathrm{gt}}*  
*\+*  
*\\lambda\_S\\mathcal{L}S*  
*\+*  
*\\lambda{\\mathrm{DMD}}\\mathcal{L}*{\\mathrm{DMD}}  
\]

Không nên gộp GT và DMD3C thành một map rồi gọi chung là teacher, vì cách đó làm mờ ranh giới giữa ground-truth supervision và teacher distillation.

DMD3C là công trình CVPR 2025 sử dụng monocular foundation model, synthetic LiDAR simulation và scale-and-shift-invariant learning để cung cấp dense supervision cho depth completion. Paper dùng BP-Net làm base depth-completion network và báo cáo KITTI RMSE 678.12 mm, đứng đầu official benchmark tại thời điểm submission.

DMD3C được chọn làm teacher vì độ chính xác metric và fine-grained geometry của nó, nhưng không nên gọi tuyệt đối là “current SOTA” trong mọi thời điểm. Cách viết an toàn:

**DMD3C is selected as the state-of-the-art metric depth-completion teacher in this work.**

## **2.2 Geometry teachers**

Geometry branch nên có vai trò hoàn toàn khác metric branch:

# **\[**

# **\\mathcal{G}**

{  
\\mathrm{DA2},  
\\mathrm{Metric3D,v2}  
}  
\]

Vai trò:

| Model | Vai trò đề xuất |
| ----- | ----- |
| Depth Anything V2 | Relative layout, thin structures, boundaries, ordinal depth |
| Metric3D v2 | Optional geometry diagnostic hoặc secondary geometry teacher |
| DSINE | Normal reliability signal, không phải depth teacher |
| DMD3C | Không tham gia geometry fusion mặc định |

DMD3C đã distill từ Depth Anything V2 trong pipeline của chính nó. Do đó DMD3C và DA2 có correlated knowledge; việc xem chúng như hai teacher độc lập có thể làm tăng trọng số cho cùng một loại lỗi. DMD3C chỉ nên giữ vai trò metric teacher, còn DA2 cung cấp relative structure.

## **2.3 Loại bỏ vòng lặp teacher tự tham chiếu**

Thiết kế cũ có nguy cơ tạo vòng lặp:

\[  
D\_{\\mathrm{DMD}}  
\\rightarrow  
R\_G^\\ast  
\\rightarrow  
C\_{\\mathrm{DMD}}  
\\rightarrow  
\\mathcal{L}\_{\\mathrm{DMD}}  
\]

vì DMD3C vừa nằm trong geometry fusion vừa được geometry fusion dùng để tính confidence cho chính nó.

Thiết kế V2:

# **\[**

# **R\_G^\\ast**

w\_{\\mathrm{DA2}}R\_{\\mathrm{DA2}}  
\+  
w\_{\\mathrm{M3D}}R\_{\\mathrm{M3D}}  
\]

và:

# **\[**

# **C\_{\\mathrm{DMD}}**

C\_{\\mathrm{sparse}}  
\\cdot  
C\_{\\mathrm{edge}}  
\\cdot  
C\_{\\mathrm{range}}  
\\cdot  
C\_{\\mathrm{agreement}}  
\]

trong đó (C\_{\\mathrm{agreement}}) chỉ so DMD3C với geometry map không chứa DMD3C.

Nếu vẫn muốn sử dụng DMD3C structure trong geometry branch, phải dùng leave-one-out fusion:

# **\[**

# **C\_{\\mathrm{DMD}}**

\\operatorname{Agreement}  
\\left(  
D\_{\\mathrm{DMD}},  
R\_G^{-\\mathrm{DMD}}  
\\right)  
\]

---

# **3\. Đánh giá novelty sau khi cập nhật**

## **3.1 Novelty tổng thể**

| Phần | Hiện tại | Sau cập nhật V2 |
| ----- | ----- | ----- |
| Teacher-role separation | Khá | Cao |
| Conflict-aware reliability | Khá | Cao |
| Student architecture | Trung bình–thấp | Khá cao |
| Boundary reconstruction | Trung bình | Khá cao |
| Metric-scale stability | Trung bình | Cao |
| Real-time deployment | Có tiềm năng | Có contribution rõ |
| Novelty tổng thể | Khoảng 6/10 | Khoảng 7.5/10 |

Mức đánh giá này giả định các module mới có ablation dương và latency thực sự tốt. Không thể khẳng định module chưa từng xuất hiện dưới mọi hình thức nếu chưa thực hiện systematic literature và patent search.

## **3.2 Novelty chính thức nên được trình bày**

### **Contribution 1 — Separated Teacher Roles**

DMD3C là dense metric teacher duy nhất. Relative monocular models không tạo hard metric labels mà chỉ cung cấp SSI, ordinal và boundary supervision.

### **Contribution 2 — Conflict-Aware Geometry Distillation**

Sparse consistency, edge risk, range risk và normal consistency xác định vùng teacher đáng tin cậy theo pixel.

### **Contribution 3 — Scale-Equivariant Sparse Anchor Routing**

Student có cơ chế đưa real metric anchors vào prediction bằng một bước residual routing đa tỉ lệ, không dùng iterative CSPN, NLSPN, DySPN hoặc optimization solver.

### **Contribution 4 — Scale-Preserving Boundary Residual Pyramid**

Student tách low-frequency metric depth và high-frequency boundary residual. Full-resolution branch chỉ dự đoán bounded high-pass residual, hạn chế làm trôi metric scale.

### **Contribution 5 — Teacher-Free Real-Time Inference**

Tất cả teacher, normal estimation và teacher fusion được thực hiện offline. Inference chỉ còn lightweight student.

---

# **4\. Các hướng kiến trúc gần đây và research gap**

CompletionFormer kết hợp convolution và Transformer để thu được cả local detail và global context, nhưng ngay bản Tiny vẫn có khoảng 41.5M parameters và mô hình đầy đủ tiếp tục dùng SPN refinement. Vì vậy việc sử dụng hybrid encoder không đủ để tạo novelty mới nếu chỉ thay backbone.

OGNI-DC và OMNI-DC cho thấy việc dự đoán depth gradient rồi tích phân có thể tăng robustness và global consistency. OMNI-DC dùng multi-resolution depth integration để giảm error accumulation khi sparse observations rất thưa, nhưng inference vẫn chứa integrator và SPN; paper báo cáo khoảng 235 ms ở (480\\times640) trên RTX 3090\.

PSD sử dụng foundation-model structure để propagate sparse depth trong cả 3D và 2D, sau đó dùng correction module để sửa distortion. Thiết kế này mạnh về OOD robustness nhưng foundation model vẫn nằm trong inference pipeline, không phù hợp với mục tiêu teacher-free real-time student của GeoDistill-RT.

LDCM năm 2026 dùng Poisson-based initialization và thay depth head bằng point-map head để học trực tiếp cấu trúc 3D. Kết quả này cho thấy student mới nên có ít nhất một ablation về 3D point consistency, nhưng full point-map output không nhất thiết phù hợp với mô hình real-time nhỏ.

EfficientPENet là preprint năm 2026 tập trung trực tiếp vào real-time depth completion. Mô hình dùng ConvNeXt, sparsity-invariant convolution và CSPN, báo cáo 36.24M parameters, 20.51 ms và 48.76 FPS trên thiết lập của tác giả. Đây là đối thủ quan trọng về Pareto accuracy–latency, nhưng số tham số vẫn tương đối lớn và refinement vẫn dựa trên CSPN.

Research gap phù hợp cho GeoDistill-RT V2 là:

Có ít phương pháp đồng thời đạt dense teacher distillation, metric-scale stability, direct sparse anchoring, high-resolution boundary recovery và teacher-free inference mà không cần iterative propagation hoặc heavy backbone.

---

# **5\. Student architecture đề xuất**

## **5.1 Tên kiến trúc**

### **GeoRT-SAR Student**

**Scale-Equivariant Anchor-Routed Student for Real-Time Depth Completion**

Hai module inference chính:

1. **SE-MSAR:** Scale-Equivariant Multi-Scale Anchor Routing.  
2. **SP-BRP:** Scale-Preserving Boundary Residual Pyramid.

---

## **5.2 Tổng quan pipeline**

RGB I \+ Sparse Depth S \+ Mask M \+ Intrinsics K  
                         ↓  
          Sparse-Depth Scale Normalization  
                         ↓  
               Ray / UV Geometry Map  
                         ↓  
         Multi-Scale Analytic Anchor Bank  
                         ↓  
       RGB Stem \+ Sparse Stem \+ Ray Stem  
                         ↓  
   Lightweight Local–Global Encoder at 1/4–1/16  
                         ↓  
   Scale-Equivariant Anchor Routing at 1/8, 1/4  
                         ↓  
        Additive Low-Frequency Metric Decoder  
                         ↓  
               Coarse Log-Depth D\_1/4  
                         ↓  
      Scale-Preserving Boundary Residual Pyramid  
                         ↓  
        Full-Resolution Metric Depth D\_full  
                         ↓  
       Adaptive Sparse Anchor Correction \+ Confidence

Không sử dụng:

* Full-resolution Transformer.  
* Iterative ConvGRU.  
* CSPN/NLSPN/DySPN trong main architecture.  
* Differentiable optimization solver.  
* Teacher hoặc normal model khi inference.  
* BEV/TPV branch.

---

# **6\. Module 1 — Sparse-depth scale normalization**

## **6.1 Mục tiêu**

Depth completion model thường dễ học phụ thuộc vào depth range của training dataset. Student cần có tính scale-equivariant:

\[  
F(I,\\beta S,\\beta M\_S)  
\\approx  
\\beta F(I,S,M\_S)  
\]

Với (M\_S) chỉ là mask nên không bị nhân scale.

Lấy median trên các sparse points hợp lệ:

# **\[**

# **m\_S**

\\operatorname{median}  
\\left{  
S(p)\\mid M(p)=1  
\\right}  
\]

Chuẩn hóa:

# **\[**

# **\\bar S(p)**

\\frac{S(p)}{m\_S+\\epsilon}  
\]

Mạng dự đoán normalized log-depth:

# **\[**

# **z(p)**

\\log  
\\left(  
\\bar D(p)+\\epsilon  
\\right)  
\]

Cuối cùng:

# **\[**

# **D\_{\\mathrm{out}}(p)**

m\_S\\exp(z\_{\\mathrm{out}}(p))  
\]

Khi toàn bộ sparse depth được nhân với (\\beta), (\\bar S) không thay đổi và output được nhân lại bởi (\\beta m\_S). Đây là tính chất rất hữu ích cho cross-sensor và cross-dataset generalization. OMNI-DC cũng cho thấy log-depth median normalization có thể bảo toàn scale equivariance khi trộn nhiều dataset.

Phần này không nên được tuyên bố là novelty độc lập; novelty nằm ở cách kết hợp nó với anchor routing và residual pyramid.

---

# **7\. Module 2 — Multi-Scale Analytic Anchor Bank**

## **7.1 Vấn đề của raw sparse concatenation**

Convolution trực tiếp trên sparse map có ba vấn đề:

* Phần lớn pixel bằng zero.  
* Zero có thể bị nhầm với depth hợp lệ nếu mask không được xử lý tốt.  
* Metric information khó lan truyền đủ xa ở các layer đầu.

Thay vì chạy propagation lặp, tạo một bank các proposal từ sparse anchors bằng masked normalized aggregation.

## **7.2 Anchor proposals**

Tại scale (s\\in{8,4}), sử dụng các radius hoặc dilation:

# **\[**

# **\\mathcal{R}**

{1,3,7,15}  
\]

Với mỗi radius (r):

# **\[**

# **A\_{s,r}(p)**

\\frac{  
\\sum\_{q\\in\\mathcal{N}*r(p)}*  
*M\_s(q)\\bar S\_s(q)*  
*}{*  
*\\sum*{q\\in\\mathcal{N}\_r(p)}  
M\_s(q)+\\epsilon  
}  
\]

Validity:

# **\[**

# **V\_{s,r}(p)**

\\mathbb{1}  
\\left\[  
\\sum\_{q\\in\\mathcal{N}\_r(p)}  
M\_s(q)\>0  
\\right\]  
\]

Có thể triển khai bằng:

* Fixed masked average pooling.  
* Sparse normalized convolution.  
* Depthwise convolution với kernel cố định.  
* Integral-image implementation nếu cần tối ưu CPU.

Module này không có parameter và chỉ chạy ở (1/8), (1/4), do đó nhẹ hơn full-resolution KNN hoặc iterative propagation.

---

# **8\. Module 3 — SE-MSAR**

## **Scale-Equivariant Multi-Scale Anchor Routing**

Đây là inference contribution chính thứ nhất.

## **8.1 Ý tưởng**

Tại mỗi pixel, không phải radius propagation nào cũng phù hợp:

* Radius nhỏ tốt gần sparse measurements và boundaries.  
* Radius lớn cần thiết trong vùng không có LiDAR.  
* Uniform averaging dễ lan depth qua hai vật thể khác nhau.  
* Iterative SPN tăng latency và memory.

Student học cách chọn giữa các analytic anchor proposals bằng một router nhỏ.

## **8.2 Routing weights**

Tại scale (s), dùng feature (F\_s), ray map (\\mathbf r\_s), anchor proposals và validity:

# **\[**

# **\\ell\_{s,r}(p)**

h\_{\\mathrm{route},s}  
\\left(  
F\_s,  
\\log(A\_{s,r}+\\epsilon),  
V\_{s,r},  
\\mathbf r\_s,  
d\_M  
\\right)  
\]

trong đó (d\_M) là khoảng cách đến sparse point hợp lệ gần nhất.

Routing probability:

# **\[**

# **\\alpha\_{s,r}(p)**

\\operatorname{softmax}*r*  
*\\left\[*  
*\\ell*{s,r}(p)  
\+  
\\log(V\_{s,r}(p)+\\epsilon)  
\\right\]  
\]

Anchor-routed target:

# **\[**

# **z\_{A,s}(p)**

\\sum\_{r\\in\\mathcal R}  
\\alpha\_{s,r}(p)  
\\log(A\_{s,r}(p)+\\epsilon)  
\]

## **8.3 Bounded residual correction**

Gọi (z\_s) là current normalized log-depth prediction:

# **\[**

# **g\_s(p)**

\\sigma  
\\left(  
h\_{g,s}(F\_s)  
\\right)  
\]

# **\[**

# **z\_s^{+}(p)**

z\_s(p)  
\+  
g\_s(p)  
\\operatorname{clip}  
\\left(  
z\_{A,s}(p)-z\_s(p),  
\-\\tau\_s,  
\\tau\_s  
\\right)  
\]

Ý nghĩa:

* Router không thay depth bằng proposal một cách cứng.  
* Correction luôn dựa trên real sparse-derived metric proposals.  
* Gate quyết định mức độ tin anchor.  
* Clip ngăn một anchor sai hoặc crossing edge phá prediction.  
* Chỉ cần một update tại mỗi scale.

Recommended initial setting:

| Scale | Radius bank | (\\tau\_s) |
| ----- | ----- | ----- |
| (1/8) | 3, 7, 15, 31 | 0.30 |
| (1/4) | 1, 3, 7, 15 | 0.15 |

Các giá trị (\\tau\_s) ở log-depth space và cần tune theo validation.

## **8.4 Điểm khác với SPN**

SE-MSAR không:

* Propagate prediction qua nhiều iteration.  
* Dự đoán affinity với toàn bộ 8-neighborhood trong mỗi iteration.  
* Giữ hidden state.  
* Chạy solver.

Nó route trực tiếp giữa một số analytic sparse-anchor proposals rồi thực hiện một bounded residual update.

Đây là formulation dễ export sang ONNX/TensorRT hơn ConvGRU hoặc custom differentiable integrator.

---

# **9\. Lightweight local–global encoder**

## **9.1 Multi-modal stems**

RGB:

\[  
F\_I^0=\\psi\_I(I)  
\]

Depth:

# **\[**

# **X\_D**

\[  
\\log(\\bar S+\\epsilon),  
M,  
d\_M  
\]  
\]

\[  
F\_D^0=\\psi\_D(X\_D)  
\]

Geometry:

# **\[**

# **X\_G**

\[  
r\_x,r\_y,r\_z,u\_{\\mathrm{norm}},v\_{\\mathrm{norm}}  
\]  
\]

\[  
F\_G^0=\\psi\_G(X\_G)  
\]

Dùng additive projected fusion thay vì concat lớn:

# **\[**

# **F^0**

W\_I F\_I^0  
\+  
g\_D\\odot W\_DF\_D^0  
\+  
g\_G\\odot W\_GF\_G^0  
\]

Trong đó:

# **\[**

# **\[g\_D,g\_G\]**

\\sigma  
\\left(  
h\_{\\mathrm{mod}}  
\[  
F\_I^0,F\_D^0,F\_G^0,M,d\_M  
\]  
\\right)  
\]

Điều này giảm memory traffic so với concat feature rộng ở mọi scale.

## **9.2 Encoder đề xuất**

* (1/2): local convolution, 24–32 channels.  
* (1/4): local convolution hoặc inverted residual, 40–48 channels.  
* (1/8): lightweight global-context block, 64–80 channels.  
* (1/16): lightweight global-context block, 96–128 channels.

Không cần attention ở (1/2) hoặc full resolution.

Hai cấu hình chính:

| Variant | Encoder |
| ----- | ----- |
| GeoRT-SAR-S | MobileViTv2-0.5 hoặc mobile CNN tương đương |
| GeoRT-SAR-M | MobileViTv2-0.75 |

Backbone không nên được coi là contribution. Nó chỉ cung cấp accuracy–latency balance; contribution nằm ở SE-MSAR và SP-BRP.

---

# **10\. Low-frequency metric decoder**

Sử dụng additive FPN:

# **\[**

# **P\_{16}**

\\delta\_{16}(E\_{16})  
\]

# **\[**

# **P\_8**

\\delta\_8(E\_8)  
\+  
\\operatorname{Up}*2(P*{16})  
\]

Sau SE-MSAR ở (1/8):

# **\[**

# **P\_8^{+}**

\\operatorname{Route}\_{8}(P\_8,A\_8)  
\]

# **\[**

# **P\_4**

\\delta\_4(E\_4)  
\+  
\\operatorname{Up}\_2(P\_8^{+})  
\]

Sau SE-MSAR ở (1/4):

# **\[**

# **P\_4^{+}**

\\operatorname{Route}\_{4}(P\_4,A\_4)  
\]

Coarse log-depth:

# **\[**

# **z\_{1/4}**

h\_D(P\_4^{+})  
\]

Coarse confidence:

# **\[**

# **s\_{1/4}**

h\_C(P\_4^{+}),  
\\qquad  
C\_{1/4}=\\exp(-s\_{1/4})  
\]

Phần decoder này chịu trách nhiệm cho:

* Global metric structure.  
* Road và large planar regions.  
* Near/mid/far depth scale.  
* Coarse object layout.

Nó không phải chịu toàn bộ trách nhiệm khôi phục object boundary.

---

# **11\. Module 4 — SP-BRP**

## **Scale-Preserving Boundary Residual Pyramid**

Đây là inference contribution chính thứ hai.

## **11.1 Vấn đề**

Nếu upsample coarse depth bằng bilinear:

* Cars và poles bị phình hoặc mờ.  
* Foreground/background bị trộn.  
* Thin structures biến mất.

Nếu dùng full-resolution depth head tự do:

* Tăng FLOPs mạnh.  
* RGB texture dễ bị copy sang depth.  
* Full-resolution branch có thể làm trôi metric scale đã học ở coarse branch.

## **11.2 Tách metric và boundary prediction**

Metric branch:

# **\[**

# **z\_{\\mathrm{metric}}**

\\operatorname{Up}*4(z*{1/4})  
\]

Boundary branch chỉ dự đoán high-frequency residual ở (1/2) và full resolution.

Tại scale (s\\in{1/2,1}):

# **\[**

# **r\_s^{\\mathrm{raw}}**

a\_s  
\\tanh  
\\left(  
h\_{R,s}  
\[  
F\_{I,s},  
F\_{\\mathrm{dec},s},  
z\_{\\mathrm{up},s},  
C\_{\\mathrm{up},s}  
\]  
\\right)  
\]

Edge gate:

# **\[**

# **E\_s**

\\sigma  
\\left(  
h\_{E,s}  
\[  
F\_{I,s},  
|\\nabla I\_s|,  
|\\nabla z\_{\\mathrm{up},s}|,  
C\_{\\mathrm{up},s}  
\]  
\\right)  
\]

High-pass projection:

# **\[**

# **\\mathcal{H}\_k(x)**

x-\\operatorname{AvgPool}\_k(x)  
\]

Boundary residual:

# **\[**

# **r\_s**

\\mathcal{H}\_k  
\\left(  
E\_s\\odot r\_s^{\\mathrm{raw}}  
\\right)  
\]

Final log-depth:

# **\[**

# **z\_{\\mathrm{full}}**

\\operatorname{Up}*4(z*{1/4})  
\+  
\\operatorname{Up}*2(r*{1/2})  
\+  
r\_1  
\]

# **\[**

# **D\_{\\mathrm{full}}**

m\_S  
\\exp  
\\left(  
z\_{\\mathrm{full}}  
\\right)  
\]

## **11.3 Tại sao high-pass residual quan trọng**

Toán tử:

\[  
\\mathcal{H}\_k(x)=x-\\operatorname{AvgPool}\_k(x)  
\]

loại bỏ phần low-frequency hoặc local DC component của residual. Vì vậy boundary branch chủ yếu sửa:

* Object contour.  
* Thin structure.  
* Foreground/background discontinuity.  
* Local shape.

Nó bị hạn chế khả năng thay đổi global metric scale hoặc toàn bộ mặt phẳng lớn.

Không nên viết rằng cơ chế này đảm bảo tuyệt đối không thay đổi metric scale, vì padding, gate và finite window vẫn có thể tạo sai lệch nhỏ. Cách diễn đạt phù hợp:

The high-pass residual formulation constrains the full-resolution branch to local structural correction and reduces metric-scale drift.

## **11.4 Cấu hình nhẹ**

### **Half-resolution head**

Input projection: 16 channels  
DWConv 3×3: 16 → 16  
Pointwise Conv: 16 → 8  
Residual head: 8 → 1  
Edge gate: 8 → 1

### **Full-resolution head**

RGB shallow feature: 8 channels  
Upsampled metric feature: 8 channels  
DWConv 3×3: 16 → 16  
Pointwise Conv: 16 → 8  
Residual head: 8 → 1  
Edge gate: 8 → 1

Không thực hiện attention hoặc standard wide convolution ở full resolution.

Recommended amplitude:

\[  
a\_{1/2}=0.10,  
\\qquad  
a\_1=0.05  
\]

trong log-depth space.

---

# **12\. Adaptive sparse anchor correction**

Sau SP-BRP:

# **\[**

# **\\lambda\_M(p)**

0.5  
\+  
0.4  
\\sigma  
\\left(  
h\_M  
\[  
C\_{\\mathrm{full}}(p),  
d\_M(p),  
|\\nabla I(p)|  
\]  
\\right)  
\]

# **\[**

# **D\_{\\mathrm{out}}(p)**

D\_{\\mathrm{full}}(p)  
\+  
\\lambda\_M(p)M(p)  
\\left\[  
S(p)-D\_{\\mathrm{full}}(p)  
\\right\]  
\]

Ba lựa chọn ablation:

### **Không correction**

\[  
D\_{\\mathrm{out}}=D\_{\\mathrm{full}}  
\]

### **Fixed soft correction**

\[  
\\lambda\_M=0.7  
\]

### **Adaptive soft correction**

\[  
\\lambda\_M=\\lambda\_M(p)  
\]

### **Hard correction**

# **\[**

# **D\_{\\mathrm{out}}**

MS+(1-M)D\_{\\mathrm{full}}  
\]

Hard correction cho sparse-anchor error bằng zero nhưng có thể tạo discontinuity nếu sensor có noise hoặc calibration error. Adaptive soft correction nên là main configuration.

---

# **13\. Final architecture configuration**

## **GeoRT-SAR-S**

Input:  
  RGB \+ sparse metric depth \+ mask \+ camera intrinsics

Normalization:  
  median sparse-depth normalization  
  normalized log-depth representation

Analytic sparse prior:  
  multi-scale masked anchor bank  
  radius/dilation proposals at 1/8 and 1/4

Encoder:  
  local CNN at 1/2 and 1/4  
  MobileViTv2-0.5 or equivalent at 1/8 and 1/16

Metric anchoring:  
  SE-MSAR at 1/8  
  SE-MSAR at 1/4  
  one bounded update per scale

Decoder:  
  additive LiteFPN  
  coarse normalized log-depth at 1/4

Full-resolution path:  
  SP-BRP at 1/2  
  SP-BRP at full resolution  
  depthwise-separable residual heads

Output:  
  full-resolution metric depth  
  full-resolution confidence  
  internal 1/4 depth and confidence

Not used:  
  teacher at inference  
  normal at inference  
  iterative propagation  
  heavy Transformer decoder  
  full-resolution attention  
  optimization solver

## **Deployment targets**

Các con số sau là **mục tiêu thiết kế**, không phải kết quả đã đo:

| Metric | Target |
| ----- | ----- |
| Parameters | 5–10M |
| Full-resolution feature width | 8–16 channels |
| Routing steps | 2 total |
| Iterative SPN steps | 0 |
| Batch-1 latency | Cần benchmark trên GPU mục tiêu |
| Memory | Thấp hơn CompletionFormer/OGNI class |
| ONNX/TensorRT | Không có custom solver |

Không nên tuyên bố real-time trước khi có latency thực nghiệm.

---

# **14\. Training objective cập nhật**

## **14.1 Metric supervision**

# **\[**

# **\\mathcal{L}\_{\\mathrm{gt}}**

\\frac{1}{|\\Omega\_{\\mathrm{gt}}|}  
\\sum\_{p\\in\\Omega\_{\\mathrm{gt}}}  
\\rho  
\\left(  
D(p)-D\_{\\mathrm{gt}}(p)  
\\right)  
\]

# **\[**

# **\\mathcal{L}\_S**

\\frac{1}{|\\Omega\_M|}  
\\sum\_{p\\in\\Omega\_M}  
|D(p)-S(p)|  
\]

# **\[**

# **\\mathcal{L}\_{\\mathrm{DMD}}**

\\frac{  
\\sum\_p  
C\_{\\mathrm{DMD}}(p)  
\\rho  
\\left(  
D(p)-D\_{\\mathrm{DMD}}(p)  
\\right)  
}{  
\\sum\_p C\_{\\mathrm{DMD}}(p)+\\epsilon  
}  
\]

## **14.2 Geometry supervision**

SSI:

# **\[**

# **\\mathcal{L}\_{\\mathrm{SSI}}**

## **\\frac{1}{|\\Omega\_G|}**

## **\\sum\_{p\\in\\Omega\_G}**

## **C\_G(p)**

## **\\rho**

## **\\left\[**

## **\\log(D(p)+\\epsilon)**

## **\\alpha^\\star R\_G^\\ast(p)**

\\beta^\\star  
\\right\]  
\]

Ordinal:

# **\[**

# **\\mathcal{L}\_{\\mathrm{ord}}**

\\frac{1}{|\\mathcal P|}  
\\sum\_{(p,q)\\in\\mathcal P}  
\\log  
\\left\[  
1+  
\\exp  
\\left(  
\-y\_{pq}  
\[  
\\log D(p)-\\log D(q)  
\]  
\\right)  
\\right\]  
\]

## **14.3 Boundary-specific supervision**

Edge band:

# **\[**

# **\\Omega\_E**

\\operatorname{Dilate}  
\\left(  
|\\nabla R\_G^\\ast|\>\\tau\_G  
;\\lor;  
|\\nabla D\_{\\mathrm{gt}}|\>\\tau\_D  
\\right)  
\]

# **\[**

# **\\mathcal{L}\_{\\mathrm{edge}}**

\\frac{1}{|\\Omega\_E|}  
\\sum\_{p\\in\\Omega\_E}  
\\rho  
\\left(  
D(p)-D\_{\\mathrm{sup}}(p)  
\\right)  
\]

Gradient matching:

# **\[**

# **\\mathcal{L}\_{\\nabla}**

\\sum\_{s\\in{1,1/2}}  
\\left\[  
|\\partial\_x z\_s-\\partial\_x z\_T|  
\+  
|\\partial\_y z\_s-\\partial\_y z\_T|  
\\right\]  
\]

## **14.4 Scale-equivariance consistency**

Sample:

\[  
\\beta  
\\sim  
\\operatorname{LogUniform}(0.5,2.0)  
\]

Run student với (S) và (\\beta S):

\[  
D\_1=F(I,S,M,K)  
\]

\[  
D\_\\beta=F(I,\\beta S,M,K)  
\]

Loss:

# **\[**

# **\\mathcal{L}\_{\\mathrm{eq}}**

## **\\frac{1}{|\\Omega|}**

## **\\sum\_p**

## **\\left|**

## **\\log(D\_\\beta(p)+\\epsilon)**

\\log(\\beta D\_1(p)+\\epsilon)  
\\right|  
\]

Nếu normalization được triển khai đúng, loss này chủ yếu kiểm tra numerical consistency và các đường dẫn vô tình sử dụng raw depth.

## **14.5 Optional 3D point consistency**

Back-project:

# **\[**

# **X(p)**

D(p)  
K^{-1}  
\\begin{bmatrix}  
u\\v\\1  
\\end{bmatrix}  
\]

# **\[**

# **\\mathcal{L}\_{3D}**

## **\\frac{1}{|\\Omega|}**

## **\\sum\_p**

## **\\rho**

## **\\left(**

## **X\_{\\mathrm{student}}(p)**

X\_{\\mathrm{DMD}}(p)  
\\right)  
\]

Đây chỉ là auxiliary training loss; inference vẫn output một depth channel. Ablation này được gợi ý bởi xu hướng point-map regression của LDCM.

## **14.6 Total loss**

\[  
\\boxed{  
\\begin{aligned}  
\\mathcal{L}  
\={}&  
\\lambda\_{\\mathrm{gt}}\\mathcal{L}*{\\mathrm{gt}}*  
*\+*  
*\\lambda\_S\\mathcal{L}S*  
*\+*  
*\\lambda{\\mathrm{DMD}}\\mathcal{L}*{\\mathrm{DMD}}  
\\  
&+  
\\lambda\_{\\mathrm{SSI}}\\mathcal{L}*{\\mathrm{SSI}}*  
*\+*  
*\\lambda*{\\mathrm{ord}}\\mathcal{L}*{\\mathrm{ord}}*  
*\\*  
*&+*  
*\\lambda*{\\mathrm{edge}}\\mathcal{L}*{\\mathrm{edge}}*  
*\+*  
*\\lambda*{\\nabla}\\mathcal{L}*{\\nabla}*  
*\\*  
*&+*  
*\\lambda*{\\mathrm{eq}}\\mathcal{L}*{\\mathrm{eq}}*  
*\+*  
*\\lambda*{3D}\\mathcal{L}\_{3D}  
\+  
\\lambda\_C\\mathcal{L}\_C  
\\end{aligned}  
}  
\]

Recommended starting weights:

| Loss | Weight |
| ----- | ----- |
| (\\lambda\_{\\mathrm{gt}}) | 1.0 |
| (\\lambda\_S) | 1.0 |
| (\\lambda\_{\\mathrm{DMD}}) | 0.3 |
| (\\lambda\_{\\mathrm{SSI}}) | 0.03 |
| (\\lambda\_{\\mathrm{ord}}) | 0.03 |
| (\\lambda\_{\\mathrm{edge}}) | 0.1 |
| (\\lambda\_{\\nabla}) | 0.03 |
| (\\lambda\_{\\mathrm{eq}}) | 0.02 |
| (\\lambda\_{3D}) | 0 hoặc 0.02 |
| (\\lambda\_C) | 0.03 |

---

# **15\. Training schedule cho 1.000–10.000 ảnh**

## **Stage A — Metric core**

Khoảng 20–30% tổng steps:

\[  
\\mathcal{L}\_{\\mathrm{gt}}  
\+  
\\mathcal{L}*S*  
*\+*  
*\\mathcal{L}*{\\mathrm{DMD}}  
\]

Bật:

* Scale normalization.  
* Anchor bank.  
* SE-MSAR.  
* Coarse decoder.

Chưa bật:

* Geometry fusion.  
* Ordinal loss.  
* Confidence weighting phức tạp.

## **Stage B — Boundary and geometry**

Khoảng 50–60% steps:

Thêm:

\[  
\\mathcal{L}*{\\mathrm{SSI}}*  
*\+*  
*\\mathcal{L}*{\\mathrm{ord}}  
\+  
\\mathcal{L}*{\\mathrm{edge}}*  
*\+*  
*\\mathcal{L}*{\\nabla}  
\]

Bật SP-BRP tại (1/2), sau đó full resolution.

## **Stage C — Confidence and robustness**

Khoảng 20% cuối:

Thêm:

* Confidence loss.  
* Scale-equivariance loss.  
* Sparse noise/outlier augmentation.  
* Optional 3D point consistency.  
* Lower learning rate.

## **Practical training settings**

* AdamW.  
* Cosine decay hoặc OneCycle.  
* AMP mixed precision.  
* EMA khoảng 0.999–0.9999.  
* Gradient clipping.  
* Pretrained lightweight encoder nếu chỉ có 1.000–10.000 ảnh.  
* Fixed optimization steps khi so sánh dataset size.  
* Tối thiểu ba random seeds cho các cấu hình cuối.

---

# **16\. Preprocessing và augmentation**

## **16.1 Geometry-safe transforms**

Sau resize:

\[  
f\_x' \= s\_xf\_x,\\quad  
f\_y'=s\_yf\_y,\\quad  
c\_x'=s\_xc\_x,\\quad  
c\_y'=s\_yc\_y  
\]

Sau crop tại offset ((x\_0,y\_0)):

\[  
c\_x'=c\_x-x\_0,\\qquad  
c\_y'=c\_y-y\_0  
\]

Horizontal flip:

\[  
c\_x'=W-1-c\_x  
\]

Ray map phải được tái tạo sau augmentation, không dùng ray map cũ.

## **16.2 Sparse-pattern augmentation**

* Random point dropout: 25%, 50%, 75%.  
* Simulated 8/16/32/64-line LiDAR.  
* Local holes.  
* Foreground-object point removal.  
* Range-dependent noise.  
* 0.5–3% outlier injection.  
* RGB–LiDAR misalignment 1–3 pixels.  
* Random depth-scale factor cho equivariance training.

OMNI-DC cho thấy đa dạng sparse patterns và scale normalization đặc biệt quan trọng với zero-shot robustness.

## **16.3 RGB augmentation**

* Brightness/contrast/gamma.  
* Motion blur nhẹ.  
* Fog hoặc rain simulation nhẹ.  
* Night-style darkening.  
* Color jitter.

Không áp dụng geometry-changing augmentation mà không cập nhật (K), sparse map và teacher outputs tương ứng.

---

# **17\. Ablation chính cho student architecture**

## **17.1 Main incremental ablation**

Đây là bảng quan trọng nhất của paper.

| ID | Architecture | Câu hỏi |
| ----- | ----- | ----- |
| S0 | RGB \+ sparse concat, mobile encoder, FPN, bilinear upsample | Baseline sạch |
| S1 | S0 \+ sparse median normalization \+ log-depth | Scale handling có giúp không? |
| S2 | S1 \+ analytic anchor bank | Sparse prefill có giá trị không? |
| S3 | S2 \+ SE-MSAR tại (1/4) | Learned routing có hơn fixed propagation không? |
| S4 | S3 \+ SE-MSAR tại (1/8) | Multi-scale routing có cần thiết không? |
| S5 | S4 \+ ordinary full-res residual head | Full-res refinement thông thường |
| S6 | S4 \+ SP-BRP tại (1/2) | High-pass residual có giúp boundary không? |
| S7 | S6 \+ SP-BRP full resolution | Full two-level residual pyramid |
| S8 | S7 \+ adaptive sparse correction | Anchor preservation |
| S9 | S8 \+ confidence head | Confidence có cải thiện hoặc calibrate được không? |

Các metric bắt buộc:

* Global RMSE.  
* MAE.  
* Edge RMSE.  
* Near/mid/far RMSE.  
* Parameters.  
* FLOPs.  
* Batch-1 latency.  
* Peak memory.

---

# **18\. Ablation cho SE-MSAR**

| ID | Thử nghiệm |
| ----- | ----- |
| R0 | Raw sparse concatenation, không anchor proposal |
| R1 | Nearest-neighbor fill |
| R2 | Single-radius masked average |
| R3 | Multi-radius uniform average |
| R4 | Learned routing weights |
| R5 | R4 \+ validity logit |
| R6 | R5 \+ distance-to-sparse input |
| R7 | R6 \+ ray map |
| R8 | Direct replacement bằng (z\_A) |
| R9 | Unbounded residual update |
| R10 | Bounded residual update — main |
| R11 | Routing tại (1/4) |
| R12 | Routing tại (1/8) và (1/4) |
| R13 | Một update mỗi scale |
| R14 | Hai updates mỗi scale |

### **Radius bank ablation**

* ({1,3,7})  
* ({1,3,7,15})  
* ({3,7,15,31})  
* ({1,3,7,15,31})

### **Gate ablation**

* (g\_s=1).  
* Learned scalar gate.  
* Learned per-pixel gate.  
* Learned per-pixel gate \+ confidence input.

Kết quả cần chứng minh:

1. Routing tốt hơn fixed analytic propagation.  
2. Một update đủ tốt hơn về Pareto so với hai updates.  
3. Multi-scale routing cải thiện far-range hoặc large-hole regions.  
4. Ray input có gain rõ hoặc bị loại khỏi final model.

---

# **19\. Ablation cho SP-BRP**

| ID | Thử nghiệm |
| ----- | ----- |
| B0 | Bilinear only |
| B1 | Learned convex upsampling |
| B2 | Ordinary full-res residual |
| B3 | Edge-gated ordinary residual |
| B4 | High-pass residual không edge gate |
| B5 | High-pass \+ edge gate — main |
| B6 | Chỉ (1/2) residual |
| B7 | Chỉ full-resolution residual |
| B8 | (1/2) \+ full-resolution pyramid |
| B9 | Standard Conv |
| B10 | DWConv \+ pointwise Conv |
| B11 | Residual width 8 |
| B12 | Residual width 16 |
| B13 | Residual width 24 |

### **Residual amplitude**

\[  
a\_1\\in{0.025,0.05,0.10,0.20}  
\]

### **High-pass kernel**

\[  
k\\in{3,5,7,11}  
\]

### **Chỉ số đánh giá riêng**

* RMSE trong 3-pixel boundary band.  
* RMSE trong 5-pixel boundary band.  
* Depth-gradient error.  
* Ordinal accuracy ở foreground/background pairs.  
* Thin-object recall.  
* Global scale bias:

# **\[**

# **\\operatorname{Bias}\_{\\log}**

\\frac{1}{|\\Omega|}  
\\sum\_p  
\\left\[  
\\log D(p)-\\log D\_{\\mathrm{gt}}(p)  
\\right\]  
\]

Kết quả cần chứng minh:

SP-BRP cải thiện edge metrics nhiều hơn ordinary residual head, trong khi global RMSE và scale bias không bị xấu đi.

---

# **20\. Backbone ablation**

Giữ decoder và routing giống nhau:

| Variant | Mục đích |
| ----- | ----- |
| Pure mobile CNN | Baseline local-only |
| MobileViTv2-0.5 | Default small |
| MobileViTv2-0.75 | Accuracy-oriented |
| ConvNeXt-style mobile blocks | So sánh modern CNN |
| Không global-context block | Kiểm tra global context thực sự cần không |
| Global block chỉ tại (1/16) | Runtime-first |
| Global block tại (1/8,1/16) | Main |

Không nên so backbone có số parameter quá khác nhau. Cần width-match hoặc báo cả Pareto curve.

CompletionFormer cho thấy CNN và Transformer có tính bổ sung, nhưng hybrid backbone chỉ nên được giữ khi gain tương xứng với FLOPs và latency.

---

# **21\. Input và modality ablation**

| ID | Input |
| ----- | ----- |
| I0 | RGB only |
| I1 | Sparse \+ mask only |
| I2 | RGB \+ sparse |
| I3 | RGB \+ sparse \+ mask |
| I4 | I3 \+ normalized UV |
| I5 | I3 \+ 3D ray |
| I6 | I3 \+ UV \+ ray |
| I7 | Raw sparse depth |
| I8 | Median-normalized sparse depth |
| I9 | Linear depth |
| I10 | Log-depth |

Mục tiêu:

* Chứng minh ray map có lợi khi camera intrinsics thay đổi.  
* Chứng minh median normalization cải thiện scale robustness.  
* Kiểm tra UV và ray có trùng thông tin hay không.  
* Loại input không tạo gain rõ để giữ student nhỏ.

---

# **22\. Output head ablation**

| ID | Head |
| ----- | ----- |
| H0 | Direct depth regression |
| H1 | Inverse-depth regression |
| H2 | Log-depth regression |
| H3 | Log-residual trên analytic prior |
| H4 | Depth \+ confidence |
| H5 | Depth \+ auxiliary normal khi train |
| H6 | Depth \+ auxiliary point map khi train |
| H7 | Full 3-channel point-map output |

Khuyến nghị:

* Main model: log-depth hoặc log-residual.  
* Point-map chỉ dùng auxiliary training.  
* Không output normal khi inference.  
* Không dùng 3-channel point map trong main small model trừ khi gain lớn.

---

# **23\. Sparse correction ablation**

| Chế độ | Ưu điểm | Rủi ro |
| ----- | ----- | ----- |
| None | Prediction mượt | Không bảo toàn sensor anchors |
| Fixed soft | Đơn giản | Không thích nghi noise |
| Adaptive soft | Cân bằng accuracy/noise | Thêm head rất nhỏ |
| Hard replacement | Sparse error bằng zero | Có thể tạo artifact |

Ngoài RMSE, phải báo cáo:

# **\[**

# **E\_{\\mathrm{anchor}}**

\\frac{1}{|\\Omega\_M|}  
\\sum\_{p\\in\\Omega\_M}  
|D(p)-S(p)|  
\]

và performance khi thêm sparse outliers.

---

# **24\. Ablation training và distillation**

## **Metric teacher**

| ID | Supervision |
| ----- | ----- |
| T0 | GT \+ sparse |
| T1 | T0 \+ DMD3C uniform |
| T2 | T1 \+ sparse consistency confidence |
| T3 | T2 \+ edge risk |
| T4 | T3 \+ range risk |
| T5 | T4 \+ leave-one-out geometry agreement |

## **Geometry teacher**

| ID | Geometry |
| ----- | ----- |
| G0 | Không geometry teacher |
| G1 | DA2 SSI |
| G2 | DA2 SSI \+ ordinal |
| G3 | G2 \+ DSINE normal reliability |
| G4 | DA2 \+ Metric3D uniform fusion |
| G5 | DA2 \+ Metric3D conflict-aware fusion |
| G6 | G5 \+ DSINE reliability |

## **Loss**

| ID | Thử nghiệm |
| ----- | ----- |
| L0 | L1 |
| L1 | Huber |
| L2 | BerHu |
| L3 | Không edge loss |
| L4 | Edge-band loss |
| L5 | Gradient loss |
| L6 | Scale-equivariance loss |
| L7 | 3D auxiliary loss |
| L8 | Joint training |
| L9 | Curriculum training |

---

# **25\. Ablation về efficiency**

## **So sánh refinement**

| Refinement | Iterations | Cần đo |
| ----- | ----- | ----- |
| None | 0 | Accuracy thấp nhất, speed cao nhất |
| SE-MSAR | 2 scale updates | Main proposal |
| CSPN | 4 | Reference |
| CSPN | 8 | Reference |
| NLSPN-like | Nhiều | Accuracy reference |
| SP-BRP | 2 residual levels | Main boundary path |

Mục tiêu không phải chứng minh SE-MSAR luôn chính xác hơn mọi SPN, mà chứng minh:

\[  
\\text{SE-MSAR \+ SP-BRP}  
\]

tạo Pareto accuracy–latency tốt hơn trong giới hạn real-time student.

## **Benchmark protocol**

* Batch size 1\.  
* Warm-up ít nhất 100 iterations.  
* Đo ít nhất 500 inference iterations.  
* CUDA synchronize trước và sau timing.  
* Báo median và P95 latency.  
* FP32 và FP16.  
* PyTorch eager.  
* Torch compile nếu dùng.  
* ONNX Runtime.  
* TensorRT nếu triển khai.  
* Peak allocated GPU memory.  
* Input resolution chính xác.

Không so latency từ các paper nếu GPU, resolution và framework khác nhau.

---

# **26\. Kế hoạch ablation theo ngân sách dữ liệu**

## **Phase 1 — Screening khoảng 1.000 ảnh**

Chạy khoảng 12–15 cấu hình:

1. S0 baseline.  
2. S1 scale normalization.  
3. S2 anchor bank.  
4. S3 routing (1/4).  
5. S4 routing (1/8+1/4).  
6. B2 ordinary residual.  
7. B5 high-pass edge-gated residual.  
8. B6 half-resolution only.  
9. B8 full pyramid.  
10. Pure CNN vs MobileViTv2-0.5.  
11. DMD3C uniform.  
12. DMD3C confidence.  
13. DA2 SSI.  
14. DA2 SSI \+ ordinal.  
15. DSINE reliability.

Chỉ dùng để phát hiện:

* Implementation bug.  
* Module không tạo gain.  
* Module quá chậm.  
* Loss gây unstable training.

Không dùng kết quả 1.000 ảnh làm kết luận chính.

## **Phase 2 — Khoảng 10.000 ảnh**

Chọn 6–8 configuration tốt nhất:

* Ba seeds.  
* Cùng số optimization steps.  
* Cùng split.  
* Cùng teacher outputs.  
* Báo mean ± standard deviation.  
* Đo latency cho mọi cấu hình.

Main comparison:

1. Baseline.  
2. Baseline \+ DMD3C.  
3. Baseline \+ DMD3C \+ teacher reliability.  
4. Baseline \+ SE-MSAR.  
5. Baseline \+ SP-BRP.  
6. Full GeoDistill-RT V2.

## **Phase 3 — Full training hoặc final validation**

Giữ ba model:

* Lightweight baseline.  
* Student architecture V2, không distillation.  
* Full student V2 \+ conflict-aware distillation.

Bảng này tách được gain từ:

\[  
\\text{Architecture}  
\\quad \\text{vs}\\quad  
\\text{Distillation}  
\]

---

# **27\. Metrics bắt buộc**

## **Accuracy**

* RMSE.  
* MAE.  
* iRMSE.  
* iMAE.  
* REL.  
* (\\delta\_1).  
* Near-range RMSE.  
* Mid-range RMSE.  
* Far-range RMSE.  
* Sparse-anchor error.

## **Boundary**

* Edge RMSE.  
* Depth-gradient MAE.  
* Ordinal accuracy.  
* Boundary-band RMSE 3 px và 5 px.  
* Thin-object region error.

## **Robustness**

* 64/32/16/8 LiDAR lines.  
* Random sparsity.  
* Outliers.  
* Range noise.  
* RGB–LiDAR misalignment.  
* Day/night.  
* Cross-dataset.

PSD và OMNI-DC cho thấy một depth-completion model có thể đạt in-domain accuracy cao nhưng vẫn thất bại khi appearance hoặc sparse-depth pattern thay đổi; robustness cần trở thành một phần của evaluation chứ không chỉ supplementary.

## **Confidence**

* Error–confidence correlation.  
* Risk–coverage curve.  
* AUSE.  
* AURG.  
* NLL.  
* RMSE theo confidence quantile.

Nếu confidence không được evaluate theo calibration, nên gọi output là reliability map.

## **Efficiency**

* Parameters.  
* MACs/FLOPs.  
* Latency median.  
* Latency P95.  
* FPS.  
* Peak memory.  
* ONNX/TensorRT compatibility.

---

# **28\. Đánh giá baseline RMSE khoảng 1.3**

RMSE 1.3 chưa thể so trực tiếp với DMD3C hoặc KITTI leaderboard nếu chưa biết:

* Đơn vị mét hay millimét.  
* Dataset và split.  
* Evaluation mask.  
* Depth cap.  
* Input resolution.  
* Số LiDAR lines hoặc sparse points.  
* Có crop phần sky hay không.  
* Ground-truth density.  
* Model được train trên 1.000 hay 10.000 ảnh.

DMD3C báo cáo 678.12 mm theo KITTI official test server. Nếu baseline 1.3 tương đương 1.3 m, nó có thể đang thấp hơn teacher đáng kể; nhưng chênh lệch protocol có thể chiếm phần lớn khác biệt nên không được so trực tiếp.

Mục tiêu screening hợp lý:

| Mốc | Kỳ vọng |
| ----- | ----- |
| DMD3C distillation | Gain lớn nhất về global RMSE |
| SE-MSAR | Gain ở large holes, far range và sparse robustness |
| SP-BRP | Gain mạnh hơn ở edge metrics so với global RMSE |
| DSINE reliability | Gain ở planar regions và teacher-conflict regions |
| Scale normalization | Gain ở cross-range và cross-dataset tests |

---

# **29\. Rủi ro nghiên cứu**

## **Rủi ro 1 — Anchor bank quá mượt**

Masked averaging có thể lan road depth lên car hoặc background lên foreground.

Giải pháp:

* Learned radius routing.  
* RGB edge gate.  
* Bounded residual.  
* Không direct replacement.

## **Rủi ro 2 — Router trở thành SPN trá hình**

Nếu dùng nhiều update hoặc neighborhood quá lớn, latency sẽ tăng và contribution trở nên gần với propagation network.

Giải pháp:

* Tối đa một update tại (1/8), một update tại (1/4).  
* Radius proposals analytic.  
* Không recurrent hidden state.

## **Rủi ro 3 — High-pass residual tạo ringing**

High-pass operator có thể tạo halo ở boundaries.

Giải pháp:

* Bounded tanh residual.  
* Edge gate.  
* (k=5) hoặc (7), không quá nhỏ.  
* Edge-band supervision.  
* Total variation nhẹ trên residual, không phải full depth.

## **Rủi ro 4 — DA2 không bổ sung DMD3C**

Do DMD3C đã sử dụng DA2, geometry loss có thể chỉ lặp lại cùng knowledge.

Giải pháp:

* DMD3C only vs DMD3C \+ DA2 ablation.  
* Chỉ áp dụng DA2 ở high-frequency disagreement regions.  
* Dùng ordinal pairs thay vì dense SSI toàn ảnh.  
* Giảm (\\lambda\_{\\mathrm{SSI}}).

## **Rủi ro 5 — “Real-time” chỉ đúng trên GPU mạnh**

Giải pháp:

* Chọn trước target hardware.  
* Benchmark FP16 batch 1\.  
* Không tính riêng neural forward mà bỏ qua preprocessing.  
* Đưa scale normalization, anchor bank và post-correction vào latency tổng.

---

# **30\. Kết luận đề xuất**

GeoDistill-RT V2 nên được định vị là:

**A real-time sparse depth-completion framework that combines DMD3C-only metric distillation and conflict-aware relative geometry supervision with a scale-equivariant anchor-routed student. The student performs one-shot multi-scale metric anchoring and scale-preserving boundary refinement without iterative propagation or teachers at inference.**

Điểm novelty mạnh nhất của training:

\[  
\\text{DMD3C metric teacher}  
\\neq  
\\text{relative geometry teachers}  
\]

Điểm novelty mạnh nhất của inference:

\[  
\\text{SE-MSAR}  
\+  
\\text{SP-BRP}  
\]

Cấu trúc contribution hoàn chỉnh:

1. **DMD3C-only dense metric teacher.**  
2. **Conflict-aware relative geometry distillation.**  
3. **Scale-equivariant multi-scale sparse anchor routing.**  
4. **Scale-preserving boundary residual pyramid.**  
5. **Teacher-free, non-iterative real-time inference.**

Ablation quan trọng nhất cần chứng minh:

\[  
\\text{Baseline}  
\\rightarrow  
\+\\text{Scale normalization}  
\\rightarrow  
\+\\text{SE-MSAR}  
\\rightarrow  
\+\\text{SP-BRP}  
\\rightarrow  
\+\\text{DMD3C}  
\\rightarrow  
\+\\text{Geometry reliability}  
\]

Nếu SE-MSAR cải thiện large-hole/far-range accuracy, SP-BRP cải thiện boundary metrics mà không làm xấu scale bias, và full model đạt Pareto accuracy–latency tốt hơn một mobile baseline có SPN, thì phần student architecture sẽ có novelty đủ rõ để đứng ngang với contribution distillation.

