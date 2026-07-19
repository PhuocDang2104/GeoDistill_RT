# Nguyên lý kiến trúc ReDC

## 1. Bài toán và ký hiệu

Đầu vào của mô hình gồm:

$$
X_r\in\mathbb{R}^{H\times W\times 3}
$$

là ảnh RGB và

$$
X_d\in\mathbb{R}^{H\times W}
$$

là sparse depth map.

Trong quá trình huấn luyện, ground-truth depth được ký hiệu:

$$
Y\in\mathbb{R}^{H\times W}.
$$

ReDC dự đoán dense depth:

$$
D=F(X_d,X_r).
$$

Toàn bộ mô hình được chia thành hai phần:

1. Backbone $F_b$: tạo coarse dense depth $\hat D$.
2. Deformable refinement module $F_r$: tinh chỉnh $\hat D$ thành depth cuối $D$.

Do đó:

$$
\boxed{
D=F_r\left(F_b(X_d,X_r)\right)
}
\tag{1}
$$

hay:

$$
\hat D=F_b(X_d,X_r),
\qquad
D=F_r$\hat D$.
$$

---

# 2. Backbone PENet hai nhánh

ReDC sử dụng backbone PENet gồm hai encoder–decoder branch:

$$
\text{Color-dominant branch}
$$

và

$$
\text{Depth-dominant branch}.
$$

Mục tiêu là xử lý riêng hai modality có bản chất khác nhau:

* RGB cung cấp biên, texture và cấu trúc vật thể.
* Sparse depth cung cấp khoảng cách metric nhưng rất thưa.

Luồng tổng quát:

$$
(X_r,X_d)
\longrightarrow
\begin{cases}
\text{Color-dominant branch}\\
\text{Depth-dominant branch}
\end{cases}
\longrightarrow
\text{confidence fusion}
\longrightarrow
\hat D.
$$

## 2.1. Color-dominant branch

Nhánh color-dominant ưu tiên khai thác thông tin từ RGB và dự đoán:

$$
D_c\in\mathbb{R}^{H\times W}
$$

là depth map thiên về thông tin màu, cùng với:

$$
C_c\in\mathbb{R}^{H\times W}
$$

là confidence logits của prediction $D_c$.

Có thể biểu diễn ở cấp độ hàm:

$$
(D_c,C_c,Z_c)=F_c(X_r,X_d),
$$

trong đó:

$$
Z_c\in\mathbb{R}^{H\times W\times C_c'}
$$

là feature map từ layer cuối của decoder color-dominant.

Nhánh này chủ yếu sử dụng RGB để xác định:

* Biên vật thể.
* Hình dáng vật thể.
* Foreground và background.
* Các vùng có cấu trúc hình ảnh tương đồng.

Tuy nhiên, vì RGB không trực tiếp chứa metric depth, $D_c$ có thể bị ảnh hưởng bởi texture hoặc ánh sáng.

## 2.2. Depth-dominant branch

Nhánh depth-dominant tập trung vào cấu trúc metric từ sparse depth và dự đoán:

$$
D_d\in\mathbb{R}^{H\times W}
$$

cùng confidence logits:

$$
C_d\in\mathbb{R}^{H\times W}.
$$

Biểu diễn tổng quát:

$$
(D_d,C_d,Z_d)=F_d(X_d,X_r,\text{features từ color branch}),
$$

trong đó:

$$
Z_d\in\mathbb{R}^{H\times W\times C_d'}
$$

là feature map từ layer cuối của depth-dominant decoder.

Nhánh này ưu tiên:

* Metric scale.
* Độ liên tục của bề mặt.
* Quan hệ gần–xa.
* Các phép đo LiDAR đáng tin cậy.

Hai nhánh không tạo ra final depth độc lập. Chúng được kết hợp bằng confidence fusion.

---

# 3. Confidence-based fusion

Tại mỗi pixel:

$$
p=(u,v),
$$

color branch dự đoán:

$$
D_c$p$,\qquad C_c$p$,
$$

và depth branch dự đoán:

$$
D_d$p$,\qquad C_d$p$.
$$

Hai confidence logits được chuẩn hóa bằng softmax:

$$
w_c$p$
=

\frac{\exp(C_c$p$)}
{\exp(C_c$p$)+\exp(C_d$p$)}
$$

và:

$$
w_d$p$
=

\frac{\exp(C_d$p$)}
{\exp(C_c$p$)+\exp(C_d$p$)}.
$$

Ta có:

$$
w_c$p$+w_d$p$=1.
$$

Coarse depth của backbone là:

$$
\boxed{
\hat D$p$
=

w_d$p$D_d$p$+w_c$p$D_c$p$
}
$$

hay viết trực tiếp:

$$
\boxed{
\hat D(u,v)
=
\frac{
D_d(u,v)e^{C_d(u,v)}
+
D_c(u,v)e^{C_c(u,v)}
}{
e^{C_d(u,v)}+e^{C_c(u,v)}
}
}
\tag{2}
$$

Ý nghĩa:

* Nếu (C_d$p$\gg C_c$p$), model ưu tiên depth branch:

$$
\hat D$p$\approx D_d$p$.
$$

* Nếu (C_c$p$\gg C_d$p$), model ưu tiên color branch:

$$
\hat D$p$\approx D_c$p$.
$$

Output $\hat D$ không còn là sparse depth mà là một depth map tương đối dense. Đây là điều kiện quan trọng để deformable refinement hoạt động tốt.

---

# 4. Tại sao không dùng deformable convolution trực tiếp trên sparse depth?

Giả sử sparse depth có validity density rất thấp:

$$
\rho$X_d$
=

\frac{
\left|\{p:X_d$p$>0\}\right|
}{
HW
}.
$$

Trên KITTI, density đầu vào chỉ khoảng:

$$
\rho$X_d$\approx 5%.
$$

Nếu lấy mẫu trực tiếp trên $X_d$, phần lớn vị trí được lấy có thể là:

$$
X_d(q)=0,
$$

tức không có measurement hợp lệ.

Do đó phép tổng:

$$
\sum_q W_{p,q}X_d(q+\delta q)
$$

dễ chứa nhiều giá trị rỗng và không cung cấp đủ thông tin để hoàn thiện depth.

ReDC vì thế sử dụng:

$$
X_d
\xrightarrow{F_b}
\hat D
\xrightarrow{F_r}
D,
$$

thay vì:

$$
X_d
\xrightarrow{\text{deformable sampling trực tiếp}}
D.
$$

Kết luận chính của paper là:

$$
\boxed{
\text{Deformable convolution nên được áp dụng trên depth map đủ dense,}
}
$$

không nên áp dụng trực tiếp trên sparse LiDAR cực thưa.

---

# 5. Feature dùng để điều khiển refinement

ReDC lấy feature cuối của hai decoder:

$$
Z_c,\qquad Z_d.
$$

Hai feature được concatenate theo chiều channel:

$$
\boxed{
Z=\operatorname{Concat}(Z_d,Z_c)
}
$$

Nếu:

$$
Z_c\in\mathbb{R}^{H\times W\times C_c'},
\qquad
Z_d\in\mathbb{R}^{H\times W\times C_d'},
$$

thì:

$$
Z\in
\mathbb{R}^{H\times W\times(C_c'+C_d')}.
$$

Feature $Z$ chứa đồng thời:

* Thông tin RGB và biên vật thể.
* Thông tin metric depth.
* Ngữ cảnh đã được backbone suy luận.
* Confidence và cấu trúc của coarse depth.

Từ $Z$, ReDC dự đoán hai đại lượng:

1. Deformable kernel weights.
2. Sampling offsets.

---

# 6. Dự đoán deformable kernel weights

Giả sử kernel có kích thước:

$$
k\times k.
$$

Số sample trên mỗi pixel là:

$$
K=k^2.
$$

Paper sử dụng:

$$
k=3
\quad\Rightarrow\quad
K=9.
$$

Một convolution $1\times1$ được dùng để dự đoán weight logits:

$$
A=\operatorname{Conv}^{W}_{1\times1}$Z$,
$$

với:

$$
A\in\mathbb{R}^{H\times W\times k^2}.
$$

Weights được lấy qua hàm sigmoid:

$$
\boxed{
W=\sigma(A)
}
$$

hay tại mỗi pixel $p$:

$$
W_p=
\left[
W_{p,1},W_{p,2},\ldots,W_{p,k^2}
\right].
$$

Trong đó:

$$
W_{p,i}\in(0,1).
$$

Paper sử dụng sigmoid thay vì softmax:

$$
W_{p,i}=\sigma(A_{p,i})
=

\frac{1}{1+e^{-A_{p,i}}}.
$$

Do đó weights không nhất thiết thỏa:

$$
\sum_i W_{p,i}=1.
$$

Các weights đóng vai trò hệ số residual correction, không đơn thuần là xác suất nội suy.

---

# 7. Dự đoán sampling offsets

Một convolution $1\times1$ khác dự đoán offset:

$$
\Delta
=

\operatorname{Conv}^{\Delta}_{1\times1}$Z$.
$$

Do mỗi sample cần hai tọa độ $(x,y)$, output có:

$$
2k^2
$$

channels:

$$
\Delta
\in
\mathbb{R}^{H\times W\times 2k^2}.
$$

Tại pixel $p$, offset của sample thứ $i$ là:

$$
\delta q_i$p$
=

\left(
\Delta x_i$p$,
\Delta y_i$p$
\right).
$$

Với $k=3$, model dự đoán:

$$
9\text{ offsets},
$$

tương ứng:

$$
18\text{ giá trị}
$$

cho mỗi pixel:

$$
(\Delta x_1,\Delta y_1,\ldots,\Delta x_9,\Delta y_9).
$$

---

# 8. Regular sampling grid

Gọi:

$$
\mathcal N$p$
$$

là lưới $k\times k$ cố định có tâm tại pixel $p$.

Với:

$$
k=3,
$$

ta có các vị trí tương đối:

$$
\mathcal R
=
\left\{
\begin{aligned}
&(-1,-1),(-1,0),(-1,1),\\
&(0,-1),(0,0),(0,1),\\
&(1,-1),(1,0),(1,1)
\end{aligned}
\right\}.
$$

Vị trí regular thứ $i$ là:

$$
q_i=p+r_i,
\qquad r_i\in\mathcal R.
$$

Trong convolution thông thường, model lấy mẫu trực tiếp tại:

$$
q_i.
$$

Trong ReDC, vị trí được dịch chuyển bởi offset học được:

$$
\boxed{
s_i$p$
=

q_i+\delta q_i$p$
}
$$

hay:

$$
s_i$p$
=

p+r_i+\delta q_i$p$.
$$

Viết theo tọa độ:

$$
s_i^x$p$
=

p_x+r_i^x+\Delta x_i$p$,
$$

$$
s_i^y$p$
=

p_y+r_i^y+\Delta y_i$p$.
$$

Nhờ đó receptive field không còn là lưới vuông cố định.

---

# 9. Bilinear sampling tại vị trí phân số

Offset có thể tạo ra tọa độ không nguyên:

$$
s_i$p$=(s_x,s_y),
\qquad
s_x,s_y\in\mathbb{R}.
$$

Do coarse depth $\hat D$ chỉ được xác định tại các pixel nguyên, ReDC sử dụng bilinear interpolation.

Gọi:

$$
\mathcal R(s)
$$

là tập bốn pixel nguyên bao quanh tọa độ $s$:

$$
\mathcal R(s)
=
\left\{
\begin{aligned}
&(\lfloor s_x\rfloor,\lfloor s_y\rfloor),\\
&(\lfloor s_x\rfloor,\lceil s_y\rceil),\\
&(\lceil s_x\rceil,\lfloor s_y\rfloor),\\
&(\lceil s_x\rceil,\lceil s_y\rceil)
\end{aligned}
\right\}.
$$

Giá trị coarse depth tại vị trí phân số $s$ được tính:

$$
\boxed{
\hat D_s
=
\sum_{t\in\mathcal R(s)}
G(s,t)\hat D_t
}
\tag{5}
$$

trong đó bilinear kernel là:

$$
\boxed{
G(s,t)
=
\max(0,1-|s_x-t_x|)
\max(0,1-|s_y-t_y|)
}
\tag{6}
$$

Với bốn pixel xung quanh, ta có:

$$
\sum_{t\in\mathcal R(s)}G(s,t)=1.
$$

Ví dụ, nếu:

$$
s=(10.2,20.7),
$$

thì bốn hàng xóm là:

$$
(10,20),(10,21),(11,20),(11,21).
$$

Các trọng số tương ứng là:

$$
G(s,(10,20))
=
(1-0.2)(1-0.7)
=
0.24,
$$

$$
G(s,(10,21))
=
(1-0.2)(1-0.3)
=
0.56,
$$

$$
G(s,(11,20))
=
(1-0.8)(1-0.7)
=
0.06,
$$

$$
G(s,(11,21))
=
(1-0.8)(1-0.3)
=
0.14.
$$

Do đó:

$$
\hat D_s
=

0.24\hat D_{10,20}
+
0.56\hat D_{10,21}
+
0.06\hat D_{11,20}
+
0.14\hat D_{11,21}.
$$

---

# 10. Deformable residual refinement

Một formulation nội suy thông thường có thể viết:

$$
D_p
=
\sum_{q\in\mathcal N$p$}
W_{p,q}$Z$\hat D_q.
\tag{3}
$$

Tuy nhiên ReDC không dự đoán trực tiếp final depth theo cách này. Paper sử dụng residual learning:

$$
\boxed{
D_p
=
\hat D_p
+
\sum_{q\in\mathcal N$p$}
W_{p,q}$Z$\hat D_{s(q)}
}
\tag{4}
$$

trong đó:

$$
s(q)=q+\delta q.
$$

Viết theo $k^2$ sample:

$$
\boxed{
D$p$
=

\hat D$p$
+
\sum_{i=1}^{k^2}
W_{p,i}
\hat D\left(p+r_i+\delta q_i$p$\right)
}
$$

Vì vị trí sample có thể không nguyên:

$$
\hat D\left(p+r_i+\delta q_i$p$\right)
=

\sum_{t\in\mathcal R(s_i$p$)}
G(s_i$p$,t)\hat D(t).
$$

Thay vào công thức tổng:

$$
\boxed{
D$p$
=

\hat D$p$
+
\sum_{i=1}^{k^2}
W_{p,i}
\sum_{t\in\mathcal R(s_i$p$)}
G(s_i$p$,t)\hat D(t)
}
$$

Đây là công thức đầy đủ của refinement trong ReDC.

Có thể định nghĩa residual:

$$
R$p$
=

\sum_{i=1}^{k^2}
W_{p,i}
\hat D(s_i$p$).
$$

Khi đó:

$$
\boxed{
D$p$=\hat D$p$+R$p$
}
$$

Backbone chịu trách nhiệm tạo prediction chính $\hat D$, còn refinement module chỉ học correction $R$.

---

# 11. Single-pass refinement

Các phương pháp propagation lặp thường có dạng:

$$
D^{(t+1)}
=

\mathcal P(D^{(t)},W),
\qquad
t=0,\ldots,T-1.
$$

Do đó:

$$
D^{(T)}
=

\underbrace{
\mathcal P(
\mathcal P(
\cdots
\mathcal P}_{T\text{ lần}}
(D^{(0)})
\cdots)).
$$

Các iteration có dependency tuần tự:

$$
D^{(t+1)}
\text{ phụ thuộc vào }
D^{(t)}.
$$

ReDC chỉ thực hiện:

$$
\boxed{
D=F_r(\hat D;W,\Delta)
}
$$

một lần duy nhất.

Weights và offsets được dự đoán một lần:

$$
(W,\Delta)=G_\theta$Z$,
$$

sau đó sampling và residual addition cũng được thực hiện một lần:

$$
D
=

\hat D
+
\operatorname{DeformableSample}(\hat D,W,\Delta).
$$

Vì vậy ReDC tránh latency của propagation nhiều vòng.

---

# 12. Hàm loss

Ground-truth KITTI vẫn chứa nhiều pixel không hợp lệ. Validity mask của ground truth được định nghĩa:

$$
M_Y$p$=\mathbf{1}(Y$p$>0),
$$

trong đó:

$$
\mathbf{1}(Y$p$>0)
=

\begin{cases}
1,&Y$p$>0,\\
0,&Y$p$=0.
\end{cases}
$$

Sai số có mask:

$$
E=(D-Y)\odot M_Y.
$$

Paper kết hợp $L_2$ và $L_1$:

$$
\boxed{
\mathcal L(D,Y)
=
\alpha
\left\|
(D-Y)\odot\mathbf{1}(Y>0)
\right\|_2
+
(1-\alpha)
\left\|
(D-Y)\odot\mathbf{1}(Y>0)
\right\|_1
}
\tag{7}
$$

với:

$$
\alpha=0.5.
$$

Viết theo từng pixel:

$$
\mathcal L_1
=

\sum_p
M_Y$p$|D$p$-Y$p$|,
$$

$$
\mathcal L_2
=

\sqrt{
\sum_p
M_Y$p$(D$p$-Y$p$)^2
}.
$$

Do đó:

$$
\mathcal L
=

0.5\mathcal L_2
+
0.5\mathcal L_1.
$$

Loss chỉ được tính tại các pixel có ground truth hợp lệ:

$$
Y$p$>0.
$$

---

# 13. Toàn bộ forward pass

Toàn bộ forward pass của ReDC có thể viết gọn như sau.

## Bước 1: backbone hai nhánh

$$
(D_c,C_c,Z_c)=F_c(X_r,X_d),
$$

$$
(D_d,C_d,Z_d)=F_d(X_d,X_r,\text{color features}).
$$

## Bước 2: confidence normalization

$$
w_c$p$
=

\frac{e^{C_c$p$}}
{e^{C_c$p$}+e^{C_d$p$}},
$$

$$
w_d$p$
=

\frac{e^{C_d$p$}}
{e^{C_c$p$}+e^{C_d$p$}}.
$$

## Bước 3: coarse depth fusion

$$
\hat D$p$
=

w_c$p$D_c$p$+w_d$p$D_d$p$.
$$

## Bước 4: feature fusion

$$
Z=\operatorname{Concat}(Z_c,Z_d).
$$

## Bước 5: predict weights

$$
W
=

\sigma\left(
\operatorname{Conv}^{W}_{1\times1}$Z$
\right).
$$

## Bước 6: predict offsets

$$
\Delta
=

\operatorname{Conv}^{\Delta}_{1\times1}$Z$.
$$

## Bước 7: deformable sampling

$$
s_i$p$
=

p+r_i+\delta q_i$p$.
$$

$$
\hat D(s_i$p$)
=

\sum_{t\in\mathcal R(s_i$p$)}
G(s_i$p$,t)\hat D(t).
$$

## Bước 8: residual refinement

$$
\boxed{
D$p$
=

\hat D$p$
+
\sum_{i=1}^{k^2}
W_{p,i}\hat D(s_i$p$)
}
$$

với:

$$
k=3,\qquad k^2=9.
$$

## Bước 9: masked training loss

$$
\boxed{
\mathcal L
=

\alpha
\left\|
(D-Y)\odot\mathbf{1}(Y>0)
\right\|_2
+
(1-\alpha)
\left\|
(D-Y)\odot\mathbf{1}(Y>0)
\right\|_1
}
$$

với:

$$
\alpha=0.5.
$$

---

# 14. Nguyên lý cốt lõi của ReDC

Toàn bộ kiến trúc có thể tóm tắt:

$$
\boxed{
(X_r,X_d)
\xrightarrow{\text{PENet backbone}}
(D_c,C_c,D_d,C_d,Z_c,Z_d)
}
$$

$$
\boxed{
(D_c,C_c,D_d,C_d)
\xrightarrow{\text{confidence fusion}}
\hat D
}
$$

$$
\boxed{
(Z_c,Z_d)
\xrightarrow{1\times1\text{ conv}}
(W,\Delta)
}
$$

$$
\boxed{
(\hat D,W,\Delta)
\xrightarrow{\text{single-pass deformable sampling}}
D
}
$$

Ý tưởng quan trọng nhất là:

$$
\boxed{
\text{Sparse depth}
\xrightarrow[\text{không nên áp dụng trực tiếp}]
{\text{deformable sampling}}
\text{dense depth}
}
$$

mà phải là:

$$
\boxed{
\text{Sparse depth + RGB}
\xrightarrow{\text{deep backbone}}
\text{dense coarse depth}
\xrightarrow{\text{deformable refinement}}
\text{final depth}
}
$$

Backbone giải quyết bài toán sparse-to-dense chính. Deformable refinement học cách chọn một số vị trí có ích trên coarse dense depth để sửa sai số và phục hồi chi tiết trong một pass duy nhất.