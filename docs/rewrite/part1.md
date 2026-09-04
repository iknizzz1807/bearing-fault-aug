## PHẦN 1 · NỀN TẢNG TOÁN HỌC CHO ML/DL — NỀN TẢNG TOÁN HỌC CHO MACHINE LEARNING / DEEP LEARNING

**Mục đích tài liệu:** Cung cấp nền tảng toán học vững chắc cho ML/DL, định hướng theo bài toán **Predictive Maintenance** (bảo trì dự đoán) dựa trên **chuỗi thời gian rung/chấn động** (vibration time series) từ cảm biến trên máy móc (motor, bearing, máy bơm, tuabin...).

**Đối tượng:** Sinh viên/kỹ sư đã có kiến thức cơ bản, cần ôn lại căn bản một cách có hệ thống, hiểu "vì sao" và "dùng khi nào" từng công cụ toán học.

**Cấu trúc tài liệu:**

1. Đại số tuyến tính
2. Giải tích (vi phân)
3. Thống kê & Xác suất
4. Xác suất & thống kê cho chuỗi thời gian

---

## Sơ đồ: vì sao học sinh toán này lại cần cho A1? (Cây bài toán con)

Trước khi đi vào chi tiết, hãy nhìn toàn cảnh một lần. Toàn bộ phần toán học dưới đây tồn tại vì **một bài toán lớn duy nhất**:

> **MỤC TIÊU LỚN:** Bảo trì dự đoán trên chuỗi rung từ cảm biến — học "cái bình thường", phát hiện lỗi **hiếm** và ước lượng thời gian sống còn lại (RUL).

Bài toán lớn này tách thành **4 bài toán con**. Mỗi bài con cần đúng một chương toán của giáo trình:

```
MỤC TIÊU LỚN: PREDICTIVE MAINTENANCE trên chuỗi rung
 |   (1) gắn cờ mẫu bất thường → phân loại nhị phân (lỗi/normal)
 |   (2) ước lượng RUL / mức hư hại  → hồi quy
 |
 ├─[1] BIỂU DIỄN DỮ LIỆU: biến tín hiệu rung thành con số
 │     có cấu trúc để máy tính xử lý → VECTOR, MA TRẬN, norm
 │     · 1 cửa sổ rung 64 mẫu = 1 vector; cả bộ dữ liệu = 1 ma trận
 │     · trọng số mạng = ma trận; GPU chạy nhanh nhờ nhân ma trận song song
 │     → cần Chương 1: ĐẠI SỐ TUYẾN TÍNH
 │
 ├─[2] HỌC = TỐI ƯU: điều chỉnh trọng số để giảm sai số dần dần
 │     → ĐẠO HÀM, GRADIENT, CHAIN RULE, GRADIENT DESCENT
 │     · đạo hàm cho biết "đi hướng nào để sai số giảm"
 │     · chain rule lan truyền gradient xuyên qua từng lớp (backprop)
 │     → cần Chương 2: GIẢI TÍCH (VI PHÂN)
 │
 ├─[3] ĐO SAI SỐ & SỰ BẤT THƯỜNG bằng ngôn ngữ xác suất
 │     → PHÂN PHỐI, KỲ VỌNG/PHƯƠNG SAI, COVARIANCE, MLE, BAYES
 │     · nhiễu rung ≈ Gaussian → chọn MSE khi dự đoán RUL
 │     · lỗi/nhị phân ≈ Bernoulli → chọn Cross-Entropy khi gắn cờ hỏng
 │     · Bayes cảnh báo về dương tính giả khi sự kiện hỏng hiếm
 │     → cần Chương 3: THỐNG KÊ & XÁC SUẤT
 │
 └─[4] DỮ LIỆU CÓ THỨ TỰ THỜI GIAN — không thể xáo trộn thoải mái
       → CHUỖI THỜI GIAN, STATIONARITY, ACF, ROLLING, LOOKBACK
       · kiểm tra tính dừng; ACF phát hiện chu kỳ quay của máy
       · rolling RMS là chỉ báo hư hại sớm (chuẩn ISO 10816)
       · lookback/sliding window tạo mẫu huấn luyện; chặn data leakage
       → cần Chương 4: CHUỖI THỜI GIAN
```

**Quy tắc đọc:** gặp bất kỳ khái niệm nào ở các chương sau, hãy tự hỏi *"bài toán con nào (1–4) đang cần nó?"* — trả lời được câu đó, bạn đã biết công thức vì sao tồn tại và dùng khi nào.

---

# 1. ĐẠI SỐ TUYẾN TÍNH

**Mục tiêu:** Nắm cách biểu diễn dữ liệu rung thành vector/ma trận, vận hành các phép toán cốt lõi (nhân ma trận, norm, chuẩn hóa), và hiểu eigenvalue/SVD/PCA để nén và giảm chiều tín hiệu — nền tảng cho mọi layer của mạng nơ-ron chạy trên GPU.

## 1.1 Vector và Ma trận

**Mục tiêu:** Hiểu vì sao "một mẫu dữ liệu là một vector" và "cả bộ dữ liệu là một ma trận", cùng quy ước kích thước (n_samples, n_features).

### 1.1.1 Vector là gì?

Một **vector** là một danh sách (mảng) có thứ tự các số. Trong ML, một điểm dữ liệu thường được biểu diễn là một vector.

**Bài toán con đang giải:** biến một phép đo rung tại một thời điểm thành con số có cấu trúc để máy tính tính toán được.

Ví dụ: mỗi giá trị đo rung tại thời điểm `t` có thể có nhiều thành phần:

- `ax = 0.52` — gia tốc rung theo trục X (đơn vị g)
- `ay = 0.61` — gia tốc rung theo trục Y
- `az = 0.44` — gia tốc rung theo trục Z
- `temp = 47.3` — nhiệt độ (°C)
- `rpm = 1498` — tốc độ quay (vòng/phút)

Ta gộp thành vector đặc trưng:

```
x = [0.52, 0.61, 0.44, 47.3, 1498]
```

Số thành phần của vector gọi là **số chiều** (dimension). Ở đây vector `x` có 5 chiều, ký hiệu `x ∈ R^5` (nằm trong không gian thực 5 chiều).

Trong bài toán rung/chấn động, một **cửa sổ** (window) 64 mẫu tín hiệu rung trên 1 trục chính là một vector 64 chiều. Cả triệu cửa sổ như vậy được lưu trong một ma trận lớn.

### 1.1.2 Ma trận là gì?

Một **ma trận** là một bảng số hình chữ nhật gồm `m` hàng và `n` cột. Ký hiệu kích thước là `m × n`.

**Bài toán con đang giải:** nhiều mẫu dữ liệu — làm sao gom lại để xử lý một lần?

Ví dụ: 3 cửa sổ, mỗi cửa sổ 4 đặc trưng (ax, ay, az, temp):

```
X = [ 0.52  0.61  0.44  47.3 ]
    [ 0.55  0.63  0.45  47.8 ]
    [ 0.60  0.70  0.49  48.5 ]
```

Đây là ma trận `3 × 4`: 3 hàng = 3 mẫu quan sát, 4 cột = 4 đặc trưng.

Trong ML, ký ước phổ biến: **mỗi hàng là một mẫu dữ liệu, mỗi cột là một đặc trưng**. Ma trận dữ liệu `X` có kích thước `(n_samples, n_features)`.

### 1.1.3 Vì sao ma trận quan trọng với ML?

- Bộ dữ liệu rung chấn động có thể có hàng chục triệu dòng — ma trận là cách lưu trữ duy nhất khả thi.
- Mọi phép toán trong mạng nơ-ron (linear layer, attention, convolution) đều là phép toán ma trận.
- GPU được thiết kế để làm phép nhân ma trận song song hàng loạt.

---

## 1.2 Các phép toán cơ bản trên ma trận

**Mục tiêu:** Nắm các phép toán nền (cộng/trừ, nhân vô hướng, nhân ma trận = nhiều dot product, transpose, đơn vị/nghịch đảo) — chúng là "khối lắp ráp" của mọi layer trong mạng.

### 1.2.1 Cộng, trừ ma trận

Cộng/trừ hai ma trận **cùng kích thước**, thực hiện theo từng phần tử:

```
[ 1  2 ]   [ 5  6 ]   [ 1+5   2+6 ]   [ 6  8 ]
[ 3  4 ] + [ 7  8 ] = [ 3+7   4+8 ] = [10 12 ]
```

### 1.2.2 Nhân ma trận với số vô hướng (scalar)

Nhân mọi phần tử với số đó:

```
3 * [ 1  2 ]   [ 3  6 ]
    [ 3  4 ] = [ 9 12 ]
```

### 1.2.3 Nhân hai ma trận (matrix multiplication)

Cho ma trận `A` kích thước `(m × k)` và `B` kích thước `(k × n)`. Tích `C = A·B` có kích thước `(m × n)`.

Phần tử hàng `i`, cột `j` của `C` được tính bằng **tổng có trọng số** (dot product) của hàng `i` của `A` với cột `j` của `B`:

```
C[i][j] = A[i][1]*B[1][j] + A[i][2]*B[2][j] + ... + A[i][k]*B[k][j]
```

**Điều kiện tiên quyết:** số cột của `A` phải bằng số hàng của `B` (cùng bằng `k`). Nếu không, phép nhân không tồn tại.

**Ví dụ số:**

```
A = [ 1  2 ]   (2×2)
    [ 3  4 ]

B = [ 5  6 ]
    [ 7  8 ]   (2×2)

C = A·B = [ 1*5+2*7    1*6+2*8 ]   [ 19  22 ]
          [ 3*5+4*7    3*6+4*8 ] = [ 43  50 ]
```

Lưu ý: **phép nhân ma trận không giao hoán**, nghĩa là `A·B ≠ B·A` nói chung. Nhưng nó **kết hợp**: `(A·B)·C = A·(B·C)`.

### 1.2.4 Dot product (tích vô hướng) của hai vector

Cho hai vector cùng chiều `a = [a1, a2, ..., an]` và `b = [b1, b2, ..., bn]`:

```
a · b = a1*b1 + a2*b2 + ... + an*bn
```

Dot product là một **số vô hướng** (scalar).

**Ví dụ số:**

```
a = [1, 2, 3]
b = [4, 5, 6]
a · b = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

Dot product có ý nghĩa hình học: đo mức **tương đồng về hướng** giữa hai vector.

- Hai vector cùng hướng: dot product lớn (dương).
- Hai vector vuông góc: dot product bằng 0.
- Hai vector ngược hướng: dot product âm.

Công thức liên hệ với góc `θ` giữa hai vector:

```
a · b = |a| * |b| * cos(θ)
```

Trong ML, dot product được dùng ở khắp nơi: **correlation** (tương quan), **attention mechanism** trong transformer, và là khối cơ bản của mọi layer tuyến tính.

**Ví dụ ứng dụng:** muốn biết cửa sổ rung hiện tại giống với "mẫu lỗi" đã biết như thế nào, ta tính dot product giữa đặc trưng chuẩn hóa của chúng. Giá trị càng cao → càng giống.

### 1.2.5 Phép nhân ma trận = tập hợp nhiều dot product

Nhân ma trận chỉ là cách viết gọn của việc tính nhiều dot product cùng lúc:

- Mỗi phần tử `C[i][j]` là dot product của hàng `i` (của `A`) với cột `j` (của `B`).
- Với ma trận dữ liệu `X (n_samples × n_features)` và ma trận trọng số `W (n_features × n_outputs)`, tích `X·W` tính **đồng thời** đầu ra cho tất cả mẫu.

Đây chính là lý do GPU "nổ máy" cho DL: hàng triệu dot product độc lập chạy song song.

### 1.2.6 Transpose (chuyển vị)

**Chuyển vị** của ma trận `A`, ký hiệu `A^T`, là ma trận đổi hàng thành cột:

```
A = [ 1  2  3 ]    (1 × 3)
A^T = [ 1 ]        (3 × 1)
      [ 2 ]
      [ 3 ]
```

Với ma trận 2 chiều:

```
A = [ 1  2 ]
    [ 3  4 ]
    [ 5  6 ]   (3×2)

A^T = [ 1  3  5 ]   (2×3)
      [ 2  4  6 ]
```

Quy tắc: phần tử `A[i][j]` sau chuyển vị thành `A^T[j][i]`.

**Vì sao dùng transpose?**

- Đổi cách biểu diễn hàng/cột cho khớp kích thước khi nhân ma trận.
- Tính covariance và trong SVD.
- Ký hiệu gọn cho dot product: `a · b = a^T * b`.

### 1.2.7 Ma trận đơn vị và ma trận nghịch đảo

**Ma trận đơn vị (identity)** ký hiệu `I`: có số 1 trên đường chéo chính, 0 ở chỗ khác. Nhân bất kỳ ma trận nào với `I` đều ra chính nó: `A·I = A`.

```
I = [ 1  0  0 ]
    [ 0  1  0 ]
    [ 0  0  1 ]
```

**Ma trận nghịch đảo** của `A`, ký hiệu `A^(-1)`, thỏa mãn:

```
A · A^(-1) = I
```

Nghịch đảo chỉ tồn tại cho ma trận **vuông** và **không suy biến** (determinant khác 0). Trong ML ta gần như không tính nghịch đảo trực tiếp (tốn kém, bất ổn định số) mà dùng các phương pháp ổn định hơn (chuẩn hóa, pseudo-inverse, gradient descent).

---

## 1.3 Norm và chuẩn hóa

**Mục tiêu:** Hiểu khái niệm "độ dài của vector" (norm), phân biệt L1/L2/Frobenius và vì sao bắt buộc chuẩn hóa khi các đặc trưng rung có thang đo rất khác nhau (0.5 g vs 1500 RPM).

### 1.3.1 Norm L2 (chuẩn Euclid) — hay dùng nhất

Norm là độ dài của một vector. Norm L2:

```
|x|_2 = sqrt(x1^2 + x2^2 + ... + xn^2)
```

**Ví dụ số:**

```
x = [3, 4]
|x|_2 = sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
```

Chính là định lý Pythagoras mở rộng nhiều chiều. Khoảng cách thông thường giữa hai điểm `a` và `b` là `|a - b|_2`.

### 1.3.2 Norm L1 (chuẩn Manhattan)

```
|x|_1 = |x1| + |x2| + ... + |xn|
```

**Ví dụ số:**

```
x = [3, -4]
|x|_1 = |3| + |-4| = 3 + 4 = 7
```

So với L2 = 5, cùng vector nhưng L1 lớn hơn vì nó cộng giá trị tuyệt đối, không bình phương.

**Vì sao phân biệt L1 và L2?** L1 và L2 phạt trọng số theo hai triết lý khác nhau, quyết định hành vi của mô hình khác hẳn:

| Tiêu chí | Norm L1 (Manhattan) | Norm L2 (Euclid) |
|---|---|---|
| Công thức | `Σ |x_i|` | `sqrt(Σ x_i²)` |
| Cách nhìn thành phần | coi mọi thành phần như nhau (tuyến tính) | bình phương → phạt nặng thành phần lớn |
| Nhạy với outlier | ít bị một giá trị bất thường chi phối | bị một giá trị lớn kéo lệch mạnh |
| Dùng trong regularization | `loss + λ·Σ|w_i|` → kéo trọng số về **chính xác 0** (sparse, tự chọn đặc trưng) | `loss + λ·Σ w_i²` (weight decay) → kéo về gần 0, không bằng 0 |
| Phù hợp khi | nghi ngờ nhiều đặc trưng vô dụng, muốn loại bỏ | hầu hết đặc trưng đều có ích, chỉ cần ép nhỏ |

Nói gọn: **L1 "tuyển chọn", L2 "cân chỉnh"**. Với feature rung (hàng chục đặc trưng thống kê/tần số), L1 giúp biết đặc trưng nào thật sự dự báo lỗi; L2 giúp mọi đặc trưng mượt hơn khi có nhiễu.

### 1.3.3 Norm Frobenius — dành cho ma trận

Frobenius norm của ma trận `A` = căn bậc hai của tổng bình phương **tất cả phần tử**:

```
|A|_F = sqrt(sum_i sum_j A[i][j]^2)
```

**Ví dụ số:**

```
A = [ 1  2 ]
    [ 3  4 ]
|A|_F = sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.48
```

Frobenius norm giống "độ lớn" của cả ma trận; được dùng để đo độ lớn của ma trận trọng số khi regularization.

### 1.3.4 Chuẩn hóa vector (normalization)

**Chuẩn hóa** = chia vector cho norm của nó để có **norm = 1** (vector đơn vị):

```
x_norm = x / |x|_2
```

**Ví dụ số:**

```
x = [3, 4]
x_norm = [3/5, 4/5] = [0.6, 0.8]
|x_norm|_2 = sqrt(0.36 + 0.64) = sqrt(1) = 1 ✓
```

**Vì sao dùng?**

- Các đặc trưng rung có thang đo khác nhau (gia tốc ~0.5 g, nhiệt độ ~47 °C, RPM ~1500). Nếu không chuẩn hóa, RPM sẽ "áp đảo" gia tốc trong tính toán khoảng cách/dot product.
- Khi chuẩn hóa, dot product giữa hai vector chuẩn hóa **chính là cosin của góc giữa chúng** (cosine similarity):

  ```
  cos_sim(a, b) = (a · b) / (|a|_2 * |b|_2)
  ```

  Được dùng để so sánh mức na ná nhau của hai cửa sổ tín hiệu / hai vec-tơ nhúng.

> **Lưu ý phân biệt:** Chuẩn hóa vector (chia theo norm) ≠ **chuẩn hóa đặc trưng** (feature scaling) như `z-score = (x - mean) / std`. Cả hai đều quan trọng ở phần 3.3 và 3.4 — và với chuỗi rung thì nơi này còn phát sinh bẫy data leakage (chỉ dùng mean/std của quá khứ, xem mục 4.4).

---

## 1.4 Eigenvalue, Eigenvector và SVD

**Mục tiêu:** Hiểu eigenvalue/eigenvector là "hướng biến thiên chính" của dữ liệu, SVD nén ma trận và PCA giảm chiều — để biến cửa sổ rung 64 chiều thành vài thành phần chính mà vẫn giữ phần lớn thông tin.

### 1.4.1 Eigenvector và eigenvalue — ý tưởng

Cho ma trận vuông `A`. Một vector khác 0 `v` được gọi là **eigenvector** của `A` nếu nhân `A` với `v` chỉ làm **giãn/kéo dài** `v` (đổi độ dài, đổi dấu) chứ không đổi hướng:

```
A · v = λ · v
```

`λ` (lambda) là **eigenvalue** tương ứng — hệ số giãn.

**Ví dụ số:**

```
A = [ 2  1 ]        v = [ 1 ]
    [ 1  2 ],           [ 1 ]

A·v = [ 2*1 + 1*1 ] = [ 3 ] = 3 * [ 1 ]
      [ 1*1 + 2*1 ] = [ 3 ] = 3 * [ 1 ]

→ λ = 3, v = [1, 1]
```

Có eigenvector khác:

```
v = [ 1 ]
    [-1 ]

A·v = [ 2*1 + 1*(-1) ] = [ 1 ] = 1 * [ 1 ]
      [ 1*1 + 2*(-1) ] = [-1 ] = 1 * [-1 ]

→ λ = 1, v = [1, -1]
```

### 1.4.2 Ý nghĩa trực quan

- Eigenvector = **hướng** mà phép biến đổi `A` "giữ nguyên" (chỉ co/giãn).
- Eigenvalue lớn = dọc theo hướng đó, `A` **khuếch đại mạnh**.
- Mỗi ma trận vuông `n×n` đối xứng có `n` eigenvector trực giao (vuông góc nhau).

Trong phân tích rung chấn: eigenvalue đo mức năng lượng dao động theo từng hướng. Hướng có eigenvalue lớn = hướng biến thiên chính của tín hiệu.

### 1.4.3 SVD — Singular Value Decomposition

Mọi ma trận `X (m × n)` (không cần vuông) đều phân tích được thành:

```
X = U · Σ · V^T
```

Trong đó:

- `U` (m × m): ma trận các vector cột trực giao — biểu diễn "chiều của các mẫu" (trái singular vector).
- `V^T` (n × n): ma trận các vector cột trực giao — biểu diễn "chiều của các đặc trưng" (phải singular vector).
- `Σ`: ma trận chéo chứa **singular values** `σ1 ≥ σ2 ≥ ... ≥ 0` — to bằng mức độ quan trọng của mỗi "thành phần".

**Vai trò trực quan của SVD — nén (compression):**

Các singular value giảm nhanh. Nếu chỉ giữ `k` giá trị lớn nhất, ta được xấp xỉ:

```
X ≈ U_k · Σ_k · V_k^T
```

với sai số nhỏ. Đây chính là **nén ma trận**: lưu ít số hơn nhưng giữ hầu hết thông tin.

Ví dụ: ma trận chứa 1000 cửa sổ × 64 đặc trưng (64.000 số). Sau SVD, giữ `k = 5` thành phần chính → chỉ cần `1000*5 + 5 + 5*64 ≈ 5325` số (~8% dung lượng) mà vẫn giữ được phần lớn biến thiên của dữ liệu.

### 1.4.4 SVD → PCA (Principal Component Analysis)

PCA dùng SVD để **giảm chiều** dữ liệu, giữ lại phương sai lớn nhất (nơi dữ liệu phân tán nhất):

1. Chuẩn hóa dữ liệu (mean = 0).
2. Tính SVD (hoặc eigenvalue decomposition của covariance matrix).
3. Giữ `k` singular vector ứng với `k` singular value lớn nhất.
4. Chiếu dữ liệu xuống `k` chiều mới (principal components).

**Vì sao dùng trong Predictive Maintenance?**

- Tín hiệu rung 64 chiều có thể nén về 3–10 chiều để: giảm nhiễu, giảm thời gian huấn luyện, trực quan hóa cụm lỗi/normal theo thời gian.
- Các "thành phần chính" thường tương ứng với chế độ dao động chính của máy (tần số quay chính, bội số 2f, harmonic).

**So sánh hướng tiếp cận giảm chiều — cái nào cho A1?**

| Cách giảm chiều | Nguyên lý | Đặc điểm | Khi nào hợp cho bài rung |
|---|---|---|---|
| **SVD/PCA** (tuyến tính) | giữ hướng phương sai lớn nhất | nhanh, diễn giải được (tần số quay, harmonic), là baseline vững | khử nhiễu, trực quan hóa, nén feature trước khi đưa vào model cổ điển |
| **Autoencoder** (phi tuyến) | học biểu diễn nén thông qua mạng nơ-ron | mạnh hơn nhưng khó diễn giải, cần nhiều dữ liệu | tự học "hình ảnh bình thường" khi hết dữ liệu lỗi (anomaly detection) |
| **Không giảm chiều** | giữ nguyên 64 chiều cho mạng sâu | mạng nơ-ron tự học biểu diễn phi tuyến | thường được ưu tiên khi dùng CNN/LSTM hiện đại |

Nói gọn: **dùng PCA khi muốn diễn giải + nhanh; nhường công việc "nén" cho chính mạng nơ-ron khi dự thi kiểu DL**. Nhưng SVD vẫn là công cụ nền tảng: nén mô hình, low-rank approximation, và trong attention (biểu diễn low-rank).

---

## 1.5 Vì sao Deep Learning cần matrix math?

**Mục tiêu:** Chỉ ra rằng toàn bộ tính toán của DL quy về **nhân ma trận trên batch** — nền tảng vì sao GPU nhanh, và cách đọc kích thước của mọi layer.

### 1.5.1 Batch — xử lý hàng loạt

Huấn luyện không xử lý từng mẫu một mà theo **batch** (`mini-batch`): ví dụ `32, 64, 128` mẫu mỗi lần.

Giả sử mỗi cửa sổ rung là vector 64 chiều. Một batch 32 cửa sổ được gộp vào ma trận:

```
X_batch  (32 × 64)
```

Mọi phép toán (tính đầu ra, tính gradient) được tính cho **cả batch cùng lúc** bằng một phép nhân ma trận.

### 1.5.2 Layer tuyến tính = nhân ma trận

Một fully-connected layer với trọng số `W` và bias `b` là:

```
Z = X · W + b
```

- `X`: đầu vào (batch × in_features)
- `W`: trọng số (in_features × out_features)
- `b`: bias (out_features)
- `Z`: đầu ra trước activation

**Ví dụ:** `X (32×64)` nhân `W (64×16)` → `Z (32×16)`: ánh xạ 32 cửa sổ rung thành 16 "đặc trưng trung gian". Quy tắc khớp kích thước: số cột của vế đầu (64) = số hàng của ma trận sau (64).

### 1.5.3 GPU và sự song song

- GPU có hàng nghìn lõi tính đơn giản (khác CPU vài chục lõi mạnh).
- Phép nhân ma trận phân rã thành vô số phép nhân–cộng **độc lập**, chạy song song tuyệt vời trên GPU.
- Do đó hiệu năng DL = hiệu năng nhân ma trận. Tăng batch → tận dụng băng thông bộ nhớ GPU tốt hơn → huấn luyện nhanh hơn (đến một giới hạn).

### 1.5.4 Tóm tắt Đại số tuyến tính

| Khái niệm | Công thức | Dùng khi nào |
|---|---|---|
| Dot product | `a·b = sum(a_i*b_i)` | Đo tương đồng, khối cơ bản của layer |
| Nhân ma trận | `C[i][j] = sum_k A[i][k]*B[k][j]` | Batch forward/backward trên GPU |
| Transpose | `A^T[j][i] = A[i][j]` | Khớp kích thước, tính covariance |
| Norm L1 | `sum(|x_i|)` | Regularization thưa, ít nhạy outlier |
| Norm L2 | `sqrt(sum(x_i^2))` | Độ dài, khoảng cách, weight decay |
| Norm Frobenius | `sqrt(sum(A[i][j]^2))` | Độ lớn ma trận trọng số |
| Chuẩn hóa | `x / |x|_2` | Cosine similarity |
| Eigen | `A·v = λ·v` | Hiểu hướng biến thiên chính |
| SVD | `X = U·Σ·V^T` | Nén, PCA, giảm chiều |

---

# 2. GIẢI TÍCH (VI PHÂN)

**Mục tiêu:** Hiểu đạo hàm/gradient dùng để "đi về hướng sai số giảm", chain rule là lõi của lan truyền ngược (backprop), gradient descent là vòng lặp học của mọi mô hình, còn Taylor và tính lồi giải thích vì sao bước nhỏ mới hội tụ.

## 2.1 Đạo hàm

**Mục tiêu:** Đạo hàm là "tốc độ thay đổi" (độ dốc) của hàm tại một điểm; nắm bảng đạo hàm cơ bản và quy tắc tuyến tính.

### 2.1.1 Định nghĩa

Đạo hàm của hàm `f(x)` tại điểm `x` đo **tốc độ thay đổi** của `f` khi `x` thay đổi một chút:

```
f'(x) = lim (h→0)  ( f(x + h) - f(x) ) / h
```

Trực giác: đạo hàm là **độ dốc** (slope) của tiếp tuyến tại `x`.

**Ví dụ số cơ bản:**

`f(x) = x^2`, tại `x = 3`:

```
f'(x) = 2x
f'(3) = 2*3 = 6
```

Nghĩa là quanh `x = 3`, nếu tăng `x` thêm 0.01 thì `f` tăng thêm khoảng `6 * 0.01 = 0.06`.

### 2.1.2 Bảng đạo hàm hay dùng

```
f(x) = c            → f'(x) = 0          (hằng số)
f(x) = x^n          → f'(x) = n*x^(n-1)  (lũy thừa)
f(x) = e^x          → f'(x) = e^x
f(x) = ln(x)        → f'(x) = 1/x
f(x) = sin(x)       → f'(x) = cos(x)
f(x) = ax + b       → f'(x) = a          (tuyến tính)
```

**Quy tắc tính nhanh (linearity of derivative):**

```
d/dx [ c1*f1(x) + c2*f2(x) ] = c1*f1'(x) + c2*f2'(x)
```

### 2.1.3 Vì sao DL cần đạo hàm?

- Huấn luyện mạng = **tối ưu hóa**: tìm trọng số `w` sao cho hàm mất mát `L(w)` nhỏ nhất.
- Đạo hàm cho biết hướng tăng của `L`. Muốn giảm, ta đi **ngược dấu đạo hàm**.
- Từ một trọng số duy nhất đến hàng triệu trọng số, ta cần đạo hàm riêng và gradient (mục 2.2).

---

## 2.2 Đạo hàm riêng và Gradient

**Mục tiêu:** Mở rộng đạo hàm sang hàm nhiều biến — đạo hàm riêng và gradient, vector chỉ hướng tăng nhanh nhất. Ngược dấu gradient = hướng giảm loss.

### 2.2.1 Đạo hàm riêng

Khi hàm có **nhiều biến** `f(x1, x2, ..., xn)`, **đạo hàm riêng** theo `xi` là đạo hàm thông thường khi **cố định tất cả biến khác**:

```
∂f/∂xi = lim (h→0) ( f(..., xi + h, ...) - f(..., xi, ...) ) / h
```

Ký hiệu `∂` (đọc là "đó") thay cho `d` khi có nhiều biến.

**Ví dụ số:** `f(x, y) = x^2 + x*y + y^3`, tại điểm `(x, y) = (2, 1)`:

```
∂f/∂x = 2x + y            → tại (2,1): 2*2 + 1 = 5
∂f/∂y = x + 3y^2          → tại (2,1): 2 + 3*1 = 5
```

### 2.2.2 Gradient — vector của mọi đạo hàm riêng

**Gradient** của hàm nhiều biến là vector chứa toàn bộ đạo hàm riêng:

```
∇f = [ ∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn ]
```

Gradient **chỉ hướng tăng nhanh nhất** của hàm tại điểm đó. Ngược dấu gradient = hướng giảm nhanh nhất = hướng ta cần đi để giảm loss.

**Ví dụ số:**

```
f(x, y) = (x - 1)^2 + (y - 2)^2
∇f = [ 2(x-1), 2(y-2) ]

Tại (3, 2): ∇f = [ 2*(2), 2*0 ] = [ 4, 0 ]
```

Hàm đạt giá trị nhỏ nhất tại `(1, 2)` (cả hai thành phần gradient = 0). Điểm đạo hàm = 0 gọi là **điểm tới hạn** (critical point) — đây là ứng viên của cực tiểu/cực đại.

### 2.2.3 Gradient trong ML

Với mạng nơ-ron, hàm mất mát phụ thuộc vào hàng triệu trọng số:

```
L(w1, w2, ..., wN)
```

Gradient `∇L` là vector N chiều. **Backpropagation** (lan truyền ngược) chính là thuật toán tính gradient này hiệu quả, dựa trên **chain rule** (mục 2.3).

---

## 2.3 Chain rule — nền tảng của Backprop (TRỌNG TÂM)

**Mục tiêu:** Hiểu chain rule ghép các đạo hàm cục bộ qua từng lớp để tạo ra gradient toàn cục — đây chính là thuật toán backprop, thứ biến "cập nhật trọng số" trở nên khả thi với hàng triệu tham số.

### 2.3.1 Chain rule cho hàm một biến

Nếu `z = f(g(x))` thì:

```
dz/dx = df/dg * dg/dx
```

Nói dễ hiểu: tốc độ thay đổi của `z` theo `x` = tích của (tốc độ thay đổi của `z` theo `g`) × (tốc độ thay đổi của `g` theo `x`).

**Ví dụ số từng bước:**

```
f(g) = g^2,  g(x) = 3x + 1, tại x = 2
g(2) = 7
df/dg = 2g = 2*7 = 14
dg/dx = 3
dz/dx = 14 * 3 = 42
```

**Kiểm chứng trực tiếp:** `z = (3x + 1)^2 = 9x^2 + 6x + 1` → `dz/dx = 18x + 6`. Tại `x = 2`: `18*2 + 6 = 42`. ✓ Hai cách cho cùng kết quả.

### 2.3.2 Chain rule nhiều biến — trọng tâm of backprop

Mạng nơ-ron là **chuỗi hàm lồng nhau**. Ví dụ mạng 1 lớp ẩn chuẩn:

```
x → z1 = W1·x + b1        (linear 1)
  → a1 = φ(z1)            (activation, ví dụ ReLU/tanh)
  → z2 = W2·a1 + b2       (linear 2)
  → ŷ = z2                (kết quả dự đoán)
  → L = (ŷ - y)^2         (MSE loss)
```

Loss phụ thuộc vào `w` qua nhiều "tầng trung gian". Muốn `∂L/∂W1`, ta lần ngược:

```
∂L/∂W1 = ∂L/∂ŷ  *  ∂ŷ/∂a1  *  ∂a1/∂z1  *  ∂z1/∂W1
```

Đây chính là **tích của các đạo hàm riêng qua từng khâu** — chain rule nhiều biến:

```
∂z/∂x = Σ_k ( ∂z/∂y_k * ∂y_k/∂x )
```

(Trong đó `y_k` là tất cả đường trung gian nối `x` đến `z`.)

### 2.3.3 Vì sao backprop = "gọn" hơn tính trực tiếp?

- Mạng 10 layer, 1000 nơ-ron mỗi layer → hàng triệu trọng số.
- Nếu tính `∂L/∂w` từng cái riêng lẻ bằng định nghĩa → tốn phí cực lớn (rất nhiều lần chạy forward).
- Backprop: một lần **forward** để tính các giá trị trung gian, một lần **backward** để truyền gradient ngược và tái sử dụng kết quả của lớp sau cho lớp trước.
- Gradient `∂L/∂w_i` bằng nhau ở các lớp cuối được tính một lần, dùng lại cho nhiều lớp — đây là sức mạnh của chain rule.

### 2.3.4 Ví dụ số hoàn chỉnh (1 nơ-ron)

Giả lập một tế bào nơ-ron: `z = w*x + b`, `a = tanh(z)` (đạo hàm `tanh' = 1 - a^2`), loss `L = (a - y)^2`.

Dữ liệu: `x = 2`, `w = 0.5`, `b = 0.1`, `y = 1`.

**Bước forward:**

```
z = 0.5*2 + 0.1 = 1.1
a = tanh(1.1) ≈ 0.8005
L = (0.8005 - 1)^2 = ( -0.1995 )^2 ≈ 0.0398
```

**Bước backward (chain rule từ cuối về đầu):**

```
∂L/∂a = 2*(a - y) = 2*(-0.1995) ≈ -0.3990
∂a/∂z = 1 - a^2 = 1 - 0.6408 ≈ 0.3592
∂z/∂w = x = 2
∂z/∂b = 1

∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w = (-0.3990) * 0.3592 * 2 ≈ -0.2866
∂L/∂b = ∂L/∂a * ∂a/∂z * ∂z/∂b = (-0.3990) * 0.3592 * 1 ≈ -0.1433
```

Cập nhật trọng số với learning rate `η = 0.1`:

```
w_new = w - η * ∂L/∂w = 0.5 - 0.1*(-0.2866) ≈ 0.5287
b_new = b - η * ∂L/∂b = 0.1 - 0.1*(-0.1433) ≈ 0.1143
```

Gradient âm → loss giảm khi tăng w → ta tăng w. Đúng chiều.

### 2.3.5 Tổng kết trực quan

```
Forward:  x → z1 → a1 → z2 → a2 → ... → ŷ → L   (tính giá trị)
Backward: ∂L/∂W1 ← ∂L/∂W2 ← ... ← ∂L/∂Wn        (tính gradient ngược)
```

Mỗi khâu của mạng chỉ cần biết hai thứ: giá trị đạo hàm cục bộ của nó và gradient truyền về từ phía sau. Chain rule ghép chúng lại = gradient toàn cục.

---

## 2.4 Gradient Descent — trực giác

**Mục tiêu:** Nắm vòng lặp "tính gradient → cập nhật trọng số" và vai trò quyết định của learning rate; phân biệt các biến thể optimizer và trực giác loss landscape.

### 2.4.1 Thuật toán

Hàm mất mát `L(w)` — mục tiêu: tìm `w` làm `L` nhỏ nhất. Gradient descent dựa trên việc **đi ngược hướng gradient**:

```
Lặp lại:
    1. Tính gradient  g = ∇L(w)
    2. Cập nhật       w ← w - η * g
```

Với `η` (eta) là **learning rate** — bước đi mỗi lần.

### 2.4.2 Loss landscape — "địa hình" của hàm mất mát

Hình dung `L(w)` như một **bề mặt địa hình**: trục ngang là trọng số, trục dọc là giá trị loss.

- Gradient descent giống **người đi bộ mù** trong sương mù: chỉ cảm nhận được độ dốc dưới chân, cứ đi xuống dốc.
- **Global minimum** — đáy trũng thấp nhất toàn vùng: điểm mong muốn.
- **Local minimum** — đáy trũng cục bộ, thấp hơn xung quanh nhưng không phải thấp nhất toàn cầu. Gradient = 0 tại đó → thuật toán "mắc kẹt".
- **Saddle point** — điểm yên ngựa: gradient = 0 nhưng không phải cực trị (một hướng xuống, một hướng lên).

### 2.4.3 Ảnh hưởng của Learning Rate

Với hàm một biến `L(w) = (w - 3)^2` (cực tiểu tại `w = 3`), bắt đầu `w = 8`, `∇L = 2(w - 3)`:

- **η = 0.1:** bước `w ← 8 - 0.1*10 = 7`, rồi 6.2, 5.6, 5.1... hội tụ chậm nhưng chắc chắn.
- **η = 0.5:** bước `w ← 8 - 0.5*10 = 3` — tới đích chỉ trong 1 bước (may mắn).
- **η = 1.0:** `w ← 8 - 10 = -2`, rồi `-2 - 1.0*2*(-5) = 8` — **dao động qua lại**, không hội tụ.
- **η = 1.2:** `w ← 8 - 12 = -4`, rồi `-4 - 1.2*2*(-7) = 12.8` — **nổ tung** (diverge), loss mỗi lúc một lớn.

**Quy luật:**

- `η` quá nhỏ → hội tụ chậm (tốn thời gian, có thể kẹt local minimum).
- `η` quá lớn → dao động / bùng nổ, không hội tụ.
- Trong thực tế dùng **learning rate schedule** (giảm dần) hoặc optimizers thích ứng (Adam, RMSProp) tự điều chỉnh bước đi theo từng trọng số.

### 2.4.4 Variants — SGD, Mini-batch, Adam

| Biến thể | Tính gradient trên | Ưu điểm | Hạn chế |
|---|---|---|---|
| **Batch GD** | toàn bộ dữ liệu mỗi bước | chính xác, ổn định | chậm, tốn bộ nhớ |
| **SGD / Mini-batch GD** | batch nhỏ (32/64/128 mẫu) | nhanh, thêm "nhiễu" ngẫu nhiên dễ thoát local minimum hẹp | nhiễu hơn, cần tinh chỉnh lr |
| **Adam** | batch nhỏ + momentum + adaptive lr | tự điều chỉnh bước theo từng tham số, hội tụ nhanh, ít tinh chỉnh | nhiều siêu tham số hơn, một số báo cáo khái quát kém hơn SGD |

Trong thực tế thi A1: khởi điểm thường dùng **Adam** với `learning_rate = 0.001`; nếu muốn kết quả tinh, cuối quá trình chuyển sang SGD momentum để hội tụ sâu hơn.

### 2.4.5 Loss landscape trong Predictive Maintenance

- Với signal có nhiễu, loss landscape có thể nhiều local minimum nông.
- Khi dùng MSE với dữ liệu noisy, landscape khá trơn. Khi dùng Cross-Entropy + softmax, landscape phức tạp hơn nhưng thực tế huấn luyện dễ hơn (mục 3.10 giải thích lý do).

---

## 2.5 Taylor và tính lồi (convexity) — khái niệm ngắn

**Mục tiêu:** Hiểu khai triển Taylor (xấp xỉ hàm bằng parabol cục bộ) lý giải vì sao gradient descent chỉ an toàn với bước nhỏ, và khái niệm lồi cho biết giới hạn của lý thuyết đối với mạng sâu.

### 2.5.1 Khai triển Taylor

Taylor cho phép **xấp xỉ hàm phức tạp tại gần điểm `x0`** bằng đa thức:

```
f(x) ≈ f(x0) + f'(x0)(x - x0)            (xấp xỉ bậc 1 — đường thẳng)
      + ½ f''(x0)(x - x0)^2              (thêm bậc 2 — parabol)
```

**Vì sao hữu ích cho DL?**

- Giải thích vì sao gradient descent hội tụ với `η` nhỏ: quanh một điểm, ta xấp xỉ loss bằng parabol, mỗi bước gradient descent thực chất là "rơi xuống đáy của parabol xấp xỉ".
- Nói rõ vì sao `η` lớn làm hỏng: xấp xỉ bậc 2 chỉ đúng trong lân cận nhỏ; bước quá xa làm xấp xỉ sai hẳn.
- Trong object detection/pose còn dùng Taylor để refine dự đoán.

### 2.5.2 Hàm lồi (convex)

Hàm `f` gọi là **lồi** nếu đoạn thẳng nối hai điểm bất kỳ trên đồ thị luôn **nằm trên đồ thị**:

```
f( t*a + (1-t)*b ) ≤ t*f(a) + (1-t)*f(b)   với mọi 0 ≤ t ≤ 1
```

Hoặc (với hàm trơn 1 biến): `f''(x) ≥ 0` với mọi x (độ cong không âm).

- Hàm lồi: **mọi local minimum đều là global minimum**. Gradient descent từ điểm nào cũng về đích.
- Ví dụ: `x^2`, `-ln(x)`, hàm MSE theo trọng số. Logistic loss cũng lồi theo tham số.

**Vì sao liên quan?**

- Các mô hình tuyến tính + loss lồi (hồi quy ridge/lasso) → dù dùng gradient descent hay thuật toán nào cũng tìm được nghiệm tối ưu toàn cục.
- Mạng sâu = hàm **không lồi** → không có bảo đảm cực tiểu toàn cục. Trong thực tế gradient descent vẫn hoạt động tốt (local minima "đủ tốt", nhiều tham số giúp né saddle point).
- Convexity giúp hiểu: không nên kỳ vọng lý thuyết chặt chẽ cho DL, mà dùng trực giác + thực nghiệm.

---

# 3. THỐNG KÊ & XÁC SUẤT

**Mục tiêu:** Dùng xác suất để (a) mô hình hóa nhiễu rung và chọn đúng hàm loss (MSE hay Cross-Entropy), (b) đo "bất thường" bằng kỳ vọng/phương sai/tương quan, (c) hiểu bẫy dương tính giả qua Bayes và đánh giá đúng khi lớp lỗi cực hiếm.

## 3.1 Khái niệm cơ bản

**Mục tiêu:** Nắm biến ngẫu nhiên, phân phối rời rạc/liên tục và trực giác của hàm mật độ xác suất.

- **Biến ngẫu nhiên (random variable):** đại lượng nhận giá trị phụ thuộc kết quả ngẫu nhiên. `X` = "giá trị amplitude rung tại thời điểm ngẫu nhiên" là một biến ngẫu nhiên.
- **Phân phối xác suất:** mô tả xác suất mỗi giá trị xuất hiện.
  - Biến **rời rạc**: hàm khối xác suất `P(X = x)`, ví dụ "số lần cảnh báo trong 1 giờ".
  - Biến **liên tục**: hàm mật độ `p(x)`, ví dụ "giá trị rung (real)".

Với hàm mật độ `p(x)` liên tục:

```
P(a ≤ X ≤ b) = ∫ p(x) dx  (từ a đến b)
```

đồng thời `∫ p(x) dx = 1` trên toàn miền (tổng xác suất = 1).

## 3.2 Các phân phối phổ biến

**Mục tiêu:** Nhận diện 3 phân phối nền tảng — Uniform (khởi tạo trọng số), Bernoulli (bài toán lỗi/nhị phân), Gaussian (nhiễu và MLE→MSE) — và biết dùng cái nào cho vấn đề nào.

### 3.2.1 Phân phối Uniform (đều)

Mọi giá trị trong khoảng `[a, b]` có **xác suất bằng nhau**.

```
p(x) = 1/(b - a)  với a ≤ x ≤ b, ngược lại p(x) = 0
```

**Ví dụ:** rung nền nhiễu trắng lý tưởng không phải uniform, nhưng **khởi tạo trọng số** thường dùng uniform trong khoảng nhỏ: `w ~ U(-0.05, 0.05)`. Điều này giúp tránh trọng số ban đầu quá lớn (làm bùng nổ) hoặc mọi nơ-ron giống nhau (đối xứng).

### 3.2.2 Phân phối Bernoulli

Biến rời rạc **0/1** (thành công/thất bại). Một tham số `p = P(X = 1)`:

```
P(X = 1) = p
P(X = 0) = 1 - p
```

**Ví dụ:** "mẫu rung này có chuẩn bị hỏng hóc hay không?" — đây chính là bài toán **bài toán phân loại nhị phân** trong Predictive Maintenance. Sản phẩm ra mô hình phóng ra `p̂` (xác suất lỗi) rồi so sánh với ngưỡng.

### 3.2.3 Phân phối Gaussian (chuẩn / normal)

Phân phối liên tục quan trọng nhất. Hai tham số: **mean `μ`** (tâm) và **phương sai `σ^2`** (độ rộng):

```
p(x) = (1 / (σ * sqrt(2π))) * exp( -0.5 * ((x - μ)/σ)^2 )
```

Hình dạng chuông đối xứng quanh `μ`; ký hiệu `X ~ N(μ, σ^2)`.

**Vì sao Gaussian phổ biến?**

- **Định lý giới hạn trung tâm (CLT):** tổng của nhiều biến ngẫu nhiên độc lập có xu hướng phân phối chuẩn, dù phân phối gốc là gì. Rất nhiều hiện tượng vật lý (nhiễu đo lường, dao động ngẫu nhiên) tuân theo điều này.
- Nhiễu rung nền (không có lỗi) thường xấp xỉ Gaussian quanh giá trị trung bình nhỏ.
- Là nền tảng của MLE dẫn đến **loss MSE** (mục 3.4–3.5).

Trong Predictive Maintenance với dữ liệu rung, ta thường kiểm tra xem phân phối amplitude ở trạng thái "ổn định" có chuẩn không; nếu không (lệch, nhiều đuôi) → nghi ngờ có hiện tượng bất thường (pitting, spalling, lỏng vít).

**So sánh nhanh hai thái cực dùng cho A1:**

| Đặc điểm | Bernoulli | Gaussian |
|---|---|---|
| Kiểu biến | rời rạc 0/1 | liên tục (real) |
| Tham số | `p = P(X=1)` | `μ` (tâm), `σ²` (độ rộng) |
| Vai trò trong đầu ra model | gắn cờ lỗi/normal | dự đoán giá trị liên tục: RUL, mức hư hại |
| Loss tương ứng (qua MLE) | Cross-Entropy | MSE |

### 3.2.4 Quy tắc 68-95-99.7 (chuẩn)

Với Gaussian:

```
P(μ - σ ≤ X ≤ μ + σ)   ≈ 0.68   (68%)
P(μ - 2σ ≤ X ≤ μ + 2σ) ≈ 0.95   (95%)
P(μ - 3σ ≤ X ≤ μ + 3σ) ≈ 0.997  (99.7%)
```

**Ứng dụng trực tiếp:** cảnh báo bất thường. Nếu giá trị rung vượt `μ + 3σ`, xác suất nó chỉ là nhiễu ngẫu nhiên rất nhỏ (<0.2%) → khả năng đang có bệnh tăng lên. Nhưng lưu ý: nếu phân phối không chuẩn thì quy tắc này không chính xác.

---

## 3.3 Kỳ vọng, Phương sai

**Mục tiêu:** Nắm hai "thước đo" trung tâm: kỳ vọng (trung bình theo xác suất) và phương sai/RMS (mức phân tán) — chỉ báo sớm hư hại bearing vì phương sai tăng trước khi biên độ trung bình đổi.

### 3.3.1 Kỳ vọng (Expected value / Mean)

Kỳ vọng = **trung bình có trọng số theo xác suất**:

```
Rời rạc:   E[X] = Σ x * P(X = x)
Liên tục:  E[X] = ∫ x * p(x) dx
Mẫu thực tế: E[X] ≈ (1/N) Σ x_i     (trung bình cộng của N mẫu)
```

**Ví dụ số (rời rạc):** số lần cảnh báo/giờ:

```
P(X=0)=0.5, P(X=1)=0.3, P(X=2)=0.2
E[X] = 0*0.5 + 1*0.3 + 2*0.2 = 0.7 (cảnh báo/giờ)
```

Tính chất tuyến tính: `E[aX + b] = a*E[X] + b` và `E[X + Y] = E[X] + E[Y]` (kể cả khi X, Y không độc lập).

### 3.3.2 Phương sai (Variance)

Phương sai đo **mức phân tán** quanh kỳ vọng:

```
Var(X) = E[(X - μ)^2] = E[X^2] - (E[X])^2
```

với `μ = E[X]`. Độ lệch chuẩn (standard deviation):

```
σ = sqrt(Var(X))
```

**Ví dụ số:** dãy 5 mẫu rung: `[1.0, 1.1, 0.9, 1.02, 0.98]`

```
μ = (1.0+1.1+0.9+1.02+0.98)/5 = 1.0
Phương sai mẫu đúng (chia N-1):
Các độ lệch: 0; 0.1; -0.1; 0.02; -0.02
   → bình phương: 0; 0.01; 0.01; 0.0004; 0.0004
Var = (0 + 0.01 + 0.01 + 0.0004 + 0.0004)/4 = 0.0208/4 = 0.0052
σ = sqrt(0.0052) ≈ 0.0721  (rung ổn định, độ lệch nhỏ)
```

So với dãy `[1.0, 1.5, 0.5, 1.3, 0.7]` (cùng μ = 1.0) nhưng `Var = 0.17`, `σ ≈ 0.412` — **phương sai tăng mạnh khi có chấn động**. Đây là một chỉ báo phát hiện sớm hỏng hóc: bệnh lý bearing thường làm **RMS/phương sai của rung tăng trước khi biên độ trung bình thay đổi**.

### 3.3.3 Đối với chuỗi thời gian

- Mean chuỗi: trung bình theo thời gian của tín hiệu.
- **RMS (Root Mean Square)** — tiêu chuẩn công nghiệp cho rung:

```
RMS = sqrt( (1/N) Σ x_i^2 )
```

Với tín hiệu có mean ≈ 0, thì `RMS ≈ σ` (phương sai). RMS là chỉ báo năng lượng dao động; tiêu chuẩn ISO 10816 dùng RMS velocity để phân loại tình trạng máy (tốt/chấp nhận/bất thường/nguy hiểm).

---

## 3.4 Covariance và Correlation

**Mục tiêu:** Đo quan hệ tuyến tính giữa 2 đại lượng bằng covariance và correlation; áp dụng cho feature selection và chẩn đoán lệch trục/bearing.

### 3.4.1 Covariance (hiệp phương sai)

Đo mức **thay đổi cùng chiều** của hai biến:

```
Cov(X, Y) = E[(X - μX)(Y - μY)]
```

- Dương: X cao thì Y có xu hướng cao. Âm: ngược lại. Gần 0: ít liên quan (tuyến tính).
- Đơn vị: tích đơn vị của X và Y, khó so sánh.

**Ví dụ số:** X = nhiệt độ (°C), Y = mức rung:

```
Điểm: X = [40, 45, 50, 55]
      Y = [0.50, 0.55, 0.60, 0.66]
μX = 47.5,  μY = 0.5775
Cov ≈ [(40-47.5)(0.5-0.5775) + (45-47.5)(0.55-0.5775)
       + (50-47.5)(0.6-0.5775) + (55-47.5)(0.66-0.5775)]/4
    ≈ [ 0.58125 + 0.06875 + 0.05625 + 0.61875 ]/4
    ≈ 0.331
```

Dương → nhiệt độ tăng, rung tăng theo. (Có thể là cả hai đều phản ánh tải/hư hại.)

### 3.4.2 Correlation (tương quan Pearson)

Chuẩn hóa covariance theo độ lệch chuẩn → **không thứ nguyên, trong `[-1, 1]`**:

```
ρ(X, Y) = Cov(X, Y) / (σX * σY)
```

- `ρ = 1`: tương quan dương hoàn hảo (Y = a + bX, b>0).
- `ρ = -1`: tương quan âm hoàn hảo.
- `ρ = 0`: **không tương quan tuyến tính** (vẫn có thể phụ thuộc phi tuyến!).

**Ví dụ số về ρ:** X là mức tải tăng dần, Y là một đặc trưng rung đo được:

```
X = [1, 2, 3, 4]
Y = [2, 3, 5, 4]

μX = 2.5 → độ lệch: -1.5, -0.5, 0.5, 1.5   → σX ≈ 1.118
μY = 3.5 → độ lệch: -1.5, -0.5, 1.5, 0.5   → σY ≈ 1.118
Cov = ((-1.5)(-1.5) + (-0.5)(-0.5) + (0.5)(1.5) + (1.5)(0.5))/4 = 4/4 = 1.0
ρ = 1.0 / (1.118 * 1.118) = 0.8
```

`ρ = 0.8`: quan hệ tuyến tính khá mạnh (Y vẫn tăng khi X tăng) nhưng chưa hoàn hảo — đúng loại "gần như cùng hướng" thường thấy giữa tải và mức rung. (Lưu ý: μ và σ đều được tính đồng nhất theo cùng công thức chia N, nên ρ không bị lệch do mẫu số.)

### 3.4.3 Vì sao quan trọng với chuỗi rung?

- **Tương quan giữa các trục X/Y/Z:** lệch trục, mất cân bằng làm thay đổi mối tương quan giữa các trục so với trạng thái bình thường.
- **Feature selection:** loại bỏ các đặc trưng tương quan quá cao (|ρ| > 0.95) để giảm độ dư thừa và overfitting.
- **Autocorrelation** (tương quan theo lag) là khái niệm trọng tâm cho chuỗi thời gian — xem mục 4.2.
- Lưu ý: correlation ≠ causation. Tương quan cao chỉ gợi ý, không chứng minh quan hệ nhân quả.

---

## 3.5 MLE — Likelihood và vì sao dùng MSE, Cross-Entropy

**Mục tiêu:** Trả lời câu hỏi "vì sao hồi quy dùng MSE, phân loại dùng Cross-Entropy?" — bằng con đường MLE chọn tham số làm dữ liệu quan sát có khả năng nhất.

### 3.5.1 Likelihood là gì?

Giả sử dữ liệu quan sát `x1, x2, ..., xN` sinh ra từ phân phối có tham số chưa biết `θ`. **Likelihood** = xác suất (mật độ) của dữ liệu **như một hàm của θ**:

```
L(θ) = p(x1 | θ) * p(x2 | θ) * ... * p(xN | θ)
```

**Nhấn mạnh sự khác biệt:**

- `p(x | θ)`: cố định θ (đã biết), hàm theo x — hàm mật độ xác suất.
- `L(θ) = p(x | θ)`: dữ liệu x cố định (đã đo), hàm theo θ — **likelihood function**.

### 3.5.2 MLE — Maximum Likelihood Estimation

**MLE:** chọn `θ` làm cho **xác suất quan sát được dữ liệu này là lớn nhất**:

```
θ̂_MLE = argmax_θ  L(θ)
```

Vì tích của nhiều số nhỏ (<1) tràn số, thực tế tối ưu **log-likelihood** (log giữ thứ tự đơn điệu):

```
log L(θ) = Σ ln p(xi | θ)
```

Tối đa hóa `log L` = tối thiểu hóa `-log L` → **negative log-likelihood (NLL)** — đây chính là cái mọi framework DL tối thiểu hóa!

### 3.5.3 MLE dẫn đến MSE — khi dữ liệu nhiễu Gaussian

Giả sử dự đoán `ŷ = f(x, w)` đúng mean, quan sát `y = ŷ + ε` với `ε ~ N(0, σ^2)`:

```
p(y | x, w) = (1/(σ sqrt(2π))) * exp( -0.5 * ((y - ŷ)/σ)^2 )
```

Log-likelihood của N mẫu:

```
log L = Σ [ -ln(σ sqrt(2π)) - 0.5 * ((yi - ŷi)/σ)^2 ]
```

Các hạng thức hằng (không phụ thuộc w) bỏ đi, tối đa `log L` tương đương tối thiểu:

```
MSE(w) = (1/N) Σ (yi - ŷi)^2
```

**Kết luận:** dùng MSE = chính là MLE dưới giả định nhiễu Gaussian, phương sai cố định. Đây là lý do MSE tự nhiên cho **hồi quy** (dự đoán giá trị liên tục như "thời gian sống còn lại RUL", "mức hư hại").

**Ví dụ số:** `ŷ = [1.0, 2.0, 3.0]`, `y = [1.1, 1.9, 3.2]`

```
MSE = ( (0.1)^2 + (-0.1)^2 + (-0.2)^2 ) / 3 = (0.01+0.01+0.04)/3 = 0.02
```

### 3.5.4 MLE dẫn đến Cross-Entropy — khi dữ liệu Bernoulli (phân loại nhị phân)

Cho bài toán lỗi/không lỗi: `y ∈ {0, 1}`. Đầu ra mô hình là xác suất `p̂ = P(y=1)`. Model dùng Bernoulli:

```
p(y | p̂) = p̂^y * (1 - p̂)^(1 - y)
```

Kiểm tra: nếu `y=1` → `p(1) = p̂`; nếu `y=0` → `p(0) = 1 - p̂`. Đúng như định nghĩa Bernoulli.

NLL (âm log-likelihood):

```
NLL = -Σ [ y_i * ln(p̂_i) + (1 - y_i) * ln(1 - p̂_i) ]
```

Đây chính là **binary cross-entropy (BCE)**.

**Ví dụ số:** mẫu lỗi `y=1`, mô hình dự đoán `p̂ = 0.9`:

```
BCE = -[ 1*ln(0.9) + 0*ln(0.1) ] = -ln(0.9) ≈ 0.105
```

Nếu mô hình kém, `p̂ = 0.2`: `BCE = -ln(0.2) ≈ 1.609` — lớn hơn nhiều → bị phạt nặng, đúng ý muốn.

**Tại sao Cross-Entropy (> MSE) cho phân loại?**

| Tiêu chí | MSE cho phân loại | Cross-Entropy (BCE) cho phân loại |
|---|---|---|
| Giả định sinh dữ liệu | nhiễu Gaussian (không hợp khi y là 0/1) | Bernoulli (đúng bản chất biến nhị phân) |
| Khi mô hình "tự tin sai" (p̂≈0 cho y=1) | phạt hữu hạn, có hạn | phạt theo **log** → tiến tới ∞, ép mô hình không dám vừa sai vừa tự tin |
| Gradient cuối softmax khi saturate | rất nhỏ (gradient vanishing) — học chậm | ổn định — học nhanh |
| Kết luận cho A1 | ok cho RUL (hồi quy), dở cho gắn cờ lỗi | **dùng cho nhãn lỗi/normal** |

Tóm gọn: biến nhị phân không có giả định Gaussian cho sai số; mô hình Bernoulli phù hợp bản chất, và hàm log phạt nặng lỗi "tự tin sai".

### 3.5.5 Mối quan hệ với log-likelihood — tóm gọn

```
Dạng bài       Giả định sinh dữ liệu    Loss tối ưu
-------------- -------------------------- -------------------------------
Hồi quy         Nhiễu Gaussian            MSE = (1/N) Σ (y - ŷ)^2
Phân loại nhị phân  Bernoulli               BCE = -Σ y ln p̂ + (1-y) ln(1-p̂)
Phân loại đa lớp   Categorical               Cross-Entropy = -Σ y_k ln p̂_k
```

Dùng đúng loss dựa trên **bản chất của đầu ra** — đừng dùng MSE cho bài toán phân loại. Trong A1: dự đoán RUL → MSE; gắn cờ lỗi sắp xảy ra → Cross-Entropy.

---

## 3.6 Định lý Bayes

**Mục tiêu:** Kết hợp niềm tin ban đầu (prior) với dữ liệu (likelihood) thành xác suất có điều kiện (posterior); giải thích vì sao cảnh báo trên sự kiện hiếm thường là dương tính giả — bài học trực tiếp cho lớp lỗi cực hiếm của A1.

### 3.6.1 Công thức

```
P(A | B) = P(B | A) * P(A) / P(B)
```

Các thành phần:

- `P(A | B)`: **posterior** — xác suất A sau khi biết B (có dữ liệu).
- `P(A)`: **prior** — xác suất A trước khi có dữ liệu (niềm tin ban đầu).
- `P(B | A)`: **likelihood** — khả năng gặp B nếu A đúng.
- `P(B)`: **evidence** — hằng số chuẩn hóa (xác suất toàn thể của B).

### 3.6.2 Ví dụ số — Predictive Maintenance

Trong nhà máy, 2% máy đang "sắp hỏng" (sự kiện F). Test rung dương tính cho F với xác suất 90%; tỷ lệ dương tính giả (báo lỗi khi tốt) là 5%.

Hỏi: một máy có **kết quả đo XẤU (dương)** thì xác suất thực sự sắp hỏng là bao nhiêu?

```
Cho:  P(F) = 0.02
      P(+ | F) = 0.90
      P(+ | OK) = 0.05
      P(+) = P(+|F)P(F) + P(+|OK)P(OK) = 0.9*0.02 + 0.05*0.98 = 0.067

P(F | +) = 0.90 * 0.02 / 0.067 ≈ 0.269
```

Chỉ **~27%**! Dù test "90% chính xác", đa số cảnh báo vẫn là **dương tính giả** — vì sự kiện hỏng hiếm (prior nhỏ). F1-score và precision phải được đánh giá cẩn thận trong bài toán hiếm (class imbalance) như thế này.

### 3.6.3 Vì sao dùng trong ML?

- **Bayesian machine learning:** thay vì 1 điểm `θ̂`, ta giữ phân phối `P(θ | data)`. Hữu ích khi ít dữ liệu, cần ước lượng **bất định** (với cảnh báo bảo trì, biết độ tin cậy của dự đoán quan trọng: tốn tiền dừng máy).
- Giải thích thiên lệch khi dữ liệu mất cân bằng (mục 3.7–3.8): prior mạnh kéo dự đoán về lớp đa số.
- Naive Bayes — baseline đơn giản hiệu quả cho phân loại đặc trưng rung.

---

## 3.7 Bias / Variance và Overfitting / Underfitting

**Mục tiêu:** Phân rã sai số thành bias/variance/noise để nhận diện underfitting và overfitting; biết cách dùng K-fold cross-validation — kèm lưu ý riêng cho chuỗi thời gian.

### 3.7.1 Phân rã bias-variance của sai số

Sai số kỳ vọng của mô hình trên mẫu mới tách thành 3 phần:

```
E[error] = bias^2 + variance + irreducible noise
```

với:

- **Bias** = sai số hệ thống: mô hình quá đơn giản, không bắt được pattern thật.
- **Variance** = mô hình quá nhạy với dữ liệu huấn luyện cụ thể: thay đổi ít dữ liệu → thay đổi lớn mô hình.
- **Irreducible noise**: nhiễu không thể loại (vd nhiễu đo về bản chất).

### 3.7.2 Bốn trạng thái điển hình

Giả sử bản chất quan hệ giữa tốc độ quay và rung là `y = 0.0004*RPM^2 + nhiễu`:

1. **Underfitting (high bias):** dùng mô hình tuyến tính `ŷ = w*RPM`. Không bắt được độ cong → sai số lớn ở cả train và test.
2. **Khớp tốt:** dùng polynomial bậc 2 → sai số train nhỏ, test nhỏ.
3. **Overfitting (high variance):** dùng polynomial bậc 10 → sai số train gần bằng 0, nhưng test **rất lớn** — mô hình học cả nhiễu và biến động ngẫu nhiên.
4. **Khớp dữ liệu thực tế:** luôn có irreducible noise → sai số tốt nhất ≈ mức nhiễu.

### 3.7.3 Định nghĩa chuẩn

- **Overfitting:** mô hình nhớ dữ liệu huấn luyện thay vì học quy luật tổng quát. Dấu hiệu: loss train nhỏ, loss validation/test cao.
- **Underfitting:** mô hình quá đơn giản, không đủ sức biểu diễn pattern. Dấu hiệu: loss train và test đều cao.

### 3.7.4 Tradeoff — đánh đổi

- Tăng độ phức tạp mô hình (thêm layer, thêm feature) → bias giảm, variance tăng.
- Điểm cân bằng tốt nhất nằm ở giữa, nơi **tổng** sai số nhỏ nhất.
- Trong DL: chọn đủ lớn để bias thấp, rồi dùng **regularization** để kìm variance (weight decay, dropout, data augmentation, early stopping, cross-validation mục 3.7.5) thay vì hoàn toàn tránh phức tạp.

**Mẹo phát hiện:** vẽ đường cong loss theo epochs:

```
Underfit:  train và val đều bắt đầu cao, vẫn giảm dần đều → cần mô hình mạnh hơn
Overfit:   train giảm, val tăng lên tiếp → dừng (early stopping) tại point val thấp nhất
```

### 3.7.5 K-fold Cross-validation

Chia dữ liệu thành `K` phần, huấn luyện `K` lần, mỗi lần dùng 1 phần làm validation và phần còn lại làm train; lấy trung bình kết quả.

```
Fold 1: valid [1]   train [2 3 4 5]
Fold 2: valid [2]   train [1 3 4 5]
...
Kết quả = mean của K lần
```

**Lưu ý chuỗi thời gian:** KHÔNG shuffle ngẫu nhiên khi cross-validation (rò rỉ quá khứ-tương lai). Phải dùng time-based split: validate trên dữ liệu **sau** thời điểm train (xem mục 4.4).

---

## 3.8 Bất cân bằng dữ liệu và tổng kết thống kê

**Mục tiêu:** Đối mặt với lớp lỗi hiếm — bỏ accuracy thuần, đánh giá bằng precision/recall/F1, và so sánh các chiến lược cân bằng lớp (oversampling, SMOTE, class weights, threshold).

### 3.8.1 Vấn đề lớp hiếm trong Predictive Maintenance

- Hầu hết thời gian máy chạy **bình thường**; sự kiện hỏng rất hiếm. Tỷ lệ có thể 1 lỗi / 10.000 mẫu.
- Mô hình luôn tiên đoán "bình thường" đạt accuracy 99.99% nhưng **vô dụng**. Đánh giá phải dùng precision/recall/F1, ROC-AUC, không dùng accuracy thuần.
- Prior trong Bayes (mục 3.6) giải thích độ lệch tự nhiên về lớp đa số → cần cân nhắc oversampling, class weights, hoặc threshold tối ưu.

**So sánh các chiến lược cân bằng lớp — chọn gì cho A1?**

| Chiến lược | Cơ chế | Ưu | Hạn chế |
|---|---|---|---|
| **Oversampling (copy)** | nhân bản thêm mẫu lỗi hiếm | đơn giản, tức thì | không thêm thông tin mới; dễ overfit nếu copy nguyên bản |
| **SMOTE** | tạo mẫu tổng hợp: nội suy giữa mẫu lỗi và k-láng giềng gần của nó | đa dạng hóa không gian đặc trưng, giảm overfit hơn copy | sinh mẫu có thể vô nghĩa nếu đặc trưng nhiễu; phải làm trong fold |
| **Class weights** | phạt sai trên lớp hiếm nặng hơn trong loss | không đụng dữ liệu, không lo leakage | phải tinh chỉnh tỉ lệ; có thể làm mô hình "khóc nhầm" |
| **Threshold tối ưu** | hạ/nâng ngưỡng quyết định trên `p̂` | không đụng model, tinh chỉnh ở bước dự đoán | cần metric mục tiêu rõ (F1, tỉ lệ báo động giả) |

Điểm chung quan trọng với chuỗi thời gian: **oversampling/SMOTE phải làm trong từng fold**, sau khi chia train/valid, để tránh cùng một mẫu (hoặc mẫu bơm từ nó) lọt cả hai phía (xem mục 4.4.4).

### 3.8.2 Bảng tóm tắt các công thức thống kê chính

```
Kỳ vọng (rời rạc)      E[X] = Σ x·P(X=x)
Phương sai             Var(X) = E[(X-μ)^2]
Độ lệch chuẩn          σ = sqrt(Var(X))
Covariance             Cov(X,Y) = E[(X-μX)(Y-μY)]
Correlation            ρ = Cov(X,Y)/(σX·σY)
MSE                    (1/N)·Σ(yi - ŷi)^2
Binary Cross-Entropy   -Σ[ yi·ln(p̂i) + (1-yi)·ln(1-p̂i) ]
Bayes                  P(A|B) = P(B|A)·P(A)/P(B)
```

---

# 4. XÁC SUẤT & THỐNG KÊ CHO CHUỖI THỜI GIAN

*(Phần đặc thù cho bài toán Predictive Maintenance bằng rung/chấn động)*

**Mục tiêu:** Nắm các khái niệm riêng của dữ liệu có thứ tự theo thời gian: tính dừng, ACF/PACF, rolling statistics, lookback/horizon và bẫy data leakage — phần đặc thù và quyết định nhất của bài toán A1.

## 4.1 Định nghĩa chuỗi thời gian và tính dừng (Stationarity)

**Mục tiêu:** Biết chuỗi thời gian là gì, "tính dừng" nghĩa là gì, cách kiểm tra và làm dừng một chuỗi rung.

### 4.1.1 Chuỗi thời gian

**Chuỗi thời gian** = dãy quan sát theo thời gian đều nhau của một biến:

```
x1, x2, x3, ..., xT
```

Với rung/chấn động:

- Tần số lấy mẫu điển hình: 10–25.6 kHz (đến vài chục kHz) cho gia tốc kế.
- 1 giây dữ liệu 25.6 kHz = 25.600 mẫu. Cần **cửa sổ** (chunk) để phân tích, thường 512–4096 mẫu.

**Ký hiệu `t`:** chỉ số thời gian/rời rạc.

### 4.1.2 Tính dừng (Stationarity) — khái niệm cốt lõi

Chuỗi gọi là **dừng (stationary)** (nghĩa hẹp, weak stationarity) nếu các đặc trưng thống kê **không đổi theo thời gian**:

```
(1) Mean không đổi:     E[x_t] = μ  (với mọi t)
(2) Phương sai không đổi: Var(x_t) = σ^2  (với mọi t)
(3) Autocovariance chỉ phụ thuộc lag k, không phụ thuộc thời điểm t:
     Cov(x_t, x_{t+k}) = γ(k)
```

**Vì sao stationarity quan trọng?**

- Hầu hết lý thuyết chuỗi thời gian (ACF, mô hình AR/ARIMA, chẩn đoán quang phổ) đòi hỏi dừng.
- Hầu hết mô hình ML/DL học quy luật chung; nếu phân phối trôi (drifting), mô hình phải được huấn luyện lại, hoặc phải dùng features không đổi theo thời gian (vd. spectral features của cửa sổ ngắn thường gần dừng hơn thô signal).
- **Ví dụ không dừng:** máy khởi động → nhiệt độ tăng dần → rung trung bình tăng dần theo thời gian. Trend không dừng.

### 4.1.3 Kiểm tra tính dừng

1. **Trực quan:** vẽ chuỗi — có trend (trôi mean) hoặc "hình phễu" (phương sai thay đổi theo thời gian)?
2. **Rolling statistics** (mục 4.3): mean/std trượt xấp xỉ hằng số?
3. **ADF test (Augmented Dickey-Fuller):** kiểm định giả thuyết "chuỗi không dừng". p-value thấp (vd < 0.05) → bác bỏ → chuỗi dừng.
4. **ACF giảm nhanh về 0** → hầu như dừng.

### 4.1.4 Xử lý khi không dừng

```
Take difference:  y_t = x_t - x_{t-1}        (loại trend tuyến tính)
Log hoặc Box-Cox: y_t = ln(x_t)              (ổn định phương sai)
Detrend bằng hồi quy: y_t = x_t - (a + b*t)
Chuẩn hóa theo cửa sổ: y_t = (x_t - μ_t) / σ_t  (z-score trượt)
```

Với chuỗi rung máy: vì có chu kỳ máy (bắt đầu-ổn định-dừng), thường tách các "runs" ổn định, hoặc dùng cửa sổ ngắn (512 mẫu) để mỗi cửa sổ xem như gần dừng.

---

## 4.2 Autocovariance và Autocorrelation

**Mục tiêu:** ACF đo tự tương quan theo lag → phát hiện chu kỳ quay và "độ nhớ" của hệ; PACF giúp chọn số lag dùng làm đặc trưng.

### 4.2.1 Autocovariance

**Autocovariance tại lag k** = covariance giữa chuỗi với chính nó, dịch đi `k` bậc:

```
γ(k) = Cov(x_t, x_{t+k}) = E[(x_t - μ)(x_{t+k} - μ)]
```

- `γ(0) = Var(x_t)`.
- `|γ(k)| ≥ |γ(k+1)|` thường khá đúng với chuỗi "trơn".

### 4.2.2 Autocorrelation / ACF

**Autocorrelation function (ACF)** — chuẩn hóa autocovariance:

```
ρ(k) = γ(k) / γ(0)  ∈ [-1, 1]
```

- `ρ(0) = 1`.
- `ρ(k) > 0`: giá trị tại `t` và `t+k` có xu hướng cùng dấu so với mean (trơn).
- `ρ(k) < 0`: đảo dấu qua lại (dao động nhanh).

**Ví dụ số:** chuỗi `[1, 2, 3, 2, 1, 2, 3, 2]` (dạng sóng):

```
μ = 2.0
x_t - μ:  [-1, 0, 1, 0, -1, 0, 1, 0]
γ(0) = (1+0+1+0+1+0+1+0)/8 = 0.5
γ(1) = [(-1)(0) + (0)(1) + (1)(0) + (0)(-1) + (-1)(0) + (0)(1) + (1)(0)]/8 = 0
ρ(1) = 0/0.5 = 0
```

Ô chạy từng ô dương/âm đuổi nhau → lag 1 không tương quan.

### 4.2.3 Vì sao ACF hữu ích cho rung?

- **Tính chu kỳ:** chuỗi rung có thành phần tuần hoàn ở tần số quay → ACF đạt đỉnh lại đúng các lag bội của chu kỳ. Điểm **đỉnh ACF → phát hiện chu kỳ/tần số chia sẻ**, đơn giản mà hiệu quả cho fault detection.
- **Độ dài nhớ của hệ thống:** ACF giảm chậm → hệ "nhớ dài", trạng thái hiện tại có thông tin về tương lai → số bước lookback cần lớn (mục 4.4).
- Độ trơn/bất thường: khi bệnh phát triển, thay đổi cấu trúc xung → ACF thay đổi (vd xuất hiện đỉnh tại tần số 2×RPM, sidebands).

### 4.2.4 Partial autocorrelation (PACF) — khái niệm

PACF `α(k)` = autocorrelation tại lag `k` **sau khi loại ảnh hưởng của các lag trung gian** (1..k-1).

```
α(1) = ρ(1)
α(2) = (ρ(2) - ρ(1)^2) / (1 - ρ(1)^2)
```

Dùng để chọn bậc mô hình AR (số lag cần dùng làm đặc trưng), thay vì ACF (dễ nhiễu chéo giữa các lag).

**So sánh ACF vs PACF — dùng cái nào?**

| Tiêu chí | ACF | PACF |
|---|---|---|
| Đo | tương quan ở lag k **tổng hợp** (gồm cả ảnh hưởng gián tiếp qua các lag trung gian) | tương quan ở lag k **sau khi đã trừ** ảnh hưởng của lag 1..k-1 |
| Hướng dùng | nhận diện chu kỳ, tính mùa, kiểm tra dừng (đỉnh tại lag bội của chu kỳ) | chọn số bậc `p` cho mô hình AR/ số lag làm đặc trưng |
| Trong A1 | phát hiện tần số quay + harmonic | tránh đưa các lag gần nhau trùng thông tin vào model |

---

## 4.3 Rolling Statistics (thống kê cửa sổ trượt)

**Mục tiêu:** Tóm tắt một cửa sổ hàng nghìn mẫu thô thành vài đại lượng (mean/std/RMS) làm feature cấp cửa sổ; biết cách chọn độ dài `W` và xử lý ngoại lệ.

### 4.3.1 Định nghĩa

Với cửa sổ độ dài `W` (vd 256, 512, 1024 mẫu), **rolling statistic** tại thời điểm `t` tính trên cửa sổ **kết thúc tại `t`**:

```
Mean trượt:    μ_t = (1/W) Σ_{i=t-W+1..t} x_i
Std trượt:     σ_t = sqrt( (1/(W-1)) Σ_{i=t-W+1..t} (x_i - μ_t)^2 )
RMS trượt:     rms_t = sqrt( (1/W) Σ_{i=t-W+1..t} x_i^2 )
```

**Ví dụ số:** chuỗi `[1, 2, 1, 2]`, `W = 2`:

```
t=2: μ = (1+2)/2 = 1.5
t=3: μ = (2+1)/2 = 1.5
t=4: μ = (1+2)/2 = 1.5
```

### 4.3.2 Vì sao dùng?

- Các chỉ số này là **feature cấp cửa sổ** — tóm tắt hàng nghìn mẫu thô thành vài số đầu vào cho model.
- Chuẩn công nghiệp ISO 10816 dựa trên **RMS velocity**. Rolling RMS tăng dần → phát hiện hư hại sớm (năng lượng rung tăng trước khi biên độ đạt ngưỡng cắt).
- Detrend + làm trơn nhiễu ngẫu nhiên; phát hiện đổi trạng thái (drift) một cách trực quan.

### 4.3.3 Chọn độ dài cửa sổ `W`

- **W nhỏ (64–256):** phản ứng nhanh, nhiễu hơn trong ước lượng; dễ sinh false alarm.
- **W lớn (1024–8192):** ước lượng vững (var giảm theo 1/W), nhưng chậm phản ứng (lag); có thể bỏ sót lỗi nhanh bất ngờ.
- **Cân nhắc:** cho phép đủ số vòng quay trong cửa sổ. Ví dụ máy 1500 RPM = 25 vòng/s. Muốn 10 vòng/cửa sổ → cửa sổ ≥ 0.4 s → @25.6 kHz ≈ **≥ 10240 mẫu**, hoặc dùng feature tần số thay vì thời gian.
- Rolling std quá lớn khi có spike → sử dụng **median + MAD** (Median Absolute Deviation) nếu bị nhiễu xung (impulse noise) làm phân phối fat-tail.

---

## 4.4 Thang đo cửa sổ, lookback: từ quá khứ → tương lai

**Mục tiêu:** Tạo dữ liệu huấn luyện TỪ CHUỖI THEO THỨ TỰ bằng lookback/horizon và sliding window, tuân thủ chiều nhân quả và chặn data leakage.

### 4.4.1 Vấn đề: phải dùng quá khứ để dự đoán tương lai

Model chuỗi thời gian học ánh xạ:

```
[ x_{t-window}, ..., x_t ] → dự đoán ở t+1, t+horizon
```

với:

- **Lookback** (window size, context length): số mẫu/bước trong quá khứ dùng làm đầu vào.
- **Horizon:** khoảng cách tới mốc cần dự đoán (vd dự đoán 1 giờ trước sự cố, hay RUL đến hỏng).
- **Strata/time alignment:** khi tạo tập hỗ trợ cho huấn luyện.

**Ví dụ cụ thể:**

Nhận cửa sổ W = 1024 mẫu rung (lookback = 1024), dự đoán xác suất lỗi trong 24 giờ (horizon = 24h). Quyết định bảo trì dựa trên dự đoán này.

### 4.4.2 Ví dụ số minh họa lookback

Cần ước lượng ban đầu dùng trung bình cửa sổ trị `a=1, b=2` trượt bậc 2 (đơn giản, minh họa ý tưởng):

```
x = [1, 2, 1, 2, ...]  (chu kỳ rung giả định)
Lookback W=2: dự đoán x_{t+1} = ave(x_t, x_{t-1})
t=3: ave(x_2, x_3) = ave(2,1) = 1.5 → dự đoán 1.5 cho x_4 (=2) — sai số 0.5
```

Vì chu kỳ 2 mẫu, lookback đủ 2 mẫu là tối thiểu; muốn dự đoán chu kỳ chuẩn hơn cần lookback ≥ chu kỳ. (Nói nôm na: đối với rung máy, lookback phải phủ **nhiều vòng quay**, không chỉ vài mẫu liền kề.)

**Quy tắc thực hành:** lookback phải **đủ dài**, nhưng dài quá làm tăng số tham số của model (trừ dùng RNN/attention xử lý chuỗi dài); feature tóm tắt cấp cửa sổ (RMS, spectral) giúp giữ thông tin với lookback ngắn.

### 4.4.3 Cấu trúc dữ liệu huấn luyện (sliding window)

Từ chuỗi dài, tạo ra nhiều mẫu bằng cửa sổ trượt:

```
X (input,  lookback=W)          y (target)
[ x1  x2  ... xW ]             [ x_{W+horizon}  hoặc  nhãn lỗi tại W+horizon ]
[ x2  x3  ... x_{W+1} ]        [ x_{W+1+horizon} hoặc nhãn ... ]
[ x3  x4  ... x_{W+2} ]        [ x_{W+2+horizon} ... ]
```

**Ví dụ:** chuỗi `x1..x8`, `W=3`, `horizon=1`, hồi quy:

```
[ x1 x2 x3 ] → y = x5
[ x2 x3 x4 ] → y = x6
[ x3 x4 x5 ] → y = x7
[ x4 x5 x6 ] → y = x8
```

### 4.4.4 Bẫy rò rỉ (data leakage) — cực kỳ quan trọng

Trong huấn luyện chuỗi thời gian, **không được để tương lai lọt vào đầu vào**:

- Chuẩn hóa/scale **chỉ dựa trên cửa sổ quá khứ** (hoặc trên phần train), KHÔNG dung mean/std của toàn bộ chuỗi (gồm cả tương lai).
- Train/valid/test split theo **thời gian** (valid sau train), không shuffle.
- K-fold: dùng **expanding/rolling origin** (train tích lũy, valid kế tiếp), lặp lại K lần như mục 3.7.5 nhưng giữ trật tự thời gian.
- Oversampling để cân bằng lớp hiếm phải làm **trong fold**, không trước khi chia (tránh cùng mẫu xuất hiện cả train lẫn valid).

**So sánh hai cách chia dữ liệu — vì sao time-based thắng cho A1?**

| Tiêu chí | Random shuffle K-fold | Time-based split |
|---|---|---|
| Nguyên tắc | trộn ngẫu nhiên rồi chia K phần | chia theo mốc thời gian: train ≤ mốc chia < valid ≤ test |
| Rủi ro | **data leakage**: mẫu tương lai lọt vào train | thấp: mô hình chỉ thấy quá khứ |
| Tính tổng quát | nếu chuỗi dừng thì ok | đúng bản chất dự báo, chấp nhận valid bị chệch nếu trend trôi |
| Kết luận | dùng khi chắc chắn chuỗi đủ dừng | **mặc định cho chuỗi rung máy** |

### 4.4.5 Baseline timeline cần chú ý

```
Quá khứ (x1..x_t) ─────────→ Mốc t ──→ Tương lai (t+1.. t+horizon)
   [input window]                [label của target]
```

Dự đoán tốt yêu cầu: tín hiệu quá khứ **có chứa thông tin có ích** (tương quan cao với tương lai — ACF giúp kiểm tra điều này), và mô hình học đúng chiều nhân quả (past → future, không reverse).

---

# TỔNG KẾT

Toàn bộ toán nền tảng này kết nối với pipeline Predictive Maintenance như sau:

1. **Thu thập:** đo gia tốc 3 trục, nhiệt độ, RPM → dữ liệu = ma trận (mục 1.1–1.2).
2. **Tiền xử lý:** chuẩn hóa đặc trưng (z-score), loại trend, kiểm tra dừng (mục 1.3, 4.1), tính cách rolling/RMS (mục 4.3).
3. **Trích đặc trưng:** từ miền thời gian (RMS, std, peak) + miền tần số (FFT → SVD/PCA có thể dùng để nén) (mục 1.4).
4. **Huấn luyện:** forward → loss (MSE cho RUL, Cross-Entropy cho lỗi/không lỗi) → backprop (chain rule) → gradient descent. Dùng đúng loss theo mục 3.5.
5. **Đánh giá:** bias-variance, cross-validation time-based, precision/recall cho lớp hiếm (mục 3.7, 3.8).
6. **Đưa ra quyết định:** dự đoán theo lookback/horizon đúng chuỗi nhân quả (mục 4.4).

**Cẩm nang kỹ thuật nhanh:**

```
Đồng bộ dữ liệu        → matrix batch, GPU (mục 1.5)
Điều chỉnh w           → gradient descent + learning rate (mục 2.4)
Huấn luyện mạng        → backprop = chain rule (mục 2.3)
Chọn loss              → phân loại: CrossEntropy; hồi quy: MSE (mục 3.5)
Chống overfit          → bias/variance, regularization, early stopping (mục 3.7)
Chuỗi thời gian        → dừng, ACF, rolling, lookback, thời gian-split (mục 4.x)
```

*(Hết tài liệu Phần 1 — Toán nền tảng. Tiếp theo có thể là Phần 2: Xử lý tín hiệu & Trích đặc trưng, và Phần 3: Mô hình ML/DL cho chuỗi thời gian.)*