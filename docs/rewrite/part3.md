## PHẦN 3 · DEEP LEARNING & GENERATIVE MODELS — DEEP LEARNING & GENERATIVE MODELS CHO CHUỖI THỜI GIAN CÔNG NGHIỆP + SINH DỮ LIỆU LỖI GIẢ

> Giáo trình luyện thi — Cuộc thi **DENSO Factory Hacks 2026**
> Bài toán **A1**: "AI bảo trì dự đoán khi thiếu dữ liệu lỗi" — dữ liệu rung chuỗi thời gian.
> Đối tượng: sinh viên AI đã biết ML cơ bản.
> Phần này trọng tâm vào: **neural network nền tảng → autoencoder → LSTM → transformer (giới thiệu) → GAN → diffusion → sinh dữ liệu lỗi giả để augment lớp hiếm**.

---

### CÂY BÀI TOÁN CON — định vị từng mô hình sẽ học trong Phần 3

Một mô hình chỉ đáng dùng khi ta biết nó giải **nhánh nào** của bài toán. Cây dưới đây phân rã MỤC TIÊU LỚN của Phần 3 thành các nhánh con; mỗi nhánh trỏ thẳng tới mục giải thích.

```
  MỤC TIÊU LỚN: (1) học đại diện từ tín hiệu rung  +  (2) sinh dữ liệu lỗi giả để bù lớp hiếm
                          │
        ┌─────────────────┴─────────────────────────────────────┐
   NHÁNH A: HỌC CẤU TRÚC THỜI GIAN                       NHÁNH B: SINH DỮ LIỆU LỖI GIẢ
   (nhận diện "thế nào là bình thường / bất thường")     (thêm mẫu cho lớp lỗi hiếm)
        │                                                      │
   ├─ Đầu vào là VECTOR đặc trưng (đã trích feature)      ├─ GAN thuần (§5.1–5.3)
   │    → Autoencoder (§2) → reconstruction error          │    nhanh, nhẹ, rủi ro mode collapse
   │    (trọng tâm của phần này)                           ├─ TimeGAN (§5.4)
   ├─ Đầu vào là CHUỖI THÔ (window × kênh cảm biến)        │    chuỗi giữ đúng động học thời gian
   │    → LSTM/GRU (§3) → phân loại window / dự báo RUL    ├─ Diffusion / DDPM (§6.1–6.3)
   ├─ Bối cảnh dài, dữ liệu LỚN                             │    rất đa dạng, học ổn định, sampling chậm
   │    → Transformer (§4) — phương án B khi đủ dữ liệu     └─ Diffusion-TS (§6.4)
   └─ Khi nào cần thêm layer sâu? (§1.6, §2.1)                   tái tạo theo tần số + conditional —
   (mô hình sâu chỉ thắng RF/XGBoost khi dữ liệu đủ lớn)         đường NÂNG CAO, cần GPU
                                                                      │
        ┌────────────────────────────────────────────────────────────────────┘
   NHÁNH C: ĐÓNG VÀO BÀI A1
        ├─ §6.5  so sánh GAN vs Diffusion → chọn đường đi thực dụng
        ├─ §7    protocol sinh+augment lớp hiếm, few-shot, FaultDiffusion, đánh giá sinh
        └─ §8    pipeline PyTorch tổng hợp cho A1 (AE + LSTM + TimeGAN tuỳ chọn + fusion)
```

**Nguyên tắc đọc:** mọi thứ đều quy về một trong hai động lực trên. AE/LSTM học **hình dạng + thứ tự** của tín hiệu; TimeGAN/Diffusion học **phân phối** của dữ liệu để bơm thêm mẫu lỗi giả. Phần còn lại (activation, loss, backprop, optimizer) chỉ là "thịt" chung cho mọi mô hình sâu.

---

1. Neural network nền tảng (MLP, activation, loss, backprop, optimizer)
2. Autoencoder (bảo trì bất thường — trọng tâm)
3. RNN / LSTM / GRU cho chuỗi thời gian
4. Transformer (giới thiệu đúng mức cần thiết)
5. Generative Adversarial Network (GAN) + TimeGAN chi tiết
6. Diffusion models (DDPM, Diffusion-TS, so sánh GAN vs Diffusion)
7. Sinh dữ liệu để augment lớp hiếm (gắn bài A1)
8. Thực hành PyTorch cho A1: pipeline tổng hợp

---

# 1. Neural network nền tảng

**Mục tiêu:** xây khối "tế bào thần kinh" đầu tiên biết học từ dữ liệu rung — một MLP có khả năng: phân loại window bình thường/bất thường, dự báo RUL, hoặc làm encoder/decoder cho Autoencoder ở §2. Hiểu xong bạn nắm được ba thứ mọi mô hình sâu đều dùng: **activation** (nguồn phi tuyến), **loss** (thước đo sai), **backprop + optimizer** (cỗ máy học).

**Bối cảnh (bài toán con):** Phần 2 đã có RF/XGBoost và Isolation Forest trên feature thủ công (RMS, kurtosis, phổ...). Câu hỏi đặt ra: *khi nào cần chuyển sang mô hình sâu?* Trả lời gọn: khi (a) ta muốn mô hình **tự học đặc trưng** thay vì trích tay, và (b) dữ liệu đủ lớn để đặc trưng tự học đó ổn định.

**So sánh — MLP/deep so với RF/XGBoost trên feature, vì sao chọn mô hình sâu:**

| Tiêu chí | RF / XGBoost trên feature tay | MLP / mô hình sâu |
|---|---|---|
| Đặc trưng | Người trích trước (RMS, spectral...) | **Tự học từ dữ liệu thô** |
| Số lượng dữ liệu cần | Ít (vài trăm mẫu vẫn tạm) | Nhiều (thường > vài nghìn) |
| Phi tuyến phức tạp | Tốt ở mức "cây" ghép | Rất tốt, đặc biệt tín hiệu tuần hoàn |
| Chi phí huấn luyện | Rẻ, ngay trên CPU | Đắt, cần GPU cho mô hình lớn |
| Vai trò trong A1 | Baseline nhanh để so sánh | Lõi của AE/lớp LSTM/GAN — học đại diện |

**Kết luận cho A1:** RF/XGBoost giữ vai trò **baseline** (chạy 1 phút, cho con số để đối chứng). Mô hình sâu chỉ trở thành lựa chọn khi ta cần thứ RF không có: **không gian ẩn** (để AE phát hiện bất thường, để GAN sinh dữ liệu). Toàn bộ Phần 3 xoay quanh "đặc trưng tự học trong không gian ẩn".

## 1.1. Từ Perceptron đến MLP (Multi-Layer Perceptron)

### 1.1.1. Perceptron: nụ cười đầu tiên

Perceptron là đơn vị tính toán nhỏ nhất của mạng nơ-ron. Với vector đầu vào `x = (x1, x2, ..., xn)`, trọng số `w = (w1, ..., wn)` và bias `b`, perceptron tính **tổng có trọng số** rồi cho qua một **hàm kích hoạt**:

```
z = w·x + b = w1·x1 + w2·x2 + ... + wn·xn + b
a = f(z)
```

- `z`: giá trị "thuần" trước kích hoạt (tổng có trọng số cộng thêm bias).
- `a`: đầu ra của neuron, là kết quả đã "kích hoạt".
- Bootstrap: `y = sign(w·x + b)`.

**Trực giác bias**: không có `b`, neuron luôn đi qua gốc tọa độ — không học được ranh giới lệch so với gốc. Bias cho phép dịch chuyển ranh giới quyết định. Ví dụ: muốn phân biệt "mức rung < 3 mm/s là bình thường theo tiêu chuẩn", đường phân cách cần nằm ở `x = 3`, không phải `x = 0`; bias chịu trách nhiệm cho sự dịch chuyển đó (`w·x + b = 0` → `3w + b = 0` → `b = -3w`).

### 1.1.2. MLP: không chỉ một neuron

Một neuron chỉ phân loại được **dữ liệu tuyến tính phân tách**. Tín hiệu rung khi bị lỗi thường có dạng phi tuyến phức tạp (xung, điều chế tần số), không cắt bằng một đường thẳng. MLP xếp thành **nhiều lớp (layers)**, mỗi lớp có nhiều neuron:

```
Lớp vào (input) → Lớp ẩn 1 (hidden) → Lớp ẩn 2 → Lớp ra (output)
 x              → a1 = f(W1·x + b1)  → a2 = f(W2·a1 + b2) → ŷ = g(W3·a2 + b3)
```

Trong đó `W` là **ma trận trọng số**, `b` là vector bias, `f` là activation phi tuyến, `g` là activation lớp ra (sigmoid/softmax cho phân loại, identity cho hồi quy).

**Forward pass (lan truyền xuôi)** — từ dữ liệu sang dự đoán:

```
# pseudocode forward pass cho mạng 1 lớp ẩn (batch = 1 mẫu)
a1 = relu(W1 @ x + b1)   # lớp ẩn, shape (hidden,)
y_hat = sigmoid(W2 @ a1 + b2)   # lớp ra, shape (1,)
```

**Cơ chế từ gốc — vì sao cần hàm kích hoạt phi tuyến?** Nếu chỉ dùng phép tính tuyến tính (không có `f`), thì dù xếp bao nhiêu lớp, tổng hợp vẫn là một phép biến đổi tuyến tính — mạng "xẹp" thành đúng một neuron. Chính `f` phi tuyến tạo nên sức mạnh biểu diễn. **Universal Approximation Theorem** nói: với đủ neuron và activation phi tuyến, MLP có thể xấp xỉ bất kỳ hàm liên tục nào. Một MLP đủ rộng cũng có thể làm encoder của Autoencoder (§2) hoặc chồng lên LSTM (§3) — đây là "gạch cơ bản" của mọi kiến trúc sâu.

## 1.2. Các hàm kích hoạt (activation functions)

| Tên | Công thức | Miền | Trực giác / lưu ý |
|---|---|---|---|
| **ReLU** | `max(0, z)` | `[0, +∞)` | Phổ biến nhất cho lớp ẩn. Đạo hàm `1` khi `z>0`, `0` khi `z<0`. Rẻ, giảm vanishing gradient. Nhược: neuron "chết" nếu luôn nhận đầu vào âm. |
| **Sigmoid** | `σ(z) = 1/(1 + e^(-z))` | `(0, 1)` | Ép kết quả về xác suất. Dùng cho lớp ra nhị phân. Đạo hàm cực đại `0.25` → khi xếp nhiều lớp, nhân liên tiếp đạo hàm nhỏ → gradient mờ dần (vanishing). |
| **tanh** | `(e^z - e^(-z))/(e^z + e^(-z))` | `(-1, 1)` | Giống sigmoid nhưng đối xứng quanh 0 → đầu ra trung bình gần 0, giúp gradient ổn định hơn. |
| **Softmax** | `p_i = e^(z_i) / Σ_j e^(z_j)` | `(0,1)`, tổng = 1 | Cho phân loại đa lớp. Chuyển logits thành phân phối xác suất. |

**Trực giác ReLU**: neuron "tắt" (0) đối với kích thích âm, "bật" tuyến tính với kích thích dương — giống cách tế bào nơ-ron sinh học chỉ phát xung khi kích thích vượt ngưỡng. ReLU giải quyết vanishing gradient vì đạo hàm là hằng số 1 trên nhánh dương: thông tin gradient không bị "nhân nhỏ dần" qua nhiều lớp.

## 1.3. Loss functions (hàm mất mát)

Loss đo **mức sai** giữa dự đoán `ŷ` và nhãn thật `y`. Mô hình tối ưu bằng cách giảm loss.

### 1.3.1. MSE (Mean Squared Error) — cho hồi quy và tái tạo tín hiệu

```
L = (1/N) · Σ (y_i - ŷ_i)²
```

- Dùng khi dự đoán **giá trị liên tục**: RUL (`máy còn sống bao lâu nữa`), nhiệt độ dự báo, hoặc **tái tạo lại tín hiệu** (autoencoder).
- MSE phạt sai lệch theo **bình phương** → sai lớn bị phạt rất nặng → gradient lớn ở sai lệch lớn → học nhanh lúc đầu.
- Đạo hàm `dL/dŷ = -(2/N)Σ(y - ŷ)`, nhân với chain rule sẽ lan về trọng số.

**Ví dụ số:** `y = 2.0`, `ŷ = 0.4` → đóng góp vào MSE là `(2.0 - 0.4)² = 2.56`; ngược lại nếu `ŷ = 1.9` thì chỉ `(0.1)² = 0.01`. Sai lệch gấp 4 lần bị phạt 256 lần — đây là lý do MSE chạy nhanh khi mô hình còn sai to, và nhạy với **outlier** (một điểm đo hỏng có thể kéo cả loss lên).

### 1.3.2. Cross-entropy — cho phân loại

**Binary Cross-Entropy (BCE)** — nhãn 0/1:

```
L = -[ y·log(p) + (1-y)·log(1-p) ]
```

- `p = σ(z)` là xác suất mô hình dự đoán lớp dương (ví dụ: "bất thường").
- Nếu `y=1`: `L = -log(p)` → muốn `p → 1`. Nếu `y=0`: `L = -log(1-p)` → muốn `p → 0`.
- **Vì sao cross-entropy thay vì MSE cho phân loại?** Với sigmoid đầu ra, MSE có đạo hàm chứa `σ'(z)` → gradient gần 0 khi mô hình sai trầm trọng (học cực chậm). Cross-entropy khử `σ'(z)`, gradient tỉ lệ `(p - y)` — sai càng to, bước cập nhật càng lớn. Hội tụ nhanh hơn hẳn.

**Categorical Cross-Entropy (CE)** — nhiều lớp. Với mã one-hot `y = (y1..yK)` và logits `z = (z1..zK)`:

```
L = -Σ_k y_k · log(softmax(z)_k)
```

Một cách viết gọn: `L = -log(p_c)` với `c` là lớp đúng — tức chỉ chịu phạt khi xác suất của lớp đúng thấp.

### 1.3.3. Tóm tắt chọn loss cho A1

| Bài toán | Loss |
|---|---|
| Phân loại window normal/anomaly (2 lớp) | BCE |
| Phân loại nhiều loại lỗi | CE |
| Dự báo RUL (số giờ còn lại) | MSE (hoặc MAE) |
| Autoencoder tái tạo window rung | MSE trên tín hiệu tái tạo |

## 1.4. Backpropagation: quy tắc chuỗi lan ngược

**Mục tiêu con:** trả lời câu hỏi "lỗi này do trọng số nào gây ra, phải chỉnh thế nào?". Ý tưởng: dùng **chain rule** (đạo hàm hàm hợp) để tính gradient của loss theo từng trọng số, rồi cập nhật trọng số ngược hướng gradient.

### 1.4.1. Chain rule

Nếu `L = f(g(h(w)))` thì:

```
dL/dw = (dL/da)·(da/dz)·(dz/dw)
```

Ví dụ mạng 1 neuron: `z = w·x + b`, `a = σ(z)`, `L = BCE(y, a)`.

```
dL/dw = dL/da · da/dz · dz/dw
      = (a - y) · a(1-a) · x
```

### 1.4.2. Backward pass từng bước (pseudocode)

Mạng 1 lớp ẩn: `x → a1 (ReLU) → ŷ (sigmoid)`, loss BCE.

```
# 1) Forward (tính sai số)
z1  = W1 @ x + b1
a1  = relu(z1)
z2  = W2 @ a1 + b2
ŷ   = sigmoid(z2)
L   = bce_loss(y, ŷ)

# 2) Backward (đạo hàm lớp ra trước)
dL_dz2  = ŷ - y                                   # đạo hàm BCE + sigmoid gộp lại
dL_dW2  = dL_dz2 · a1^T                           # shape của W2
dL_db2  = dL_dz2
dL_da1  = W2^T @ dL_dz2

# 3) Lan tiếp qua hidden
dL_dz1  = dL_da1 · relu'(z1)                       # relu'=1 nếu z1>0 ngược lại 0
dL_dW1  = dL_dz1 · x^T
dL_db1  = dL_dz1

# 4) Cập nhật (SGD)
W1 = W1 - lr * dL_dW1
b1 = b1 - lr * dL_db1
W2 = W2 - lr * dL_dW2
b2 = b2 - lr * dL_db2
```

**Trực giác từng bước**:
1. Luôn tính đạo hàm **từ lớp ra về lớp vào** (ngược chiều forward) vì loss chỉ phụ thuộc vào đầu ra, ta duyệt ngược để biết "lỗi do ai gây ra".
2. Ở mỗi lớp, gradient = gradient từ lớp trên × đạo hàm cục bộ (activation + linear).
3. "Cập nhật ngược hướng gradient": nếu `dL/dW > 0`, tăng `W` sẽ tăng loss → phải giảm `W`. Công thức `W = W - lr·dL/dW`.

Trong PyTorch, toàn bộ việc này chỉ là `loss.backward()` — autograd (tự động đạo hàm) làm hết.

## 1.5. Optimizers: cách đi xuống dốc

**Mục tiêu con:** sau khi có gradient, Chung hỏi "bước đi bao lớn và theo hướng nào?". Optimizer trả lời.

### 1.5.1. SGD (Stochastic Gradient Descent)

```
θ = θ - η · ∇L(θ)
```

- `θ`: toàn bộ tham số mô hình, `η`: learning rate, `∇L`: gradient.
- Trực giác: đi một bước ngược hướng dốc. Mỗi bước trên một **batch nhỏ** ngẫu nhiên (stochastic) → nhanh, nhiễu giúp thoát cực tiểu cục bộ nhỏ.
- Nhược: dao động vuông góc khi "thung lũng" lõm mạnh theo một hướng; không biết đổi nhịp khi sắp tới cực tiểu.

**Ví dụ lr**: `lr = 0.001` cho Adam là khởi điểm an toàn cho mạng huấn luyện trên dữ liệu chuẩn hóa (feature scale ~ 1).

### 1.5.2. Momentum

```
v = γ·v + η·∇L
θ = θ - v
```

- `γ` thường `0.9`. Trực giác: quả bóng lăn xuống dốc — tích lũy "đà" qua các bước, hướng chuyển động trung bình hóa các gradient nhiễu.
- Lợi ích: giảm dao động zíc-zắc, tăng tốc xuống đáy thung lũng.

### 1.5.3. Adam (Adaptive Moment Estimation) — lựa chọn mặc định

Adam giữ trung bình di động của gradient (`m`, "momentum") và bình phương gradient (`v`, "tỉ lệ học tự thích ứng"):

```
m_t = β1·m_(t-1) + (1-β1)·g_t                     # trung bình gradient
v_t = β2·v_(t-1) + (1-β2)·g_t²                     # trung bình gradient bình phương
m̂_t = m_t / (1 - β1^t)                             # hiệu chỉnh lệch lúc đầu
v̂_t = v_t / (1 - β2^t)
θ   = θ - η · m̂_t / (√v̂_t + ε)                     # ε ≈ 1e-8 tránh chia 0
```

Tham số mặc định: `β1=0.9`, `β2=0.999`, `ε=1e-8`.

**Trực giác**: mỗi tham số có **learning rate riêng** — tham số gradient dao động mạnh (v lớn) thì bước nhỏ; gradient ổn định (v nhỏ) thì bước lớn. Adam "tự tay lái" cho từng tham số → ít phải chỉnh lr, hội tụ tốt trong đa số bài toán, gồm cả GAN/diffusion. Nhược: cần nhớ `m`, `v` → tốn bộ nhớ (~2× tham số), và có thể overfit noise ở cuối train.

## 1.6. Overfitting và các biện pháp chống

**Overfitting**: mô hình nhớ "thuộc lòng" tập train (kể cả nhiễu) nhưng kém trên dữ liệu mới. Nguy cơ rất cao khi **dữ liệu lỗi hiếm** như bài A1 — mô hình chỉ cần vài mẫu lỗi là học vẹt. (Nhánh "khi nào nên dùng mô hình sâu/sâu hơn" trong cây bài toán con: mô hình càng sâu càng nhiều tham số → càng dễ overfit → với dữ liệu ít, dùng mô hình **nhỏ hơn** thay vì sâu hơn.)

| Phương pháp | Công thức / cơ chế | Trực giác |
|---|---|---|
| **Dropout** | Ngẫu nhiên gán 0 cho một phần neuron mỗi batch | Mạng "mù" vài đường nối mỗi lần → nhiều tập hợp con mô hình cùng học → trung bình hóa ẩn, giảm phụ thuộc vào đơn neuron |
| **Weight decay (L2)** | `L' = L + (λ/2)Σ w²` | Phạt trọng số lớn → ép trọng số nhỏ, mô hình "đơn giản" (ít gồ ghề), tránh vừa khít nhiễu. Gradient thêm `λ·w`. |
| **Early stopping** | Theo dõi validation loss, dừng khi bắt đầu tăng | Dừng đúng lúc mô hình còn tổng quát — học chậm lại vùng piano nhưng chưa học vẹt nhiễu |
| **Batch normalization** | Chuẩn hóa từng batch: `x̂ = (x-μ_b)/σ_b`, rồi nhân `γ` cộng `β` | Giữ activation cùng phân phối qua các lớp → gradient ổn định, cho phép lr cao hơn và train nhanh hơn |

Nguyên tắc quan trọng: **dropout/weight decay hoạt động kém hiệu quả khi dữ liệu lỗi quá ít** — mô hình không có gì để học từ. Khi mọi thứ khác thất bại, chiến lược kiên cường nhất là **(1) xử lý imbalance vĩ mô** (loss có trọng số, oversampling có kiểm soát) và **(2) generative augmentation** — chủ đề chính của phần 5–7.

## 1.7. PyTorch căn bản: tensor, autograd, vòng lặp huấn luyện

**Mục tiêu con:** biến toán học §1.1–1.6 thành code chạy được — nền tảng cho mọi mô hình ở các phần sau.

### 1.7.1. Tensor: dữ liệu đa chiều

Tensor giống numpy array nhưng có tính năng auto-diff. Data A1: window rung dạng `(batch, time_steps, n_signals)`.

```python
import torch
x = torch.randn(8, 128, 3)   # 8 windows, 128 time steps, 3 tín hiệu (x,y,z)
print(x.shape)               # torch.Size([8, 128, 3])
```

### 1.7.2. Autograd: tự động tính gradient

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
loss = (w * x) ** 2
loss.backward()                       # lan ngược
print(w.grad)                         # tensor([24.]) = 2*w*x**2 = 2*3*4
```

Chỉ các tensor có `requires_grad=True` mới nhận `.grad`. Mọi toán tử PyTorch đều đăng ký đạo hàm của mình vào đồ thị tính toán.

### 1.7.3. Vòng lặp huấn luyện chuẩn

```python
import torch, torch.nn as nn

model    = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn  = nn.BCEWithLogitsLoss()          # BCE + sigmoid gộp, ổn định số

for epoch in range(20):
    for x_batch, y_batch in dataloader:    # x:(B,64), y:(B,)
        optimizer.zero_grad()              # xóa gradient cũ
        logits = model(x_batch).squeeze()
        loss   = loss_fn(logits, y_batch)
        loss.backward()                    # tính gradient
        optimizer.step()                   # cập nhật trọng số
    print(epoch, loss.item())
```

Lưu ý: đặt model ở chế độ `model.train()` khi train và `model.eval()` khi đánh giá (quan trọng với dropout/batchnorm).

---

# 2. Autoencoder — công cụ bảo trì bất thường TRỌNG TÂM

**Mục tiêu:** HỌC "THẾ NÀO LÀ BÌNH THƯỜNG" — một mô hình **không cần nhãn**, nén dữ liệu rung normal thành một không gian ẩn nhỏ rồi tái tạo lại. Dùng cái "sai số tái tạo" (reconstruction error) làm điểm bất thường để phát hiện lỗi — kể cả lỗi **chưa từng thấy**. Đây là vũ khí số 1 cho A1 vì ta chỉ có dữ liệu normal dồi dào.

**Bối cảnh (bài toán con):** phân loại window thành normal/bất thường khi **không có nhãn lỗi đủ**. So với MLP có giám sát (§1) cần cả hai lớp, AE chỉ cần một lớp để học "phong cách" normal.

**So sánh — Autoencoder vs PCA vs bin-wise feature + IsolationForest, vì sao chọn AE:**

| Tiêu chí | PCA + threshold | Bin-wise feature + IsolationForest (Phần 2) | **Autoencoder** |
|---|---|---|---|
| Đặc trưng | Chiếu tuyến tính → không bắt phi tuyến | Đường histogram/share feature do người chọn | **Học phi tuyến tự động** |
| Không gian ẩn | Có (principal components) | Không | Có (latent) — vừa nén vừa **tái tạo được** |
| Bắt cấu trúc phức tạp (xung, điều chế) | Kém | Trung bình | Tốt nhờ DNN encoder |
| Vai trò trong A1 | Baseline tuyến tính | Baseline nhanh | **Phương án chính** (src/ae.py) |

Cả ba đều "học normal"; AE thắng vì mũi tên nén–giải nén phi tuyến bắt được cấu trúc tín hiệu (dáng sóng, đỉnh xung) mà PCA tuyến tính không làm nổi.

## 2.1. Kiến trúc: encoder – bottleneck – decoder

Autoencoder (AE) học cách **nén rồi khôi phục** dữ liệu mà không cần nhãn:

```
        x ──► [Encoder φ] ──► z (bottleneck, chiều nhỏ) ──► [Decoder ψ] ──► x̂
```

- **Encoder** `z = φ(x)`: nén `x ∈ R^d` thành vector tiềm ẩn `z ∈ R^p` với `p < d`. Các lớp DNN nén dần thông tin.
- **Bottleneck / latent space**: điểm giữa mỏng nhất — buộc mô hình giữ lại **những gì quan trọng nhất** để tái tạo, bỏ đi nhiễu/dư thừa. Đây chính là **bản nén thông tin** chứa "đặc trưng nền tảng" của dữ liệu.
- **Decoder** `x̂ = ψ(z)`: căng ngược `z` thành đầu ra cùng chiều với `x`.

**Trực giác tổng quát**: như nén file ảnh — giữ lại "bản chất" bức ảnh, xả ra gần giống bản gốc. Chiều bottleneck càng nhỏ → mô hình càng phải khái quát hóa mạnh (chỉ giữ đặc trưng phổ biến của dữ liệu train). Trong code repo của bạn (`src/ae.py`), encoder là `Linear(→32) → ReLU → Linear(→8)`, bottleneck = 8; decoder là `Linear(→32) → ReLU → Linear(→d_in)`. Với `d_in = 16` feature mỗi window, ta có chuỗi chiều rộng **16 → 32 → 8 → 32 → 16** — tức "phình ra rồi nén về 8 rồi phình lại": mô hình phải gói toàn bộ "bản chất normal" vào 8 con số.

_vẽ mạch feature AE 16→32→8→32→16:_
```
 x:[16] ─L16-32→ [32] ─ReLU→ [32] ─L32-8→ [8]z ─L8-32→ [32] ─ReLU→ [32] ─L32-16→ x̂:[16]
        └─────── ENCODER ───────┘   bottleneck    └──────────── DECODER ────────────┘
```

## 2.2. Huấn luyện: tái tạo + reconstruction loss

Mục tiêu: đầu ra `x̂` giống đầu vào `x` nhất. Với tín hiệu rung (số thực liên tục), loss tái tạo thường dùng **MSE**:

```
L_recon = (1/N) · Σ_i || x_i - x̂_i ||² = (1/N) · Σ_i Σ_k (x_i,k - x̂_i,k)²
```

Trong đó `x̂ = ψ(φ(x))`. Loss nhỏ → decoder khôi phục được tín hiệu từ latent → encoder đã bắt được "phong cách" dữ liệu học.

**Ví dụ số** (một mẫu, 2 chiều): `x = [1.0, 2.0]`, `x̂ = [0.9, 0.4]` → `L = (0.1)² + (1.6)² = 0.01 + 2.56 = 2.57`. Gradient sẽ đẩy decoder xuất ra chiều thứ hai gần `2.0` hơn — cụ thể, đạo hàm theo thành phần thứ hai là `2·(x̂2 - x2) = 2·(0.4-2.0) = -3.2`, nên `x̂2` sẽ được đẩy lên.

Điểm mấu chốt: **AE train hoàn toàn không dùng nhãn** → dùng được khi chỉ có dữ liệu "bình thường", đúng hoàn cảnh A1.

## 2.3. Anomaly detection với AE

### 2.3.1. Logic chính

1. **Train AE chỉ trên dữ liệu NORMAL** (phong phú, dễ thu).
2. Vì chỉ học "vẻ ngoài của normal", AE tái tạo mẫu normal rất tốt, nhưng **tái tạo kém mẫu lạ** — lỗi rung, mòn trục, xung bất thường có cấu trúc khác hẳn.
3. Điểm bất thường = **reconstruction error**:

```
Anomaly score: e = || x - φ,ψ(x) ||²   (MSE giữa chuỗi gốc và tái tạo)
```

4. Ngưỡng `th`: có thể lấy từ phân vị 99% của `e` trên tập validation normal (thủ công) hoặc dùng Otsu / double-check bằng dữ liệu lỗi thật nếu có. Trong `src/ae.py` có sẵn `threshold_from_err(rec_err, percentile=99.0)` làm đúng việc này.

### 2.3.2. Ví dụ trực giác

| Mẫu | Tái tạo | MSE | Kết luận |
|---|---|---|---|
| Normal: sin ổn định 1× 50Hz | gần như giống | 0.01 | normal |
| Bắt đầu mài mòn ổ trục: xung cộng vào | chỉ tái tạo phần "sin mượt", bỏ xung | 0.85 | bất thường ⇒ báo sớm |
| Va chạm / chập: méo hoàn toàn | tái tạo ra thứ "lạ" không giống | 3.2 | bất thường mạnh |

**Trực giác sâu**: AE là "chuyên gia về normal". Khi gặp thứ nó chưa từng thấy, nó phải "nói bừa" → sai lệch lớn. Điều này khiến AE **phát hiện được cả lỗi chưa từng gặp** (unseen faults) — vô giá trong nhà máy vì ta không thể dự trù hết mọi kiểu lỗi.

### 2.3.3. Biến thể nhanh

- **AE trên window**: chia rung thành window (vd 128 điểm × 3 trục), mỗi window là một "ảnh 2D" — AE xử lý như ảnh nhỏ.
- **AE trên feature (khuyến nghị đầu tiên)**: trích nhẹ feature (RMS, kurtosis, spectral...) mỗi window rồi cho vào FeatureAE nhỏ — chạy CPU được, ổn định (nền tảng của `src/ae.py`).
- **Conv-AE**: dùng Conv1d cho encoder/decoder → bắt cấu trúc cục bộ (đỉnh xung, hình dạng sóng).

## 2.4. Variational Autoencoder (VAE) — giới thiệu ý tưởng

**Mục tiêu:** thêm "bước đệm" từ AE sang generative models: vừa giữ khả năng anomaly detection, vừa có **latent có xác suất mượt** để về sau sinh dữ liệu mới (bàn đạp cho GAN/Diffusion §5–6).

VAE thay vì map `x → z` (một điểm), map `x → phân bố z` (trung bình `μ` + độ lệch `σ`):

```
encoder: z ~ N(μ(x), diag(σ²(x)))
generator: x̂ ~ p(x | z)   (thường decoder cho ra giá trị trung bình)
```

Mẫu `z` được lấy bằng **reparametrization trick**:

```
z = μ + σ ⊙ ε ,   ε ~ N(0, I)
```

(trick này làm cho gradient chảy xuyên qua phép lấy mẫu — phép lấy mẫu trở thành phép nhân khả vi).

**ELBO (Evidence Lower Bound) — ý tưởng** (không cần chứng minh đầy đủ): VAE cực đại hóa cận dưới xác suất log của dữ liệu:

```
L = E_{z~q}[ log p(x|z) ]  -  KL( q(z|x) || p(z) )
```

Hai số hạng đóng vai trò:
1. **Số hạng tái tạo** `E[log p(x|z)]`: gần `-MSE(x, x̂)`, yêu cầu x̂ giống x.
2. **Số hạng KL**: ép `q(z|x)` gần phân phối chuẩn tiên nghiệm `N(0, I)`, làm latent space **mượt và liên tục** — hai điểm z gần nhau cho ra hai mẫu x tương tự nhau.

**Lợi ích** so với AE thuần: latent có cấu trúc xác suất đẹp → có thể **sinh mẫu mới** bằng cách lấy `z ~ N(0, I)` rồi decode. Trong bài A1: VAE vừa làm AE anomaly detection vừa là **bước đệm khái niệm** cho generative models (GAN, diffusion) ở phần sau.

## 2.5. Code PyTorch — AE trên window feature (khớp src/ae.py)

Đây là mạch đúng file `src/ae.py`: encoder `Linear(d_in,32) → ReLU → Linear(32,8)`, decoder `Linear(8,32) → ReLU → Linear(32,d_in)`. Với `d_in=16` ta có **16 → 32 → 8 → 32 → 16**.

```python
import torch, torch.nn as nn

class FeatureAE(nn.Module):
    def __init__(self, d_in, d_hidden=32, d_latent=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_latent))          # 16 -> 32 -> 8
        self.dec = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_in))              # 8 -> 32 -> 16

    def forward(self, x):
        return self.dec(self.enc(x))                # x̂

# ------------------ training ------------------
ae    = FeatureAE(d_in=16)              # ví dụ: window nén thành 16 feature
opt   = torch.optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

normal_loader = ...                    # CHỈ dữ liệu normal
for epoch in range(60):                # default epochs trong ae.py
    for x in normal_loader:            # x: (B, 16)
        x_hat = ae(x)
        loss  = criterion(x_hat, x)    # reconstruction loss
        opt.zero_grad(); loss.backward(); opt.step()

# ------------------ anomaly detection ------------------
def anomaly_score(ae, x):
    ae.eval()
    with torch.no_grad():
        score = ((ae(x) - x) ** 2).mean(dim=-1)   # MSE theo từng window
    return score

# ngưỡng 99% từ reconstruction error của tập normal:
from src.ae import threshold_from_err
thr = threshold_from_err(rec_err_normal, percentile=99.0)
print("ngưỡng:", thr)
```

Thuật ngữ cần nhớ: `encoder`, `decoder`, `bottleneck/latent`, `reconstruction loss`, `anomaly score`, `ELBO`, `reparametrization trick`.

---

# 3. RNN / LSTM / GRU cho chuỗi thời gian

**Mục tiêu:** học **THỨ TỰ THỜI GIAN** — dự đoán từ trạng thái của cả chuỗi rung (chứ không phải từng điểm độc lập). LSTM/GRU giải quyết hai bài toán con: (a) phân loại window có lỗi hay không (dùng `h_T` cuối), (b) dự báo RUL (hồi quy số giờ còn lại). Đây là nhánh "học cấu trúc thời gian" trong cây bài toán con.

**Bối cảnh (bài toán con):** MLP xử lý window như vector vô thứ tự — quên mất rằng "xung lỗi xuất hiện giữa cửa sổ rồi dập tắt". Mô hình tuần tự duyệt từng bước và giữ "ký ức" để bắt quan hệ giữa các bước liên tiếp.

**So sánh — RNN thuần vs LSTM vs GRU:**

| Tiêu chí | RNN thuần | LSTM | GRU |
|---|---|---|---|
| Bộ nhớ dài | Kém (vanishing gradient) | Tốt (cell state + cổng) | Tốt (ít cổng hơn) |
| Số tham số | Ít nhất | Nhiều nhất (~4× RNN) | Trung bình (~3× RNN) |
| Tốc độ train / dữ liệu ít | Nhanh nhưng học được ít | Đủ nhanh, giàu biểu diễn | **Tốt nhất khi dữ liệu ít** (ít overfit) |
| Dùng cho A1 | Không nên | Nên (đủ mạnh) | Nên (ưu việt với data hiếm) |

Với A1 dữ liệu lỗi hiếm, **GRU/LSTM nhỏ** là lựa chọn sáng suốt hơn Transformer lớn (xem §4.4).

## 3.1. Mô hình tuần tự và nỗi đau vanishing gradient

### 3.1.1. Tại sao cần mô hình tuần tự?

Cửa sổ rung 128 điểm đặt cạnh nhau thành vector thì MLP xử lý được — nhưng MLP **quên mất thứ tự** và không chia sẻ tham số theo thời gian. Rung lỗi có tính **tuần tự**: một cú va chạm tạo xung kéo theo dập tắt; tần số thay đổi theo thời gian. RNN duyệt chuỗi từng bước `t`, giữ **trạng thái ẩn** `h_t` như "bộ nhớ" chứa bối cảnh tới thời điểm `t`:

```
h_t = tanh( W_h·h_{t-1} + W_x·x_t + b )
y_t = W_y·h_t + b_y          (nếu cần đầu ra từng bước)
```

- `x_t`: mẫu rung tại bước `t`.
- `h_t`: trạng thái ẩn — chứa "ký ức" của toàn bộ quá khứ tới `t`, được cập nhật tuần tự.
- Tham số `W_h, W_x` **mỗi bước dùng chung** → mô hình có thể xử lý chuỗi dài bất kỳ, và số tham số không phình theo độ dài chuỗi.

**Trực giác**: như đọc một câu — hiểu từng chữ trong ngữ cảnh của những chữ trước đó `h_{t-1}`, rồi cập nhật "ý tưởng đang đọc" `h_t`.

### 3.1.2. Vì sao RNN thuần khó train? — Vanishing / Exploding gradient

Khi lan ngược qua `T` bước, gradient nhân với `T` lần ma trận `W_h`:

```
∂L/∂W_h ≈ Σ_t (hệ số) · (W_h)^(T-t) · ...
```

- Nếu `‖W_h‖ < 1`: `(W_h)^k → 0` → **gradient tàn lụi**: các bước xa không học được → mất "bộ nhớ dài hạn".
- Nếu `‖W_h‖ > 1`: `(W_h)^k → ∞` → **gradient bùng nổ**: cập nhật trọng số nhảy loạn → NaN. (Biện pháp khẩn: gradient clipping `g = g·c/‖g‖` nếu `‖g‖ > c`.)

LSTM/GRU ra đời để chống vanishing gradient bằng cách cho phép thông tin "đi xuyên suốt" qua **đường cao tốc** cổng điều khiển (xem 3.2).

## 3.2. LSTM: Long Short-Term Memory

### 3.2.1. Ý tưởng

Thêm **cell state** `C_t` — "băng chuyền" chạy suốt chuỗi với các cổng quyết định **ghi thêm gì, quên gì, đọc ra gì**. `h_t` vẫn là trạng thái ẩn ngắn phục vụ đầu ra.

```
        ┌────────────────────────────────────────────────┐
   x_t ─┤  trong đó: cổng quên f, cổng vào i, ứng viên C̃, │
h_{t-1} ─┤  cổng ra o                                       │
        └────────────────────────────────────────────────┘
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t   ← "băng chuyền" (dòng gradient không bị triệt tiêu)
   h_t = o_t ⊙ tanh(C_t)
```

### 3.2.2. Công thức bốn cổng

Gọi `h_{t-1}` là state ẩn trước, `x_t` là đầu vào, concatenation `[h_{t-1}, x_t]`:

```
Nhớ quên:  f_t = σ( W_f·[h_{t-1}, x_t] + b_f )
Nhớ vào:   i_t = σ( W_i·[h_{t-1}, x_t] + b_i )
         C̃_t = tanh( W_C·[h_{t-1}, x_t] + b_C )   # ứng viên thông tin mới
Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Nhớ ra:    o_t = σ( W_o·[h_{t-1}, x_t] + b_o )
State ẩn:  h_t = o_t ⊙ tanh(C_t)
```

(`⊙` là phép nhân từng phần tử; `σ` sigmoid ra giá trị 0–1 đóng vai trò "cái vòi" đóng/mở.)

**Trực giác từng cổng**:
- **Forget gate `f_t`** (0–1): "Bộ nhớ cũ trong C_{t-1} còn hữu dụng không?" — 0 nghĩa là quên sạch, 1 là giữ nguyên. Ví dụ: khi máy đổi chế độ vận hành, cần quên đặc trưng chế độ cũ.
- **Input gate `i_t`** (0–1): "Thông tin mới `C̃` có đáng ghi không? Ghi bao nhiêu?" — với xung bất thường mới xuất hiện, i_t mở để ghi dấu hiệu này.
- **Candidate `C̃_t`**: ứng viên nội dung mới được đánh giá.
- **Cell update**: `C_t = (quên phần cũ) + (thêm phần mới)` — một phép cộng tuyến tính đơn giản, **dòng gradient chảy xuyên qua không bị khuếch đại/triệt tiêu** → chống vanishing gradient gốc rễ.
- **Output gate `o_t`**: "Phần nào của bộ nhớ C_t sẽ lộ ra để làm đầu ra `h_t`?"

**Vì sao LSTM hợp chuỗi rung?** Xung lỗi xuất hiện trong vài ms rồi biến mất; LSTM có thể **giữ dấu ấn** của xung qua nhiều bước (nhờ cell state) để phán quyết "window có chứa sự kiện bất thường" dù sự kiện không kéo dài hết window.

## 3.3. GRU: Gated Recurrent Unit (ngắn gọn)

GRU gộp hai cổng, bớt một state:

```
Cập nhật:   z_t = σ( W_z·[h_{t-1}, x_t] + b_z )        # "phao giữ" cũ
Đặt lại:    r_t = σ( W_r·[h_{t-1}, x_t] + b_r )        # "bỏ" bao nhiêu quá khứ
Ứng viên:   h̃_t = tanh( W_h·[x_t, r_t ⊙ h_{t-1}] + b_h )
State ẩn:   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

- `z_t` (update): đóng vai trò kết hợp forget+input của LSTM — quyết định giữ lại bao nhiêu trạng thái cũ `h_{t-1}` so với dùng trạng thái mới.
- `r_t` (reset): cho phép quên đi bối cảnh cũ khi cần — hữu ích khi tín hiệu thay đổi đột ngột (va chạm).
- GRU: ít tham số hơn LSTM (train nhanh, ít overfit với dữ liệu ít), hiệu năng tương đương trong đa số bài toán chuỗi vừa.

## 3.4. Ứng dụng cho A1: hai hướng chính

### (a) Phân loại window normal vs abnormal (nhị phân hoặc đa-lớp-lỗi)

```
window (T bước, n tín hiệu) → LSTM (lấy h_T cuối) → Linear → logit
```

- Lấy trạng thái ẩn cuối cùng `h_T` (tóm tắt toàn bộ cửa sổ) cho vào head phân loại.
- Hoặc **lấy trung bình** tất cả `h_t` (mean pooling) nếu muốn quan tâm đều các thời đoạn.
- Nhãn: BCE hoặc CE; test là "cửa sổ này bình thường hay sắp hỏng".

### (b) Dự báo RUL (Remaining Useful Life) hồi quy

```
history window (tín hiệu đã qua) → LSTM → Linear (1 đầu ra) → rul̂
```

- Đầu vào là cửa sổ quá khứ; đầu ra là **số giờ còn lại trước khi hỏng** — bài toán hồi quy (loss MSE).
- RUL "phạt không đối xứng" nếu giám khảo dùng asymmetric loss: dự đoán muộn (nguy hiểm) phạt nặng hơn dự đoán sớm. Thường làm bằng cách **giảm lr / thêm bias trong head** hoặc training có trọng số.

### (c) Sequence-to-sequence (nâng cao, cùng ý tưởng)

Nếu muốn mô phỏng "cả đoạn tương lai" thay vì một con số: encoder LSTM đọc lịch sử → latent → decoder LSTM sinh từng bước kế tiếp (giống bài máy dịch). Đây cũng chính là **mô-đun cốt lõi** của TimeGAN (§5.4) khi nó dùng RNN sinh latent theo từng bước thời gian.

## 3.5. Code ngắn — LSTM classifier PyTorch

```python
import torch, torch.nn as nn

class LSTMFault(nn.Module):
    def __init__(self, n_signal=3, hidden=64, n_layers=1, n_class=1):
        super().__init__()
        self.lstm = nn.LSTM(n_signal, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_class)

    def forward(self, x):                 # x: (B, T, n_signal)
        out, (h, c) = self.lstm(x)        # out: (B, T, hidden)
        last = out[:, -1, :]              # trạng thái ẩn cuối
        return self.head(last).squeeze(-1)   # logit giờ

model     = LSTMFault()
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    for x_b, y_b in loader:              # y_b: 0 hoặc 1
        logit = model(x_b)
        loss  = loss_fn(logit, y_b)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # chống bùng
        optimizer.step()
    # đánh giá: (logit > 0) => predict 1
```

Để đổi sang GRU chỉ cần thay `nn.LSTM(...)` bằng `nn.GRU(...)`; `out[:, -1, :]` không đổi.

---

# 4. Transformer (giới thiệu đúng mức cần thiết)

**Mục tiêu:** nắm đủ ý tưởng để biết **KHI NÀO dùng** và khi nào KHÔNG — vì với A1, Transformer chỉ là phương án B. Hiểu cơ chế attention cũng giúp đọc hiểu Diffusion-TS (§6.4) dùng transformer làm denoiser.

**Bối cảnh (bài toán con):** LSTM truyền ngữ cảnh xa qua trung gian (`h_t`) → thông tin suy yếu, duyệt tuần tự → chậm. Attention cho phép mọi vị trí **nối thẳng** với mọi vị trí khác trong một hop.

**So sánh — RNN/LSTM vs Transformer:**

| Tiêu chí | LSTM/GRU | Transformer |
|---|---|---|
| Kết nối ngữ cảnh xa | Qua trung gian `h_t` (suy yếu dần) | **Trực tiếp** 1 hop (attention) |
| Song song hóa | Kém (duyệt tuần tự) | Tốt (tính song song mọi cặp) |
| Complexity theo độ dài | `O(T)` bước tuần tự | `O(T²)` cặp vị trí |
| Dữ liệu cần | Ít → vừa | Rất nhiều (tránh overfit) |
| Với A1 (dữ liệu lỗi vài chục–vài trăm) | **Chọn** | Không nên (trừ ngữ cảnh đủ lớn) |

## 4.1. Self-attention: Q, K, V

RNN duyệt tuần tự (chậm, khó song song hóa, khó giữ ngữ cảnh xa). Transformer cắt hết sự tuần tự: **mỗi phần tử nhìn trực tiếp mọi phần tử khác** bằng cơ chế attention. Trước khi tính, mỗi vị trí `i` được chiếu thành ba vector:

```
Query_i  = W_Q · x_i        (ẩn danh: "tôi đang tìm kiếm gì?")
Key_i    = W_K · x_i        ("tôi đang cung cấp loại thông tin gì?")
Value_i  = W_V · x_i        ("nội dung thực sự của tôi là gì?")
```

Độ liên quan giữa vị trí `i` và `j` là **dot-product** Query với Key, chia căn chiều để giữ độ lớn ổn định, rồi softmax hóa:

```
Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V
```

**Từng phần tử**:
- `Q·K^T`: ma trận score, phần tử `(i,j)` = mức "vị trí i quan tâm tới j".
- `√d_k`: chuẩn hóa để tích nội không bùng (phương sai tỉ lệ `d_k`).
- `softmax` trên dòng: chuẩn hóa thành trọng số tổng bằng 1.
- Nhân với `V`: đầu ra của vị trí `i` = **tổng có trọng số** các `Value` đã chú ý.

**Trực giác cho chuyện rung**: phần tử "đỉnh xung" có Query hỏi "ai là hàng xóm của ta?" — hàng xóm xa nhưng có đặc trưng tương tự sẽ được đánh giá quan trọng → nhận biết mô-típ lỗi lặp lại ở bất kỳ đâu trong cửa sổ, dù cách xa nhau.

## 4.2. Multi-head attention & positional encoding

- **Multi-head**: chạy nhiều `Attention(·)` song song với `W_Q/K/V` khác nhau (thường 8 head), mỗi head học một kiểu quan hệ khác nhau (một head để ý biên độ sóng, một head để ý tần số, một head để ý độ trễ giữa các xung...), đầu ra nối lại rồi chiếu tuyến tính.
- **Attention trái / phải / full**: có thể chọn "chỉ nhìn lại quá khứ" (causal mask) cho dự báo tương lai để tránh leak.

**Positional encoding (PE)**: vì attention không mang thứ tự, thêm tín hiệu vị trí vào embedding. Ý tưởng phổ biến — sin/cos theo chu kỳ:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Các tần số khác nhau (chu kỳ khác nhau) như **la bàn đa tần số**: tần số cao mô tả vị trí gần nhau, tần số thấp mô tả vị trí xa — cho phép mô hình suy ra thứ tự tương đối. (Với chuỗi rung có thể thay bằng PE học được hoặc chỉ dùng đơn vị thời gian thật.)

## 4.3. Liên hệ với chuỗi thời gian: attention thay thế temporal dependency

- Trong RNN, mối quan hệ dài hạn chỉ được truyền "ngấm ngầm" qua `h_t`; trong Transformer, **mọi cặp vị trí nối trực tiếp** một hop — không mất thông tin qua nhiều trung gian, không chịu vanishing gradient theo độ dài.
- Có thể đưa vào biến thời gian: độ trễ giữa tọa độ = chiều dài sóng, đỉnh-đỉnh... làm **input feature** cho Query/Key.
- Nhược: complexity `O(T²)` (T² cặp), cần nhiều dữ liệu, nhiều tham số riêng.

## 4.4. Trong thực tế bài A1 (lời khuyên quan trọng)

> **LSTM nhỏ hiệu quả hơn Transformer lớn khi dữ liệu ít.**

Vì sao?
1. Transformer cần rất nhiều dữ liệu để học global attention pattern; A1 dữ liệu lỗi chỉ vài chục–vài trăm window. LSTM nhỏ (1 lớp, 32–64 hidden) có ~hàng nghìn tham số, trong khi Transformer có triệu tham số → **overfit nhanh**.
2. Attention toàn cục phù hợp ngữ cảnh dài; window rung 128–256 bước là **ngữ cảnh vừa phải** — LSTM/Dilated Conv đủ mạnh và ổn định hơn.
3. Chi phí giám khảo tính điểm thường dựa ROC/AUC trên test thật: mô hình tổng quát hóa tốt với ít dữ liệu thắng, không phải mô hình "ngầu" nhất.

Trong AI pipeline của bạn: ưu tiên Autoencoder (không giám sát) + LSTM/GRU nhỏ (có giám sát). Transformer chỉ nên là phương án B khi dữ liệu đủ lớn (hàng chục nghìn mẫu) — hoặc dùng **Diffusion-TS-style transformer encoder** trong phần sinh dữ liệu (6.4), khi dữ liệu tổng sinh nhiều hơn.

---

# 5. Generative Adversarial Network (GAN)

**Mục tiêu:** SINH DỮ LIỆU LỖI GIẢ — học phân phối của lớp lỗi hiếm rồi sinh thêm mẫu để "dày" lớp đó, giúp classifier có giám sát học tốt hơn. Đây là nhánh B của cây bài toán con (bù lớp hiếm).

**Bối cảnh (bài toán con):** ở §2 ta dùng AE để *đánh giá* bất thường, nhưng vẫn thiếu **mẫu lỗi** cho bước phân loại. GAN giải bài toán ngược: học `p_data` để vẽ ra những mẫu mới chưa từng tồn tại.

**So sánh sơ bộ — GAN thuần vs TimeGAN vs Diffusion (chi tiết ở §6.5):**

| Tiêu chí | GAN thuần | **TimeGAN** | Diffusion (DDPM/Diffusion-TS) |
|---|---|---|---|
| Giữ động học thời gian | Kém | **Tốt** (supervised + embedding) | Tốt (Fourier + reconstruct) |
| Ổn định train | Khó (minimax) | Khó nhưng kiểm soát được | Rất ổn định |
| Chi phí | Nhẹ | Nhẹ–vừa | Nặng (cần GPU) |
| Chất lượng với dữ liệu hiếm (A1) | Collapse dễ | Chấp nhận được (few-shot) | Cao nếu fine-tune tốt |

## 5.1. Ý tưởng minimax game

GAN = trò chơi hai người chơi:
- **Generator G**: nhận nhiễu `z` ngẫu nhiên, cố sinh dữ liệu giả khiến D không phân biệt được.
- **Discriminator D**: nhận dữ liệu (thật `x` hoặc giả `G(z)`), cố phân loại thật/giả.

Mục tiêu chung (minimax):

```
min_G max_D  V(D, G) = E_{x~p_data}[ log D(x) ] + E_{z~p_z}[ log(1 - D(G(z))) ]
```

- `D(x) ≈ 1` (giỏi nhận thật) → số hạng thứ nhất lớn khi D tốt.
- `D(G(z)) ≈ 0` (nhận ra giả) → số hạng thứ hai lớn khi D tốt; G muốn làm `D(G(z)) → 1`.
- **Trạng thái cân bằng (Nash equilibrium)**: `D(x) = 1/2` với mọi x — D vô phương phân biệt, khi đó `p_g = p_data` (phân phối dữ liệu giả khớp dữ liệu thật).

**Trực giác**: như nghệ sĩ giả mạo (G) và chuyên gia giám định (D) đua nhau; nghệ sĩ ngày càng khéo vì bị giám định phát hiện, giám định ngày càng tinh vì nghệ sĩ làm khó. Điểm cân bằng lý tưởng: giám định phải đoán mò.

## 5.2. Kiến trúc train — hai model đấu nhau

```
Real data x ───► D ──► D(x) ≈ 1        (nhãn 1: thật)
Noise z ───► G ──► G(z) ─► D ─► D(G(z)) ≈ 0   (nhãn 0: giả)
              └────── ép D thấy "0" nhưng G dùng loss này để học ┘
```

Training loop (luân phiên):
1. **Cập nhật D**: batch thật (nhãn 1) + batch giả (nhãn 0) → loss BCE.
2. **Cập nhật G**: cố định D, sinh batch giả với nhãn cheat `1` → loss BCE nhỏ khi D nhận nhầm giả là thật. Gradient từ D chảy ngược vào G → G chỉnh hướng sinh.

```python
# phác thảo vòng lặp GAN (nhớ D đấu G đối kháng)
for step in range(steps):
    # (1) train D
    real = batch_real_data()                 # x_real
    d_real = D(real);  loss_d_real = BCE(d_real, 1)
    z = torch.randn(batch, z_dim)
    fake = G(z);       d_fake = D(fake.detach()); loss_d_fake = BCE(d_fake, 0)
    loss_d = (loss_d_real + loss_d_fake)/2
    d_opt.zero_grad(); loss_d.backward(); d_opt.step()

    # (2) train G (cheat: ép fake thành 1)
    fake = G(z)
    loss_g = BCE(D(fake), 1)
    g_opt.zero_grad(); loss_g.backward(); g_opt.step()
```

## 5.3. Mode collapse và training instability

- **Mode collapse**: G "hack" D bằng cách chỉ sinh một kiểu mẫu luôn đánh lừa được; dữ liệu sinh nghèo. Dấu hiệu: các mẫu sinh lặp lại, loss D bất thường.
- **Instability**: D mạnh quá → gradient G rỗng; D yếu quá → G vô định; hai bên "giằng co" oscillate. Cực đại `-log D(G(z))` bão hòa số học.
- Biện pháp kinh điển: **WGAN-GP** (Wasserstein loss + gradient penalty), label smoothing (dùng nhãn 0.9 thay 1), learning rate riêng cho G và D (`lr_g < lr_d`), thêm noise vào input D, spectral normalization.

## 5.4. GAN cho chuỗi thời gian: TimeGAN (NeurIPS 2019)

### 5.4.1. Vì sao GAN thuần sinh chuỗi khó?

Chuỗi thời gian không chỉ cần **hình dạng** đúng mà còn cần **động học — mối quan hệ các bước liên tiếp** (autocorrelation, xu hướng, tần số). GAN thuần (z ngẫu nhiên → sequence) dễ sinh ra chuỗi "đẹp từng khung" nhưng **vô nghĩa về thứ tự thời gian**: bước t+1 không liên quan bước t.

### 5.4.2. Ý tưởng cốt lõi TimeGAN

TimeGAN huấn luyện **trong không gian embedding** thay vì không gian dữ liệu thô, và kết hợp đồng thời nhiều mất mát, để vừa khớp phân phối tĩnh vừa khớp **phân phối có điều kiện theo thời gian**:

```
1. Reconstruction loss: E[ || x - x̂ ||² ]      (autoencoder embedding: emb + rec)
2. Supervised loss:     E[ || h_{t+1} - s(h_t) ||² ]   (supervisor học bước chuyển tiếp)
3. Unsupervised (adversarial) loss: GAN trong latent (dis vs gen)
4. Embedding supervision: buộc mẫu synthetic giữ động học (supervisor chạy trên h̃ giả)
```

> **Ghi chú nguồn:** TimeGAN gốc (jsyoon0823/TimeGAN) viết bằng **TensorFlow 1.15** — không chạy được trên Python ≥ 3.8/3.11. Bản repo này dùng **`src/timegan_torch.py`** — triển khai lại bằng **PyTorch, trung thành kiến trúc/paper, chạy CPU**, được triệu gọi bởi `scripts/03_train_timegan.py`. Mô tả dưới đây khớp đúng cách bản code đó tổ chức mạng và loss.

**Cấu trúc mạng (khớp `src/timegan_torch.py`) — 5 module, tất cả đều là GRU:**

```
          X (thật, (B,T,d))
              │
     [Embedder e] : GRU ───────────► H (real embedding)
              │                          │
      [Recovery r] : GRU+Linear          │
              │                          ▼
              └──► X̂ (tái tạo)     [Supervisor s] : GRU  → dự đoán H_{t+1}
                                     [Generator g] : GRU   → H̃ (fake embedding) từ Z
                                     [Discriminator d] : GRU+Linear → logit(thật/giả)
```

- **Embedder `e_φ`: `X → H`** — nén chuỗi thành latent chuẩn (GRU). `Embedder(d, hidden, n_layer)`.
- **Recovery `r_ξ`: `H → X̂`** — đưa latent về không gian dữ liệu (GRU + Linear). Ghép với Embedder thành một AE trong latent. `Recovery(hidden, d, n_layer)`.
- **Generator `g_θ`: `Z → H̃`** — nhận nhiễu `z ~ N(0,I)` (shape `(B,T,z_dim)`) sinh **latent tổng hợp**, KHÔNG sinh raw. `Generator(z_dim, hidden, n_layer)`.
- **Supervisor `s_ψ`: `H → H_{t+1}`** — mô hình chuyển tiếp: học `h_t → h_{t+1}` để giữ động học. `Supervisor(hidden, n_layer)`.
- **Discriminator `d_φ'`**: phân biệt `H` (thật) vs `H̃` (giả) ngay trong latent (GRU + Linear → logit). `Discriminator(hidden, n_layer)`.

**Bốn loss trong vòng lặp (khớp code):**

1. **Reconstruction loss**: `rec_loss = MSE(X̂, X)` — học Embedder+Recovery tái tạo chuỗi từ latent.
2. **Supervised loss (trên thật)**: `sup_loss = MSE( sup(H[:, :-1]), H[:, 1:] )` — supervisor dự đoán bước kế `h_{t+1}` khớp bước thật.
3. **Unsupervised / adversarial loss**: discriminator phân biệt `dis(H)` và `dis(H̃)` bằng BCE; generator chịu `g_adv = BCE(dis(H̃), 1)` (cố đánh lừa).
4. **Embedding supervision (trên giả)**: bảo động học cho mẫu synthetic — code lấy `H_fake_next = emb(rec(H_fake)[:, :-1])` (đẩy một bước giả qua recovery rồi nhúng lại bằng embedder để tạo "bước kế") và buộc `sup(H_fake[:, :-1])` khớp nó: `g_sup = MSE(sup(H_fake[:,:-1]), H_fake_next)`.

Code thực thi theo hai pha lặp liên tiếp trong một vòng:
```
# pha 1 (joint): cập nhật emb+rec+sup theo (rec + 10·sup), rồi cập nhật gen theo (g_adv + 10·g_sup)
# pha 2 (adversarial): cập nhật dis trên (BCE thật=1, giả=0); các mạng dùng Adam lr=1e-3
```

Trọng số `10.0` cho supervised loss giúp mô hình "nghiêng" về giữ động học thời gian — đúng ý đồ paper. Khi sinh dữ liệu: `z → generator → H̃ → recovery → X̃` (chuỗi giả trong không gian quan sát).

**Vì sao hợp chuỗi**: bằng cách "đưa về không gian embedding (tĩnh) + buộc mô phỏng động học (supervised)", TimeGAN buộc mẫu sinh **giữ đúng tương quan thời gian** — điều GAN thuần không làm được. Đây là chuẩn vàng cho sinh chuỗi gen/tín hiệu, phù hợp để tạo thêm dữ liệu rung lỗi cho A1 (dù với dữ liệu lỗi rất ít, cần kỹ thuật few-shot ở mục 7).

**Công thức tổng loss** (ý niệm):

```
min_{G,S,E} max_D  λ_recon · L_recon(E,R) + λ_sup · L_sup(S) + L_adv(D, G, E)
```

_(λ_sup = 10 trong code)_

## 5.5. Lưu ý thực hành với chuỗi rung

- Chuẩn hóa mạnh (z-score từng tín hiệu) trước khi vào GAN — GAN cực nhạy scale.
- Sinh window độc lập (vd 128 bước) rồi dùng cho classifier — tránh sinh chuỗi dài (khó học tần số cao).
- Luôn luôn **đánh giá chất lượng sinh** bằng discriminative score (mục 7.4) trước khi dùng cho train.

---

# 6. Diffusion models — TRỌNG TÂM nhánh hiện đại nhất

**Mục tiêu:** sinh dữ liệu lỗi **rất đa dạng, ít mode collapse** hơn GAN — học cách "lột lớp nhiễu" ngược lại để tái tạo dữ liệu từ nhiễu thuần. Đây là đường **nâng cao** cho A1 (cần GPU, đắt hơn TimeGAN).

**Bối cảnh (bài toán con):** TimeGAN làm tốt với dữ liệu ít nhưng có thể thiếu đa dạng. Khi bạn cần **chất lượng tần số cao** (giữ đúng BPFO/BPFI của lỗi ổ trục) và đã có GPU — diffusion là ứng cử viên mạnh nhất.

**Khi nào chọn Diffusion thay vì TimeGAN** (so sánh đầy đủ xem §6.5): ổn định huấn luyện (không minimax), mẫu đa dạng, giữ tần số tốt; đánh đổi bằng chi phí sampling cao (nhiều forward) và bắt buộc lượng tính toán lớn.

## 6.1. DDPM (Denoising Diffusion Probabilistic Models) — ý tưởng

Diffusion = **học cách đảo ngược quá trình "bôi nhiễu dần thành trắng"**:

1. **Forward (làm hư)**: từ dữ liệu sạch `x_0`, dần thêm Gaussian noise qua `T` bước (vd T=1000) cho tới khi thành `x_T ~ N(0, I)` thuần nhiễu.
2. **Reverse (khôi phục)**: học mạng thần kinh `ε_θ` dự đoán nhiễu từng bước, lật ngược quá trình để từ `x_T` khôi phục `x_0`.

Trực giác: như xóa mờ rồi vẽ lại từng lớp bụi — mô hình vừa thêm "thịt mới" vừa tin vào cấu trúc dữ liệu gốc.

### Forward noising — công thức

Mỗi bước thêm nhiễu Gaussian xác định bởi băng nhiễu `β_t ∈ (0,1)` (nhỏ, tăng dần):

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)
```

Nhờ tính chất Gaussian, **nhảy thẳng `x_0 → x_t`**:

```
x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε ,   ε ~ N(0, I)
với ᾱ_t = Π_{s=1}^t (1-β_s)
```

- Khi `t` lớn, `ᾱ_t → 0` → `x_t ≈ ε`: dữ liệu đã thành nhiễu thuần.
- Công thức trên là tất cả những gì ta cần để "chụp ảnh" `x_t` ở mọi bước mà không cần lặp forward.

### Reverse denoising — công thức

Nếu biết `q(x_{t-1}|x_t)` thì có thể quay ngược. Rất tiếc nó phụ thuộc vào toàn `x_0` và không tractable — ta **xấp xỉ bằng mạng** `p_θ`:

```
p_θ(x_{t-1} | x_t) = N( x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t,t) )
```

Với đầu ra μ_θ được bóc tách: mạng dự đoán **nhiễu đã thêm** `ε_θ(x_t, t)`, rồi

```
x_{t-1} = (1/√α_t) · ( x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t) ) + σ_t·z
```

`α_t = 1-β_t`; `σ_t` là độ nhiễu tùy lịch trình (thường `σ_t² = β_t`). Vế phải trừ đi phần nhiễu dự đoán rồi giữ "định hướng" dữ liệu còn lại.

## 6.2. Mất mát — epsilon-prediction

DDPM huấn luyện **mạng dự đoán nhiễu**: lấy ngẫu nhiên bước `t`, cộng nhiễu thật `ε`, yêu cầu mạng bóc ra `ε_θ`:

```
L = E_{t, x_0, ε}[ || ε - ε_θ( x_t, t ) ||² ]
```

- `x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε`, `t ~ Uniform(1..T)`.
- Về bản chất gần **MSE giữa nhiễu thật và nhiễu dự đoán** — mạng học "lượng nhiễu đã bị đổ vào ở bước t, tôi lột ra đúng bao nhiêu".
- Mạng `ε_θ` thường là UNet (ảnh) hoặc transformer/MLP (chuỗi), nhận thêm biến bước `t` (time embedding) để biết "đang ở bước nào".

### Ví dụ số (bước cô đọng)
- `x_0` = window rung thật (chuẩn hóa).
- Chọn `t = 500`, lấy `ε` ngẫu nhiên; tính `x_500 = √(ᾱ)·x_0 + √(1-ᾱ)·ε`.
- Mạng trả `ε_θ(x_500, 500)`; loss = `‖ε - ε_θ‖²`.
- Ở `t` nhỏ (nhiễu ít): gần như khôi phục chính xác; ở `t` lớn: nhiệm vụ là "bóc nhiễu hỗn loạn ra khỏi nền tín hiệu" — học cấu trúc toàn cục dữ liệu.

## 6.3. Sampling — lặp khử nhiễu

Sinh mẫu mới = bắt đầu từ nhiễu thuần rồi **lặp T lần** đi ngược về dữ liệu:

```
x_T ~ N(0, I)
for t from T down to 1:
    z ~ N(0, I)           (z = 0 nếu t = 1)
    x_{t-1} = (1/√α_t)·( x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ(x_t, t) ) + σ_t·z
return x_0               # mẫu sinh
```

- Mỗi vòng: "bóc nhiễu dự đoán một lớp, cộng chút nhiễu mới để khử sai số".
- Tốc độ: cần T lần forward → chậm (có acceleration DDIM để giảm bước).

## 6.4. Diffusion-TS (ICLR 2024) — diffusion cho chuỗi thời gian

**Ý tưởng chính**: huấn luyện denoiser bằng cách dự đoán **chính dữ liệu sạch x_0** (reconstruction) thay vì dự đoán nhiễu `ε`, kết hợp **phân rã trend/seasonal** và **Fourier loss** — rất hợp chuỗi rung có thành phần chu kỳ/tần số.

> **Lưu ý đường nâng cao:** Diffusion-TS có chi phí tính toán cao (denoiser transformer + nhiều bước sampling) → **cần GPU**. Với đề thi giới hạn thời gian, chỉ theo đường này sau khi AE + TimeGAN đã chạy tốt; code minh họa trong `src/Diffusion-TS/`.

### Các điểm chính

1. **Kiến trúc**: encoder-decoder dạng transformer làm denoiser `G_θ(x_t, t, c)` — không dùng UNet, mà dùng transformer cho chuỗi; hỗ trợ **conditional generation theo class** (điều kiện `c` = "normal" hoặc từng loại lỗi).
2. **Reconstruct thay vì predict noise**: mạng xuất trực tiếp ước lượng `x̂_0 = G_θ(...)`, loss:

```
L = E[ || x_0 - G_θ(x_t, t, c) ||² ]
```

Trực giác: với chuỗi, "dự đoán x_0" thường học ổn định hơn "dự đoán noise" (logit thuần tín hiệu), vì tín hiệu có cấu trúc (biên độ, tần số) rõ hơn noise vô định hình.

3. **Phân rã trend/seasonal**: đầu vào `x_t` được tách thành **trend** (xu hướng chậm, nhẵn) và **seasonal** (chu kỳ lặp); hai nhánh denoise riêng rồi gộp. Với rung: seasonal nắm thành phần tần số quay cơ bản (vd 1×, 2×), trend nắm trôi chậm theo mòn máy.

4. **Fourier loss**: thêm loss trên biến đổi Fourier:

```
L_fourier = E[ || F(x_0) - F(x̂_0) ||² ]  (F = DFT, phổ biên độ/tần số)
```

Vì chuỗi tín hiệu "sống" ở miền tần số nhiều hơn miền thời gian (2 dòng chỉ khác nhau ở pha, giống phổ chuỗi mẹ). Bắt phổ → mẫu sinh giữ đúng tần số đặc trưng của lỗi (BPFO, BPFI...), không "bóp méo sóng" thành cấu trúc tần số sai.

5. **Conditional generation**: diffusion gắn thêm biến điều kiện lớp vào denoiser; lúc sampling ngược, đưa `c = loại lỗi` vào mỗi bước → sinh **đúng chủng loại** lỗi mong muốn. Đây chính là công cụ ta cần để "sản xuất hàng loạt dữ liệu lỗi loại X".

### Công thức gộp loss (ý niệm)

```
L = E[ ||x_0 - G_θ(x_t,t,c)||² ] + λ_F · E[ ||F(x_0) - F(G_θ(x_t,t,c))||² ]
```

Khi chỉ có vài mẫu lỗi thật, huấn luyện conditioned diffusion thường được tinh chỉnh (fine-tune từ pretrain trên normal) hoặc dùng kỹ thuật few-shot (mục 7.2–7.3).

## 6.5. So sánh GAN vs Diffusion cho sinh dữ liệu

| Tiêu chí | GAN | Diffusion (DDPM/Diffusion-TS) |
|---|---|---|
| **Chất lượng mẫu / độ đa dạng** | Có thể cao nhưng rủi ro mode collapse — thiếu đa dạng | Rất đa dạng (bao phủ cả distribution), ít mode collapse |
| **Độ ổn định huấn luyện** | Nổi tiếng khó — G vs D giằng co, cần điều chỉnh nhiều | Ổn định, loss đơn giản (MSE dự đoán), không trò chơi minimax |
| **Tốc độ sampling** | Rất nhanh (1 forward) | Chậm (T=1000 bước) — cần DDIM để giảm số bước |
| **Dữ liệu ít (A1)** | Học nhanh "vá" nhưng collapse dễ | Học chắc nhưng cần đủ bước; few-shot cần pretrain+fine-tune |
| **Struct chuỗi (tần số)** | Kém tự nhiên (dễ mất tương quan thời gian) | Tốt nếu dùng Fourier+reconstruct (Diffusion-TS) |
| **Tài nguyên / Việt hóa code** | Nhẹ, code cổ điển | Nặng hơn (nhiều forward), cần GPU tầm trung |

**Kết luận thực dụng cho A1**: Nếu chỉ muốn augment vài trăm mẫu để classifier lên hạng — **GAN (TimeGAN)**: nhanh, đủ. Nếu muốn chất lượng tần số, đa dạng lỗi, không chịu mode collapse và có GPU + ít thời gian — **Diffusion (kiểu Diffusion-TS)**: chất lượng cao hơn, đặc biệt cho rare-class generation với fine-tune. Trong đề thi giới hạn thời gian: **ưu tiên đường GAN-TimeGAN + AE; diffusion là đường nâng cao.**

---

# 7. Sinh dữ liệu để AUGMENT lớp hiếm (gắn bài A1)

**Mục tiêu:** khép lại vòng đời: dùng generative model đã học (§5–6) để **bù thêm mẫu lỗi giả**, rồi train classifier tốt hơn trên lỗi **THẬT** — và đảm bảo việc đánh giá không bị "nhiễu" bởi mẫu sinh.

**Bối cảnh (bài toán con):** AE/LSTM đã có; lớp lỗi vẫn chỉ vài mẫu. Đây là công đoạn "phễu" nối generative (§5–6) với supervised (§3): sinh → augment → train → đánh giá trên thật.

**So sánh cách bù lớp hiếm (chọn đường nào):**

| Cách | Chi phí | Giữ bản chất lỗi | Rủi ro |
|---|---|---|---|
| Oversampling đơn giản (lặp mẫu lỗi) | Rẻ nhất | Không đổi | Overfit — classifier học vẹt |
| Augmentation tín hiệu (shift, noise, warp) | Rẻ | Một phần | Không tạo khái niệm mới |
| **Generative (TimeGAN/Diffusion) — §5–6** | Trung bình–cao | Tốt | Cần đánh giá sinh (7.4) |
| Clustering + SMOTE (Phần 2) | Rẻ | Trung bình | Khó trên chuỗi dài |

## 7.1. Protocol đầy đủ (khung làm việc)

```
[Bước 1] Tập train: normal (đầy đủ) + rất ít fault thật (few-shot, F_m)
[Bước 2] Huấn luyện generator G:
           - optional: pretrain trên normal trước
           - fine-tune có điều kiện bằng F_m (cho từng loại lỗi)
[Bước 3] Sinh F_gen = G(F_m)  → hàng trăm/thousands mẫu lỗi giả
[Bước 4] Train classifier giám sát trên: normal + F_m (thật) + F_gen (giả)
[Bước 5] Đánh giá CHỈ trên fault THẬT của tập test (đảm bảo không dùng mẫu giả để scoring)
```

Nguyên tắc vàng:
- **Không bao giờ đánh giá bằng mẫu sinh** — test set phải là fault thật do giám khảo giữ. Dùng synthetic chỉ giúp mô hình học "khái niệm lỗi" rộng hơn.
- Data leakage cấm: nếu generator học luôn cả window test → mẫu sinh nhiễu nhẹ vẫn bị classifier "nhớ mặt" → đánh giá ảo. Luôn split theo **thời điểm máy** (trước khi train generator).
- Trọng số: có thể đặt mẫu giả đóng góp ít hơn mẫu thật (sample weighting `w_fake = 0.3·w_real`) — chống overfit vào "hơi hướng" sinh.

## 7.2. Vì sao sinh trên class hiếm khó? Few-shot generation

1. **Lượng thông tin cực nhỏ**: generator học "dấu hiệu lỗi" từ 5–10 mẫu → dễ bắt nhiễu của từng mẫu cụ thể thay vì "bản chất lỗi"; nếu 3 mẫu nhiễm nhiễu A, model sinh ra toàn "lỗi-có-nhiễu-A".
2. **Đa dạng kém**: vài mẫu không mô tả hết biến thể của lỗi (các mức độ mòn, vị trí ổ trục, tải trọng...). Generator có thể sinh ra các mẫu gần-giống-nhau (mode collapse) → classifier học "bàn sao" → không tổng quát cho lỗi mới.
3. **Class-conditional sinh yếu**: generator chưa có đủ "lực" để phân biệt "lỗi loại A vs B" mà chỉ vài mẫu.
4. **Đánh giá khó khăn**: FID không dùng được với vài chục mẫu; discriminative score kém tin (bài phân biệt quá dễ).

**Các kỹ thuật few-shot generation**:
- **Pretrain + fine-tune**: generator học phân phối normal (dữ liệu dồi dào) trước; sau đó fine-tune bằng vài mẫu lỗi (chỉ vài epoch, lr nhỏ) → mô hình "biết dáng dữ liệu" không bắt nhiễu mẫu lẻ. **→ Áp dụng trực tiếp cho TimeGAN: train trên normal rồi fine-tune bằng F_m.**
- **Data augmentation tín hiệu**: time-shift, scale biên độ, cộng noise yếu, jitter thời gian, warping (dilate/compress), tạo dạng biến thể — tăng "bề dày" class lỗi trước khi sinh.
- **Sinh theo phễu**: từ mẫu lỗi thật làm "seed", sinh quanh vùng latent (sampling gần z thật) → các mẫu lỗi "thuộc same family".
- **VAE/latent interpolation**: chèn giữa 2 mẫu lỗi trong latent → nhân đôi đa dạng có kiểm soát.

## 7.3. FaultDiffusion (arXiv 2511.15174, 11/2025) — hướng few-shot fault TS generation

**Lưu ý quan trọng**: đây là paper rất mới (tháng 11/2025). Nếu bạn chưa đọc trực tiếp, hãy lấy phần mô tả sau làm **khung hiểu chung** và luôn đối chiếu bản gốc để lấy chi tiết siêu chính xác khi cần trích dẫn trong báo cáo.

**Khung mô tả theo hướng chung (nhất quán tinh thần paper và đề A1)**:
- **Chủ đề**: diffusion model sinh dữ liệu **fault time series trong giám sát/điều khiển công nghiệp** — đúng việc ta cần cho A1 (rung cảm biến công nghiệp, thiếu dữ liệu lỗi).
- **Few-shot**: tận dụng sức ít mẫu lỗi thật (few-shot generation) — khớp A1 với dữ liệu lỗi cực hiếm. Cách làm thường thấy ở kỹ thuật này: học phân phối từ lượng normal lớn làm nền, rồi **condition theo loại lỗi** và fine-tune bằng ít mẫu fault.
- **Dấu hiệu khớp A1**:
  - Đầu vào là multivariate time series (nhiều kênh cảm biến: rung x/y/z, nhiệt độ...).
  - Khả năng sinh **mẫu lỗi đa dạng theo lớp** (class-conditional) để augment lớp hiếm rồi train classifier.
  - Điểm mạnh diffusion: ít mode collapse hơn GAN — quan trọng khi dữ liệu hiếm vì ta không muốn sinh ra "bản photo" trùng lặp che lấp classifier.

**Khuyến nghị thực tế khi áp dụng**: bắt đầu từ bản code chuẩn của DDPM/Diffusion-TS, đổi input thành window rung chuẩn hóa; fine-tune có điều kiện bằng vài mẫu lỗi thật; đánh giá bằng discriminative score trước khi đưa vào tuổi đời. Nếu ngại chi tiết toán, có thể dùng **kiến trúc Diffusion-TS** làm nền tảng — cộng đồng có code PyTorch dễ dùng (đúng mục đích giảng dạy, không cần train từ đầu).

## 7.4. Đánh giá dữ liệu sinh

Chất lượng mẫu sinh không được đo bằng mắt. Ba "công cụ đo" chuẩn nghiên cứu:

### (1) Discriminative score (đo "độ giống")
Train **post-hoc classifier** phân biệt real vs fake (nhãn 1 = thật, 0 = giả; kiểm chứng bằng cross-validation). AUC càng gần 0.5 → không phân biệt được → dữ liệu sinh giống thật. AUC ≈ 1 → yếu, sinh lộ "phải gọi".

```
Train: classifier nhỏ (LSTM 1 lớp hoặc MLP trên feature) trên real+synthetic
Metric: AUC(real, fake);  mục tiêu AUC ≈ 0.5
```

### (2) Predictive score (đo "giữ được động học")
Train **forecasting model** (vd LSTM) dự đoán bước tiếp theo **trên dữ liệu thật**; rồi **train/fine-tune riêng trên dữ liệu sinh** và so performance (MSE forecast, hoặc accuracy downstream task). Nếu generative đủ tốt, mô hình học từ synthetic cũng gần bằng học từ real:

```
T1 = test_forecast(model trained on REAL)
T2 = test_forecast(model trained on SYNTHETIC)
predictive score = (T1 - T2)  → càng gần 0 càng tốt
```

### (3) Visualization — PCA / t-SNE / UMAP
Chiếu real + synthetic xuống 2–3 chiều, kiểm tra:
- **Chồng lấp**: synthetic nằm đúng vùng phân phối real (không tách cụm riêng, không bám vào góc xa).
- **Không phải "photocopy"**: các điểm synthetic phải rải rộng, không đập chồng lên nhau chính xác (dấu hiệu mode collapse).
- **Bảo toàn cụm class**: từng loại lỗi synthetic nên màu đúng vùng của loại lỗi real tương ứng.

**Checklist ngắn**: (1) AUC-discriminative ≥ 0.5 gần tốt ~0.6 rất tốt; (2) predictive gap nhỏ; (3) t-SNE chồng lấp + phủ rộng. Chỉ khi qua cả ba, mới đưa synthetic vào pipeline train chính.

---

# 8. Thực hành PyTorch cho A1 — pipeline tổng hợp (code giả định, đủ hiểu)

Phần này tổng hợp toàn bộ thành một luồng chuẩn. **Đây chỉ là code giả định để bạn hiểu thứ tự các khối**, chưa phải code chạy production (bỏ qua normalizer, seed, eval chi tiết...).

## 8.1. Tổng quan pipeline

```
[1] Chuẩn bị: raw rung → window (B, T, n_ch) → chuẩn hóa z-score
[2] Head 1 — AE anomaly: train AE trên normal → anomaly score mọi window
[3] Head 2 — Classifier LSTM: train có giám sát trên normal + fault(thật + sinh)
[4] (Optional) GAN/TimeGAN sinh thêm fault → augment
[5] Fusion: đầu ra = fusion(anomaly score, classifier proba) → ngưỡng
[6] Đánh giá: AUC/ROC/precision-recall trên fault THẬT của test
```

## 8.2. Code giả định từng khối

```python
import torch, torch.nn as nn
import torch.nn.functional as F

# ---------- 0) chuẩn bị window ----------
# raw: (n_samples, n_ch); window_len=128; hop=32
def make_windows(data, L=128, hop=32):
    ws = [data[i:i+L] for i in range(0, len(data)-L+1, hop)]
    return torch.tensor(np.stack(ws)).float()      # (B, L, n_ch)

# chuẩn hóa theo từng kênh (dùng stat từ normal train)
# x = (x - mu) / sigma  per channel

# ---------- 1) AE anomaly (mục 2) ----------
class AE(nn.Module):
    def __init__(self, L, n_ch, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(start_dim=1),
            nn.Linear(L*n_ch, 64), nn.ReLU(), nn.Linear(64, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, L*n_ch))
    def forward(self, x):
        B = x.size(0); z = self.enc(x)
        return self.dec(z).view(B, x.size(1), x.size(2))

ae = AE(L=128, n_ch=3)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
for e in range(30):
    for xb in normal_loader:
        opt.zero_grad(); loss = ((ae(xb)-xb)**2).mean(); loss.backward(); opt.step()

def anomaly_score(ae, xb):
    ae.eval()
    with torch.no_grad(): return ((ae(xb)-xb)**2).mean(dim=(1,2))

# ---------- 2) Optional: sinh thêm fault (TimeGAN/GAN — mục 5.4) ----------
# Giả sử có hàm generate_faults(n): return tensor (n, L, n_ch)
# synthetic = generate_faults(batch, cond="bearing")   # dùng TimeGAN/Diffusion đã train

# ---------- 3) LSTM classifier (mục 3.5) ----------
class LSTMFault(nn.Module):
    def __init__(self, n_ch=3, hidden=48, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

model = LSTMFault(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# Dataloader: normal (majority) + fault_real + fault_synthetic (weight thấp hơn)
for e in range(20):
    for xb, yb in combined_loader:       # yb ∈ {0,1}
        opt.zero_grad(); logit = model(xb)
        w_yb = torch.where(yb == 1, 5.0, 1.0)   # fault=5x, normal=1x
        # giảm trọng số mẫu SINH thêm nếu sợ overfit "hơi hướng" generator
        # w_yb = torch.where(yb == 1, torch.where(is_synth, 1.5, 5.0), 1.0)
        loss = F.binary_cross_entropy_with_logits(logit, yb, weight=w_yb)
        loss.backward(); opt.step()

# ---------- 4) Fusion + ngưỡng ----------
# score = anomaly_score(ae, x)  (chuẩn hóa min-max 0..1)
# proba = torch.sigmoid(model(x))
# final = alpha*score_norm + (1-alpha)*proba;  thresh = quantile(0.99, final[normal])
# predict = final > thresh
```

## 8.3. Lưu ý deployment tối giản

- **Cửa sổ trượt (sliding window)**: test trong thực tế là online — tính trên window mới nhất, muốn có "nhiệt độ cảnh báo" theo thời gian thì trung bình trượt score (EMA) trước khi nâng cờ.
- **Threshold & metric**: bài mất cân bằng nặng → chọn ngưỡng theo **precision-recall curve** hoặc **F1/F2** (F2 nhấn cảnh báo sớm — nhà máy sẵn sàng chịu false alarm thay vì bỏ lỡ lỗi), đừng dùng accuracy.
- **Danh sách models khả dụng**: AE/ConvAE (không giám sát), LSTM/GRU classifier (có giám sát), TimeGAN/Diffusion-TS (sinh thêm lỗi), MLP/XGBoost trên feature (baseline nhanh).
- **Nguyên tắc 80/20 cho cuộc thi**: (1) làm đúng baseline AE + LSTM trên window chuẩn hóa; (2) thêm augmentation có kiểm soát (time-swap, noise); (3) sinh thêm bằng generative; (4) fusion + calibrate ngưỡng. Đừng đuổi theo Transformer/diffusion to trước khi baseline ngon.

---

# Phụ lục nhanh — bảng thuật ngữ tiếng Việt–Anh

| Tiếng Việt | Tiếng Anh |
|---|---|
| Lan truyền xuôi / ngược | Forward / backpropagation |
| Hàm kích hoạt | Activation function |
| Chuỗi có trọng số | Weighted sum |
| Bottleneck / không gian tiềm ẩn | Bottleneck / latent space |
| Mất mát tái tạo | Reconstruction loss |
| Điểm bất thường | Anomaly score |
| Cổng quên / vào / ra | Forget / input / output gate |
| Trạng thái tế bào | Cell state |
| Quá trình lan ngược gradient | Backpropagation through time (BPTT) |
| Sụp đổ mode | Mode collapse |
| Không gian embedding | Embedding space |
| Thêm nhiễu thuận / khử nhiễu ngược | Forward noising / reverse denoising |
| Tách xu hướng–chu kỳ | Trend–seasonal decomposition |
| Điểm phân biệt / dự báo | Discriminative / predictive score |
| Few-shot tạo sinh | Few-shot generation |