## PHẦN 2 · MACHINE LEARNING CỐT LÕI & ANOMALY DETECTION — MACHINE LEARNING CỐT LÕI CHO BÀI TOÁN PREDICTIVE MAINTENANCE & ANOMALY DETECTION

> Giáo trình luyện thi — Cuộc thi **DENSO Factory Hacks 2026**
> Bài toán **A1**: "AI bảo trì dự đoán khi thiếu dữ liệu lỗi" — dữ liệu là chuỗi thời gian rung của máy móc.
> Đối tượng: sinh viên / kỹ sư AI.
> Mức độ: từ nền tảng đến ứng dụng trực tiếp.
> Bản này được viết khớp với hiện trạng code của team (`src/pipeline.py`, `src/features.py`, `scripts/01`, `scripts/02`).

---

### Cây bài toán con — bản đồ toàn Phần 2

Trước khi vào chi tiết, hãy nhìn bức tranh tổng. Bài toán A1 không phải "một việc duy nhất" mà là chuỗi bài toán con nối nhau. Biết mình đang ở nhánh nào sẽ biết mình đang giải gì:

```
MỤC TIÊU LỚN: phát hiện LỖI SỚM trên chuỗi rung, trong khi dữ liệu lỗi CỰC HIẾM
   │
   ├─(1) Tín hiệu dài hàng nghìn điểm → vài con số có nghĩa?
   │      → Mục 2. Feature engineering (window/overlap, FFT, 16 feature, chuẩn hoá)
   │
   ├─(2) Có nhãn lỗi → phân loại "bình thường hay lỗi"?
   │      → Mục 3. Học có giám sát (LogReg → Decision Tree/RF → XGBoost)
   │
   ├─(3) KHÔNG có nhãn lỗi → tách "cái gì khác bình thường"?
   │      → Mục 5. Anomaly detection (OC-SVM / Isolation Forest / Autoencoder / statistical)
   │
   ├─(4) "Thế nào là tốt" khi lớp hiếm — chọn mô hình nào, ngưỡng nào?
   │      → Mục 4. Metrics (Confusion / Precision-Recall-F1 / PR vs ROC / threshold theo chi phí)
   │
   ├─(5) Lớp lỗi quá ít → làm sao bù?
   │      → Mục 6. Imbalance + augmentation (undersampling / SMOTE / GAN / diffusion)
   │
   ├─(6) Làm sao CHỨNG MINH model thật sự có giá trị (không "chém gió")?
   │      → Mục 6.4. Protocol trước/sau augmentation + chống data leakage
   │
   ├─(7) (Mở rộng) còn bao lâu nữa thì hỏng? Không chỉ 0/1?
   │      → Mục 7. RUL dự báo ngắn
   │
   └─(8) Gom mọi nhánh vào một luồng chạy được
          → Mục 8. Pipeline thực hành A1 (sơ đồ 8.1, quyết định 8.2, checklist 8.3)
```

Đọc nhanh theo tư duy đúng:
- Kéo mũi tên từ trái sang phải: **tín hiệu → feature → model → metrics → (bù lớp) → protocol → (RUL) → pipeline**.
- Nhánh (2) và (3) là hai con đường song song phụ thuộc vào một câu hỏi duy nhất: **"ta có bao nhiêu nhãn lỗi?"**. Có nhiều → đi nhánh (2); gần như không → đi nhánh (3).
- Nhánh (6) là "bộ chia trung thực" — nó quyết định mọi con số bạn báo cáo có đáng tin hay không. Đây là phần giám khảo chấm điểm mạnh nhất ở đề A1.

**Mục lục Phần 2:**

1. Overview bài toán A1
2. Feature engineering cho tín hiệu rung
3. Học có giám sát cổ điển
4. Metrics (trọng tâm — bài toán mất cân bằng mạnh)
5. Anomaly detection / học không giám sát
6. Xử lý dữ liệu mất cân bằng + thiếu dữ liệu lỗi
7. RUL dự báo ngắn
8. Pipeline thực hành cho A1
- Phụ lục A: Bảng công thức nhanh · Phụ lục B: Từ vựng Anh–Việt

---

# 1. Overview bài toán A1

> **Mục tiêu:** nắm chính xác ta đang giải bài toán gì — đầu vào, đầu ra, thứ khó nhất — để không lạc hướng khi chọn feature và model.

## 1.1. Mô tả bài toán

Bài toán A1 của cuộc thi DENSO Factory Hacks 2026 xoay quanh **bảo trì dự đoán (predictive maintenance)**:

- **Input**: chuỗi thời gian của các cảm biến gắn trên máy móc nhà máy:
  - **Rung (vibration)** — cảm biến gia tốc, thường đo ở tần số lấy mẫu cao (kHz trở lên). Đây là tín hiệu CHÍNH.
  - **Nhiệt độ (temperature)** — nhiệt độ động cơ, ổ trục, vỏ máy.
  - **Dòng điện (current)** — dòng tiêu thụ của motor.
  - Có thể thêm: áp suất, tiếng ồn, điện áp...
- **Output**: nhãn trạng thái cho từng cửa sổ thời gian (window):
  - `0` = **bình thường (normal)**.
  - `1` = **bất thường / sắp hỏng (anomaly / fault)**.

Nói cách khác: **phát hiện bất thường** (fault detection / anomaly detection) trên dữ liệu cảm biến. Mô hình cần trả lời câu hỏi: "Tại thời điểm này, máy có hoạt động đúng không?"

## 1.2. Thách thức cốt lõi

Vấn đề nan giải nhất của bài toán này — và cũng chính là tên đề bài A1 — là:

> **Dữ liệu lỗi cực hiếm (fault data is extremely scarce).**

Cụ thể hơn:

1. **Mất cân bằng nghiêm trọng (severe class imbalance)**: Trong nhà máy thực tế, máy chạy bình thường 99%+ thời gian. Số mẫu bất thường có thể chỉ chiếm 0.1%–2% dữ liệu.
2. **Khan hiếm nhãn lỗi (label scarcity)**: Máy hiếm khi hỏng, và hỏng thì người ta cũng thường không ghi nhãn đầy đủ. Thu thập dữ liệu lỗi là đắt, nguy hiểm và mất thời gian.
3. **Lỗi đa dạng**: Lỗi có thể là mài mòn ổ trục, lệch trục (misalignment), mất cân bằng rotor, nới lỏng ốc, chạm stato — mỗi loại có "dấu vân tay" tín hiệu khác nhau. Có lỗi phát triển từ từ (mài mòn), có lỗi đột ngột (chập, vỡ).
4. **Chuỗi thời gian liên quan đến thời gian thực**: Không thể shuffle ngẫu nhiên train/test như dữ liệu bảng thông thường vì sẽ gây **data leakage** (rò rỉ dữ liệu tương lai vào quá khứ).

Hiện trạng team đang làm đúng tinh thần này: file `src/pipeline.py` mô phỏng đúng tỉ lệ "normal nhiều, fault hiếm" — với dataset IMS (run-to-failure), mỗi test-run được chia theo thời gian thành: 0→70% NORMAL, 70→90% bỏ qua (vùng mơ hồ, nhãn không rõ), 90→100% FAULT. Vùng FAULT chỉ chiếm một phần nhỏ cuối chuỗi — đó chính là "dữ liệu lỗi hiếm" của đề bài.

## 1.3. Vì sao dữ liệu rung lại quan trọng nhất?

Tham số máy móc phản ứng rất nhanh với trục trặc cơ khí:

- **Lệch trục / mất cân bằng** → khung 1× (vòng quay), 2×... thành phần tần số thay đổi.
- **Hỏng ổ trục (bearing)** → xung lực lặp lại ở tần số đặc trưng BPFO/BPFI/BSF/FTF.
- **Mài mòn** → năng lượng dao động tăng, dạng sóng từ sin thuần túy thành nhiễu phức tạp.

Tại sao phải lấy feature? Vì dữ liệu rung thô thường có tần số lấy mẫu hàng chục kHz (tức hàng chục nghìn điểm mỗi giây). Đưa trực tiếp raw signal vào mô hình truyền thống là bất khả thi — số chiều quá lớn và nhạy nhiễu. **Feature engineering** nén chuỗi dài hàng nghìn điểm thành vài chục con số có ý nghĩa vật lý, giúp các mô hình cổ điển "ăn" được.

Ví dụ minh họa tổng quan: một chuỗi rung 1 giây ở 20 kHz = 20.000 điểm. Sau khi gom cửa sổ 0,1 s, mỗi cửa sổ (2.000 điểm) được nén thành ~40 feature. Dữ liệu từ "20.000 chiều" thành "40 chiều", rất nhẹ và hiệu quả cho logistic regression / XGBoost. (Trong code hiện tại của team, mỗi window được nén thành đúng **16 feature** — bảng đầy đủ ở Mục 2.6.)

## 1.4. Luồng tư duy chung để giải A1

```
[Raw vibration time series]
        │
        ▼
[Windowing + Feature extraction]   ← miền thời gian + miền tần số + chuẩn hóa
        │
        ▼
[Model]  ─────────────┬────────────────────────────┐
        │              │                            │
[Supervised]     [Unsupervised/One-class]      [Hybrid]
LogReg / RF /    IsolationForest / OC-SVM /    train trên majority,
XGBoost          Autoencoder                   đánh ngưỡng trên normal,
        │                                         flag phần tail hiếm
        ▼
[Threshold tuning theo business cost]
        │
        ▼
[Predict 0/1 → cảnh báo → dashboard / API]

[Có dữ liệu lỗi đủ? ──yes──▶ Supervised metrics: Precision/Recall/F1]
[Có dữ liệu lỗi đủ? ──no ───▶ Unsupervised + validating bằng heuristic]
```

👉 Mục 8 sẽ đưa pipeline chi tiết hơn, khớp với luồng chạy thực tế của 3 script trong repo.

---

# 2. Feature engineering cho tín hiệu rung

> **Mục tiêu:** nén chuỗi thời gian thô (hàng nghìn điểm) thành các đại lượng vô hướng có ý nghĩa vật lý, tối đa sức phân biệt normal vs anomaly. Đây là nơi **kiếm được nhiều điểm nhất** ở A1 — một feature tốt còn hơn một model xịn trên feature dởm.

**Bài toán con mà mục này giải:** nhánh (1) của cây — "tín hiệu dài → vài con số có nghĩa". Nếu bỏ qua nhánh này, mọi model phía sau (supervised lẫn unsupervised) đều vận hành trên dữ liệu quá chiều, quá nhiễu và không mang thông tin vật lý.

## 2.1. Window và overlap cơ bản (nền tảng trước tiên)

> **Mục tiêu:** chia chuỗi dài thành các khối ngắn để mỗi khối cho ra một bộ feature + một nhãn 0/1 riêng — đây là "đơn vị dự đoán" của A1.

Trước khi tính feature, ta phải chia chuỗi thời gian dài thành **cửa sổ (window)** ngắn:

- **Cửa sổ (window)**: một khối liên tiếp các mẫu. Ví dụ chuỗi rung 60 s @ 20 kHz = 1.200.000 mẫu. Nếu lấy window = 0,1 s → 2.000 mẫu/window → 600 windows.
- **Overlap (chồng lấp)**: các cửa sổ kế nhau được phép đè lên nhau một phần.

Nếu hai cửa sổ kế nhau cách nhau `stride` mẫu:
`overlap_ratio = (window_len - stride) / window_len`

Ví dụ: window = 2.000 mẫu, stride = 200 mẫu → overlap 90%.

**Vì sao cần overlap?**

- Nhãn của cảm biến thường được gắn theo khoảng thời gian (ví dụ: "từ giây 30 đến giây 40 là lỗi"). Nếu cửa sổ nằm lọt vào khoảng bình thường nhưng lại gồm đúng ranh giới lỗi, ta sẽ "lỡ" sự kiện.
- Overlap tăng số lượng mẫu huấn luyện → sinh thêm dữ liệu từ cùng một tín hiệu (rất hữu ích khi dữ liệu ít).
- Giúp dự đoán mượt hơn theo thời gian: thay vì nhảy từng nhịp, ta có thể lấy trung bình/đa số phiếu của các cửa sổ liền kề.
- Hạn chế: overlap làm các cửa sổ **phụ thuộc lẫn nhau** → khi tách train/test phải chia theo thời gian (không shuffle), và tránh để một sự kiện lỗi xuất hiện trong cả train lẫn test.

**Cỡ window chọn sao?**

- Đủ nhỏ để "bắt kịp" sự thay đổi nhanh của tín hiệu (nếu window quá dài sẽ làm mịn nhòe cú sốc/đợt rung bất thường ngắn).
- Đủ lớn để có đủ mẫu cho ước lượng thống kê ổn định và đủ độ phân giải tần số (xem 2.4.3).
- Quy tắc ngón tay cái: window lớn hơn **một chu kỳ xoay của máy** là hợp lý (nếu 1.800 vòng/phút ≈ 30 vòng/s, một chu kỳ ≈ 0,033 s → window ≥ 0,05–0,1 s).

Ví dụ số:
- Tần số lấy mẫu `fs = 20_000 Hz`, window 2.000 mẫu = 0,1 s.
- Đủ thời gian cho Fourier phân biệt tần số bội 20 kHz/2000 = 10 Hz (khoảng hai đỉnh gần nhau tách được nếu cách ≥ 10 Hz).

Ở hiện trạng team: `src/pipeline.py` dùng `win=512`, `stride=256` (overlap 50%) trên dữ liệu IMS có `fs = 12_000 Hz`; nếu dataset quá dài, stride được tự nới để tổng số window không vượt mốc an toàn cho máy tính.

## 2.2. Feature miền thời gian (time-domain features)

> **Mục tiêu:** từ từng window, trích các đại lượng mô tả **hình dạng và mức độ** dao động trong miền thời gian — đây là lớp feature trực quan nhất, ai cũng tính được và gần như luôn có ích.

**Bài toán con:** mỗi window là N mẫu; ta cần nén chúng thành vài con số. Bối cảnh: có feature "bắt năng lượng" (std, RMS), có feature "bắt sự kiện tức thời" (peak), có feature "bắt hình dạng phân bố" (skewness, kurtosis), có feature "bắt tỉ lệ" (crest/shape/impulse factor). Dùng đồng bộ cả bốn họ để bao phủ đủ loại lỗi.

So sánh nhanh vai trò hai feature "năng lượng" hay bị nhầm: **std vs RMS** — về mặt toán học, nếu tín hiệu đã trừ mean thì `RMS == std`. Nhưng trong thực tế bộ lọc tiền xử lý trừ mean không hoàn toàn, và nhiều dataset tính RMS trên tín hiệu CÓ thành phần DC → không bằng std. **Kết luận cho A1: tính cả hai, giữ cả hai — không mất gì, thêm độ an toàn.**

Đặt chuỗi trong window có `N` mẫu: `x = [x1, x2, ..., xN]`. (Ở đây N là số mẫu trong một cửa sổ.)

### 2.2.1. Mean (trung bình)

Công thức:
`mean(x) = (1/N) * sum(xi)` với `i = 1..N`
hay `mean = (x1 + x2 + ... + xN) / N`

**Ý nghĩa vật lý**: mức DC của tín hiệu. Với gia tốc kế lý tưởng đặt trên máy rung đối xứng, mean ≈ 0. Nếu mean dạt khỏi 0 (bias) có thể là lỗi cảm biến, drift nhiệt độ, hoặc lệch trục tĩnh. Nó gần như không bắt được "mạnh yếu" của rung — nên thường được loại bỏ trước khi tính các feature khác.

### 2.2.2. Standard deviation (độ lệch chuẩn) — std

Công thức:
`std = sqrt( (1/N) * sum((xi - mean)^2) )`

**Ý nghĩa vật lý**: mức độ phân tán của tín hiệu quanh giá trị trung bình — nói lên **"mạnh hay yếu"** của dao động xoay quanh điểm cân bằng. Khi máy mòn, chi tiết bị lỏng, lệch, phản lực tăng → biên độ dao động lan rộng → std tăng. Đây là feature thô nhanh nhất để nhận biết "máy rung hơn bình thường".

### 2.2.3. RMS (Root Mean Square) — căn bậc hai trung bình bình phương

Công thức:
`RMS = sqrt( (1/N) * sum(xi^2) )`

Đây chính là tổng năng lượng trung bình của tín hiệu theo thời gian.

**Ý nghĩa vật lý**: RMS là thước đo **độ lớn hiệu dụng** của dao động — đại lượng tiêu chuẩn trong kỹ thuật rung (tiêu chuẩn ISO 10816 đánh giá độ rung máy dựa trên RMS tốc độ/vận tốc). RMS tăng đều khi đặc tính mòn/lệch trục dần tiến triển, do đó là feature tốt cho cả **fault detection** lẫn **RUL prediction**.

Lưu ý: nếu tín hiệu đã được trừ mean, `RMS == std` về mặt toán học (vì `sum(xi^2)/N - mean^2 = variance`). Nhưng về mặt kỹ thuật TA NÊN tính cả hai, vì:
1. Việc trừ mean thường xảy ra trong bộ lọc tiền xử lý — có thể không hoàn toàn.
2. Trong một số dataset, `RMS` được tính trên tín hiệu CÓ mean (bao gồm cả thành phần DC) → không bằng std.
Giữ cả hai sẽ không mất gì.

### 2.2.4. Peak (đỉnh) & Peak-to-peak

Công thức:
`peak = max(|xi|)` — biên độ tuyệt đối lớn nhất trong window.
`peak_to_peak = max(xi) - min(xi)` — khoảng dao động đỉnh-đỉnh.

**Ý nghĩa vật lý**: Bắt **sự kiện tức thời** — cú va đập, xung va chạm, khe hở cơ khí. Một ổ trục nứt vỡ sẽ sinh ra xung lực lớn dù RMS chưa tăng nhiều. Peak rất nhạy nhưng cũng dễ bị nhiễu đo (một spike nhiễu làm tăng peak). Vì vậy peak thường kết hợp với chuẩn hóa và ngưỡng hợp lý để tránh báo động giả vì nhiễu.

### 2.2.5. Skewness (độ lệch bất đối xứng)

Công thức:
`skewness = ( (1/N) * sum((xi - mean)^3) ) / std^3`

**Ý nghĩa vật lý**: độ bất đối xứng của phân bố biên độ. Tín hiệu rung đối xứng (sin + nhiễu đối xứng) → skewness ≈ 0. Khi có hiện tượng **chạm/cọ xát** một chiều (rubbing), hoặc lỗi tạo ra xung lực một phía → phân bố lệch → skewness dương hoặc âm rõ rệt. Skewness giúp phát hiện loại lỗi tạo ra biên độ bất đối xứng.

### 2.2.6. Kurtosis (độ nhọn)

Công thức (kurtosis dư, excess kurtosis):
`kurtosis = ( (1/N) * sum((xi - mean)^4) ) / std^4 - 3`

(`-3` để chuẩn hóa: phân phối chuẩn Gaussian có kurtosis = 0.)

**Ý nghĩa vật lý**: mức độ "nhọn/thừa đuôi" của phân bố. Tín hiệu rung bình thường gần Gaussian → kurtosis ≈ 0. Khi xuất hiện **xung lực lặp lại (impulses)** — dấu hiệu kinh điển của ổ trục hỏng, vỡ răng bánh răng — phân bố trở nên nhọn, đuôi dày → **kurtosis tăng vọt** (lên 5, 10, 20+). Đây là một trong những feature **nhạy nhất** với hỏng ổ trục ở giai đoạn sớm.

### 2.2.7. Crest factor (hệ số đỉnh)

Công thức:
`crest_factor = peak / RMS`

**Ý nghĩa vật lý**: tỉ lệ giữa đỉnh và mức năng lượng. Tín hiệu sin thuần túy có crest factor = √2 ≈ 1,414. Tín hiệu có xung lực sắc nhọn (giàu impulse) → crest factor cao (3, 5, 10...). Khi máy mòn dần, đôi khi RMS tăng nhanh hơn peak → crest factor **giảm** — một dạng tương đối hóa giúp phát hiện sự thay đổi bản chất tín hiệu.

### 2.2.8. Shape factor (hệ số hình dạng)

Công thức:
`shape_factor = RMS / mean_abs` với `mean_abs = (1/N) * sum(|xi|)`

**Ý nghĩa vật lý**: quan hệ giữa mức năng lượng và mức tuyệt đối trung bình — phản ánh "hình dạng" dạng sóng. Sin thuần túy có shape factor ≈ 1,11. Khi tín hiệu chứa nhiều xung (spiky) thì shape factor cao hơn. Dùng kết hợp với crest factor để phân biệt dạng sóng.

### 2.2.9. Impulse factor (hệ số xung)

Công thức:
`impulse_factor = peak / mean_abs`

**Ý nghĩa vật lý**: tương tự crest factor nhưng so với mức trung bình tuyệt đối thay vì RMS. Nhạy với xung đột ngột (impulsive faults): khi có va đập, peak tăng nhanh hơn mean_abs → impulse factor tăng. Rất hữu ích cho phát hiện lỗi dạng impact (khe hở, nứt vỡ bề mặt).

### 2.2.10. Bảng tổng kết feature thời gian + khoảng giá trị tham khảo

| Feature | Công thức | Ý nghĩa | Bắt loại lỗi gì |
|---|---|---|---|
| mean | `(1/N)*sum(xi)` | mức DC, drift | lỗi cảm biến, lệch tĩnh |
| std | `sqrt((1/N)*sum((xi-mean)^2))` | mức năng lượng dao động | mòn tăng dần (mọi loại) |
| RMS | `sqrt((1/N)*sum(xi^2))` | năng lượng tích phân | tiêu chuẩn ISO, mòn tiến triển |
| peak | `max(|xi|)` | biên độ đỉnh | xung, va đập |
| peak-to-peak | `max(xi)-min(xi)` | khoảng dao động | xung lớn, clearance |
| skewness | `((1/N)*sum((xi-mean)^3))/std^3` | bất đối xứng | chạm một chiều (rubbing) |
| kurtosis | `((1/N)*sum((xi-mean)^4))/std^4 - 3` | độ nhọn/đuôi dày | **ổ trục hỏng, xung lặp lại** |
| crest factor | `peak/RMS` | tỉ đỉnh/năng lượng | lỗi dạng impulse |
| shape factor | `RMS/mean_abs` | hình dạng dạng sóng | thay đổi dạng sóng |
| impulse factor | `peak/mean_abs` | tỉ đỉnh/trung bình tuyệt đối | lỗi impact |

**Ví dụ số minh họa kurtosis:**

- Tín hiệu sin thuần: `x = [1, 0, -1, 0, ...]` → phân bố cong hình chữ M, kurtosis sau trừ 3 âm nhẹ (≈ -1,5).
- Tín hiệu nhiễu Gaussian lý tưởng → kurtosis = 0.
- Tín hiệu bình thường + 3 xung lực cực lớn → phân bố có đuôi dài → kurtosis 8–20.

Kinh nghiệm: theo dõi đồng thời RMS (năng lượng) + kurtosis (độ nhọn). Nếu cả hai cùng tăng → lỗi cơ khí tiến triển. Nếu RMS ổn định mà kurtosis tăng → xuất hiện xung rời rạc, nghi ngờ lỗi loại impact.

## 2.3. Giới thiệu biến đổi Fourier và feature miền tần số

> **Mục tiêu:** thêm lớp feature miền tần số để bắt **"năng lượng tập trung ở đâu"** — thứ mà feature thời gian gần như mù — nhất là với lỗi ổ trục (BPFO/BPFI) và mài mòn lan tỏa.

**Bài toán con:** nhiều lỗi cơ khí mang "dấu vân tay tần số" rất riêng (đỉnh ở 1×, 2×, BPFO...). Feature thời gian chỉ thấy biên độ tăng; feature tần số thấy NHỮNG TẦN SỐ NÀO tăng → phân biệt được loại lỗi.

### 2.3.1. Vì sao miền tần số?

Tín hiệu rung thường là "chồng chập" của nhiều thành phần dao động ở các tần số khác nhau:
- Tần số quay cơ bản `f1` (1×), hài bậc 2×f1, 3×f1...
- Tần số đặc trưng của ổ trục (BPFO, BPFI...) — giá trị phụ thuộc số viên bi, đường kính.
- Nhiễu nền trắng.

Miền tần số cho ta nhìn thấy **"năng lượng tập trung ở đâu"** — giúp phân biệt các loại lỗi cực rõ. Ví dụ:

- **Mất cân bằng rotor (unbalance)**: đỉnh lớn ở đúng 1×.
- **Lệch trục (misalignment)**: đỉnh ở 1×, 2× thậm chí 3×.
- **Ổ trục hỏng**: các đỉnh nhỏ ở BPFO/BPFI cộng với "mặt nâng" nhiễu tần số cao.

### 2.3.2. FFT trong một đoạn ngắn

FFT biến đổi N mẫu miền thời gian thành N (phức) hệ số tần số. Cách dùng gọn:

```
X[k] = FFT(x)[k]                        # k = 0..N-1
Amp[k] = |X[k]|                          # biên độ (magnitude) — dùng trong code src/features.py
Power[k] = |X[k]|^2                       # công suất (power spectrum)
freq[k] = k * fs / N                      # tần số tương ứng (Hz)
```

- `k = 0` là tần số DC.
- `k = N/2` tương ứng tần số Nyquist (xem 2.4).
- Chỉ nửa phổ đầu (từ k=0 đến k=N/2) chứa thông tin mới; nửa sau đối xứng với nửa đầu (với tín hiệu thực). Code team dùng `rfft` (chỉ lấy nửa phổ thực) — đúng tinh thần này.

Lưu ý nhanh: không dùng feature "toàn bộ N hệ số FFT" làm input mô hình — quá chiều, quá nhạy nhiễu, và không dịch bất biến theo pha. Thay vào đó ta nén phổ thành các **feature vô hướng** dưới đây.

### 2.3.3. Spectral centroid (tâm phổ)

Công thức:
`centroid = sum(freq[k] * Power[k]) / sum(Power[k])` với `k` chạy trên miền tần số quan tâm.

**Ý nghĩa vật lý**: "trọng tâm" của phổ — tần số trung bình có trọng số theo năng lượng. Centroid nhích lên cao khi tín hiệu chứa nhiều năng lượng ở tần số cao (nghi ngờ mài mòn, hỏng viên bi, phát ra tiếng rít tần số cao). Đây là feature "định hướng" nhanh: bình thường centroid thấp, khi lỗi mài mòn → nhích cao.

### 2.3.4. Spectral spread (độ trải phổ)

Công thức:
`spread = sqrt( sum( Power[k] * (freq[k] - centroid)^2 ) / sum(Power[k]) )`

**Ý nghĩa vật lý**: mức độ "loang rộng" của năng lượng quanh centroid — tương tự std nhưng trên miền tần số. Tín hiệu hẹp băng (nhiều hài rời rạc rõ) có spread nhỏ; tín hiệu nhiễu broadband (mài mòn lan tỏa, kêu rè) có spread lớn. Kết hợp: centroid nói "ở đâu", spread nói "rộng hay hẹp".

### 2.3.5. Spectral flatness (độ phẳng phổ)

Công thức:
`flatness = exp( (1/K) * sum( ln(Power[k] + epsilon) ) ) / ( (1/K) * sum(Power[k]) + epsilon )`

với `+epsilon` tránh log(0) khi Power = 0 (epsilon thường 1e-12 trong code).

**Ý nghĩa vật lý**: tỉ số **trung bình hình học / trung bình số học** của phổ, nằm trong (0, 1]:

- flatness → 1: phổ "phẳng" như nhiễu trắng (không có đỉnh nổi bật) → âm thanh ồn rè, mang tính ngẫu nhiên.
- flatness → 0: phổ "nhọn" với vài đỉnh sắc (tonal, harmonic) → máy chạy gọn rõ ràng.

Khi lỗi kiểu ồn/mài mòn lan rộng, flatness **tăng**. Khi xuất hiện các hài mới từ lỗi, có thể giảm. Đây là feature hữu ích phân biệt lỗi rời rạc vs nhiễu lan tỏa.

### 2.3.6. Band energy (năng lượng theo dải tần)

Chia phổ thành các dải tần và tính năng lượng từng dải:

```
E[band b] = sum( Power[k] ) với k thuộc dải b
```

Ví dụ dải (Hz): [0, 50), [50, 100), [100, 200), [200, 500), [500, 1000), [1000, 2000), [2000, 5000), [5000, 10000].

**Ý nghĩa vật lý và cách dùng**:
- Tần số quay f1 < 100 Hz thường; năng lượng khổng lồ ở dải thấp = cân bằng/lệch.
- Năng lượng ở vùng BPFO–BPFI (thường hàng trăm Hz – vài kHz) tăng = ổ trục hỏng.
- Năng lượng ở cực cao (mid–high kHz) tăng = mài mòn, khô dầu, chạm.
Đây có lẽ là nhóm feature dễ "ăn điểm" nhất vì tương đối trực quan với thợ máy — có thể tỉ lệ hóa: `E_rel[b] = E[band b] / total_energy`.

So sánh trong code hiện tại: `src/features.py` chọn kiểu "tổng hợp gọn" hơn thay vì 8 dải tần — nó lấy `spec_energy` (tổng năng lượng phổ) + `spec_centroid` + `spec_spread` + `spec_flatness` + `freq_mean` + `freq_std`. Nếu bạn mở rộng, thêm band energy 8 dải (theo bảng trên) là hướng rõ ràng nhất để tăng điểm.

### 2.3.7. Window function (hàm cửa sổ) — khái niệm bắt buộc

> **Mục tiêu:** trước FFT, nhân tín hiệu với hàm cửa sổ để giảm "rò rỉ phổ" — nếu bỏ qua, đỉnh phổ bị loang sang tần số kế cận và feature miền tần số bị nhiễu.

Trước FFT nên nhân tín hiệu với **window function**:

- `hann[k] = 0.5 * (1 - cos(2*pi*k/(N-1)))`
- `hamming[k] = 0.54 - 0.46*cos(2*pi*k/(N-1))`
- Rectangular (không nhân gì) — đơn giản nhất nhưng gây rò rỉ phổ (spectral leakage): một đỉnh sắc sẽ "loang" sang tần số kế cận.

Vì sao? Vì chuỗi ngắn N mẫu cắt đột ngột tạo ra "sườn" gián đoạn ở hai đầu → xuất hiện các tần số giả. Window làm mềm hai đầu → giảm leakage, đỉnh phổ sắc hơn. **Tổng năng lượng sẽ đổi** do window — nếu dùng window, hãy dùng đồng nhất cho mọi window và mọi file để so sánh công bằng. (Code team dùng `np.hanning(win)` nhất quán cho mọi window — đúng.)

## 2.4. Tần số lấy mẫu, Nyquist và aliasing (nền tảng — đọc kỹ)

> **Mục tiêu:** biết giới hạn tin cậy của phổ (0 → fs/2) và tránh cái bẫy "đỉnh ảo" do aliasing — một sai lầm nhỏ ở đây làm mọi feature tần số vô nghĩa.

**Bài toán con:** kết quả FFT chỉ đúng trong một dải tần giới hạn. Làm feature miền tần số mà không biết dải này thì có thể "phát hiện lỗi" ở tần số không tồn tại.

### 2.4.1. Tần số Nyquist

> **Định lý lấy mẫu (Shannon–Nyquist)**: để tái dựng chính xác một tín hiệu liên tục, tần số lấy mẫu phải **lớn hơn hai lần** tần số cao nhất chứa trong tín hiệu.

Công thức:
`f_Nyquist = fs / 2`

Ví dụ: `fs = 20 kHz` → `f_Nyquist = 10 kHz`. Nghĩa là: ta chỉ tin tưởng FFT ở vùng 0–10 kHz. Mọi thành phần năng lượng được FFT "báo" ở dải trên 10 kHz là ẢO (do aliasing, xem 2.4.2), trừ khi phần cứng đã có anti-aliasing filter.

### 2.4.2. Aliasing (biệt danh tần số)

Khi tín hiệu thực chứa tần số > f_Nyquist nhưng vẫn bị lấy mẫu với fs, các tần số cao này sẽ "ngụy trang" thành tần số thấp hơn trong dải hợp lệ:

`f_gia = | f_thuc - n * fs |` với n là số nguyên làm kết quả nằm trong [0, fs/2].

Ví dụ số: `fs = 1000 Hz` → Nyquist 500 Hz. Một thành phần thực ở **700 Hz** sẽ được FFT hiển thị ở:
`|700 - 1000| = 300 Hz`. Nhà phân tích tưởng có dao động 300 Hz — sai bản chất, không thể phát hiện lỗi vì đỉnh "bị dời chỗ".

**Hệ quả thực hành**:
- Luôn đọc metadata để biết fs chính xác của từng file.
- Khi tính FFT: `freq_max_hop_tin = fs/2`. Feature band energy chỉ tính trong [0, fs/2].
- Nếu AI trước xử lý giảm lấy mẫu (downsample) 2×, nhớ lọc thông thấp trước khi downsample (vì sau khi giảm fs, tần số cao cũ sẽ gây aliasing).
- Với chuỗi khác fs giữa các file: đưa feature band energy theo **tỷ lệ f/f1** (hoặc chuẩn hóa theo harmonic) để so sánh được.

### 2.4.3. Độ phân giải tần số

Với window N mẫu @ fs, bước tần số giữa hai bin liên tiếp:
`Delta_f = fs / N`

Ví dụ: `fs=20_000`, `N=2_000` → Delta_f = 10 Hz. Hai đỉnh cách nhau dưới 10 Hz sẽ bị gộp làm một. Muốn phân giải tốt hơn: tăng N (lấy window dài hơn).

> **Quan hệ bất định trong thực hành**: window dài → phân giải tần số tốt, nhưng nhòe theo thời gian; window ngắn → bắt nhanh sự kiện nhưng phổ thô. Đây là lý do overlap được dùng: giữ window đủ dài cho FFT mượt mà vẫn có nhiều decision point theo thời gian.

## 2.5. Chuẩn hóa (normalization / scaling)

> **Mục tiêu:** đưa các feature khác đơn vị về cùng thang đo để mô hình (đặc biệt là mô hình gradient/khoảng cách) không bị chiều có biên độ lớn "áp đảo".

**Bài toán con:** các feature có "đơn vị" rất khác nhau (RMS ~ 1e-2 g, kurtosis ~ 10, band energy ~ 1e4). Mô hình như Logistic Regression (gradient) hoặc khoảng cách (kNN, SVM) **sẽ bị méo** nếu không chuẩn hóa — chiều có biên độ lớn "áp đảo" chiều nhỏ.

So sánh ba cách chuẩn hóa — và vì sao chọn RobustScaler cho rung:

| Cách | Phép | Chống outlier | Khuyến nghị A1 |
|---|---|---|---|
| Z-score | `(x-mu)/sigma` | Trung bình | nếu dùng chuẩn hóa trước RF/XGB ok |
| Min–max | `(x-min)/(max-min)` | Kém | ít dùng với rung |
| **RobustScaler** | `(x-median)/IQR` | Tốt | ⭐ mạnh nhất cho rung |

**Kết luận cho A1:** tín hiệu rung có đặc điểm về bản chất là chứa **spike/outlier** (xung va đập, cú sốc). Min–max bị một outlier kéo toàn dải → mọi giá trị bình thường dồn cục vào một chỗ → mất hết sức phân biệt. RobustScaler dùng median + IQR nên "không bị kéo" — giữ được hình dạng cụm lẫn làm lộ outlier.

Lưu ý quan trọng theo hiện trạng team: pipeline dùng **z-score** (`fit_zscore` + `zscore` trong `src/features.py`) thay vì RobustScaler — vì z-score đủ tốt sau khi đã có feature 16 chiều và chuẩn hóa bằng mean/std fit TRÊN TRAIN. Nếu bạn muốn nâng cấp, thay bằng RobustScaler là điều chỉnh nhỏ và hoàn toàn hợp lý cho rung.

### 2.5.1. Z-score (standardization)

Công thức:
`z = (x - mu) / sigma`

- `mu`, `sigma` khai báo train; áp dụng cho test với CÙNG `mu`, `sigma` đó (không tính lại trên test — nếu không sẽ giới thiệu leakage thông tin test vào quy trình).
- Kết quả: trung bình 0, std 1.
- **Khi dùng**: mô hình giả định phân bố Gauss (Logistic, SVM); khi feature có thể có độ lệch ít nghiêm trọng; khi không rõ nên dùng gì → chọn mặc định.

### 2.5.2. Min–max scaling

Công thức:
`x_scaled = (x - x_min) / (x_max - x_min)` → nằm trong [0, 1].

- Nhạy cảm với outlier: một outlier kéo toàn dải → mọi giá trị bình thường dồn cục vào 0,1–0,3. Kém nếu dữ liệu có giá trị cực (thường gặp ở rung — có spike).
- **Khi dùng**: dữ liệu đã biết có biên độ ràng buộc (nhiệt độ 20–80°C); khi bắt buộc feature nằm trong [0,1] (neural có activation bão hòa như sigmoid).

### 2.5.3. RobustScaler

Công thức:
`x_scaled = (x - median) / IQR` với `IQR = Q3 - Q1` (tứ phân vị), hoặc `(x - median)/1.349*MAD`.

- Trung vị & IQR **chống nhiễu** — không bị kéo bởi vài outlier vài nghìn lần.
- **Khi dùng**: dữ liệu thống kê có nhiều outlier/xung — **điển hình của tín hiệu rung**. Rất khuyên dùng cho A1.

### 2.5.4. Ví dụ số minh họa

Dãy RMS của 5 window: `[0.5, 0.6, 0.55, 8.0, 0.7]` (có outlier 8.0 do một cú sốc thật hoặc nhiễu).

- min–max: min=0.5, max=8.0 → 0.55 → `(0.55-0.5)/7.5 = 0.0067` → bị "ép" gần 0, các giá trị bình thường không phân biệt được.
- z-score: mu ≈ 2.07, sigma ≈ 3.24 → 0.55 → `(0.55-2.07)/3.24 ≈ -0.47`.
- RobustScaler: median = 0.6, Q1=0.55, Q3=0.7, IQR=0.15 → 0.55 → `(0.55-0.6)/0.15 ≈ -0.33`; chiều 8.0 → `(8-0.6)/0.15 ≈ 49` rất cao → dễ phát hiện ngay.

RobustScaler giữ được cả "hình dạng cụm" lẫn làm lộ outlier.

## 2.6. Gợi ý danh sách feature tổng hợp cho A1 (mức mở rộng)

Bảng 16 feature mà `src/features.py` đang tính cho mỗi window (RAW, chưa chuẩn hoá):

```
-- Miền thời gian (10)
mean, std, rms, peak, peak2peak, skewness, kurtosis,
crest_factor, shape_factor, impulse_factor

-- Miền tần số (6) — FFT (rfft) với window function hann
spec_centroid, spec_spread, spec_flatness, spec_energy,
freq_mean, freq_std
```

Nếu mở rộng để ăn thêm điểm, bổ sung nhóm gợi ý sau:

```
-- Miền thời gian mở rộng
zero_crossing_rate  (số lần đổi dấu — đo "tần suất" nhanh)

-- Miền tần số mở rộng (trước FFT với window function hann)
spectral_rolloff,
band_energy[8 dải] (chuẩn hóa tổng = 1),
harmonic_ratio = E( tại f1, 2f1, 3f1 ) / total_energy

-- Chỉ số thống kê phụ
diff_features: std của sai phân bậc nhất (đo tốc độ thay đổi);
range_ratio = peak_to_peak / rms
```

Mẹo mạnh: **tính feature theo cả chuỗi raw lẫn chuỗi đã lọc bandpass** (dải quanh tần số quay) gấp được khả năng phân tách lỗi. Chọn feature sẵn tốt hơn là dùng toàn bộ và để model tự gánh.

---

# 3. Học có giám sát cổ điển

> **Mục tiêu của cả mục:** trả lời nhánh (2) của cây bài toán con — khi ta ĐÃ CÓ nhãn lỗi (một phần), dựng mô hình phân loại 0/1 đạt điểm cao nhất. Giả định: ta đã có **nhãn** (0/1) cho từng window — hoặc từ dữ liệu thi (kèm ít dữ liệu lỗi), hoặc từ dữ liệu bổ sung sau khi khai thác.

**Bài toán con:** bài toán phân loại nhị phân trên dữ liệu bảng feature đã nén. Bối cảnh đặc thù A1: ít mẫu lỗi, dữ liệu nhiễu, cần diễn giải được. Vì vậy thứ tự đi là: mô hình đơn giản và diễn giải được (LogReg) → mô hình mạnh dạng cây (RF) → mô hình thường thắng giải (XGBoost).

## 3.1. Logistic Regression (hồi quy logistic)

> **Mục tiêu:** có một baseline NHANH, diễn giải được, để đối chiếu mọi model khác — đồng thời "đọc" được feature nào ảnh hưởng lỗi mạnh qua trọng số w.

### 3.1.1. Ý tưởng

Dù tên là "regression", đây là thuật toán **phân loại nhị phân**. Ý tưởng: dự đoán **xác suất** thuộc lớp 1 từ tổ hợp tuyến tính của các feature, rồi "bóp" vào khoảng (0,1) bằng hàm sigmoid.

### 3.1.2. Công thức

Linear score:
`z = w0 + w1*x1 + w2*x2 + ... + wp*xp`

Sigmoid:
`p = 1 / (1 + exp(-z))`

Hàm sigmoid có tính chất:
- `z = 0` → `p = 0.5`
- `z → +inf` → `p → 1`
- `z → -inf` → `p → 0`

Quyết định (decision boundary):
`prediction = 1 nếu p >= threshold, ngược lại 0`, threshold mặc định = 0.5.

Decision boundary trong không gian feature: nghiệm của `z = 0`, tức `w0 + w1*x1 + ... = 0` — là một **siêu phẳng** trong không gian feature. Vì là siêu phẳng tuyến tính, Logistic **không bắt được quan hệ phi tuyến** tốt — nhưng với thư viện scikit-learn và data chuẩn hóa tốt, nó vẫn là mô hình "benchmark nhanh + diễn giải được" — bạn có thể đọc trọng số w để biết feature nào ảnh hưởng mạnh (hệ số dương → tăng xác suất lỗi).

### 3.1.3. Loss function (cross-entropy)

Với N mẫu, nhãn `y_i ∈ {0,1}`:

`Loss = -(1/N) * sum( y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) )`

Giải thích trực quan:
- Nếu `y_i=1` nhưng mô hình đoán `p_i` nhỏ → `ln(p_i)` rất âm → loss lớn → bị phạt nặng.
- Nếu `y_i=0` nhưng mô hình đoán `p_i` cao → `ln(1-p_i)` rất âm → loss lớn.
Nhờ đó mô hình học để đưa p gần nhãn thật.

Logistic tối ưu hóa bằng gradient descent (thường là phiên bản L-BFGS trong library). Nên thêm **regularization** (xem 3.4).

### 3.1.4. Ví dụ số cực ngắn

Có 1 feature: RMS. Học được `z = -3.5 + 40*RMS` (sau chuẩn hóa). Với window có RMS=0.08 (chuẩn hóa → ~0 ):
`z = -3.5 + 40*0 ≈ -3.5` → `p = 1/(1+e^3.5) ≈ 0.03` → dự đoán 0 (bình thường).
Với window có RMS lớn (chuẩn hóa ≈ 0.15):
`z = -3.5 + 40*0.15 = 2.5` → `p = 1/(1+e^-2.5) ≈ 0.92` → dự đoán 1 (bất thường).

## 3.2. Decision Tree → Random Forest

> **Mục tiêu:** đi từ một cây đơn (dễ hiểu nhưng dễ overfit) đến Random Forest — model dạng cây đầu tiên thực sự mạnh với dữ liệu tabular ít mẫu, kèm "validation miễn phí" bằng OOB.

### 3.2.1. Decision Tree: ý tưởng

Cây quyết định chia không gian feature theo các **luật if–else**:
"Nếu `kurtosis > 6` và `RMS > 0.05` → lỗi". Mỗi nút chọn một feature + ngưỡng phân tách tốt nhất sao cho sau khi tách, các nhánh con **thuần nhất hơn** (ít lẫn lớp).

### 3.2.2. Độ đo chia (split criterion)

Gini impurity của một tập hợp:
`Gini = 1 - sum_c (p_c)^2`, với `p_c` là tỉ lệ lớp c.
- Tập thuần 1 lớp → Gini = 0.
- Hai lớp đều 50/50 → Gini = 0.5.

Thuật toán thử mọi (feature, ngưỡng), chọn phân tách làm **giảm Gini nhiều nhất**:
`gain = Gini(parent) - (n_l/n) * Gini(left) - (n_r/n) * Gini(right)`

Một tiêu chí khác: entropy `H = -sum_c p_c ln(p_c)`, gain theo information gain.

### 3.2.3. Rủi ro của một cây đơn: overfitting

Cây đơn sâu đến mức mỗi lá thuần 100% thường **quá khớp**: nhớ nhiễu của train. Kết quả: trên test có thể tệ hơn Logistic. Vì vậy người ta dùng rừng cây — **ensemble**.

### 3.2.4. Random Forest: Bagging + random feature

- **Bagging** (Bootstrap aggregating):
1. Từ N mẫu train, lấy mẫu **có hoàn lại** (bootstrap) N mẫu → tập con cho mỗi cây (mỗi cây nhìn một phiên bản khác của train, vài cây thiếu mẫu này, vài cây thiếu mẫu kia).
2. Huấn luyện M cây; mỗi cây khi tách chỉ được xét **ngẫu nhiên subset các feature** (thường sqrt(p)).
3. Dự đoán = **đa số phiếu** (bình chọn) giữa các cây.

Vì sao hiệu quả?:
- Mỗi cây thiên về (variance cao) nhưng thiên lệch theo các hướng nhiễu khác nhau → khi lấy trung bình, **sai số phương sai giảm** mà bias gần như không đổi.
- Random feature làm cây giảm tương quan nhau hơn (không cùng nhiễu), tăng lợi ích của ensemble.

### 3.2.5. Out-of-bag (OOB) score

Vì bootstrap lấy mẫu có hoàn lại, với N mẫu, xác suất một mẫu không được chọn trong một cây ≈ `(1-1/N)^N ≈ e^-1 ≈ 0.368`. Các mẫu "bỏ lỡ" (out-of-bag) của một cây KHÔNG tham gia train cây đó → có thể dùng để đánh giá ngay trong lúc train mà **không cần tách thêm validation set**:

- Với mỗi mẫu, dự đoán bằng chỉ các cây mà mẫu đó thuộc OOB.
- Gộp tất cả → OOB score ≈ ước lượng generalization không chệch.

OOB rất tiện trong bài toán thiếu dữ liệu: tận dụng toàn bộ dữ liệu train mà vẫn có hồi kiểm (cross-validation style) miễn phí. Lưu ý: vẫn phải tách **test theo thời gian** riêng để đánh giá cuối cùng (tránh leak về chuỗi thời gian — xem 5.6).

## 3.3. Gradient Boosting / XGBoost

> **Mục tiêu:** model mạnh nhất họ cây cho dữ liệu tabular đã feature-engineer — có regularization chống overfit, bất biến scale, và là điểm khởi đầu thắng giải cho A1.

### 3.3.1. Ý tưởng cốt lõi: "sửa sai từng bước"

Khác với Random Forest (xây M cây song song rồi lấy trung bình), **boosting** xây cây **tuần tự**: cây tiếp theo được huấn luyện để **bù đắp sai lệch mà các cây trước để lại**.

Đặc biệt gradient boosting:
```
Bắt đầu: mô hình khởi tạo F0(x) = mean(y)  (hồi quy) hoặc log(odds) (phân loại)
Vòng lặp m = 1..M:
 1. Tính "phần dư" (residual) r_i = y_i - F_{m-1}(x_i)   (hồi quy)
    Hoặc: tính gradient của loss theo dự đoán hiện tại  (tổng quát)
 2. Fit một cây nhỏ h_m để dự đoán r_i từ x_i
 3. Cập nhật: F_m(x) = F_{m-1}(x) + lr * h_m(x)
Kết quả: F_M(x) = F_0(x) + lr*h_1(x) + lr*h_2(x) + ... + lr*h_M(x)
```

Trực giác: cây 1 bắt xu hướng lớn; sai chỗ nào, cây 2 nhắm đúng chỗ còn sai; tiếp tục... Mỗi cây "thợ sửa" bé (thường nông, max_depth 3–6), nhiều cây cộng lại thành mô hình mạnh và đúng hơn. **learning rate (lr)** kiểm soát mức độ mỗi cây được hiệu chỉnh — lr bé + nhiều cây thường hội tụ tốt hơn lr lớn.

### 3.3.2. Loss function đối với phân loại

Phân loại (mục tiêu xác suất) cho A1 thường dùng **binary logistic loss**:

`Loss = -y*ln(p) - (1-y)*ln(1-p)`  (với N mẫu thì trung bình loss trên N).

XGBoost còn tối ưu bằng **second-order** thông tin: dùng gradient `g = dL/dp` và Hessian `h = d^2L/dp^2`. Ý tưởng: không chỉ biết "lỗi bao nhiêu" (gradient) mà còn biết "lỗi" biến thiên thế nào (Hessian) → xác định lá cây và giá trị mỗi lá chính xác hơn so với naive gradient. Đây là một phần vì sao XGBoost nhanh+chắc chắn so với boosting thế hệ đầu (GBM).

### 3.3.3. Regularization L1 / L2

Công thức loss XGBoost tổng:
`L_total = sum_i L(y_i, p_i) + lambda * sum_l w_l^2 + alpha * sum_l |w_l|`

Hai số hạng phạt trên **trọng số lá** `w_l`:
- **L2** (`lambda * w^2`): phạt giá trị lá lớn → lá ôn hòa, giảm overfit (giống ridge).
- **L1** (`alpha * |w|`): phạt phi tuyến tính, có thể **ép số lá về 0** → kiểu sparsity.

Cùng với `max_depth`, `min_child_weight`, `subsample`, `colsample` — những tham số này giúp chống overfit mạnh mẽ, cực kỳ quan trọng vì dữ liệu A1 ít và mất cân bằng.

### 3.3.4. Vì sao XGBoost mạnh với dữ liệu tabular / feature-engineered?

1. **Xử lý phi tuyến + interaction**: tự động tìm ra luật phức hợp giữa feature (`kurtosis cao AND RMS tăng`) mà không cần ta thủ công khai báo.
2. **Bất biến với scale**: cây chỉ so sánh ngưỡng nên không cần chuẩn hóa feature (có thể dùng trực tiếp) — tiết kiệm bước tiền xử lý.
3. **Robust với outlier và dữ liệu thiếu**: ngưỡng chia dựa trên thứ hạng giá trị, không bị vài outlier kéo; tự học hướng xử lý missing theo phân phối.
4. **Nhạy tốt số mẫu**: với đúng số lượng thi, ensemble tree học được cấu trúc nhưng ít chịu overfit nếu regularization tốt.
5. **Đo importance**: `gain` và `frequency` của từng feature → diễn giải được, giúp rút gọn chọn feature.

Kinh nghiệm: XGBoost/LightGBM thường là **điểm khởi đầu thắng giải** cho bài toán tabular lỗi hiếm ở A1.

## 3.4. So sánh Random Forest vs XGBoost — khi dùng cái nào?

> **Mục tiêu:** biết khi nào chọn RF, khi nào chọn XGB — cả hai đều là ứng viên cuối cùng của A1, nhưng tính chất khác nhau.

| Tiêu chí | Random Forest | XGBoost / Gradient Boosting |
|---|---|---|
| Cách xây cây | Song song, lấy trung bình | Tuần tự, sửa lỗi |
| Bias/variance | Giảm variance | Giảm bias tuần tự |
| Khả năng fit dữ liệu nhiễu | Tốt hơn, ít overfit tự nhiên | Cần regularization cẩn thận |
| Tốc độ train | Nhanh (song song được dễ) | Chậm hơn khi lặp tuần tự (LightGBM nhanh hơn) |
| Feature importance | Có (impurity / permutation) | Có (gain, cover, frequency) |
| Hiệu chỉnh ngưỡng chính xác trên dữ liệu hiếm | Dễ bí (ít mẫu lỗi) | Hiệu quả hơn với nhiều cây nhỏ |
| Khi nào dùng | Dữ liệu nhiều nhiễu, cần nhanh, cần OOB validation | Dữ liệu tabular feature-engineered, cần maximize chính xác |

**Kết luận cho A1 — vì sao chọn cái này hơn cái kia:**
- Nếu dữ liệu lỗi thực sự chỉ vài chục mẫu: **RF an toàn hơn** — ít tham số, ít overfit mạnh, có OOB sạch.
- Nếu có cỡ vài trăm mẫu lỗi trở lên: **XGBoost gần như luôn đạt top** với GridSearch nhẹ.
- Quy trình đề xuất (đúng tinh thần "đi từ đơn giản"):
1. Bắt đầu bằng **Logistic** (chạy nhanh, đánh giá feature, baseline metrics).
2. Rồi **Random Forest** với OOB để có ước lượng sạch.
3. Rồi **XGBoost** (với GridSearch nhẹ trên `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `scale_pos_weight`) — gần như luôn đạt top cho A1.
- Test cả hai và chọn theo **threshold-independent metric** (PR-AUC) trên test theo thời gian.

Đúng với hiện trạng: `scripts/02_compare_augmentation.py` dùng **RandomForest** làm model chấm trước/sau augmentation — hợp lý vì đây là mốc nhanh, ổn định với ít lỗi thật; nâng lên XGBoost là bước tối ưu tiếp theo.

### 3.4.1. Xử lý imbalance trong XGBoost

`scale_pos_weight = n_negative / n_positive`. Ví dụ 100.000 normal, 500 lỗi → scale ≈ 200. Điều này nhân trọng số loss cho lớp thiểu số — hướng mô hình học kỹ lớp lỗi.

---

# 4. Metrics cho bài toán mất cân bằng (trọng tâm)

> **Mục tiêu của cả mục:** trả lời nhánh (4) — "thế nào là tốt" khi lớp hiếm. Định nghĩa sai "tốt" = chọn sai model = thất bại dù model giỏi. Metrics ở đây ảnh hưởng TRỰC TIẾP tới cách bạn báo cáo cho giám khảo.

> ⚠️ Trong A1, chọn sai metric = chọn sai mô hình. Accuracy là "cạm bẫy" lớn nhất. Mục này phải nắm gốc.

## 4.1. Accuracy — vì sao gây hiểu lầm khi class lệch?

**Mục tiêu:** biết vì sao accuracy KHÔNG dùng được khi lỗi hiếm.

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Với dữ liệu 99% bình thường, một mô hình "ngu" luôn đoán 0 sẽ đạt **99% accuracy**.

Ví dụ số: dữ liệu có 10.000 normal, 100 lỗi. Model luôn dự đoán 0:
- TP=0, TN=10.000, FP=0, FN=100.
- Accuracy = 10000/10100 = **99,0%** — nghe rất hay nhưng chẳng bắt được **một** lỗi nào! Trong bảo trì dự đoán, đó là thảm họa: máy vỡ mà không báo.

Vì vậy phải dùng metrics tập trung vào lớp quan tâm: lỗi (positive).

## 4.2. Confusion matrix (ma trận nhầm lẫn)

**Mục tiêu:** có bức tranh bốn ô đầy đủ — nền tảng của mọi metric chống mất cân bằng.

Ta quy ước "positive = lớp bất thường/lỗi".

```
                Dự đoán 0        Dự đoán 1
Thực tế 0      TN (True Neg)     FP (False Pos - báo động giả)
Thực tế 1      FN (False Neg - bỏ sót lỗi)   TP (True Pos - bắt đúng lỗi)
```

Bốn ô này nền tảng cho mọi metric:
- **TP**: lỗi thật được báo → tốt.
- **FP**: báo lỗi khi thật bình thường → **báo động giả** → chi phí dừng máy vô ích.
- **FN**: lỗi thật bị bỏ sót → **nguy hiểm nhất** (máy tiếp tục chạy rồi vỡ).
- **TN**: bình thường và được xác nhận bình thường.

## 4.3. Precision, Recall, F1

**Mục tiêu:** đo đúng hai mặt của chất lượng dự đoán lớp hiếm: "bắt được bao nhiêu lỗi thật" (recall) và "báo bao nhiêu % là đúng" (precision), rồi gộp bằng F1.

Công thức:
`Precision = TP / (TP + FP)`
`Recall = TP / (TP + FN)`
`F1 = 2 * Precision * Recall / (Precision + Recall)` (harmonic mean)

**Ý nghĩa thực tế**:
- **Precision**: "Trong những gì mô hình BÁO là lỗi, bao nhiêu % là thật lỗi?" → đo mức báo động giả. Precision cao = ít làm phiền nhà máy.
- **Recall**: "Trong tất cả lỗi THẬT, mô hình bắt được bao nhiêu %?" → đo độ an toàn. Recall thấp = sót lỗi = nguy hiểm.
- **F1**: cân bằng hài hòa hai mục tiêu trên khi nhãn mất cân bằng.

### 4.3.1. Ví dụ số quyết định

Giả thiết: 10.000 normal, 100 lỗi. Mô hình A dự đoán:
- TP=80, FN=20 → bỏ sót 20 lỗi.
- FP=500, TN=9500 → 500 báo động giả.

Tính:
- Precision = 80/(80+500) = 80/580 = **13,8%**
- Recall = 80/(80+20) = 80/100 = **80%**
- Accuracy = (80+9500)/10100 = 9580/10100 = **94,9%**
- F1 = 2*(0.138*0.80)/(0.138+0.80) = 2*0.11044/0.938 = 0.22088/0.938 ≈ **23,5%**

Nhận xét: accuracy cao (94,9%) nhưng precision tệ — mô hình báo ồ ạt, gây 500 báo động giả. Trong nhà máy, 500 lần dừng máy vô lý là không thể chấp nhận.

Kiểm tra song song: Model B dự đoán TP=70, FN=30, FP=100, TN=9900:
- Precision = 70/170 = 41,2%; Recall = 70/100 = 70%; F1 = 2*(0.412*0.7)/(0.412+0.7) ≈ **51,9%**.

Model B recall thấp hơn chút nhưng precision tốt hơn nhiều → F1 cao hơn gấp đôi. **Bài học: chọn mô hình theo F1/PR, không theo accuracy.**

## 4.4. PR curve vs ROC/AUC — vì sao PR quan trọng cho lớp hiếm

> **Mục tiêu:** biết PR-AUC/AP là thước đo chính của A1, ROC/AUC chỉ là tham khảo phụ — vì ROC bị "che mắt" bởi số normal khổng lồ.

**Bài toán con:** một metric duy nhất mà không cần chọn threshold trước (threshold-independent), để so sánh model công bằng.

### 4.4.1. Tư tưởng: model cho xác suất → di chuyển threshold

Hầu hết model (Logistic, RF, XGB) output ra **xác suất** `p ∈ [0,1]`, rồi ta so với threshold `t` (mặc định 0.5) để quyết định 0/1. **Thay đổi `t` → thay đổi cặp (Precision, Recall)**. Ví dụ:

- `t` thấp (0,1): gần như mọi mẫu bị báo lỗi → Recall ≈ 100% nhưng Precision rất thấp.
- `t` cao (0,9): chỉ báo khi thực sự chắc chắn → Precision cao, Recall thấp.

### 4.4.2. PR curve

Vẽ **Precision (trục đứng) vs Recall (trục ngang)** khi quét `t` từ 1 → 0:

- Điểm (0, 1) khi `t` cao → Precision tối đa, Recall 0.
- Điểm (Recall=1, Precision thấp) khi `t` thấp.

**PR-AUC** = diện tích dưới đường PR. PR-AUC cao = model giữ Precision cao khi recall tăng — đúng thứ ta cần khi lớp hiếm.

### 4.4.3. ROC/AUC — cảnh báo

ROC vẽ **TPR (Recall) vs FPR (`FP/(FP+TN)`)** — FPR có mẫu số gồm toàn normal (rất nhiều) nên **dễ lạc quan** với lớp hiếm:

Ví dụ: 10.000 normal, 100 lỗi. Model báo 500 lỗi (FP=500):
- FPR = 500/10500 = 4,76% (trông bé nhỏ).
- Nhưng 500 báo động giả là 500 lần dừng máy — chi phí thực tế KHỔNG LỒ.
FPR nhìn "bé" vì mẫu số gồm toàn normal (nhiều) — hay cheat số đo. **ROC/AUC bị ảnh hưởng bởi TN**, còn PR chỉ quan tâm TP/FP/FN → nhạy hơn với lớp thiểu số.

So sánh nhanh:

| Tiêu chí | PR-AUC / AP | ROC-AUC |
|---|---|---|
| Nhạy với lớp hiếm | Cao — nhìn thẳng vào TP/FP/FN | Thấp — TN khổng lồ "nuốt" FPR |
| Diễn giải ổn định khi class lệch | Tốt | Dễ lạc quan giả tạo |
| Vai trò ở A1 | ⭐ metric chính | Tham khảo phụ |

**Kết luận cho A1 (class hiếm):** ưu tiên **PR-AUC**, **Average Precision (AP)** và báo cáo Precision/Recall tại một vài threshold cụ thể. ROC/AUC chỉ mang tính tham khảo phụ. (Các script team vẫn in AUC như một con số "phân biệt tổng quát" — đọc kèm với recall/precision, đừng xem nó là số quyết định.)

## 4.5. Threshold tuning & precision–recall tradeoff (chọn theo business)

> **Mục tiêu:** chọn threshold không phải "mặc định 0.5" mà theo chi phí thật của nhà máy — đây là câu chuyện khiến giám khảo tin bạn hiểu vấn đề.

### 4.5.1. Precision–recall tradeoff

Bạn không thể muốn cả hai cùng cao với một model cố định: giữa chúng có **đánh đổi (tradeoff)**. Khi tăng threshold → precision tăng, recall giảm (và ngược lại). Chọn điểm nào phụ thuộc **chi phí**:

### 4.5.2. Chi phí bỏ sót lỗi vs báo động giả — phân tích business

Trong nhà máy, hãy ước lượng:
- **C_cost**: chi phí BỎ SÓT một lỗi thật (máy hỏng, dừng sản xuất, sửa chữa lớn, mất an toàn) — thường rất lớn.
- **C_fp**: chi phí mỗi lần báo động giả (dừng máy kiểm tra vô ích, nhân công kiểm tra lại, mất nhịp sản xuất) — nhỏ hơn nhưng không bằng 0.
- **C_tp**: lợi ích bắt được một lỗi thật (sửa chủ động gọn nhẹ, tránh hỏng lớn).

Quy tắc chọn threshold: chọn `t` làm cho **tổng chi phí kỳ vọng nhỏ nhất**:
`Cost = FP * C_fp + FN * C_fault`

Ví dụ số: mỗi tuần máy có ~2 lỗi thật cần bắt. Nếu `C_fault = 50 triệu VND`, `C_fp = 2 triệu VND`:
- Model at t=0.5: FN=0, FP=8/tuần → Cost = 0*50tr + 8*2tr = **16tr**.
- Model at t=0.8: FN=0.3, FP=1/tuần (mỗi ~3 tuần sót 1 lỗi) → Cost = 0.3*50tr + 1*2tr = 15tr + 2tr = **17tr**.
- Model at t=0.65: FN=0.1, FP=3 → Cost = 0.1*50tr + 3*2tr = 5tr+6tr = **11tr** ← tối ưu.

Thay vì "cố định threshold mặc định", đọc đường PR, tìm điểm lướt trên–trái (precision giữ cao khi recall vẫn chấp nhận được), rồi chốt vài threshold liên tiếp cho 3 mức cảnh báo (warning / alarm / critical).

### 4.5.3. Mẹo thực hành threshold

- XGBoost output `predict_proba`; không dùng `predict` mặc định khi class lệch.
- Tìm threshold tối ưu trên **validation theo thời gian** (không tinh trên test — sẽ chủ quan).
- Với A1: chọn theo `max F1`, theo `min cost`, hoặc theo ràng buộc recall ≥ 90% → precision tối đa được trong điều kiện đó.

## 4.6. MAE / RMSE cho RUL (nếu làm dự báo tuổi thọ)

> **Mục tiêu:** biết bộ metric RIÊNG khi chuyển từ phân loại 0/1 sang hồi quy RUL — Precision/Recall không còn đúng nữa.

Khi chuyển sang bài toán **RUL (Remaining Useful Life)** — dự đoán số giờ (hoặc số chu kỳ) còn lại trước khi hỏng — đây là **hồi quy**, không còn dùng Precision/Recall nữa.

`MAE = (1/N) * sum(|y_i - yhat_i|)`
`RMSE = sqrt( (1/N) * sum((y_i - yhat_i)^2) )`

So sánh:
- **MAE**: trung bình độ lệch tuyệt đối — trực quan, đơn vị nguyên bản (giờ), không phạt mạnh outlier.
- **RMSE**: bình phương rồi căn — **phạt nặng lỗi lớn**. Nếu vài dự đoán lệch cực xa (dự đoán 500 giờ nhưng thật 2 giờ) → RMSE kéo lên rất cao.

Cho RUL, **lỗi âm (dự đoán ít hơn thật) là an toàn hơn lỗi dương** (dự đoán 2 giờ nhưng thật 500 giờ → máy vỡ trong lúc ta đang chờ). Một vài contest dùng **asymmetric score**:

`score = sum( exp(-yhat_i/13) - 1 )` nếu dự đoán muộn (over-predict, tức báo lỗi trễ — nguy hiểm), `sum( exp(yhat_i/10) - 1 )` nếu dự đoán sớm.

Lưu ý: model A1 thường trả về 0/1 trước, RUL là mở rộng — chỉ khi đề yêu cầu cụ thể về thời gian còn lại mới cần (chi tiết thêm ở Mục 7).

**Kết thúc mục 4 — checklist**:
- [ ] Báo cáo Confusion matrix + Precision/Recall/F1 tại vài threshold.
- [ ] Báo PR-AUC (hoặc AP) thay cho accuracy.
- [ ] Chọn threshold theo business cost, không mặc định 0.5.
- [ ] Nếu làm RUL: dùng MAE/RMSE + asymmetric score nếu contest quy định.

---

# 5. Anomaly detection / học KHÔNG giám sát

> **Mục tiêu của cả mục:** trả lời nhánh (3) — khi KHÔNG có (hoặc gần như không có) nhãn lỗi, chỉ học trên dữ liệu bình thường rồi "soi" cái gì khác bình thường.

> Đúng tinh thần A1: **thiếu dữ liệu lỗi**. Khi không có (hoặc gần như không có) nhãn lỗi, ta không thể train supervised. Giải pháp: chỉ học trên dữ liệu bình thường rồi "soi" cái gì khác bình thường.

Làm rõ phân biệt 3 hướng trong mục này:
1. **Unsupervised**: không có nhãn gì → group/cluster, lọc outlier.
2. **One-class / semi-supervised**: train CHỈ trên normal → đánh dấu bất kỳ thứ gì lệch khỏi normal.
3. **Statistical**: phân tích phân bố → ngưỡng động.

## 5.1. One-class SVM (OC-SVM)

> **Mục tiêu:** dựng một "biên bao" quanh cụm normal trong không gian feature; điểm nằm ngoài biên = bất thường. Chọn khi số mẫu normal vừa phải (hàng nghìn), dimension thấp–trung bình.

### 5.1.1. Ý tưởng

Thay vì tìm siêu phẳng phân tách 2 lớp, OC-SVM tìm một **vùng chứa hầu hết dữ liệu normal** và đánh dấu những điểm nằm ngoài là bất thường. Nói nôm na: một "quả cầu" mềm bao quanh cụm dữ liệu normal trong không gian feature.

### 5.1.2. Kernel trick

Không gian feature thật có thể phức hợp — không một siêu phẳng (hay quả cầu) đơn giản nào bao được. **Kernel** ánh xạ dữ liệu lên không gian chiều cao hơn để tìm biên dễ hơn:

- Linear: không gian giữ nguyên.
- RBF: `K(x,z) = exp(-gamma * ||x - z||^2)` — phổ biến nhất; đo "độ tương tự" giảm dần theo khoảng cách. Kiểm soát "độ mềm" biên bằng `gamma` & `nu`.

### 5.1.3. Tham số `nu`

`nu ∈ (0,1]` là tỉ lệ lỗi tối đa được phép (upper bound) và là tỉ lệ support vectors tối thiểu. Nói nôm na: `nu` nói với model "khoảng % dữ liệu normal có thể bị coi là outlier nếu cần".

- `nu=0.05`: cho phép coi ~5% normal là biên lỏng → biên rộng, ít báo động giả.
- `nu=0.001`: biên ép sát dữ liệu → nhạy bắt lỗi nhưng dễ báo giả.
Chọn `nu` bằng kinh nghiệm với chi phí business (giống threshold).

## 5.2. Isolation Forest

> **Mục tiêu:** tách outlier cực nhanh bằng cách so "độ sâu cô lập" trong cây ngẫu nhiên — lựa chọn hàng đầu khi dữ liệu chiều cao (nhiều feature) và cần tốc độ.

### 5.2.1. Ý tưởng nghịch đảo

Isolation Forest đi ngược logic "đo khoảng cách đến tâm": nó cố tìm một **cây quyết định ngẫu nhiên** phân rời dữ liệu càng nhanh càng tốt.

- Mỗi cây: chọn ngẫu nhiên feature + ngưỡng ngẫu nhiên để tách dữ liệu thành 2 nửa, lặp lại đến khi mỗi điểm nằm riêng lá.
- **Điểm dị biệt (outlier) bị tách ra CHỈ sau vài bước** (nằm trong vùng thưa xa, dễ chia) → **độ sâu (path length) ngắn**.
- **Điểm normal nằm trong cụm đông, cần nhiều lần chia mới cô lập** → path length dài.

Điểm bất thường = điểm có **path length trung bình ngắn** trên nhiều cây.

### 5.2.2. Ưu điểm

- Rất nhanh và nhẹ (không tính khoảng cách O(N²)).
- Hoạt động tốt trên dữ liệu dimension cao (hàng chục feature).
- Không cần giả định phân bố Gauss.

### 5.2.3. Hạn chế

- Nhạy với số cây & kích thước; có thể bỏ sót outlier cụm (nếu lỗi tạo cụm riêng đông thì lại không "cô lập nhanh").
- Ngưỡng đánh giá (`contamination`) vẫn phải đặt bằng tay — ta phải cài ~ tỉ lệ lỗi ước lượng.

**So sánh OC-SVM vs Isolation Forest vs Autoencoder — chọn cái nào cho A1?**

| Tiêu chí | OC-SVM | Isolation Forest | Autoencoder |
|---|---|---|---|
| Ý tưởng cốt lõi | Biên bao quanh normal (kernel) | Độ sâu cô lập trong cây ngẫu nhiên | Sai số tái dựng của mạng nén-giải nén |
| Giả định phân bố | Không (nhờ kernel) | Không | Không |
| Dữ liệu chiều cao / nhiều feature | Kém (slow, cần dimension vừa) | Tốt | Tốt (học biểu diễn) |
| Số mẫu normal cần | Hàng nghìn (không hàng triệu) | Ít hơn | Khá nhiều, dễ overfit nếu normal thiếu đa dạng |
| Tốc độ | Chậm ở quy mô lớn | Rất nhanh | Phụ thuộc mạng, cần GPU khi lớn |
| Diễn giải | Tương đối | Tương đối | Khó |
| Robustness với nhiễu nghiệp vụ | Medium | Cao (thứ hạng ngưỡng) | Cao nếu regularize tốt |

**Kết luận cho A1:** kết hợp IsolationForest như một **tầng 1 nhanh nhạy** (loại bất thường quá rõ), OC-SVM hoặc Autoencoder làm **tầng 2 chi tiết** hơn; nếu đủ dữ liệu normal, Autoencoder cho biểu diễn mạnh nhất (chi tiết ở Phần 3). Ở hiện trạng team, `scripts/01_baseline_anomaly.py` chọn **Autoencoder** làm mốc vô giám sát — chính là "tầng chi tiết" này, và `results/baseline_ae.json` lưu con số baseline (xem Mục 6.4).

## 5.3. Autoencoder + reconstruction error (giới thiệu — chi tiết phần Deep Learning)

**Autoencoder** là mạng neural: nén dữ liệu về không gian ẩn chiều thấp (encoder), rồi giải nén về gần giống đầu vào (decoder). Nó **chỉ được huấn luyện trên dữ liệu NORMAL**.

Công thức tổng quát:
```
z = f_enc(x)          # encode x thành vector ẩn
x_hat = g_dec(z)      # tái dựng x
reconstruction_error = || x - x_hat ||^2   (thường là MSE)
```

Khi test:
- Mẫu **normal** → tái dựng tốt → reconstruction error thấp.
- Mẫu **bất thường** (dạng chưa từng thấy) → không nén được → error cao → nhận diện là lỗi.

Trực giác: autoencoder "nhớ" khuôn dạng số liệu normal; thứ không giống normal sẽ không khớp.

**Ứng dụng A1**:
- Input có thể là **vector feature** (như 16 feature của `src/features.py` — đúng cách `scripts/01` dùng) hoặc **đoạn spectrogram/waveform** của cửa sổ.
- Output = `reconstruction_error` → tiếp tục dùng **statistical threshold** (5.4) để quyết định 0/1.
- Điểm mạnh: học biểu diễn mạnh mẽ, mở rộng theo dữ liệu; nhược điểm: cần khá nhiều dữ liệu normal, dễ overfit nếu normal thiếu đa dạng, khó giải thích.
Đi sâu về architecture, loss, đánh giá sẽ ở phần **Deep Learning (part 3)**.

## 5.4. Statistical / heuristic: z-score, rolling threshold, EWMA, CUSUM

> **Mục tiêu:** phát hiện bất thường CHỈ bằng ngưỡng trên một-không-ít biến (thường RMS/band energy) — nhanh, cài ngay, nhà máy hiểu được. Là họ "vũ khí" rẻ và minh bạch nhất; không cần train model.

Đây là các kỹ thuật "chỉ dùng 1 biến (thường là RMS/band energy)" — nhanh, cài được ngay, có ý nghĩa với nhà máy.

### 5.4.1. Z-score (chung cho 1 biến đo)

**Mục tiêu:** chênh bao nhiêu độ lệch chuẩn so với mức bình thường — ngưỡng theo thang σ, trực giác với kỹ sư.

Cách: tính mu, sigma trên **toàn bộ dữ liệu normal** (hoặc cửa sổ đầu record xem là ổn định). Ngưỡng: nếu `|z| = |x - mu|/sigma > T` với T thường 3–4 → báo lỗi. Z-score T=3 nghĩa "chỉ ~0,3% mẫu normal nằm ngoài" (theo xấp xỉ Gauss).

Giới hạn: phải biết dữ liệu gần Gauss (RMS gần vậy); nếu có nhiều cụm chế độ vận hành khác nhau (idle vs full load) thì scale toàn cục kém.

### 5.4.2. Rolling threshold (ngưỡng cuốn theo thời gian)

Thay vì 1 mu/sigma toàn cục, dùng **rolling window**: với điểm tại thời gian t, tính mu, sigma từ cửa sổ trước đó (ví dụ 5 phút lui về). Ngưỡng:
`alarm khi x[t] > mu_roll + k * sigma_roll`

Ưu: bám theo drift chậm của máy. Nhược: nếu lỗi diễn biến từ từ (càng tăng dần), rolling cũng "ăn theo" lỗi → chậm bắt; cần **baseline cố định** từ dữ liệu sạch khi máy mới.

### 5.4.3. EWMA (Exponential Weighted Moving Average)

Công thức đệ quy:
`EWMA[t] = alpha * x[t] + (1 - alpha) * EWMA[t-1]`

- `alpha` (0 < alpha < 1): trọng số cho mẫu mới; alpha gần 0 → trơn lâu (ít nhạy), alpha gần 1 → phản ứng nhanh (dễ nhiễu).
- EWMA làm mịn nhiễu nhanh để bắt **xu hướng** nhẹ. Ngưỡng: báo khi `EWMA[t] > mu + lambda * sigma_ewma` (lambda ví dụ 3).

### 5.4.4. CUSUM (Cumulative Sum) — kiểm tra thay đổi phát hiện

CUSUM tích lũy chênh lệch nhỏ có hướng của biến theo thời gian — cực nhạy với **drift nhỏ nhưng kéo dài** (lỗi từ từ) mà z-score/EWMA bỏ lỡ.

Công thức hai hướng:
```
S_plus[t] = max(0, S_plus[t-1] + (x[t] - mu0) - K)
S_minus[t] = max(0, S_minus[t-1] + (mu0 - x[t]) - K)
Báo lỗi khi S_plus[t] > h  hoặc  S_minus[t] > h
```

- `mu0`: giá trị trung bình kỳ vọng (bình thường).
- `K`: mức dịch chuyển tối thiểu muốn phát hiện (mức độ nhạy); thường K = 0.5 * sigma.
- `h`: ngưỡng báo (thường vài lần K, ví dụ h = 4–5 * sigma_shift).

**So sánh 4 kỹ thuật statistical — chọn cái nào?**

| Kỹ thuật | Bắt lỗi gì | Nhạy drift nhỏ kéo dài | Phức tạp cài đặt | Ảnh hưởng outlier đơn lẻ |
|---|---|---|---|---|
| Z-score | Nhảy vọt đột ngột | Kém | Rất thấp | Cao |
| Rolling | Drift chậm, theo cục bộ | Trung bình (có thể "ăn theo" lỗi) | Thấp | Trung bình |
| EWMA | Xu hướng nhẹ, làm mịn nhiễu | Tốt | Thấp | Thấp |
| CUSUM | Drift nhỏ kéo dài | Rất tốt | Trung bình | Thấp |

**Kết luận cho A1:** bắt đầu bằng z-score hoặc EWMA (đơn giản, dễ giải thích với giám khảo); nếu biết lỗi phát triển từ từ (mài mòn) → thêm CUSUM hoặc ngưỡng percentile ổn định trên baseline.

**Cảnh báo** với mọi phương pháp statistical: phải xác định rõ liệu "normal" có phải là một chế độ ổn định hay có nhiều chế độ (vận hành khác nhau). Nếu có nhiều chế độ, hãy tách theo chế độ (clustering hoặc chia dòng) trước khi tính ngưỡng.

## 5.5. Cài ngưỡng khi KHÔNG có nhãn

> **Mục tiêu:** khi không có nhãn lỗi thật, vẫn phải chốt được ngưỡng "báo động" — bằng percentile, heuristic vật lý, hoặc quy tắc thời gian.

Vì không có nhãn lỗi thật, ta phải "đoán" ngưỡng. Các cách:

### 5.5.1. Percentile (phân vị)

Cách: tính mọi anomaly score (reconstruction error, isolation factor, khoảng cách...), rồi chọn ngưỡng ở phân vị cao:
`threshold = percentile(scores, 99.0)` (hoặc `99.5`, `99.9`).

Chỉ số: 1% hay 0,5% dữ liệu bị coi là bất thường. Nếu ta biết từ kinh nghiệm nhà máy "1% thời gian là có vấn đề nhỏ", dùng percentile 99. Nếu lỗi hiếm hơn, 99.9.
(Đúng cách `scripts/01_baseline_anomaly.py` làm: `--thr 99.0` → ngưỡng = percentile 99 của reconstruction error trên tập normal.)

### 5.5.2. Heuristic theo vật lý

- Với RMS rung: so với giá trị trong datasheet máy (ví dụ ISO: máy "tốt" < 2,8 mm/s; "cảnh báo" 2,8–4,5; "dừng" > 7,1). → lấy ngưỡng từ tiêu chuẩn.
- Với band energy ở tần số ổ trục: so sánh độ tăng cho phép (ví dụ tăng > 10× so với baseline normal).
- Với multiple sensors: ngưỡng theo mức tăng tương đối so với chính chuỗi từng máy (khác nhau giữa máy mới và máy cũ).

### 5.5.3. Kết hợp "score + thời gian": ngưỡng mềm

Tránh báo động do spike nhiễu đơn: quyết định "lỗi" chỉ khi score vượt ngưỡng trong **k window liên tiếp** (ví dụ 5/10). Hoặc dùng trung bình 3 điểm gần nhau → giảm báo động giả mà không bỏ lỡ chuỗi lỗi kéo dài.

### 5.5.4. Self-validation khi dữ liệu normal sạch

Nếu ta biết 2–3 chuỗi dữ liệu đầu là sạch (máy vừa kiểm định): dùng nó làm **reference baseline** để calibrate mu, sigma, ngưỡng percentile. Nhưng KHÔNG dùng nó làm "test tự do" — vẫn phải dành một phần để đánh giá cuối (xem 5.6).

## 5.6. Cross-validation cho anomaly: hạn chế & data leakage

> **Mục tiêu:** hiểu vì sao CV cổ điển (shuffle) sai với chuỗi thời gian, và cách chia **train = quá khứ, test = tương lai** là bất biến cho A1.

### 5.6.1. Vấn đề cơ bản với dữ liệu chuỗi thời gian

K-fold CV cổ điển **shuffle ngẫu nhiên** các mẫu → trộn mẫu trong quá khứ với mẫu trong tương lai. Điều này là **data leakage**: model "nhìn thấy" tương lai trong quá trình train → điểm trên CV cao giả tạo, nhưng khi deploy (chỉ biết quá khứ), model tệ hơn nhiều — **kết quả đánh giá bị lệch nghiêm trọng**.

Hệ quả với anomaly detection: nếu các mẫu normal giai đoạn đầu giống nhau và mẫu lỗi nằm cuối, shuffle sẽ "lén" cho model biết dạng lỗi ở phase hoặc window kề — bất công.

### 5.6.2. Quy tắc vàng: KHÔNG shuffle giữa past/future

Đúng nhất với deploy: **train = quá khứ, test = tương lai**.

```
[data theo thời gian]
|---------- train (past) ----------||------ test (future) ------|
```

- Với **anomaly score** (không nhãn): dùng **TimeSeriesSplit** — chia theo khối thời gian liên tiếp: lần 1 train tháng 1 → test tháng 3; lần 2 train tháng 1–3 → test tháng 4;...
- Khi làm **sliding block**: vùng train phải TẤT CẢ trước vùng test trong cùng fold. Không có bất kỳ điểm nào trong test nằm trước thời điểm train lân cận.
- Đặc biệt với smooth features (rolling mean, EWMA): rolling window **chỉ được tính từ dữ liệu quá khứ**, không tính từ tương lai. Nói cách khác, tính feature bằng cách "chỉ nhìn về trước" (causal). Với offline dataset, dễ mắc lỗi âm thầm dùng rolling giữa tâm → phải kiểm tra code.

### 5.6.3. Hạn chế của CV khi thiếu nhãn/thiếu dữ liệu lỗi

1. Không có metrics chuẩn vàng: không biết số lỗi thật → mọi ngưỡng percentile chỉ là giả định; kết quả CV "tốt" không đảm bảo bắt đúng lỗi thật.
2. Ít đại diện quần thể: nếu chỉ 1–2 chuỗi lỗi thật, chia fold có thể làm test chỉ chứa 1 chuỗi → độ tin cậy thấp (variance cao).
3. Nếu thống kê học (z-score/Rolling) dùng rolling window để tính ngưỡng trên train thì sẽ "đúng" trên train nhưng chưa chắc generalize.

**Cách giảm thiểu**:
- Báo cáo **phân phối anomaly score** trên train/test riêng (histogram) — giúp nhìn trực giác liệu ngưỡng có hợp lý.
- Đánh giá trên **chuỗi thời gian thật** (file thứ n so với file thứ 1) — so sánh khi máy mới (baseline) vs khi máy cũ.
- Nếu có nhãn ít ỏi, dùng nó CHỈ cho test cuối, tuyệt đối không cho chọn threshold (tránh overfit vào vài chuỗi lỗi hiếm đó).

---

# 6. Xử lý dữ liệu mất cân bằng + thiếu dữ liệu lỗi (nhánh chính)

> **Mục tiêu của cả mục:** trả lời nhánh (5) và (6) — làm sao giúp model học lớp lỗi hiếm tốt hơn (augmentation), và làm sao chứng minh điều đó bằng protocol trung thực.

> Giả định: có TOÀN BỘ normal + MỘT PHẦN rất nhỏ lỗi thật (a few). Mục này trả lời: làm sao để mô hình học hiệu quả và đánh giá trung thực.

## 6.1. Oversampling / undersampling

> **Mục tiêu:** cân bằng tỉ lệ lớp (50/50 hoặc gần đó) để model không "lười" chỉ đoán lớp đông — nhưng phải biết cái giá của từng cách.

### 6.1.1. Undersampling (giảm mẫu lớp đông)

- Loại ngẫu nhiên mẫu normal để cân bằng (50/50 hoặc gần đó).
- Lợi: nhẹ, nhanh, bias về lớp lỗi chứ không bị áp đảo.
- Hại: **vứt bỏ dữ liệu normal quý** → normal ít hơn dẫn đến học phân bố normal kém, mất ổn định. Chỉ nên dùng khi dữ liệu normal cực nhiều và lỗi có nghĩa.

### 6.1.2. Oversampling (nhân mẫu lớp thiểu số)

- **Nhân bản ngẫu nhiên** các mẫu lỗi cho bằng số normal.
- Lợi: giữ mọi dữ liệu.
- Hại: nhân bản tạo **trùng lặp chính xác** → model có thể "thuộc lòng" từng mẫu lỗi → overfit, và CV shuffle sẽ bị "trùng mẫu giữa train/test" (leakage!) nếu nhân bản xong rồi mới chia train/test.

⚠️ **Quy tắc quan trọng**: thực hiện resampling **TRONG fold** (chỉ trên train split) — không bao giờ resample toàn bộ rồi mới tách, nếu không test sẽ chứa bản sao của các mẫu train → metric giả tạo. (Đúng cơ chế `scripts/02`: lỗi giả chỉ ghép vào `X_base`/`y_base` — tức vào TRAIN; `X_test`/`y_test` không bao giờ chạm.)

## 6.2. SMOTE (Synthetic Minority Over-sampling Technique)

> **Mục tiêu:** sinh thêm mẫu lỗi TỔNG HỢP (không phải nhân bản) ở khoảng giữa các mẫu lỗi thật — đa dạng hóa lớp hiếm hơn oversampling đơn thuần.

### 6.2.1. Ý tưởng

Thay vì nhân bản y hệt, SMOTE sinh **mẫu tổng hợp nằm trên đoạn thẳng nối giữa các mẫu lỗi gần nhau**:

Cách:
1. Với mỗi mẫu lỗi `a`, chọn ngẫu nhiên k láng giềng cùng lớp lỗi (k-NN trong không gian feature, k thường 5).
2. Chọn một láng giềng `b`, tạo điểm mới:
`x_new = a + lambda * (b - a)`, với `lambda ~ Uniform(0,1)`.
3. Lặp cho đến đủ số mẫu mong muốn.

### 6.2.2. Ví dụ số

Feature (kurtosis=5.0, RMS=0.4) và láng giềng (kurtosis=7.0, RMS=0.6). Với lambda=0.5:
`x_new = (5.0+0.5*(7.0-5.0), 0.4+0.5*(0.6-0.4)) = (6.0, 0.5)`.
Mẫu mới nằm "ở giữa" hai mẫu lỗi thật.

### 6.2.3. Ưu / nhược

- Ưu: đa dạng hóa lớp thiểu số hơn nhân bản; thường cải thiện recall/F1 khi dữ liệu lỗi hiếm (trong trường hợp số mẫu lỗi đủ để ước lượng cụm).
- Nhược / nguy hiểm:
  1. **Nhạy nhiễu**: nếu một mẫu lỗi là nhiễu đo, SMOTE sinh thêm mẫu nhiễu quanh nó → model học lỗi giả.
  2. **Nội suy sai vị trí**: với feature không smooth (kurtosis có thể nhảy), nội suy tuyến tính tạo mẫu "không có thật" nằm giữa cụm normal → có thể gây lẫn lộn.
  3. **Nguồn dữ liệu lỗi quá ít** (như 5–20 mẫu) thì SMOTE kém; ít nhất nên có vài chục mẫu.
  4. Vẫn phải resample trong fold.

Biến thể đáng biết: **Borderline-SMOTE** (tập trung sinh quanh biên), **SMOTE-ENN / SMOTE-Tomek** (kết hợp undersampling để dọn nhiễu và làm sạch biên).

## 6.3. Data augmentation bằng dữ liệu tổng hợp (GAN / diffusion) — giới thiệu

> **Mục tiêu:** khi vài mẫu lỗi thật là quá ít cho mọi heuristic, dùng generative model học phân bố lớp lỗi để "bơm" thêm mẫu — là deliverable nổi bật của đề A1 ("bộ sinh lỗi giả").

Khi đã có vài mẫu lỗi thật nhưng quá ít, ta có thể **sinh thêm mẫu lỗi bằng generative model** (huấn luyện để "mô phỏng" phân bố lớp lỗi). Hai họ phổ biến:

### 6.3.1. GAN (Generative Adversarial Network)

- Generator G sinh ra giả; Discriminator D đoán giả/thật. Đấu nhau (adversarial) cho đến khi G sinh ra mẫu không phân biệt được với thật.
- Sinh mẫu lỗi giả từ mẫu lỗi ít ỏi: G học phân bố lớp lỗi → sinh thêm.
- Nhược: khó train (mode collapse — G chỉ sinh vài dạng), cần dữ liệu đủ để "huấn luyện ổn định"; với 5 mẫu lỗi thì GAN thường không ăn — chỉ nên dùng khi ≥ vài trăm mẫu lỗi.

### 6.3.2. Diffusion model

- Học cách **khử nhiễu (denoise)**: thêm nhiễu dần vào mẫu thật đến khi gần như trắng, rồi học ngược để tái dựng → sinh mẫu mới.
- Thường ổn định hơn GAN, chất lượng tốt, nhưng nặng tính toán → thực dụng hơn khi data ít? Vẫn cần lượng mẫu đáng kể cho phân bố lỗi.

**So sánh SMOTE vs GAN/diffusion — chọn cái nào?**

| Tiêu chí | SMOTE | GAN | Diffusion |
|---|---|---|---|
| Nguyên lý | Nội suy tuyến tính giữa mẫu thiểu số | Đấu G-D để học phân bố | Học khử nhiễu ngược để sinh |
| Số mẫu lỗi tối thiểu | Vài chục | Vài trăm | Vài trăm (nặng hơn) |
| Chất lượng mẫu | Chạy theo cụm, nhạy nhiễu | Tốt nhưng dễ mode collapse | Tốt, ổn định hơn |
| Chi phí | Rẻ, chạy CPU | Trung bình–cao | Cao |
| Rủi ro | Nội suy vào vùng "không có thật" | Train mất ổn định | Nặng, cần tài nguyên |
| Vai trò ở A1 | ⭐ Khởi điểm nhanh, đủ dùng | Tầng nâng cao, cần nhiều lỗi thật | Xu hướng, deliverable "xịn" |

**Kết luận cho A1:** SMOTE là điểm khởi đầu rẻ và đủ dùng; khi muốn điểm cộng "bộ sinh lỗi giả" (đúng yêu cầu đề), nâng lên **GAN/diffusion** — nhưng luôn nhớ: mẫu sinh chỉ phụ trợ, con số quyết định là protocol trung thực (6.4). (Trong repo, `scripts/03` sinh `synthetic_faults.npy` bằng TimeGAN rồi `scripts/02 --gen npy` dùng lại — đúng hướng này.)

> Chi tiết triển khai: **phần Deep Learning (part 3)**. Ở đây chỉ ghi nhận: augmentation tổng hợp THƯỜNG là phụ (chỉ giúp), không thay thế được một protocol đánh giá trung thực. Nếu nguyên bản có 50 mẫu lỗi thật, hãy giữ nguyên cho test.

## 6.4. Protocol so sánh trước/sau augmentation — vì sao tránh data leakage

> **Mục tiêu của mục này (quan trọng nhất trong Phần 2):** định nghĩa một giao thức đánh giá TRUNG THỰC để chứng minh augmentation có giá trị hay không — và cập nhật đúng hiện trạng split của team.

### 6.4.1. Protocol chuẩn mực (khi có MỘT VÀI lỗi thật)

```
1. Tách theo thời gian TRƯỚC KHI làm bất cứ điều gì khác:
   train_norm_windows       = normal windows (thời gian trước, vùng khỏe)
   train_fault_windows      = vài windows lỗi THẬT (giữ nguyên, không sinh thêm)
   test_future_windows      = windows (cả normal cả lỗi THẬT) nằm SAU cùng deadline
2. Augmentation (SMOTE / GAN / copy):
   CHỈ áp dụng lên train (train_norm + train_fault)
   test_future_window KHÔNG BAO GIỜ bị sinh thêm, không bị trộn
3. Fit scaler / feature transform CHỈ trên train → áp lên test.
   (scaler trước augmentation cho công bằng; không fit trên toàn bộ)
4. Đánh giá cuối: Precision/Recall/F1 (hoặc PR-AUC) TRÊN test_future chỉ.
5. Báo cáo cả hai:
   - baseline (train KHÔNG augmentation)
   - augmented (train CÓ augmentation)
   → soi được việc augmentation có thật sự giúp generalization hay chỉ "khớp" mồi.
```

### Hiện trạng split của team (cập nhật mới nhất — phải đọc kỹ)

Trước đây một nhánh thí nghiệm cũ lấy **test normal là 20% NORMAL CUỐI** (phần normal sát vùng hỏng). Đó là một **test bất công**: phần normal ấy thực ra đã suy giảm, gần với lỗi thật → model bị đánh giá trên vùng mơ hồ → recall thấp giả tạo (có lần chỉ ~0.013). Không phải model yếu mà là TEST SAI CHỖ.

`src/pipeline.py` hiện tại đã sửa và định nghĩa split đúng như sau — **bạn báo cáo theo đúng cách này**:

```
Mỗi test-run theo thời gian:
  [0% ---- 70% ---- 90% ---- 100%]
  [ NORMAL khỏe | vùng mơ hồ | FAULT ]
  - 0 → 70%   : NORMAL (toàn bộ)
  - 70 → 90%  : BỎ QUA (nhãn không rõ, tránh nhãn nhiễu)
  - 90 → 100% : FAULT (dữ liệu lỗi THẬT, chính là thứ hiếm)

Trong riêng vùng NORMAL của MỖI run (tính trên phần 0–70%):
  [0% — 70% | 70% — 90% | 90% — 100%]
  [ TRAIN    |  TEST     |  GREY (loại) ]
  - TRAIN : normal thật khỏe       → model học "bình thường là gì"
  - TEST  : normal chưa thấy NHƯNG VẪN KHỎE  → test đánh giá
  - GREY  : normal đã suy giảm, sát vùng hỏng → LOẠI khỏi cả train lẫn test
            (giữ GREY trong test → model khó phân biệt normal-gần-hỏng với lỗi
             thật → recall/AUC thấp do test "bất công", không phải do model yếu)

FAULT: trộn ngẫu nhiên trong cụm lỗi → 20% TRAIN / 80% TEST.
       Test dùng lỗi THẬT chưa từng thấy khi train → chống data leakage.

Chuẩn hoá z-score: fit mean/std TRÊN TRAIN (normal + fault train) rồi áp cho
train + test + synthetic. Scaler không bao giờ nhìn test.
```

### 6.4.2. Con số baseline THẬT của team (chạy trên split mới này)

Chạy trên dataset IMS (`--limit 300`, win=512, stride=256), kết quả lưu trong `results/`:

**Baseline vô giám sát — AutoEncoder** (`scripts/01`, `results/baseline_ae.json`):
- Ngưỡng (percentile 99 của reconstruction error trên normal): **0.0164**
- Precision **0.387** · Recall **0.031** · F1 **0.057** · AUC **0.524**

Đọc: AE thuần "học normal rồi đo độ lạ" bắt được rất ít lỗi thật (recall ~3%), AUC gần 0.5 (đoán bừa). Đây chính là lý do đề bài muốn ta **thêm dữ liệu lỗi giả để model giám sát học tốt hơn** — tức bài toán của script 2.

**So sánh trước/sau augmentation — RandomForest** (`scripts/02 --gen heuristic`, `results/compare_augmentation.json`):

| metric | trước aug | sau aug (heuristic) | Δ |
|---|---|---|---|
| precision | 0.416 | 0.418 | +0.002 |
| recall | **0.477** | **0.482** | +0.005 |
| f1 | 0.444 | 0.447 | +0.003 |
| auc | **0.582** | **0.580** | −0.002 |

Semantics: recall = % lỗi thật bắt được (0.477 nghĩa ~48/100); precision = % cảnh báo đúng; AUC = phân biệt tổng quát (0.5 = đoán bừa).

Hai điều rút ra:
1. **Split MỚI (test normal khỏe) cho recall thực tế cao** — chuyển từ ~0.013 (test bất công, trúng vùng suy giảm) lên **0.477** — vì test giờ hỏi đúng câu: "phân biệt normal khỏe với lỗi thật".
2. So với mốc AE vô giám sát (recall 0.031), RF có nhãn + chút lỗi thật đã **nhảy vọt lên recall 0.477 → 0.482**. Heuristic augmentation chỉ đẩy thêm nhẹ (+0.005 recall); bản `--gen npy` (TimeGAN) và tinh chỉnh feature/threshold là hướng để tăng tiếp.

### 6.4.3. Vì sao tránh data leakage (điểm mấu chốt)

1. **Nếu shuffle 80/20 ngẫu nhiên**: mẫu lỗi ở file4 có thể rơi vào train, còn bản sao ở phần khác của CHÍNH file đó rơi vào test → mô hình "gặp" mẫu tương tự nhau → recall cao giả tạo. Khi deploy máy mới, không có "bản sao" đó → hỏng thật.
2. **Nếu fit scaler trên toàn bộ**: mu/sigma chứa thông tin test (thống kê tương lai) → feature trong train "bị nhìn trước" → mọi metric tin cậy thấp.
3. **Nếu SMOTE trên toàn bộ rồi tách**: mẫu sinh ra giữa hai mẫu lỗi thật có thể sinh ra một điểm gần với test → leak.

Nguyên tắc tóm gọn: **"test tương lai là vùng đất thiêng — augmentation, scaling, chọn threshold, chọn feature đều chỉ thực hiện trong train."**

### 6.4.4. Kỳ vọng thực tế của augmentation

- Tốt cho: giúp model học biên lớp lỗi ổn định hơn, giảm variance với mẫu lỗi ít.
- Không thần kỳ: nếu 5 mẫu lỗi, không thể sinh ra "lỗi mới thuộc loại khác"; SMOTE/GAN chỉ nội suy quanh các mẫu đã có.
- Kết luận: augmentation là **phụ trợ**, không phải thuốc chữa bách bệnh. Ưu tiên số 1 vẫn là feature engineering + threshold tối ưu + model tốt trên dữ liệu thật. Báo cáo so sánh trước/sau (như 6.4.2) đúng chuẩn "deliverable A1" mà giám khảo chấm.

---

# 7. RUL (Remaining Useful Life) dự báo ngắn

> **Mục tiêu:** khi đề mở rộng từ "0/1" sang "còn bao lâu nữa hỏng", biết cách chuyển bài toán thành hồi quy và chọn loss/phương án đánh giá đúng.

Khi A1 mở rộng: không chỉ trả 0/1 mà muốn biết **còn bao nhiêu giờ trước khi hỏng** → bài toán **hồi quy**.

## 7.1. Thiết lập bài toán

- Input: window các feature tại thời điểm t.
- Output: RUL = số giờ (hoặc chu kỳ hoặc vòng quay) còn lại.
  Ví dụ dataset hình thành từ chuỗi hư hỏng: file từ khi máy mới (RUL=365 ngày) giảm dần đến 0 (hỏng).
- Nếu nhãn chỉ "một khoảng thời gian trước khi hỏng" → gán tuyến tính hoặc chạm sàn (capping) RUL tại giá trị tối đa quan tâm: `RUL_capped = min(RUL, RUL_max)` với RUL_max như 30 ngày → model dễ học hơn.

## 7.2. Loss functions cho hồi quy

- **MSE**: `(1/N) * sum((y - yhat)^2)` — dùng XGBRegressor / Neural. Phạt lỗi lớn nặng.
- **MAE**: `(1/N) * sum(|y - yhat|)` — robust hơn với outlier, trực quan.
- **Huber loss**: MSE khi lỗi nhỏ, MAE khi lỗi lớn — dung hòa 2 loại trên, rất ổn với chuỗi thật (outlier nặng).
- **Asymmetric / quantile loss**: phạt lỗi âm và dương khác nhau — phù hợp business RUL (muốn không trì hoãn báo hỏng). Quantile loss:
`L = max( tau*(y - yhat), (tau-1)*(y - yhat) )` với tau là quantile muốn (tau=0.5 → giống MAE; tau=0.8 → phạt mạnh khi yhat < y, tức dự đoán sớm quá).

So sánh nhanh: MSE phạt lỗi lớn nặng (muốn không có cú lệch cực xa), MAE trực quan nhất (đơn vị giờ), quantile loss cho ta "phương án an toàn" của RUL. **Cho A1 (nếu làm RUL):** bắt đầu bằng MAE để diễn giải, bổ sung quantile/asymmetric nếu contest quy định score riêng.

## 7.3. Đánh giá

- **MAE**: dễ hiểu (trung bình lệch ± giờ).
- **RMSE**: phạt chệch xa.
- Nếu contest có **score riêng** (exponential như NASA PHM 2008): dùng hẳn score contest:
  `Score = sum( exp(-(y-yhat)/13) - 1 )` nếu y-yhat < 0 (dự đoán sớm — ít nguy hiểm), và `sum( exp((y-yhat)/10) - 1 )` nếu y-yhat > 0 (dự đoán muộn — nguy hiểm hơn).

## 7.4. Lưu ý chuỗi thời gian trong RUL

1. RUL các window cùng một máy **không độc lập** — ta không thể shuffle. Phải chia theo máy/chuỗi: train trên máy lẻ, test trên máy chẵn — hoặc theo thời gian.
2. Feature hướng về tương lai: rolling statistics phải causal.
3. RUL đánh giá bằng xu hướng đơn điệu tăng của error khi gần hỏng — vẽ `predicted_vs_true` theo thời gian để kiểm tra drift.

**Khuyến nghị nhanh**: nếu đề A1 chủ yếu là 0/1, đừng lăn tăn RUL — chắc chắn feature + protocol + metrics (mục 4 & 6) trước; RUL chỉ làm khi có thời gian và yêu cầu rõ.

---

# 8. Pipeline thực hành cho A1

> **Mục tiêu của cả mục:** gom toàn bộ nhánh trên cây bài toán con thành một luồng chạy được từ raw signal đến báo cáo — kèm quyết định nhanh và checklist trước nộp.

## 8.1. Sơ đồ tổng thể

```
                     ┌──────────────────────────────────────────┐
 RAW DATA (CSV/Parquet: time, vib_x, vib_y, vib_z, temp, current)
                     │
                     ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  1. LOAD & INSPECT                                              │
 │     • metadata tần số lấy mẫu fs, đơn vị (g, mm/s, °C, A)      │
 │     • eda: plot waveform, histogram, boxplot theo thời gian     │
 │     • phát hiện missing/gap; quyết định resample nếu cần        │
 └─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  2. WINDOWING + FEATURE ENGINEERING                             │
 │     • chọn window (0.05–0.2s) & stride (overlap 50–90%)          │
 │     • nhân window function (hann) trước FFT                      │
 │     • time-domain: mean, std, rms, peak, p2p, skew, kurt, cf,... │
 │     • freq-domain : FFT → spectrum → centroid/spread/flatness/   │
 │                     band_energy[8] + harmonic features            │
 │     • sensor-level: nối tất cả sensor cùng window → 1 dòng feature│
 └─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  3. SPLIT THEO THỜI GIAN (KHÔNG SHUFFLE)                        │
 │     train(quá khứ) ──▶ validation(gần sau) ──▶ test(tương lai nhất)│
 │     (A1: normal 3-zone [Train|Test|Grey], fault 20/80 — Mục 6.4) │
 └─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  4. FEATURE SCALING (fit TRÊN TRAIN, áp sang val/test)          │
 │     • RobustScaler cho tín hiệu rung; z-score/log nếu cần        │
 │     • handle missing, cột constant, outlier theo IQR             │
 └─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │  5. MODEL (so sánh nhánh)                                             │
 │      a. Supervised: LogReg → RF → XGBoost (scale_pos_weight, tune)      │
 │      b. Unsupervised: IsolationForest / OC-SVM / (Autoencoder part DL)   │
 │      c. Hybrid: dùng a. sau khi đã có vài lỗi, b. khi chưa có nhãn      │
 │      d. Augmentation (SMOTE / gen) CHỈ trong train, compare trước/sau   │
 └────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │  6. METRICS + THRESHOLD (business-oriented)                           │
 │     • Confusion matrix + Precision/Recall/F1 tại vài threshold        │
 │     • PR-AUC / AP; ROC/AUC tham khảo                                  │
 │     • chọn threshold theo cost (C_fault vs C_fp) hoặc min recall      │
 │     • (RUL nếu có: MAE/RMSE, quantile loss)                            │
 └────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │  7. CHẠY TRÊN TEST CHƯA TỪNG CHẠM, GHI KẾT QUẢ, DIỄN GIẢI            │
 │     • báo cáo ví dụ: hệ thống "alarm" tại giây X — đúng/giả?           │
 │     • feature importance → vẽ top features (kurtosis, E_band_hi,...)   │
 └────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │  8. TRIỂN KHAI (demo + dashboard)                                      │
 │     • batch thao tác trên file mới → ghi kết quả CSV                 │
 │     • dashboard: theo thời gian (vib + anomaly score + alert)         │
 │     • API stream: nhận window → trả (prob, class, rms, kurtosis)      │
 └────────────────────────────────────────────────────────────────────────┘
```

Đối chiếu repo hiện tại: bước 3–5 được gom sẵn trong `src/pipeline.py` (`split_and_scale`, `to_train_test_arrays`) để ba script dùng chung; bước 2 trong `src/features.py`; bước 5b/6 trong `scripts/01`; bước 5d/6 trong `scripts/02`; bước 5a–6 kết hợp trong `scripts/03`. Muốn nộp bài "reproducible một cú", hãy giữ mọi tham số qua CLI (như các script đang làm).

## 8.2. Quyết định quan trọng rút gọn (decision guidance)

| Câu hỏi | Câu trả lời ngắn |
|---|---|
| Có ≥ vài trăm mẫu lỗi nhãn? | Ưu tiên supervised (XGBoost) |
| Có 0 – vài chục mẫu lỗi? | Unsupervised/one-class + ngưỡng percentile; hoặc supervised trên normal-only + score |
| Feature miền nào quan trọng nhất? | band_energy vùng ổ trục tần số cao, kurtosis, crest factor, spectral centroid |
| Nên chuẩn hóa gì? | RobustScaler (rung có nhiều outlier); hiện tại repo dùng z-score fit trên train — hợp lệ |
| Threshold chọn sao? | Theo business cost, không mặc định 0.5 |
| Chia dữ liệu sao? | Theo thời gian, KHÔNG shuffle; test normal lấy từ vùng KHỎE (Mục 6.4) |
| Test normal lấy ở đâu? | Từ vùng khỏe của riêng vùng NORMAL ([Train|Test|Grey] 0.7/0.9/1.0), KHÔNG lấy normal-sát-hỏng |
| Fault chia sao khi rất hiếm? | 20% train / 80% test, đúng một cụm lỗi thật, chống leakage |
| Augmentation đáng tin? | Chỉ khi có bảng trước/sau trên CÙNG test (results/compare_augmentation.json) |

## 8.3. Checklist trước nộp bài (demo day)

1. File EDA + notebook sạch: plot waveform, spectrum normal vs fault.
2. Feature bảng (địa chỉ cột rõ ràng, đơn vị ghi chú).
3. Model cuối + hyperparams + lý do chọn.
4. Bảng metrics trên test: confusion, precision/recall/F1, PR-AUC.
5. Threshold được chọn có rationale business (đừng nói "vì F1 max" — hãy nói chi phí bỏ sót lỗi bao nhiêu).
6. Baseline vs augmentation so sánh được (chứng minh augmentation có hiệu quả hay không) — dùng trước/sau trên CÙNG test khỏe (như bảng 6.4.2).
7. Dashboards: cảnh báo thời gian thực demo bằng file thực (kể cả khi chạy offline).
8. Pipeline reproducible: script build train/val/test theo thời gian, train, eval một cú.
9. Kể rõ câu chuyện split (normal 3-zone, fault 20/80, scaler fit trên train) — giám khảo chấm điểm phần "hiểu leakage".

---

# Phụ lục A: Bảng công thức nhanh dùng mọi lúc

| Đại lượng | Công thức |
|---|---|
| RMS | `RMS = sqrt((1/N) * sum(xi^2))` |
| std | `std = sqrt((1/N) * sum((xi - mean)^2))` |
| Skewness | `skew = ((1/N) * sum((xi-mean)^3)) / std^3` |
| Kurtosis (dư) | `kurt = ((1/N) * sum((xi-mean)^4)) / std^4 - 3` |
| Crest factor | `crest = peak / RMS` |
| Shape factor | `shape = RMS / mean_abs` |
| Impulse factor | `impulse = peak / mean_abs` |
| Spectral centroid | `C = sum(f_k*P_k) / sum(P_k)` |
| Spectral spread | `S = sqrt( sum(P_k*(f_k-C)^2) / sum(P_k) )` |
| Spectral flatness | `F = (prod(P_k)^(1/K)) / (mean(P_k))` |
| Band energy (dải b) | `E_b = sum(P_k) với k thuộc dải b` |
| Z-score | `z = (x - mu)/sigma` |
| Min-max | `x' = (x - xmin)/(xmax - xmin)` |
| RobustScaler | `x' = (x - median)/IQR` |
| Sigmoid | `p = 1/(1+exp(-z))` |
| Cross-entropy | `-sum_i (y_i*ln p_i + (1-y_i)*ln(1-p_i))/N` |
| Gini | `G = 1 - sum_c p_c^2` |
| EWMA | `E_t = alpha*x_t + (1-alpha)*E_{t-1}` |
| CUSUM | `S_t = max(0, S_{t-1} + (x_t - mu0) - K)` |
| Precision | `Precision = TP/(TP+FP)` |
| Recall | `Recall = TP/(TP+FN)` |
| F1 | `F1 = 2*P*R/(P+R)` |
| Accuracy | `Acc = (TP+TN)/(TP+TN+FP+FN)` |
| MAE | `(1/N)*sum(|y-yhat|)` |
| RMSE | `sqrt((1/N)*sum((y-yhat)^2))` |
| Huber | MSE nếu |err|<delta, ngược lại MAE |

# Phụ lục B: Các từ vựng Anh–Việt qua lại hay gặp

- Predictive maintenance — bảo trì dự đoán
- Fault / Anomaly — lỗi / bất thường
- Window / Stride / Overlap — cửa sổ / bước nhảy / chồng lấp
- Sampling frequency — tần số lấy mẫu
- Nyquist frequency — tần số Nyquist
- Aliasing — biệt danh tần số
- Feature engineering — kỹ thuật trích đặc trưng
- Spectral centroid / spread / flatness — tâm phổ / độ trải phổ / độ phẳng phổ
- Band energy — năng lượng dải tần
- Normalization / Standardization — chuẩn hóa / chuẩn hóa z
- Class imbalance — mất cân bằng lớp
- Confusion matrix — ma trận nhầm lẫn
- Precision / Recall / F1 — độ chính xác dương / độ phủ / F1
- Threshold — ngưỡng
- False alarm — báo động giả
- One-class SVM — SVM một lớp
- Isolation Forest — rừng cô lập
- Reconstruction error — sai số tái dựng
- Data leakage — rò rỉ dữ liệu
- SMOTE — kỹ thuật quá mẫu tổng hợp thiểu số
- Remaining Useful Life (RUL) — tuổi thọ còn lại
- Cross-validation — kiểm định chéo

---

*Hết giáo trình phần 2. Bước tiếp theo: phần 3 về Deep Learning (RNN/LSTM, Autoencoder chi tiết, GAN/diffusion cho augmentation) với nền tảng của giáo trình này.*