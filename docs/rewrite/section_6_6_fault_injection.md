## 6.6. Hướng thứ nhất — Fault Injection / Mô phỏng vật lý: tạo dữ liệu lỗi bằng cách "bơm lỗi"

> **Vị trí:** mục này nằm NGAY sau §6.5 (so sánh GAN vs Diffusion) và trước Chương 7 (sinh dữ liệu để augment lớp hiếm). Nó cùng với §6.7 (sẽ viết sau) tạo thành **cụm "2 hướng sinh dữ liệu lỗi"**:
> - **Hướng 1 (mục này) = Physics-based / fault injection** → DỰNG lỗi theo quy luật vật lý.
> - **Hướng 2 (§6.7) = Generative** → HỌC để sinh lỗi (GAN/TimeGAN/Diffusion, đã học ở §5–§6).
>
> Cả hai cùng giải quyết một deliverable của đề A1: **"bộ sinh dữ liệu lỗi"** — chỉ khác cách làm. Mục này đi sâu hướng thứ nhất, thứ mà cột công nghệ gợi ý của BTC mô tả là **"Mô phỏng vật lý – fault injection"**.

**Mục tiêu:** mục này xây một phương án **KHÔNG cần học từ dữ liệu** để tạo dữ liệu lỗi: ta **dựng chữ ký tín hiệu của từng loại lỗi cơ khí theo đúng quy luật vật lý**, rồi "bơm" chữ ký đó vào tín hiệu bình thường. Kết quả là một bộ sinh dữ liệu lỗi **chạy được trên CPU, trong vài giây, sinh ra hàng loạt mẫu lỗi có nhãn chính xác tuyệt đối (100% là lỗi)** — và quan trọng nhất, ta **điều khiển được CHÍNH XÁC chế độ lỗi** mình muốn (hỏng vòng ngoài, vòng trong, lệch trục, mất cân bằng…). Đây là điều mà chờ máy hỏng thật gần như không thể.

**Khác biệt cốt lõi với Hướng generative (§5–§6):**

| | Hướng generative (GAN/Diffusion) | Hướng physics (mục này) |
|---|---|---|
| Nguồn kiến thức | **Học phân bố lỗi thật** | **Dựng theo công thức cơ học** |
| Cần dữ liệu lỗi thật | Cần (dù ít, để fine-tune) | **Không cần lỗi thật** — chỉ cần hiểu vật lý |
| Độ đa dạng | Rất tốt (học được biến thể) | Trung bình (bám đúng chế độ đã mô hình hoá) |
| Độ chính xác nhãn | Nhãn do mình gán, có thể nhiễu | **Tuyệt đối**: mẫu nào cũng đúng lỗi |
| Kiểm soát chế độ lỗi | Khó (sinh ra "đại khái" giống lỗi) | **Rất cao** (chọn đúng BPFO/misalignment) |
| Tốc độ / chi phí | Trung bình–cao (train model) | **Rẻ, tức thì** |
| Nhược điểm lớn nhất | Dễ cho model học "vẹt" phân bố giả | **Simulation gap**: lỗi giả không trùng 100% lỗi thật |

**Bối cảnh / bài toán con:** khi nào ta phải dùng hướng này? Trả lời bằng ba tình huống thực tế trong A1:

1. **Không có (hoặc quá ít) dữ liệu lỗi thật để generative học.** TimeGAN/Diffusion (§5.4, §6.4) cần một lượng tối thiểu mẫu lỗi để bắt "dáng" của lỗi. Nhưng đề A1 đặt đúng bài toán *"scarcity"*: có thể team chỉ có vài chục mẫu lỗi, thậm chí zero. Với zero, generative **bất lực**; physics thì không cần bất kỳ mẫu lỗi nào. → Là lý do mục này được đặt *trước* §6.7 như "phương án cứu cánh khi chưa có gì".
2. **Muốn kiểm soát CHẾ ĐỘ lỗi.** Generative sinh ra "một dạng lỗi tổng quát". Nhưng khi thuyết trình trước giám khảo, ta muốn nói rõ: "tôi bơm đúng vòng ngoài (BPFO=81 Hz), tôi bơm lệch trục (2×RPM)". Physics cho phép **chỉ định** chế độ ngay từ tham số đầu vào — thứ rất hợp để vẽ "bảng sweep tham số" cho báo cáo.
3. **Muốn nhãn tuyệt đối chính xác từng mẫu.** Dữ liệu fault thật (IMS 90–100%) vẫn có thể lẫn normal-mới-bắt-đầu-xuống-cấp; generative sinh ra có thể "nửa lỗi nửa không". Còn mẫu bơm vật lý: **ta chủ động cộng xung lỗi vào → chắc chắn mang dấu lỗi** → nhãn không thể sai.

Tóm tắt bài toán con mà mục này giải: **"Khi không có lỗi thật để học và/hoặc muốn tạo lỗi đúng chế độ + nhãn sạch 100%, hãy bơm chữ ký lỗi theo công thức cơ học vào tín hiệu normal."**

---

### 6.6.1. Bản chất là gì (từ gốc): tín hiệu = healthy + fault signature + noise

Trước khi học AI, ta cần một mô hình tín hiệu đơn giản nhưng đủ dùng. Dữ liệu rung từ một cảm biến chưa bao giờ "sạch": nó luôn là **tổng** của nhiều thành phần. Hãy nghĩ mỗi thành phần như một "lớp" chồng lên nhau:

```
x_thông(t)   =  x_healthy(t)   +  x_fault(t)   +  n(t)
                └── máy khỏe    ──┘ └── chữ ký lỗi ──┘ └── nhiễu nền ──┘
```

- **`x_healthy(t)`** — nhịp đập "bình thường" của máy: độ rung nền ổn định, chủ yếu là tần số quay vài hài. Đây chính là dữ liệu **normal** mà ta có RẤT nhiều (IMS 0–70%, code dùng làm train normal).
- **`x_fault(t)`** — **chữ ký lỗi**: phần tín hiệu "thừa" do bệnh cơ khí sinh ra (xung lặp khi viên bi lăn qua vết xước, đỉnh 2×RPM khi lệch trục…). Phần này có **hình dạng rất đặc trưng** và lặp lại chu kỳ — chính là thứ ta sẽ *dựng*.
- **`n(t)`** — nhiễu nền ngẫu nhiên (ma sát, điện tử, môi trường).

Trực quan quan trọng nhất: **một lỗi cơ khí không sinh ra "một cơn nhiễu lạ", mà sinh ra một MẪU DAO ĐỘNG lặp lại ở một tần số rất riêng.** Giống như nhịp tim và tiếng "tách" của van đóng mở: nó nhịp nhàng, và tần số của nó nói lên bệnh gì.

Ý tưởng của *fault injection* vì thế gọn như sau: **ta biết trước tần số nhấp nháy của từng bệnh (nhờ vật lý cơ khí), ta tạo ra đúng cái "nhịp tách" đó bằng một hàm toán học, rồi cộng vào tín hiệu normal.** Không cần bất kỳ mẫu lỗi thật nào để làm việc này.

**Vì sao cách này nhanh, rẻ và kiểm soát được?**

- **Nhanh/rẻ:** chỉ là cộng vài mẫu vào mảng numpy → chạy trong mili-giây trên CPU, không cần GPU, không cần train. Trái lại TimeGAN mất hàng chục phút.
- **Sinh được NHIỀU mẫu:** lấy 1000 window normal, mỗi window bơm một chế độ lỗi với tham số khác nhau → có CHỤC NGHÌN mẫu lỗi, nhiều gấp hàng nghìn lần vài mẫu lỗi thật.
- **Kiểm soát chế độ:** chỉ cần đổi số `BPFO` / `amp` / `mod` → sinh ra "chế độ lỗi" khác nhau. Muốn vòng ngoài thì BPFO, muốn lệch trục thì 2×RPM.
- **Đây chính là thứ "chờ máy hỏng thật" không thể làm:** máy hỏng thật là *biến cố ngẫu nhiên, tốn hàng tháng, đắt đỏ, và không ai muốn máy hỏng vì mình cần dữ liệu*. Bơm lỗi thì bấm nút là có.

> **Đừng hiểu lầm:** nói "không cần học" không có nghĩa là **không cần hiểu**. Ngược lại, ta chỉ làm đúng khi hiểu cơ chế hỏng để chọn tham số chuẩn. Thứ "tiết kiệm" ở đây là **không cần mẫu lỗi thật**, chứ **hiểu vật lý thì bắt buộc**.

---

### 6.6.2. Cơ sở vật lý: chữ ký tần số của từng loại lỗi (có công thức)

Đây là "xương sống" của mục. Nếu bạn chưa từng học cơ khí, hãy đọc chậm — mỗi đoạn đi từ *cái gì* → *công thức* → *số cụ thể*.

Đặt ký hiệu chung:
- `f_r` = tần số quay của trục (Hz) = `RPM / 60`. Ví dụ trục quay 1800 vòng/phút → `f_r = 30 Hz`.
- Một ổ bi (ball bearing) gồm: **vòng ngoài** (outer race), **vòng trong** (inner race), **vòng lăn** (cage, giữ khoảng cách các bi), và các **viên bi** (balls). Hình:

```
          ┌─────────────────────────────┐
          │        VÒNG NGOÀI          │
          │   o  o  o  o  o  o  o  o   │   ← các viên bi (số lượng Nb)
          │     (cage / vòng lăn)      │
          │        VÒNG TRONG          │
          └─────────────────────────────┘
```

Khi một bề mặt có vết khuyết (xước/pitting), mỗi lần viên bi lăn **qua vết** đó sẽ sinh ra một **va đập (impact)**. Các va đập này lặp lại với một chu kỳ rất ổn định, và tần số lặp đó được tính nhờ hình học của ổ bi. Cón bốn "tần số đặc trưng" kinh điển (còn gọi là **Bearing Defect Frequencies** — bạn đã gặp ở Chương 2, mục §2.3.1):

| Ký hiệu | Tên đầy đủ | Nghĩa |
|---|---|---|
| **BPFO** | Ball Pass Frequency **Outer** | Vết xước ở **vòng ngoài** → mỗi bi đập vào vết |
| **BPFI** | Ball Pass Frequency **Inner** | Vết xước ở **vòng trong** → mỗi bi đập vào vết |
| **BSF** | **Ball** Spin Frequency | **Viên bi** tự quay quanh trục của nó (vết ở ngay viên bi) |
| **FTF** | Fundamental **Train** Frequency | Vận tốc của **vòng lăn/cage** (vết ở cage) |

#### Công thức (theo hình học ổ bi)

Đặt: `Nb` = số viên bi, `d` = đường kính viên bi, `D` = đường kính vòng lăn (pitch diameter), `φ` = góc tiếp xúc (contact angle), `f_r` = tần số quay trục.

```
BPFO = (Nb / 2) · f_r · (1 - (d/D)·cos φ)          # vòng ngoài (outer race)
BPFI = (Nb / 2) · f_r · (1 + (d/D)·cos φ)          # vòng trong (inner race)
BSF  = (D / 2d)     · f_r · (1 - (d/D)²·cos²φ)     # viên bi (ball spin)
FTF  = (f_r / 2)    · (1 - (d/D)·cos φ)            # vòng lăn / cage
```

**Đọc công thức để hiểu bản chất:**

- `(Nb/2)` và `(1 ± (d/D)·cosφ)`: ổ bi càng nhiều bi, trục càng quay nhanh → bi càng thường xuyên đập vào đúng chỗ khuyết → tần số càng cao.
- `(d/D)·cosφ` là hiệu ứng "bán kính": đường kính viên bi so với đường kính vòng lăn quyết định tỉ lệ mà viên bi "trượt–quay" trên mỗi vòng của trục. Dấu **+** cho BPFI (vòng trong quay nhanh hơn nên bi đập đúng chỗ khuyết nhiều hơn) và dấu **–** cho BPFO (vòng ngoài đứng yên, bi đập ít hơn) — đó là lý do **BPFI > BPFO** khi cùng tham số.
- Nói cách khác: **BPFO < BPFI**, và cả hai đều là bội số-thập-phân (không nguyên) của `f_r`. Chính đặc điểm "không nguyên" này khiến các tần số này **không trùng** với hài của tần số quay → dễ phân biệt trên phổ.

#### Ví dụ số cụ thể

Chọn ổ bi điển hình (giá trị tham khảo như BTC gợi ý): `Nb = 9`, `d/D = 0.4`, `φ = 0°` (góc 0 làm `cosφ = 1`, gọn nhất), trục quay `f_r = 30 Hz` (1800 RPM):

```
BPFO = (9/2)·30·(1 - 0.4·1)            = 4.5·30·0.6 = 81.0  Hz
BPFI = (9/2)·30·(1 + 0.4·1)            = 4.5·30·1.4 = 189.0 Hz
BSF  = (1/(2·0.4))·30·(1 - 0.4²·1)     = 1.25·30·0.84 = 31.5 Hz
FTF  = (30/2)·(1 - 0.4·1)              = 15·0.6 = 9.0  Hz
```

Và **hài bậc hai** của BPFO: `2·BPFO = 162 Hz`, `3·BPFO = 243 Hz` — rất hay xuất hiện và giúp xác nhận lỗi (xem §6.6.7).

> **Giá trị thực tế khác:** nếu bạn dùng ổ SKF-6205 của bộ dữ liệu CWRU (`Nb=9`, `d=0.3125"`, `D=1.537"` → `d/D≈0.203`, `φ=0`), thì với `f_r=30 Hz`:
> ```
> BPFO ≈ (9/2)·30·(1 - 0.203) ≈ 107.5 Hz ;  BPFI ≈ (9/2)·30·(1 + 0.203) ≈ 162.4 Hz
> ```
> Số liệu này hữu ích **khi bạn bơm lỗi vào đúng dữ liệu CWRU**. Việc chọn `d/D` nào chỉ quyết định *con số* kết quả, còn *cấu trúc công thức* và *bản chất "đối đỉnh tại BPFO"* thì không đổi.

#### (a) Mất cân bằng (unbalance) — chữ ký ở **1×RPM**

**Nguyên nhân:** khối lượng không phân bố đối xứng quanh trục quay (bụi bẩn bám lệch, trục không đối xứng). Mỗi vòng quay, điểm nặng "nện" một lần → lực ly tâm đổi chiều đúng `f_r` lần/giây.

**Chữ ký:** **một đỉnh duy nhất rất lớn ở đúng `f_r` (1×RPM)**, thường kèm hài yếu 2×, 3×. Biên độ tỉ lệ với bình phương tốc độ (`∝ f_r²`) — nên máy càng nhanh càng thấy rõ.

```
x_fault(t) = A_u · sin(2π·f_r·t + φ_0)
```

#### (b) Lệch trục (misalignment) — chữ ký ở **2×RPM** (và 1×, 3×)

**Nguyên nhân:** trục nối không đồng tâm (song song lệch hoặc góc lệch) → mỗi vòng quay trục bị "uốn" hai lần (lên-xuống và lần bên kia) → **hai** va đập mỗi vòng.

**Chữ ký:** đỉnh nổi bật ở **`2·f_r` (2×RPM)**, có thể có cả `1·f_r` và `3·f_r`. Đây là dấu hiệu phân biệt với unbalance (1×): nếu 2× hơn hẳn 1× → nghi ngờ **lệch trục**, còn 1× hơn hẳn 2× → nghi ngờ **mất cân bằng**.

```
x_fault(t) = A_m1·sin(2π·f_r·t) + A_m2·sin(2π·2·f_r·t)  + A_m3·sin(2π·3·f_r·t)
```

#### (c) Mòn lan tỏa / độ nhám (diffuse wear) — năng lượng dải tần cao

**Nguyên nhân:** bề mặt bị mòn đều, không còn bén; ma sát tăng, sinh rung "rè" nhiều tần số.

**Chữ ký:** không phải một đỉnh đơn, mà là **"mặt nâng" (noise floor) dải cao tăng** → `spec_centroid` (tâm phổ) **nhích lên**, `spec_spread` tăng, và `spec_flatness` **tăng** (phổ trở nên "phẳng" hơn vì nhiễu broadband). Đây là lý do `src/features.py` có sẵn `spec_centroid`, `spec_flatness`, `spec_spread`: chính là để bắt loại lỗi này. Khi lỗi nặng, năng lượng còn **dịch lên quanh tần số cộng hưởng riêng** của cấu trúc (tạo cụm sideband quanh tần số tự nhiên).

#### (d) Vòng trong bị xước (inner race) — điều chế (modulation)

**Chữ ký đặc biệt:** xung lặp ở **BPFI**, nhưng **biên độ bị chính trục quay "đập" theo chu kỳ** → tạo **sideband** (dải bên) quanh BPFI.

**Vì sao có điều chế?** Vết xước nằm trên vòng trong — mà vòng trong **quay** cùng trục. Khu vực chịu tải chính (load zone) thì đứng yên về phía (thường là hướng trọng lực). Khi vết xước quay **vào** vùng tải → va đập mạnh; khi quay **ra** khỏi vùng tải → va đập yếu. Vậy biên độ xung thay đổi đúng `f_r` lần/giây.

```
x_fault(t) = [1 + m·cos(2π·f_r·t)] · Σ_k h(t - k·T_BPFI)      # m = modulation index
```

Toán học nói rằng nhân một sóng mang với `(1 + m·cos(2π·f_r·t))` sinh ra **hai tần số mới** quanh sóng mang:

```
f_sideband = f_carrier ± f_mod   →  BPFI ± f_r     (ví dụ: 189 ± 30 = 159 và 219 Hz)
```

Tương tự, bất kỳ lỗi nào làm "quẩn" biên độ theo chu kỳ `f_r` đều tạo cặp sideband `f_carrier ± f_r` quanh tần số chính. Đây là "ngôn ngữ" phổ quan trọng nhất trong bảo trì dự đoán — giám khảo rất dễ hỏi vì sao có đỉnh kép quanh BPFI.

**Tổng kết bảng chữ ký (dùng để điền "bảng sweep" trong báo cáo):**

| Loại lỗi | Đỉnh chính | Điều chế / sideband | Feature bị đẩy lên |
|---|---|---|---|
| Mất cân bằng | `1·f_r` | không | RMS, năng lượng dải thấp |
| Lệch trục | `2·f_r` (+1×, 3×) | không | RMS, năng lượng dải thấp-trung |
| Vòng ngoài (BPFO) | BPFO, 2×BPFO, 3×BPFO | ít | `spec_energy`, `kurtosis`, `peak` |
| Vòng trong (BPFI) | BPFI, 2×BPFI | **BPFI ± f_r** | `spec_energy`, `kurtosis`, `peak`, `spec_centroid` |
| Viên bi (BSF) | BSF, 2×BSF | BSF ± FTF | `kurtosis`, `spec_energy` |
| Cage (FTF) | FTF (thấp) | khó thấy | biến thiên chậm |
| Mòn lan tỏa | cụm cao-tần | "mặt nâng" | `spec_flatness` ↑, `spec_centroid` ↑, `spec_spread` ↑ |

**Vì sao xuất hiện đỉnh tại `f`, `2f`, và sideband (điều chế)?** — Một chuỗi xung tuần hoàn bất kỳ, khi phân tích Fourier, luôn bung ra **chuỗi hài** ở `f, 2f, 3f, …` với biên độ giảm dần. Thêm vào đó, xung va đập còn "kích thích" tần số cộng hưởng cơ cấu (thường vài kHz) → các cụm năng lượng quanh tần số cộng hưởng. Còn **sideband** đến từ điều chế biên độ như giải thích ở trên. Khi đọc phổ, ta tìm đúng dấu hiệu này để xác định chế độ lỗi.

---

### 6.6.3. Toán học của việc "bơm": mô hình tín hiệu + ví dụ số

Bây giờ biến ý tưởng thành công thức tính được. Mục tiêu: từ một window `x_h(t)` (normal, thật) → sinh `x_f(t)` (lỗi). Có hai "chiều" làm và đều nên hiểu:

#### Cách 1 — Bơm ở miền TÍN HIỆU (rung thô) [khuyến khích]

Đây là cách "huấn luyện trung thực" nhất: sửa tín hiệu rồi để pipeline trích feature. Mô hình chuẩn của lỗi ổ trục dạng va đập:

```
x_f(t) = x_h(t)  +  A · Σ_k h(t - k·T)  +  n'(t)
                         └── xung tuần hoàn ──┘  └── nhiễu điều khiển ──┘
```

với:
- `T = 1 / f_defect` — khoảng cách giữa hai va đập (s). Ví dụ BPFO = 81 Hz → `T = 1/81 ≈ 0.01235 s`.
- `A` — biên độ va đập (điều khiển mức độ nặng của lỗi).
- `h(u)` — **dạng xung**: một dao động tắt dần, ở tần số cộng hưởng `f_n`:

```
h(u) = e^(-u/τ) · sin(2π·f_n·u)        với u ≥ 0, còn lại = 0
```

- `τ` — hằng số thời gian tắt dần (quyết định xung "đanh" hay "bè").
- `f_n` — tần số cộng hưởng của cấu trúc (vài kHz).
- `n'(t)` — nhiễu trắng có **SNR điều khiển** (thêm để mẫu không giống hệt nhau, và mô phỏng máy thật nhiễu nền).

Khi muốn **điều chế** (vòng trong/BPFI), nhân thêm lớp biên độ:

```
x_f(t) = x_h(t)  +  A · [1 + m·cos(2π·f_r·t)] · Σ_k h(t - k·T)
                              └── m = modulation index (0 → 1) ──┘
```

`m` càng lớn → càng điều chế sâu → sideband `BPFI ± f_r` càng rõ.

> **Tóm tắt 4 nút tham số quan trọng:** `A` (mức nặng lỗi), `T=1/f_defect` (loại lỗi), `m` (điều chế, cho vòng trong), `SNR` (độ sạch của mẫu). Bảng sweep trong báo cáo sẽ quét đúng 4 nút này.

#### Ví dụ số cụ thể (theo số liệu mục 6.6.2)

Chọn lỗi **vòng ngoài**, `BPFO = 81 Hz`, `f_n = 3000 Hz`, `τ = 0.5 ms`, `fs = 12000 Hz`, window `win = 512` mẫu:

```
T     = 1 / 81          ≈ 0.012346 s
samples / chu kỳ        = fs·T = 12000·0.012346 ≈ 148 mẫu
số xung trong 512 mẫu   ≈ 512 / 148 ≈ 3.46 xung   (≈ 3–4 va đập trong window)

u (tính bằng mẫu, mỗi xung kéo dài 10 ms):  h(u) = e^(-u/0.0005)·sin(2π·3000·u)
```

Hiện tượng: mỗi 148 mẫu lại "nện" một cú tắt dần. Trên phổ, energy tập trung thành cụm quanh `3000 Hz` (cộng hưởng) với **các răng cách nhau đúng 81 Hz** — chính là dấu vân tay `BPFO`. Nếu thêm `m = 0.5` (giả vòng trong): xuất hiện cặp răng ở `BPFI ± f_r = 189 ± 30 = 159 Hz & 219 Hz`.

**Sơ đồ ý tưởng (ASCII):**

```
miền thời gian:
x_h(t)  ────────────────  (nhiễu nền nhỏ, ổn định)

bơm xung:
            ┌┐            ┌┐            ┌┐
x_f(t) ─────┘└──────┬─────┘└──────┬─────┘└──────
              k=0   │       k=1   │       k=2
            T=148 mẫu         T=148 mẫu

miền tần số (phổ):
                      ▼ cộng hưởng f_n=3000Hz
   ▂        ▂        ▂        ▂        ▂        ▂     ← các răng cách nhau 81 Hz
       ___________________▂▂▂▂▂▂▂▂___________________
       0     81    162    243   ...      3000     (Hz)
```

#### Cách 2 — Bơm ở miền FEATURE (đã có 16 feature)

Nhanh hơn nữa: lấy window normal **đã trích** 16 feature (khớp `src/features.py`), rồi **đẩy đúng các feature** phản ánh chữ ký lỗi. Ví dụ với lỗi va đập:

```
kurtosis  += Δkurt     (xung → độ nhọn tăng)
crest_factor += Δcrest (peak / RMS tăng → "đỉnh" nổi lên)
spec_energy  *= (1 + ΔE)   (năng lượng phổ tăng)
spec_centroid += Δfc       (năng lượng dồn lên tần số cao)
```

**Nhược điểm của cách 2:** ta tự tay co giãn feature nên dễ "sai vật lý" (vd đẩy spec_energy lên nhưng quên đẩy tỉ lệ đúng với kurtosis). Model có thể học một "cụm dị thường phi tuyến" không tương ứng kết cấu lỗi thật. **Khuyến nghị:** ưu tiên **Cách 1** (bơm tín hiệu → pipeline tự trích feature) vì nó bảo toàn mối tương quan vật lý giữa các feature; Cách 2 chỉ dùng khi muốn sinh cực nhanh và không cần độ thật cao. Team đang đi theo hướng **feature-based** (chỗ `--gen npy` đọc raw features cho `src/features.py`) — nên việc chọn Cách 1 là hợp lý: nó cho ra raw features chuẩn hoá bằng đúng scaler của train.

---

### 6.6.4. Áp dụng trong code/project của team

Hiện `scripts/02_compare_augmentation.py` chỉ có `--gen heuristic|npy`:

- `heuristic` = bóp biên độ 2.5× + nhiễu (rất thô, chỉ để chạy thử — xem `_heuristic_faults`).
- `npy` = đọc lỗi giả từ TimeGAN (`--gen-npy results/synthetic_faults.npy`).

**Đề xuất thêm `--gen physics`** — một nguồn lỗi giả **thứ ba**, mạnh hơn `heuristic` vì bám đúng vật lý (xung BPFO/BPFI, sideband, lệch trục). Kết quả: script so sánh **3 bộ sinh** (heuristic / physics / npy=TimeGAN) → bảng "2 bộ sinh, cái nào tốt hơn" mà SPEC yêu cầu. Bổ sung vào docstring khối chọn gen:

```python
if args.gen == "heuristic":
    X_syn = _heuristic_faults(Xn_train, rng, n_rows=6 * n_fault)
elif args.gen == "physics":
    # (mới) bơm lỗi vào TÍN HIỆU normal rồi mới trích feature → giữ đúng tương quan vật lý
    X_syn = _physics_faults(Xn_train, rng, d, n_rows=6 * n_fault)
    # X_syn là RAW features (16 cột) → z-score bằng scaler train (như nhánh npy)
    X_syn = ft.zscore(X_syn, d["scaler_mu"], d["scaler_sd"])
```

**Dữ liệu nguồn:** tín hiệu normal lấy từ **IMS 0–70% đầu** (= vùng khỏe mà `pipe.split_and_scale` đã cắt cho `Xn_train`). Mẹo: vì `Xn_train` đã là **feature**, để bơm ở miền tín hiệu ta cần lấy lại **window raw normal** — cụ thể: (i) dùng `windows_to_features` chiếu ngược? Không — đơn giản nhất là load raw một lần nữa trong `_physics_faults` (gọi `ds.make_windows` trên chính `runs_normal` train), hoặc (ii) giữ một bản `windows_train` song song ở bước split. Khi gộp vào `pipeline.split_and_scale`, hãy trả thêm `d["windows_normal_train"]` để script 02 dùng lại.

**Nguyên tắc bất biến (bắt buộc giữ):**

- Sinh lỗi giả **CHỈ trong train** — chống leakage. Test (cả normal-khỏe 0.70–0.90 và fault thật 90–100%) **không bao giờ** dùng mẫu bơm hay scaler riêng.
- Tuân thủ **3-zone split** trong `src/pipeline.py`: train normal = `[0, 0.70]` của vùng NORMAL mỗi run; test normal = `[0.70, 0.90]`; grey = `[0.90, 1.0]` → loại. Fault thật thì **20/80** (train/test). Z-score fit **trên train** (`fit_zscore`) rồi áp cho mọi thứ.
- Khi thêm `--gen physics`, mẫu sinh cũng phải được **z-score bằng `d["scaler_mu"]`/`d["scaler_sd"]`** (giống nhánh `npy`), tuyệt đối không fit scaler riêng.

**Bảng "sweep" cho báo cáo (điền con số sau khi chạy):**

| Tham số | Giá trị quét | Ý nghĩa/ghi chú |
|---|---|---|
| `A` (biên độ) | 0.2 / 0.5 / 1.0 / 2.0 | lỗi nhẹ → nặng |
| `f_defect` | BPFO / BPFI / 2×RPM | chọn loại lỗi |
| `m` (modulation) | 0 / 0.3 / 0.6 | 0 = không điều chế (vòng ngoài), >0 = vòng trong |
| `SNR` (dB) | 20 / 40 / ∞ | độ sạch của mẫu sinh |
| `n_rows` | 2× / 4× / 6× số lỗi thật | tỉ lệ tăng cường |

---

### 6.6.5. Ví dụ mã Python (giải thuật bơm lỗi)

Chỉ trình bày **giải thuật**, không yêu cầu chạy. Hàm nhận tín hiệu khỏe + tham số → trả tín hiệu có lỗi:

```python
import numpy as np

def damped_impulse(u, fn=3000.0, tau=0.0005):
    """Dạng xung va đập tắt dần: h(u)=exp(-u/tau)*sin(2*pi*fn*u), u>=0."""
    u = np.maximum(u, 0.0)
    return np.exp(-u / tau) * np.sin(2 * np.pi * fn * u)

def inject_bearing_fault(sig, fs, f_defect, amp=0.5, fn=3000.0, tau=0.0005,
                         m=0.0, f_rot=0.0, noise_snr=None, rng=None):
    """Bơm chữ ký lỗi ổ trục vào tín hiệu normal `sig`.

    sig      : (n,) tín hiệu healthy (window bình thường).
    f_defect : tần số va đập, Hz (BPFO/BPFI/BSF — xem 6.6.2).
    amp      : biên độ va đập `A`.
    fn/tau   : tần số cộng hưởng & hằng số tắt dần của xung h(u).
    m        : modulation index; >0 khi mô phỏng vòng trong / điều chế.
    f_rot    : f_r (Hz), cần khi m>0 để tạo sideband BPFI±f_r.
    noise_snr: nếu là số (dB) thì thêm nhiễu trắng với SNR này; None = không thêm.
    """
    n = len(sig)
    t = np.arange(n) / fs
    x_f = sig.astype(float).copy()

    T = 1.0 / f_defect                     # chu kỳ giữa các va đập (s)
    period_samples = int(round(T * fs))    # số mẫu mỗi chu kỳ
    impulse_len = int(round(0.010 * fs))   # xung kéo dài 10 ms

    k_starts = np.arange(0, max(1, n - 1), period_samples)   # các mốc bắt đầu xung
    for k0 in k_starts:
        L = min(impulse_len, n - k0)
        if L <= 0:
            continue
        u = np.arange(L) / fs
        h = amp * damped_impulse(u, fn, tau)          # xung thô
        if m > 0.0:                                    # điều chế biên độ theo f_rot
            t_local = (k0 / fs) + u
            h *= 1.0 + m * np.cos(2 * np.pi * f_rot * t_local)
        x_f[k0:k0 + L] += h                            # cộng chữ ký vào tín hiệu

    if noise_snr is not None and rng is not None:
        # thêm nhiễu trắng có SNR điều khiển: P_sig/P_noise = 10^(SNR/10)
        P_sig = np.mean(x_f ** 2)
        P_noise = P_sig / (10 ** (noise_snr / 10))
        x_f += rng.normal(0, np.sqrt(P_noise), n)
    return x_f

# Ví dụ dùng trong pipeline train (chỉ train — TÁCH khỏi test):
# windows_healthy = <raw windows normal, lấy từ vùng [0,0.70] mỗi run>
#   for sig in windows_healthy:
#       sig_fault = inject_bearing_fault(sig, fs=FS, f_defect=BPFO, amp=0.5,
#                                        fn=3000, tau=0.0005, m=0.0)
#       feats = ft.raw_features(make_windows(sig_fault, win, stride), fs=FS)
```

Một số điểm kỹ thuật trong code, hãy tự giải thích khi thuyết trình:

- **Bơm vào tín hiệu rồi MỚI trích feature** (`ft.raw_features`) — không tự co feature, nên mối quan hệ vật lý giữa các feature được bảo toàn.
- **Tham số `m` mô phỏng vòng trong/sideband** — đây là điểm "ăn tiền" vì chứng tỏ bạn hiểu điều chế (mục 6.6.2d).
- **Inspect chéo**: sau khi sinh, hãy in `spec_energy`, `kurtosis`, `spec_centroid` của mẫu bơm và so với mẫu normal → xác nhận feature thực sự dịch chuyển đúng hướng.

---

### 6.6.6. Ưu / nhược + so sánh nội bộ

**Ưu điểm:**

1. **Nhanh + rẻ:** mili-giây trên CPU, không cần GPU, không cần huấn luyện.
2. **Kiểm soát chế độ lỗi chính xác:** chỉ định BPFO/BPFI/lệch trục/1× qua tham số → dễ minh hoạ, dễ thuyết trình, dễ lập "bảng sweep".
3. **Nhãn tuyệt đối:** mẫu bơm nào cũng chắc chắn là lỗi, không "nửa normal nửa lỗi".
4. **Sinh hàng loạt:** lấy N window normal, bơm N lần với tham số khác nhau → chục nghìn mẫu.
5. **Hiểu cơ chế → thuyết trình mạnh:** bạn nhìn phổ tự chỉ ra đỉnh BPFO/sideband — giám khảo ấn tượng vì bạn *hiểu lỗi*, không chỉ chạy code.

**Nhược điểm (phải trung thực):**

1. **Simulation gap:** mô hình `h(u)` của bạn chỉ *xấp xỉ* lỗi thật. Lỗi thật còn chứa phi tuyến (ma sát, biến dạng đàn hồi, tải trọng) mà bạn không mô hình hoá hết được → mẫu sinh **không trùng 100%** lỗi thật, đôi khi lệch khá xa.
2. **Model học quá khớp vào mô phỏng:** nếu classifier chỉ thấy lỗi giả "quá sạch, quá đều", nó có thể `overfit` quy luật mô phỏng của ta → khi chạm lỗi thật lộn xộn ở test lại kém. Đây chính là rủi ro ngược với mục đích.
3. **Cần hiểu vật lý:** chọn sai `d/D`, `Nb`, `τ`, `f_n` → tần số/chữ ký sai → bơm ra "lỗi" không khớp thực tế. Sai ở bước này là sai nền tảng.
4. **Đa dạng hạn chế hơn generative:** bạn mô phỏng được *các chế độ đã biết*, khó tạo *biến thể mới mẻ* mà mình chưa nghĩ ra.

**Vì sao vẫn cần generative (+ dữ liệu thật) để "chốt"?**

Physics cho ta lượng **nhưng có thể thiếu chất** (quá sạch, thiên lệch). Generative/Dữ liệu thật cho chất. Chiến lược đúng là kết hợp:

```
Bơm vật lý  →  sinh NHIỀU mẫu (về lượng)  →  mở rộng "biên" của class lỗi
Dữ liệu thật/TimeGAN → neo "chất": giữ phân phối gần bản chất lỗi thật
Huấn luyện: normal + lỗi thật (20%) + lỗi bơm + lỗi sinh  →  đánh giá trên lỗi THẬT
```

Trong A1, ta dùng physics làm **cột đối chiếu** ("cái nào tốt hơn" giữa bơm và generative), còn **TimeGAN làm "nhân vật chính"** (đúng keyword đề, có số liệu end-to-end). Physics + TimeGAN là bộ đôi: **số lượng** + **chất lượng**.

---

### 6.6.7. Bẫy / lưu ý thực hành (rất dễ mất điểm)

1. **KHÔNG bơm lỗi vào trong TEST.** Test chỉ dùng **lỗi thật chưa thấy** + normal khỏe. Mọi augmentation, scaling, chọn ngưỡng, chọn feature đều nằm trong **train**. Bơm vào test = data leakage = điểm hỏng.
2. **Simulation gap — đừng coi mẫu bơm là "fault thật" để đánh giá.** Mẫu bơm chỉ để *train*, không bao giờ để *test*. Nếu bạn đo AUC trên lỗi bơm, kết quả sẽ "ảo" (quá đẹp) vì mẫu do chính bạn sinh.
3. **Lỗi tần số Naive — dùng SAI `fs`.** Tần số đặc trưng tính bằng Hz, nhưng khi chuyển sang "số mẫu mỗi chu kỳ" bạn dùng `T_in_samples = fs / f_defect`. Nếu `fs` sai, bạn bơm xung sai tần số → chữ ký không còn là BPFO nữa. **Đặc biệt:** `src/pipeline.py` mặc định `FS = 12000` (khớp CWRU **12 kHz** dưới tải), nhưng **IMS thật là 20 kHz**. Nếu team dùng `--dataset ims`, hãy chỉnh `fs=20000` trước khi bơm và trích feature, nếu không con số BPFO sẽ lệch.
4. **Xác minh phổ (FFT) của lỗi giả có đúng đỉnh BPFO không.** Trước khi đưa vào train, phải vẽ/check phổ: có răng tại đúng `BPFO`, `2·BPFO`… không? Có sideband `BPFI ± f_r` (nếu khai báo vòng trong)? Một hàm `_assert_spectrum(x_f, fs, f_defect)` so sánh đỉnh tìm được với `f_defect` (sai số vài Hz) là đáng có. Sai sót ở đây phá toàn bộ thí nghiệm mà ta không nhận ra.
5. **Cửa sổ quá ngắn → đỉnh phổ mờ.** `win=512`, `fs=12000` → độ phân giải tần số `fs/win = 23.4 Hz/bin`. BPFO=81 Hz nằm **giữa** bin 3 (70.3 Hz) và bin 4 (93.8 Hz) → đỉnh bị "loang" mờ, khó tách. Muốn thấy rõ BPFO hãy dùng window dài hơn (vd 2048 mẫu → ~5.9 Hz/bin), hoặc tính phổ trên window dài rồi mới lấy feature (chú ý bơm cả khi cắt window).
6. **Đừng để model học nền nhiễu mô phỏng mà không có ở thực tế.** Nếu bạn cộng `noise_snr` kiểu gaussian vào mẫu bơm, mà lỗi thật không có nhiễu kiểu đó, model có thể học "dấu hiệu là có gaussian" thay vì "dấu hiệu là lỗi". Hãy để mức nhiễu của mẫu bơm **thấp & giống mức nhiễu của normal thật**, hoặc thêm nhiễu từ một bản normal thật (bootstrap nhiễu) thay vì gaussian thuần.
7. **Nhãn mẫu bơm đều là 1 (fault)** — đừng quên `y_syn = np.ones(...)`, và đừng đưa nhầm mẫu normal vào mảng synthetic.

---

### 6.6.8. Kết nối với phần còn lại của giáo trình

- **Chương 2 — đặc trưng miền tần số (mục §2.3.1, §2.3.6):** nơi giới thiệu khái niệm **BPFO/BPFI/BSF/FTF** và "đỉnh ở 1×, 2×"; mục này mở rộng thành **công thức tính** + **hướng bơm chữ ký**. Hãy đọc lại §2.3 để nhớ tần số đặc trưng là gì trước khi vào đây. `src/features.py` (16 feature, có `spec_centroid`, `spec_flatness`, `spec_energy`…) chính là "cầu nối" để chuyển từ đồ thị vật lý sang số liệu model.
- **§6.5 — so sánh GAN vs Diffusion:** đây là "anh em" của mục này — cả hai đều trả lời *làm gì khi thiếu lỗi*. §6.5 bàn về hai bộ sinh generative; §6.6 bổ sung hướng **vật lý** để có cái nhìn toàn diện (generative + physics) trước khi sang Chương 7.
- **§6.7 (kế tiếp) — hướng generative:** mục này cùng §6.7 tạo cụm "2 hướng sinh dữ liệu lỗi". Nếu §6.6 là "dựng lỗi theo công thức", thì §6.7 là "học để sinh lỗi". Trong Chương 7, hai hướng sẽ được **hợp nhất** vào một pipeline augment + đánh giá.
- **Cây bài toán con của Phần 3 (đầu chương, nhánh "Sinh dữ liệu lỗi giả"):** nhánh đó đang liệt kê GAN/TimeGAN/Diffusion; hãy **bổ sung một nhánh con** cho "Physics-based / fault injection" để cây phản ánh đúng 2 hướng: `(a) dựng lỗi theo vật lý (§6.6)` và `(b) học phân bố để sinh lỗi (§5–§6, §6.7)`.
- **SPEC.md / sheet của BTC:** cột **"Mô phỏng vật lý – fault injection"** trong "CÔNG NGHỆ GỢI Ý" — đây chính là cột mà mục này cover. Khi nộp, đây là một điểm "đủ bộ": bạn có cả **physics** (mục này) lẫn **generative** (§5–§6) để đối chiếu, thay vì chỉ lo một đường.
