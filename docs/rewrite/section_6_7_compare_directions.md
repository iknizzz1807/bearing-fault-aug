## 6.7. Đối chiếu hai hướng sinh dữ liệu lỗi (vật lý vs generative) & cột "công nghệ gợi ý"

### 6.7.1. Mục tiêu

**Mục tiêu:** sau khi đã học xong Hướng 1 (mô phỏng vật lý / bơm lỗi) ở mục 6.6 và Hướng 2 (generative: GAN/TimeGAN/Diffusion) ở mục 5–6, mục này chốt **khi nào chọn hướng nào** và **vì sao team nên đi CẢ HAI nhưng lệch ưu tiên** (khoảng 60% generative, 20% vật lý, 0% diffusion ngay lúc này). Đây là "nút thắt" vì hai hướng **không đối nghịch** mà **bổ sung** cho nhau; hiểu sai ở đây sẽ dẫn tới hai lỗi ngược nhau: hoặc "bỏ mất độ thật" (chỉ chăm mô phỏng vật lý) hoặc "không kiểm soát được câu chuyện" (chỉ chăm generative).

### 6.7.2. Bối cảnh / bài toán con

Độc giả vừa đi hết hai con đường dài:

- **Hướng 1 (vật lý / fault injection):** hiểu cơ chế hỏng (BPFO/BPFI/BSF/FTF, lệch trục, sideband, mài mòn lan tỏa) → "bơm" lỗi theo công thức vật lý vào tín hiệu normal. Trong repo hiện tại chỉ mới là heuristic (`scripts/02 --gen heuristic`: nhân biên độ 2.5 + nhiễu Gaussian), chưa phải vật lý đúng nghĩa.
- **Hướng 2 (generative):** học phân bố lỗi thật (dù chỉ vài mẫu) rồi sinh lỗi giả giống thật. TimeGAN (chạy được trên CPU, `scripts/03` → `results/synthetic_faults.npy`), Diffusion (cần GPU, chỉ là ghi chú lý thuyết).

Vấn đề là **cả hai đều trả lời cùng một câu hỏi** — "làm sao có thêm lỗi giả để train supervised?" — nên dễ nhầm là "chọn 1 trong 2". Mục này làm rõ chúng dùng cho **hai vai trò khác nhau** trong cùng một pipeline: một bên cho **độ bao phủ + kiểm soát + kể chuyện**, một bên cho **độ bám phân bố thật**. Đồng thời mục này đối chiếu với đúng cột **"CÔNG NGHỆ GỢI Ý"** của BTC: `Mô phỏng vật lý - GAN/TimeGAN - Diffusion - fault injection - PyTorch` — nó **gợi ý cả hai cụm** chứ không chọn cụm nào, nên đi một mình là "lệch khỏi kỳ vọng của đề".

Để "cái nhìn tổng" rõ ràng, hãy so sánh ngay **đầu vào – cơ chế – đầu ra** của hai hướng:

| Hướng | Đầu vào | Cơ chế | Đầu ra (dữ liệu lỗi giả) |
|---|---|---|---|
| **Vật lý** | Cấu hình máy (số bi $n$, đường kính bi $d$, đường kính vòng $D$, góc tiếp xúc $\phi$, tần số trục $f_r$) + 1 tín hiệu normal | Công thức tần số đặc trưng → điều chế/nhân xung rồi **cộng chồng** vào tín hiệu khỏe | Lỗi giả **có nhãn chế độ** (BPFO nặng độ 3, lệch trục 2×…) và **đúng tần số** mặc dù có **simulation gap** |
| **Generative** | 1 vài mẫu lỗi **thật** (vài chục window) | GAN/TimeGAN/Diffusion **học phân bố** rồi lấy mẫu (sample) | Lỗi giả **giống phân bố lỗi thật** nhưng **không đặt tên được chế độ**, đôi khi trùng lặp (mode collapse) |

Cách nhớ nhanh: **vật lý trả lời "lỗi này XẢY RA THẾ NÀO"**, còn **generative trả lời "lỗi này TRÔNG RA làm sao"**. Hai câu hỏi khác nhau → hai vai trò khác nhau, không loại trừ.

### 6.7.3. Bảng so sánh tổng hợp hai hướng

| Tiêu chí | Hướng 1 — Vật lý / fault injection | Hướng 2 — Generative (GAN/TimeGAN/Diffusion) |
|---|---|---|
| **Cơ chế** | Dựng theo **công thức vật lý** cơ chế hỏng → bơm vào tín hiệu normal | **Học phân bố** lỗi thật → sinh mẫu mới theo phân bố đó |
| **Cần dữ liệu lỗi thật nhiều?** | **Cực ít** — có thể 0 mẫu lỗi (chỉ cần hiểu cơ chế + cấu hình máy) | Từ **ít đến vừa** — cần vài chục/trăm mẫu lỗi để học phân bố (subset > 0) |
| **Chất lượng bám bản chất lỗi** | Đúng **cơ chế tần số** (đỉnh BPFO/BPFI đúng, sideband đúng) nhưng bỏ sót hiện tượng "không nằm trong mô hình" | Bám **phân bố tổng thể** của lỗi thật, kể cả hiện tượng khó mô tả bằng công thức |
| **Độ "thật" so với lỗi thực (simulation gap)** | **Tồn tại gap** — lỗi mô phỏng không hệt lỗi thực (nhiễu nền, chế độ chạy, già hóa phụ thuộc máy) | **Nhỏ hơn** — học chính từ lỗi thật nên trung thực với dữ liệu thực hơn; nhưng vẫn có "mode collapse" (GAN) |
| **Tốc độ / chi phí** | **Rất nhanh, rẻ** — vài dòng công thức, chạy CPU tức thì | **Chậm hơn** — train model, iteration, tốn thời gian điều chỉnh |
| **Yêu cầu tài nguyên (GPU?)** | **Không cần** — thuần CPU/numpy | TimeGAN: **chạy được trên CPU** (nhưng chậm). Diffusion: **bắt buộc GPU** |
| **Mức độ kiểm soát chế độ lỗi** | **Tuyệt đối** — biết trước mẫu này là BPFO hay lệch trục, bơm đúng khẩu phần | **Thấp** — model tự sinh, khó chỉ định "tạo đúng kiểu lỗi X" |
| **Nhãn chính xác từng mẫu** | **100% đúng** — mẫu nào cũng mang nhãn lỗi + tên chế độ rõ ràng | **Đúng về nhãn "lỗi"** nhưng mẫu sinh có thể "rác" (mode collapse, trùng lặp, không thực sự là lỗi hữu ích) |
| **Rủi ro chính** | **Simulation gap** (lỗi giả quá "sạch"/"đúng mô hình" → model học nhầm) | **Mode collapse** (GAN sinh lặp một kiểu) / **overfit vào vài mẫu lỗi thật** / **data leakage** nếu không cô lập test |
| **Phù hợp khi nào** | Khi hiểu rõ cơ chế hỏng, muốn **kiểm soát + kể chuyện** + **nhãn sạch** + **rẻ** | Khi có **một ít lỗi thật** và muốn **bám phân bố thật**, linh hoạt với chế độ lỗi khó mô tả |

### 6.7.4. Vì sao hai hướng bổ sung nhau, không loại trừ

Hai hướng bổ sung vì chúng mạnh ở **những chiều ngược nhau** của cùng một vấn đề:

- **Vật lý** mạnh ở **độ bao phủ chế độ lỗi + khả năng kiểm soát + chi phí rẻ + nhãn sạch.** Nó nói được chính xác "đây là lỗi BPFO nặng độ 3" — thứ mà generative gần như không làm được (generative sinh ra "một cái gì đó giống lỗi" nhưng không nói được tên chế độ).
- **Generative** mạnh ở **độ trung thực với phân bố thật + mềm dẻo.** Có những hiện tượng lỗi (nhiễu nền, độ già hóa, chế độ tải phụ thuộc máy) mà ta **không thể viết công thức** — vật lý mù chỗ này, generative học được.

**Kết hợp chúng theo vai trò khác nhau, không thay thế nhau:**

1. **Dùng vật lý sinh lớp lỗi "đã biết cơ chế"** — ví dụ xung BPFO/BPFI, sideband quanh trục chính. Lớp này cần độ **chính xác tần số + nhãn chế độ** để minh chứng khoa học. Với ổ bi, tần số đặc trưng tính theo cấu hình: $f_{BPFO}=\frac{n}{2}\left(1-\frac{d}{D}\cos\phi\right) f_r$ (lỗi vòng ngoài), $f_{BPFI}=\frac{n}{2}\left(1+\frac{d}{D}\cos\phi\right) f_r$ (lỗi vòng trong), còn lệch trục/không cân bằng cho đỉnh tại $1\times f_r, 2\times f_r$ kèm **sideband** quanh chúng. Đây là nơi vật lý "bất khả chiến bại" về độ chính xác tần số.
2. **Dùng generative học phần "chưa hiểu rõ"** — phần lỗi thật mà con người khó mô tả bằng công thức (nhiễu nền theo máy, già hóa phi tuyến, chế độ tải biến đổi). Chính TimeGAN/Diffusion học bù các chiều mà mô hình vật lý bỏ sót.

**Vì sao không thể thay thế nhau?** Nếu chỉ dùng vật lý, model học lỗi giả **quá sạch/đúng công thức** → sai lệch nặng khi gặp lỗi thật có nhiễu thực (simulation gap). Ngược lại, nếu chỉ dùng generative với **rất ít** lỗi thật, model chỉ học được **phân bố hẹp** của vài mẫu đó → thiếu đa dạng chế độ lỗi và dễ mode collapse. Ghép hai bộ sinh → model thấy được **cả "phân bố thật" lẫn "danh mục chế độ + độ sạch"** → robust hơn.

**Ví dụ kết hợp thực tế trong repo:** nâng `--gen heuristic` (chỉ nhân biên độ 2.5 + nhiễu) thành `--gen physics` (bơm xung BPFO/BPFI/sideband theo công thức trên) để tạo **cột đối chiếu**; song song giữ `--gen npy` (TimeGAN sinh từ lỗi thật) làm **bộ sinh chính**. Kết quả là hai bộ dữ liệu lỗi:
- Bộ "physics" → kiểm soát được chế độ, nhãn sạch, kể chuyện khoa học.
- Bộ "generative" → bám phân bố thật, chạy được trên CPU, đúng keyword của đề.

Chạy `scripts/02 --gen physics` và `--gen npy` trên cùng một test (3-zone split, test chỉ nhận lỗi thật chưa thấy) → ta có **bảng so sánh trước/sau cho từng bộ sinh**, chứng minh "bộ nào giúp recall/F1 tăng nhiều hơn" — chính là deliverable mà giám khảo cần.

### 6.7.5. Đối chiếu cột "công nghệ gợi ý" của BTC (quan trọng)

Cột BTC: `Mô phỏng vật lý - GAN/TimeGAN - Diffusion - fault injection - PyTorch`. Phân rã từng thuật ngữ:

| Thuật ngữ | Thuộc hướng | Vai trò trong pipeline của team | Trạng thái code hiện có |
|---|---|---|---|
| **Mô phỏng vật lý** | Hướng 1 (vật lý) | Mô hình cơ chế hỏng (xung BPFO/BPFI, sideband, lệch trục) → sinh lỗi giả "có cơ sở" | **Chưa có** — mới dạng heuristic (`--gen heuristic` nhân biên độ + nhiễu, `scripts/02`) |
| **fault injection** | Hướng 1 (vật lý) | "Bơm" lỗi vào tín hiệu normal theo chế độ đã chọn, nhãn 100% chính xác | **Chưa có** — heuristic ở `scripts/02` chưa đúng vật lý |
| **GAN/TimeGAN** | Hướng 2 (generative) | Học phân bố lỗi thật hiếm → sinh lỗi giả. **Trụ chính** của team | **Có** — `scripts/03_train_timegan.py`, `src/timegan_torch.py` → `results/synthetic_faults.npy` |
| **Diffusion** | Hướng 2 (generative) | Hướng nâng cao đạt chất lượng tần số cao hơn, giữ đúng BPFO/BPFI | **Chỉ lý thuyết** — `src/Diffusion-TS` (clone, chưa chạy), **bắt buộc GPU** |
| **PyTorch** | Framework (không phải hướng) | Nền tảng triển khai TimeGAN / (sau này) Diffusion | **Có** — `src/timegan_torch.py` |

**Kết luận từ bảng trên:**
- Team đã cover **khoảng một nửa cột** — phần generative: `GAN/TimeGAN` + `PyTorch` (chạy được, có số liệu). Phần còn thiếu là **mảng vật lý** (`Mô phỏng vật lý` + `fault injection`).
- Muốn **bám sát 100% gợi ý** của BTC → cần bổ sung mảng vật lý, tức nâng `--gen heuristic` → `--gen physics` (fault injection có cơ sở vật lý) dùng làm **cột đối chiếu** — đúng kế hoạch trong `SPEC.md` (khoảng 20% ưu tiên).
- Đồng thời cột này cho thấy đề cũng kỳ vọng **"mô tả chế độ lỗi"** (một phần **input** của A1): khi ta bơm lỗi BPFO/sideband, ta buộc phải mô tả rõ chế độ hỏng → đây là điểm cộng khi trả lời đề.

### 6.7.6. Bản đồ / cây quyết định chọn hướng

Dạng "NẾU … THÌ …" — trả lời tuần tự để chọn chiến lược:

```
BẮT ĐẦU: cần thêm "lỗi giả" để augment lớp hiếm
│
├─ (1) Có hiểu đầy đủ cơ chế lỗi (BPFO/BPFI/sideband...)?
│     ├─ YES → ưu tiên VẬT LÝ (kiểm soát + nhãn sạch + rẻ)
│     └─ NO  → ưu tiên GENERATIVE (học phần "chưa hiểu rõ")
│
├─ (2) Có GPU + đủ thời gian không?
│     ├─ NO  →  VẬT LÝ (CPU tức thì) + TimeGAN (chạy được trên CPU)   ← khớp team này
│     └─ YES →  Có thể thử DIFFUSION (chất lượng tần số cao hơn)
│
├─ (3) Muốn kiểm soát chính xác chế độ lỗi để KỂ CHUYỆN / trình bày?
│     ├─ YES →  VẬT LÝ (bơm đúng chế độ, đặt tên chế độ rõ ràng)
│     └─ NO  →  GENERATIVE (không cần chỉ định chế độ)
│
├─ (4) Muốn độ trung thực với dữ liệu thật (khi có vài lỗi thật)?
│     ├─ YES →  GENERATIVE (học chính từ lỗi thật)
│     └─ NO  →  VẬT LÝ (không phụ thuộc dữ liệu lỗi thật)
│
└─ (5) Cần nhãn tuyệt đối từng mẫu (báo cáo sạch)?
      ├─ YES →  VẬT LÝ (nhãn 100% đúng + tên chế độ)
      └─ NO  →  GENERATIVE (nhãn "lỗi" đúng nhưng mẫu có thể "rác")
```

Bảng tóm tắt quyết định (đọc nhanh, song song với cây ở trên):

| Tình huống của team | Chọn hướng chính | Chọn hướng phụ/bổ sung |
|---|---|---|
| Có hiểu cơ chế hỏng, muốn kiểm soát & kể chuyện | Vật lý | Generative để bù độ thật |
| Có vài lỗi thật, muốn bám phân bố | Generative | Vật lý để kiểm soát chế độ |
| Không có GPU | **Vật lý + TimeGAN** (CPU được) | — (bỏ Diffusion) |
| Có GPU + thời gian | Generative (Diffusion) | Vật lý để check cơ chế |
| Muốn nhãn sạch từng mẫu để báo cáo | Vật lý | Generative bù chiều chưa hiểu |
| Muốn đa dạng chế độ lỗi (nhiều loại hỏng) | Vật lý (mỗi chế độ 1 bộ bơm) | Generative bồi thêm biến thể |

**Tóm tắt thành một câu quyết định:** không có câu trả lời "chọn hẳn một hướng". Team chọn **generative làm trụ** (để có độ bám phân bố thật + đúng keyword đề + chạy được trên CPU) và **vật lý làm lớp so sánh** (để check cơ chế + kiểm soát chế độ + kể chuyện), bởi team **không có GPU** (câu 2 trả lời NO) và **muốn cả hai** (câu 1, 3, 4, 5 trả lời YES theo từng vai trò).

### 6.7.7. Khuyến nghị chiến lược cho team (khớp SPEC.md)

Chiến lược đề xuất — **trụ chính + lớp so sánh + ghi chú tương lai**, khớp `SPEC.md`:

| Hạng | Thành phần | Tỷ trọng | Việc cụ thể | Lý do |
|---|---|---|---|---|
| **1** | **TimeGAN / generative** | ~60% | Giữ `scripts/03_train_timegan.py` → `results/synthetic_faults.npy` → `scripts/02 --gen npy`. Chạy 3000+ iteration khi có thời gian | Đúng keyword đề ("GAN/TimeGAN"), chạy được trên CPU, có **số liệu thật** (`results/compare_augmentation.json`), dễ bảo vệ trước giám khảo |
| **2** | **Fault injection vật lý** | ~20% | Nâng `--gen heuristic` → `--gen physics`: bơm xung BPFO/BPFI/sideband theo công thức, nhãn chế độ rõ ràng. Dùng làm **cột đối chiếu** với bộ generative | Cover nốt phần "Mô phỏng vật lý - fault injection" trong cột gợi ý; tạo góc "2 bộ sinh, cái nào tốt hơn" cho slide; bổ sung "mô tả chế độ lỗi" (input của A1) |
| **3** | **Diffusion** | ~0% giờ | Chỉ ghi trong báo cáo là "hướng nâng cao đạt chất lượng tần số cao hơn"; **đừng đầu tư** vì cần GPU | Không khả thi với tài nguyên hiện tại |

**Bảng thứ tự ưu tiên việc cần làm (chốt):**

1. **(A)** Đăng ký đội hạn 31/08 — ưu tiên số 1 (ngoài phạm vi kỹ thuật, nhưng khẩn).
2. **(B)** Giữ TimeGAN chạy 3000+ iteration (chất lượng synthetic tốt hơn).
3. **(C)** Làm `--gen physics` (bơm BPFO/sideband, chạy CPU).
4. **(D)** Ghép 3 kết quả (baseline AE + TimeGAN + fault injection) → bảng so sánh cho slide.

**Cách dựng `--gen physics` (rất ngắn gọn, chạy CPU):** mở rộng `_heuristic_faults` trong `scripts/02_compare_augmentation.py` thành hàm bơm lỗi theo chế độ:

```
f_bpfo = n/2 * (1 - d/D*cos(phi)) * f_r      # tần số đặc trưng vòng ngoài
# 1) lấy window normal x(t)
# 2) tạo chuỗi xung đơn vị tại mỗi chu kỳ 1/f_bpfo (mô phỏng viên bi đập lỗi)
# 3) điều chế xung (gia tốc xung, không phải sóng sin) rồi nhân hệ số "mức nặng"
# 4) cộng vào x(t), thêm nhiễu nền theo mức x(t)
# 5) dán nhãn: fault + tag chế độ (vd in 1 dòng mô tả "BPFO sideband, nặng 2x")
```
Chỉ cần bơm vào **miền thời gian + đảm bảo phổ có đỉnh tại `f_bpfo`/`f_bpfi`** là đủ làm "cột đối chiếu"; không cần mô phỏng cơ khí 3D phức tạp. Kết quả lưu `.npy` dưới dạng RAW feature (giống `scripts/03`, để `scripts/02 --gen npy` tái dùng và chuẩn hóa bằng scaler train — **tránh lộ test**).

**Điểm "ăn điểm" trên slide:** nêu rõ góc **"2 hướng sinh dữ liệu lỗi (vật lý vs generative) — vì sao bổ sung nhau"** phù hợp với **cột công nghệ gợi ý của đề** (`Mô phỏng vật lý - GAN/TimeGAN - Diffusion - fault injection - PyTorch`). Góc này cho thấy team **hiểu sâu** (không chỉ "chạy một cái GAN rồi nộp") và **trả lời đúng deliverable**: so sánh accuracy trước/sau bằng hai bộ sinh độc lập.

### 6.7.8. Kết luận / tóm tắt để nhớ

- **Generative là chính** (để có **độ bám phân bố thật** + đúng keyword đề + chạy được trên CPU), **vật lý là phụ** (để **kiểm soát chế độ lỗi + nhãn sạch + kể chuyện + rẻ**).
- Cột **công nghệ của đề gợi ý CẢ HAI cụm** (`Mô phỏng vật lý - fault injection` + `GAN/TimeGAN - Diffusion`), nên đi một mình là lệch khỏi kỳ vọng.
- **`PyTorch` là framework, không phải hướng** — nó chỉ là nền tảng để chạy TimeGAN (đã có `src/timegan_torch.py`); đừng nhầm nó thành một hướng sinh dữ liệu.
- Chốt chiến lược: TimeGAN ~60% (chính), fault injection vật lý ~20% (so sánh), Diffusion ~0% (ghi chú nâng cao) — thống nhất với `SPEC.md`, không dùng lỗi giả nào ở **test** (3-zone split, test chỉ nhận lỗi thật chưa từng thấy).
