# GIÁO TRÌNH BẢO TRÌ DỰ ĐOÁN & SINH DỮ LIỆU LỖI

## Cho bài toán A1 — DENSO Factory Hacks 2026
### Predictive-Maintenance AI Despite Scarce Fault Data

> **Ai nên đọc:** thành viên đội thi A1 có nền tảng ML/DL cơ bản, cần hệ thống lại
> từ toán → machine learning → deep learning → generative models → tài nguyên, kế hoạch.
>
> **Mục tiêu học xong:** hiểu và triển khai được pipeline "học cái bình thường → sinh
> thêm dữ liệu lỗi giả → so sánh accuracy trước/sau" — đúng 3 deliverable của đề.

---

## Mở đầu: đọc nhanh trong 5 phút
- **Bài toán:** máy móc nhà máy gắn cảm biến (rung/nhiệt/dòng); dữ liệu lỗi cực hiếm;
  cần cảnh báo sớm + ước tính tuổi thọ còn lại (RUL) mà ít báo động giả.
- **Thách thức lõi:** không đủ mẫu lỗi thật để học có giám sát.
- **Ý tưởng giải:** (1) học "thế nào là bình thường" rồi gắn cờ cái khác thường
  (anomaly detection); và/hoặc (2) **dùng GAN/Diffusion sinh thêm dữ liệu lỗi giả** rồi
  luyện mô hình phân loại, và **so sánh độ chính xác trước/sau** (deliverable chấm điểm).
- **Timeline:** đăng ký đội **≤31/08**, Factory Tour **11/09**, nộp Vòng 1 slide **≤12/10**,
  chung kết **16/11**.

> ⚠️ **HÔM NAY LÀ 30/08/2026.** Hạn đăng ký đội là **31/08 — mai**. Làm ngay trước khi đọc
> tiếp: lên **densohackathon.vn** lập tài khoản đội (3–5 người), điền ý tưởng sơ bộ,
> chốt owner tài khoản. Quên mốc này thì toàn bộ giáo trình này vô nghĩa.

---

## BẢN ĐỒ TƯ DUY A1 — mục tiêu lớn → bài toán con (đọc trước tiên)

Mục tiêu lớn và 6 bài toán con phải giải. Mỗi bài con ghi rõ **Mục tiêu / Chọn / Vì sao** và
trỏ tới phần tương ứng của giáo trình.

```
MỤC TIÊU LỚN: "AI bảo trì dự đoán phát hiện lỗi sớm trên chuỗi rung,
DÙ dữ liệu lỗi cực hiếm"   (đăng ký ≤31/08 → nộp slide ≤12/10 → chung kết 16/11)
 │
 ├─[a] NÉN TÍN HIỆU TẦN SỐ CAO → ĐẶC TRƯNG CÓ NGHĨA          → Phần 2, src/features.py
 ├─[b] ĐỊNH NGHĨA "THẾ NÀO LÀ TỐT/XẤU" CHO LỚP HIẾM         → Phần 2 mục 4
 ├─[c] PHÁT HIỆN BẤT THƯỜNG KHI KHÔNG CÓ NHÃN               → Phần 2 mục 5, Phần 3
 ├─[d] PHÂN LOẠI CÓ GIÁM SÁT KHI CÓ ÍT NHÃN                 → Phần 2 mục 3
 ├─[e] BÙ ĐẮP LỚP HIẾM = SINH DỮ LIỆU LỖI GIẢ               → Phần 3 mục 5–7, Phần 4 §3–4
 └─[f] CHỨNG MINH AUGMENTATION CÓ ÍCH = PROTOCOL TRƯỚC/SAU  → Phần 2 mục 6.4, scripts/02
```

| Bài toán con | Mục tiêu | Chọn | Vì sao | Trỏ tới |
|---|---|---|---|---|
| **[a] Nén tín hiệu** | Biến chuỗi rung 12–25 kHz thành vector nhỏ gọn mà vẫn giữ dấu hiệu hỏng | Windowing `win=512, stride=256` + **16 features** (thời gian: mean/std/peak/kurtosis… + tần số: spectral centroid/flatness/fft) + z-score | Không học trực tiếp tín hiệu kHz bằng torch/sklearn nhỏ; features đủ cho AE + RF, nhanh và giải thích được | Phần 2 · `src/features.py`, `src/pipeline.py` |
| **[b] Định nghĩa tốt/xấu cho lớp hiếm** | Đo bằng thước đo không bị "im lặng" khi fault hiếm | **Precision / Recall / F1 / PR-AUC**; quét ngưỡng P95–P99.5 | Mất cân bằng ~1:6000 làm accuracy/AUC "nói dối" (AUC 0.52 mà bỏ sót gần hết lỗi); DENSO chấm bằng chi phí báo giả vs bỏ lọt | Phần 2 mục 4 |
| **[c] Phát hiện bất thường khi không có nhãn** | Baseline "học cái bình thường → gắn cờ khác thường" khi chưa có lỗi nào | **Autoencoder** + reconstruction error + ngưỡng P99; nâng cấp: IsolationForest / OC-SVM | Đúng kịch bản nhà máy mới lắp cảm biến chưa thu được lỗi; cho "cột mốc trước" để so sánh | Phần 2 mục 5 · Phần 3 · `scripts/01` |
| **[d] Phân loại có giám sát với ít nhãn** | Biến "khác thường" thành nhãn dùng được với vài mẫu fault thật (20%) | **RandomForest / XGBoost** trên 16 features | Mạnh với tabular nhỏ, có feature importance để kể chuyện, chạy CPU được | Phần 2 mục 3 · `scripts/02` |
| **[e] Bù đắp lớp hiếm bằng lỗi giả** | Cân bằng train từ ~1:6000 về gần 1:1 bằng lỗi tổng hợp giống thật | **TimeGAN** (CPU) trước; **Diffusion-TS / FaultDiffusion** (GPU) khi có máy | Generative nhìn trúng bản chất đề "scarce fault data"; sinh đúng class fault theo ý; metric discriminative chứng minh "giống thật" | Phần 3 mục 5–7 · Phần 4 §3–4 |
| **[f] Chứng minh augmentation có ích** | Có con số "trước < sau" đáng tin, giám khảo không bắt bài | **3-zone split theo thời gian** (TRAIN/TEST/GREY) + generator **chỉ học train** + test = lỗi THẬT chưa thấy | Leakage là lý do duy nhất khiến số đẹp bị loại; đây đúng deliverable 3 BTC chấm | Phần 2 mục 6.4 · `src/pipeline.py` (3-zone), `scripts/02` |

**Đọc cây này thế nào:** bất kỳ câu hỏi nào của đội cũng rơi vào 1 trong 6 nhánh trên. Bắt đầu từ nhánh
bài toán đang vướng → đọc "Vì sao" trước → mở phần giáo trình được trỏ tới.

---

## Cách dùng giáo trình này

| Vai của bạn | Đọc gì để nhanh | Chi tiết tra thêm ở đâu |
|---|---|---|
| **Sinh viên muốn nắm gốc rễ** | Đọc **tuần tự** từ Phần 1 (toán) → Phần 2 (ML cổ điển, feature, anomaly, classification) → Phần 3 (DL + generative) → Phần 4 (tài nguyên + kế hoạch) | Mỗi phần tự đứng được; nếu vội, đọc phần được Bản đồ tư duy trỏ tới rồi quay lại đủ khi rảnh |
| **Kỹ sư cần nhanh / đang code** | Bản đồ tư duy A1 ở trên (5 phút) + **Phần 4 §8 — Quyết định nhanh (decision guidance)** để chọn model/dữ liệu theo tình huống | Mở đúng file được trỏ tới trong bảng: `src/features.py`, `src/pipeline.py`, `scripts/01`, `scripts/02`, `scripts/03` |
| **Người làm slide Vòng 1** | Phần 4: §1 (tổng quan + deliverable) · §3 (paper cite đúng ID) · §5 (2 nhánh hướng đi) · §7 (cấu trúc slide) · §1.5 (bối cảnh mùa trước) + **số liệu có sẵn trong `results/`** | Đọc bài "HIỆN TRẠNG TEAM + HÀNH ĐỘNG NGAY" dưới đây để lấy con số thật (30/08/2026) chứ không đoán |

> Quy tắc chung: con số nào đưa vào slide MUST có source trong `results/`. Không viết số tay
> vào slide — tái chạy command ghi ở Phụ lục A để sinh lại.

---

<hr style="page-break-before: always;"></hr>

## PHẦN 4 · KHO TÀI NGUYÊN, HƯỚNG ĐI & KẾ HOẠCH — KHO TÀI NGUYÊN (DATASET / PAPER / REPO) + HƯỚNG ĐI & KẾ HOẠCH CHIẾN THẮNG

## BÀI TOÁN A1 — DENSO FACTORY HACKS 2026

> **Văn bản này là tài liệu chiến lược duy nhất cho đội A1.** Nó bó 5 thứ lại thành một:
> (1) catalog tài nguyên đã **verify link** (dataset + paper + repo), (2) hướng đi kỹ thuật
> 2 nhánh kèm phân tích khả thi, (3) kế hoạch chi tiết theo tuần từ 26/08 → 12/10/2026,
> (4) chiến lược slide Vòng 1 + bộ câu hỏi cho Factory Tour, (5) **hiện trạng team tại 30/08**
> kèm hành động ngay.
>
> Mọi link bên dưới đều **đã thử tải được** trong quá trình chuẩn bị (08/2026). Nếu một link
> chết, đừng đoán link khác — dùng mirror tại https://data.phmsociety.org/nasa/ (PHM Society
> đã mirror toàn bộ dataset NASA PHM) và tìm paper qua arXiv ID thay vì đoán đường dẫn PDF.

---

## ⚠️ HIỆN TRẠNG TEAM + HÀNH ĐỘNG NGAY (cập nhật 30/08/2026)

**Trạng thái tổng:** sẵn data + code + số baseline. Đội còn vỏn vẹn 1 tháng để biến số thật thành câu chuyện thắng.

| Khoản | Hiện trạng (30/08) | Còn thiếu |
|---|---|---|
| Đăng ký đội | Hạn **31/08 (mai)** — chưa chốt là TRỌNG ĐIỂM số 1 | Ý tưởng sơ bộ trên **densohackathon.vn**, owner tài khoản |
| Dữ liệu | IMS (test_1/2/3) + FEMTO đã tải; CWRU **fault** đã tải, **normal 97–100.mat hỏng** (serve thiếu bytes) → fallback IMS làm nguồn normal | C-MAPSS (optional) |
| Paper / Repo | 3 bài đã tải/đọc online: FaultDiffusion, TimeGAN, Diffusion-TS; 2 repo đã clone (`vendor/`) | — |
| Code | `src/` (data, features, ae, pipeline 3-zone), `scripts/01` , `scripts/02` , `scripts/03` (TimeGAN) chạy được | Synthetic thật chưa sinh đủ |
| Số đo | AE baseline yếu → **cột mốc "trước" hoàn hảo**; heuristic gain nhỏ → cần sinh tốt hơn | Số "sau" đẹp bằng diffusion |

**Số liệu thực tế đang có (`results/`, ngày 30/08):**
- **AE baseline** (`baseline_ae.json`, IMS, `--limit 300 --epochs 60`): ngưỡng 0.0164,
  precision 0.387, recall 0.031, F1 0.057, AUC 0.524. AE gần đoán bừa trên feature IMS → đúng vai "trước".
- **RF heuristic** (`compare_augmentation.json`, IMS): TRƯỚC 0.416 / 0.477 / 0.444 / 0.582 →
  SAU 0.418 / 0.482 / 0.447 / 0.580 (precision/recall/F1/AUC). **Gain gần như bằng 0** →
  heuristic sinh lỗi kém; đây là lý do chính để chuyển lên TimeGAN/diffusion.
- `pipeline.py` đã cài **3-zone split** 30/08: normal `[0..0.70) TRAIN · [0.70..0.90) TEST khỏe ·
  [0.90..1.0] GREY bỏ`; fault 20% train / 80% test thật; z-score fit trên TRAIN → chống leakage.

**Hành động NGAY, theo thứ tự ưu tiên:**

1. **[Hôm nay/mai] Đăng ký đội ≤31/08** — densohackathon.vn, chốt owner. Không trượt mốc này.
2. **Sửa tiếp model:** nâng bằng **XGBoost** + sweep ngưỡng P95–P99.5 (tăng recall tại precision
   chấp nhận); thêm assertion test chỉ fault THẬT (chống leak).
3. **TimeGAN ≥3000 iterations:** chạy `scripts/03_train_timegan.py` cho đủ số steps, sinh `.npy`,
   đo **discriminative score** (~0.5–0.65 là đẹp) → đưa vào script 02 `--gen npy`.
4. **Dashboard MVP + báo cáo trước/sau** (bảng + PR curve) — đầu vào trực tiếp cho slide Vòng 1.
5. **Factory Tour 11/09:** chuẩn bị sẵn 10 câu hỏi (§7.4) + 1-pager protocol trước/sau để hỏi kỹ sư DENSO.

**Chốt câu chuyện:** bán "một protocol tái sử dụng" (bất kỳ dataset rung nào → AE baseline →
sinh lỗi giả → supervised → báo cáo trước/sau), không bán "AI". Mọi con số phải đi qua `results/`.

---

## Mục lục CHƯƠNG 4

2. [Catalog DATASET](#2-catalog-dataset)
3. [Catalog PAPER](#3-catalog-paper)
4. [Catalog REPO](#4-catalog-repo)
5. [Hướng đi kỹ thuật 2 nhánh](#5-hướng-đi-kỹ-thuật-2-nhánh--đánh-giá-kỹ--đề-xuất)
6. [Kế hoạch chi tiết theo tuần (26/08 → 12/10/2026)](#6-kế-hoạch-chi-tiết-theo-tuần-2608--12102026)
7. [Chiến lược slide Vòng 1 (hạn 12/10)](#7-chiến-lược-slide-vòng-1-hạn-1210)
8. [Quyết định nhanh (decision guidance)](#8-quyết-định-nhanh-decision-guidance)

---

## 1. Tổng quan bài A1 + đọc lại deliverable + diễn giải giám khảo mong gì

### 1.1 Bài toán trong một câu

> **"Predictive-Maintenance AI Despite Scarce Fault Data"** — Xây dựng hệ AI bảo trì dự
> đoán (phát hiện bất thường sớm + ước lượng tuổi thọ RUL) trong điều kiện **rất ít dữ
> liệu lỗi** — đúng như thực tế nhà máy: có hàng tháng dữ liệu "máy chạy tốt", nhưng rung
> động/ánh sáng khi máy sắp hỏng thì gần như không có mặt hàng nào trên kệ.

Ngữ cảnh công nghiệp điển hình (dựng theo README của team):

- Máy móc (môtơ + ổ bi, bơm, trục…) chạy liên tục, được gắn cảm biến rung / nhiệt độ / dòng điện.
- Có **nhiều** dữ liệu trạng thái "máy khỏe" (normal), **rất ít** dữ liệu "máy sắp hỏng / vừa hỏng" (fault).
- Input: chuỗi thời gian đa biến (vibration X/Y/Z, current, nhiệt độ) ~10 Hz (bản đồ với dữ liệu rung 12 kHz của CWRU / IMS).
- Output mong muốn: (0) cảnh báo bình thường / (1) bất thường + RUL (giờ còn lại).
- Goal: **phát hiện sớm (recall cao) NHƯNG ít báo động giả (precision cao)** — trade-off kinh điển của bảo trì dự đoán (báo giả = tắt máy xem xét lãng phí; bỏ sót = hỏng lớn gấp chục lần).

### 1.2 Deliverable chính xác của đề (sau 3 tháng)

Ba đầu ra **bắt buộc** cần nộp (đối chiếu `RESOURCES.md §5`):

| # | Deliverable | Diễn giải sản phẩm cụ thể | Trạng thái hiện tại của team |
|---|-------------|---------------------------|------------------------------|
| 1 | **Bộ sinh dữ liệu bất thường** | Code pipeline sinh "dữ liệu lỗi giả" (synthetic fault) — có thể là GAN hoặc Diffusion cho chuỗi thời gian, hoặc heuristic. Kèm cách dùng rõ ràng (CLI, config). | Đã có nhánh heuristic trong script 02; **thiếu** sinh bằng diffusion thật |
| 2 | **Tập dữ liệu tăng cường** | File `.npy` / `.csv` chứa các window "lỗi giả" đã sinh — dùng được ngay để feed vào train. | Chưa có (mới sinh transient trong script) |
| 3 | **Báo cáo so sánh độ chính xác dự báo TRƯỚC / SAU tăng cường** | Bảng số liệu + biểu đồ (PR curve trước/sau) trên cùng một test set **lỗi thật chưa thấy**. | Đã có protocol trong script 02 + kết quả heuristic thô |

### 1.3 Giám khảo mong gì? (diễn giải)

Đây là hackathon **công nghiệp** (DENSO = nhà sản xuất linh kiện ô tô, nhà máy họ có vô số
thiết bị rung). Đội giám khảo gồm kỹ sư DENSO + chuyên gia AI. Họ chấm theo thứ tự ưu tiên:

1. **Tôn trọng kịch bản "scarce fault data"** — đây là mấu chốt. Bài nộp sẽ bị loại ngay nếu
   team làm supervised classification bằng cách "có đủ lỗi thật, chia train/test bình thường".
   Scarce phải được xử lý bằng: (a) unsupervised baseline, rồi (b) sinh thêm lỗi giả.
2. **Trung thực về số liệu/leakage** — test phải là **lỗi THẬT chưa từng thấy**, không một
   pixel nào của test xuất hiện trong train (kể cả bóng dáng qua augmentation). Ai cũng có
   thể "đẹp" nếu cheat; đội thắng là đội đẹp *thật* và chỉ rõ protocol trong slide.
3. **Một nghiên cứu có cấu trúc, không phải "chạy được"** — giám khảo nghề sẽ hỏi: baseline
   là gì? ngưỡng chọn thế nào? vì sao gain lại đến? paper nào bạn dựa trên? Câu trả lời nằm
   trong log, JSON, và slide.
4. **Khả năng áp dụng thật** — "máy của em/đội gặp tình huống này, thu thập data lỗi thế nào?"
   Nếu team trả lời được sau Factory Tour và map vào pipeline → điểm hợp đồng.
5. **Tính "so sánh trước/sau" phải rõ** — càng ít text, càng nhiều biểu đồ: PR curve + bảng
   Precision/Recall/F1/AUC trước/sau, ngưỡng P99, và ví dụ window lỗi giả vs lỗi thật
   (trực quan → thuyết phục).

**Ký hiệu quan trọng nhất:** Đề bài nhấn "predictive-maintenance AI *despite scarce fault
data*". Đừng bán "AI" trừu tượng; hãy bán **"một protocol tái sử dụng được: bất kỳ máy nào,
bất kỳ dataset rung nào → baseline AE → sinh lỗi giả → train supervised → báo cáo trước/sau"**.

### 1.4 Context stack code hiện có (đã đọc mã nguồn)

- **Environment:** `.venv/` Python 3.11.9, scipy 1.13.1 (xem `requirements.txt`).
- **`src/data.py`** — loader CWRU (tách normal/fault bằng label), NASA IMS (3 test, tự nhận diện
  file timestamp, `limit_per_test`), FEMTO (mọi `.csv`); hàm `make_windows(sig, win=512, stride=256)`.
  IMS test_1: ~163840 sample/chuỗi, 8 cột; test_3: 4 cột.
- **`src/features.py`** — 16 features vector-hoá trên ma trận window: thời gian (mean, std, rms,
  peak, peak2peak, skewness, kurtosis, crest, shape, impulse) + tần số (spectral centroid/spread/
  flatness/energy, freq_mean/freq_std) + z-score theo cột.
- **`src/ae.py`** — `FeatureAE` (Linear 16→32→8→32→16), train chỉ trên normal, ngưỡng P99,
  metrics precision/recall/F1/AUC; tự chọn CUDA nếu có.
- **`src/pipeline.py`** — 3-zone split TRONG TỪNG test-run (sửa ngày 30/08 sau khi phát hiện
  test normal cũ lấy trúng vùng "suy giảm sát hỏng"): normal 0–70% mỗi run chia tiếp
  `[0..0.70) → TRAIN khỏe`, `[0.70..0.90) → TEST khỏe chưa thấy`, `[0.90..1.0] → GREY bỏ hẳn`.
  Fault 90–100% mỗi run: trộn ngẫu nhiên 20% train / 80% test. Z-score fit TRÊN TRAIN, áp cho
  cả test lẫn synthetic (chống data leakage, scaler dùng lại ở script 02).
- **`scripts/01_baseline_anomaly.py`** — AE học "bình thường" → ngưỡng P99 → báo cáo trên lỗi thật.
- **`scripts/02_compare_augmentation.py`** — protocol TRƯỚC/SAU augmentation, cờ `--gen heuristic|npy`,
  cờ `--limit`, `--win`, `--stride`, `--gen-npy`. Test = normal khỏe chưa thấy (3-zone như trên)
  + fault THẬT chưa thấy (80% fault).
- **`scripts/03_train_timegan.py`** — nhánh đang mở: train TimeGAN sinh lỗi giả (target ≥3000 iterations),
  xuất `.npy` → feed script 02 `--gen npy`.
- **Project ngôn ngữ:** docstring/comment tiếng Việt giải thích "script chạy gì & số nghĩa gì",
  tên hàm/CLI tiếng Anh — 3 script mỗi cái đều có câu chuyện tự giải thích ở đầu file.

> Mã tham chiếu cụ thể trong tài liệu này dùng đúng tên file/flag nêu trên để không mất công soi.
> **Kết quả baseline hiện hữu** (`results/baseline_ae.json`, `--limit 300 --epochs 60`, dataset=ims):
> ngưỡng 0.0164, precision 0.387, recall 0.031, F1 0.057, AUC 0.524. AE thuần trên feature IMS
> vẫn yếu (gần 0.5 = đoán bừa) → đây chính là chỗ "trước" hoàn hảo để kể câu chuyện tăng cường
> (RF trong script 02 đã kéo recall lên ~0.48 nhờ dữ liệu + model có giám sát).
> Lưu ý lịch sử: kết quả cũ ghi "recall 0.013, AUC 0.542" sai nguồn gốc — lấy trước khi sửa split,
> test normal khi đó trúng vùng đã suy giảm → test bất công nên recall tụt thảm khốc.
>
> **Cập nhật 30/08 — đọc kỹ:** số chính xác hiện tại là AE (P 0.387 / R 0.031 / F1 0.057 / AUC 0.524)
> và RF heuristic TRƯỚC 0.416 / 0.477 / 0.444 / 0.582 → SAU 0.418 / 0.482 / 0.447 / 0.580 (`results/compare_augmentation.json`).
> Gain heuristic ≈ 0 → **bằng chứng cần bộ sinh thật (TimeGAN/diffusion)**, không dùng heuristic làm số chính.

### 1.5 Bối cảnh mùa trước — SPARK / DiffusionAD (dùng cho slide)

Đội vô địch Mùa 3 (2025): **SPARK** — Viện Trí tuệ nhân tạo, Đại học Công nghệ (UET), ĐHQG Hà Nội.

**Giải pháp: "DiffusionAD"** — phát hiện lỗi bất thường trong sản xuất bằng **diffusion model,
không cần dữ liệu nhãn** (unsupervised/self-supervised). Thuộc nhánh AI/Data (phát hiện lỗi
linh kiện — hướng thị giác máy).

**Nguồn đối chiếu:** VnReview (khởi động mùa 4), FPT, Facebook UET/AI-UET, bài BUV về đội á quân
FSiL (BUV + Bách Khoa, giải "End-to-End AI"). *(Nguồn cấp thứ cấp — nếu trích chi tiết kỹ thuật
trong slide nên tìm bài VnReview/FPT gốc.)*

**Ý nghĩa chiến lược cho A1 (điểm ăn điểm slide):**
- Năm ngoái giám khảo đã trao giải cao nhất cho tư duy **"diffusion + dữ liệu lỗi khan hiếm/không nhãn"** — đúng tinh thần của A1.
- **Điểm khác biệt phải nêu:** DiffusionAD thắng ở mảng **ảnh (CV)**; đề A1 được team đưa cùng tư duy đó sang **chuỗi thời gian rung (TS)** — chưa ai chứng minh ở nhóm này.
- **Tránh:** copy y hệt họ (thắng bằng unsupervised anomaly); A1 yêu cầu deliverable "bộ sinh dữ liệu + so sánh trước/sau" → nhấn mạnh nhánh augmentation có giám sát.

---

## 2. Catalog DATASET

### 2.1 Bảng tổng quan

| Dataset | Loại | Mô tả ngắn | Số sample | Tần số | Task chính dùng cho | Trạng thái team |
|---|---|---|---|---|---|---|
| **CWRU Bearing** (Case Western Reserve Univ.) | Có nhãn | Rung ổ bi 2HP môtơ, 3 vị trí (DE/FE/BA), 3 kích thước lỗi + normal-motor | ~hàng trăm file `.mat` (mỗi file 12s, ~120k sample@12k) | 12 kHz / 48 kHz | Anomaly detection + fault classification | Đã tải fault (105, 118, 130…), **normal 97–100.mat hỏng** |
| **NASA IMS Bearings** | Run-to-failure (không nhãn hạng) | 3 test, nhiều ổ bi chạy đến hỏng thật | test_1: ~2156 file × 163840 sample × 8 cột; test_2: 984 file; test_3: 6324 file × 4 cột | 20 kHz, ghi mỗi 10 phút | Anomaly + **RUL** (phần đầu = normal, cuối = suy giảm) | Đã tải, dùng chính |
| **FEMTO / PRONOSTIA (FEMTO-ST)** | Accelerated life test | 17 máy, 6 điều kiện vận hành, chạy nhanh đến hỏng | 17 máy × (train/test), file `acc.csv` | 25.6 kHz | RUL + anomaly | Đã tải zip |
| **NASA Turbofan (C-MAPSS)** | Mô phỏng | 100 động cơ, 21 sensor + 3 op condition, có RUL thật | 4 tập train/test, FD001 100 máy | 1 Hz (chu kỳ) | **RUL chuẩn ngành** | Chưa tải (optional) |

> Mirror dự phòng cho cả 4: https://data.phmsociety.org/nasa/ (nếu S3 chậm/chết).

---

### 2.2 CWRU Bearing — chi tiết

- **Nguồn:** https://engineering.case.edu/bearingdatacenter/download-data-file
  (tải tay bằng trình duyệt hoặc `wget`; trang này kiểu portal nên script bỏ qua — dùng browser).
- **Đặc điểm:**
  - Ổ bi thử nghiệm trên môtơ 2 HP; lỗi được tạo bằng tia EDM (điện phóng): **kích thước lỗi
    0.007", 0.014", 0.021", 0.028"** trên inner race (IR), outer race (OR), ball (B), + dataset normal-motor.
  - 3 vị trí đo gia tốc: Drive End (DE), Fan End (FE), Base (BA). Mỗi file `.mat` chứa biến
    `*_DE_time`, `*_FE_time` (rung theo thời gian) và `*_RPM` thuộc tính.
  - Tần số 12 kHz (đa số, file "12k Drive End Bearing Fault Data") và 48 kHz ("48k").
  - 4 tải: 0 / 1 / 2 / 3 HP → RPM khác nhau (1797/1772/1750/1730).
  - **Verify cụ thể:** các file fault như `105.mat`, `118.mat` (IR007…) đọc được bằng
    `scipy.io.loadmat`; **4 file `Normal Baseline Data/97.mat…100.mat` bị hỏng khi tải từ
    server chính thức (serve thiếu bytes)** → `src/data.py` tự bỏ qua file lỗi (`loadmat` ném
    exception → `continue`), vì vậy hiện tại **không có normal đọc được từ CWRU** → fallback
    IMS là nguồn normal.
- **Task phù hợp:**
  - *Anomaly detection:* fault (IR/OR/B) là "bất thường", mọi normal-motor là "bình thường" — nhãn sạch.
  - *Fault classification:* 3 loại lỗi + 4 mức nghiêm trọng → 12 lớp (nếu muốn slide đẹp hơn).
- **Cách chia normal/fault hợp lý (giữ nhãn sạch):**
  - Đã có nhánh label trong loader: `label=0` cho normal, `label=1` cho fault.
  - Split *theo file*, KHÔNG cắt window rồi trộn — mỗi file chỉ ở một tập (tránh window cùng
    file nằm cả train lẫn test).
  - Normal ~hàng chục file → 80/20 train/test; fault chia 20% train (scarce) / 80% test.
  - Lưu ý: các mức tải khác nhau có biên độ khác nhau → đừng lẫn tải 0HP vào test mà train chỉ
    có 3HP, vì AE sẽ "sợ" biên độ chứ không "sợ" lỗi.
- **Hạn chế:**
  - Lỗi tạo bằng EDM = `seeded fault` **không tự nhiên như suy giảm thật** (không có giai đoạn
    "mọc vết nứt"); vết lỗi tĩnh, biên độ rất mạnh → dễ tách với normal hơn thực tế nhà máy.
  - Normal baseline bị hỏng trên server chính thức → phải dùng nguồn khác hoặc fallback IMS.
  - Có `RPM` không đổi trong mỗi file nên khó mô phỏng load thay đổi.
  - Format `.mat` mỗi file có meta khác nhau → code phải dò `*_DE_time` thay vì tên cứng.

---

### 2.3 NASA IMS Bearings — chi tiết

- **Nguồn:** https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
  (script `scripts/download_data.sh` đã tự tải + giải nén; zip ~kiểu `.rar` của NASA nhưng
  download script xử lý được).
- **Đặc điểm:**
  - 3 test **run-to-failure thật** (không phải mô phỏng): ổ bi chạy liên tục đến khi hỏng.
  - `test_1`: 4 ổ bi (8 kênh = 2 accel × 4 ổ), ~2156 file, mỗi file ~20s (163840 sample) ghi
    mỗi 10 phút. **Test_1 là chuỗi tiêu chuẩn** trong mọi paper (RUL learning).
  - `test_2`: 4 ổ bi, 984 file, 8 kênh.
  - `test_3`: 4 ổ bi, 6324 file nhưng chỉ **4 kênh** (2 accel × 2 hướng？ — đọc được 4 cột),
    thời điểm rời đầy đủ hơn.
  - Tần số lấy mẫu 20 kHz. Mỗi file `2003.10.22.12.06.24` (timestamp) là một "snapshot": load
    ra ma trận (20480, n_cols) → `.ravel()` thành tín hiệu đa kênh dài.
  - Có dữ liệu mô tả khi nào ổ bi hỏng (event log đi kèm) → phục vụ gán RUL tham khảo.
- **Task phù hợp:**
  - **Anomaly:** 70% đầu mỗi test = normal; 10% cuối = suy giảm/hỏng — cách team đang dùng.
  - **RUL:** dùng chỉ số health đoạn cuối; càng "late" càng fault mạnh.
- **Cách chia normal/fault hợp lý (protocol đã chuẩn hoá trong code, `src/pipeline.py`):**
  - Với MỖI test-run (KHÔNG ghép phẳng 3 test làm 1): lấy `runs_normal = files[:0.7×m]`,
    `runs_fault = files[0.9×m:]` (đoạn 0.7–0.9 suy giảm dần bỏ qua — nhãn nhiễu).
  - Trong vùng NORMAL (0–70%) của từng run chia tiếp 3 vùng để test normal KHÔNG bị "trúng
    vũng normal đã xuống cấp sát vùng hỏng" (bug cũ → test bất công, recall tụt ~0.01):
      `[0..0.70) → TRAIN khỏe`, `[0.70..0.90) → TEST khỏe chưa thấy`, `[0.90..1.0] → GREY bỏ hẳn`.
  - **Không trộn lẫn test**: train/test theo thứ tự thời gian của *cùng* file-set; danger nếu
    trộn cuối test_3 vào train rồi test trên cuối test_1 — mức suy giảm khác nhau.
  - FAULT: gộp lỗi thật (90–100% mỗi run) rồi trộn ngẫu nhiên → 20% train-scarce / 80% test-thật.
    Z-score fit trên TRAIN, chung cho test + synthetic (chống leakage).
- **Hạn chế:**
  - **Nặng:** tải hết ~8 GB RAM → dùng `--limit` (mặc định 300 file/test) cho prototype, bỏ giới
    hạn chỉ khi chạy báo cáo cuối.
  - Nhãn không có per-file "fault/normal" chính thức → ta suy diễn từ vị trí trong chuỗi
    (0.7/0.9 heuristic) nên nhãn hơi mơ hồ ở vùng 0.7–0.9.
  - 8 kênh của test_1 thực chất là 4 ổ bi → dùng đúng semantics mới đúng (mỗi ổ bi là một máy)
    nếu muốn câu chuyện "đa thiết bị".
  - File không có đuôi; vài file rỗng → code lọc `arr.size == 0`.

---

### 2.4 FEMTO / PRONOSTIA — chi tiết

- **Nguồn:** https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip
  (đã tải `FEMTO_Bearing.zip`, giải nén → `data/FEMTO/` có các sub-dataset `Bearing1_*`, `Bearing2_*`).
- **Đặc điểm:**
  - **Accelerated life test**: ổ bi nhỏ chạy dưới tải cực mạnh để hỏng nhanh (vài giờ thay vì
    vài tháng) → có đủ giai đoạn "health → suy giảm → hỏng" trong thời gian ngắn.
  - 6 điều kiện vận hành (operating condition) khác nhau về tải/tốc độ; mỗi condition có nhiều
    "máy" (bearing) — tổng 17 máy.
  - Dữ liệu mỗi máy lưu trong file `acc.csv` (gia tốc chiều? theo file — header có tên cột),
    ghi 25.6 kHz với windows rời; kèm metadata điều kiện vận hành.
  - Dataset chuẩn của **IEEE PHM 2012 Prognostic Challenge** — có ground-truth RUL (số giây trước hỏng).
- **Task phù hợp:**
  - **RUL** là thế mạnh (có nhiều paper dùng FEMTO score RUL).
  - **Anomaly:** phần đầu mỗi máy = normal, phần cuối = fault → chia như IMS.
- **Cách chia normal/fault:**
  - Theo từng máy: `sigs_n = files[:0.6×m]`, `sigs_f = files[0.85×m:]` — nhìn thống kê RMS trước.
  - **Train/test phải khác máy** (generalization giữa các bearing) hoặc cùng máy chia thời gian —
    nêu rõ chọn gì trong slide; an toàn nhất: train trên N máy, test trên máy chưa từng thấy.
- **Hạn chế:**
  - File CSV không đồng nhất (số cột/đặt tên thay đổi theo sub-dataset) → loader hiện tại đơn
    giản `.ravel()` có thể trộn kênh không đúng semantics.
  - Chưa có nhãn chuẩn ở loader hiện tại → `01_baseline_anomaly.py --dataset femto` có raise
    `SystemExit("FEMTO chưa có nhãn chuẩn")` → đừng dùng FEMTO để chạy baseline liền; dùng ims
    hoặc cwru cho demo chính, FEMTO để thêm phần RUL sau.
  - Số máy ít (17) → không đủ để làm fault classification.

---

### 2.5 NASA Turbofan (C-MAPSS) — chi tiết

- **Nguồn:** https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
  (optional — chỉ tải nếu muốn tăng độ sâu về RUL).
- **Đặc điểm:**
  - Mô phỏng 100 động cơ, mỗi động cơ một chuỗi cycle (chu kỳ bay) đến khi hỏng; 21 sensor +
    điều kiện vận hành; **chuẩn vàng của PHM** (cùng dòng NASA PHM Challenge 2008).
  - 4 tập con: FD001 (1 op condition, 1 mode lỗi), FD002, FD003, FD004. Train/test riêng, test
    có RUL thật dùng để scoring.
  - Tần số 1 Hz theo cycle (không phải Hz sinh học như rung) → **khác biệt bản chất dữ liệu với CWRU/IMS**.
- **Task phù hợp:**
  - **RUL regression** (dự báo số cycle còn lại) — script so sánh sẽ cần nhánh regressor nếu dùng.
  - Có bộ evaluation chính thức để tự minh chứng "đạt score" — thuyết phục nếu cần câu chuyện benchmark.
- **Cách chia:**
  - Dùng luôn train/test có sẵn của NASA (không tự chia lại).
  - Nếu muốn anomaly: coi 30% cycle đầu = normal, 5% cuối = fault.
- **Hạn chế:**
  - Dữ liệu mô phỏng (không phải vibration thật) → khác bản chất với bài "máy rung"; dùng trong
    slide như "bổ trợ chứng minh protocol tổng quát" chứ không phải dữ liệu chính.
  - Không thể xài trực tiếp `make_windows` semantics của rung nếu không đọc lại cấu trúc file
    (mỗi dòng = 1 cycle, cột = sensor).
  - Cần tuning riêng (nhiều paper, ít thời gian) → xếp optional.

### 2.6 Bảng quyết định "dùng dataset nào ở đâu"

| Bước | Dataset chính | Lý do | Dự phòng |
|---|---|---|---|
| Baseline AE (script 01) | **IMS** | có normal + fault thật đọc được ngay, chuỗi liền mạch | CWRU fault-only không có normal |
| Augmentation protocol (script 02) | **IMS** | chuỗi dài → nhiều window → đủ train/test | CWRU (nếu lấy normal từ nguồn khác) |
| Demo fault classification (slide) | **CWRU** | nhãn 3 loại lỗi sạch → ảnh spectrogram đẹp | — |
| Phần RUL (điểm cộng) | **FEMTO** hoặc **C-MAPSS** | có ground truth RUL | FEMTO nếu đã có, C-MAPSS nếu cần đa dạng |
| Câu chuyện benchmark "protocol tổng quát" | cả 4 | mỗi dataset là một "nhà máy khác" | — |

---

## 3. Catalog PAPER

### 3.1 Bảng tổng quan

| # | Paper | Venue / Năm | Link (đã verify) | Cần đọc vì |
|---|---|---|---|---|
| 1 | **FaultDiffusion: Few-Shot Fault Time Series Generation with Diffusion Model** | arXiv **2511.15174** (2025) | abs https://arxiv.org/abs/2511.15174 · PDF https://arxiv.org/pdf/2511.15174 | **Trùng gần 100% đề A1**; trích dẫn chính trong slide |
| 2 | **TimeGAN: Time-series Generative Adversarial Networks** | NeurIPS 2019 | PDF https://papers.nips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf | Nền tảng GAN sinh TS + 2 metric chuẩn hoá |
| 3 | **Diffusion-TS: Interpretable Diffusion for General Time Series Generation** | ICLR 2024 | https://openreview.net/pdf?id=4h1apFjO99 (đọc online; openreview có thể chặn script tải) | SOTA diffusion TS sinh theo class — nhánh augmentation thật |

> **Lưu ý dấu hiệu quan trọng về trích dẫn:** có **nhiều** bài khác cùng tên "FaultDiffusion"
> tồn tại trên arXiv (tên generic). Khi trích dẫn/cite luôn ghi kèm arXiv ID **2511.15174**;
> đừng đoán link khác. Khi cần kiểm tra bài chuẩn nhất, dùng arXiv API:
> `curl "https://export.arxiv.org/api/query?id_list=2511.15174"`.

---

### 3.2 FaultDiffusion (arXiv 2511.15174) — giải thích chi tiết

**Vì sao đọc trước tiên:** Đây là paper hiếm hoi trực tiếp giải "sinh dữ liệu lỗi **few-shot**
cho industrial time series" — đúng y bối cảnh đề A1 ("scarce fault data"). Đọc nó biết được:

- Sự khác biệt với diffusion thuần: có cơ chế **few-shot conditioning** (đào tạo với vài mẫu
  fault thật để bộ sinh biết "dáng vẻ" của lỗi) thay vì sinh mù một class.
- Cách model để giữ **tính nhất quán vật lý của tín hiệu rung** (spectrum không bị nhiễu ngẫu
  nhiên) — chính là thứ mà "GAN sinh lung tung rồi bị chấm điểm kém" hay dính.
- Protocol đánh giá mà họ dùng để chứng minh synthetic "giống thật": thường là (1) t-SNE nhìn
  chồng lấn phân phối, (2) classifier-based discrimination (train bộ phân biệt synthetic vs real),
  (3) downstream metric khi augment vào train → **ba thứ đó y hệt thứ team cần in trong slide**.

**Ứng dụng trong dự án A1:** dùng kỹ thuật few-shot conditioning của nó làm "thiết kế bộ sinh
lỗi giả" (deliverable 1): train với vài cửa sổ fault thật (20% fault) + đầy đủ normal, giam
phân phối sinh về đúng vùng tần số 'lỗi giống thật'. Đây là **tài liệu tham khảo mạnh nhất**
để trả lời câu giám khảo: "Làm sao synthetic không bị model tách ra ngay lập tức?"

**Lưu ý trích dẫn:** cite đúng dạng `FaultDiffusion (arXiv:2511.15174, 2025)`. Nếu repo code
của bài tồn tại hãy link repo gốc (chưa verify repo tại thời điểm viết — không đoán, chỉ cite paper).

---

### 3.3 TimeGAN (NeurIPS 2019) — giải thích chi tiết

**Vì sao đọc:**
- Bài kinh điển đầu tiên coi time series như một đồ thị và rèn generator GAN bằng **supervised
  loss trên autocorrelation** (temporal dynamics) — đây là lý do nó sinh ra chuỗi "khớp moment
  thống kê + khớp nhịp (autocorrelation)", thứ mà vanilla GAN không làm được.
- Paper định nghĩa **2 metric kinh điển để đánh giá chất lượng sinh** mà team nên tái sử dụng y
  nguyên: `discriminative score` (train classifier "real vs synthetic" → accuracy càng gần 0.5
  càng tốt) và `predictive score` (dùng chuỗi tổng hợp học dự báo 1 bước rồi đo lỗi dự báo).
- **Repo chuẩn** có sẵn (xem §4) → chi phí thử nghiệm thấp nhất trong 3 paper.

**Ứng dụng trong dự án A1:**
- Nhanh: chạy CPU được cho chuỗi dài vừa; hợp để sprint tuần 5–6 sinh **một cách nhanh** so sánh trước/sau.
- Vì generator là MLP/RNN + discriminator phủ distribution, cần có "normal (nhiều) + ít fault
  thật (ít)" để sinh — đúng thiết kế đề.
- Dùng luôn 2 metric của nó để slide có con số về **"synthetic giống thật đến mức nào"** trước
  khi nói về gain downstream.

**Lưu ý:** TimeGAN nhạy tuning (đặc biệt trong bài nhiều paper tái lập thất bại) — nếu có GPU
hãy thử Diffusion-TS trước (ổn định hơn); TimeGAN là phương án CPU dự phòng.

---

### 3.4 Diffusion-TS (ICLR 2024) — giải thích chi tiết

**Vì sao đọc:**
- Diffusion cho TS đạt chất lượng sinh cao hơn GAN (đặc biệt bao phủ chế độ hiếm — "mode
  coverage") và **có guidance theo class/condition**: bạn cho "lỗi loại gì" → model sinh đúng.
- Bài giải thích rõ cách biến dữ liệu TS thành 2D (thời gian × kênh) và diffusion trên liên
  kết biến — từ đó giúp hiểu ánh xạ "synthetic window" sang `make_windows` của team.
- Có repo chuẩn (xem §4) tương đối "plug-and-play" với Tutorial notebook.

**Ứng dụng trong dự án A1:**
- Nhánh augmentation **thật**: train trên normal + fault thật, **condition theo class** để sinh
  đúng class "fault" với số lượng mong muốn → viết ra `.npy` → feed `scripts/02 --gen npy`.
- Vì model sinh theo chuỗi trong thang thời gian gốc, phải chỉnh chiều dài = `win=512` (hoặc
  chunk) để ra window khớp pipeline — phần dễ sai nhất.
- Cần GPU CUDA (thực tế chạy được CPU nhưng rất lâu) → nằm lịch sau khi chắc chắn có máy GPU.

**Lưu ý:** File PDF trên OpenReview **đôi khi chặn script `wget`/crawler** → mở bằng trình duyệt
để đọc online, không cần tải về; slide chỉ cite ICLR 2024 chứ không cần link PDF chính xác.

### 3.5 Bảng "đọc → làm gì" tóm tắt

| Paper | Đọc để… | Biến thành gì trong repo |
|---|---|---|
| FaultDiffusion | hiểu few-shot conditioning + tạo ảnh t-SNE phân phối | phương án thiết kế generator + nội dung slide §"scarce data" |
| TimeGAN | đồ thị GAN TS + 2 metric | metric đánh giá synthetic (discriminative/predictive) trong `results/` |
| Diffusion-TS | condition-by-class + chiêu guidance | generator thật → `.npy` → script 02 `--gen npy` |

---

## 4. Catalog REPO

### 4.1 Bảng tổng quan

| Repo | Paper | Ngôn ngữ / Độ khó | Yêu cầu phần cứng | Mục tiêu dùng trong team |
|---|---|---|---|---|
| https://github.com/jsyoon0823/TimeGAN | TimeGAN (NeurIPS 2019) | TensorFlow 1.x → cần môi trường riêng; **chạy CPU được** | CPU OK, GPU tùy chọn | sinh dữ liệu lỗi giả phương án B, metric đánh giá |
| https://github.com/Y-debug-sys/Diffusion-TS | Diffusion-TS (ICLR 2024) | PyTorch; Tutorial_0/1/2.ipynb | **GPU CUDA khuyến nghị** (CPU rất chậm) | generator chính → `.npy` → script 02 |

> Cả hai clone được bây giờ và cơ bản là "đun sôi" — nhưng đừng để vào `src/` chính nếu cần
> tách môi trường (hai repo có dependency khác nhau, có thể xung đột với `.venv/` của team).

---

### 4.2 TimeGAN — cách dùng

**Clone vào thư mục riêng (không phải `.venv`)**
```bash
cd A1_predictive_maintenance
mkdir -p vendor && cd vendor
git clone https://github.com/jsyoon0823/TimeGAN.git
cd TimeGAN
```

**Môi trường** (vì là TensorFlow 1.x / tf.keras cũ, tách riêng venv để không đụng `.venv` của team):
```bash
python -m venv .venv-tf1
source .venv-tf1/bin/activate
pip install "tensorflow==2.10.*"  # phiên bản tương thích code TF1-API
# (một số bản của repo hỗ trợ TF2 mode eager; chạy thử Tutorial trước)
```

**Chạy huấn luyện + sinh:**
```bash
python3 -m main_timegan.py --data_name stock
# sau khi train, script sinh file: timeGAN.saved_model + output/...
# dùng Python API trong repo: load data, gọi `timegan` sinh n mẫu class fault
```

**Sinh "dữ liệu lỗi giả" từ TimeGAN theo kịch bản A1:**
1. Tạo tensor 2D `(n_seq, time_len, n_dim)` với `n_seq` = số window (normal là chủ yếu +
   vài window fault thật — few-shot), `time_len = win (512)`, `n_dim` = 16 *features* (nếu bạn
   sinh trên feature space) *hoặc* = số kênh sensor (nếu sinh trên signal thô — đọc khớp với
   `make_windows`).
2. Train generator (**target ≥3000 iterations** — đồng bộ `scripts/03_train_timegan.py`).
3. Sinh N window "fault giả": `generate(noise)` → reshape về `(N, 16)` (nếu train trên feature)
   hoặc `(N, 512)` (signal → sau này tính feature bằng `features.extract_features`).
4. Trong script 02: nếu đã là feature vector → `X_syn.astype(np.float32)`; nếu là signal → chạy
   `features.extract_features(X_syn, fs=12000)` rồi nạp.

> **Ghi chú tinh tế:** TimeGAN cho phép **điều kiện class** khá lỏng (chỉ một generator với
> phân phối trộn). Nếu thấy class fault quá lẫn với normal, hãy khắc phục bằng cách *oversample*
> fault thật trong bộ train của TimeGAN (upweight), đây là trick đúng paper FaultDiffusion đã nêu.

**Kiểm tra chất lượng sinh:** tái dùng 2 metric của paper — train classifier "real vs synth"
trên window/features; accuracy ≈ 0.5 → chất lượng tốt. Ghi số này vào slide.

---

### 4.3 Diffusion-TS — cách dùng

**Clone + môi trường GPU:**
```bash
cd A1_predictive_maintenance
mkdir -p vendor && cd vendor
git clone https://github.com/Y-debug-sys/Diffusion-TS.git
cd Diffusion-TS
python -m venv .venv-dts
source .venv-dts/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# kiểm tra GPU:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Dataset format của repo:** bỏ dữ liệu mình vào thư mục `Data/` theo định dạng chúng đọc
(`.npy` hoặc `.csv` 2D `(n_samples, seq_len*n_dim)` tùy tutorial); đọc `Tutorial_0/1/2.ipynb`
để biết chính xác.

**Chạy tutorial (từng bước)**
```bash
jupyter nbconvert --to script Tutorial_1*.ipynb --output forward_tutorial  # hoặc mở jupyter
python forward_tutorial.py       # train
python backward_tutorial.py      # sample -> lưu .npy
```
- `Tutorial_1` (forward): fit model + lưu `ckpt`.
- `Tutorial_2` (backward): sinh mẫu mới với guidance theo `conditional-column` nếu dataset của
  bạn có cột class.

**Sinh "fault giả" cho pipeline A1:**
1. Build dataset 2D: mỗi mẫu = window 512 của 1 kênh (hoặc window đa kênh flatted).
   - Nếu dùng signal thô: reshape về `(n, win)=512` → sau khi sinh chạy
     `features.extract_features` (fs=12000) để khớp input script 02.
   - Nếu dùng feature: tự tính trước rồi lưu thẳng `.npy` dạng `(n,16)`.
2. Train (khuyến nghị **có guidance theo class 0/1**: normal / fault) — chỉ có vậy khi train
   xong guide `sample(num_cond=1)` cho hàng loạt "fault".
3. Lưu `np.save("synthetic_faults.npy", arr_synth)` với `arr_synth.shape[1]==16` (feature).

**Feed vào script 02:**
```bash
python scripts/02_compare_augmentation.py --dataset ims --gen npy \
    --gen-npy results/synthetic_faults.npy --limit 300
```

> **Checklist nhập môn Diffusion-TS:** (1) khớp chiều dài chuỗi → win=512; (2) có cột class
> không? (không thì tự thêm label fault=1 cho synthetic); (3) giá trị scale hợp lý (chuẩn hoá
> về ~0–1 hoặc z-score trước khi train diffusion); (4) máy GPU ≥8GB VRAM cho seq dài.

---

### 4.4 Xung đột môi trường — mẹo vận hành

| Khu | Dùng gì |
|---|---|
| `.venv/` (team) | scipy, torch CPU/GPU, sklearn — mọi script A1 chính |
| `vendor/TimeGAN/.venv-tf1` | TF1-API — chỉ để sinh `.npy` |
| `vendor/Diffusion-TS/.venv-dts` | torch + các dep của repo — sinh `.npy` |

Khi scripting tự động: mỗi bước sinh nên **export** ra `.npy` rồi làm việc với file, không import
chéo — tránh lệ thuộc môi trường của nhau.

---

## 5. HƯỚNG ĐI KỸ THUẬT 2 NHÁNH (đánh giá kỹ + đề xuất)

### 5.1 Nhánh 1 — Unsupervised anomaly + threshold (baseline bắt buộc)

**Mô tả pipeline đang có** (`scripts/01_baseline_anomaly.py`):
1. Chuyển mỗi chuỗi rung thành windows `(win=512, stride=256)` → features 16 chiều
   (`src/features.py`).
2. Train `FeatureAE` **chỉ trên window normal** (MSE trên reconstruction + ReLU MLP 16→32→8→32→16).
3. Tính reconstruction error trên toàn bộ (normal + test fault); ngưỡng = **P99 của error
   normal-train** (`threshold_from_err(rec_err, 99.0)`).
4. Báo cáo precision/recall/F1/AUC trên test có fault thật.

**Cách chọn ngưỡng — đúng bài toán:**
- P99 (mặc định): cho phép 1% normal bị báo động giả → xem precision. Nếu recall quá tệ (như
  kết quả hiện hành 0.031), thử P95, P97, P99.5 → vẽ **precision-recall curve khi quét percentile**
  để chọn điểm "đủ recall, đủ precision" (đây là biểu đồ giám khảo thích xem).
- Mục tiêu không phải recall 1.0: mỗi mức ngưỡng là một trade-off; nêu rõ team chọn ngưỡng theo
  chi phí giả định (vd: 1 lỗi bỏ lọt = phí sửa máy X; 1 báo giả = phí stop-line Y).

**Nhược điểm / hạn chế (bắt buộc kể trong slide):**
- AE học "xấp xỉ bình thường" → **vùng suy giảm SỚM** (fault mới chớm, gần normal) có
  reconstruction error gần bằng normal → ổ bi suy giảm chậm sẽ rơi dưới ngưỡng → recall thấp.
- AE không dùng thông tin thời gian (window độc lập) → tín hiệu "đỏ dần theo thời gian" bị bỏ.
- Ngưỡng tĩnh theo percentile → không thích ứng máy chạy load đổi (CWRU nhiều RPM).
- Precision thấp khi fault hiếm (class imbalance nặng) — default threshold phóng đại số "anomaly".
- Chỉ phù hợp khi **không có nhãn lỗi** (phase 1 của sản phẩm; bộ cảm biến mới lắp, chưa thu
  được lỗi). Đúng là deliverable 01 nhưng **không phải đích chiến thắng**.

**Khi nào dùng nhánh này:** bắt buộc làm để lấy baseline con số. Đồng thời là "vật cản" so sánh
trong slide (trước augmentation) — phần "trước" của báo cáo.

---

### 5.2 Nhánh 2 — Augment + supervised classifier (KHẢ NĂNG THẮNG CAO)

**Ý tưởng:** Dùng normal (nhiều) + **rất ít fault thật (few-shot)** huấn luyện bộ sinh (GAN/
diffusion) → sinh *nhiều* window lỗi giả → gộp vào train → huấn luyện classifier supervised
(RandomForest/XGBoost trên 16 features, hoặc 1D-CNN) → **đánh giá trên test = normal-unseen +
fault THẬT chưa từng thấy**.

**Pipeline hiện có khớp 100%** (`scripts/02_compare_augmentation.py`):
- Test-split: normal 80% train / 20% test-unseen; fault 20% train-scarce / 80% test-thật.
- Base model: chỉ normal + ít fault thật (trước). Augmented model: thêm synthetic fault (sau).
- Metrics: precision/recall/F1/AUC trên *cùng* test.
- Cờ `--gen heuristic` (nhanh, protocol demo) và `--gen npy` (synthetic thật từ diffusion).

**Điểm chống gian lận — phải quán triệt cả đội:**
1. **Test là lỗi THẬT, chưa từng thấy** — không xào synthetic vào test, không dùng chính
   sample train để "đoán".
2. **Generator chỉ dùng tập train** (20% fault + normal-train): tuyệt đối không train generator
   trên window fault thuộc tập test (nếu generator nhìn test → đo trước/sau là trò cười).
3. **Không có bóng dáng**: synthetic sinh từ generator đã học train → nếu test có sample của
   cùng ổ bi nhưng khác time, vẫn được; nhưng **không được sinh từ chính sample test**.
4. **Metric cần báo cả recall & precision** — đừng chỉ AUC (AUC kể chuyện thiếu trong scenario
   hiếm fault: bộ classifier tối ưu AUC vẫn có thể zero recall ở ngưỡng mặc định).
5. **Stratified k-fold** hoặc hold-out theo thời gian (train trước, test sau) — đúng bạn đã làm
   trong `_load_pipeline` (split theo thứ tự mảng), nói rõ trong slide để chứng minh sạch.

**Cách sinh giả 3 mức (từ rẻ đến đắt — phân ở từng tuần):**
| Mức | Cách sinh | Phần cứng | Chất lượng | Dùng khi nào |
|---|---|---|---|---|
| heuristic | trộn noise + nhân biên độ lên normal (`0.8` noise, x2.5 biên độ) | CPU | thô, "linearly separable" | prototype protocol (đã có — gain ≈ 0, đừng dùng làm số chính) |
| TimeGAN | GAN TS với autocorrelation loss | CPU | trung bình, ít mode diversity | không có GPU, cần số tạm (`scripts/03`, ≥3000 iter) |
| Diffusion-TS / FaultDiffusion | diffusion condition-by-class/few-shot | GPU | cao nhất, bám phân phối thật | demo chính + báo cáo cuối |

**Trọng tâm báo cáo "trước/sau" (deliverable 3):**
- Bảng: Precision / Recall / F1 / AUC × (before, after, Δ).
- PR curve (2 đường) — điểm nhìn quan trọng nhất; nếu synthetic tốt, đường "after" nằm phía
  trên bên phải với recall cao hơn tại cùng precision.
- Bổ sung 2 metric của TimeGAN (discriminative/predictive) để chứng minh "synthetic = thật".
- Ảnh so sánh: a) spectrogram 1 window lỗi thật vs 1 window lỗi giả (nhìn bằng mắt giống);
  b) t-SNE/Umap của normal/fault-thật/fault-giả để thấy fault-giả bao phủ vùng fault-thật.

**Câu "bản chất gain" trả lời giám khảo:** trước → model chỉ thấy vài mẫu fault, tỉ lệ nhất
định giỏi nhất cũng chỉ là must-chance ≈ 1/6000 (imbalance) → recall ~0. Sau → model thấy đủ
dạng fault → recall nhảy lên. Nếu gain không như kỳ vọng, đừng xóa số liệu — **báo "không
đổi/giảm nhẹ" kèm lý do** (bộ sinh synthetic trùng dạng, threshold chưa xoay) cũng là kết quả
trung thực hợp lệ. Thực tế 30/08: heuristic cho Δ recall ≈ 0.005 → đúng vậy, đó là động lực
nâng cấp generator chứ không phải bug.

---

### 5.3 Phân tích khả thi theo thời gian / GPU / nhân lực

| Nhánh | Thời gian hiện thực | GPU cần? | Nhân lực (5 người) | Mức rủi ro |
|---|---|---|---|---|
| Baseline AE (nhánh 1) | ~1–2 ngày (đã có code chạy) | không (torch CPU OK) | 1 người | thấp |
| Heuristic augmentation | ~0.5 ngày (đã có) | không | 1 người | không |
| TimeGAN sinh fault | 3–6 ngày (tuning, ≥3000 iter) | không bắt buộc | 1 người | trung bình (TF1 cũ, venv riêng) |
| Diffusion-TS sinh fault | 5–10 ngày (train + tune seq_len) | **bắt buộc GPU ≥8GB** | 1–2 người | trung bình → cao |
| Báo cáo + slide | 2–3 ngày | — | cả đội | thấp nếu chạy sớm |

**Khuyến nghị cuối cùng (chốt):** chạy NHÁNH 1 + heuristic ngay tuần 2–3 để có số; đặt GPU cho
Diffusion-TS tuần 6; nếu hết GPU hồi tuần 8 → fallback TimeGAN CPU + kiểm soát câu chuyện bằng
2 metric của TimeGAN. **Không bao giờ để "số slide" chờ diffusion**; bảng trước/sau của heuristic
đã đủ minh chứng protocol, diffusion chỉ làm số đẹp hơn.

---

## 6. KẾ HOẠCH CHI TIẾT THEO TUẦN (26/08 → 12/10/2026)

> Lưu ý: hôm nay 30/08 — **tuần 1 đang ở ngày cuối**. Mốc CỨNG đầu tiên là **đăng ký 31/08 (mai)**.
> Factory Tour 11/09, nộp Vòng 1 12/10, chung kết 16/11.

### 6.1 Bảng timeline tổng quan

| Tuần | Ngày (2026) | Giai đoạn | Việc chính | Đầu ra (exit criteria) |
|---|---|---|---|---|
| 1 | 26/08–31/08 | Khởi động + đăng ký | đăng ký đội (**≤31/08**), lập repo, chốt stack, tải data | repo sạch, data sẵn, **đội đăng ký thành công** |
| 2 | 01/09–07/09 | Baseline | feature + AE + ngưỡng | `results/baseline_ae.json` + PR curve |
| 3 | 08/09–11/09 | Baseline hoàn thiện | quét ngưỡng P95–P99.5, thêm dashboard sơ bộ | bảng confident, dashboard MVP, chuẩn bị câu hỏi tour |
| 4 | 11/09 (Bootcamp + Factory Tour) | *Học từ nhà máy* | hỏi kỹ sư DENSO, ghi chép, chỉnh hiểu biết về "lỗi thật" | danh sách Q&A, refinements process |
| 5 | 14/09–20/09 | Augmentation heuristic | gắn pipeline 02 + heuristic lên 3 dataset | bảng compare heuristic trước/sau |
| 6 | 21/09–27/09 | Sinh synthetic (GPU) | clone Diffusion-TS + TimeGAN (≥3000 iter), sinh fault giả | synthetic .npy đầu tiên (vài nghìn window) |
| 7 | 28/09–04/10 | Sinh synthetic hoàn thiện | tune, sinh đủ, evaluate bằng 2 metric TimeGAN | siêu chất lượng synthetic + discriminative score |
| 8 | 05/10–09/10 | Báo cáo + dashboard | composite báo cáo trước/sau, dashboard RUL+alert | báo cáo PDF/dashboard hoàn chỉnh |
| 9 | 10/10–12/10 | Slide Vòng 1 + nộp | viết slide, chạy lại số liệu sạch, nộp | file nộp 12/10 |

---

### 6.2 Tuần 1 (26/08–31/08) — Khởi động & ĐĂNG KÝ

**Mục tiêu không thể trượt:** đăng ký xong trước 31/08 trên **densohackathon.vn** (1 tài khoản đội
= 1 bài nộp — đội 3–5 người; xác định ai là owner tài khoản). **Đã ở ngày cuối tuần 1 (30/08) —
ưu tiên hàng đầu hôm nay/mai.**

**Checklist:**
- [ ] **Đăng ký đội + điền ý tưởng sơ bộ (deadline 31/08).** — LÀM NGAY.
- [ ] Git repo tồn tại; README + `.gitignore` (data/ bỏ qua; data lớn 300MB+FEMTO không lên git).
- [ ] Xác nhận 3–5 thành viên + phân vai: (a) Data & feature, (b) Mô hình baseline/AE,
      (c) GAN/Diffusion pipeline, (d) Dashboard/frontend, (e) Slide + tài liệu. Dù vai nào
      cũng phải đọc `RESOURCES.md`.
- [ ] Chạy `bash scripts/download_data.sh` (idempotent — tự bỏ qua phần đã có); kiểm tra
      `data/NASA_IMS/test_{1,2,3}/` và `data/FEMTO/`.
- [ ] CWRU: tải tay từ trang chính thức phần fault (105, 118, 130…); **bỏ qua normal bị hỏng**
      nếu vẫn không lấy được — dùng IMS làm nguồn normal.
- [ ] Chạy thử `python scripts/01_baseline_anomaly.py --dataset ims --limit 200` → có JSON.
- [ ] Chạy thử `python scripts/02_compare_augmentation.py --dataset ims --gen heuristic --limit 200`
      → hiểu protocol.
- [ ] Trả lời được bảng câu hỏi "đúng giám khảo": deliverable là gì? leakage là gì trong bối cảnh này?

**Vai không được để thiếu:** người viết slide đọc toàn bộ catalog paper ngay tuần 1 để slide
không "đoán paper".

---

### 6.3 Tuần 2–3 (01/09–11/09) — Feature + AE baseline

**Tuần 2:** hoàn thiện baseline trên IMS + CWRU.
- [ ] Chạy baseline với vài cấu hình `--win`(256/512/1024) × `--stride`(128/256) × `--thr`
      (95/97/99/99.5) → ghi 1 bảng sweep.
- [ ] Vẽ **PR curve** của AE (quét ngưỡng) → chọn ngưỡng hợp lý (không cứng P99).
- [ ] Ghi `results/` nhiều JSON để slide sau này "kể" quá trình chọn ngưỡng.
- [ ] (Nếu có) thử `--dataset cwru` (chỉ fault để xem recall — normal không đọc được thì bỏ).

**Tuần 3:** dọn dashboard MVP (alert line + RUL ước lượng) + chuẩn bị Factory Tour.
- [ ] Dashboard: hiển thị score bất thường theo thời gian, đánh dấu vượt ngưỡng, đếm trong
      test thực.
- [ ] Thu thập 10 câu hỏi Factory Tour (xem §7.4) — giao cho 2 người có mặt.
- [ ] Tự set expectation: số AE hiện tại (AUC 0.52) sẽ TĂNG khi dùng supervised-aug (đó là
      câu chuyện chính, không phải bug).
- [ ] Viết nháp intro slide survey "scarce fault data in manufacturing" (thu thập 3–5 số liệu
      công bố để mở đầu).

**Exit criteria:** bảng sweep ngưỡng + PR curve saved trong `results/`; dashboard chạy được cục bộ.

---

### 6.4 Tuần 4 (11/09) — Bootcamp + Factory Tour: Hỏi kỹ sư DENSO

**Đây là "phòng thí nghiệm minh bạch" duy nhất — tận dụng tối đa.** Mục tiêu: thu được câu
trả lời để (a) chỉnh hướng kỹ thuật, (b) làm slide thấm thía bài toán thực.

**Checklist trước tour:**
- [ ] In sẵn 10 câu hỏi + bảng so sánh dataset (mang theo file này).
- [ ] Phân công người ghi chép (answers log) — không để cuộc trò chuyện trôi.
- [ ] Chuẩn bị 1-pager mô tả protocol trước/sau để kỹ sư góp ý trực tiếp.

**Sau tour (ngay trong tuần 4):**
- [ ] Kết xuất Q&A log bằng markdown trong repo (`notes/factory_tour_notes.md`).
- [ ] Map 3 câu hỏi quan trọng nhất vào pipeline (nếu câu trả lời đổi thiết kế: ví dụ
      "sensor nào ít nhiễu nhất khi có lỗi" → chọn kênh, "bao lâu giữa cảnh báo → hỏng" → nhịp
      threshold).
- [ ] Bổ sung phần "Factory insight" vào slide (điểm cộng: chứng minh đội nghe và áp dụng).

---

### 6.5 Tuần 5–9 (14/09–09/10) — Augmentation + So sánh + Dashboard

**Tuần 5 — pipeline augmentation heuristic chạy trên 2–3 dataset.**
- [ ] Chạy `--gen heuristic` trên `ims` và `cwru` (cwru: `--limit` flood? xem hàm `_load_pipeline`
      có hỗ trợ cwru — có nhánh cwru). Ghi 2 bảng before/after.
- [ ] Tạo 2 biểu đồ PR curve trước/sau (lưu vào `results/*.png`).
- [ ] Xác nhận đường leak: file JSON có `"n_synth"`, `"n_test_fault"` — kiểm tra luôn test chỉ
      fault THẬT (không trùng train) bằng code check (người làm data viết assertion).

**Tuần 6 — synthetic thật (GPU):**
- [ ] Clone 2 repo, set môi trường (đúng bài học §4). Bật GPU ≥8GB (rủi ro tìm máy: xếp sớm).
- [ ] Fabric dữ liệu train generator từ IMS: normal 80% train, 20% fault thật (không chạm test).
- [ ] Output `.npy` thử nghiệm 1000–3000 window phần fault; chạy `02 --gen npy --gen-npy …`.
- [ ] Đo discriminative score (TimeGAN metric): trainer "real vs fake" classifier trên features
      → accuracy nên ~0.5–0.65 (đẹp là 0.5–0.6).

**Tuần 7 — tuning & mở rộng:**
- [ ] Tuning: seq_len, guidance-strength, % upweight fault thật lên 0.35–0.5 (follow FaultDiffusion).
- [ ] Sinh đủ tổng synthetic target: sao cho class balance trong train ≈ 1:1 (thay vì 1:6000) →
      recall tăng mạnh.
- [ ] Chạy 3 cấu hình generator → chọn tốt nhất bằng metric dưới; lưu tất cả trong `results/`.
- [ ] Có ít nhất 1 kết quả "2 metric TimeGAN trên synthetic của tôi" để nói trong slide.

**Tuần 8 — báo cáo cuối + dashboard hoàn chỉnh:**
- [ ] Ghost-run lại toàn bộ pipeline từ sạch (tải data → heuristic → synthetic → compare) với
      seed cố định → số cuối cùng quyết định.
- [ ] Dashboard bản cuối: chuỗi thời gian đa kênh + anomaly score + ngưỡng + alert + RUL.
- [ ] Báo cáo markdown/PDF với bảng before/after + PR curve + spectrogram(tín hiệu thật vs giả)
      + t-SNE + phần methodology/limitations.
- [ ] File `.npy` synthetic lưu trong `results/` (làm deliverable "tập dữ liệu tăng cường").

**Tuần 9 (10/10–12/10) — slide + nộp:** xem §7; chạy lại số liệu để mọi con số khớp file nộp.

**Exit criteria tổng:** `scripts/*` chạy end-to-end (README), 3 deliverable đủ, repository
"reproducible log" (mọi số trong slide được sinh từ command ghi trong README).

---

### 6.6 Checklist nộp cuối cùng (đối chiếu deliverable BTC)

- [ ] **Bộ sinh dữ liệu bất thường:** code + README hướng dẫn (diffusion/TimeGAN/heuristic) —
      kèm cách chạy.
- [ ] **Tập dữ liệu tăng cường:** `results/synthetic_faults.npy` + mô tả số lượng/đặc điểm.
- [ ] **Báo cáo trước/sau:** bảng Precision/Recall/F1/AUC + PR curve + n_synth/n_test_fault rõ.
- [ ] **Dashboard** alert + RUL.
- [ ] **Slide Vòng 1** (10–15 slide) — xem §7.
- [ ] File nộp đúng định dạng (pdf/slides) lên đúng tài khoản đội trước 12/10 23:59.

---

## 7. CHIẾN LƯỢC SLIDE VÒNG 1 (hạn 12/10)

### 7.1 Nguyên tắc vàng

- **10–15 slide, không hơn.** Mỗi slide 1 ý. Giám khảo đọc <60s/slide.
- **Số liệu là vua:** mọi nhận định phải có con số (`results/*.json`) đi kèm; hình minh họa
  (PR curve, spectrogram, t-SNE) thay lời.
- **Kể chuyện "kịch bản scarce data"** — mở đầu bằng đau thật (thiếu fault data), kết bằng
  protocol tái sử dụng được.
- **Cite paper hợp lệ:** FaultDiffusion (arXiv:2511.15174), TimeGAN (NeurIPS 2019), Diffusion-TS
  (ICLR 2024) — ghi đúng venue/ID, không đoán link.

### 7.2 Cấu trúc slide đề xuất

| # | Slide | Nội dung | Hình chính |
|---|---|---|---|
| 1 | **Title** | Tên đội, "Predictive Maintenance with Scarce Fault Data — AI Augmentation Approach" | logo + tên |
| 2 | **Problem** | Nhà máy: nhiều normal, gần như không có lỗi → baseline supervised chết; chi phí báo giả vs bỏ lọt | 1 con số fact + sơ đồ dòng |
| 3 | **Objective + Deliverables** | 3 deliverable của đề (sinh fault, dataset tăng cường, so sánh trước/sau) | 3 box |
| 4 | **Dataset** | CWRU / IMS / FEMTO — 1 bảng nhỏ (nguồn, thời lượng, nhãn). Nói rõ "IMS là main, CWRU phụ" | bảng + thumbnail tín hiệu |
| 5 | **Baseline approach (unsupervised)** | Autoencoder + reconstruction error + P99 → "trước" | sơ đồ AE + PR curve "before" |
| 6 | **Augmentation — the core** | Sinh lỗi giả bằng TimeGAN / Diffusion-TS / FaultDiffusion; heuristic làm crosscheck | sơ đồ pipeline sinh |
| 7 | **Augmentation protocol (anti-leak)** | test = normal-unseen + fault THẬT chưa thấy; generator train chỉ trên train; split theo thời gian | sơ đồ split |
| 8 | **Baseline metrics (before)** | bảng + PR curve "before" + scan ngưỡng P95–99.5 | bảng số + PR |
| 9 | **Augmented metrics (after)** | bảng + PR curve "after" + đồ thị so | 2 đường PR chồng |
| 10 | **Synthetic quality** | discriminative/predictive score (TimeGAN), t-SNE normal/fault-thật/fault-giả | t-SNE + spectrogram thật vs giả |
| 11 | **Why it works** | few-shot conditioning của FaultDiffusion → synthetic bám dạng lỗi thật; ít noise mode-collapse | câu + 1 hình |
| 12 | **Limitations + Risks** | threshold tĩnh, máy khác sensor, GPU cần thiết → nêu trung thực + direction | 3 bullet |
| 13 | **Roadmap → RUL + next** | từ anomaly → RUL (FEMTO/C-MAPSS), dashboard alert | timeline mini |
| 14 | **Team & Plan** | 3–5 người, roles, sprint 9 tuần | bảng team |
| 15 | **Thank You / Q&A** | link repo demo + invite hỏi | — |

> Có thể gộp 12–13 nếu cần giữ 13 slide; đừng thêm slide "About us" dài.

### 7.3 Điểm "ăn điểm" quan trọng

1. **Frame "scarce data" làm research chủ đề:** mở đầu bằng phát biểu nhận thức (có survive thị
   trường AI công nghiệp: rất ít bài làm đúng scarce-data; đa số cheat). Trung thực = khác biệt.
2. **PR curve TRƯỚC/SAU:** một dòng "Recall 0.01 → 0.8, same precision 0.7" đáng giá 1000 chữ.
   Đảm bảo đường sau nằm phía trên phải. *(Số thật hiện tại: RF recall 0.477 → 0.482 — chưa đủ
   ấn tượng; phải nâng bằng XGBoost tuning + TimeGAN/diffusion trước khi đưa lên slide.)*
3. **Trích dẫn paper đúng:** FaultDiffusion (arXiv:2511.15174) cite ở slide 11 có weight hơn
   nếu bạn ghi thêm "method we adapted for few-shot conditioning". Dùng 2 metric TimeGAN ở
   slide 10 → chứng tỏ biết cách đánh giá generative model.
4. **Một slide "leakage check" sẽ gây ấn tượng mạnh:** "chúng tôi chủ động auto-assert rằng
   test chỉ có fault thật" + code snippet nhỏ.
5. **Con số nhất quán repo↔slide:** chạy cùng command, seed cố định; slide ghi đúng JSON.
6. **Dashboard mini demo chạy trực tiếp** (nếu cho phép) — live demo ứng dụng vào vòng chung
   kết (16/11), nên "chỉ demo an toàn" (không phụ thuộc mạng).

### 7.4 Câu hỏi nên hỏi kỹ sư DENSO tại Factory Tour (11/09)

Không hỏi câu "set up giúp em" — hỏi câu làm lộ **đặc tính lỗi thật** để đội tinh chỉnh dataset:

1. "Khi máy bắt đầu hỏng (vỡ ổ bi / lệch trục), sensor nào cho tín hiệu rõ NHẤT trước tiên:
   rung hướng nào, tần số nào?" → chọn kênh + band tần số trong features.
2. "Khoảng cách giữa dấu hiệu cảnh báo sớm và hỏng hẳn là bao lâu (giờ/ngày) trong dây chuyền
   của anh chị?" → config ngưỡng + cửa sổ dự báo.
3. "Anh chị có dữ liệu lỗi cũ nào (history log) không? Khoảng bao nhiêu % tổng dữ liệu?" →
   xác định mức "scarce" thực tế; nếu họ có cả TB normal + TB fault thì bài toán ít khắc nghiệt hơn.
4. "Nhiễu vận hành bình thường khác với lỗi thế nào (rung bình thường đã cao do máy chạy
   nhiều trục)?" → tune baseline tránh báo giả.
5. "Cách thu thập data lỗi hiện tại là thủ công hay tự động; mỗi lỗi ghi được bao nhiêu giây?" →
   đánh giá khả thi mở rộng method để giám khảo tin "protocol tái sử dụng".
6. "Loại lỗi phổ biến nhất của thiết bị rung tại DENSO là gì (bearings, gears, misalignment)?" →
   chọn dataset CWRU/FEMTO hay cần thêm loại lỗi khác.
7. "Hệ thống hiện tại có cảnh báo gì chưa (thủ công theo lịch định kỳ?) → muốn AI thay/nâng?" →
   vị trí sản phẩm của AI trong quy trình.
8. "Tiêu chí KPI của họ chấp nhận bao nhiêu báo giả / cho phép bỏ lọt bao nhiêu lỗi?" —
   đặt ngưỡng đúng bài toán (nếu trả lời "0 bỏ lọt" → threshold rất thận trọng; "ít báo giả" → PR trade-off).
9. "Có thu được tín hiệu trước-sau bảo trì khi thay ổ bi mới không (để làm augmentation chuẩn
   hơn)?" → insight "normal mới" so với "normal mòn dần".
10. "Nhân viên vận hành có kinh nghiệm nhận diện âm thanh/mùi trước khi hỏng không — có thể gán
    nhãn phụ không?" → phương án bổ sung nhãn thô.

**Cách dùng các câu trả lời:**
- Mỗi câu trả lời ghi vào `notes/factory_tour_notes.md` kèm action trong pipeline (vd: câu 1 →
  thêm channel-weighted feature; câu 4 → thêm band-pass filter cho normal-nuance).
- Nhấn mạnh 2–3 insight hay nhất ngay trong slide 12 "Factory-driven refinement" — đây là điểm
  riêng biệt team có mà nhóm khác không.

---

## 8. QUYẾT ĐỊNH NHANH (DECISION GUIDANCE)

Dành cho kỹ sư cần chọn nhanh "dùng cái gì" mà không phải đọc lại toàn bộ. Map trực tiếp với
6 bài toán con [a]–[f] trong Bản đồ tư duy A1.

### 8.1 Bảng chọn mô hình theo bài toán con

| Bài toán con | Chọn mặc định | Khi nào đổi | Trỏ tới (code) |
|---|---|---|---|
| [a] Nén tín hiệu | 16 features, `win=512, stride=256` | window ngắn → `win=1024`; nhiễu tần số cao → thêm band-pass | `src/features.py`, `src/pipeline.py` |
| [b] Định nghĩa tốt/xấu | PR-AUC + precision/recall/F1, ngưỡng P99 | có chi phí thực (báo giả vs bỏ lọt) → chọn ngưỡng theo chi phí | metrics trong `scripts/01`, `scripts/02` |
| [c] Không có nhãn | Autoencoder + reconstruction error | AE dưới trung bình → IsolationForest / OC-SVM | `scripts/01` (AE) |
| [d] Ít nhãn | RandomForest trên 16 features | cần chính xác hơn / tuning → XGBoost | `scripts/02` |
| [e] Sinh lỗi giả | TimeGAN (CPU) ≥3000 iter | có GPU → Diffusion-TS (condition-by-class); sát đề → FaultDiffusion few-shot | `scripts/03` · `vendor/` |
| [f] Chứng minh trước/sau | `--gen heuristic` + 3-zone split | có synthetic .npy → `--gen npy` | `scripts/02`, `src/pipeline.py` |

### 8.2 Decision guidance — luồng quyết định (đọc theo thứ tự)

1. **Đăng ký đội chưa?** Hạn 31/08 (mai). Chưa → dừng mọi việc, đăng ký trước.
2. **Có nhãn lỗi không?**
   - Không → chạy AE baseline (`scripts/01`), ngưỡng P99. Đó là cột mốc "trước" (deliverable 3).
   - Có (ít) → sang bước 3.
3. **Có GPU CUDA ≥8GB không?**
   - Không → TimeGAN CPU (`scripts/03`, ≥3000 iter) sinh `.npy`; heuristic làm đối chứng thêm.
   - Có → Diffusion-TS condition-by-class sinh fault giả; FaultDiffusion nếu trả lời được câu
     "synthetic bị tách ra ngay không".
4. **Cần số nhanh?** → heuristic trước, synthetic sau. Không để số slide chờ GPU.
5. **Bất kỳ con số trước/sau nào** cũng phải đi qua 3-zone split, generator chỉ học train, test
   chỉ lỗi THẬT. Vi phạm → số bỏ đi (leakage).
6. **Chọn đáp án "rẻ nhất" đủ chứng minh protocol**, nâng cấp sau nếu còn thời gian/GPU.

---

## PHỤ LỤC A — Các lệnh chạy end-to-end (để tái lập số liệu trong slide)

```bash
# Bước 0 — chuẩn bị
source .venv/bin/activate

# Bước 1 — data
bash scripts/download_data.sh                        # idempotent: NASA IMS + FEMTO
# (CWRU tải tay từ trang chính thức; normal hỏng → fallback IMS)

# Bước 2 — baseline (mục tiêu "trước")
python scripts/01_baseline_anomaly.py --dataset ims --limit 300 \
    --out results/baseline_ae_ims300.json
# quét ngưỡng
python - <<'PY'
from src import ae, data, features
import numpy as np, glob
# ... (sweep threshold bằng nội bộ src, xem README)
PY

# Bước 3 — augmentation protocol (heuristic — chạy nhanh, số tạm)
python scripts/02_compare_augmentation.py --dataset ims --gen heuristic --limit 300 \
    --out results/compare_heuristic.json

# Bước 3b — TimeGAN (CPU, ≥3000 iterations) → sinh .npy
python scripts/03_train_timegan.py --dataset ims ...
#   output -> results/synthetic_faults.npy
python scripts/02_compare_augmentation.py --dataset ims --gen npy \
    --gen-npy results/synthetic_faults.npy --limit 300 \
    --out results/compare_timegan.json

# Bước 4 — synthetic thật (sau khi có GPU; xem §4)
#  vendor/Diffusion-TS/... sample -> results/synthetic_faults.npy
python scripts/02_compare_augmentation.py --dataset ims --gen npy \
    --gen-npy results/synthetic_faults.npy --limit 300 \
    --out results/compare_diffusion.json
```

---

## PHỤ LỤC B — Những cạm bẫy dễ khiến đội "ăn điểm xấu" (đọc 1 lần/tuần)

1. **Leak**: generator nhìn thấy window test; hoặc split window cùng file vào cả train/test.
2. **Số liệu đẹp giả**: train-test chung file CWRU; hoặc báo AUC mà không báo recall/precision.
3. **Synthetic "giả vờ nhưng chấm sai"**: model phân loại real-vs-fake quá dễ (accuracy ~1.0)
   → nêu trong slide như warning.
4. **Quên đăng ký 31/08** → cả dự án vô nghĩa (đặt reminder + chốt owner).
5. **FEMTO/C-MAPSS vướng format**: load bằng `.csv`; mọi cột khớp; không để lẫn file zip.
6. **GPU tìm muộn** → diffusion trễ → fallback TimeGAN CPU kịp; đừng chờ GPU đến phút chót.
7. **Bỏ mặc venv**: 3 môi trường (team / TimeGAN / Diffusion-TS) tách bạch,
   không `pip install` lung tung và làm hỏng `.venv/` chính.
8. **Bồn chồn vì gain heuristic ≈ 0** → đó là bằng chứng cần generator thật, đưa thẳng vào slide
   phần "motivation" thay vì giấu.

---

## PHỤ LỤC C — Tài liệu nên đọc lại theo vai

| Vai | Đọc cái gì |
|---|---|
| Cả đội | README.md, RESOURCES.md, file này mục 1–5 |
| Data & feature | file này mục 2 (dataset), 5.1 (threshold) |
| Mô hình/ML | file này mục 3–4 (paper + repo), 5.2 |
| Dashboard | file này mục 6.3/6.5 (dashboard spec) |
| Slide | file này mục 1.5 (mùa trước), 3 (trích dẫn), 7 (slide) |
| Người hỏi DENSO | mục 7.4 (câu hỏi tour) |
| Kỹ sư quyết định nhanh | file này mục 8 (decision guidance) + Bản đồ tư duy A1 ở mở đầu |

---

*Tài liệu khép chặt với mã nguồn hiện tại của team (A1_predictive_maintenance) và đã dùng đúng
số liệu thật từ `results/` (cập nhật 30/08/2026). Mọi link trong tài liệu này đều đã được verify
tại ngày 08/2026.*