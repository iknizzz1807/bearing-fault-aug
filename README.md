# Dự án A1 — Predictive-Maintenance AI Despite Scarce Fault Data

AI bảo trì dự đoán khi **thiếu dữ liệu lỗi**.

## Bài toán
- Máy móc chạy liên tục, gắn cảm biến (rung / nhiệt độ / dòng điện).
- Nhiều dữ liệu "máy khỏe" nhưng **rất ít dữ liệu "máy sắp hỏng"** → không đủ ví dụ lỗi thật để train supervised thông thường.
- Cần: cảnh báo sớm + ước tính tuổi thọ còn lại (RUL), ít báo động giả.

## Input / Output / Goal
- **Input:** chuỗi thời gian đa biến (vibration, current, nhiệt độ).
- **Output:** cảnh báo normal(0) / fault(1) + RUL.
- **Goal:** recall cao (bắt sớm) nhưng precision cao (ít báo giả) — đánh đổi kinh điển.

## Cách tiếp cận
So sánh nhiều trục: baseline (AE/RF), augmentation (heuristic / TimeGAN / physics),
feature engineering, deep temporal. Báo cáo trước/sau trên cùng test **chống leakage**.

> **KẾT QUẢ THỰC NGHIỆM (01/09/2026, đã sửa leak — số tuyệt đối):**
> - **Baseline RF + feature đạt trần khi dữ liệu lỗi đủ**: IMS recall **0.886** / AUC **0.982**; CWRU **1.0 / 1.0**.
> - **Augmentation CÓ GIÁ TRỊ khi dữ liệu lỗi THẬT SỰ khan hiếm**: với **bộ sinh tối ưu (interpolate lỗi thật) + đủ số lượng lỗi giả**:
>   - 50 lỗi train → recall **0.490 → 0.778**; 20 lỗi train → recall **0.383 → 0.738**; 10 lỗi → 0.252 → **0.689** (interp_align+amp, §3.4.6c).
> - **Bộ sinh tối ưu = sinh lỗi giả từ chính các lỗi thật (interpolation)**, KHÔNG phải bơm vào nền normal:
>   interpolate > heuristic (0.46) > **TimeGAN/generative (0.43)** ≈ physics (0.42) ở kịch bản khan hiếm.
> - **Hướng tối ưu thêm đã kiểm chứng** (§3.4.1): **tăng số lỗi giả** giúp rõ (800→3200: 0.649→0.685; K=50 còn 0.72→0.80);
>   **mix thêm physics đa tần số** giúp chút (0.677→0.682). Các hướng "lan rộng" (k=3/scale rộng/nhiễu mạnh) đều phản tác dụng.
> - **Nhưng khi dữ liệu lỗi ĐÃ đủ (≥100–200 lỗi)** thì baseline chạm trần và augmentation gần như không đóng góp.
> - **Deep temporal (LSTM/CNN) KÉM hơn RF** (recall ~0.43).
> - **Model mới / bộ sinh phức tạp KHÔNG thắng khi feature kỹ thuật đã đủ mạnh**: self-supervised
>   contrastive kém hơn feature 24D (0.358 vs 0.418, §3.5); ECDF/KS distribution alignment không thay
>   được interpolate (0.698 vs 0.718, §3.4.2); DDPM/diffusion KÉM NHẤT (0.43–0.59, §3.4.4, loss phẳng
>   không học được từ 20–50 mẫu); TTA và ensemble đều không giúp (script 09, §3.4.3). Xác nhận: nút thắt
>   là **dữ liệu lỗi khan hiếm**, không phải model hay bộ sinh.
> - **Cải tiến mới NHẤT (đáng giá nhất): `amp_align`** = interpolate + chuẩn biên độ theo phân bố
>   std lỗi thật. Bù đúng điểm yếu của interpolate (bản trộn quá "dịu", thiếu amplitude cực đoan):
>   **K=10 +0.070 → 0.61**, **K=20 +0.051 → 0.78**, K=50 hòa ~0.79. **Càng hiếm càng giúp** (§3.4.3).
> - **MỚI NHẤT: `interp_align` (phase-aware mixup)**: interpolate nhưng **căn pha 2 cửa sổ lỗi bằng
>   FFT cross-correlation trước khi trộn**, khử destructive interference làm interpolate sinh bản
>   mềm. Cộng chuẩn biên độ (`--amp`): **K=10 → 0.69, K=20 → 0.74** — **tốt nhất ở vùng đề bài quan
>   tâm (10–20 lỗi)**; K=50 ~0.78. **PHƯƠNG ÁN CHÍNH THỨC** (§3.4.5, số đã sửa leak §3.4.6c).
> - **AUDIT code (4 subagent + verify thủ công) — ĐÃ SỬA 2 BUG, số tuyệt đối trong SPEC §3.4.6c**:
>   **① leak train/test** trong `_raw_train_windows` (bộ sinh lấy lỗi thật từ CẢ test; split fault
>   là window-level) → làm recall "sau" phồng lên, giờ đã sửa (K=20: 0.786→**0.738**);
>   **② physics nhận FEATURE 24D làm tín hiệu** → kết luận cũ "physics không giúp" do code sai,
>   đã sửa (physics K=20: 0.432→0.440, vẫn thua interpolate). **Thứ tự bộ sinh không đổi**, mọi
>   kết luận định tính về "interpolation thắng khi khan hiếm" giữ nguyên giá trị.
> - Kết luận chính: **augmentation sinh dữ liệu lỗi có ý nghĩa đúng khi lỗi thật hiếm (10–50 mẫu)**
>   — đây là kịch bản mà đề bài quan tâm — và **cách tốt nhất là interpolation từ lỗi thật
>   + căn pha cross-correlation + chuẩn biên độ (`--gen interp_align --amp 1`), sinh đủ số lượng
>   (n_synth=3200)**. Chi tiết: `SPEC.md §3.4/§3.4.1/§3.4.2/§3.4.3/§3.4.4/§3.4.5/§3.4.6/§3.5`.
> - **Đã kiểm chứng thực nghiệm các feature "vô địch Kaggle"**: feature khác biệt của VSB Power Line
>   1st (peak height std/mean, envelope) tách mạnh nhưng **trùng thông tin** với feature 24D đã có, và
>   `sawtooth_rmse` (feature #1 của họ) **không chuyển được** sang window 512 (sep=0.06) → thêm đều
>   nằm trong nhiễu (±0.004). Kết luận: bài ổ bi khác bài partial-discharge; phần dư địa nằm ở **số
>   lượng mẫu lỗi**, không ở feature (SPEC §3.1.1/§3.1.2).
> - **n_synth CÓ GIỚI HẠN (KHÔNG đơn điệu)**: sinh 8000 lỗi giả từ chỉ 10 lỗi thật → các bản trộn
>   trùng manifold, ghi đè 10 mẫu gốc → recall **TỤT −0.143 (0.632→0.489 ở K=10)**, tái lập được;
>   K=20/50 chỉ +0.02/+0.01 (trong nhiễu). → **Giữ n_synth=3200 là tối ưu** (§3.4.6a).
> - **Wavelet denoise kiểu LANL (db4+universal threshold) KHÔNG giúp**: +0.008 trong nhiễu ở K=20;
>   và ngưỡng tổng quát của LANL (segment 150k) quá mạnh cho cửa sổ 512 → phải giảm `mult` (§3.4.6b).

## Cấu trúc repo (thực tế)
```
A1_predictive_maintenance/
├── README.md          # tổng quan + cách chạy
├── RESOURCES.md       # data/paper/repo đã verify link
├── SPEC.md            # hướng đi + toàn bộ kết quả thực nghiệm (đọc đầu tiên)
├── requirements.txt
├── src/               # code tái sử dụng
│   ├── data.py        # loader IMS / CWRU (bao gồm hàm đếm+load vùng) / FEMTO
│   ├── features.py    # 24 feature (thời gian + tần số + envelope spectrum)
│   ├── ae.py          # autoencoder + metrics (precision/recall/F1/AUC)
│   ├── physics.py     # MÔ PHỎNG VẬT LÝ: BPFO/BPFI/BSF/FTF + sinh lỗi giả
│   ├── generator.py   # BỘ SINH TỐI ƯU: interpolate + amp_align + interp_align (căn pha) + spectral_mixup + ECDF/KS align
│   ├── contrastive.py # self-supervised SimCLR encoder (thử — kém feature 24D)
│   ├── ddpm.py        # DDPM nhẹ sinh lỗi (thử — kém nhất khi khan hiếm)
│   ├── timegan_torch.py  # TimeGAN bản PyTorch (chạy CPU/GPU)
│   ├── TimeGAN/       # clone tham khảo
│   └── Diffusion-TS/  # clone tham khảo (cần GPU lớn, chưa chạy)
├── scripts/
│   ├── download_data.sh    # tải NASA IMS + FEMTO
│   ├── download_papers.sh  # tải paper tham khảo
│   ├── 01_baseline_anomaly.py   # AE chỉ học 'bình thường', chấm ngưỡng
│   ├── 02_compare_augmentation.py # RF: so sánh TRƯỚC/SAU aug (heuristic|npy)
│   ├── 03_train_timegan.py  # sinh lỗi giả TimeGAN → .npy
│   ├── 04_train_lstm.py     # LSTM / 1D-CNN trên window thô
│   ├── 05_physics_aug.py    # sinh lỗi giả VẬT LÝ (BPFO/BPFI) + đánh giá
│   ├── 06_scarcity_aug.py   # KHAN HIẾM LỖI nhân tạo (hold-out test cố định, bóp lỗi train)
│   ├── 07_timegan_vs_interp.py  # so TimeGAN (generative) vs interpolation ở CÙNG mức khan hiếm
│   ├── 08_contrastive_vs_aug.py # so self-supervised contrastive vs feature 24D (kém)
│   ├── 09_tta_ensemble.py     # TTA + ensemble (thử — không giúp, RF không invariance)
│   └── 10_denoise_ab.py       # A/B wavelet denoise (LANL 1st) trước khi feature (không giúp)
├── data/     # dataset thô (git-ignored)
└── results/  # JSON + npy kết quả
```

## Workflow (chạy lệnh, không notebook)
```bash
source .venv/bin/activate

# 3) baseline RF + feature (kết quả chính)
python scripts/02_compare_augmentation.py --dataset ims --gen heuristic --limit 300

# 4) augmentation heuristic (xem protocol)
python scripts/02_compare_augmentation.py --dataset ims --gen heuristic

# 5) augmentation TimeGAN thật (cần GPU — máy có RTX 4050 6GB + CUDA)
python scripts/03_train_timegan.py --dataset ims --iterations 4000 --n-samples 4000
python scripts/02_compare_augmentation.py --dataset ims --gen npy \
     --gen-npy results/synthetic_faults_ims_FIXED.npy --limit 300

# 6) augmentation VẬT LÝ (BPFO — chạy CPU, nhanh)
python scripts/05_physics_aug.py --dataset ims --fault-type bpfo --limit 300

# 7) KHAN HIẾM LỖI nhân tạo (thí nghiệm trọng yếu §3.4 — test cố định 8100 lỗi)
#    --gen optimized = bộ sinh TỐI ƯU (interpolate lỗi thật) → recall 0.38 → 0.65 ở 20 lỗi train
python scripts/06_scarcity_aug.py --dataset ims --gen optimized --n-fault 50 20 10

# 8) so TimeGAN (generative) vs interpolation ở CÙNG mức khan hiếm (cần GPU)
python scripts/07_timegan_vs_interp.py --dataset ims --n-fault 50 20 --iterations 1200

# 9) deep temporal (so sánh)
python scripts/04_train_lstm.py --dataset ims --model lstm
```

> **Ghi chú `--limit`:** IMS có ~9.5k file/tổng (tải hết ~8GB RAM). `--limit` giới hạn số
> file **mỗi vùng** (normal = limit file đầu, fault = limit/4 file cuối) — không còn cắt
> theo "300 file đầu rồi chia" (bug cũ đã sửa, xem SPEC.md §4).

## Dataset public
- **NASA IMS Bearings**: run-to-failure, 3 test, lỗi nằm ở cuối vòng đời (chia theo thời gian).
- **CWRU Bearing**: 12k/48kHz, .mat, nhiều loại lỗi + tải. **Đã bổ sung 4 file normal (97–100
  mat) từ mirror** vì trang chính thức serve thiếu bytes.
- **FEMTO/PRONOSTIA**: accelerated life test.

## Paper / repo tham khảo (xem RESOURCES.md)
- **TimeGAN** (NeurIPS 2019), **FaultDiffusion** (2025) — sinh chuỗi thời gian.
- Clone: `git clone https://github.com/jsyoon0823/TimeGAN.git  src/`
- CWRU official: https://engineering.case.edu/bearingdatacenter/download-data-file

## Timeline
- 31/08: đã hoàn tất đăng ký đội.
- 11/09: Bootcamp + Factory Tour — chuẩn bị câu hỏi thu thập data lỗi thật.
- 12/10: nộp Vòng 1 (pipeline + báo cáo so sánh + dashboard).

## Việc cần làm (ưu tiên)
- [x] Thí nghiệm **khan hiếm lỗi nhân tạo** (script 06) → "augment cứu recall khi đói lỗi" — ĐÃ XONG (§3.4).
- [x] **Bộ sinh tối ưu** (interpolate lỗi thật) — ĐÃ XONG: `src/generator.py` (recall 0.38 → 0.65 ở 20 lỗi).
- [x] So **TimeGAN vs interpolation** — ĐÃ XONG (script 07): TimeGAN kém nhất khi khan hiếm (0.43), interpolate thắng (0.65).
- [x] **Model mới / bộ sinh phức tạp** — ĐÃ XONG (không thắng): contrastive kém feature 24D (§3.5), ECDF/KS align không thay được interpolate (§3.4.2).
- [ ] Gom số liệu thật vào bảng slide Vòng 1 (nguồn `results/*.json`).
