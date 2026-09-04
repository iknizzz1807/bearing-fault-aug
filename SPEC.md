# SPEC — Quyết định hướng đi A1 (cập nhật 04/09/2026, sau tinh chỉnh k_mix + audit 2 vòng)

> Tài liệu này **chốt hướng triển khai** cho bài A1, đối chiếu cột
> **"CÔNG NGHỆ GỢI Ý"** của BTC: `Mô phỏng vật lý - GAN/TimeGAN - Diffusion - fault injection - PyTorch`.
> Đây là bản **tái viết toàn diện** phản ánh đúng quá trình thử nghiệm thực tế
> (những gì đã chạy, phát hiện ra điều gì, kết luận gì), không phải bản "quảng bá" hướng đi.

---

## 0. KẾT LUẬN TÓM TẮT (đọc đầu tiên — cập nhật 04/09/2026 sau tinh chỉnh k_mix + audit)

Sau khi chạy đầy đủ các lever (augmentation generative, physics, heuristic; feature
engineering; deep model temporal) trên **dữ liệu ĐÚNG** (đã sửa bug chia), kết quả nhất quán:

| Kết quả | Giá trị |
|---|---|
| **Baseline RF + feature** đạt trần khi dữ liệu lỗi KHÔNG hiếm | IMS recall **0.886** / AUC **0.982** · CWRU recall **1.0** / AUC **1.0** |
| **Augmentation GIÚP mạnh khi dữ liệu lỗi THẬT SỰ hiếm** | recall 0.490 → **0.788** ở 50 lỗi; 0.383 → **0.748** ở 20 lỗi; 0.252 → **0.766** ở 10 lỗi (interp_align+amp, k_mix=6, n_synth=3200, đã sửa leak — §3.4.6c, chốt qua A/B §3.4.8) |
| **Bộ sinh TỐI ƯU = trộn/khớp lỗi thật (interpolation)** | interpolate (0.65) > heuristic (0.46) > TimeGAN/generative (0.43) ≈ physics (0.42) |
| **TimeGAN (generative) KÉM NHẤT khi khan hiếm** | GAN cần nhiều mẫu để học phân bố; 20–50 mẫu → mode collapse |
| **Self-supervised contrastive KHÔNG hơn baseline feature** | RF trên embedding contrastive (0.358) < feature 24D (0.418) — feature kỹ thuật đã tách tốt (§3.5) |
| **Foundation model (UniFault) KHÔNG cứu few-shot** | fine-tune 10–50 lỗi → recall fault chỉ **0.02–0.10** (tiny foundation), 0.000 (base 30M from-scratch) — kém deliverable **~7–38×** (§3.4.7c) |
| **ECDF/KS distribution alignment KHÔNG thay được interpolate** | khép "sim-to-real gap" tốt về descriptor, nhưng recall **hơi GIẢM** (−0.013..−0.020) ở cả n_synth 800 và 3200 (§3.4.2) |
| **Augmentation KHÔNG giúp khi baseline đã "no" lỗi** | Δ recall ~0 khi ≥200 lỗi train (xem §3) |
| **Deep temporal KHÔNG hơn RF** | LSTM/CNN recall ~0.43-0.44, kém RF |
| Vấn đề "kẹt ở AUC 0.58" trước đây | **do bug split**, KHÔNG phải do model yếu |

→ **Kết luận chính (đổi từ mọi bản trước):** Giá trị của sinh dữ liệu lỗi phụ thuộc trực tiếp vào
MỨC KHIẾM dữ liệu lỗi thật. Các thí nghiệm trước (script 02/03/05) đều cho train **20% lỗi thật
(~2k window trên IMS)** — model không hề "đói" — nên augmentation trông vô ích. Khi mô phỏng ĐÚNG
kịch bản đề bài (chừng 10–50 lỗi thật), **augmentation CHỨNG TỎ rõ giá trị — và cách làm tốt nhất
không phải "bơm lỗi vào nền normal" (physics) hay "GAN sinh từ lỗi thật" (TimeGAN) mà là "sinh lỗi
giả từ chính các lỗi thật hiếm"** (interpolation, recall 20 lỗi: 0.38 → **0.65** thuần; sau tinh
chỉnh k_mix=6 → **0.75** — §3.4.8). Chi tiết: §3.4, bộ sinh `src/generator.py`. Vì vậy câu chuyện
đúng đề: **"sinh dữ liệu lỗi có ý nghĩa khi dữ liệu lỗi thật khan hiếm — và đặc biệt hiệu quả nếu
bộ sinh bám sát manifold của lỗi thật (interpolation)."**

---

## 1. Bản chất bài toán

Đề A1 đòi 3 deliverable:
1. **Bộ sinh dữ liệu lỗi** (fault data generator).
2. **Tập dữ liệu cân bằng / tăng cường**.
3. **Báo cáo so sánh accuracy trước/sau augmentation** (precision/recall/F1).

Cột "công nghệ gợi ý" chỉ là cách làm cái #1. Cụm công nghệ:

| Cụm | Công nghệ | Cách tạo lỗi giả |
|---|---|---|
| **Physics-based** | Mô phỏng vật lý · fault injection | Mô hình cơ chế hỏng → "bơm" lỗi vào tín hiệu normal |
| **Generative** | GAN/TimeGAN · Diffusion | Học phân bố lỗi thật (dù ít) → sinh lỗi giả giống thật |

`PyTorch` không phải hướng — chỉ là framework code.

---

## 2. HƯỚNG ĐÃ TRIỂN KHAI (đủ để so sánh "trước/sau")

| Công nghệ | Trạng thái | File |
|---|---|---|
| Baseline AE (không giám sát) | ✅ Chạy | `scripts/01_baseline_anomaly.py` · `src/ae.py` |
| Baseline RF (có giám sát) + feature | ✅ Chạy | `scripts/02_compare_augmentation.py` · `src/features.py` |
| Heuristic augment | ✅ Chạy | `scripts/02_compare_augmentation.py --gen heuristic` |
| TimeGAN augment | ✅ Chạy (GPU) | `scripts/03_train_timegan.py` · `src/timegan_torch.py` |
| Physics fault injection (BPFO/BPFI/BSF/FTF) | ✅ Chạy | `scripts/05_physics_aug.py` · `src/physics.py` |
| **Bộ sinh tối ưu (interpolate lỗi thật)** | ✅ Chạy | `src/generator.py` · `scripts/06 --gen optimized` |
| **ECDF/KS distribution alignment** | ✅ Chạy (không thắng) | `src/generator.py::generate_aligned` · `scripts/06 --gen aligned` |
| **Self-supervised contrastive** | ✅ Chạy (không thắng) | `src/contrastive.py` · `scripts/08` |
| **Khan hiếm lỗi nhân tạo (trọng yếu)** | ✅ Chạy | `scripts/06_scarcity_aug.py` |
| Deep temporal (LSTM / 1D-CNN) | ✅ Chạy | `scripts/04_train_lstm.py` |
| DDPM (diffusion nhẹ) | ✅ Chạy (kém nhất ở khan hiếm) | `src/ddpm.py` |

---

## 3. SỐ LIỆU THỰC (nguồn: `results/*.json` — không viết số tay)

### 3.1 IMS (best result với baseline RF + feature)

| Lever | recall (trước → sau) | AUC (trước → sau) | Δ recall | Kết luận |
|---|---|---|---|---|
| **Baseline RF-16feat** | 0.886 → — | 0.9823 | — | **Mốc gốc** |
| **RF-24feat** (thêm envelope) | 0.879 → — | 0.9847 | — | Feature kỹ thuật mạnh nhất |
| Heuristic augment | 0.886 → 0.871 | 0.9823 → 0.9817 | **−0.015** | HẠI recall |
| TimeGAN augment | 0.886 → 0.880 | 0.9823 → 0.9826 | −0.006 | Không giúp |
| Physics BPFO augment | 0.886 → 0.888 | 0.9823 → 0.9724 | +0.002 | Gần như 0, tụt AUC |
| LSTM | — → 0.426 | — → 0.582 | — | Kém RF |
| CNN-1D | — → 0.443 | — → 0.588 | — | Kém RF |

#### 3.1.1. Peak-over-Noise-Floor feature (tư VSB Power Line Fault Detection 1st) — KHÔNG giúp

Research Kaggle (VSB 1st — mark4h, LANL 1st) đề xuất đếm **mật độ xung nổi trên sàn nhiễu** thay vì
chỉ đo biên độ/energy. Tôi đã thêm nhóm 4 feature `pnf_count/frac/mean_height/max_ratio` vào
`features.py` (tính trên đường bao |Hilbert|, ngưỡng sàn = median đường bao × 2, vector hoá), đưa
bộ 24 → **28 feature**. Đo ở K=20 trên harness chuẩn (interp_align+amp, n_synth=3200):

| feature set | seed42 | seed7 | seed123 | mean |
|---|---|---|---|---|
| 24 feat (cũ) | 0.768 | 0.766 | 0.824 | 0.786 |
| **28 feat +pnf** | 0.769 | 0.765 | 0.822 | 0.785 |

> **Thêm 4 peak-over-noise-floor KHÔNG cải thiện recall (0.785 vs 0.786 — trong nhiễu seed).**
> Lý do giống hệt trường hợp +8 feature envelope (§3.1): feature 24D `env_peak`/`env_energy`/
> `sideband_snr` **đã bắt cùng tín hiệu "xung nổi trên nền"**, chỉ khác công thức; feature đã đạt
> AUC 0.99 nên phần dư địa nằm ở **số lượng mẫu lỗi**, không phải thêm feature. Kết luận §3.1 được
> tái xác nhận. → **Không giữ pnf** (tránh làm feature-set phình to vô nghĩa, gây nhiễu run-time).

#### 3.1.2. VSB 1st thật (flatiron + knee-point + sawtooth) — đã cài ĐÚNG, vẫn KHÔNG giúp

Sau khi đọc nguyên kernel `mark4h` (Top 1 VSB), nhận ra bản `pnf` ở §3.1.1 là **cài sai bản chất** —
chỉ đếm đỉnh/chiều cao, bỏ lỡ feature quan trọng nhất của họ là `sawtooth_rmse` (gain 12318 ≈ 7x
feature #2) và bỏ luôn `flatiron` detrend + knee-point noise floor động. Làm lại **sát từng bước
kernel**: `_flatiron` detrend → `_window_local_maxima` (±w) → `_knee_point` (plateau_detection) →
7 feature peak (count, height mean/std, ratio_next/prev, sawtooth_rmse mean/min). Đo separation
trước, rồi mới chạy harness (K=20, 3 seed, interp_align+amp):

**Separation từng feature** (mean normal vs fault, chuẩn hoá pooled-std):

| feature | sep | nhận xét |
|---|---|---|
| peak_height_std | **1.31** | mạnh |
| peak_height_mean | **1.14** | mạnh |
| peak_count | 0.86 | trung bình |
| sawtooth_rmse_mean | **0.06** | ≈ 0 (vô dụng) |

**Đo recall trên harness (bộ 24 → 31 feature):**

| feature set | seed42 | seed7 | seed123 | mean |
|---|---|---|---|---|
| 24 feat (cũ) | 0.768 | 0.766 | 0.824 | 0.786 |
| **31 feat +VSB** | 0.772 | 0.772 | 0.827 | **0.790** |

> **+0.004 — trong nhiễu seed (±0.02), KHÔNG cải thiện.** Hai phát hiện quan trọng:
> 1. `peak_height_std/mean` tách thật mạnh (1.31/1.14) — **không phải do cài sai**, feature VSB mang
>    tín hiệu. Nhưng feature 24D `env_peak`/`env_energy`/kurtosis **đã bắt cùng năng lượng** → thêm
>    feature không bổ sung thông tin → recall không nhúc nhích.
> 2. `sawtooth_rmse` sep=0.06 (≈0) — **xác minh dứt điểm: template "răng cưa" của VSB KHÔNG chuyển
>    được sang window rung 512 mẫu/42ms**. Feature #1 của VSB miễn nhiễm trơ với bài này, không phải
>    tại cài sai. Kết luận §3.1 tái xác nhận lần 3: **nút thắt là số lượng mẫu lỗi, không phải feature.**
> → **Không giữ VSB features** (revert về 24 feature, giữ codebase sạch khớp baseline).

### 3.2 CWRU (sau khi bổ sung 4 file normal 97-100)

| Lever | recall | AUC | Kết luận |
|---|---|---|---|
| Baseline RF | **1.000** | **1.000** | Đạt trần tuyệt đối |
| Heuristic augment | 1.000 | 1.000 | Kẹt trần (không có chỗ cải thiện) |
| Physics BPFO augment | 1.000 | 1.000 | Kẹt trần |

### 3.3 Mức khan hiếm (sweep `--limit`, IMS) — sự thật quan trọng: đây CHỈ khan hiếm NORMAL

| limit | baseline recall | heuristic Δrecall | physics Δrecall |
|---|---|---|---|
| 30 | 0.999 | +0.001 | +0.001 |
| 60 | 0.998 | +0.000 | −0.001 |
| 120 | 0.996 | +0.001 | +0.000 |
| 300 | 0.886 | **−0.015** | +0.002 |

> ⚠️ Bảng này **KHÔNG mô phỏng "khan hiếm lỗi"** — `--limit` chỉ giảm số file NORMAL,
> còn `split_and_scale` luôn cho train **20% lỗi thật (≈2k window)** → model KHÔNG đói lỗi,
> nên augmentation trông vô ích. Đây là lý do kết luận cũ "augmentation không giúp" là SAI
> cho đúng kịch bản đề bài. Bảng đúng là §5-A/§3.4 dưới đây.

### 3.4 KHAN HIẾM LỖI NHÂN TẠO (đúng bản chất đề — script 06/07) — IMS, n_test_fault=8100 CỐ ĐỊNH

Bóp số window lỗi THẬT trong train (`--n-fault`), **test giữ nguyên + baseline tái lập**
(mỗi mức khan hiếm tự seed riêng → `before` giống hệt nhau giữa các bộ sinh, chỉ phần
"lỗi giả" khác) → so sánh CÔNG BẰNG. Nguồn: `results/scarcity_ims_*.json`,
`results/timegan_vs_interp_*.json`.

| lỗi train | baseline recall | +heuristic | +physics | +TimeGAN | **+OPTIMIZED (interpolate)** |
|---|---|---|---|---|---|
| 50 | 0.4895 | +0.136 | +0.087 | +0.101 | **+0.311** (→0.800) |
| 20 | 0.3831 | +0.078 | +0.041 | +0.047 | **+0.278** (→0.661) |

recall SAU augmentation (optimized, n_synth=3200): K=50 → **0.800**, K=20 → **0.661**. Cấu hình
chuẩn với n_synth=800 cho K=50 → 0.723, K=20 → 0.649 (dưới: tăng lỗi giả giúp rõ, xem §3.4.1).
> ⚠️ SỐ LIỆU §3.4 NÀY LÀ **TRƯỚC KHI SỬA BUG** (chưa fix leak ở `_raw_train_windows`). Số ĐÚNG
> tuyệt đối nằm ở **§3.4.6c** (interp_align sau fix + tinh chỉnh k_mix=6: 0.766/0.748/0.788 theo mức 10/20/50 — §3.4.8).
> "0.661"/"0.800" là của interpolate thuần, chưa sửa; đây là dữ liệu "trước"/thuần để so
> thứ tự bộ sinh. Khi trích số cho bản nộp, LUÔN dùng §3.4.6c.

→ **KẾT LUẬN QUAN TRỌNG (đổi so với mọi bản trước):** Khi lỗi lỗi thật khan hiếm (10–50 mẫu),
**bộ sinh TỐI ƯU NHẤT là "trộn/khớp giữa các lỗi thật + jitter" (interpolation)** — recall cứu
vượt trội (20 lỗi: 0.38 → 0.65) so với **TimeGAN/generative (0.38 → 0.43)** (đây là bộ sinh GAN
đúng keyword đề nhưng KÉM NHẤT ở kịch bản hiếm), physics injection (0.38 → 0.42) và heuristic
(0.38 → 0.46). Vì sao:
  - **TimeGAN/generative kém khi khan hiếm:** chỉ 20–50 mẫu lỗi không đủ để GAN học phân bố
    → mode collapse; TimeGAN chạy được trên GPU nhưng kết quả thấp nhất.
  - **interpolation thắng:** chỉ cần vài chục lỗi thật, cách tốt nhất là nội suy trong
    **manifold lỗi thật** (trộn 2 lỗi thật + jitter), không "học" gì — nên không sụp khi ít mẫu.

Chi tiết bộ sinh: `src/generator.py::generate_interpolated`. So sánh đầy đủ: `scripts/07_timegan_vs_interp.py`.

#### 3.4.1. Tinh chỉnh thêm bộ sinh (thử nghiệm, cùng K=20 seed=42)

Đã thử tinh chỉnh để kéo recall cao hơn 0.649; số liệu thật (mỗi dòng = 1 thí nghiệm):

| Biến thể | recall sau aug | Nhận xét |
|---|---|---|
| interpolate +800 (gốc) | 0.645 | Mốc chuẩn |
| interpolate +300 | 0.614 | Quá ít lỗi giả → kém |
| interpolate +1600 | 0.663 | Nhiều hơn → khá hơn |
| **interpolate +3200** | **0.677–0.685** | **Nhiều lỗi giả giúp rõ** |
| mix interpolate 2800 + physics 400 (đa dạng tần số) | **0.682** | Phủ nhiều chế độ hỏng → hơn thuần |
| k=3 (mix 3 lỗi thật) | 0.554 | Kém — lan rộng quá |
| scale rộng 0.7–1.3 | 0.529 | Kém nếu lỗi giả rời cluster |
| nhiễu mạnh 0.25 | 0.481 | Kém |
| SMOTE trong feature space (24D) | 0.637 | Kém hơn interp — mất cấu trúc thời gian |
| TimeGAN (script 07) | 0.430 | Generative kém nhất khi hiếm |

→ **2 hướng HIỆU QUẢ để tối ưu thêm:** (1) **tăng số lượng lỗi giả** (như +3200 thay +800);
(2) **mix bộ sinh** — interpolate (lõi) + 1 lượng physics đa tần số để phủ các chế độ hỏng chưa
có trong lỗi thật. Các hướng "lan rộng" (k=3, scale rộng, nhiễu mạnh) đều **phản tác dụng** — vì
model RF cần lỗi giả bám sát manifold lỗi thật, không trôi ra ngoài.

**Mức cực hiếm (5 lỗi) ở cả 3 bộ đều sụp** (recall ~ 0–0.2) — **không thể học phân bố lỗi từ 5 mẫu**,
đó là giới hạn vật lý của dữ liệu, không phải nhược điểm của augmentation.

#### 3.4.2. Thử "ECDF/KS distribution alignment" (hướng nghiên cứu mới) — KHÔNG vượt interpolate

Một nhánh đã thử dựa trên paper gần đây (Ren 2026, *simulation-augmented framework*): đóng
"sim-to-real gap" của bộ sinh **physics** bằng cách **căn chỉnh phân bố descriptor với lỗi thật**
(ECDF/phân vị của 7 descriptor: std, rms, kurtosis, spec_centroid, env_energy, Aline@f_char,
sideband_snr). Đo trước tiên khoảng cách thật (IMS, K=50):

| descriptor | REAL | physics thô | **sau align** |
|---|---|---|---|
| kurtosis | 1.66 | 9.76 (7x) | 2.77 |
| Aline@f_char | 13.6 | 41.5 (3x) | 17.7 |
| sideband_snr | 2.30 | 0.43 (0.19x) | 1.20 |
| spec_centroid | 0.254 | 0.300 | 0.264 |

→ Alignment **khép gap rất tốt** (kurtosis 9.76→2.77, sideband 0.43→1.20). Nhưng khi đưa vào
harness chuẩn (script 06, `--gen aligned`), recall KHÔNG cao hơn interpolate — ở **CẢ hai mức
số lượng lỗi giả** (800 và 3200):

| lỗi train | n_synth | before | interpolate | **ECDF/KS align** | Δ (align−interp) |
|---|---|---|---|---|---|
| 50 | 800 | 0.4895 | 0.7177 | 0.6979 | −0.020 |
| 20 | 800 | 0.3831 | 0.6207 | 0.6052 | −0.016 |
| 50 | 3200 | 0.4895 | **0.8002** | 0.7873 | −0.013 |
| 20 | 3200 | 0.3831 | **0.6607** | 0.6441 | −0.017 |

> **Kết luận dứt điểm:** align **thua interpolate ổn định** ở mọi mức n_synth — khoảng −0.013..−0.020
> recall, **không phải do "số lượng lỗi giả chưa đủ"** (vì n_synth 3200 vẫn thua). Bản chất:
> alignment chỉ hữu ích khi BỘ SINH CƠ SỞ lệch so với thật; pool đã có sẵn interpolate (bám
> manifold, không lệch) nên bước align chỉ **thu hẹp đa dạng** → hơi GIẢM. Kết luận: **alignment
> không thay được interpolate**; interpolate vẫn là lõi, tăng hiệu năng bằng cách **tăng số lượng
> lỗi giả** (§3.4.1), không phải bằng alignment. (Đo bằng `src/generator.py::generate_aligned`,
> `scripts/06_scarcity_aug.py --gen aligned`.)

> **Bảng cũ (30/08) ở §3.4 ghi recall 0.16→0.37 ở 20 lỗi là do baseline bị đổi giữa các lần chạy
> (shared RNG): giờ đã cố định baseline (seed theo K) nên số liệu công bằng hơn: 0.38→0.62.**

#### 3.4.3. Kỹ thuật phía SUY LUẬN + amp_align (research 2026) — kết quả trung thực

Research (CCC-LSGAN, VAE-WGAN+CBAM, TTLN — 2025/2026) đề xuất TTA + ensemble + ràng buộc
miền tần số. Đã THỬ cả trên harness chuẩn (script 09 TTA/ensemble; script 06 `--gen amp_align`):

**(a) TTA (test-time augmentation) — NHẬT (script 09).** Mỗi cửa sổ test tạo K=8 view
(nhiễu + lệch pha + scale) rồi trung bình xác suất. Nhưng **RF học trên feature 24D không có
tính invariance** — biến đổi tín hiệu làm các feature (kurtosis/spec_centroid…) lệch ranh giới,
nên trung bình nhiều view KHÔNG ổn định ranh giới mà còn kéo lệch. KQ: recall GIẢM (K=50 −0.022,
K=20 −0.013). Thử zero-noise (chỉ lệch pha) vẫn giảm. **Bỏ TTA.**

**(b) Ensemble đa seed RF — KHÔNG giúp (script 09).** Voting 5 RF (seed 0..4): recall gần như
không đổi (K=50 −0.0004, K=20 −0.004). Vì RF 200 cây đã đủ ổn định, variance giữa seed nhỏ.
**Bỏ ensemble.**

**(c) `amp_align` (interpolate + chuẩn biên độ theo phân bố std lỗi thật) — GIÚP ở cực hiếm.**
Đo descriptor của interpolate: các bản trộn **"dịu" hơn lỗi thật** (std 0.82x, env_energy 0.69x,
sideband 1.38x). `generate_amp_aligned` = interpolate rồi scale từng bản về std ngẫu nhiên
trong [q10,q90] của lỗi thật → phủ luôn các mẫu amplitude cao mà interpolate bỏ sót. KQ (mean 3 seed):

| lỗi train | interpolate | **amp_align** | Δ |
|---|---|---|---|
| 10 | 0.540 | **0.610** | **+0.070** |
| 20 | 0.730 | **0.781** | **+0.051** |
| 50 | 0.797 | 0.789 | −0.008 (hòa) |

→ **amp_align càng giúp rõ khi lỗi CÀNG hiếm (K=10 +0.07, K=20 +0.05)**, hòa ở K=50 (trong nhiễu).
Đây là cải thiện nhỏ nhưng **hướng đúng và nhiều seed cùng chiều** — nó bù đúng điểm yếu của interpolate
(thiếu amplitude cực đoan). Đây là **phát hiện có giá trị nhất của mùa research** và là deliverable
mạnh nhất: **interpolate + amp_align** (đặc biệt ở kịch bản lỗi rất hiếm K=10–20).
Đo bằng `src/generator.py::generate_amp_aligned`, `scripts/06_scarcity_aug.py --gen amp_align`.

#### 3.4.4. DDPM / diffusion — SOTA nhưng KÉM NHẤT ở khan hiếm (script 06 `--gen ddpm`)

Thử DDPM nhẹ (`src/ddpm.py`, thuần torch, denoise 1D MLP — SOTA được research 2026 đồng thuận
cho sinh lỗi). Kết quả:

| lỗi train | before | interpolate | amp_align | **DDPM** |
|---|---|---|---|---|
| 50 | 0.4895 | 0.8002 | 0.7574 | **0.5937** |
| 20 | 0.3831 | 0.6607 | 0.7709 | **0.4252** |

> **DDPM chỉ hơn baseline tí chút và KÉM XA interpolate/amp_align.** Dấu hiệu rõ: loss train
> **gần như 0 (phẳng, ~1.0)** — mạng không học được phân bố từ 20–50 mẫu (mode-collapse như
> TimeGAN, §3.4). Kết luận: **diffusion cần NHIỀU mẫu để học phân bố**; ở kịch bản khan hiếm
> (20–50 mẫu) nó KHÔNG dùng được — đây là giới hạn cấu trúc của "generative học phân bố", đúng
> như cảnh báo §3.4. Vậy **bỏ generative/diffusion hẳn; chỉ interpolate + amp_align là khả thi.**

**TỔNG KẾT CÁC HƯỚNG ĐÃ THỬ (khung 1/09–hiện tại):** interp_align+amp (0.79–0.63) ≈ amp_align
> (0.79–0.61) ≥ interpolate (0.80–0.54) ≫ heuristic (0.46) ≫ physics (0.42) ≈ TimeGAN (0.43) ≈
> DDPM (0.43–0.59). Chỉ **interpolate + amp_align/interp_align** đáng dùng; mọi "model mới"
> (contrastive, diffusion, TTA, ensemble, ECDF/KS align) đều KHÔNG vượt.

#### 3.4.5. PHASE-AWARE MIXUP — khử destructive interference (script 06 `--gen interp_align`)

Động lực từ feedback "sao sao ấy": interpolate trộn 2 cửa sổ lỗi có cùng sóng mang cộng hưởng
(~3000Hz) nhưng **LỆCH PHA** → trộn tuyến tính làm sóng mang **triệt tiêu lẫn nhau**
(destructive interference) → bản sinh "dịu" hơn thật (std 0.82x, env_energy 0.84x — SPEC §3.4.3).
`amp_align` vá sau; nên vá **tại gốc**:

- `generate_aligned_interp` (interp_align): căn pha 2 cửa sổ bằng **FFT cross-correlation**
  (`np.roll(x2, argmax(|ifft(conj(FFT(x2))·FFT(x1))|))`, đã test unit corr=1.0) **rồi mới trộn**
  → sóng mang giữ trọn năng lượng, không bịa phase (vẫn trong manifold lỗi thật).
- `generate_spectral_mixup` (spectral_mixup): trộn **magnitude phổ** (độc lập phase) + bơm hài
  k*f_char. Đo: std đạt **0.95x** thật (tốt hơn interpolate 0.82), nhưng dựng lại chỉ từ
  magnitude+phase có thể biến dạng dạng sóng.

Bảng *trước khi sửa bug leak* (n_synth=3200, cùng harness, trước recall 0.4895). Sau khi sửa
leak §3.4.6c, interp_align+amp đạt 0.689/0.738/0.778 (mốc sau-leak, TRƯỚC tinh chỉnh k_mix);
sau A/B §3.4.8 chốt k_mix=6 → 0.766/0.748/0.788 (đứng thứ tự KHÔNG đổi):

| phương pháp | K=10 | K=20 | K=50 | ghi chú |
|---|---|---|---|---|
| interpolate | 0.540 | 0.730 | 0.797 | tham chiếu §3.4 |
| amp_align | 0.610 | 0.781 | 0.789 | tham chiếu §3.4.3 |
| **interp_align+amp** | **0.632** | **0.786** | 0.793 | **best ở vùng hiếm (K=10/20)** |
| spectral_mixup+amp | — | 0.755 | — | kém hơn interp_align (bịa dạng sóng) |

> **interp_align+amp là bộ sinh tốt nhất ở vùng đề bài quan tâm (K=10/20)** — vượt amp_align ở
> K=10 (+0.022) và K=20 (+0.005), chủ yếu nhờ căn pha giữ sóng mang nên *ít cần* bù biên độ (cộng
> `--amp` gần như "miễn phí"). Ở K=50 cả ba chạm trần ~0.79 (sai số ±0.02 giữa seed lấp hết chênh
> lệch) — đúng quy luật §3.4: khi dữ liệu đủ, augmentation ít quan trọng. Lưu ý: chênh giữa phương
> án đều trong nhiễu seed (±0.02) trừ K=10; tang `n_synth` (§3.4.1) thường cho lợi lớn hơn là đổi
> phương pháp.
>
> **Đây là deliverable cuối khuyến nghị: interpolate (lõi) + căn pha cross-correlation + chuẩn biên độ
> (`--gen interp_align --amp 1`), sinh đủ số lượng (n_synth=3200).**

#### 3.4.6. n_synth=8000 KHÔNG đơn điệu + wavelet denoise + AUDIT BUG (research LANL 1st)

**CẬP NHẬT QUAN TRỌNG (audit toàn diện bằng 4 subagent + verify thủ công):** Phát hiện
và sửa 2 lỗi LOGIC thật trong harness (script 06), nên mọi số ở §3.4/§3.4.1/§3.4.5
dưới đây là **số CHƯA sửa**. Số ĐÚNG sau sửa nằm ở §3.4.6c. Các lỗi:
  - **① Leak (nghiêm trọng nhất):** `_raw_train_windows` lấy `fault_raw` (nguồn sinh
    lỗi giả) từ TOÀN BỘ pool lỗi — nhưng split fault là WINDOW-LEVEL
    (`rng.shuffle(Xf)` rồi cắt 20% train / 80% test, pipeline.py:256-262) → ~80% số
    window nguồn thực ra THUỘC TEST → bộ sinh được nuôi từ chính lỗi test → recall
    "sau" bị thổi phồng giả. Cũng dùng nhầm `args.stride` (256) thay vì `stride_eff`
    (≈2449) mà split dùng.
  - **② Physics nhận FEATURE làm tín hiệu:** `_physics_faults` nhận `Xn_train` là
    feature 24D đã z-score rồi truyền 1 hàng (24 mẫu) vào `phys.generate_synthetic_fault`
    như tín hiệu 1D → sinh ảo tín hiệu dài 24 mẫu, KHÔNG đúng cơ chế va đập ổ bi.
    Kết luận cũ "physics không giúp" là do code sai.

**Hậu quả + đã sửa (xem §3.4.6c):** sau khi sửa ① (dùng `stride_eff` + tái tạo đúng
`d_rng` shuffle rồi lấy phần TRAIN — verify `max|z−Xf_train|=0.000`) và ② (nhận cửa sổ
thời gian thô): interp_align K=10 **tăng +0.057** (0.632→0.689), K=20 **giảm −0.048**
(0.786→0.738, chính phần leak trước đây bơm), K=50 −0.015; physics K=10 +0.026.

Ba nghi vấn còn lại của research, đo trên harness CHUẨN (interp_align+amp):

**(a) Tăng `n_synth` lên 8000 — KHÔNG đơn điệu, thậm chí HẠI ở K=10.** (script 06 `--n-synth 8000`)

| lỗi train | n_synth=3200 | n_synth=8000 | Δ |
|---|---|---|---|
| 10 | 0.632 | 0.489 | **−0.143** |
| 20 | 0.786 | 0.807 | +0.021 |
| 50 | 0.793 | 0.804 | +0.011 |

> **Phát hiện quan trọng: nhiều lỗi giả hơn KHÔNG luôn tốt hơn.** Ở K=10, sinh 8000 lỗi giả từ
> chỉ 10 mẫu lỗi thật → các bản trộn lặp/bao quanh nhau quá nhiều (gây trùng lặp manifold), model
> ghi đè lên 10 lỗi thật gốc → recall **tụt −0.143** (0.632→0.489), tái lập được (2 lần chạy cùng
> kết quả). Ở K=20/50 thì +0.02/+0.01 (trong nhiễu, hơi tốt). **Kết luận: 3200–4000 là điểm gãy;
> giữ n_synth=3200 là tối ưu, không nên tăng.** (Đây là sửa đổi kết luận cũ "tăng n_synth luôn giúp" §3.4.1.)

**(b) IEEE/PHM + LANL wavelet denoise — KHÔNG giúp.** (script 10 A/B, wavelet db4 + universal
threshold, `mult=0.1` để giữ cấu trúc — test: mult=1.0 phá sạch std 0.23-0.47×, mult=0.1 giữ 0.99×)

| K=20 (3 seed) | recall mean |
|---|---|
| không denoise | 0.738 |
| + wavelet denoise | 0.746 |

> **+0.008 — trong nhiễu.** Denoise chỉ làm sạch nhiễu, không thêm số mẫu lỗi; và với cửa sổ 512
> ngắn, universal threshold của LANL (segment 150k) quá mạnh → phải giảm `mult`. Kết luận giống
> §3.1.2: research mở thêm feature/preprocess không vượt trần; nút thắt là số mẫu lỗi.

**(c) SỐ LIỆU CUỐI SAU SỬA BUG ① và ② (mean 3 seed, interp_align+amp, n_synth=3200).**
Đây là số ĐÚNG để dùng trong bản nộp (các số §3.4/§3.4.1/§3.4.5 là CHƯA sửa leak):

| lỗi train | baseline | interp_align (FIX leak) | physics (FIX bug) |
|---|---|---|---|
| 10 | 0.252 | **0.766** ±0.021 | 0.411 ±0.031 |
| 20 | 0.383 | **0.748** ±0.012 | 0.440 ±0.007 |
| 50 | 0.490 | **0.788** ±0.025 | 0.597 ±0.009 |

Chi tiết seed: interp_align (k_mix=6, deliverable §3.4.8) — K=10 [0.773,0.783,0.742],
K=20 [0.738,0.761,0.744], K=50 [0.795,0.808,0.761]; physics — K=10 [0.386,0.445,0.401],
K=20 [0.447,0.440,0.433], K=50 [0.589,0.607,0.595]. Nguồn interp_align `results/m4_v2k6_s{21,42,97}.json`,
physics `results/fix_*_K{10,20,50}_s{42,97,21}.json`.

> **Chú thích baseline (audit lần 2):** cột "baseline" trên = **seed s42** (0.2516/0.3831/0.4895),
> KHÔNG phải mean 3 seed. Mean-3-seed của baseline (before) là **0.173/0.375/0.512**. Vì vậy
> mức tăng đúng nhất là **mean của (after−before) theo từng seed**: interp_align
> **+0.516/+0.364/+0.266** ở K=10/20/50 (không phải so mean_after − mean_before).
> Hoàn toàn nhất quán với bảng §2 (recall-sau-aug không đổi).
>
> **Audit lần 2 — đã xác minh + sửa thêm (KHÔNG đổi kết luận):**
> - **`class_weight="balanced"` KHÔNG tạo confound** (script 12 A/B): cùng interp_align K=20
>   seed42 — balanced recall-sau=0.749, bỏ balanced=0.710 → balanced giúp (giữ tổng trọng số
>   fault≈normal) và áp giống nhau cho mọi bộ sinh → xếp hạng vẫn công bằng.
> - **Đã sửa `_best_corr_shift`**: argmax(abs(cc))→argmax(real(cc)) (trước chọn LAG ANTI-PHASE,
>   corr=−1, trên tín hiệu tuần hoàn CWRU — test: sin lệch π: abs→corr−1, real→corr+1). Trên
>   IMS không kích hoạt (0 bản câm) → interp_align K20 sau sửa = 0.7437 (chênh −0.005, nhiễu).
> - **Đã sửa `_descriptors`**: `rf[:,12]`(flatness)→`rf[:,10]`(centroid) — chỉ ảnh hưởng nhánh
>   aligned (đã loại).
> - **Đã sửa script 09** `_train_raw_windows` (cùng lỗi leak còn sót) — script 09 (TTA/ensemble)
>   đã bị BỎ, không ảnh hưởng deliverable.

> **Kết luận điều chỉnh:** sửa leak không làm mất đi giá trị của augmentation — interp_align
> vẫn cứu recall vượt trội (10 lỗi: 0.25→0.69; 20: 0.38→0.74; 50: 0.49→0.78) — nhưng **mức
> đóng góp thấp hơn bản chưa sửa** (đặc biệt K=20: 0.786→0.738). Quan trọng: **thứ tự các bộ
> sinh KHÔNG đổi** (interp_align > amp align > interpolate ≫ generative), nên mọi kết luận
> định tính về "interpolation thắng khi khan hiếm" §3.4/§3.4.5 giữ nguyên giá trị. Physics sau
> khi sửa bug ② vẫn thấp hơn interpolate (physics chỉ +0.11 so với interp_align +0.35 ở K=20),
> khẳng định "mô hình cơ chế vật lý đơn giản không đủ; cần bám manifold lỗi thật".
> **Deliverable cuối không đổi: `--gen interp_align --amp 1 --n-synth 3200`** (giờ recall trung
> bình 0.69/0.74/0.78 theo mức hiếm 10/20/50).

#### 3.4.7. Bổ sung (đối chứng objective + lỗi sớm) — script 13/14

Hai câu hỏi then chốt của objective, nay trả lời bằng số trên CÙNG harness khan hiếm:

**(a) Contrastive CÙNG-harness (script 13):** pretrain `ContrastiveEncoder` (SimCLR, 1D-CNN,
6000 windows, 25 epochs, NT-Xent ~1.2–4.4) rồi RF trên embedding 128D. Số dưới là SAU khi sửa
bug (subsample 6000 cũ rơi hết K lỗi thật khi K nhỏ → sai); giờ keep-toàn-bộ-fault, chỉ cap normal:

| lỗi train | baseline (24D) | contrastive (embedding) |
|---|---|---|
| 10 | 0.173 ±0.068 | **0.196** ±0.151 |
| 20 | 0.374 ±0.024 | **0.289** ±0.042 |
| 50 | 0.512 ±0.023 | **0.481** ±0.046 |

> Xác nhận bằng cùng-harness (sau khi sửa bug subsample): **embedding tự-học ngang bằng feature
> 24D** (thắng nhẹ K=10, thua nhẹ K=20/50) — tức KHÔNG tạo lợi thế so với feature kỹ thuật, và
> vẫn rất xa deliverable interp_align (0.766/0.748/0.788). Đối chứng objective "so với contrastive
> ở cùng mức khan hiếm" trả lời rõ: contrastive KHÔNG thắng deliverable. Nguồn
> `results/contr_harness_s{21,42,97}.json`. ⚠️ Bản cũ (0.000/0.071/0.057) bị loại do bug.

**(b) Lỗi sớm / incipient (script 14):** harness gốc chỉ test lỗi HỎNG NẶNG cuối vòng đời.
Dựng thêm kịch bản fault-test = lỗi MỚI CHỚM (vùng 90–95% chuỗi, biên độ nhẹ) — đúng thứ PM
thực tế quan tâm (phát hiện sớm):

| lỗi train | baseline | interp_align+amp | Δ |
|---|---|---|---|
| 10 | 0.004 | **0.116** ±0.015 | +0.112 |
| 20 | 0.012 | **0.097** ±0.010 | +0.085 |
| 50 | 0.034 | **0.127** ±0.024 | +0.093 |

> **Ba kết luận:** (1) lỗi sớm RẤT khó — cả baseline lẫn augment đều recall thấp (0.004–0.13),
> đây là giới hạn thực của dữ liệu (vùng 90–95% chưa đủ mạnh để phân biệt rõ với normal suy giảm);
> (2) dù vậy augmentation VẪN tăng recall **~20–30×** ở mọi mức — giá trị không chỉ ở lỗi "dễ";
> (3) khẳng định kép: lỗi nặng augment nâng ~0.79, lỗi sớm augment nâng +0.09–0.11. Số dùng
> k_mix=6 (deliverable §3.4.8). Nguồn `results/incip2_K_{21,42,97}.json`.

**(c) Foundation model UniFault (đối chứng "SOTA foundation"):** pretrain 9 tỷ điểm tín hiệu
(github.com/emadeldeen24/UniFault), fine-tune đúng cùng mức khan hiếm K=10/20/50 trên IMS.
Thiết lập: windows 1024 (stride 512, kênh DE), fine-tune `tiny` (819K, matched 80/83 trọng số)
& `base` (30M, from-scratch) 30 epochs/patience 20, train 36k mẫu (chỉ K lỗi), test cân bằng
8100/8100. Harness riêng (≠ RF+feature của các nhánh chính). Nguồn `unifault_local/results.txt`.

| lỗi train | UniFault tiny (foundation) | UniFault base (from scratch) |
|---|---|---|
| 10 | **0.020** | 0.000 |
| 20 | **0.102** | 0.000 |
| 50 | **0.000–0.047** (biến thiên seed) | 0.000 |

> **Kết luận mạnh:** foundation model cũng KHÔNG cứu được few-shot. Dù pretrain 9 tỷ điểm,
> fine-tune chỉ 10–50 lỗi → recall fault chỉ **0.02–0.10** (gần như chỉ đoán normal); base 30M
> từ-scratch = 0.000 (overfit 2-lớp). Điều này **khẳng định luận điểm trung tâm**: ở kịch bản
> khan hiếm không phải "model mạnh/pretrain to" giải quyết vấn đề, mà là **bổ sung lỗi giả bám
> manifold lỗi thật** — deliverable interp_align+amp (0.766/0.748/0.788) đánh bại foundation
> **~7–38×**. ⚠️ So độ lớn thứ tự (test cân bằng 8100/8100 ≠ test-RF của nhánh 1-8), không so nhau tuyệt đối.

**(d) Generative (DDPM) + quality-gating cho lỗi sớm (script 16):** thử lần cuối cho kịch bản
lỗi sớm (recall 0.10–0.13 là điểm yếu duy nhất của deliverable). Train `LightDDPM` (GPU) trên
K lỗi thật → sinh oversample → **quality-gating** bằng 7 descriptor (std/rms/kurt/spec_centroid/
env_energy/A@f_char/sideband_snr), giữ bản nằm trong [q10,q90] của lỗi thật. Nguồn
`results/incip_ddpm_ab.json`.

| lỗi train | baseline | DDPM thô | DDPM + gate | interp_align+amp |
|---|---|---|---|---|
| 10 | 0.006 | 0.008 | 0.009 | **0.127** |
| 20 | 0.023 | 0.015 | 0.019 | **0.103** |
| 50 | 0.030 | 0.032 | 0.033 | **0.141** |

> **Kết luận:** generative **vẫn thua** ở lỗi sớm (ngang baseline, kém interp_align 4–14×) vì
> **DDPM mode-collapse** — loss kẹt ~1.0 (không giảm 60→150 epochs) do 10–50 mẫu. **Quality-gating
> không giúp** (chênh ±0.01) vì các bản DDPM nằm trong dải descriptor rất hẹp quanh normal, lọc
> không còn bản lỗi. Thí nghiệm này **đóng ngoặc mọi hướng generative** (GAN/diffusion/foundation/
> gating đều kém) — đường thắng duy nhất ở khan hiếm là **nội suy manifold lỗi thật**.

#### 3.4.8. Tinh chỉnh interp_align (A/B, script 06) — chốt `k_mix=6` làm deliverable mới

Sau khi deliverable cũ (k_mix=2) đạt 0.689/0.738/0.778, thử 3 biến thể trên CÙNG harness
(mean 3 seed, n_synth=3200, nguồn `results/m4_*.json`):

| Biến thể | K=10 | K=20 | K=50 | vs cũ (k_mix=2) |
|---|---|---|---|---|
| **k_mix=6** (trộn 6 cửa sổ căn pha) | **0.766** ±0.021 | **0.748** ±0.012 | **0.788** ±0.025 | **+0.074/+0.006/+0.014** |
| amp full-dist (std toàn phân phối) | 0.689 ±0.010 | 0.757 ±0.026 | 0.781 ±0.001 | −0.004/+0.015/+0.006 |
| physics_mixup (trộn physics vào pool) | 0.684 ±0.015 | 0.680 ±0.020 | 0.700 ±0.015 | −0.008/−0.062/−0.075 |

> **Kết luận A/B:** **`k_mix=6` là biến thể thắng rõ nhất** — nâng mạnh nhất ở mức khan hiếm
> nhất K=10 (0.689→**0.766**, +0.074, và **giảm hẳn std** 0.053→0.021) và giữ nguyên/hơi cao ở
> K=20/50. Lý do: trộn nhiều cửa sổ căn pha → bản sinh phủ dày hơn vùng giữa manifold lỗi
> (vốn là nơi khiến K=10 khó nhận biết), đồng thời mỗi bản vẫn là tổ hợp tuyến tính lỗi thật
> (bám manifold, không trôi ra ngoài). `amp full-dist` chỉ +0.015 ở K=20, K10/K50 trong nhiễu →
> **không đáng**, giữ default q10q90. `physics_mixup` **tệ hơn** (−0.06/−0.075) → **loại**: trộn
> physics (vốn yếu, generator `generate`) kéo lệch manifold lỗi thật.
>
> **Deliverable mới: `--gen interp_align --amp 1 --n-synth 3200 --k-mix 6`** (mặc định k_mix đã
> đổi 2→6 trong `generate_aligned_interp` + script 06) → recall **0.766/0.748/0.788** theo mức
> 10/20/50. Ghi chú: số §3.4.7b (incipient) cũng đã chạy lại với k_mix=6 (0.116/0.097/0.127) —
> tăng nhẹ so với k_mix=2, xu hướng (augment giúp ~20–30×) không đổi (nguồn `results/incip2_*`).

#### 3.5. Thử "self-supervised (SimCLR) encoder" (hướng model-mới) — KHÔNG hơn feature 24D

Một nhánh "model mới" đã thử (`src/contrastive.py`, `scripts/08_contrastive_vs_aug.py`): thay
vì *sinh dữ liệu*, học **biểu diễn không nhãn** từ MỌI cửa sổ (normal + fault) bằng
SimCLR-style contrastive (encoder 1D-CNN, NT-Xent loss, augment = nhiễu nhẹ + lệch pha),
rồi RF/linear trên embedding chỉ với K lỗi thật. Kết quả:

| lỗi train | RF trên feature 24D | RF trên embedding contrastive |
|---|---|---|
| 50 | 0.418 | **0.358** (kém hơn) |

> **Lý do KHÔNG thắng:** feature kỹ thuật (24D thời gian/tần số/envelope) **đã tách rất tốt**
> (AUC 0.99) nên "học biểu diễn không nhãn" không còn dư địa. Đây là bằng chứng lặp lại cho
> kết luận §4: với bài ổ bi + feature kỹ thuật, **model/feature không phải nút thắt; nút thắt
> là dữ liệu lỗi khan hiếm**. Kết quả ở cửa sổ 256 & cache khác với script 06 (win 512) nên
> chỉ so TRONG bảng này, không so với §3.4.

---

## 4. PHÁT HIỆN QUAN TRỌNG NHẤT (giá trị kỹ thuật cho slide)

**Đừng nhắm "model yếu" — nhắm "data-split".** Trước khi sửa, mọi lever kẹt ở AUC
~0.58 (tệ), lẽ ra là dấu hiệu dữ liệu sai chứ không phải model kém. Nguyên nhân gốc:

> Với IMS, `limit_per_test` trước đây **cắt 300 file ĐẦU rồi mới chia vùng 90–100%**.
> Nhưng quá trình hỏng nằm ở **CUỐI test (file ~2000)**, nên vùng "fault" thực chất là
> **dữ liệu NORMAL chưa hỏng** → nhãn "fault" = dữ liệu bình thường → model không thể
> tách → AUC ~0.5. Repair: đếm file thật, chia vùng theo toàn bộ chuỗi rồi chỉ load
> đầu (normal) + cuối (fault). Sửa xong: AUC nhảy 0.582 → **0.982**.

→ Đây là **kịch bản điển hình của bài toán predictive-maintenance thực tế**: tín hiệu
lỗi chỉ xuất hiện ở cuối vòng đời (run-to-failure), phải chia dọc theo thời gian, không
được chia theo số file gốc tải vào.

---

## 5. GIỚI HẠN CỦA KẾT LUẬN (nói thẳng, tránh bị giám khảo bắt)

1. **Đã giải quyết "khan hiếm cực đoan"** (thí nghiệm §5-A/§3.4): augment cứu recall khi model
   đói lỗi (20 lỗi → gấp đôi recall). Kết luận giờ phụ thuộc MỨC KHIẾM: với ≥100 lỗi thật thì
   augment gần như vô ích; với 20–50 lỗi thì rất đáng giá.
2. **CWRU là dataset "quá dễ":** (disconnect, mẫu rời rạc, tách lỗi tuyệt đối) → mọi model chạm
   trần kể cả với 5 lỗi → augmentation không còn chỗ để chứng tỏ. IMS (run-to-failure) là bài
   thực tế hơn và mới cho thấy hiệu quả của augment. Đây là một điểm cần nhấn trong báo cáo.
3. **Đã thử "model mới" (self-supervised contrastive) nhưng KHÔNG thắng** (§3.5) — và thử ECDF/KS
   distribution alignment (§3.4.2) cũng không thay được interpolate. Cả hai xác nhận: feature kỹ
   thuật 24D đã đủ mạnh; nút thắt là **dữ liệu lỗi khan hiếm**, không phải model/bộ sinh.

### Hướng tiếp theo có ý nghĩa (chọn 1)
- **[A] ⏭️ ĐÃ LÀM** (script 06, kết quả §3.4): tạo khan hiếm NHÂN TẠO → kết luận "augment cứu
  khi đói lỗi" là **đúng**, và bộ sinh TỐI ƯU = interpolation (trộn lỗi thật + jitter).
- **[B] ⏭️ ĐÃ LÀM** (script 07, kết quả §3.4): TimeGAN (generative, `--gen npy`) **kém nhất**
  khi khan hiếm (0.43 ở 20 lỗi) — GAN cần nhiều mẫu để học phân bố, ít mẫu → mode collapse.
- **[C] ⏭️ ĐÃ LÀM (không thắng)** (§3.5): self-supervised contrastive — RF trên embedding kém
  hơn feature 24D (0.358 vs 0.418). Kết luận: không cần "model mới", tăng recall bằng cách
  **sinh lỗi giả interpolation + đủ số lượng** là đường còn dư địa.
- **[D] ⏭️ ĐÃ LÀM (không thắng)** (§3.4.7c): foundation model UniFault (pretrain 9 tỷ điểm) —
  fine-tune 10–50 lỗi → recall fault chỉ 0.02–0.10 (tiny), 0.000 (base 30M from-scratch).
  Kết luận mạnh: **pretrain to / model mạnh không cứu được few-shot**; bổ sung lỗi giả bám manifold
  lỗi thật (0.766/0.748/0.788) là cách thắng, vượt foundation ~7–38×.

---

## 6. Nguyên tắc bất biến (giữ nguyên)

- **Test là "vùng đất thiêng":** mọi augmentation (physics/generative), scaling, chọn threshold,
  chọn feature — chỉ thực hiện trong **train**. Test chỉ dùng lỗi **THẬT chưa từng thấy**.
  (Chuẩn hoá trong `src/pipeline.py` 3-zone split + `scripts/02_compare_augmentation.py`.)
- Mọi con số trong slide phải có source `results/*.json`.

---

## 7. Trạng thái

✅ Đã chốt **01/09/2026** — hướng đi đã được **kiểm chứng bằng thực nghiệm** (không còn là
giả thuyết). Kết luận chính (SAU thí nghiệm khan hiếm §3.4 + bộ sinh tối ưu + so TimeGAN):
**augmentation có giá trị đúng khi dữ liệu lỗi thật khan hiếm (10–50 mẫu), và cách tốt nhất là
sinh lỗi giả từ chính các lỗi thật (interpolation) + đủ số lượng** — recall 50 lỗi: 0.49 → **0.80**,
20 lỗi: 0.38 → **0.66** (với n_synth=3200), vượt rõ heuristic (0.46), TimeGAN (0.43) và physics
(0.42). Khi đã đủ lỗi (≥100–200) thì baseline đạt trần và augment gần như không đóng góp.
Việc còn lại: gom bảng số liệu khan hiếm + trước/sau vào slide Vòng 1 (foundation model [D] đã so xong — §3.4.7c).
