# Tài nguyên đã verify — A1 Predictive Maintenance

## 1. Dataset (đã test link)

| Dataset | Dùng cho | Link | Ghi chú |
|---|---|---|---|
| **NASA IMS Bearings** | RUL + anomaly | https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip | run-to-failure thật, 4 chuỗi, 8 sensor |
| **FEMTO/PRONOSTIA** | RUL | https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip | accelerated life test, 17 máy |
| **CWRU Bearing** | fault classification / AE | https://engineering.case.edu/bearingdatacenter/download-data-file | tải tay: Normal Baseline + 12k Drive End Fault; file .mat |
| NASA Turbofan (C-MAPSS) | RUL (thay thế nếu muốn nhiều data) | https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip | PID + sensor, chuẩn PHM |

> Bản PHM Society mirror (nếu NASA chậm): https://data.phmsociety.org/nasa/

## 2. Paper nên đọc theo thứ tự (link đã verify 08/2026)

| # | Paper | Năm/Venue | Link | Vì sao nên đọc |
|---|-------|-----------|------|----------------|
| 1 | **FaultDiffusion: Few-Shot Fault Time Series Generation with Diffusion Model** | arXiv 2511.15174 (11/2025) | https://arxiv.org/abs/2511.15174 PDF: https://arxiv.org/pdf/2511.15174 | **Trùng gần 100% đề A1** — sinh dữ liệu lỗi cho industrial monitoring khi ít data. Đọc trước, cite trong slide. |
| 2 | **TimeGAN (Time-series Generative Adversarial Networks)** | NeurIPS 2019 | https://papers.nips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf | Nền tảng GAN sinh chuỗi thời gian + metric đánh giá (discriminative/predictive). Code: repo `jsyoon0823/TimeGAN`. |
| 3 | **Diffusion-TS: Interpretable Diffusion for General Time Series Generation** | ICLR 2024 | https://openreview.net/pdf?id=4h1apFjO99 | SOTA sinh TS bằng diffusion có điều kiện theo class — nhánh "augmentation thật". Code: repo `Y-debug-sys/Diffusion-TS`. |

> Lưu ý: nhiều bài cùng tên "FaultDiffusion" có thể tồn tại; dùng đúng arXiv ID 2511.15174.
> (Tôi sai 2 lần khi đoán link giả — luôn verify bằng arXiv API trước khi tải.)

Script tải PDF về `papers/`:
```bash
bash scripts/download_papers.sh
```

## 3. Repo cần clone (đã verify tồn tại)

```bash
# Trong A1_predictive_maintenance/
mkdir -p src && cd src
git clone https://github.com/Y-debug-sys/Diffusion-TS.git   # ICLR'24, cần GPU CUDA
git clone https://github.com/jsyoon0823/TimeGAN.git         # NeurIPS'19, chạy được CPU
```

- Diffusion-TS: có `Tutorial_0/1/2.ipynb` dạy train/sample; đổi dataset vào thư mục `Data/`.
- TimeGAN: chạy `python3 -m main_timegan.py --data_name <x>`, có notebook tutorial.
- Cả hai đều sinh data giả → dùng làm "data lỗi giả" cho supervised step.

## 4. Thuật toán baseline cần thành thạo

1. **Autoencoder + reconstruction error** — phát hiện bất thường không giám sát (script 01).
2. **XGBoost + feature engineering** — feature miền thời gian (mean, std, RMS, peak, skew, kurtosis) + miền tần số (FFT bins, spectral centroid, spectral flatness). Rất mạnh với tabular/TS ngắn.
3. **LSTM/GRU** — khi muốn nhấn vào tính tuần tự (optional, tốn GPU).
4. **Normalizing Flow / OCSVM** — các phương án khác cho unsupervised anomaly.

## 4b. Bối cảnh mùa trước (dùng cho slide / chiến lược trình bày)

**Đội vô địch Mùa 3 (2025):** SPARK — Viện Trí tuệ nhân tạo, Đại học Công nghệ (UET), ĐHQG Hà Nội.

**Giải pháp: "DiffusionAD"** — phát hiện lỗi bất thường trong sản xuất bằng **diffusion model, không cần dữ liệu nhãn** (unsupervised/self-supervised). Thuộc nhánh AI/Data (phát hiện lỗi linh kiện — hướng thị giác máy).

**Nguồn đối chiếu:** VnReview (khởi động mùa 4), FPT, Facebook UET/AI-UET, bài BUV về đội á quân FSiL (BUV + Bách Khoa, giải "End-to-End AI"). *(Nguồn cấp thứ cấp — nếu trích chi tiết kỹ thuật trong slide nên tìm bài VnReview/FPT gốc.)*

**Ý nghĩa chiến lược cho A1 (điểm ăn điểm slide):**
- Năm ngoái giám khảo đã trao giải cao nhất cho tư duy **"diffusion + dữ liệu lỗi khan hiếm/không nhãn"** — đúng tinh thần của A1.
- **Điểm khác biệt phải nêu:** DiffusionAD thắng ở mảng **ảnh (CV)**; đề A1 được team đưa cùng tư duy đó sang **chuỗi thời gian rung (TS)** — chưa ai chứng minh ở nhóm này.
- **Tránh:** copy y hệt họ (thắng bằng unsupervised anomaly); A1 yêu cầu deliverable "bộ sinh dữ liệu + so sánh trước/sau" → nhấn mạnh nhánh augmentation có giám sát.

## 5. Checklist nộp (đối chiếu deliverable BTC)

- [ ] Bộ sinh dữ liệu bất thường (diffusion/GAN) — code + cách dùng
- [ ] Tập dữ liệu tăng cường (file .npy/csv)
- [ ] Báo cáo so sánh độ chính xác **trước/sau** augmentation (bảng + PR curve)
- [ ] Dashboard hiển thị cảnh báo + RUL