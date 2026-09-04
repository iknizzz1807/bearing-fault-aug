# Bài toán A1 — Predictive-Maintenance AI Despite Scarce Fault Data

> Nguồn: trang chính thức Denso Factory Hacks 2026 (đơn vị tổ chức FPT + DENSO Việt Nam).
> File này lưu **chi tiết bài toán + thể thức + deliverable** để đối chiếu khi làm slide/báo cáo.
> Hướng đi & kết quả thực nghiệm xem **`SPEC.md`**; tổng quan repo xem **`README.md`**.

---

## 1. Chủ đề lớn (Theme Core 2025)

**"Kiến tạo Nhà máy Thông minh trong Tương lai — Dự đoán dẫn lối. Tự động linh hoạt. Kết nối tức thời."**

Bài A1 thuộc nhóm **01 · PREDICTIVE & KNOWLEDGE AI**:
> Khai thác sức mạnh của AI nhằm chuyển đổi dữ liệu và kinh nghiệm thành tri thức có thể
> tái sử dụng, đồng thời nâng cao khả năng dự báo và hỗ trợ vận hành.

Mục tiêu nhóm "Các bài toán có thể xoay quanh" (những mục khớp nhất với A1):
- Ứng dụng AI **dự báo lỗi, rủi ro hoặc các tình huống bất thường** trong sản xuất.

## 2. Tổng quan bài toán A1

| Trường | Nội dung |
|---|---|
| **Mã bài** | A1 |
| **Tên** | Predictiv-Maintenance AI Despite Scarce Fault Data |
| **Tóm tắt** | **AI bảo trì dự đoán khi thiếu dữ liệu lỗi** — huấn luyện AI bảo trì dự đoán trong điều kiện thiếu dữ liệu về các chế độ lỗi. |

## 3. Sản phẩm đầu ra sau 3 tháng (3 deliverable BẮT BUỘC)

1. **Bộ sinh dữ liệu bất thường** (fault data generator).
2. **Tập dữ liệu tăng cường** (augmented dataset).
3. **Báo cáo so sánh độ chính xác dự báo trước và sau khi tăng cường dữ liệu.**

## 4. Công nghệ gợi ý (cột chính thức của BTC)

```
Python - Mô phỏng vật lý - GAN/Diffusion cho dữ liệu chuỗi thời gian - Scikit-learn/PyTorch
```

| Cụm | Ứng dụng vào A1 | Trạng thái trong repo |
|---|---|---|
| **Python** | Ngôn ngữ chính | ✅ |
| **Mô phỏng vật lý** | Σinh lỗi giả bằng BPFO/BPFI/BSF/FTF (fault injection) | ✅ `src/physics.py`, `scripts/05_physics_aug.py` |
| **GAN/Diffusion cho chuỗi thời gian** | Học phân bố lỗi → sinh mẫu giả | ✅ TimeGAN (`src/timegan_torch.py`, `scripts/03`); ⚠️ Diffusion cần GPU lớn (chưa chạy) |
| **Scikit-learn** | Model phân loại (RandomForest) + metrics | ✅ `scripts/02_compare_augmentation.py` |
| **PyTorch** | Deep model (AE, TimeGAN, LSTM/CNN) | ✅ `src/ae.py`, `04_train_lstm.py` |

> **Diễn giải:** cột công nghệ gợi ý chỉ là **cách làm** deliverable #1. `PyTorch` là framework,
> không phải hướng. Chi tiết các hướng đã thử & kết quả xem `SPEC.md`.

## 5. Thể thức / thông tin cuộc thi

| Mục | Thông tin |
|---|---|
| **Đơn vị tổ chức** | FPT + DENSO Việt Nam |
| **Ngày sự kiện** | 11 tháng 09 năm 2026 (Thứ Sáu) |
| **Thời gian** | 09:30 – 16:00 |
| **Email** | densofactoryhacks@gmail.com |
| **Hotline** | 0833033789 |
| **Trụ sở FPT** | Tòa FPT Tower, 10 Phạm Văn Bạch, Cầu Giấy, Hà Nội |
| **DenSO VN** | Lô E1, KCN Thăng Long, Thiên Lộc, Hà Nội |

### Cấu trúc cụm bài toán (9 bài, 5 chủ đề)
- **PREDICTIVE & KNOWLEDGE AI**: A1, A2, A3
- **HUMANOID & ROBOT**: H1, H2, H3
- **DATA UTILIZATION**: D1, D2, D3

## 6. Đối chiếu với thực tế làm được (quan trọng)

Nội dung brief khớp với những gì repo đã triển khai:

| Deliverable brief | Trạng thái | Ghi chú |
|---|---|---|
| Bộ sinh dữ liệu bất thường | ✅ | 2 bộ sinh: TimeGAN (`03`), physics BPFO (`05`), heuristic (`02`) |
| Tập dữ liệu tăng cường | ✅ | Ghép lỗi giả vào train (thực hiện trong train, không lộ test) |
| Báo cáo trước/sau | ✅ | `results/*.json` (IMS + CWRU) |

**Kết quả đo được:** baseline RF + feature đạt trần (IMS recall 0.886/AUC 0.98; CWRU 1.0/1.0);
augmentation **không cải thiện**. Vì vậy trọng tâm báo cáo nên là: *so sánh + giải thích vì sao
sinh dữ liệu không cần thiết cho các bộ dữ liệu này, và nêu rõ giới hạn (chưa mô phỏng khan hiếm
cực đoan)* — xem `SPEC.md §5`.

## 7. Liên kết tài liệu

- Hướng đi + toàn bộ kết quả: `SPEC.md`
- Tổng quan repo + cách chạy: `README.md`
- Giáo trình kỹ thuật (BPFO/feature/GAN): `docs/giaotrinh_A1.md` (+ `.pdf`)
- Nguồn dữ liệu/paper: `RESOURCES.md`
