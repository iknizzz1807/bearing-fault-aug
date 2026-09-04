#!/usr/bin/env python3
"""Ghép 4 phần giáo trình thành 1 master markdown (bìa + chương, bỏ mục lục lặp)."""
import re
from pathlib import Path

AGENTS = Path("/tmp/opencode/agents")
OUT = Path("/tmp/opencode/giaotrinh_A1.md")

cover = """# GIÁO TRÌNH BẢO TRÌ DỰ ĐOÁN & SINH DỮ LIỆU LỖI

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
- **Timeline:** đăng ký ≤31/08, Factory Tour 11/09, nộp Vòng 1 slide ≤12/10, chung kết 16/11.

---
"""

parts = [
    ("part1_math.md", "PHẦN 1 · NỀN TẢNG TOÁN HỌC CHO ML/DL", 1),
    ("part2_ml_core.md", "PHẦN 2 · MACHINE LEARNING CỐT LÕI & ANOMALY DETECTION", 2),
    ("part3_dl_generative.md", "PHẦN 3 · DEEP LEARNING & GENERATIVE MODELS", 3),
    ("part4_catalog_roadmap.md", "PHẦN 4 · KHO TÀI NGUYÊN, HƯỚNG ĐI & KẾ HOẠCH", 4),
]


def clean_embedded_toc(text: str) -> str:
    """Bỏ các mục 'Mục lục'/'MỤC LỤC' nhúng bên trong mỗi phần (TOC sẽ tự sinh ở cuối)."""
    # xoá heading mục lục + các bullet của nó cho tới heading cấp 1 hoặc 2 kế tiếp
    lines = text.split("\n")
    out, skipping = [], False
    for ln in lines:
        if re.match(r"^#{1,6}\s*(Mục lục|MỤC LỤC|Mục Lục)\s*$", ln.strip()):
            skipping = True
            continue
        if skipping:
            if re.match(r"^#{1,6}\s", ln.strip()) or (ln.strip() and not ln.startswith(" ") and not ln.startswith("-") and not ln.startswith("*") and not ln.startswith("1.")):
                skipping = False
                out.append(ln)
            elif ln.strip() == "":
                continue
            else:
                continue
        elif ln.strip() == "## Mục lục" or ln.strip() == "## MỤC LỤC":
            skipping = True
            continue
        else:
            out.append(ln)
    return "\n".join(out)


def demote_h1_to_h2(text: str, label: str) -> str:
    """Đổi dòng '# TITLE' đầu tiên thành '## LABEL — TITLE'."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if re.match(r"^#\s+", ln):
            title = ln[2:].strip()
            lines[i] = f"## {label} — {title}"
            # thêm page-break trước mỗi phần
            lines.insert(i, '<div style="page-break-before: always;"></div>')
            break
    return "\n".join(lines)


chunks = [cover]
for fname, label, _ in parts:
    p = AGENTS / fname
    txt = p.read_text(encoding="utf-8")
    txt = clean_embedded_toc(txt)
    txt = demote_h1_to_h2(txt, label)
    chunks.append(txt)

OUT.write_text("\n\n".join(chunks), encoding="utf-8")
print(f"OK  {OUT}  ({OUT.stat().st_size//1024} KB, {OUT.read_text().count(chr(10))} dòng)")