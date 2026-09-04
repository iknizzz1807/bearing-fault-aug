#!/usr/bin/env python3
"""SCRIPT 1/3 — Baseline (mốc so sánh): AutoEncoder tách bất thường.

╔══════════════════════════════════════════════════════════════════════╗
║  Script này CHẠY CÁI GÌ? (tóm tắt 1 đoạn)                            ║
║  Dữ liệu lỗi hiếm → ta không đủ mẫu lỗi để dạy model "đây là lỗi".    ║
║  Nên làm ngược lại: dạy model biết "thế nào là BÌNH THƯỜNG"          ║
║  (chỉ cho AutoEncoder học dữ liệu normal), rồi đo độ SAI LỆCH khi     ║
║  tái tạo (reconstruction error). Chuỗi nào tái tạo "trật" quá mức     ║
║  → nghi là bất thường.                                               ║
║                                                                      ║
║  Cụ thể từng bước:                                                   ║
║   1. Tải tín hiệu rung, cắt thành các cửa sổ (window), tính 16 đặc    ║
║      trưng (mean, rms, skewness, ...) cho mỗi window.                ║
║   2. Chỉ dùng NORMAL để huấn luyện AutoEncoder (AE): AE học cách né   ║
║      đúng dữ liệu bình thường ra đầu ra.                              ║
║   3. Với mọi window (kể cả lỗi), lái AE tái tạo lại nó → tính lỗi    ║
║      tái tạo (số càng lớn = càng "lạ"), chọn ngưỡng cảnh báo.        ║
║   4. Window nào có lỗi tái tạo vượt ngưỡng → gọi là ANOMALY (lỗi).   ║
║   5. Đo model bắt đúng bao nhiêu % lỗi thật (recall), cảnh báo đúng   ║
║      bao nhiêu % (precision) và AUC.                                  ║
║                                                                      ║
║  VAI TRÒ trong dự án:                                                ║
║    Đây là con số "TRƯỚC augmentation" — là MỐC để script 2 so sánh:  ║
║    sau khi thêm dữ liệu lỗi giả, model có bắt lỗi thật tốt hơn  ●   ║
║    không? Nếu script 2 "sau" > "trước" → augmentation có ích →       ║
║    đây chính là deliverable mà Ban giám khảo chấm điểm ở đề A1.       ║
╚══════════════════════════════════════════════════════════════════════╝

CÁCH CHẠY:
    python scripts/01_baseline_anomaly.py --dataset ims --limit 300

CÁC THAM SỐ QUAN TRỌNG:
    --dataset  ims hoặc cwru      (nên dùng ims — đủ cả normal lẫn fault)
    --limit    300                (tải 300 file cho mỗi test-run; giữ máy không treo)
    --epochs   60                 (số vòng luyện của AutoEncoder)
    --thr      99.0               (ngưỡng: "bất thường" là những window nằm trên
                                   percentile 99 của lỗi tái tạo trên tập normal)

ĐẦU RA TIẾNG VIỆT CÓ NGHĨA LÀ GÌ? (ví dụ kết quả thật đã chạy):
    threshold  0.0164 : nếu lỗi tái tạo của 1 window > 0.0164 → coi là lỗi.
    precision  0.387  : trong 100 lần cảnh báo "có lỗi" → 39 lần là đúng thật.
    recall     0.031  : trong 100 lỗi thật có trên test → model bắt được 3 cái.
    f1         0.057  : trung bình điều hoà của precision & recall (một con số gộp).
    auc        0.524  : 0.5 = đoán bừa, 1.0 = phân biệt hoàn hảo.
                       (0.52 gần 0.5 → AE thuần "tách bất thường" chưa giỏi —
                       đó là lý do đề bài muốn ta thêm dữ liệu lỗi giả để
                       model GIÁM SÁT (script 2) bắt lỗi tốt hơn.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ae                      # noqa: E402  (AutoEncoder + thước đo trong thư mục src/)
from src import pipeline as pipe        # noqa: E402  (tải & chia dữ liệu dùng chung)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    # ---------- đọc tham số dòng lệnh ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--win", type=int, default=512)      # độ dài 1 window (điểm dữ liệu rung)
    ap.add_argument("--stride", type=int, default=256)   # bước trượt giữa 2 window liền nhau
    ap.add_argument("--epochs", type=int, default=60)    # số vòng AE luyện
    ap.add_argument("--thr", type=float, default=99.0,   # percentile làm ngưỡng cảnh báo
                    help="ngưỡng = percentile reconstruction error trên tập normal")
    ap.add_argument("--limit", type=int, default=300,
                    help="IMS: số file/test tải tối đa (giữ máy không treo)")
    ap.add_argument("--out", default=str(RESULTS / "baseline_ae.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(42)   # cố định seed → chạy lại ra kết quả giống hệt (tái lập được)
    RESULTS.mkdir(exist_ok=True)

    # ---------- 1. Chuẩn bị dữ liệu ----------
    # load_normal_fault_signals → trả về (runs_normal, runs_fault):
    #   mỗi cái là danh sách các "test-run"; mỗi test-run là một chuỗi tín hiệu theo thời gian.
    # split_and_scale → chia train/test/chuẩn hoá đúng quy ước của dự án
    #   (xem docstring trong src/pipeline.py). Nó trả về dict d["..."] chứa sẵn:
    #   - Xn_train : windows NORMAL dùng để huấn luyện (model học "bình thường" là gì)
    #   - Xf_train : vài windows LỖI THẬT hiếm hoi (20% số lỗi)
    #   - Xn_test  : windows NORMAL chưa từng thấy khi train (test đánh giá)
    #   - Xf_test  : windows LỖI THẬT chưa từng thấy khi train (test đánh giá)
    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    d = pipe.split_and_scale(signs_n, signs_f, args.win, args.stride, rng)
    # X_base  = toàn bộ train (normal + lỗi thật hiếm) → script này chỉ lấy phần normal
    # X_test  = toàn bộ test  (normal chưa thấy + lỗi thật) → sẽ chấm điểm ở đây
    X_base, _, X_test, y_test = pipe.to_train_test_arrays(d)
    print(f"[data] train normal={d['n_normal_train']} fault={d['n_fault_train']} | "
          f"test normal={d['n_normal_test']} fault={d['n_fault_test']} (stride={d['stride_eff']})")

    # ---------- 2. Chỉ cho AE học NORMAL ----------
    # Ý tưởng: AE chỉ nhìn "cái gì là bình thường". Lúc test, nó sẽ tái tạo kém
    # (lỗi tái tạo lớn) với dữ liệu bất thường / lỗi — đó là cơ chế phát hiện.
    Xn_train = d["Xn_train"]    # chỉ lấy phần NORMAL (không lấy lỗi — đúng triết lý vô giám sát)
    print(f"[train] AE học {len(Xn_train)} windows bình thường (feature dim={Xn_train.shape[1]}) ...")
    # train_ae trả về (mô hình đã luyện, lỗi tái tạo trên CHÍNH tập train)
    # rec_err_train = "AE tái tạo dữ liệu train trật bao nhiêu" — dùng để đặt ngưỡng
    model, rec_err_train = ae.train_ae(Xn_train, epochs=args.epochs)

    # ---------- 3. Chấm ngưỡng cảnh báo ----------
    # Trên tập normal, ta đo mức lỗi tái tạo "bình thường" nhất là bao nhiêu.
    # threshold = percentile 99 của các lỗi trên tập normal:
    #   → 99% window bình thường có lỗi tái tạo NHỎ HƠN ngưỡng này.
    # Khi đó, 1 window có lỗi tái tạo VƯỢT ngưỡng → không giống normal → báo "lỗi".
    # (Càng tăng --thr thì càng ít báo động → recall giảm, precision tăng.)
    threshold = ae.threshold_from_err(rec_err_train, args.thr)
    print(f"[thr] ngưỡng anomaly (P{args.thr}): {threshold:.4f}")

    # ---------- 4. Đánh giá trên test (normal-chưa-thấy + lỗi THẬT) ----------
    # Cho toàn bộ test đi qua AE → mỗi window được 1 điểm "bất thường" (lỗi tái tạo).
    # report_anomaly so sánh: điểm > ngưỡng thì kết luận "LỖI", rồi đối chiếu
    # với nhãn thật (y_test) tính ra precision/recall/F1/AUC.
    scores = ae.rec_error(model, X_test)
    report = ae.report_anomaly(y_test, scores, threshold)
    print("\n=== KẾT QUẢ BASELINE (TRƯỚC augmentation) ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # ---------- 5. Lưu kết quả ra JSON (để script sau/đồ thị dùng lại) ----------
    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "report": report, "threshold": threshold}, f, indent=2)
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()