#!/usr/bin/env python3
"""SCRIPT 5 — Tăng cường dữ liệu lỗi bằng MÔ PHỎNG VẬT LÝ (BPFO/BPFI) chuyên sâu.

Đây là "lever" thứ ba trong bộ so sánh augmentation của đề A1 (thiếu dữ liệu lỗi).
Ba OPTION sinh lỗi giả đã thử, mỗi bên một cơ chế khác nhau (đặt cạnh nhau để
đối chứng, xem SPEC.md §1, §2):
    - heuristic (script 02): biến đổi biên độ + nhiễu — nhanh nhưng "bừa".
    - TimeGAN  (script 03) : học phân bố lỗi thật — khó kiểm soát, tốn GPU.
    - PHYSICS  (script này): chèn xung va đập đúng tần số đặc trưng ổ bi lăn vào
                             nền rung KHỎE THẬT — KIỂM SOÁT ĐƯỢC, ĐÚNG VẬT LÝ.

Điểm khác biệt của script này: lỗi giả được sinh BẰNG VẬT LÝ ổ bi lăn, không nhờ GAN
hay heuristic biên độ. Dùng src/physics.py — chèn tín hiệu va đập lặp đúng tần số đặc
trưng (BPFO/BPFI/BSF/FTF) vào nền rung KHỎE THẬT:

    x_f(t) = x_h(t) + A * Σ_k h(t - k*T) * w(t) + n(t)

Vì sao chọn BPFO (mặc định --fault-type bpfo)?
    BPFO (Ball Pass Frequency Outer) là nhịp hạt lăn đập vết nứt ở rãnh NGOÀI —
    một lỗi ổ bi phổ biến, xung va đập RÕ nhất nên dễ thấy model có "bắt" được
    hành vi có cấu trúc tần số hay không. Có thể đổi BPFI/BSF/FTF để so sánh chéo.

Quy trình (đúng anti-leakage, giữ nguyên cách đánh giá của script 02):
    1. Tải tín hiệu normal + fault (pipe.load_normal_fault_signals).
    2. Lấy cửa sổ KHỎE từ vùng TRAIN-normal (tránh lòe test/grey).
    3. Với mỗi cửa sổ khỏe, chạy physics_aug_worker → n_copy cửa sổ lỗi giả.
    4. raw_features trên cửa sổ lỗi giả → feature RAW (chưa chuẩn hoá).
    5. pipe.split_and_scale → chia train/test + scaler (fit trên train).
    6. Chuẩn hoá lỗi giả bằng đúng scaler train.
    7. Train RF "trước" (normal + ít lỗi thật) và "sau" (+ lỗi giả).
    8. Đánh giá CẢ HAI trên CÙNG test (normal khỏe chưa thấy + lỗi thật chưa thấy).

CÁCH CHẠY:
    python scripts/05_physics_aug.py --dataset ims --fault-type bpfo --out results/compare_physics.json

KẾT QUẢ THỰC (đã chạy --fault-type bpfo, xem results/compare_physics_FIXED.json):
    recall 0.886 → 0.888 (gần như không đổi), AUC 0.9823 → 0.9724 (tụt nhẹ).
→ Physics injection KHÔNG giúp vượt RF-on-features, cùng chiều với heuristic và
TimeGAN: ở dataset này sinh lỗi giả (dù đúng vật lý) không cải thiện recall/AUC —
xem kết luận chung ở SPEC.md §3. Đây là một "comparison lever" để trình bày thẳng
trong báo cáo, không phải thất bại.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as ds          # noqa: E402
from src import features as ft      # noqa: E402
from src import physics as phys     # noqa: E402
from src import pipeline as pipe    # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Các mốc baseline đã chạy trước (dùng để đối chứng khi in kết quả)
_BASELINES = {
    "RF-24feat baseline": {"recall": 0.479, "auc": 0.585},
    "RF-16feat baseline": {"recall": 0.477, "auc": 0.582},
    "TimeGAN-npy (script 03)": {"recall": 0.478, "auc": 0.5826},
}


def _healthy_train_windows(runs_normal: list[list[np.ndarray]], win: int,
                           stride: int, rng: np.random.Generator,
                           max_windows: int) -> np.ndarray:
    """Lấy cửa sổ KHỎE từ vùng TRAIN-normal của các test-run.

    Chia mỗi run theo thời gian, lấy 0→NORMAL_TRAIN_END (70%) số cửa sổ đầu tiên
    tương ứng vùng normal khỏe dùng để huấn luyện trong pipeline. Nhờ vậy cửa sổ
    khỏe dùng để bơm lỗi giả KHÔNG nằm trong vùng test/grey → không lòe đánh giá.

    Trả về ma trận (n_windows, win). Nếu tổng vượt `max_windows` thì trộn ngẫu
    nhiên rồi giới hạn (giữ máy khỏi treo, vẫn đa dạng).
    """
    all_windows = []
    for run in runs_normal:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride)
            if len(w):
                i_train = max(1, int(len(w) * pipe.NORMAL_TRAIN_END))
                all_windows.append(w[:i_train])
    if not all_windows:
        raise SystemExit("Không lấy được cửa sổ khỏe nào — kiểm tra dữ liệu/--win/--stride.")
    stacked = np.concatenate(all_windows)
    if len(stacked) > max_windows:
        idx = rng.choice(len(stacked), size=max_windows, replace=False)
        stacked = stacked[idx]
    return stacked


def _make_synthetic_faults(healthy_windows: np.ndarray, fault_type: str, fs: float,
                           rotation_hz: float, geometry: dict, depth: float,
                           rng: np.random.Generator, n_copy: int) -> np.ndarray:
    """Bơm lỗi giả vật lý vào từng cửa sổ khỏe.

    Mỗi cửa sổ khỏe (win,) → n_copy cửa sổ lỗi giả (mỗi bản nhiễu/pha hơi khác
    nhau nhờ dịch tần số ±2% qua rng). Ghép tất cả về ma trận (n_copy * n_windows, win).

    Lưu ý: mình bám sát logic của `physics_aug_worker`, nhưng gọi trực tiếp
    `generate_synthetic_fault` để truyền `depth` theo tham số --depth (worker gốc
    cố định depth=0.30, không nhận tham số này → --depth sẽ vô tác dụng).
    """
    f_fault = {"bpfo": phys.bpfo, "bpfi": phys.bpfi, "bsf": phys.bsf, "ftf": phys.ftf}[
        fault_type](rotation_hz, geometry)
    chunks = []
    for w in healthy_windows:
        for i in range(n_copy):
            jitter = f_fault * (1 + rng.uniform(-0.02, 0.02))
            chunks.append(phys.generate_synthetic_fault(w, jitter, fs, depth=depth, seed=i))
    return np.asarray(chunks)


def _score(clf, X: np.ndarray, y: np.ndarray) -> dict:
    """Bộ 4 chỉ số đánh giá cho 1 model trên 1 tập X (có nhãn thật y).

    - precision = % cảnh báo đúng / tổng cảnh báo
    - recall    = % lỗi thật bị bắt / tổng lỗi thật
    - f1        = trung bình điều hoà precision + recall
    - auc       = khả năng phân biệt normal vs lỗi ở MỌI ngưỡng (0.5 = đoán bừa)
    """
    p, r, f1, _ = precision_recall_fscore_support(y, clf.predict(X), average="binary",
                                                  zero_division=0)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}


def main():
    # ---------- đọc tham số dòng lệnh ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--limit", type=int, default=300,
                    help="số file/test tải (giữ máy khỏi treo)")
    ap.add_argument("--win", type=int, default=512, help="độ dài cửa sổ (mẫu)")
    ap.add_argument("--stride", type=int, default=256, help="bước trượt cửa sổ")
    ap.add_argument("--fault-type", choices=["bpfo", "bpfi", "bsf", "ftf"], default="bpfo",
                    help="loại lỗi giả: BPFO/BPFI/BSF/FTF")
    ap.add_argument("--rotation-hz", type=float, default=30.0,
                    help="tần số quay trục (Hz) — IMS chạy ~2000 RPM ≈ 30 Hz")
    ap.add_argument("--n-synth-factor", type=int, default=3,
                    help="số bản lỗi giả sinh từ MỖI cửa sổ khỏe")
    ap.add_argument("--depth", type=float, default=0.5,
                    help="biên độ xung va đập (tỉ lệ std tín hiệu khỏe)")
    ap.add_argument("--max-healthy-windows", type=int, default=2000,
                    help="giới hạn số cửa sổ khỏe dùng để bơm lỗi")
    ap.add_argument("--out", default=str(RESULTS / "compare_physics.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    RESULTS.mkdir(exist_ok=True)

    # ---------- tham số hình học ổ bi (mặc định của src/physics.py) ----------
    geometry = phys.bearing_geometry()

    # ---------- 1. Dữ liệu + chia train/test + scaler ----------
    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    d = pipe.split_and_scale(signs_n, signs_f, args.win, args.stride, rng)
    X_base, y_base, X_test, y_test = pipe.to_train_test_arrays(d)
    n_fault = len(d["Xf_train"])
    print(f"[data] train normal={d['n_normal_train']} fault={d['n_fault_train']} | "
          f"test normal={d['n_normal_test']} fault={d['n_fault_test']}")

    # ---------- 2. Cửa sổ KHỎE từ vùng TRAIN-normal (anti-leakage) ----------
    healthy = _healthy_train_windows(signs_n, args.win, args.stride, rng,
                                     args.max_healthy_windows)
    print(f"[phys] {len(healthy)} cửa sổ khỏe (từ vùng train-normal) để bơm lỗi giả")

    # ---------- 3. Bơm lỗi giả vật lý ----------
    synth_windows = _make_synthetic_faults(healthy, args.fault_type, pipe.FS,
                                           args.rotation_hz, geometry, args.depth,
                                           rng, args.n_synth_factor)
    # ---------- 4. Feature cho lỗi giả (RAW, chưa chuẩn hoá) ----------
    synth_raw = ft.raw_features(synth_windows, fs=pipe.FS)
    # ---------- 5. Chuẩn hoá bằng ĐÚNG scaler train ----------
    X_syn = ft.zscore(synth_raw, d["scaler_mu"], d["scaler_sd"])
    y_syn = np.ones(len(X_syn))
    print(f"[phys] sinh {len(X_syn)} cửa sổ lỗi giả ({args.fault_type}, "
          f"rotation={args.rotation_hz:.1f}Hz, depth={args.depth})")

    # ---------- 6. Tập train "SAU" = train gốc + lỗi giả ----------
    X_aug = np.concatenate([X_base, X_syn])
    y_aug = np.concatenate([y_base, y_syn])
    print(f"[aug] thêm {len(X_syn)} lỗi giả → train tổng {len(X_aug)} "
          f"(fault chiếm {100 * (n_fault + len(X_syn)) / len(X_aug):.1f}%)")

    # ---------- 7. Huấn luyện 2 model (RandomForest, class_weight="balanced") ----------
    clf_before = RandomForestClassifier(n_estimators=200, random_state=0,
                                        class_weight="balanced").fit(X_base, y_base)
    clf_after = RandomForestClassifier(n_estimators=200, random_state=0,
                                       class_weight="balanced").fit(X_aug, y_aug)

    # ---------- 8. Đánh giá CẢ HAI trên CÙNG test ----------
    r_before = _score(clf_before, X_test, y_test)
    r_after = _score(clf_after, X_test, y_test)

    # ---------- 9. Bảng so sánh + đối chiếu baseline ----------
    print("\n=== PHYSICS AUGMENTATION: TRƯỚC / SAU ===")
    print(f"{'metric':<10} {'trước':>12} {'sau':>12} {'Δ':>8}")
    for m in ["precision", "recall", "f1", "auc"]:
        print(f"{m:<10} {r_before[m]:>12.4f} {r_after[m]:>12.4f} {r_after[m] - r_before[m]:+.4f}")
    print("  recall = % lỗi thật bị bắt | precision = % cảnh báo đúng | AUC = phân biệt tổng quát")

    print("\n=== ĐỐI CHIẾU VỚI BASELINE ĐÃ CHẠY TRƯỚC ===")
    print(f"{'model':<28} {'recall':>8} {'auc':>8}")
    for name, v in _BASELINES.items():
        print(f"{name:<28} {v['recall']:>8.3f} {v['auc']:>8.3f}")
    print(f"{'PHYSICS-' + args.fault_type.upper() + ' (sau)':<28} "
          f"{r_after['recall']:>8.3f} {r_after['auc']:>8.3f}")
    delta = r_after["auc"] - max(v["auc"] for v in _BASELINES.values())
    print(f"\n  → Auc 'sau' so với baseline tốt nhất: {delta:+.4f}. "
          f"{'CẢI THIỆN' if delta > 0 else 'KHÔNG TỐT HƠN — đây là một phát hiện, không phải thất bại'}.")

    # ---------- 10. Lưu JSON ----------
    with open(args.out, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "gen": "physics",
            "fault_type": args.fault_type,
            "rotation_hz": args.rotation_hz,
            "depth": args.depth,
            "n_synth_factor": args.n_synth_factor,
            "before": r_before,
            "after": r_after,
            "n_synth": int(len(X_syn)),
            "n_healthy": int(len(healthy)),
            "n_test_fault": int((y_test == 1).sum()),
            "n_test_normal": int((y_test == 0).sum()),
        }, f, indent=2)
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
