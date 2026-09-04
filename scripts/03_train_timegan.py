#!/usr/bin/env python3
"""SCRIPT 3/3 — Huấn luyện TimeGAN để SINH "LỖI GIẢ" cho augmentation.

╔══════════════════════════════════════════════════════════════════════╗
║  SCRIPT NÀY LÀM GÌ? (tóm tắt 1 đoạn)                                  ║
║  TimeGAN (NeurIPS 2019) là một TRONG CÁC OPTION sinh dữ liệu lỗi giả  ║
║  loại GENERATIVE (học phân bố lỗi thật rồi sinh lỗi giả giống thật).  ║
║  Ba hướng đã thử cho đề A1: heuristic (script 02), TimeGAN (script     ║
║  này), physics (script 05). Đây là bên "học — sinh", đối trọng với     ║
║  physics dùng công thức (xem SPEC.md §1).                              ║
║  1. Ăn vào: các window LỖI THẬT (90–100% cuối mỗi test — rất hiếm).   ║
║  2. Học: phân bố dữ liệu của lỗi thật này (hình dạng, tần số, ...).   ║
║  3. Sinh: hàng nghìn window "lỗi giả" GIỐNG THẬT mà lúc trước ta       ║
║     không hề có (vì lỗi thật chỉ có vài nghìn, normal có triệu).      ║
║  4. Lưu: lỗi giả dưới dạng RAW feature ra file .npy → script 2        ║
║     (--gen npy) đọc, chuẩn hoá bằng scaler train trộn vào tập train.  ║
║                                                                       ║
║  ĐIỂM YẾU CỦA GENERATIVE (lý do đặt cạnh physics): không chỉnh được   ║
║  hình dạng lỗi, dễ mode collapse/nét phân bố, tốn GPU.                 ║
║                                                                       ║
║  KẾT QUẢ THỰC (đã chạy — xem SPEC.md §3.1): augmentation KHÔNG giúp.  ║
║  recall 0.886 → 0.880, AUC 0.9823 → 0.9826 (sau khi sửa split). Cả ba ║
║  lever heuristic/physics/TimeGAN đều vậy → kết luận chung ở SPEC.       ║
║  Script được GIỮ LẠI như một experiment/deliverable của đề (bộ sinh    ║
║  dữ liệu lỗi), không phải bước bắt buộc cho kết quả cuối.              ║
║                                                                       ║
║  KỸ THUẬT: repo gốc là TensorFlow 1.15 → hỏng trên Py3.11; ta dùng     ║
║  bản viết lại PyTorch (src/timegan_torch.py) trung thành paper. Chạy   ║
║  trên GPU (CUDA) khi có, ngược lại CPU. Kiến trúc 4 mạng: Embedder +   ║
║  Recovery + Generator + Discriminator, loss = joint (tự tái tạo +      ║
║  giám sát + adversarial).                                              ║
║                                                                       ║
║  TIẾN TRÌNH DỮ LIỆU:                                                  ║
║    lỗi thật → cắt window → z-score từng window → TimeGAN học → sinh    ║
║    N window giả → extract RAW feature → .npy → script 2.               ║
╚══════════════════════════════════════════════════════════════════════╝

CÁCH CHẠY (mặc định ghi results/synthetic_faults_ims_FIXED.npy):
    python scripts/03_train_timegan.py --dataset ims --limit 200 \
        --iterations 1500 --n-samples 3000 \
        --out results/synthetic_faults_ims_FIXED.npy

GHI CHÚ THỜI GIAN:
    --iterations 1500  ≈ ~10–30 phút tùy máy. --iterations 3000+  chậm hơn
    nhưng lỗi giả tốt hơn. Code dùng torch → tự dùng GPU (CUDA) khi có, nếu
    không thì chạy CPU (device đặt trong timegan_torch.py).

THÔNG SỐ CHÍNH:
    --seq-len    256   độ dài 1 window (số điểm rung ≈ 20 ms).
    --n-windows  2000  số window lỗi THẬT dùng để học (dữ liệu hiếm chính là đây).
    --iterations 1500  số bước luyện GAN.
    --n-samples  3000  số window lỗi GIẢ muốn sinh ra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as ds               # noqa: E402  (loader dữ liệu)
from src import features as ft           # noqa: E402  (tính 16 feature)
from src import pipeline as pipe         # noqa: E402  (tải + chia dữ liệu)
from src.timegan_torch import TimeGAN    # noqa: E402  (bản PyTorch của TimeGAN)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _fault_windows(signals: list[np.ndarray], win: int, n_target: int,
                   seed: int = 0) -> np.ndarray:
    """Cắt tín hiệu lỗi thật thành windows (win, 1) và chuẩn hoá z-score TỪNG WINDOW.

    MỤC ĐÍCH: TimeGAN học "HÌNH DẠNG" của lỗi (mẫu dao động), không học thang
    biên độ tuyệt đối. Nên trước khi học, mỗi window tách riêng được chuẩn hoá
    về trung bình 0, độ lệch 1 — khử đi sự khác biệt độ lớn giữa các sensor/test.

    ĐẦU RA: mảng (n_target, win, 1) — kích thước 3D (batch, chuỗi, kênh) đúng
    định dạng TimeGAN cần.
    """
    windows = []
    for sig in signals:
        arr = np.asarray(sig, dtype=np.float32).reshape(-1)   # tín hiệu 1D
        # Số window tối đa cắt ra từ tín hiệu dài này (stride = win//2, chồng lấp 50%)
        if len(arr) < win:
            continue   # tín hiệu quá ngắn → không cắt được window nào
        stride = win // 2
        n = (len(arr) - win) // stride + 1
        for i in range(n):
            row = arr[i * stride: i * stride + win]          # cắt đúng win điểm
            if row.std() < 1e-8:
                continue   # window gần như hằng → bỏ, tránh chia cho 0
            row = (row - row.mean()) / (row.std() + 1e-8)    # z-score từng window
            windows.append(row[:, None])                    # (win, 1)
    # Phải đủ số window theo yêu cầu, nếu không thì báo lỗi rõ ràng
    if len(windows) < n_target:
        raise SystemExit(f"chỉ có {len(windows)} window lỗi, cần ≥ --n-windows {n_target}. Tăng --limit.")
    # Chọn ngẫu nhiên (mặc định seed=0) đúng n_target window để TimeGAN không
    # phải học cả trăm nghìn window — vừa đủ đa dạng lại nhanh.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))[:n_target]
    return np.asarray(windows, dtype=np.float32)[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--limit", type=int, default=200,
                    help="IMS: số file/test tải (bản lỗi, cho generator)")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="độ dài window (số điểm rung ≈ 20ms)")
    ap.add_argument("--n-windows", type=int, default=2000,
                    help="số window lỗi thật dùng để học (dữ liệu hiếm chính là đây)")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=3000,
                    help="số window lỗi giả muốn sinh ra")
    ap.add_argument("--out", default=str(RESULTS / "synthetic_faults.npy"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    # ---------- 1. Lấy dữ liệu lỗi THẬT (90–100% cuối mỗi test IMS) ----------
    # load_normal_fault_signals trả về (runs_normal, runs_fault) — mỗi run là 1 test.
    # Ở đây chỉ quan tâm runs_fault (lỗi thật) → làm phẳng thành 1 danh sách tín hiệu.
    _, runs_fault = pipe.load_normal_fault_signals(args.dataset, args.limit)
    signs_fault = pipe.flatten_runs(runs_fault)
    # Cắt thành windows + z-score từng window → (n, seq_len, 1)
    X = _fault_windows(signs_fault, args.seq_len, args.n_windows)
    print(f"[1] windows lỗi thật: {X.shape} (khan hiếm — đây là chất liệu để sinh giả)")

    # ---------- 2. Train TimeGAN ----------
    # TimeGAN học phân bố của lỗi thật. Joint loss = 3 phần:
    #   (i)  reconstruction   : qua được tái tạo EMBEDDER→RECOVERY khớp không
    #   (ii) supervised        : chuỗi giả diễn biến giống chuỗi thật không
    #   (iii) adversarial (GAN): generator đánh lừa discriminator không
    # iterations = số vòng luyện. printfreq chỉ in loss thỉnh thoảng (đỡ loãng màn hình).
    print(f"[2] train TimeGAN (iter={args.iterations}, hidden={args.hidden}) ...")
    tg = TimeGAN(hidden_dim=args.hidden, num_layers=args.layers, batch_size=args.batch)
    losses = tg.train(X, iterations=args.iterations, printfreq=max(1, args.iterations // 5))

    # ---------- 3. Sinh lỗi giả + extract RAW feature ----------
    # sample(n) → n window "lỗi giả" (n, seq_len, 1)
    syn = tg.sample(args.n_samples)
    np.save(str(RESULTS / "synthetic_windows.npy"), syn)   # bản thô để trực quan sau này
    # Trích 16 feature (mean, rms, FFT...) như pipeline, KHÔNG chuẩn hoá
    # (RAW) — vì script 2 sẽ chuẩn hoá bằng scaler của TRAIN để không lộ test.
    feats = ft.raw_features(syn[:, :, 0], fs=pipe.FS)
    np.save(args.out, feats)
    print(f"[3] sinh {syn.shape[0]} window lỗi giả → {args.out} (RAW feature {feats.shape})")
    print("    (script 02 sẽ chuẩn hoá bằng scaler train khi dùng --gen npy)")

    # ---------- 4. Lưu báo cáo (loss cuối, thông số) ----------
    with open(str(RESULTS / "timegan_report.json"), "w") as f:
        json.dump({"synthetic_n": int(feats.shape[0]), "feature_dim": int(feats.shape[1]),
                   "seq_len": args.seq_len,
                   "losses": {k: float(v[-1]) for k, v in losses.items()}}, f, indent=2)
    print("[4] lưu results/timegan_report.json")
    print("\nBước tiếp theo:")
    print("  python scripts/02_compare_augmentation.py --dataset ims --gen npy \\")
    print("      --gen-npy results/synthetic_faults.npy")


if __name__ == "__main__":
    main()