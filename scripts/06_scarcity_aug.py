#!/usr/bin/env python3
"""SCRIPT 6 — Thí nghiệm trọng yếu: "KHAN HIẾM NHÂN TẠO" (SPEC.md §5-A).

Đây là thí nghiệm mà mọi lever trước đó (script 02/03/05) CHƯA HỎI ĐÚNG câu của đề A1.

TẠI SAO PHẢI CÓ SCRIPT NÀY (đọc kỹ trước khi bỏ qua):
    Các script 02 (heuristic), 03 (TimeGAN), 05 (physics) — cùng script 02 chạy với
    `--limit` — đều kết luận "augmentation không giúp". NHƯNG ở đó, train NHẬN 20%
    lỗi THẬT (IMS ≈ 2k window), tức là model "TRƯỚC" chưa hề "đói" dữ liệu lỗi. Kết
    luận "augmentation vô ích" vì thế chỉ đúng khi baseline ĐÃ đủ dữ liệu lỗi, KHÔNG
    chứng minh gì về kịch bản đề bài thực sự quan tâm: "chỉ có vài mẫu lỗi".

    Script này MÔ PHỎNG ĐÚNG kịch bản đó: bóp số window lỗi thật trong train xuống
    còn rất ít (--n-fault [200, 100, 50, 20, 5]), giữ TEST cố định, rồi xem các
    nguồn lỗi giả (heuristic / physics BPFO / TimeGAN-npy) có "cứu" recall hay không.

QUY TRÌNH (mỗi mức khan hiếm):
    1. pipe.split_and_scale(..., max_fault_train=K) → test CỐ ĐỊNH, train chỉ K lỗi.
    2. Model "TRƯỚC"  = RF học normal + K lỗi thật.
    3. Model "SAU"    = RF học train trên + thêm lỗi giả (n_synth).
    4. Đánh giá CẢ HAI trên CÙNG test (normal khỏe chưa thấy + lỗi thật chưa thấy).
    5. Ghi bảng: mỗi mức K → recall trước / sau / Δ → thấy đường có hồi phục không.

CÁCH CHẠY:
    python scripts/06_scarcity_aug.py --dataset ims --gen physics
    python scripts/06_scarcity_aug.py --dataset ims --gen heuristic
    python scripts/06_scarcity_aug.py --dataset cwru --gen physics

KẾT QUẢ sẽ trả lời trực tiếp câu "sinh dữ liệu lỗi có cần cho bài khan hiếm không".
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
from src.generator import OptimizedFaultGenerator   # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

FAULT_TRAIN_DEFAULT = [200, 100, 50, 20, 5]   # mức "khan hiếm lỗi" cần thử (window)


def _score(clf, X: np.ndarray, y: np.ndarray) -> dict:
    """Bộ 4 chỉ số (precision/recall/f1/auc) cho 1 model trên 1 tập test."""
    p, r, f1, _ = precision_recall_fscore_support(y, clf.predict(X), average="binary",
                                                  zero_division=0)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}


def _heuristic_faults(Xn_train: np.ndarray, rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """Lỗi giả kiểu heuristic: lấy window NORMAL bóp biên độ 2.5x + nhiễu.

    Giữ ngắn gọn: chỉ là một trong các nguồn lỗi giả để đối chứng; chất lượng thật
    biểu hiện qua physics/TimeGAN. Xem thêm `_heuristic_faults` ở scripts/02.
    """
    idx = rng.integers(0, len(Xn_train), size=min(n_rows, len(Xn_train)))
    fake = Xn_train[idx].copy()
    fake *= 2.5
    return fake + rng.normal(0, 0.8, fake.shape)


def _physics_faults(norm_raw: np.ndarray, rng: np.random.Generator, n_rows: int,
                    fault_type: str, rotation_hz: float, geometry: dict, depth: float) -> np.ndarray:
    """Lỗi giả theo vật lý ổ bi (BPFO/BPFI/BSF/FTF) — NHẬN CỬA SỔ THỜI GIAN THÔ.

    FIX BUG (audit §3.4.6/bug): trước đây nhận `Xn_train` là FEATURE đã z-score (24 cột)
    rồi truyền 1 hàng (24 mẫu) vào `phys.generate_synthetic_fault` như tín hiệu 1D
    → sinh ảo tín hiệu dài 24 mẫu, không đúng cơ chế. Giờ nhận window THỜI GIAN THÔ
    (512 mẫu) làm nền, chèn đúng xung va đập BPFO/BPFI/BSF/FTF, rồi trích feature sau.
    """
    f_fault = {"bpfo": phys.bpfo, "bpfi": phys.bpfi, "bsf": phys.bsf, "ftf": phys.ftf}[
        fault_type](rotation_hz, geometry)
    idx = rng.integers(0, len(norm_raw), size=min(n_rows, len(norm_raw)))
    chunks = []
    for i, w in enumerate(norm_raw[idx]):
        jitter = f_fault * (1 + rng.uniform(-0.02, 0.02))
        chunks.append(phys.generate_synthetic_fault(w.astype(np.float64), jitter,
                                                     pipe.FS, depth=depth, seed=i))
    return np.asarray(chunks)


def _raw_train_windows(signals_n: list[list[np.ndarray]], signals_f: list[list[np.ndarray]],
                       win: int, stride: int, n_fault: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Lấy cửa sổ THỜI GIAN THÔ (chưa chuẩn hoá) cho vùng TRAIN theo đúng anti-leakage.

    Trả về (normal_train_win, fault_train_win):
      - normal: cửa sổ từ vùng KHỎE (0→NORMAL_TRAIN_END mỗi run) — dùng làm nền bơm lỗi.
      - fault : K cửa sổ lỗi THẬT (nguồn khan hiếm) — dùng để `calibrate` bộ sinh.

    FIX LEAK (audit): split trong pipeline là WINDOW-LEVEL cho fault (rng.shuffle(Xf)
    rồi cắt 20% train / 80% test). Trước đây hàm này lấy `fault_raw` từ TOÀN BỘ pool
    lỗi (gồm cả 80% window THUỘC TEST) → bộ sinh được nuôi từ chính lỗi test (thổi
    phồng recall). Giờ tái tạo ĐÚNG `d_rng` (seed+K) để shuffle HOÁN VỊ index rồi lấy
    phần TRAIN (20% đầu), khớp hệt `Xf_train` trong `split_and_scale`. Ngoài ra phải
    dùng `stride_eff` (không phải `args.stride` gốc) để khớp pool của split.
    """
    norm = []
    for run in signals_n:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride)
            if len(w):
                i_train = max(1, int(len(w) * pipe.NORMAL_TRAIN_END))
                norm.append(w[:i_train])
    norm = np.concatenate(norm) if norm else np.empty((0, win))

    flt = []
    for run in signals_f:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride)
            if len(w):
                flt.append(w)
    flt = np.concatenate(flt) if flt else np.empty((0, win))
    # Khớp hệt phân chia fault-train của split_and_scale (window-level, rng = seed+K):
    n_all = len(flt)
    n_f_train_all = max(1, int(0.2 * n_all))           # 20% pool — đúng như pipeline
    idx = np.arange(n_all)
    rng.shuffle(idx)
    flt_train = flt[idx[:n_f_train_all]]               # Phần 20% ĐẦU dành cho train
    # Sau đó mới cắt xuống n_fault (khan hiếm): phần vượt bị bỏ, giống split_and_scale.
    flt_train = flt_train[:n_fault]
    return norm, flt_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--gen", choices=["heuristic", "physics", "optimized", "aligned", "amp_align", "interp_align", "spectral_mixup", "physics_mixup", "ddpm", "npy"], default="optimized",
                    help="nguồn lỗi giả dùng để tăng cường (optimized = interpolate | aligned = ECDF/KS align | amp_align = interpolate+chuẩn amplitude | interp_align = interpolate+căn pha cross-correlation | spectral_mixup = trộn magnitude phổ+bơm hài | physics_mixup = interp_align trộn physics | ddpm = diffusion)")
    ap.add_argument("--amp", type=float, default=0.0,
                    help=">0 thì sau khi sinh sẽ chuẩn biên độ (std về phân bố lỗi thật như amp_align), áp cho interp_align/spectral_mixup (0 = tắt)")
    ap.add_argument("--amp-mode", choices=["q10q90", "full"], default="q10q90",
                    help="q10q90 = std về [q10,q90] (mặc định/cũ) | full = std resample từ TOÀN BỘ phân phối lỗi thật")
    ap.add_argument("--k-mix", type=int, default=6,
                    help="số cửa sổ lỗi thật trộn trong interp_align (2 = hành vi cũ; 6 = mặc định mới, chọn qua A/B SPEC §3.4.8)")
    ap.add_argument("--phys-frac", type=float, default=0.30,
                    help="tỉ lệ cửa sổ lấy từ physics-injection trong physics_mixup (0..1)")
    ap.add_argument("--harm-inject", type=float, default=0.0,
                    help="cường độ bơm hài vật lý tại k*f_char cho spectral_mixup (0..1, 0 = tắt)")
    ap.add_argument("--gen-npy", type=str, default=str(RESULTS / "synthetic_faults.npy"),
                    help="file lỗi giả từ TimeGAN (chỉ dùng khi --gen npy)")
    ap.add_argument("--limit", type=int, default=300, help="số file/test tải")
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--n-fault", type=int, nargs="+", default=FAULT_TRAIN_DEFAULT,
                    help="các mức số window lỗi thật cho train (khan hiếm)")
    ap.add_argument("--n-synth", type=int, default=600,
                    help="số lỗi giả thêm vào train 'SAU' (cố định cho mọi mức)")
    ap.add_argument("--ddpm-epochs", type=int, default=300,
                    help="số epoch train DDPM (chỉ dùng khi --gen ddpm)")
    ap.add_argument("--fault-type", choices=["bpfo", "bpfi", "bsf", "ftf"], default="bpfo")
    ap.add_argument("--rotation-hz", type=float, default=30.0)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", help="in chi tiết tham số học được")
    ap.add_argument("--out", default=str(RESULTS / "scarcity_aug.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    RESULTS.mkdir(exist_ok=True)
    geometry = phys.bearing_geometry()

    # Tải tín hiệu MỘT LẦN; mỗi mức khan hiếm chỉ đổi `max_fault_train`.
    print(f"=== KHAN HIẾM LỖI NHÂN TẠO — {args.dataset} (aug={args.gen}) ===")
    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    all_normal = pipe.flatten_runs(signs_n)
    stride_eff = pipe._effective_stride(all_normal, args.win, args.stride)
    if stride_eff != args.stride and args.verbose:
        print(f"  [note] stride {args.stride} → nới {stride_eff} (giữ số window)")

    rows = []
    for K in args.n_fault:
        # Mỗi mức khan hiếm dùng MỘT rng mới (seed phụ thuộc K) → `before` (train/test
        # split + scaler) giống hệt nhau cho MỌI --gen → so sánh CÔNG BẰNG: chỉ phần
        # "lỗi giả" khác nhau, còn baseline thì tái lập được.
        d_rng = np.random.default_rng(args.seed + K)
        d = pipe.split_and_scale(signs_n, signs_f, args.win, args.stride, d_rng,
                                 max_fault_train=K)
        X_base, y_base, X_test, y_test = pipe.to_train_test_arrays(d)
        n_fault = len(d["Xf_train"])
        Xn_train = d["Xn_train"]
        gen_rng = np.random.default_rng(args.seed + 1000 + K)   # riêng cho sinh lỗi giả
        # rng tái tạo phân chia FAULT-TRAIN (window-level) của split_and_scale:
        # chính là `d_rng` mới (seed+K) — vì split dùng d_rng duy nhất `rng.shuffle(Xf)`,
        # nên nếu shuffle index bằng default_rng(seed+K) ta được CÙNG hoán vị.
        fault_split_rng = np.random.default_rng(args.seed + K)

        # ---- lỗi giả: sinh trên train-normal (anti-leakage), chuẩn hoá bằng scaler train ----
        if args.gen == "heuristic":
            X_syn = _heuristic_faults(Xn_train, gen_rng, args.n_synth)
            y_syn = np.ones(len(X_syn))
        elif args.gen == "physics":
            raw = _physics_faults(Xn_train, gen_rng, args.n_synth, args.fault_type,
                                  args.rotation_hz, geometry, args.depth)
            X_syn = ft.zscore(raw, d["scaler_mu"], d["scaler_sd"])
            y_syn = np.ones(len(X_syn))
        elif args.gen == "optimized":
            # Bộ sinh TỐI ƯU (kết quả SPEC §3.4): TỐT NHẤT khi lỗi hiếm là TRỘN
            # (interpolate) giữa các cửa sổ lỗi THẬT + jitter — bám sát manifold lỗi
            # thật thay vì bơm xung vào nền normal. Cần cửa sổ thời gian THÔ.
            norm_raw, fault_raw = _raw_train_windows(signs_n, signs_f, args.win,
                                                     stride_eff, n_fault, fault_split_rng)
            gen = OptimizedFaultGenerator(fs=pipe.FS)
            gen.calibrate(fault_raw, norm_raw)          # học f_char/depth (để ghi chú)
            synth_sig = gen.generate_interpolated(fault_raw, args.n_synth, gen_rng)
            raw = ft.raw_features(synth_sig, fs=pipe.FS)   # lỗi giả (thời gian) → feature
            X_syn = ft.zscore(raw, d["scaler_mu"], d["scaler_sd"])
            y_syn = np.ones(len(X_syn))
            if args.verbose:
                print(f"  [optimized] interpolate {len(fault_raw)} lỗi thật → "
                      f"{len(X_syn)} bản (f_char={gen.f_char:.1f}Hz depth={gen.depth:.2f})")
        elif args.gen == "aligned":
            # ECDF/KS DISTRIBUTION ALIGNMENT (SPEC §3.4.2): physics + interpolate làm
            # POOL, rồi lọc các bản có descriptor nằm trong dải phân vị lỗi THẬT
            # (amplitude/sideband/kurtosis hợp lý) → "sim-to-real gap" được khép.
            norm_raw, fault_raw = _raw_train_windows(signs_n, signs_f, args.win,
                                                     stride_eff, n_fault, fault_split_rng)
            gen = OptimizedFaultGenerator(fs=pipe.FS)
            gen.calibrate(fault_raw, norm_raw)
            synth_sig = gen.generate_aligned(fault_raw, norm_raw, args.n_synth, gen_rng)
            raw = ft.raw_features(synth_sig, fs=pipe.FS)
            X_syn = ft.zscore(raw, d["scaler_mu"], d["scaler_sd"])
            y_syn = np.ones(len(X_syn))
            if args.verbose:
                print(f"  [aligned] ECDF/KS align {len(fault_raw)} lỗi thật → "
                      f"{len(X_syn)} bản (f_char={gen.f_char:.1f}Hz depth={gen.depth:.2f})")
        elif args.gen == "amp_align":
            # interpolate + chuẩn biên độ theo phân bố std lỗi thật (SPEC §3.4.3) —
            # bù khoảng "dịu" của interpolate (std/env_energy thấp hơn thật ~0.7-0.85x).
            norm_raw, fault_raw = _raw_train_windows(signs_n, signs_f, args.win,
                                                     stride_eff, n_fault, fault_split_rng)
            gen = OptimizedFaultGenerator(fs=pipe.FS)
            gen.calibrate(fault_raw, norm_raw)
            synth_sig = gen.generate_amp_aligned(fault_raw, args.n_synth, gen_rng)
            raw = ft.raw_features(synth_sig, fs=pipe.FS)
            X_syn = ft.zscore(raw, d["scaler_mu"], d["scaler_sd"])
            y_syn = np.ones(len(X_syn))
            if args.verbose:
                print(f"  [amp_align] interpolate+chuẩn amplitude {len(fault_raw)} lỗi thật → "
                      f"{len(X_syn)} bản (f_char={gen.f_char:.1f}Hz)")
        elif args.gen in ("interp_align", "spectral_mixup", "physics_mixup"):
            # PHASE-AWARE MIXUP (SPEC §3.4.5): khử destructive interference.
            #  - interp_align   : interpolate nhưng căn pha 2 cửa sổ bằng cross-correlation
            #                     trước khi trộn → sóng mang không triệt tiêu (std sát thật).
            #  - spectral_mixup : trộn MAGNITUDE phổ (độc lập phase) + bơm hài vật lý; bộ sinh
            #                     "phantasma" phiên bản reviewer. Sau đó cũng qua feature.
            #  - physics_mixup  : interp_align nhưng trộn cả lỗi physics vào pool (A/B Mục 4).
            # Cả ba đều nhận cửa sổ thời gian thô như interpolate.
            norm_raw, fault_raw = _raw_train_windows(signs_n, signs_f, args.win,
                                                     stride_eff, n_fault, fault_split_rng)
            gen = OptimizedFaultGenerator(fs=pipe.FS)
            gen.calibrate(fault_raw, norm_raw)
            if args.gen == "interp_align":
                synth_sig = gen.generate_aligned_interp(fault_raw, args.n_synth, gen_rng,
                                                        k_mix=args.k_mix)
            elif args.gen == "physics_mixup":
                synth_sig = gen.generate_physics_mixup(fault_raw, norm_raw, args.n_synth,
                                                       gen_rng, phys_frac=args.phys_frac)
            else:
                synth_sig = gen.generate_spectral_mixup(fault_raw, args.n_synth, gen_rng,
                                                        harmonic_inject=args.harm_inject)
            if args.amp > 0:   # post-scale biên độ chuẩn theo phân bố std lỗi thật
                approx_fc = gen.f_char
                aa = OptimizedFaultGenerator(fs=pipe.FS)
                aa.f_char = approx_fc
                aa._calibrated = True
                synth_sig = aa._amp_rescale(synth_sig, fault_raw, gen_rng,
                                            full_dist=(args.amp_mode == "full"))
            raw = ft.raw_features(synth_sig, fs=pipe.FS)
            X_syn = ft.zscore(raw, d["scaler_mu"], d["scaler_sd"])
            y_syn = np.ones(len(X_syn))
            if args.verbose:
                print(f"  [{args.gen}] {len(fault_raw)} lỗi thật → {len(X_syn)} bản "
                      f"(f_char={gen.f_char:.1f}Hz, amp={args.amp}/{args.amp_mode}, "
                      f"k_mix={args.k_mix}, phys={args.phys_frac})")
        elif args.gen == "ddpm":
            # Diffusion (denoising) sinh lỗi giả từ lỗi THẬT — SOTA generative. Cần
            # cửa sổ thời gian thô để train mô hình rồi sinh, sau đó trích feature.
            norm_raw, fault_raw = _raw_train_windows(signs_n, signs_f, args.win,
                                                     stride_eff, n_fault, fault_split_rng)
            from src.ddpm import LightDDPM
            dd = LightDDPM(seq_len=args.win, T=200, seed=args.seed + K)
            dd.train(fault_raw, epochs=args.ddpm_epochs, batch=64, log_every=0)
            synth_sig = dd.sample(args.n_synth)
            raw = ft.raw_features(synth_sig, fs=pipe.FS)
            X_syn = ft.zscore(raw, d["scaler_mu"], d["scaler_sd"])
            y_syn = np.ones(len(X_syn))
            if args.verbose:
                print(f"  [ddpm] sinh {len(X_syn)} bản từ {len(fault_raw)} lỗi thật")
        else:  # npy — đọc lỗi giả từ TimeGAN (script 03)
            p = Path(args.gen_npy)
            if not p.exists():
                raise SystemExit(f"Thiếu file {p}. Chạy script 03 trước (--gen npy).")
            raw = np.load(p)
            if raw.shape[1] != len(ft.FEATURE_NAMES):
                raise SystemExit(f"file npy có {raw.shape[1]} cột, cần {len(ft.FEATURE_NAMES)}.")
            X_syn = ft.zscore(raw, d["scaler_mu"], d["scaler_sd"])
            y_syn = np.ones(len(X_syn))

        X_aug = np.concatenate([X_base, X_syn])
        y_aug = np.concatenate([y_base, y_syn])

        clf_before = RandomForestClassifier(n_estimators=200, random_state=0,
                                            class_weight="balanced").fit(X_base, y_base)
        clf_after = RandomForestClassifier(n_estimators=200, random_state=0,
                                           class_weight="balanced").fit(X_aug, y_aug)

        r_before = _score(clf_before, X_test, y_test)
        r_after = _score(clf_after, X_test, y_test)
        rows.append({
            "n_fault_train": K, "n_train_normal": int(len(Xn_train)),
            "n_test_fault": int((y_test == 1).sum()),
            "before": r_before, "after": r_after,
            "d_recall": r_after["recall"] - r_before["recall"],
            "d_auc": r_after["auc"] - r_before["auc"],
        })
        print(f"[K={K:>4}] train_fault={d['n_fault_train']:>5} "
              f"before recall={r_before['recall']:.4f} → after={r_after['recall']:.4f} "
              f"(Δ={r_after['recall'] - r_before['recall']:+.4f}) | "
              f"test_fault={rows[-1]['n_test_fault']}")

    # ---- bảng tổng hợp ----
    print("\n=== BẢNG TỔNG: recall trước / sau / Δ theo mức khan hiếm ===")
    print(f"{'train_fault':>12} {'test_fault':>11} {'recall_bf':>10} {'recall_af':>10} {'Δrecall':>9} {'auc_af':>8}")
    for r_ in rows:
        print(f"{r_['n_fault_train']:>12} {r_['n_test_fault']:>11} "
              f"{r_['before']['recall']:>10.4f} {r_['after']['recall']:>10.4f} "
              f"{r_['d_recall']:>+9.4f} {r_['after']['auc']:>8.4f}")

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "gen": args.gen,
                   "fault_type": args.fault_type, "n_synth": args.n_synth,
                   "rows": rows}, f, indent=2)
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
