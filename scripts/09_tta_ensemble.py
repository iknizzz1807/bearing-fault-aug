#!/usr/bin/env python3
"""SCRIPT 09 — TTA (test-time augmentation) + ENSEMBLE cho bài khan hiếm dữ liệu lỗi.

PHASE 1 & 2 — những cải thiện phía SUY LUẬN (research 2026: CCC-LSGAN, VAE-WGAN+CBAM,
TTLN đều dùng TTA + ensemble để tăng điểm, KHÔNG cần sinh thêm dữ liệu). Rẻ, và
KHÔNG chạm vào holdout (đúng §6: "test là vùng đất thiêng" — chỉ làm ở suy luận).

A) TTA (test-time augmentation):
   Mỗi cửa sổ test x → K "view" x_k (nhiễu nhẹ + lệch pha + scale nhỏ) → mỗi view
   qua RF → K xác suất → TRUNG BÌNH → xác suất ổn định hơn (giảm phương sai do pha).
   LÀM Ở SIGNAL SPACE: biến đổi tín hiệu thời gian rồi trích LẠI feature 24D —
   không trộn feature trực tiếp (mất ý nghĩa vật lý).

B) ENSEMBLE: nhiều RF (đa seed) trên cùng expanded train → trung bình xác suất
   (giảm phương sai của thuật toán RF). Kết hợp với TTA.

HARNESS (sạch, tái lập): `build_split` tự dựng ĐÚNG logic `pipe.split_and_scale`
(cùng `_effective_stride`, 3-zone normal, 20/80 fault same-permutation, scaler
fit-on-train) + TRẢ VỀ RAW WINDOW của test. Baseline (K=1) và TTA dùng CHÍNH XÁC
cùng test → Δ recall thuần từ TTA, không nhiễu.

CÁCH CHẠY:
    python scripts/09_tta_ensemble.py --n-fault 50 20 --n-synth 3200 --tta 8
    python scripts/09_tta_ensemble.py --n-fault 50 --tta 4 --ensemble 5
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
from src import pipeline as pipe    # noqa: E402
from src.generator import OptimizedFaultGenerator   # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------- #
#  HELPERS
# ---------------------------------------------------------------------- #
def _score(clf, X, y) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y, clf.predict(X), average="binary",
                                                  zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}


def _fit_rf(X, y, seed=0):
    return RandomForestClassifier(n_estimators=200, random_state=seed,
                                  class_weight="balanced").fit(X, y)


# ---------------------------------------------------------------------- #
#  build_split — tái dựng split của pipeline nhưng TRẢ VỀ RAW WINDOW TEST
# ---------------------------------------------------------------------- #
def _windows_per_run(runs, win, stride_eff):
    """Giống `pipe._features_per_run` nhưng trả RAW WINDOW (mỗi run = 1 ma trận)."""
    out, acc = [], []
    for run in runs:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride_eff)
            if len(w):
                acc.append(w)
        if acc:
            out.append(np.concatenate(acc))
            acc = []
    return out


def build_split(signs_n, signs_f, win, stride, rng, max_fault_train):
    """Tái dựng `pipe.split_and_scale` + trả về (X_base, y_base, X_test, y_test,
    test_windows_norm, test_windows_fault, mu, sd).

    ĐIỂM MẤU CHỐT để khớp HỆT `split_and_scale`: pipeline làm `rng.shuffle(Xf)`
    (in-place trên ma trận feature). Vì `Generator.shuffle` chỉ phụ thuộc SỐ HÀNG N,
    ta thay bằng `idx = arange(N); rng.shuffle(idx)` rồi index CẢ features LẪN raw
    windows theo `idx` → nhận CÙNG thứ tự (test/train giống hệt pipeline) MÀ vẫn có
    raw window. Scalerr fit-on-train cũng giữ nguyên.
    """
    all_normal = pipe.flatten_runs(signs_n)
    stride_eff = pipe._effective_stride(all_normal, win, stride)

    Xn_run_feats = pipe._features_per_run(signs_n, win, stride_eff)
    Xn_run_windows = _windows_per_run(signs_n, win, stride_eff)

    Xf = pipe.windows_to_features(pipe.flatten_runs(signs_f), win, stride_eff)
    Xf_windows = np.concatenate([ds.make_windows(sig, win=win, stride=stride_eff)
                                 for run in signs_f for sig in run])

    # normal: 3-zone (features + windows cùng slice theo index)
    Xn_train_z, Xn_test_z, Xn_test_w_z, Xn_grey_z = [], [], [], []
    for Xn_run, wr in zip(Xn_run_feats, Xn_run_windows):
        i_tr = int(len(Xn_run) * pipe.NORMAL_TRAIN_END)
        i_te = int(len(Xn_run) * pipe.NORMAL_TEST_END)
        Xn_train_z.append(Xn_run[:i_tr])
        Xn_test_z.append(Xn_run[i_tr:i_te])
        Xn_test_w_z.append(wr[i_tr:i_te])
        Xn_grey_z.append(Xn_run[i_te:])
    Xn_train_raw = np.concatenate(Xn_train_z) if Xn_train_z else np.empty((0, Xf.shape[1]))
    Xn_test_raw = np.concatenate(Xn_test_z) if Xn_test_z else np.empty((0, Xf.shape[1]))
    Xn_test_windows = np.concatenate(Xn_test_w_z) if Xn_test_w_z else np.empty((0, win))

    # fault: shuffle theo CHỈ SỐ (khớp hệt `rng.shuffle(Xf)` vì chỉ phụ thuộc N)
    idx = np.arange(Xf.shape[0])
    rng.shuffle(idx)
    Xf = Xf[idx]
    Xf_windows = Xf_windows[idx]
    n_f_test = int((1 - 0.2) * len(Xf))
    Xf_test_raw = Xf[-n_f_test:]
    Xf_test_windows = Xf_windows[-n_f_test:]
    n_f_train = max(1, int(0.2 * len(Xf)))
    if max_fault_train is not None:
        n_f_train = min(n_f_train, int(max_fault_train))
    Xf_train_raw = Xf[:n_f_train]

    mu, sd = ft.fit_zscore(np.concatenate([Xn_train_raw, Xf_train_raw]))
    X_base = np.concatenate([ft.zscore(Xn_train_raw, mu, sd),
                             ft.zscore(Xf_train_raw, mu, sd)])
    y_base = np.concatenate([np.zeros(len(Xn_train_raw)), np.ones(len(Xf_train_raw))])
    X_test = np.concatenate([ft.zscore(Xn_test_raw, mu, sd), ft.zscore(Xf_test_raw, mu, sd)])
    y_test = np.concatenate([np.zeros(len(Xn_test_raw)), np.ones(len(Xf_test_raw))])
    return (X_base, y_base, X_test, y_test, Xn_test_windows, Xf_test_windows, mu, sd)


# ---------------------------------------------------------------------- #
#  TTA — sinh K view signal-space rồi trích feature lại, dùng scaler TRAIN
# ---------------------------------------------------------------------- #
def tta_views(windows2d, rng, n_views, noise=0.05, jitter=4, scale_lo=0.9, scale_hi=1.1):
    """Với mỗi cửa sổ: n_views bản (nhiễu + lệch pha + scale nhẹ). Trả (n, K, win)."""
    n, L = windows2d.shape
    out = np.empty((n, n_views, L))
    for i in range(n):
        s = windows2d[i]
        std = s.std() + 1e-8
        for k in range(n_views):
            sc = rng.uniform(scale_lo, scale_hi)
            v = s * sc + rng.normal(0, noise * std, L)
            sh = rng.integers(-jitter, jitter + 1)
            out[i, k] = np.roll(v, sh) if sh else v
    return out


def tta_predict_proba(clf, test_windows, mu, sd, rng, n_views):
    """RF infer trên K view/cửa sổ → trung bình xác suất (n,) cho class=1."""
    n, K, L = test_windows.shape
    probs = np.empty((n, K))
    flat = test_windows.reshape(-1, L)
    # trích feature cho TẤT CẢ view rồi z-score bằng scaler train (chống lộ)
    feats = ft.raw_features(flat, fs=pipe.FS)
    feats = ft.zscore(feats, mu, sd)
    p = clf.predict_proba(feats)[:, 1].reshape(n, K)
    return p.mean(axis=1)


def _decision(prob1, thresh=0.5):
    return (prob1 >= thresh).astype(int)


def _score_probs(y, prob1, thresh=0.5) -> dict:
    pred = _decision(prob1, thresh)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "auc": float(roc_auc_score(y, prob1))}


# ---------------------------------------------------------------------- #
#  MAIN
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--n-fault", type=int, nargs="+", default=[50, 20])
    ap.add_argument("--n-synth", type=int, default=3200)
    ap.add_argument("--tta", type=int, default=8, help="số view/cửa sổ test (1 = không TTA)")
    ap.add_argument("--ensemble", type=int, default=1, help="số RF đa seed (1 = không ensemble)")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--jitter", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "tta_ensemble.json"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    print(f"=== TTA+Ensemble ({args.dataset}) | tta={args.tta} ens={args.ensemble} "
          f"n_synth={args.n_synth} ===")

    rows = []
    for K in args.n_fault:
        d_rng = np.random.default_rng(args.seed + K)
        (X_base, y_base, X_test, y_test,
         wn_test, wf_test, mu, sd) = build_split(signs_n, signs_f, args.win, args.stride,
                                                  d_rng, max_fault_train=K)

        # --- expanded train: interpolate lỗi thật (bộ sinh đang giữ, SPEC §3.4) ---
        fac = K
        stride_eff = pipe._effective_stride(pipe.flatten_runs(signs_n), args.win, args.stride)
        norm_raw, fault_raw = _train_raw_windows(signs_n, signs_f, args.win, stride_eff,
                                                 fac, np.random.default_rng(args.seed + K))
        gen = OptimizedFaultGenerator(fs=pipe.FS)
        gen.calibrate(fault_raw, norm_raw)
        synth = gen.generate_interpolated(fault_raw, args.n_synth,
                                          np.random.default_rng(args.seed + 2000 + K))
        feats_syn = ft.zscore(ft.raw_features(synth, fs=pipe.FS), mu, sd)
        X_aug = np.concatenate([X_base, feats_syn])
        y_aug = np.concatenate([y_base, np.ones(len(feats_syn))])

        # test windows (normal + fault) làm TTA trong signal space
        test_windows = np.concatenate([wn_test, wf_test])
        test_prob1 = np.empty(len(X_test))

        # vòng ensemble (đa seed RF) với TTA (n_views) — chỉ phía suy luận
        for e in range(args.ensemble):
            clf = _fit_rf(X_aug, y_aug, seed=e)
            if args.tta > 1:
                views = tta_views(test_windows, np.random.default_rng(args.seed + 3000 + K + e),
                                  args.tta, args.noise, args.jitter)
                p1 = tta_predict_proba(clf, views, mu, sd,
                                       np.random.default_rng(args.seed + 4000 + K + e), args.tta)
            else:
                feats = ft.zscore(ft.raw_features(test_windows, fs=pipe.FS), mu, sd)
                p1 = clf.predict_proba(feats)[:, 1]
            test_prob1 = test_prob1 + p1 if e else p1
        test_prob1 /= args.ensemble

        # baseline (không TTA/ensemble) = cùng train, predict trực tiếp
        clf0 = _fit_rf(X_aug, y_aug, seed=0)
        feats0 = ft.zscore(ft.raw_features(test_windows, fs=pipe.FS), mu, sd)
        s_base = _score_probs(y_test, clf0.predict_proba(feats0)[:, 1])
        s_tta = _score_probs(y_test, test_prob1)

        rows.append({
            "n_fault_train": K,
            "n_test_fault": int((y_test == 1).sum()),
            "before_interp": s_base,           # interpolate (n_synth) không TTA
            "tta_ensemble": s_tta,             # interpolate + TTA(+ensemble)
            "d_recall": round(s_tta["recall"] - s_base["recall"], 4),
        })
        print(f"[K={K:>3}] interp recall={s_base['recall']:.4f} → "
              f"+TTA({args.tta}/ens{args.ensemble})={s_tta['recall']:.4f} "
              f"(Δ={s_tta['recall'] - s_base['recall']:+.4f})", flush=True)

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "win": args.win, "n_synth": args.n_synth,
                   "tta": args.tta, "ensemble": args.ensemble, "rows": rows}, f, indent=2)
    print(f"Đã lưu: {args.out}")


def _train_raw_windows(signs_n, signs_f, win, stride, n_fault, rng):
    """Cửa sổ THÔ vùng TRAIN (normal khỏe + K lỗi thật) — theo đúng 3-zone anti-leak.

    FIX LEAK (giống scripts/06::_raw_train_windows): split fault trong pipeline là
    WINDOW-LEVEL (`rng.shuffle(Xf)` rồi cắt 20% train / 80% test) và dùng `stride_eff`,
    KHÔNG phải `args.stride` gốc. Trước đây `_train_raw_windows` lấy `fault` từ TOÀN BỘ
    pool (gồm cả 80% window THUỘC TEST) → bộ sinh được nuôi từ chính lỗi test.
    Giờ: dùng `stride_eff` + tái tạo đúng `d_rng` (seed+K) shuffle rồi lấy phần TRAIN.
    """
    norm = []
    for run in signs_n:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride)
            if len(w):
                i_tr = max(1, int(len(w) * pipe.NORMAL_TRAIN_END))
                norm.append(w[:i_tr])
    norm = np.concatenate(norm) if norm else np.empty((0, win))
    fault = []
    for run in signs_f:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride)
            if len(w):
                fault.append(w)
    fault = np.concatenate(fault) if fault else np.empty((0, win))
    # Khớp hệt phân chia fault-train của split_and_scale (window-level, rng = seed+K):
    n_all = len(fault)
    n_f_train_all = max(1, int(0.2 * n_all))
    idx = np.arange(n_all)
    rng.shuffle(idx)
    fault = fault[idx[:n_f_train_all]][:n_fault]
    return norm, fault


if __name__ == "__main__":
    main()
