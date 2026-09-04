#!/usr/bin/env python3
"""SCRIPT 14 — Kiểm chứng AUGMENTATION trên LỖI SỚM / INcipient (fault-test nhẹ).

VÌ SAO: harness hiện tại (script 06) nạp lỗi chỉ từ `files[i_fault:][-n_fault:]` =
vài chục file CUỐI (hỏng NẶNG). Giai đoạn 90%→95% chuỗi (lỗi vừa CHỚM, biên độ nhẹ,
sát normal suy giảm) KHÔNG hề được đưa vào test. Predictive-maintenance thực tế quan
tâm PHÁT HIỆN SỚM — ta phải chứng minh augment (interp_align+amp) vẫn có giá trị ngay
cả khi test chứa lỗi sớm (biên độ thấp, khó tách).

CÁCH LÀM (giữ nguyên anti-leakage, chỉ đổi CỬA SỔ FAULT-TEST):
  - Train/Normal test: giữ hệt như split_and_scale (normal khỏe 0-90%).
  - Fault-train   : ngẫu nhiên K từ vùng FAULT (90-100%) — như harness.
  - Fault-TEST    : ĐỔI sang vùng lỗi SỚM (p_incip_low..p_incip_high của chuỗi, mặc định
                    0.90→0.95) — nơi ổ bi mới bắt đầu suy giảm. Vì train/test chia theo
                    thời gian & khác vùng → không rò rỉ; và lỗi sớm KHÔNG trùng lỗi nặng.
  So sánh trên CÙNG fault-test sớm: baseline vs interp_align+amp.

Cách chạy:
  python scripts/14_incipient_ab.py --n-fault 10 20 50 --incip 0.90 0.95 --out results/incipient_ab.json
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


def _score(clf, X, y) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y, clf.predict(X), average="binary",
                                                  zero_division=0)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}


def raw_windows(sigs: list[np.ndarray], win: int, stride: int) -> np.ndarray:
    out = []
    for s in sigs:
        w = ds.make_windows(s, win=win, stride=stride)
        if len(w):
            out.append(w)
    return np.concatenate(out) if out else np.empty((0, win))


def load_faults_window(dataset: str, lo: float, hi: float, limit: int):
    """Nạp cửa sổ lỗi trong [lo,hi] chuỗi (theo file thật) — dùng cho test incipient."""
    runs = []
    for test_dir in sorted(ds.DATA_DIR.joinpath("NASA_IMS").iterdir()):
        if not test_dir.is_dir():
            continue
        files = list(ds.ims_files(test_dir))
        if len(files) < 3:
            continue
        n = len(files)
        i_lo, i_hi = int(n * lo), int(n * hi)
        seg = files[i_lo:i_hi] if i_hi > i_lo else files[-max(3, limit // 4):]
        runs.append(ds.load_ims_paths(seg))
    return [r for run in runs for r in run]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims"], default="ims")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n-fault", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--incip-lo", type=float, default=0.90)
    ap.add_argument("--incip-hi", type=float, default=0.95)
    ap.add_argument("--n-synth", type=int, default=3200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "incipient_ab.json"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    win, stride = 512, 256
    fs = pipe.FS

    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    all_normal = pipe.flatten_runs(signs_n)
    stride_eff = pipe._effective_stride(all_normal, win, stride)
    print(f"stride_eff={stride_eff}  incipient=[{args.incip_lo},{args.incip_hi}]")

    # ---- NORMAL test windows sớm (khỏe, 70-90% vùng normal) như split_and_scale ----
    def zone_normal_split():
        n_tr, n_te = [], []
        for run in signs_n:
            for sig in run:
                w = ds.make_windows(sig, win=win, stride=stride_eff)
                if len(w):
                    i_tr = int(len(w) * pipe.NORMAL_TRAIN_END)
                    i_te = int(len(w) * pipe.NORMAL_TEST_END)
                    n_tr.append(w[:i_tr]); n_te.append(w[i_tr:i_te])
        return (np.concatenate(n_tr) if n_tr else np.empty((0, win)),
                np.concatenate(n_te) if n_te else np.empty((0, win)))

    n_tr_raw, n_te_raw = zone_normal_split()
    n_tr_n = n_tr_raw  # normal train (thô)

    # ---- LỖI SỚM làm fault-TEST (không dùng để train) ----
    incip_signals = load_faults_window(args.dataset, args.incip_lo, args.incip_hi, args.limit)
    print(f"  incipient fault signals loaded: {len(incip_signals)}")
    f_inc_w = raw_windows(incip_signals, win, stride_eff)
    print(f"  incipient fault windows: {f_inc_w.shape}")

    # ---- NORMAL test features (z-score theo train-pool) ----
    # dùng scaler của 'd' cho chính xác; nhưng d cần fault_train. Ta dựng 1 lần.

    rows = []
    # Dựng 1 lần các vùng nền không phụ thuộc K:
    # features cho n_tr và n_te (z-score bằng mu/sd của train stack)
    Xn_tr_f = ft.raw_features(n_tr_n, fs=fs)
    Xn_te_f = ft.raw_features(n_te_raw, fs=fs)
    Xinc_f = ft.raw_features(f_inc_w, fs=fs)

    for K in args.n_fault:
        # --- fault-train: K cửa sổ từ pool FAULT NẶNG (đúng như harness) ---
        pool_f = raw_windows(pipe.flatten_runs(signs_f), win, stride_eff)
        idx = np.arange(len(pool_f))
        rng_f = np.random.default_rng(args.seed + K)
        rng_f.shuffle(idx)
        n_all = max(1, int(0.2 * len(pool_f)))
        f_tr_raw = pool_f[idx[:n_all]][:K]

        # --- z-score: fit trên train stack (n_tr + K fault) ---
        Xf_tr_f = ft.raw_features(f_tr_raw, fs=fs)
        mu, sd = ft.fit_zscore(np.concatenate([Xn_tr_f, Xf_tr_f]))
        Xb = np.concatenate([ft.zscore(Xn_tr_f, mu, sd), ft.zscore(Xf_tr_f, mu, sd)])
        yb = np.concatenate([np.zeros(len(Xn_tr_f)), np.ones(len(Xf_tr_f))])
        X_te = np.concatenate([ft.zscore(Xn_te_f, mu, sd), ft.zscore(Xinc_f, mu, sd)])
        y_te = np.concatenate([np.zeros(len(Xn_te_f)), np.ones(len(Xinc_f))])

        # baseline
        clf0 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(Xb, yb)
        r0 = _score(clf0, X_te, y_te)

        # interp_align + amp
        gen = OptimizedFaultGenerator(fs=fs)
        gen.calibrate(f_tr_raw, n_tr_n)
        g_rng = np.random.default_rng(args.seed + 1000 + K)
        synth = gen.generate_aligned_interp(f_tr_raw, args.n_synth, g_rng)
        aa = OptimizedFaultGenerator(fs=fs); aa.f_char = gen.f_char; aa._calibrated = True
        synth = aa._amp_rescale(synth, f_tr_raw, g_rng)
        X_syn = ft.zscore(ft.raw_features(synth, fs=fs), mu, sd)
        X_aug = np.concatenate([Xb, X_syn])
        y_aug = np.concatenate([yb, np.ones(len(X_syn))])
        clf1 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(X_aug, y_aug)
        r1 = _score(clf1, X_te, y_te)

        rows.append({"n_fault_train": K, "incipient": f"{args.incip_lo}-{args.incip_hi}",
                     "baseline": r0, "interp_align_amp": r1,
                     "n_test_fault_incipient": int((y_te == 1).sum()),
                     "n_train_normal": len(Xn_tr_f)})
        print(f"[K={K:>3}] baseline={r0['recall']:.4f} → interp_align={r1['recall']:.4f} "
              f"(Δ={r1['recall']-r0['recall']:+.4f}) | incipient_test={int((y_te==1).sum())}",
              flush=True)

    print("\n=== BẢNG: INCIPIENT FAULT (lỗi sớm) ===")
    print(f"{'K':>4} {'baseline':>9} {'interp':>9} {'Δ':>9}")
    for r_ in rows:
        print(f"{r_['n_fault_train']:>4} {r_['baseline']['recall']:>9.4f} "
              f"{r_['interp_align_amp']['recall']:>9.4f} "
              f"{r_['interp_align_amp']['recall']-r_['baseline']['recall']:>+9.4f}")

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "n_synth": args.n_synth,
                   "incipient": [args.incip_lo, args.incip_hi],
                   "stride_eff": stride_eff, "rows": rows}, f, indent=2)
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
