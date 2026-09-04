#!/usr/bin/env python3
"""SCRIPT 07 — So sánh TimeGAN (generative) vs INTERPOLATE ở CÙNG mức khan hiếm.

Bổ sung cho script 06 (chỉ so heuristic/physics/optimized). Ở đây ta huấn luyện
TimeGAN trên ĐÚNG K cửa sổ lỗi thật (không phải toàn bộ vùng lỗi) để so CÔNG BẰNG
với bộ sinh `--gen optimized` (interpolate) — cả hai đều chỉ được cấp K lỗi thật.

Vì TimeGAN cần đủ mẫu để học phân bố nên khi K quá nhỏ (20) thường kém; đây chính là
điều muốn chứng minh: interpolation không cần "học" mẫu, chỉ trộn lỗi thật → ổn định
và hiệu quả hơn khi dữ liệu lỗi rất hiếm.

CÁCH CHẠY:
    python scripts/07_timegan_vs_interp.py --n-fault 50 20 --iterations 1500

KẾT QUẢ sẽ so 4 nguồn trên CÙNG một baseline (test cố định 8100 lỗi):
    - Trước (no aug)
    - +heuristic      (script 06)
    - +physics        (script 06)
    - +TimeGAN        (script này — train trên K lỗi thật)
    - +interpolate    (script 06 / `--gen optimized`)
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
from src.timegan_torch import TimeGAN               # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _score(clf, X, y) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y, clf.predict(X), average="binary",
                                                  zero_division=0)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}


def _fault_windows(runs_fault, n, rng, win):
    out = []
    for run in runs_fault:
        for s in run:
            out.append(ds.make_windows(s, win=win, stride=win))
    out = np.concatenate(out)
    return out[rng.choice(len(out), size=min(n, len(out)), replace=False)]


def _normal_windows(runs_normal, rng, win):
    out = []
    for run in runs_normal:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=win)
            if len(w):
                i_train = max(1, int(len(w) * pipe.NORMAL_TRAIN_END))
                out.append(w[:i_train])
    return np.concatenate(out)


def _heuristic(Xn_train, rng, n):
    idx = rng.integers(0, len(Xn_train), size=n)
    return Xn_train[idx] * 2.5 + rng.normal(0, 0.8, Xn_train[idx].shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--n-fault", type=int, nargs="+", default=[50, 20])
    ap.add_argument("--n-synth", type=int, default=800)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "timegan_vs_interp.json"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    rows = []
    for K in args.n_fault:
        d_rng = np.random.default_rng(args.seed + K)
        d = pipe.split_and_scale(signs_n, signs_f, args.win, args.stride, d_rng,
                                 max_fault_train=K)
        X_base, y_base, X_test, y_test = pipe.to_train_test_arrays(d)
        rng = np.random.default_rng(args.seed + 1000 + K)
        norm_raw = _normal_windows(signs_n, rng, args.win)
        fault_raw = _fault_windows(signs_f, K, rng, args.win)

        row = {"n_fault_train": K, "before": _score(*_fit_eval(X_base, y_base, X_test, y_test))}

        # ---- heuristic ----
        X_syn = _heuristic(d["Xn_train"], rng, args.n_synth)
        row["heuristic"] = _score(*_fit_eval(
            np.concatenate([X_base, X_syn]), np.concatenate([y_base, np.ones(len(X_syn))]),
            X_test, y_test))

        # ---- TimeGAN: train trên ĐÚNG K lỗi thật ----
        Xw = np.array([_z(w)[:, None] for w in fault_raw], dtype=np.float32)
        tg = TimeGAN(hidden_dim=24, num_layers=3, batch_size=32)
        tg.train(Xw, iterations=args.iterations, printfreq=max(1, args.iterations // 5))
        synw = tg.sample(args.n_synth)
        Xt = ft.zscore(ft.raw_features(synw[:, :, 0], fs=pipe.FS), d["scaler_mu"], d["scaler_sd"])
        row["timegan"] = _score(*_fit_eval(
            np.concatenate([X_base, Xt]), np.concatenate([y_base, np.ones(len(Xt))]),
            X_test, y_test))

        # ---- interpolate (bộ sinh tối ưu) ----
        gen = OptimizedFaultGenerator(fs=pipe.FS)
        gen.calibrate(fault_raw, norm_raw)
        syni = gen.generate_interpolated(fault_raw, args.n_synth, rng)
        Xi = ft.zscore(ft.raw_features(syni, fs=pipe.FS), d["scaler_mu"], d["scaler_sd"])
        row["interpolate"] = _score(*_fit_eval(
            np.concatenate([X_base, Xi]), np.concatenate([y_base, np.ones(len(Xi))]),
            X_test, y_test))
        rows.append(row)
        print(f"[K={K}] before={row['before']['recall']:.4f} "
              f"heuristic={row['heuristic']['recall']:.4f} "
              f"timegan={row['timegan']['recall']:.4f} "
              f"interp={row['interpolate']['recall']:.4f}", flush=True)

    print("\n=== BẢNG RECALL: trước / heuristic / TimeGAN / interpolate ===")
    for r_ in rows:
        print(f"K={r_['n_fault_train']:>4}: before={r_['before']['recall']:.4f} "
              f"heur={r_['heuristic']['recall']:.4f} "
              f"timegan={r_['timegan']['recall']:.4f} "
              f"interp={r_['interpolate']['recall']:.4f}")
    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "n_synth": args.n_synth,
                   "iterations": args.iterations, "rows": rows}, f, indent=2)
        print(f"Đã lưu: {args.out}")


def _z(w):
    w = w - w.mean()
    return w / (w.std() + 1e-8)


def _fit_eval(Xb, yb, Xt, yt):
    clf = RandomForestClassifier(n_estimators=200, random_state=0,
                                 class_weight="balanced").fit(Xb, yb)
    return clf, Xt, yt


if __name__ == "__main__":
    main()


def _fit_eval(Xb, yb, Xt, yt):
    clf = RandomForestClassifier(n_estimators=200, random_state=0,
                                 class_weight="balanced").fit(Xb, yb)
    return clf, Xt, yt
