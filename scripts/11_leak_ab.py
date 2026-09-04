"""11_leak_ab.py — A/B kiểm chứng DATA LEAKAGE trong _raw_train_windows (script 06).

PHÁT HIỆN (từ audit): `_raw_train_windows` lấy cửa sổ lỗi THẬT (nguồn sinh lỗi giả)
từ TOÀN BỘ pool lỗi (gồm cả 80% window thuộc TEST), vì split trong pipeline là
window-level (`rng.shuffle(Xf)` rồi cắt 20% train / 80% test). Ngoài ra nó dùng
`args.stride` (256) trong khi split dùng `stride_eff` (≈2449) — càng lệch.

Script này tái tạo ĐÚNG `Xf_train_raw` (20 cửa sổ lỗi THẬT dành cho train, khớp hệt
phân chia của `split_and_scale`) rồi so recall:
  - [BUGGY ] nguồn sinh = fault_raw từ pool toàn bộ, stride 256 (như script 06 hiện tại)
  - [FIXED ] nguồn sinh = fault_raw = đúng 20 cửa sổ lỗi TRAIN (stride_eff, khớp Xf_train)
Mọi thứ khác giữ y hệt (scaler fit-train, RF class_weight balanced, same test).

Chạy:
    python scripts/11_leak_ab.py --n-fault 20 --seed 42 --n-synth 3200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pipeline as pipe, data as ds, features as ft
from src.generator import OptimizedFaultGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

RESULTS = Path(__file__).resolve().parent.parent / "results"


def raw_windows(sigs, win, stride):
    out = []
    for s in sigs:
        w = ds.make_windows(s, win=win, stride=stride)
        if len(w):
            out.append(w)
    return np.concatenate(out) if out else np.empty((0, win))


def fault_rank_train(fault_raw_all, n_fault, seed, K):
    """Trả về đúng n_fault cửa sổ lỗi TRAIN (khớp Xf_train của split_and_scale).

    Trong `split_and_scale`, Xf = windows_to_features(flatten(signs_f), win, stride_eff)
    rồi `rng.shuffle(Xf)` (in-place trên features) rồi cắt [:n_f_train].
    Để lấy các cửa sổ thời gian THÔ tương ứng, ta shuffle đúng index bằng cùng rng
    (Fisher–Yates trên axis 0 của mảng 1D = cùng hoán vị như trên 2D) rồi lấy [:n_fault]
    từ pool raw. `_features_per_run`/`windows_to_features` giữ đúng thứ tự thời gian
    (chỉ ghép theo run), nên index shuffle trên feature CHÍNH LÀ index trên raw pool.
    """
    idx = np.arange(len(fault_raw_all))
    rng = np.random.default_rng(seed + K)  # d_rng = seed + K, đúng như split_and_scale
    rng.shuffle(idx)
    return fault_raw_all[idx[:n_fault]]


def score(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ims")
    ap.add_argument("--n-fault", type=int, default=20)
    ap.add_argument("--n-synth", type=int, default=3200)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    args = ap.parse_args()

    signa_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    all_normal = pipe.flatten_runs(signa_n)
    stride_eff = pipe._effective_stride(all_normal, args.win, args.stride)

    rows = []
    for K in [args.n_fault]:
        d_rng = np.random.default_rng(args.seed + K)
        d = pipe.split_and_scale(signa_n, signs_f, args.win, args.stride, d_rng,
                                 max_fault_train=K)
        X_base, y_base, X_test, y_test = pipe.to_train_test_arrays(d)
        scaler_mu, scaler_sd = d["scaler_mu"], d["scaler_sd"]
        gen_rng = np.random.default_rng(args.seed + 1000 + K)

        # ---- nguồn sinh lỗi giả: 2 phiên bản ----
        # BUGGY: như script 06 `_raw_train_windows` (pool toàn bộ, stride=args.stride)
        flt_buggy = raw_windows(pipe.flatten_runs(signs_f), args.win, args.stride)
        fault_raw_buggy = flt_buggy[gen_rng.choice(len(flt_buggy), size=min(K, len(flt_buggy)), replace=False)]

        # FIXED: đúng K cửa sổ lỗi TRAIN (stride_eff, khớp Xf_train)
        flt_all_eff = raw_windows(pipe.flatten_runs(signs_f), args.win, stride_eff)
        fault_raw_fixed = fault_rank_train(flt_all_eff, K, args.seed, K)

        # Dùng CÙNG gen_rng mới cho cả hai để công bằng (tái lập phân hoạch sinh)
        def run_generator(fault_raw, tag):
            gen = OptimizedFaultGenerator(fs=pipe.FS)
            norm_raw = raw_windows(pipe.flatten_runs(signa_n), args.win, stride_eff)
            gen.calibrate(fault_raw, norm_raw)
            synth = gen.generate_aligned_interp(fault_raw, args.n_synth, gen_rng)
            if args.amp > 0:
                aa = OptimizedFaultGenerator(fs=pipe.FS)
                aa.f_char = gen.f_char
                aa._calibrated = True
                synth = aa._amp_rescale(synth, fault_raw, gen_rng)
            rawf = ft.raw_features(synth, fs=pipe.FS)
            X_syn = ft.zscore(rawf, scaler_mu, scaler_sd)
            y_syn = np.ones(len(X_syn))
            X_aug = np.concatenate([X_base, X_syn])
            y_aug = np.concatenate([y_base, y_syn])
            clf = RandomForestClassifier(n_estimators=200, random_state=0,
                                        class_weight="balanced").fit(X_aug, y_aug)
            rb = score(RandomForestClassifier(n_estimators=200, random_state=0,
                                              class_weight="balanced").fit(X_base, y_base),
                       X_test, y_test)
            ra = score(clf, X_test, y_test)
            rows.append({"tag": tag, "source_windows": len(fault_raw),
                         "before": rb, "after": ra})
            print(f"[{tag:>7}] nguồn={len(fault_raw):>5} win "
                  f"before={rb['recall']:.4f} → after={ra['recall']:.4f} "
                  f"(Δ={ra['recall']-rb['recall']:+.4f})")

        run_generator(fault_raw_buggy, "BUGGY")
        run_generator(fault_raw_fixed, "FIXED")

    print("\n=== TÓM TẮT A/B LEAKAGE (K=%d) ===" % args.n_fault)
    print("test_fault=%d  stride_eff=%d" % (int((y_test == 1).sum()), stride_eff))
    for r_ in rows:
        print(f"  {r_['tag']:>7} nguồn={r_['source_windows']:>5} "
              f"before={r_['before']['recall']:.4f} after={r_['after']['recall']:.4f}")

    out = RESULTS / f"leak_ab_K{args.n_fault}_s{args.seed}.json"
    RESULTS.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n_fault": args.n_fault, "seed": args.seed,
                   "n_synth": args.n_synth, "stride_eff": stride_eff,
                   "test_fault": int((y_test == 1).sum()), "rows": rows}, f, indent=2)
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
