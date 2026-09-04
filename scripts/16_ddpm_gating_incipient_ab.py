#!/usr/bin/env python3
"""SCRIPT 16 — Hướng A: INCIPIENT fault + DDPM sinh lỗi giả + QUALITY-GATING (A/B).

MỤC TIÊU: kịch bản lỗi sớm (incipient, vùng 90–95% vòng đời, biên độ nhẹ) là điểm yếu
của deliverable (recall chỉ 0.10–0.13, §6.2). Hướng này thử generative: train LightDDPM
trên K lỗi thật (GPU) rồi sinh lỗi giả, và thêm một BỘ LỌC CHẤT LƯỢNG (quality-gating)
để chỉ giữ các bản có phân bố 7 descriptor nằm trong dải [q10,q90] của lỗi thật
(loại bản méo/mode-collapse). So sánh A/B trên CÙNG harness incipient:
    baseline  (RF, K lỗi thật, không augment)
    DDPM thô   (RF + lỗi giả DDPM sinh ra, KHÔNG lọc)
    DDPM+gate  (RF + lỗi giả DDPM ĐÃ lọc descriptor)
Đối chứng thêm: interp_align+amp (deliverable §6.2) để xem generative có vượt interpolation.

CÁCH CHẠY (GPU local):
  python scripts/16_ddpm_gating_incipient_ab.py --n-fault 10 20 50 --out results/incip_ddpm_ab.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as ds          # noqa: E402
from src import features as ft      # noqa: E402
from src import pipeline as pipe    # noqa: E402
from src.ddpm import LightDDPM      # noqa: E402
from src.generator import OptimizedFaultGenerator   # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# 7 descriptor: std, rms, kurtosis, spec_centroid, env_energy, Aline@f_char, sideband_snr
DESC_NAMES = ["std", "rms", "kurt", "spec_centroid", "env_energy", "align_fchar", "sideband_snr"]


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


def descriptors(windows: np.ndarray, fs: float, f_char: float) -> np.ndarray:
    """7 descriptor "task-agnostic" (tái dùng logic generator._descriptors)."""
    w = np.asarray(windows, dtype=np.float64)
    win = w.shape[1]
    rf = ft.raw_features(w, fs=fs)
    s = np.std(w, axis=1)
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    kur, cen, env_e = rf[:, 6], rf[:, 10], rf[:, 17]
    Xc = w - w.mean(axis=1, keepdims=True)
    amp = np.abs(hilbert(Xc, axis=1))
    amp = amp - amp.mean(axis=1, keepdims=True)
    Xf = np.abs(np.fft.rfft(amp, axis=1)) ** 2
    fr = np.fft.rfftfreq(win, 1.0 / fs)
    jc = int(np.argmin(np.abs(fr - f_char)))
    Al = np.sqrt(Xf[:, jc]) / (rms + 1e-12)
    band = np.abs(fr - f_char) < 0.1 * f_char
    nb = np.abs(fr - 2 * f_char) < 0.1 * f_char
    snr = Xf[:, band].mean(axis=1) / (Xf[:, nb].mean(axis=1) + 1e-12)
    return np.stack([s, rms, kur, cen, env_e, Al, snr], axis=1)


def quality_gate(synth: np.ndarray, fault_train: np.ndarray, fs: float, f_char: float,
                 n_target: int, q_low: float = 0.10, q_high: float = 0.90,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Giữ các bản synthetic có descriptor nằm trong [q_low, q_high] của lỗi thật.

    Trả về đúng `n_target` bản (oversample rồi lọc; nếu thiếu bổ sung bằng cách
    resample descriptor-target bên trong dải để giữ đủ số lượng).
    """
    fw = np.asarray(fault_train, dtype=np.float64)
    syn = np.asarray(synth, dtype=np.float64)
    Dr = descriptors(fw, fs, f_char)
    lo = np.quantile(Dr, q_low, axis=0)
    hi = np.quantile(Dr, q_high, axis=0)
    Ds = descriptors(syn, fs, f_char)
    keep = np.all((Ds >= lo) & (Ds <= hi), axis=1)
    newsyn = syn[keep]
    if len(newsyn) >= n_target:
        sel = np.arange(len(newsyn))
        rng = rng or np.random.default_rng(0)
        rng.shuffle(sel)
        return newsyn[sel[:n_target]]
    # không đủ bản hợp lệ → giữ bản đã lọc rồi bổ sung bằng cách 'project' mỗi bản
    # còn thiếu về gần một target descriptor trong dải (scale std/rms, giữ tần số).
    rng = rng or np.random.default_rng(0)
    out = list(newsyn)
    i = 0
    while len(out) < n_target:
        cand = syn[i % len(syn)]
        tgt = rng.uniform(lo, hi)
        c_des = descriptors(cand[None, :], fs, f_char)[0]
        # chỉnh std/rms về gần target (giữ nguyên shape phổ)
        ratio = tgt[0] / (c_des[0] + 1e-12)
        cand = cand * float(np.clip(ratio, 0.5, 2.0))
        out.append(cand)
        i += 1
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims"], default="ims")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n-fault", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--incip-lo", type=float, default=0.90)
    ap.add_argument("--incip-hi", type=float, default=0.95)
    ap.add_argument("--n-synth", type=int, default=3200)
    ap.add_argument("--ddpm-epochs", type=int, default=150)
    ap.add_argument("--ddpm-t", type=int, default=200)
    ap.add_argument("--ddpm-batch", type=int, default=64)
    ap.add_argument("--ddpm-lr", type=float, default=2e-4)
    ap.add_argument("--ddpm-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q-low", type=float, default=0.10)
    ap.add_argument("--q-high", type=float, default=0.90)
    ap.add_argument("--out", default=str(RESULTS / "incip_ddpm_ab.json"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    win, stride = 512, 256
    fs = pipe.FS

    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    all_normal = pipe.flatten_runs(signs_n)
    stride_eff = pipe._effective_stride(all_normal, win, stride)
    print(f"stride_eff={stride_eff}  incipient=[{args.incip_lo},{args.incip_hi}]", flush=True)

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
    n_tr_n = n_tr_raw
    print(f"normal train windows={n_tr_raw.shape}, normal test windows={n_te_raw.shape}",
          flush=True)

    incip_signals = load_faults_window(args.dataset, args.incip_lo, args.incip_hi, args.limit)
    print(f"incipient fault signals loaded: {len(incip_signals)}", flush=True)
    f_inc_w = raw_windows(incip_signals, win, stride_eff)
    print(f"incipient fault windows: {f_inc_w.shape}", flush=True)

    # features nên train/test
    Xn_tr_f = ft.raw_features(n_tr_n, fs=fs)
    Xn_te_f = ft.raw_features(n_te_raw, fs=fs)
    Xinc_f = ft.raw_features(f_inc_w, fs=fs)
    print(f"n_inc_fault_test={len(Xinc_f)}", flush=True)

    rows = []
    for K in args.n_fault:
        # --- fault-train: K cửa sổ từ pool FAULT NẶNG (đúng như harness) ---
        pool_f = raw_windows(pipe.flatten_runs(signs_f), win, stride_eff)
        idx = np.arange(len(pool_f))
        rng_f = np.random.default_rng(args.seed + K)
        rng_f.shuffle(idx)
        n_all = max(1, int(0.2 * len(pool_f)))
        f_tr_raw = pool_f[idx[:n_all]][:K]

        Xf_tr_f = ft.raw_features(f_tr_raw, fs=fs)
        mu, sd = ft.fit_zscore(np.concatenate([Xn_tr_f, Xf_tr_f]))
        Xb = np.concatenate([ft.zscore(Xn_tr_f, mu, sd), ft.zscore(Xf_tr_f, mu, sd)])
        yb = np.concatenate([np.zeros(len(Xn_tr_f)), np.ones(len(Xf_tr_f))])
        X_te = np.concatenate([ft.zscore(Xn_te_f, mu, sd), ft.zscore(Xinc_f, mu, sd)])
        y_te = np.concatenate([np.zeros(len(Xn_te_f)), np.ones(len(Xinc_f))])

        # ---- A. baseline (RF, K lỗi thật, không augment) ----
        clf0 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(Xb, yb)
        r0 = _score(clf0, X_te, y_te)

        # ---- f_char cho descriptor-gating (học từ lỗi thật) ----
        gen = OptimizedFaultGenerator(fs=fs)
        gen.calibrate(f_tr_raw, n_tr_n)
        f_char = gen.f_char
        print(f"  [K={K}] f_char={f_char:.1f} Hz | fault_train={len(f_tr_raw)}", flush=True)

        # ---- B. DDPM (GPU): train trên lỗi thật, sinh oversample ----
        ddpm = LightDDPM(seq_len=win, T=args.ddpm_t, lr=args.ddpm_lr,
                         seed=args.ddpm_seed + K)
        ddpm.train(f_tr_raw, epochs=args.ddpm_epochs, batch=args.ddpm_batch, log_every=50)

        # sinh nhiều hơn n_synth để quality-gate có dư
        oversample = int(args.n_synth * 2.5)
        ddpm_rng = np.random.default_rng(args.ddpm_seed + 1000 + K)
        synth_raw = ddpm.sample(oversample, alpha_std=1.0)

        # ---- interp_align + amp (đối chứng deliverable §6.2) ----
        g_rng = np.random.default_rng(args.seed + 1000 + K)
        synth_ia = gen.generate_aligned_interp(f_tr_raw, args.n_synth, g_rng)
        aa = OptimizedFaultGenerator(fs=fs); aa.f_char = f_char; aa._calibrated = True
        synth_ia = aa._amp_rescale(synth_ia, f_tr_raw, g_rng)

        # ---- đo recall cho từng biến thể ----
        def eval_synth(synth_arr, label):
            X_syn = ft.zscore(ft.raw_features(synth_arr, fs=fs), mu, sd)
            X_aug = np.concatenate([Xb, X_syn])
            y_aug = np.concatenate([yb, np.ones(len(X_syn))])
            clf = RandomForestClassifier(n_estimators=200, random_state=0,
                                         class_weight="balanced").fit(X_aug, y_aug)
            return _score(clf, X_te, y_te), len(X_syn)

        r_ddpm, n_ddpm = eval_synth(synth_raw, "ddpm_raw")
        # BẮT BUỘC giữ đủ n_synth bản ĐÃ LỌC để so sánh công bằng cùng số lượng
        synth_gated = quality_gate(synth_raw, f_tr_raw, fs, f_char, args.n_synth,
                                   args.q_low, args.q_high,
                                   rng=np.random.default_rng(args.seed + 2000 + K))
        r_gate, n_gate = eval_synth(synth_gated, "ddpm_gate")
        r_ia, n_ia = eval_synth(synth_ia, "interp_align")

        rows.append({
            "n_fault_train": K,
            "incipient": f"{args.incip_lo}-{args.incip_hi}",
            "baseline": r0,
            "ddpm_raw": r_ddpm,
            "ddpm_gated": r_gate,
            "interp_align_amp": r_ia,
            "n_synth_per_variant": n_ia,
            "n_synth_gate_req": args.n_synth,
            "n_synth_gate_kept": int((np.all((descriptors(synth_raw, fs, f_char) >= np.quantile(descriptors(f_tr_raw, fs, f_char), args.q_low, axis=0)) &
                                              (descriptors(synth_raw, fs, f_char) <= np.quantile(descriptors(f_tr_raw, fs, f_char), args.q_high, axis=0)), axis=1)).sum()),
            "n_test_fault_incipient": int((y_te == 1).sum()),
            "n_train_normal": len(Xn_tr_f),
            "f_char": float(f_char),
        })
        print(f"  [K={K:>3}] baseline={r0['recall']:.4f} | ddpm_raw={r_ddpm['recall']:.4f} "
              f"| ddpm_gated={r_gate['recall']:.4f} | interp_align={r_ia['recall']:.4f}",
              flush=True)

    print("\n=== BẢNG: INCIPIENT (lỗi sớm) — DDPM vs GATE vs interp_align ===")
    print(f"{'K':>4} {'baseline':>9} {'ddpm_raw':>9} {'ddpm_gate':>10} {'interp':>9}")
    for r_ in rows:
        print(f"{r_['n_fault_train']:>4} {r_['baseline']['recall']:>9.4f} "
              f"{r_['ddpm_raw']['recall']:>9.4f} {r_['ddpm_gated']['recall']:>10.4f} "
              f"{r_['interp_align_amp']['recall']:>9.4f}")

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "n_synth": args.n_synth,
                   "incipient": [args.incip_lo, args.incip_hi],
                   "stride_eff": stride_eff, "ddpm": {"epochs": args.ddpm_epochs,
                                                       "T": args.ddpm_t,
                                                       "batch": args.ddpm_batch,
                                                       "lr": args.ddpm_lr},
                   "gate": {"q_low": args.q_low, "q_high": args.q_high},
                   "rows": rows}, f, indent=2)
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
