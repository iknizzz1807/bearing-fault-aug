"""10_denoise_ab.py — A/B: có/không wavelet-denoise TRƯỚC khi tính feature (LANL 1st).

Vấn đề cần trả lời: LANL 1st (team Psi/Zoo) denoise bằng wavelet (db4 + universal
threshold) rồi mới trích feature. Mình chưa từng denoise raw → 24 feature mình tính
trên tín hiệu nhiễu. Câu hỏi: denoise có làm sạch 24 feature giúp model học tốt hơn
ở kịch bản KHIẾM (K=20) hay không?

CÁCH ĐO (tránh missout do harness khác):
  - Tái dựng CHÍNH XÁC split pipeline bằng trick shuffle-equivalence (như script 09,
    đã verify: rng.shuffle(matrix) == rng.shuffle(arange(N)) rồi index) → cùng train/test,
    cùng scaler fit-on-train cho MỌI biến thể → so sánh CÔNG BẰNG.
  - Biến thể duy nhất: DENOISE raw windows (train + test + synth) trước khi
    ft.raw_features. Generator là interp_align+amp (best §3.4.5) cho cả hai nhánh.

Chạy: python scripts/10_denoise_ab.py --n-fault 20 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.fft import rfft, irfft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pipeline as pipe, data as ds, features as ft  # noqa: E402
from src.generator import OptimizedFaultGenerator  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import recall_score, roc_auc_score  # noqa: E402


def denoise_wavelet(windows: np.ndarray, wavelet: str = "db4", level: int = 1,
                    mult: float = 0.1) -> np.ndarray:
    """Denoise wavelet THẬT theo LANL 1st: wavedec(db4) + universal hard threshold.

    Đúng từng bước của kernel thắng giải (team Psi/Zoo):
      1. `maddest` = mean(|coeff - mean|) làm ước lượng MAD.
      2. sigma = (1/0.6745) * maddest(coeff[-level])  — ước lượng nhiễu.
      3. uthresh = sigma * sqrt(2*log(n)) * mult       — universal threshold.
      4. hard-threshold các hệ số chi tiết, tái dựng (waverec).
    LƯU Ý SCALE: LANL dùng trên segment 150k mẫu; window 512 của ta ngắn hơn nên
    universal threshold mặc định QUÁ mạnh (phá sạch std tín hiệu). `mult` giảm ngưỡng
    để bảo toàn cấu trúc (test: mult=0.1 giữ std ~0.99, mult=1 phá còn 0.23-0.47).
    """
    import pywt
    w = np.asarray(windows, dtype=np.float64).copy()
    if w.ndim == 1:
        w = w[None, :]
    n, win = w.shape

    def maddest(d):
        return np.mean(np.abs(d - np.mean(d)))

    out = np.empty_like(w)
    for i in range(n):
        coeff = pywt.wavedec(w[i], wavelet, mode="per")
        sigma = (1.0 / 0.6745) * maddest(coeff[-level]) + 1e-12
        uthresh = mult * sigma * np.sqrt(2.0 * np.log(win))
        coeff[1:] = [pywt.threshold(c, value=uthresh, mode="hard") for c in coeff[1:]]
        out[i] = pywt.waverec(coeff, wavelet, mode="per")[:win]
    return out[0] if windows.ndim == 1 else out


def _windows_per_run(runs, win, stride_eff):
    out = []
    acc = []
    for run in runs:
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride_eff)
            if len(w):
                acc.append(w)
        if acc:
            out.append(np.concatenate(acc))
            acc = []
    return out


def build_split_denoise(signs_n, signs_f, win, stride, rng, max_fault_train, do_denoise):
    """Như build_split (script 09) nhưng (tuỳ chọn) denoise windows trước khi feature.

    Trick shuffle-equivalence: rng.shuffle(Xf) chỉ phụ thuộc SỐ HÀNG N → index raw &
    feature theo cùng idx → giữ nguyên train/test như pipeline. scaler fit-on-train.
    """
    all_normal = pipe.flatten_runs(signs_n)
    stride_eff = pipe._effective_stride(all_normal, win, stride)

    Xn_run_feats, Xn_run_windows = [], []
    for run in signs_n:
        feats_run, w_run = [], []
        for sig in run:
            w = ds.make_windows(sig, win=win, stride=stride_eff)
            if len(w):
                w = denoise_wavelet(w) if do_denoise else w
                feats_run.append(ft.raw_features(w, fs=pipe.FS))
                w_run.append(w)
        feats_run = np.concatenate(feats_run) if feats_run else np.empty((0, 24))
        w_run = np.concatenate(w_run) if w_run else np.empty((0, win))
        Xn_run_feats.append(feats_run)
        Xn_run_windows.append(w_run)

    # fault
    Xf_parts, Xf_windows_parts = [], []
    for sig in pipe.flatten_runs(signs_f):
        w = ds.make_windows(sig, win=win, stride=stride_eff)
        if len(w):
            w = denoise_wavelet(w) if do_denoise else w
            Xf_parts.append(ft.raw_features(w, fs=pipe.FS))
            Xf_windows_parts.append(w)
    Xf = np.concatenate(Xf_parts) if Xf_parts else np.empty((0, 24))
    Xf_windows = np.concatenate(Xf_windows_parts) if Xf_windows_parts else np.empty((0, win))

    # normal 3-zone
    Xn_train_z, Xn_test_z, Xn_test_w, Xn_grey = [], [], [], []
    Xn_train_w_z = []
    for Xn_run, wr in zip(Xn_run_feats, Xn_run_windows):
        i_tr = int(len(Xn_run) * pipe.NORMAL_TRAIN_END)
        i_te = int(len(Xn_run) * pipe.NORMAL_TEST_END)
        Xn_train_z.append(Xn_run[:i_tr])
        Xn_test_z.append(Xn_run[i_tr:i_te])
        Xn_test_w.append(wr[i_tr:i_te])
        Xn_grey.append(Xn_run[i_te:])
        Xn_train_w_z.append(wr[:i_tr])
    Xn_train_raw = np.concatenate(Xn_train_z)
    Xn_test_raw = np.concatenate(Xn_test_z)
    Xn_train_windows = np.concatenate(Xn_train_w_z) if Xn_train_w_z else np.empty((0, win))

    # fault: shuffle theo index (khớp hệt rng.shuffle(Xf))
    idx = np.arange(Xf.shape[0])
    rng.shuffle(idx)
    Xf = Xf[idx]
    Xf_windows = Xf_windows[idx]
    n_f_test = int((1 - 0.2) * len(Xf))
    Xf_test_raw = Xf[-n_f_test:]
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
    # raw windows đã shuffle → generator sinh lỗi từ ĐÚNG tập lỗi train (khớp Xf_train_raw)
    Xf_train_windows = Xf_windows[:n_f_train]
    return X_base, y_base, X_test, y_test, mu, sd, Xf_train_windows, Xn_train_windows


def run(K, seed, denoise, n_synth=3200):
    signs_n, signs_f = pipe.load_normal_fault_signals("ims", 300)
    rng = np.random.default_rng(seed + K)
    Xb, yb, Xt, yt, mu, sd, fault_windows, norm_windows = build_split_denoise(
        signs_n, signs_f, 512, 256, rng, K, denoise)

    gen_rng = np.random.default_rng(seed + 1000 + K)
    gen = OptimizedFaultGenerator(fs=pipe.FS)
    fw = np.asarray(fault_windows)
    nw = np.asarray(norm_windows)
    gen.calibrate(fw, nw)                                   # như script 06: fault + normal
    syn_sig = gen.generate_aligned_interp(fw, n_synth, gen_rng)
    # --amp: post-scale biên độ chuẩn theo phân bố std lỗi thật
    aa = OptimizedFaultGenerator(fs=pipe.FS)
    aa.f_char = gen.f_char
    aa._calibrated = True
    syn_sig = aa._amp_rescale(syn_sig, fw, gen_rng)
    if denoise:
        syn_sig = denoise_wavelet(syn_sig)
    raw = ft.raw_features(syn_sig, fs=pipe.FS)
    X_syn = ft.zscore(raw, mu, sd)

    Xa = np.concatenate([Xb, X_syn])
    ya = np.concatenate([yb, np.ones(len(X_syn))])

    clf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
    clf.fit(Xa, ya)
    pred = clf.predict(Xt)
    rec = recall_score(yt, pred)
    auc = roc_auc_score(yt, clf.predict_proba(Xt)[:, 1])

    # đối chứng "before" recall (không aug) — phải khớp script 06 (K=20: ~0.3831)
    clf0 = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
    clf0.fit(Xb, yb)
    rec0 = recall_score(yt, clf0.predict(Xt))
    return rec, auc, rec0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fault", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--n-synth", type=int, default=3200)
    a = ap.parse_args()
    rec, auc, rec0 = run(a.n_fault, a.seed, a.denoise, a.n_synth)
    print(f"K={a.n_fault} seed={a.seed} denoise={a.denoise} before={rec0:.4f} after={rec:.4f} auc={auc:.4f}")
