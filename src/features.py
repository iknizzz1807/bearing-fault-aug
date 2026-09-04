"""Feature engineering vector-hoá (không vòng lặp Python/window).

Nhiệm vụ: biến mỗi cửa sổ tín hiệu rung (win,) thành vector đặc trưng cố định để
đưa vào model phân loại (RF nhị phân). Toàn bộ tính trên ma trận (n_windows, win)
với các phép toán dọc theo axis=1 → nhanh hơn vài chục lần so với lặp từng window.

VÌ SAO CÓ ĐỦ 3 NHÓM (time + freq + envelope) = 24 FEATURE?
    Một tín hiệu lỗi ổ bi thể hiện qua NHIỀU góc khác nhau:
      - time (10) : biên độ / độ lệch / độ nhọn / xung (crest, shape, impulse) —
                    bắt sự thay đổi cường độ rung khi bắt đầu hỏng.
      - freq (6)  : phân bố năng lượng phổ (centroid, spread, flatness, energy).
      - env (8)   : đường bao sau khử điều chế — bắt XUNG VA ĐẬP của lỗi (xem dưới).
    Gộp cả ba cho model "nhìn" từ nhiều góc, tăng khả năng tách normal vs fault.

    LƯU Ý SỐ FEATURE 16 → 24: trước đây chỉ time + freq (16 cột), sau bổ sung 8
    feature envelope thành 24. Thực nghiệm cho thấy thêm feature KHÔNG cải thiện
    đáng kể (xem SPEC.md §3.1): RF-24feat recall ~0.879 so với RF-16feat ~0.886.
    Vì vậy đây được giữ như một "lever" đã thử, làm đối chứng — không phải thứ
    quyết định kết quả.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert

FEATURE_NAMES = [
    "mean", "std", "rms", "peak", "peak2peak", "skewness", "kurtosis",
    "crest_factor", "shape_factor", "impulse_factor",
    "spec_centroid", "spec_spread", "spec_flatness", "spec_energy",
    "freq_mean", "freq_std",
    # ---- nhóm chẩn đoán rung bổ sung (envelope / xung / thống kê) ----
    "env_rms", "env_energy", "env_peak", "env_centroid", "env_spread",
    "env_kurtosis", "zero_crossing", "abs_energy",
]


def _time_features(X: np.ndarray) -> np.ndarray:
    n, win = X.shape
    rms = np.sqrt(np.mean(X**2, axis=1))
    peak = np.max(np.abs(X), axis=1)
    pptp = np.ptp(X, axis=1)
    mean_abs = np.mean(np.abs(X), axis=1)
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    ef = np.stack([
        mean,
        std,
        rms,
        peak,
        pptp,
        skew(X, axis=1),
        kurtosis(X, axis=1),
        peak / (rms + 1e-12),                        # crest factor
        rms / (mean_abs + 1e-12),                    # shape factor
        peak / (mean_abs + 1e-12),                   # impulse factor
    ], axis=1)
    return ef


def _freq_features(X: np.ndarray, fs: float) -> np.ndarray:
    n, win = X.shape
    window = np.hanning(win)
    Xf = np.abs(rfft(X * window, axis=1))
    freqs = rfftfreq(win, 1.0 / fs)
    power = Xf**2
    energy = power.sum(axis=1, keepdims=True)
    valid = energy[:, 0] > 1e-12

    # mọi phép chia đều ở dạng vector — tránh chia 0 bằng +eps
    centroid = (freqs[None, :] * Xf).sum(axis=1) / (Xf.sum(axis=1) + 1e-12)
    spread = np.sqrt(((freqs[None, :] - centroid[:, None])**2 * Xf).sum(axis=1)
                     / (Xf.sum(axis=1) + 1e-12))
    log_power = np.mean(np.log(power + 1e-12), axis=1)
    flatness = np.exp(log_power) / (np.mean(power, axis=1) + 1e-12)
    f_mean = (freqs[None, :] * Xf).sum(axis=1) / (Xf.sum(axis=1) + 1e-12)
    f_std = np.sqrt(((freqs[None, :] - f_mean[:, None])**2 * Xf).sum(axis=1)
                    / (Xf.sum(axis=1) + 1e-12))
    return np.stack([centroid, spread, flatness, energy[:, 0], f_mean, f_std], axis=1)


def _envelope_features(X: np.ndarray, fs: float) -> np.ndarray:
    """Envelope (đường bao) qua biến đổi Hilbert — bắt đúng XUNG VA ĐẬP của lỗi ổ bi.

    Tín hiệu rung khi ổ bi hỏng có các "nhịp va đập" tần số BPFO/BPFI bị ĐIỀU CHẾ
    bởi chuyển động quay. Phổ thô (FFT trực tiếp) khó thấy các nhịp vì chúng bị
    "che" bởi phổ nền; nhưng đường bao |analytic signal| sau Hilbert thì lộ rõ tần
    số va đập (chính là BPFO/BPFI sau khi khử điều chế) → đây là TRỤ CỘT trong chẩn
    đoán rung (envelope analysis / demodulation), chính là lý do nhóm này được thêm
    vào bộ 16 → 24 feature.

    Trả về (n, 8): env_rms, env_energy, env_peak, env_centroid, env_spread,
                   env_kurtosis, zero_crossing, abs_energy
    """
    n, win = X.shape
    amp = np.abs(hilbert(X, axis=1))              # đường bao (n, win)
    rms_env = np.sqrt(np.mean(amp**2, axis=1))
    peak_env = np.max(np.abs(amp), axis=1)
    kurt_env = kurtosis(amp, axis=1)

    Xf = np.abs(rfft(amp, axis=1))                 # phổ của đường bao
    freqs = rfftfreq(win, 1.0 / fs)
    power = Xf**2
    energy_env = power.sum(axis=1)
    centroid_env = (freqs[None, :] * Xf).sum(axis=1) / (Xf.sum(axis=1) + 1e-12)
    spread_env = np.sqrt(((freqs[None, :] - centroid_env[:, None])**2 * Xf).sum(axis=1)
                         / (Xf.sum(axis=1) + 1e-12))

    # zero crossing: số lần tín hiệu đổi dấu — thể hiện "tần số nhịp" chậm/rõ
    signs = np.signbit(X)
    zc = (signs[:, 1:] != signs[:, :-1]).sum(axis=1)

    abs_energy = np.sum(X**2, axis=1)              # tổng năng lượng thời gian

    return np.stack([rms_env, energy_env, peak_env, centroid_env,
                     spread_env, kurt_env, zc.astype(float), abs_energy], axis=1)


def raw_features(windows: np.ndarray, fs: float = 12000) -> np.ndarray:
    """(n, win) -> (n, len(FEATURE_NAMES)) — KHÔNG chuẩn hoá.

    Ghép 3 khối: time (10) + freq (6) + envelope (8) = 24 feature cho mỗi cửa sổ.
    Trả về feature THÔ (raw) để bước sau fit MỘT scaler CHUNG trên train rồi áp
    đồng bộ cho normal + fault + synthetic. Nếu chuẩn hoá riêng từng cụm, model sẽ
    không thấy được khác biệt phân bố giữa chúng → mất ý nghĩa so sánh."""
    return np.hstack([
        _time_features(windows),
        _freq_features(windows, fs),
        _envelope_features(windows, fs),
    ])


def fit_zscore(train_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit (mean, std) cho z-score từ tập train."""
    mu = train_feats.mean(axis=0)
    sd = train_feats.std(axis=0) + 1e-12
    return mu, sd


def zscore(feats: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (feats - mu) / sd


def extract_features(windows: np.ndarray, fs: float = 12000,
                     mu: np.ndarray | None = None, sd: np.ndarray | None = None) -> np.ndarray:
    """(n, win) -> (n, len(FEATURE_NAMES)).

    Mặc định tự z-score theo chính cụm (tiện cho prototype). Khi cần chuẩn hoá
    thống nhất train/test, truyền `mu`/`sd` fit từ train (xem fit_zscore)."""
    feats = raw_features(windows, fs)
    if mu is None or sd is None:
        mu, sd = fit_zscore(feats)
    return (feats - mu) / sd