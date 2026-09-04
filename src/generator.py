"""generator.py — Bộ sinh lỗi giả TỐI ƯU cho kịch bản "khan hiếm dữ liệu lỗi".

╔══════════════════════════════════════════════════════════════════════════╗
║  ĐÂY LÀ CỐT LÕI CỦA BÀI (giải thích nếu bạn thắc mắc vì sao cần nó):        ║
║  Người dùng yêu cầu "tối ưu thuật toán sinh lỗi giả để từ 20–50 lỗi thật      ║
║  + sinh lỗi giả → hiệu năng tốt nhất". Phát hiện quan trọng: bộ sinh cũ        ║
║  (`physics.py`) dùng TAM SỐ VẬT LÝ CỐ ĐỊNH (hình học ổ bi mặc định, quay     ║
║  30Hz) và TẠO RA NHIỀU BẢN GẦN NHƯ GIỐNG HỆT NHAU (correlation = 1.0, xem    ║
║  SPEC §3.4) → rất ít thông tin, không bám đúng lỗi thật.                       ║
║                                                                              ║
║  Module này sửa cả hai vấn đề:                                                ║
║    1. HỌC ĐẶC TRƯNG LỖI THẬT từ 20–50 mẫu hiếm (tần số đặc trưng + biên độ    ║
║       + phổ hài). Thay vì đoán 92.3Hz, ta ĐO xem lỗi thật rung ở tần số nào.   ║
║    2. ĐA DẠNG HOÁ: sinh nhiều biến thể (khác pha, khác mức nặng, khác nền     ║
║       normal, khác bậc hài) → LỚP phủ đầy đủ không gian lỗi → model học        ║
║       được khái niệm "lỗi" thay vì thuộc lòng một mẫu.                          ║
╚══════════════════════════════════════════════════════════════════════════╝

CÁCH HOẠT ĐỘNG (tương thích pipeline + feature chuẩn):
    1. `calibrate(fault_windows, fs)`  : đo từ mấy cửa sổ lỗi THẬT →
        chiết ra `f_char` (tần số đặc trưng, Hz), `depth` (biên độ lỗi),
        bảng biên độ các bậc hài, và độ nặng trung bình.
    2. `generate(normal_windows, n, rng)` : sinh `n` cửa sổ lỗi giả.
        Mỗi bản = LẤY MỘT CỬA SỔ NORMAL NGẪU NHIÊN làm nền rồi bơm xung va đập
        theo mô hình hài (f_char, 2f, 3f…) với THAM SỐ NGẪU NHIÊN:
           - depth ngẫu nhiên (bám độ nặng lỗi thật ± ngẫu nhiên)
           - phase ngẫu nhiên (lỗi xảy ra lúc nào cũng được → đa dạng)
           - trộn hài ngẫu nhiên (không phải lúc nào hài to cũng mạnh nhất)
           - modulate vòng quay bật/tắt
           - nhiễu cường độ nhẹ (bản chồng lên nền khác nhau là đã khác nhau)
        → KỂ CẢ HAI BẢN CÙNG depth cũng KHÁC nhau vì (a) nền normal khác, (b) phase
        khác, (c) hài trộn khác. Đây chính là "đa dạng hoá" mà bộ cũ thiếu.

ĐẦU RA: mảng (n, win) tín hiệu lỗi giả (miền thời gian). Sau đó cứ đưa vào
`features.raw_features` rồi z-score bằng scaler TRAIN (chống rò rỉ) — giống hệt
luồng của `scripts/02`/`03`/`05`/`06`.

TẠI SAO LẤY NỀN LÀ NORMAL THAY VÌ LỖI THẬT (design rationale):
    Nếu chỉ trộn 20 lỗi thật với nhau, model sẽ "thuộc lòng" 20 mẫu đó (qua fit)
    và sụp trên lỗi thật mới. Bơm lỗi vào nền NORMAL (không phải lỗi) → sinh vô
    hạn tổ hợp pha x nền x độ nặng, model học "cấu trúc lỗi" thay vì "20 mẫu".
    Đồng thời tần số / biên độ vẫn ĐƯỢC HIỆU CHỈNH theo lỗi thật (không bịa).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert


class OptimizedFaultGenerator:
    """Sinh lỗi giả học từ dữ liệu lỗi thật hiếm + đa dạng hoá (xem docstring module)."""

    def __init__(self, fs: float):
        self.fs = float(fs)
        self.f_char: float | None = None     # Hz — tần số lỗi đặc trưng (học từ dữ liệu)
        self.depth: float = 0.5              # biên độ xung (tỉ lệ std nền)
        self.harmonics: np.ndarray | None = None  # biên độ hài tương đối [f, 2f, 3f, ...]
        self._calibrated = False

    # ------------------------------------------------------------------ #
    #  BƯỚC 1: HỌC ĐẶC TRƯNG TỪ LỖI THẬT
    # ------------------------------------------------------------------ #
    def calibrate(self, fault_windows: np.ndarray, normal_windows: np.ndarray | None = None,
                  fmin: float = 5.0, fmax: float = 300.0, n_harmonics: int = 4) -> dict:
        """Đo `f_char`, `depth`, `harmonics` từ các cửa sổ lỗi THẬT.

        Phương pháp (chuẩn trong chẩn đoán rung / envelope analysis):
          - Tách DC rồi lấy đường bao |Hilbert| → phổ của envelope (loại bỏ tần số
            dao động nền, giữ lại "nhịp va đập" của lỗi = chính là tần số BPFO/BPFI
            sau khi khử điều chế).
          - Làm MỊN phổ và tìm đỉnh trong dải [fmin, fmax] — phải giới hạn dải vì
            ở tần số cao (thường > 300–400Hz) phổ bị nhiễu và có thể che mất đỉnh
            đặc trưng (thực nghiệm IMS: nếu để fmax lớn sẽ bắt nhầm vào nhiễu cao
            tần, còn đỉnh thật ~23Hz nằm ở rìa). Phạm vi hợp lý là vùng tần số ổ bi
            lăn (vài chục Hz → vài trăm Hz), hài bậc cao vẫn nằm trong đó.
          - `f_char` = tần số có công suất envelope lớn nhất (đã mịn) trong dải này.
          - `depth`  = std(lỗi)/std(normal) — độ nặng trung bình của lỗi so với nền.
          - `harmonics` = công suất các bậc hài f, 2f, 3f, 4f (chuẩn hoá về max).

        Trả về dict chứa các giá trị học được (tiện ghi log / tái sử dụng).
        """
        fw = np.asarray(fault_windows, dtype=np.float64)
        f_char, power, freqs = self._dominant_envelope_freq(fw, fmin, fmax)
        self.f_char = f_char

        # độ nặng: nếu có normal để so sánh thì std ratio, còn không thì lấy std của lỗi
        if normal_windows is not None and len(normal_windows):
            nstd = np.std(normal_windows) + 1e-12
            self.depth = float(np.clip(np.std(fw) / nstd, 0.1, 5.0))
        else:
            self.depth = float(np.clip(np.std(fw) / 0.1, 0.1, 5.0))

        # biên độ hài: lấy công suất tại k*f_char (chọn bin gần nhất), chuẩn hoá
        amps = []
        for h in range(1, n_harmonics + 1):
            j = int(np.argmin(np.abs(freqs - h * f_char)))
            amps.append(power[j])
        amps = np.asarray(amps, dtype=np.float64)
        amps = amps / (amps.max() + 1e-12)       # hài 1 = 1.0
        self.harmonics = amps
        self._calibrated = True
        return {"f_char": self.f_char, "depth": self.depth, "harmonics": amps.tolist()}

    def _dominant_envelope_freq(self, fw: np.ndarray, fmin: float, fmax: float):
        """Tìm tần số envelope chiếm ưu thế (đã mịn phổ, giới hạn dải đặc trưng ổ bi).

        Vì sao phải LÀM MỊN (smoothing) phổ: chỉ vài chục cửa sổ lỗi thật → phổ rất
        nhiễu, một bin ngẫu nhiên ở tần số cao có thể "thắng" argmax. Trung bình
        cửa sổ 3–5 bin quanh mỗi điểm làm đỉnh liền mạch nổi lên, đỉnh lạ bị hạ.
        """
        Xc = fw - fw.mean(axis=1, keepdims=True)
        amp = np.abs(hilbert(Xc, axis=1))
        amp = amp - amp.mean(axis=1, keepdims=True)      # bỏ DC của envelope
        Xf = np.abs(rfft(amp, axis=1)) ** 2
        freqs = rfftfreq(Xc.shape[1], 1.0 / self.fs)
        power = Xf.mean(axis=0)                          # công suất trung bình
        # làm mịn phổ bằng trung bình di động (k=5) rồi mới tìm đỉnh → chống nhiễu
        k, kern = 5, np.ones(5) / 5
        power_s = np.convolve(power, kern, mode="same")
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not mask.any():
            raise ValueError(f"Không có bin tần số trong [{fmin},{fmax}] — kiểm tra fs/win.")
        i0 = int(np.argmax(power_s[mask]))
        return float(freqs[mask][i0]), power, freqs

    # ------------------------------------------------------------------ #
    #  BƯỚC 2: SINH NHIỀU BIẾN THỂ (đa dạng hoá)
    # ------------------------------------------------------------------ #
    def generate(self, normal_windows: np.ndarray, n: int,
                 rng: np.random.Generator) -> np.ndarray:
        """Sinh `n` cửa sổ lỗi giả từ `normal_windows` làm nền.

        Mỗi bản lỗi giả tuân theo mô hình xung lặp (như mục tiêu của đề:
        "fault injection"):
            x_f(t) = x_h(t) + A * Σ_k h(t - k*T) * w(t) + n(t)
        với T = 1/f_char, h(t) = xung va đập, w(t) = điều chế vòng quay.

        ĐIỂM MẠNH SO VỚI BỘ CŨ: các bản KHÁC NHAU chứ không giống hệt, vì mỗi lần
        ta ngẫu nhiên:
          - chọn một nền normal khác nhau (nền rung thật luôn khác)
          - chọn độ nặng depth quanh giá trị học được
          - chọn phase bắt đầu của chuỗi xung
          - trộn các bậc hài theo tỉ lệ ngẫu nhiên
          - bật/tắt điều chế, đổi tốc độ điều chế

        QUAN TRỌNG (được hiệu chỉnh từ thực nghiệm, xem SPEC §3.4):
          Lỗi ổ bi thật (IMS) biểu hiện qua HAI đặc trưng quyết định:
            (1) AMPLITUDE: std/rms/peak/env_energy tăng ~1.85x so với nền.
            (2) TẦN SỐ CAO: lỗi kích thích CỘNG HƯỞNG cơ khí (IMS ~3000 Hz) làm phổ
                dịch lên (spec_centroid tăng). Tần số đặc trưng (BPFO/BPFI ~23–90 Hz)
                chỉ là tần số VA ĐẬP (điều chế), KHÔNG phải sóng mang.
          Bộ cũ chỉ bơm xung tần số thấp (≈f_char) → kéo spec_centroid XUỐNG sai
          hướng và không tăng amplitude → model chẳng học được gì. Bộ mới:
            - xung mang tần số CỘNG HƯỞNG cao (resonance_hz ~3000 Hz);
            - các xung đó xuất hiện theo chu kỳ T = 1/f_char (điều chế bởi lỗi);
            - CHUẨN HOÁ toàn bộ tín hiệu về std mục tiêu ≈ std lỗi thật (amplitude).
        """
        if not self._calibrated or self.f_char is None:
            raise RuntimeError("Gọi calibrate() trước khi generate().")
        nw = np.asarray(normal_windows, dtype=np.float64)
        n_out = int(n)
        fs = self.fs
        win = nw.shape[1]

        resonance = getattr(self, "resonance_hz", 3000.0)   # tần số cộng hưởng cơ (Hz)

        results = np.empty((n_out, win))
        for i in range(n_out):
            base = nw[rng.integers(0, len(nw))]                 # nền normal ngẫu nhiên
            depth_i = float(np.clip(self.depth * rng.uniform(0.6, 1.7), 1.2, 8.0))
            # tần số va đập + trộn hài ngẫu nhiên (dịch nhẹ ±2% như trượt thực tế)
            f_inst = self.f_char * (1 + rng.uniform(-0.02, 0.02))
            harms = self.harmonics * rng.uniform(0.3, 1.0, size=len(self.harmonics))
            harms /= (harms.max() + 1e-12)
            res_hz = resonance * (1 + rng.uniform(-0.15, 0.15))  # cộng hưởng cũng trượt nhẹ

            x = np.zeros(win)
            gap = max(2, int(round(fs / f_inst)))        # chu kỳ va đập (điều chế)
            phase = rng.integers(0, gap)                  # pha ngẫu nhiên
            for h, a in enumerate(harms, start=1):
                per = max(4, int(gap / h))                # hài bậc h → va đập nhanh hơn
                imp = self._impact(int(per), res_hz, fs) * a
                for k in range(int(np.ceil((win - phase) / per))):
                    start = int(phase + k * per)
                    if start >= win:
                        break
                    seg = imp[:max(0, win - start)]
                    x[start:start + len(seg)] += seg
            # điều chế vòng quay (bật 70% các bản — lỗi thật không phải lúc nào cũng đều)
            if rng.random() < 0.7:
                mod_period = max(gap * 2, int(fs / 5))
                t = np.arange(win)
                x = x * (0.6 + 0.4 * np.cos(2 * np.pi * t / mod_period * rng.uniform(0.8, 1.4)))

            # Chuẩn hoá về đúng mức std mục tiêu (đặc trưng lỗi thật là AMPLITUDE tăng):
            # tỉ lệ xung so với toàn tín hiệu được giữ từ mô hình, chỉ kéo độ lớn tổng.
            x_unit = x / (x.std() + 1e-12)          # chuẩn hoá mô hình xung về 1 std
            results[i] = base + depth_i * base.std() * x_unit
        return results

    def generate_amp_aligned(self, fault_windows: np.ndarray, n: int,
                             rng: np.random.Generator, noise_frac: float = 0.10,
                             jitter_smp: int = 8) -> np.ndarray:
        """interpolate nhưng HẬU XỬ LÝ sao cho một số descriptor khớp lỗi THẬT.

        Phát hiện (SPEC §3.4.3): interpolate trộn trực tiếp lỗi thật nên bám manifold,
        NHƯNG các bản trộn có xu hướng "dịu" hơn bản gốc — std/env_energy thấp hơn
        thật (~0.7–0.85x), sideband cao hơn (~1.4x). Tức là interpolate sinh lỗi
        "trung bình", ít phủ các mẫu cực đoan (amplitude cao / sideband thấp) mà chẩn
        đoán cần.

        `generate_amp_aligned` = interpolate rồi chuẩn hoá từng bản về **std** mục tiêu
        lấy từ PHÂN BỐ std của lỗi thật (resample std trong dải [q10, q90] của thật):
            x_syn = x_syn * (std_target / std(x_syn))
        → mỗi bản giữ độ lớn tín hiệu bằng một bản lỗi thật ngẫu nhiên (không quá
        dịu / không quá to), bù đúng khoảng thiếu amplitude của interpolate. Giữ nguyên
        cấu trúc thời gian (chỉ đổi biên độ toàn cục) → không phá manifold.
        """
        syn = self.generate_interpolated(fault_windows, n, rng, noise_frac, jitter_smp)
        return self._amp_rescale(syn, fault_windows, rng)

    def _amp_rescale(self, syn: np.ndarray, fault_windows: np.ndarray,
                     rng: np.random.Generator, full_dist: bool = False) -> np.ndarray:
        """Chuẩn biên độ từng bản sinh về std lấy từ phân bố lỗi thật.

        Tách riêng để tái dùng cho interp_align / spectral_mixup (--amp). Không phụ
        thuộc calib; chỉ cần cửa sổ lỗi thật để rút phân bố std.

        `full_dist=False` (MẶC ĐỊNH, hành vi cũ): bóp std mục tiêu về [q10, q90] của
        lỗi thật — loại 20% cực đoan hai đầu, tránh sinh bản quá nhỏ/quá to.
        `full_dist=True`: resample std mục tiêu từ TOÀN BỘ phân phối std lỗi thật
        (giữ cả mẫu cực đoan) — lỗi thật test có cả bản "êm" lẫn bản "rung mạnh",
        nên lỗi giả trải hết dải giúp model học đủ dải độ rung.
        """
        syn = np.asarray(syn, dtype=np.float64)
        fw = np.asarray(fault_windows, dtype=np.float64)
        stds_real = np.std(fw, axis=1)
        lo, hi = np.quantile(stds_real, 0.10), np.quantile(stds_real, 0.90)
        out = syn.copy()
        for i in range(len(syn)):
            if full_dist:
                # resample từ phân phối std THẬT (vd bootstrap): chọn ngẫu nhiên 1 bản thật
                target = stds_real[rng.integers(0, len(stds_real))]
            else:
                target = rng.uniform(lo, hi)
            s_now = syn[i].std() + 1e-12
            out[i] = syn[i] * (target / s_now)
        return out

    def generate_interpolated(self, fault_windows: np.ndarray, n: int,
                              rng: np.random.Generator, noise_frac: float = 0.10,
                              jitter_smp: int = 8) -> np.ndarray:
        """Sinh lỗi giả bằng TRỘN (interpolation) giữa các cửa sổ lỗi THẬT + jitter.

        Đây là chiến lược TỐI ƯU nhất khi dữ liệu lỗi khan hiếm (chỉ 20–50 mẫu) —
        kết quả thực nghiệm SPEC §3.4: recall cải thiện vượt trội so với physics
        injection hay heuristic (K=20: ~0.59-0.65 vs ~0.43). Vì sao?
          - Khi chỉ có vài chục lỗi thật, điều mình muốn là nội suy trong KHÔNG GIAN
            LỖI THẬT (bám sát manifold) — trộn 2 lỗi thật bằng hệ số ngẫu nhiên a
            rồi thêm nhiễu nhẹ + dịch pha → ra vô số biến thể VẪN giống lỗi thật,
            không "rơi ra ngoài" như bơm xung vào nền normal.
          - Physics injection (bơm theo f_char) chỉ tạo đúng MỘT kiểu lỗi; khi test
            chứa nhiều chế độ hỏng khác nhau thì model học lệch → recall kém hơn.

        Mỗi bản: x_syn = a*x_fault_i + (1-a)*x_fault_j + nhiễu + roll pha.
        `a ~ U(0,1)`, chọn ngẫu nhiên 2 cửa sổ lỗi thật (kể cả cùng cửa sổ → jitter).
        `noise_frac`: biên độ nhiễu (× std tín hiệu tổng).
        `jitter_smp`: số mẫu trượt pha ngẫu nhiên (đa dạng thời gian, chống trùng).
        """
        fw = np.asarray(fault_windows, dtype=np.float64)
        n_out = int(n)
        results = np.empty((n_out, fw.shape[1]))
        for i in range(n_out):
            if len(fw) >= 2:
                # trộn 2 cửa sổ lỗi thật (i,j độc lập; có thể trùng nếu len nhỏ)
                i0, j0 = rng.integers(0, len(fw), size=2)
                a = rng.random()
                base = a * fw[i0] + (1 - a) * fw[j0]
            else:
                base = fw[0].copy()
            # jitter pha thời gian (chống trùng + mô phỏng lệch pha lỗi)
            sh = rng.integers(-jitter_smp, jitter_smp + 1)
            if sh:
                base = np.roll(base, sh)
            # nhiễu nhẹ quanh std của chính nó
            results[i] = base + rng.normal(0, noise_frac * base.std() + 1e-12, base.shape)
        return results

    # ------------------------------------------------------------------ #
    #  PHASE-AWARE MIXUP (chống destructive interference — đo SPEC §3.4.5)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _best_corr_shift(a: np.ndarray, b: np.ndarray) -> int:
        """Độ trượt (vòng) khiến `a` đạt tương quan CAO NHẤT với `b`, dùng FFT.

        Cơ sở (đã kiểm chứng bằng test unit): `np.roll(a, k)` với
        `k = argmax(|ifft(conj(FFT(a)) * FFT(b))|)` cho hệ số tương quan = 1.0 với
        b khi a là bản lệch pha của b. → Dùng để căn pha hai cửa sổ lỗi trước khi
        trộn, triệt tiêu hiện tượng "trộn hai sóng mạng lệch pha → triệt tiêu sóng
        mang" (destructive interference) làm interpolate sinh bản mềm (std/env_energy
        thấp hơn thật) — SPEC §3.4.3.
        """
        FFa, FFb = np.fft.fft(a), np.fft.fft(b)
        cc = np.fft.ifft(np.conj(FFa) * FFb)
        # argmax(real(cc)) — KHÔNG dùng abs: abs sẽ chọn LAG ANTI-PHASE (corr âm, trộn
        # tự triệt tiêu) trên tín hiệu tuần hoàn (đã test: sin lệch π → abs chọn corr=−1,
        # real chọn corr≈+0.97). Trên IMS ít gặp nhưng trên CWRU (sóng thuần) có 18% cặp
        # bị chọn nhầm. Xem §3.4.6c (đã xác minh không ảnh hưởng kết luận interp_align).
        return int(np.argmax(np.real(cc)))

    def generate_aligned_interp(self, fault_windows: np.ndarray, n: int,
                                rng: np.random.Generator, noise_frac: float = 0.10,
                                jitter_smp: int = 8, k_mix: int = 6) -> np.ndarray:
        """interpolate NHƯNG căn pha hai cửa sổ bằng cross-correlation trước khi trộn.

        Giải quyết trực tiếp vấn đề interruptive interference của `generate_interpolated`:
        gọi hai cửa sổ lỗi x1, x2 có cùng sóng mang (cộng hưởng ~3000Hz, SPEC §3.4)
        nhưng LỆCH PHA → trộn a*x1 + (1-a)*x2 làm sóng mang triệt tiêu → bản trộn
        "mềm" hơn (std/env_energy ~0.7–0.85x thật). Tác giả đề xuất trộn trong miền
        magnitude phổ để thoát lệch pha. Ta chọn cách ỔN ĐỊNH & bám manifold nhất:
        giữ NGUYÊN tín hiệu thời gian, chỉ dịch X2 về trùng pha với X1 (cross-corr),
        RỒI mới trộn tuyến tính.

        `k_mix` >= 2 (MẶC ĐỊNH 6, đã chọn qua A/B Mục 4 — SPEC §3.4.8): trộn `k_mix` cửa sổ
        lỗi thật — mọi cửa sổ được căn pha về cửa sổ THỨ NHẤT rồi lấy tổng trọng số
        (trọng số Dirichlet ngẫu nhiên), mở rộng điểm phủ manifold lỗi (vùng giữa nhiều
        mẫu). k_mix=2 giữ đúng hành vi cũ (a, 1-a) với a~U(0,1) = Dirichlet(1,...,1).
        Không phá manifold vì mọi bản vẫn là tổ hợp tuyến tính của lỗi thật đã căn pha.

        Kết quả: mỗi bản vẫn nằm trong không gian lỗi thật (không bịa phase), sóng mang
        giữ được năng lượng → std/env_energy sát thật hơn, mạnh hơn amp_align (vốn chỉ
        scale biên độ sau). (Tốt hơn trộn magnitude: trộn magnitude + giữ phase làm
        biến dạng dạng sóng.)
        """
        fw = np.asarray(fault_windows, dtype=np.float64)
        n_out = int(n)
        # k_mix hợp lệ: tối đa số lỗi thật có (ít nhất 2), tối thiểu 2
        km = int(np.clip(k_mix, 2, max(2, len(fw))))
        results = np.empty((n_out, fw.shape[1]))
        for i in range(n_out):
            if len(fw) >= 2:
                # chọn `km` cửa sổ lỗi thật + trọng số Dirichlet (a~U(0,1) khi km=2)
                picks = rng.integers(0, len(fw), size=km)
                w = rng.dirichlet(np.ones(km))
                # căn mọi cửa sổ về pha của cửa sổ ĐẦU (chuẩn pha chung: lỗi thật cộng hưởng)
                ref = fw[picks[0]]
                base = np.zeros(fw.shape[1])
                for m, idx in enumerate(picks):
                    wm = fw[idx]
                    if m > 0:
                        k = self._best_corr_shift(wm, ref)
                        if k:
                            wm = np.roll(wm, k)
                    base += w[m] * wm
            else:
                base = fw[0].copy()
            sh = rng.integers(-jitter_smp, jitter_smp + 1)
            if sh:
                base = np.roll(base, sh)
            results[i] = base + rng.normal(0, noise_frac * base.std() + 1e-12, base.shape)
        return results

    def generate_spectral_mixup(self, fault_windows: np.ndarray, n: int,
                                rng: np.random.Generator, noise_frac: float = 0.10,
                                jitter_smp: int = 8, harmonic_inject: float = 0.0,
                                n_harm: int = 3) -> np.ndarray:
        """Trộn trong MIỀN MAGNITUDE PHỔ + bơm hài vật lý (phiên bản reviewer đề xuất).

        Cách làm (từng bản):
          1. Lấy 2 cửa sổ lỗi thật, FFT.
          2. Trộn MAGNITUDE (độc lập phase): |S_new| = a|S1| + (1-a)|S2|.
             Kể cả khi x1, x2 lệch pha, magnitude KHÔNG triệt tiêu → sóng mang giữ trọn.
          3. Phase: lấy phase của cửa sổ CÓ NĂNG LƯỢNG CAO hơn (coherent source) — nhưng
             để tránh gọi sự phụ thuộc pha, dùng phase của bản nào coi là "thân" hơn bằng
             cách chọn phase theo tỉ trọng (weighted phase). Nếu không, chỉ cần phase của
             cửa sổ thứ nhất (vẫn tạo ra tín hiệu thực vì magnitude ≥ 0).
          4. iFFT → tín hiệu thời gian.
          5. [tuỳ chọn] bơm hài vật lý tại k*f_char để tăng sắc nét thành phần lỗi.
          6. thêm nhiễu + jitter như interpolate.

        Vì magnitude giữ nguyên toàn bộ "độ mạnh" của sóng mang (không triệt tiêu),
        bản sinh có std/env_energy SÁT lỗi thật — mạnh hơn interpolate thuần.
        `harmonic_inject` (0..1): phần magnit lại thêm vào các bin k*f_char (nếu 0 thì
        tắt). Nhớ rằng S trong rfft đã chứa mọi bin; chỉ boost bin gần k*f_char.
        """
        fw = np.asarray(fault_windows, dtype=np.float64)
        n_out = int(n)
        win = fw.shape[1]
        Fs = np.abs(np.fft.rfft(fw, axis=1))            # (N, F) magnitude
        Ph = np.angle(np.fft.rfft(fw, axis=1))          # (N, F) phase
        freqs = np.fft.rfftfreq(win, 1.0 / self.fs)
        results = np.empty((n_out, win))
        for i in range(n_out):
            if len(fw) >= 2:
                a = rng.random()
                i0, j0 = rng.integers(0, len(fw), size=2)
                # trộn magnitude (không triệt tiêu) + weighted phase (coherent)
                M = a * Fs[i0] + (1 - a) * Fs[j0]
                P = a * Ph[i0] + (1 - a) * Ph[j0]
                if harmonic_inject > 0:
                    fc = getattr(self, "f_char", None)
                    if fc:
                        for h in range(1, n_harm + 1):
                            jb = int(np.argmin(np.abs(freqs - h * fc)))
                            M[jb] *= (1 + harmonic_inject)
                S = M * np.exp(1j * P)
                base = np.fft.irfft(S, n=win)
            else:
                base = fw[0].copy()
            sh = rng.integers(-jitter_smp, jitter_smp + 1)
            if sh:
                base = np.roll(base, sh)
            results[i] = base + rng.normal(0, noise_frac * base.std() + 1e-12, base.shape)
        return results

    @staticmethod
    def _impact(length: int, res_hz: float, fs: float) -> np.ndarray:
        """Mẫu xung va đập: hình sin TẦN SỐ CỘNG HƯỞNG tắt dần nhanh.

        Lỗi ổ bi kích thích cộng hưởng cơ khí (hàng trăm Hz – kHz), không phải tần số
        va đập thấp. Tần số `res_hz` (IMS ~3000Hz, xem `generate`) chính là dải năng
        lượng mà lỗi thật đẩy lên → sau khi bơm, phổ dịch LÊN giống lỗi thật.
        """
        n = max(4, int(length))
        t = np.arange(n)
        decay = np.exp(-t / max(1, n * 0.12))                  # tắt nhanh như va đập
        omega = 2 * np.pi * res_hz / fs
        return np.sin(omega * t) * decay

    # ------------------------------------------------------------------ #
    #  BƯỚC 3: ECDF/KS DISTRIBUTION ALIGNMENT (sim-to-real gap)
    # ------------------------------------------------------------------ #
    def _descriptors(self, windows: np.ndarray, f_char: float) -> np.ndarray:
        """7 descriptor "task-agnostic" so sánh phân bố synthetic vs REAL (Ren 2026).

        Trả về (n, 7): std, rms, kurtosis, spec_centroid, env_energy,
        Aline@f_char (biên độ envelope tần số va đập), sideband_snr.
        """
        from src import features as ft
        w = np.asarray(windows, dtype=np.float64)
        win = w.shape[1]
        rf = ft.raw_features(w, fs=self.fs)
        s = np.std(w, axis=1)
        rms = np.sqrt(np.mean(w ** 2, axis=1))
        kur, cen, env_e = rf[:, 6], rf[:, 10], rf[:, 17]
        Xc = w - w.mean(axis=1, keepdims=True)
        amp = np.abs(hilbert(Xc, axis=1))
        amp = amp - amp.mean(axis=1, keepdims=True)
        Xf = np.abs(rfft(amp, axis=1)) ** 2
        fr = rfftfreq(win, 1.0 / self.fs)
        jc = int(np.argmin(np.abs(fr - f_char)))
        Al = np.sqrt(Xf[:, jc]) / (rms + 1e-12)
        band = np.abs(fr - f_char) < 0.1 * f_char
        nb = np.abs(fr - 2 * f_char) < 0.1 * f_char
        snr = Xf[:, band].mean(axis=1) / (Xf[:, nb].mean(axis=1) + 1e-12)
        return np.stack([s, rms, kur, cen, env_e, Al, snr], axis=1)

    def generate_aligned(self, fault_windows: np.ndarray, normal_windows: np.ndarray,
                         n: int, rng: np.random.Generator, oversample: int = 25,
                         p_band: tuple[float, float] = (0.05, 0.95),
                         no_align: bool = False):
        """Sinh lỗi giả rồi CĂN CHỈNH PHÂN BỐ (ECDF) với lỗi thật.

        Ý TƯỞNG (paper Ren 2026 — "simulation-augmented framework", ECDF/KS alignment):
        bộ sinh physics có "sim-to-real gap" (tần số/biên độ/sideband lệch — SPEC
        §3.4.2). Thay vì sinh blind rồi ném vào model, ta:
          1) Tạo POOL = physics injection (đa dạng biến thể) TRỘN interpolate
             (bám sát manifold lỗi thật — best theo SPEC §3.4).
          2) Đo 7 descriptor của từng bản pool.
          3) Với mỗi descriptor, loại các bản nằm NGOÀI dải phân vị [p_band] của
             lỗi THẬT (bản phi vật lý: quá nhọn/quá to/thiếu sideband).
          4) Nếu đủ `n` bản hợp lệ thì dùng chúng; nếu còn thiếu, bổ sung interpolate
             (chắc chắn bám manifold).

        Trả về (n, win) — tập bản lỗi giả có PHÂN BỐ descriptor khớp lỗi thật.
        `no_align=True` trả POOL THÔ (bỏ qua lọc) để đo hiệu quả của alignment.
        """
        if not self._calibrated or self.f_char is None:
            raise RuntimeError("Gọi calibrate() trước khi generate_aligned().")
        fw = np.asarray(fault_windows, dtype=np.float64)
        nw = np.asarray(normal_windows, dtype=np.float64)
        f_char, n_out = self.f_char, int(n)
        nf = int(oversample) * n_out
        pool = np.concatenate([self.generate(nw, nf, rng),        # physics (cần nền normal)
                               self.generate_interpolated(fw, nf, rng)])   # manifold
        if no_align:
            return pool[:n_out]
        Dm = self._descriptors(pool, f_char)
        Dr = self._descriptors(fw, f_char)
        lo = np.quantile(Dr, min(p_band), axis=0)
        hi = np.quantile(Dr, max(p_band), axis=0)
        keep = np.all((Dm >= lo) & (Dm <= hi), axis=1)
        cand = pool[keep] if keep.any() else pool
        if len(cand) < n_out:
            extra = self.generate_interpolated(fw, n_out - len(cand), rng)
            cand = np.concatenate([cand, extra])
        return cand[:n_out]

    def generate_physics_mixup(self, fault_windows: np.ndarray, normal_windows: np.ndarray,
                               n: int, rng: np.random.Generator, phys_frac: float = 0.30,
                               noise_frac: float = 0.10, jitter_smp: int = 8) -> np.ndarray:
        """interp_align NHƯNG trộn cả lỗi PHYSICS-INJECTION vào pool đầu vào.

        Lỗi IMS test chứa nhiều CHẾ ĐỘ hỏng (không chỉ BPFO) mà interpolation thuần
        (trộn lỗi thật) có thể phủ THIẾU (nó chỉ bám các mẫu có trong train). Physics
        bơm xung theo tần số đặc trưng học được → bổ sung biến thể cấu trúc lỗi khác.
        Ý TƯỞNG: với mỗi cửa sổ sinh, với xác suất `phys_frac` lấy cửa sổ lỗi thật từ
        POOL TRỘN (physics + thật), còn lại vẫn trộn lỗi thật thuần (như interp_align).

        QUAN TRỌNG (anti-bug kiểu subsample): `phys_frac` chỉ trộn vào pool, KHÔNG làm
        rơi lỗi thật — mọi cửa sổ sinh đều là tổ hợp tuyến tính của các cửa sổ lỗi
        (thật + physics), không phụ thuộc cap. Physics phải KHÔNG tính 'true' như
        lỗi thật thật sự, nhưng vẫn trong manifold gần lỗi.

        Cần `normal_windows` (nền để physics injection). Bộ sinh `generate` đã calibrate
        (f_char sẵn trước).
        """
        if not self._calibrated or self.f_char is None:
            raise RuntimeError("Gọi calibrate() trước khi generate_physics_mixup().")
        fw = np.asarray(fault_windows, dtype=np.float64)
        nw = np.asarray(normal_windows, dtype=np.float64)
        n_out = int(n)
        # pool head: sinh physics đủ để trộn (mới cần nền, số ít redundant)
        n_phys = max(1, int(n_out * phys_frac))
        phys = self.generate(nw, n_phys, rng)
        pool = np.concatenate([fw, phys])           # lỗi thật + lỗi physics trộn chung
        results = np.empty((n_out, fw.shape[1]))
        for i in range(n_out):
            # 2 cửa sổ: với prob phys_frac, ít nhất 1 cửa sổ từ 'physics'; kèm cửa sổ thật
            p = rng.random()
            if p < phys_frac:
                i0 = rng.integers(len(fw), len(fw) + len(phys))  # lấy từ PHYSICS (sau thật)
            else:
                i0 = rng.integers(0, len(fw))
            j0 = rng.integers(0, len(pool))
            x1, x2 = pool[i0], pool[j0]
            a = rng.random()
            k = self._best_corr_shift(x2, x1)
            x2a = np.roll(x2, k) if k else x2
            base = a * x1 + (1 - a) * x2a
            sh = rng.integers(-jitter_smp, jitter_smp + 1)
            if sh:
                base = np.roll(base, sh)
            results[i] = base + rng.normal(0, noise_frac * base.std() + 1e-12, base.shape)
        return results



__all__ = ["OptimizedFaultGenerator"]
