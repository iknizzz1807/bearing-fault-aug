"""physics.py — Sinh dữ liệu lỗi ổ bi bằng MÔ PHỎNG VẬT LÝ (fault injection).

Ý tưởng (hướng 2 trong SPEC.md): thay vì để GAN học rồi sinh lỗi giả (khó kiểm
soát, hay "nát" như TimeGAN), mình dùng CÔNG THỨC vật lý ổ bi lăn để chế tín hiệu
lỗi giả ĐÚNG tần số đặc trưng:

  x_f(t) = x_h(t) + A * Σ_k h(t - k*T) * w(t) + n(t)

Trong đó:
  - x_f(t): tín hiệu lỗi giả
  - x_h(t): tín hiệu KHỎE thật (cũng chính là nền rung bình thường)
  - h(t)  : mẫu "va đập xung" (impulse) tại mỗi lần hạt lăn đập qua vết nứt
  - T     : chu kỳ giữa 2 lần va đập = 1 / f_fault (f_fault lấy theo BPFO/BPFI/BSF)
  - w(t)  : điều chế theo vòng quay (xung chỉ mạnh khi vết nứt vào vùng tải)
  - n(t)  : nhiễu ngẫu nhiên nhỏ (bảo đảm tín hiệu không lặp y hệt)

Các tần số đặc trưng ổ bi (công thức chuẩn từ chẩn đoán rung):
  BPFO = n/2 * fr * (1 - d/D * cos(phi))       # Ngoài lần (outer race)
  BPFI = n/2 * fr * (1 + d/D * cos(phi))       # Trong lần (inner race)
  BSF  = D/(2d) * fr * (1 - (d/D*cos(phi))^2)  # Hạt lăn (ball spin)
  FTF  = fr/2 * (1 - d/D * cos(phi))           # Lồng (train cage)
với n = số hạt lăn, d = đường kính hạt, D = đường kính vòng, phi = góc tiếp xúc,
    fr = tốc độ quay trục (Hz).

Giải thích ngắn gọn để trình bày trong slide:
  - Một ổ bi hỏng sinh ra "nhịp va đập" lặp lại ở tần số cố định (~BPFO/BPFI/BSF).
  - Tín hiệu rung đo được = xung đó + nền rung bình thường + nhiễu cảm biến.
  - Mình "bơm" đúng nhịp đó vào tín hiệu khỏe → có mẫu lỗi giả đúng vật lý, không
    cần dữ liệu lỗi thật. Đây là cách tạo dữ liệu lỗi khi dữ liệu thật khan hiếm.

VÌ SAO CHỌN PHYSICS LÀ MỘT OPTION (và đặt nó CẠNH các leverage khác)?
  Đề A1 đưa ra nhóm công nghệ: "Mô phỏng vật lý · GAN/TimeGAN · Diffusion ·
  fault injection" (xem SPEC.md §1). Đây là một trong các cách sinh lỗi giả.
  Ba hướng đã thử & so sánh:
    - heuristic (script 02): biến đổi biên độ + nhiễu — nhanh, nhưng "lỗi giả"
      không có cấu trúc tần số vật lý → gần như vô nghĩa về mặt cơ chế.
    - TimeGAN generative (script 03): học phân bố lỗi thật — nhưng không kiểm
      soát được hình dạng lỗi, dễ mode collapse/nhiễu, chậm, cần GPU.
    - Diffusion: tự viết nhẹ (src/ddpm.py) — nhưng generative kém ở khan hiếm.
    - PHYSICS (file này): dùng CÔNG THỨC vật lý ổ bi lăn → lỗi giả KIỂM SOÁT
      ĐƯỢC, ĐÚNG VẬT LÝ (tần số BPFO/BPFI/BSF/FTF), không bị mode collapse, tái
      lập nhờ seed.

  Vì sao physics đáng làm "lever chính" để so sánh trước/sau augmentation?
    Physics cho phép kiểm soát chính xác tần số, pha, biên độ của lỗi → biết rõ
    mình đang "bơm cái gì" vào dữ liệu. Nếu kết quả không đổi thì kết luận
    "augmentation không giúp" có căn cứ vững hơn so với heuristic (bơm bừa)
    hay GAN (khó giải thích phân bố học được).
"""

from __future__ import annotations

import numpy as np


def bearing_geometry(n_balls: int = 8, ball_diameter: float = 0.0150,
                     pitch_diameter: float = 0.0650, contact_angle: float = 0.0,
                     **_) -> dict:
    """Tham số hình học ổ bi lăn (mặc định: ổ bi tiêu biểu, đơn vị mét).

    Trả về dict có các trường dùng cho công thức tần số. Nhớ: `contact_angle`
    đầu vào là ĐỘ, bên trong quy về radian cho cos().
    """
    return {
        "n_balls": int(n_balls),
        "ball_diameter": float(ball_diameter),
        "pitch_diameter": float(pitch_diameter),
        "contact_angle": float(np.deg2rad(contact_angle)),
    }


def bpfo(fr: float, geo: dict) -> float:
    """Ball Pass Frequency Outer — tần số hạt lăn đập vết nứt ngoài."""
    return geo["n_balls"] / 2.0 * fr * (1 - geo["ball_diameter"] / geo["pitch_diameter"]
                                        * np.cos(geo["contact_angle"]))


def bpfi(fr: float, geo: dict) -> float:
    """Ball Pass Frequency Inner — tần số hạt lăn đập vết nứt trong."""
    return geo["n_balls"] / 2.0 * fr * (1 + geo["ball_diameter"] / geo["pitch_diameter"]
                                        * np.cos(geo["contact_angle"]))


def bsf(fr: float, geo: dict) -> float:
    """Ball Spin Frequency — tần số quay của hạt lăn."""
    ratio = geo["ball_diameter"] / geo["pitch_diameter"] * np.cos(geo["contact_angle"])
    return geo["pitch_diameter"] / (2 * geo["ball_diameter"]) * fr * (1 - ratio**2)


def ftf(fr: float, geo: dict) -> float:
    """Fundamental Train Frequency — tần số quay của lồng giữ hạt."""
    return fr / 2.0 * (1 - geo["ball_diameter"] / geo["pitch_diameter"]
                       * np.cos(geo["contact_angle"]))


def _impact(train_len: int, pulse_frac: float = 0.08) -> np.ndarray:
    """Mẫu xung va đập đơn (hình sin giảm dần) độ dài `train_len`*pulse_frac.

    Mỗi lần hạt lăn đập vết nứt sinh ra 1 xung ngắn có tần số dao động RẤT CAO
    (vài kHz một cách tương đối, thường là tần số cộng hưởng của vòng ổ). Ta mô
    hình giản đơn bằng sin giảm dần có tần số quanh 5% lần tần số lấy mẫu.
    """
    n = max(8, int(train_len * pulse_frac))
    t = np.arange(n)
    freq_ratio = 0.05                       # tần số dao động ≈ 5% fs
    decay = np.exp(-t / (n * 0.15))         # tắt dần nhanh như xung va đập
    return np.sin(2 * np.pi * freq_ratio * t) * decay


def generate_synthetic_fault(healthy: np.ndarray, freq_fault: float, fs: float,
                             depth: float = 0.30, modulation: bool = True,
                             noise: float = 0.10, pulse_frac: float = 0.08,
                             seed: int = 0) -> np.ndarray:
    """Sinh tín hiệu lỗi giả từ tín hiệu khỏe bằng mô hình xung lặp.

    x_f(t) = x_h(t) + A * Σ_k h(t - k*T) * w(t) + n(t)

    Tham số:
      healthy   : tín hiệu KHỎE (nền rung thật), mảng 1D.
      freq_fault: tần số lỗi (Hz) — lấy từ bpfo/bpfi/bsf.
      fs        : tần số lấy mẫu (Hz).
      depth     : A — biên độ xung va đập (tỉ lệ với std của tín hiệu khỏe).
      modulation: có nhân envelope vòng quay (xung chỉ mạnh trong vùng tải) không.
      noise     : biên độ nhiễu ngẫu nhiên thêm vào (tỉ lệ std).
      pulse_frac: chiều dài xung (phần của 1 chu kỳ lỗi).
      seed      : hạt ngẫu nhiên cho nhiễu (tái lập).

    Trả về: mảng cùng shape với `healthy` — tín hiệu lỗi giả.
    """
    healthy = np.asarray(healthy, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    n = len(healthy)
    amp = depth * (np.std(healthy) + 1e-12)

    # khoảng cách giữa 2 lần va đập (mẫu) = 1 period của tần số lỗi
    gap = max(2, int(round(fs / max(freq_fault, 1e-9))))
    x = np.zeros(n)
    n_imp = int(np.ceil(n / gap))
    imp = _impact(gap, pulse_frac)
    for k in range(n_imp):
        start = k * gap
        end = start + len(imp)
        if end > n:
            break
        x[start:end] += imp[:end - start]

    # điều chế theo vòng quay: xung mạnh hơn khi vết nứt đi vào vùng tải (1 lần/quay)
    if modulation:
        # 1 vòng quay = fs / fr? Ta dùng tần số lỗi làm mốc: envelope chậm hơn.
        # Đơn giản: nhân sin(chậm) quanh 1 chu kỳ quay — ước lượng fr từ freq_fault.
        # Nếu có tần số vòng quay riêng thì truyền thêm; ở đây chấp nhận heuristic.
        mod_period = max(gap * 2, int(fs / 5))   # 2 lần chu kỳ lỗi ≈ vùng tải
        t = np.arange(n)
        env = 0.55 + 0.45 * np.cos(2 * np.pi * t / mod_period)
        x = x * env

    x += rng.normal(0, noise * np.std(healthy) + 1e-12, n)   # nhiễu (không lặp y hệt)
    return healthy + amp * x


def physics_aug_worker(healthy: np.ndarray, fault_type: str, fs: float,
                       rotation_hz: float, geometry: dict, rng: np.random.Generator,
                       n_copy: int = 1) -> np.ndarray:
    """Hàm tiện: từ 1 tín hiệu khỏe sinh `n_copy` bản lỗi giả (an toàn, tái lập).

    `fault_type` chọn tần số: 'bpfo' | 'bpfi' | 'bsf' | 'ftf'.
    `rng`: numpy Generator (bên ngoài cấp để cả toàn bộ quá trình cùng 1 seed quyết định).
    Trả về mảng (n_copy, len(healthy)) — mỗi hàng 1 bản lỗi giả, với tần số pha lỗi
    nhấp nhô nhẹ để đa dạng (bắt chước độ trượt thực tế của ổ bi).
    """
    f_fault = {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}[fault_type](
        rotation_hz, geometry)
    out = []
    for i in range(n_copy):
        # dịch nhẹ tần số mỗi bản (~±2%) → đa dạng mà vẫn đúng bản chất
        jitter = f_fault * (1 + rng.uniform(-0.02, 0.02))
        out.append(generate_synthetic_fault(healthy, jitter, fs, seed=i))
    return np.asarray(out)


__all__ = [
    "bearing_geometry", "bpfo", "bpfi", "bsf", "ftf",
    "generate_synthetic_fault", "physics_aug_worker",
]
