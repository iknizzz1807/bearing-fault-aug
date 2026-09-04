"""contrastive.py — Encoder self-supervised (SimCLR-style) cho tín hiệu rung.

Đây là hướng C trong SPEC.md: dùng nhóm model mà research ghi nhận *vượt trội*
data-augmentation (+21% theo các paper gần đây) — học biểu diễn KHÔNG CẦN NHÃN rồi
mới dùng ít nhãn lỗi. Mục đích: đối chiếu "model mới" với "sinh dữ liệu lỗi"
(interpolation/heuristic/physics/TimeGAN) ở CÙNG mức khan hiếm.

Nguyên tắc (SimCLR, Chen et al. 2020):
  1. Mỗi cửa sổ rung → tạo 2 "view" khác nhau (augment: nhiễu nhẹ + lệch pha thời gian).
  2. Encoder (1D-CNN) map cả hai view → embedding.
  3. NT-Xent loss: kéo hai view của CÙNG một mẫu LẠI gần nhau, đẩy các mẫu KHÁC nhau
     ra xa. Học được embedding mà các cửa sổ giống "kiểu" (normal vs fault) tụ thành cụm.
  4. Sau pretrain: lấy embedding làm feature → train classifier với VÀI lỗi thật.

VÌ SAO KHÔNG DÙNG FOUNDATION MODEL LỚN (Chronos/RmGPT):
  - Chronos là forecasting, không phải classification; RmGPT/YOTOnet repo không có code
    (xem SPEC §research). Cài transformers + tải model lớn vừa rủi ro phiên bản vừa lệch
    định hướng bài (phân loại normal/fault, không dự báo chuỗi tương lai).
  - Encoder contrastive nhỏ tự viết: chạy được trên GPU, không cần net, đo được ngay.

Lưu ý tín hiệu rung: augmentation phải NHẸ để không phá đặc trưng lỗi (chỉ nhiễu + lệch pha
nhỏ). Lấy 1 khác lệch pha lớn (roll) sẽ biến "fault" thành "normal" — sai. Dùng noise nhỏ +
jitter pha ±vài mẫu.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_views(windows: np.ndarray, rng: np.random.Generator,
                  noise: float = 0.05, shift: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Tạo 2 view (augment nhẹ) cho mỗi cửa sổ rung — dùng cho contrastive pretrain.

    Positive pair = 2 view của cùng một cửa sổ. Augment chỉ NHẸ:
      - nhiễu Gaussian biên độ nhỏ (noise × std tín hiệu)
      - lệch pha thời gian ±`shift` mẫu (mô phỏng lệch điểm bắt đầu của cửa sổ)
    Không dùng scale/roll lớn vì sẽ làm "lỗi" nghe như "thường".
    """
    n = len(windows)
    v1 = np.empty_like(windows)
    v2 = np.empty_like(windows)
    for i in range(n):
        s = windows[i]
        v1[i] = s + rng.normal(0, noise * s.std() + 1e-8, s.shape)
        sh = rng.integers(-shift, shift + 1)
        v2[i] = np.roll(s, sh) + rng.normal(0, noise * s.std() + 1e-8, s.shape)
    return v1, v2


class Encoder1D(nn.Module):
    """Encoder 1D-CNN: (B, 1, L) -> (B, 256). Nhận các kênh tần số + thời gian."""

    def __init__(self, in_channels: int = 1, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, x):                       # x: (B, C, L)
        h = self.features(x).squeeze(-1)        # (B, 128)
        return self.proj(h)


class ContrastiveEncoder:
    """Wrapper: pretrain SimCLR trên window rung rồi extract embedding.

    API:
      model = ContrastiveEncoder(feat_dim=..., seq_len=256)
      model.pretrain(windows, epochs, batch)     # học không cần nhãn
      emb = model.encode(windows)                # (n, out_dim) embedding
    """

    def __init__(self, seq_len: int = 256, out_dim: int = 128, temperature: float = 0.1,
                 lr: float = 1e-3, seed: int = 0):
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.temperature = temperature
        self.lr = lr
        self.seed = seed
        self.net = Encoder1D(in_channels=1, out_dim=out_dim).to(DEVICE)

    @staticmethod
    def _nt_xent(z1, z2, temperature, batch_size):
        """Normalized Temperature-scaled Cross Entropy (SimCLR loss)."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)                 # (2B, D)
        logits = (z @ z.T) / temperature               # similarity matrix
        # mask loại bỏ chính nó
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, -1e9)
        positives = torch.cat([torch.arange(batch_size, 2 * batch_size),
                               torch.arange(0, batch_size)], dim=0).to(z.device)
        loss = F.cross_entropy(logits, positives)
        return loss

    def pretrain(self, windows: np.ndarray, epochs: int = 20, batch_size: int = 256,
                 lr: float | None = None, log_every: int = 0) -> dict:
        """Pretrain SimCLR trên mảng (n, seq_len). Trả về loss cuối mỗi epoch."""
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr or self.lr)
        n = len(windows)
        history = []
        print(f"[contrastive] pretrain trên {n} cửa sổ (không nhãn) — {epochs} epochs")
        for ep in range(1, epochs + 1):
            self.net.train()
            idx = rng.permutation(n)
            total, nb = 0.0, 0
            for i in range(0, n, batch_size):
                b = idx[i:i + batch_size]
                xw = windows[b]
                v1, v2 = augment_views(xw, rng)
                z1 = self.net(torch.as_tensor(v1[:, None], dtype=torch.float32, device=DEVICE))
                z2 = self.net(torch.as_tensor(v2[:, None], dtype=torch.float32, device=DEVICE))
                loss = self._nt_xent(z1, z2, self.temperature, len(b))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item(); nb += 1
            avg = total / max(1, nb)
            history.append(avg)
            if log_every and ep % log_every == 0:
                print(f"    epoch {ep}/{epochs} | nt_xent={avg:.4f}")
        return {"loss_history": history}

    @torch.no_grad()
    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Trả về embedding (n, out_dim) cho các cửa sổ — dùng làm feature cho classifier."""
        self.net.eval()
        out = []
        for i in range(0, len(windows), 512):
            b = windows[i:i + 512]
            z = self.net(torch.as_tensor(b[:, None], dtype=torch.float32, device=DEVICE))
            out.append(z.cpu().numpy())
        return np.concatenate(out, axis=0)


__all__ = ["ContrastiveEncoder", "Encoder1D", "augment_views"]
