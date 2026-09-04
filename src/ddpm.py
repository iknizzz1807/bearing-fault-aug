"""ddpm.py — Denoising Diffusion Probabilistic Model (nhẹ, 1D) sinh tín hiệu rung lỗi.

SOTA 2026 cho sinh dữ liệu lỗi, đối trọng "generative" với interpolate. Mục tiêu: học
phân bố cửa sổ lỗi THẬT (20–50 mẫu) rồi sinh lỗi giả mới. Đây là bản THUẦN torch, chạy
được GPU 6GB, không cần thư viện ngoài. (Bản tự viết — không dùng clone Diffusion-TS.)

CẤU TRÚC:
  - Forward: x_t = sqrt(a_hat_t)*x0 + sqrt(1 - a_hat_t)*eps  (nhiễu cộng dồn tuyến tính).
  - Reverse (nhiễu dự đoán eps-theta): small 1D CNN (hoặc MLP) map (x_t, t) -> eps.
  - train: denoise một step (eps loss trên batch đã noised ngẫu nhiên t).
  - sample: khử nhiễu dần t=T..1, với nhiễu nhỏ (cố định/ngẫu nhiên tùy predict_eps).

CẢNH BÁO (đã biết từ §3.4): diffusion cần NHIỀU mẫu để học phân bố — với 20–50 mẫu rất dễ
mode collapse / học sai như TimeGAN. Ta vẫn chạy để ĐO, không đoán.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _linear_beta_schedule(T: int, lo: float = 1e-4, hi: float = 2e-2) -> torch.Tensor:
    return torch.linspace(lo, hi, T, device=DEVICE)


class TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):                     # t: (B,) hoặc scalar
        t = t.float().view(-1, 1)
        return self.embed(t)


class Denoise1D(nn.Module):
    """Hàm khử nhiễu eps_theta(x_t, t): MLP 1D nhẹ (đủ cho windows 512)."""

    def __init__(self, seq_len: int, time_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.time_emb = TimeEmbed(time_dim)
        self.net = nn.Sequential(
            nn.Linear(seq_len + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, x_t, t):                # x_t: (B, L)
        te = self.time_emb(t)
        h = torch.cat([x_t, te], dim=1)
        return self.net(h)


class LightDDPM:
    """Wrapper nhỏ gọn: train/sample. API giống hệt ContrastiveEncoder để dễ dùng.

      model = LightDDPM(seq_len=512, T=200)
      model.train(X, epochs)
      Xsyn = model.sample(n)
    """

    def __init__(self, seq_len: int = 512, T: int = 200, lr: float = 2e-4, seed: int = 0):
        self.seq_len = seq_len
        self.T = T
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)
        self.den = Denoise1D(seq_len).to(DEVICE)
        betas = _linear_beta_schedule(T)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0).to(DEVICE)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return X / (np.std(X) + 1e-8)

    def train(self, windows: np.ndarray, epochs: int = 200, batch: int = 64,
              log_every: int = 0) -> dict:
        X = np.asarray(windows, dtype=np.float32)
        X = self._normalize(X)
        Xt = torch.as_tensor(X, device=DEVICE)
        opt = torch.optim.Adam(self.den.parameters(), lr=self.lr)
        n = len(Xt)
        rng = np.random.default_rng(self.seed)
        history = []
        print(f"[ddpm] train {n} mẫu lỗi thật, {epochs} epochs, T={self.T}", flush=True)
        for ep in range(1, epochs + 1):
            self.den.train()
            idx = rng.permutation(n)
            total, nb = 0.0, 0
            for i in range(0, n, batch):
                b = idx[i:i + batch]
                x0 = Xt[b]                                   # (B, L)
                B = len(b)
                t = torch.randint(0, self.T, (B,), device=DEVICE)
                eps = torch.randn_like(x0)
                a_hat = self.alpha_hat[t].unsqueeze(1)       # (B,1)
                x_t = torch.sqrt(a_hat) * x0 + torch.sqrt(1 - a_hat) * eps
                eps_pred = self.den(x_t, t.float())
                loss = F.mse_loss(eps_pred, eps)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.den.parameters(), 1.0)
                opt.step()
                total += loss.item(); nb += 1
            avg = total / max(1, nb)
            history.append(avg)
            if log_every and ep % log_every == 0:
                print(f"    epoch {ep}/{epochs} | mse={avg:.5f}", flush=True)
        return {"loss_history": history}

    @torch.no_grad()
    def sample(self, n: int, alpha_std: float = 1.0) -> np.ndarray:
        self.den.eval()
        x = torch.randn(n, self.seq_len, device=DEVICE)
        for t in range(self.T - 1, -1, -1):
            tb = torch.full((n,), t, device=DEVICE)
            eps_pred = self.den(x, tb.float())
            if t > 0:
                z = torch.randn_like(x)
                a_hat_t = self.alpha_hat[t]
                a_hat_prev = self.alpha_hat[t - 1]
                alpha_t = self.alphas[t]
                # posterior: x_{t-1} ~ N(mu, beta_tilde)
                mu = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - a_hat_t) * eps_pred)
                sigma = torch.sqrt(
                    (1 - a_hat_prev) / (1 - a_hat_t) * self.betas[t])
                x = mu + sigma * z
            else:
                a_hat_t = self.alpha_hat[0]
                x = (x - torch.sqrt(1 - a_hat_t) * eps_pred) / torch.sqrt(a_hat_t)
        return x.cpu().numpy() * alpha_std


__all__ = ["LightDDPM"]
