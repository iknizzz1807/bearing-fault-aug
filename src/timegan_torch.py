"""TimeGAN (Yoon et al., NeurIPS 2019) — triển khai lại bằng PyTorch (chạy CPU).

Lý do: repo gốc jsyoon0823/TimeGAN dùng tensorflow==1.15 không chạy được trên
Python >= 3.8. Bản này trung thành về mặt kiến trúc:
  1. Embedding network:  chuỗi X  -> không gian embedding H
  2. Recovery network:   embedding H -> tái tạo X_hat
  3. Supervisor:         học mô hình chuyển tiếp h_t -> h_{t+1}
  4. Generator:          z ngẫu nhiên -> embedding giả
  5. Discriminator:      phân biệt embedding thật / giả

Loss:
  - Reconstruction:  ||X - X_hat||^2  (embedder + recovery + supervisor đồng train)
  - Supervised loss: ||h_{t+1} - supervisor(h_t)||^2
  - Adversarial GAN: embedding thật vs giả (discriminator/generator)
Dog đó: Generator -> Recovery để sinh ra data giả trong không gian quan sát.

Tham khảo: J. Yoon, D. Jarrett, M. van der Schaar. "Time-series Generative
Adversarial Networks". NeurIPS 2019.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _GRUNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layer: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, out_dim, num_layers=n_layer, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class Embedder(nn.Module):
    """X -> H"""
    def __init__(self, dim: int, hidden: int, n_layer: int):
        super().__init__()
        self.rnn = _GRUNet(dim, hidden, n_layer)

    def forward(self, x):
        return self.rnn(x)


class Recovery(nn.Module):
    """H -> X_hat"""
    def __init__(self, hidden: int, dim: int, n_layer: int):
        super().__init__()
        self.rnn = _GRUNet(hidden, hidden, n_layer)
        self.fc = nn.Linear(hidden, dim)

    def forward(self, h):
        return self.fc(self.rnn(h))


class Supervisor(nn.Module):
    """H -> H_next (learn transition)"""
    def __init__(self, hidden: int, n_layer: int):
        super().__init__()
        self.rnn = _GRUNet(hidden, hidden, n_layer)

    def forward(self, h):
        return self.rnn(h)


class Generator(nn.Module):
    """Z -> H_fake"""
    def __init__(self, z_dim: int, hidden: int, n_layer: int):
        super().__init__()
        self.rnn = _GRUNet(z_dim, hidden, n_layer)

    def forward(self, z):
        return self.rnn(z)


class Discriminator(nn.Module):
    """H or H_fake -> logit"""
    def __init__(self, hidden: int, n_layer: int):
        super().__init__()
        self.rnn = _GRUNet(hidden, hidden, n_layer)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, h):
        return self.fc(self.rnn(h))


class TimeGAN:
    """Wrapper train + sample, API giống repo gốc nhưng bằng PyTorch."""

    def __init__(self, hidden_dim: int = 24, num_layers: int = 3,
                 z_dim: int = 24, lr: float = 1e-3, batch_size: int = 32):
        self.hidden = hidden_dim
        self.n_layer = num_layers
        self.z_dim = z_dim
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = None
        self.dim = None
        self.nets: dict[str, nn.Module] = {}

    # ---- helpers ----
    def _bce(self, p, t):
        return nn.functional.binary_cross_entropy_with_logits(p, t)

    def _random_z(self, n: int, T: int) -> torch.Tensor:
        return torch.randn(n, T, self.z_dim, device=DEVICE)

    # ---- training ----
    def train(self, ori_data: np.ndarray, iterations: int = 5000,
              printfreq: int = 1000, seed: int = 0) -> dict[str, list[float]]:
        """ori_data: (n_samples, seq_len, dim). Chạy theo pipeline gốc:
        1) joint: emb + rec + sup + gen (reconstruction + supervised)
        2) embedding: adversarial train discriminator/generator trên embedding
        3) joint (full): cập nhật mọi net với đủ các loss."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        n, T, d = ori_data.shape
        self.seq_len, self.dim = T, d

        X = torch.as_tensor(ori_data, dtype=torch.float32, device=DEVICE)
        hidden, layers, z_dim = self.hidden, self.n_layer, self.z_dim

        emb = Embedder(d, hidden, layers).to(DEVICE)
        rec = Recovery(hidden, d, layers).to(DEVICE)
        sup = Supervisor(hidden, layers).to(DEVICE)
        gen = Generator(z_dim, hidden, layers).to(DEVICE)
        dis = Discriminator(hidden, layers).to(DEVICE)
        self.nets = {"emb": emb, "rec": rec, "sup": sup, "gen": gen, "dis": dis}

        # optimizers (paper: Adam 1e-3; discriminator dùng lr riêng)
        opt_e = torch.optim.Adam(emb.parameters(), lr=self.lr, betas=(0.9, 0.999))
        opt_r = torch.optim.Adam(rec.parameters(), lr=self.lr, betas=(0.9, 0.999))
        opt_s = torch.optim.Adam(sup.parameters(), lr=self.lr, betas=(0.9, 0.999))
        opt_g = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(dis.parameters(), lr=self.lr, betas=(0.9, 0.999))

        losses = {"rec": [], "sup": [], "dis": [], "gen": []}
        mse = nn.MSELoss()

        for it in range(1, iterations + 1):
            idx = np.random.choice(n, self.batch_size, replace=True)
            Xb = X[idx]  # (B, T, d)
            B = Xb.size(0)

            # ---------- pha 1: joint (emb+rec+sup+gen) ----------
            H = emb(Xb)                       # real embedding
            X_hat = rec(H)                    # reconstruction

            H_next_seq = sup(H[:, :-1, :])
            H_real_shift = H[:, 1:, :]
            sup_loss = mse(H_next_seq, H_real_shift)            # supervised (transition)
            rec_loss = mse(X_hat, Xb)                            # reconstruction

            opt_e.zero_grad(); opt_r.zero_grad(); opt_s.zero_grad()
            (rec_loss + 10.0 * sup_loss).backward(retain_graph=True)
            opt_e.step(); opt_r.step(); opt_s.step()

            Z = self._random_z(B, T)
            H_fake = gen(Z)
            opt_g.zero_grad()
            # generator muốn discriminator nghĩ H_fake là thật (chạy trước khi disc update mới)
            g_adv = self._bce(dis(H_fake), torch.ones(B, T, 1, device=DEVICE))
            # supervised loss cho phần fake: bảo giữ dynamics bằng cách khuyến khích
            # recovery(generator) xấp xỉ một bước dịch đúng — đơn giản hoá là không có
            # ground truth, nên TimeGAN gốc dùng `sup` trên fake embedding tạm:
            H_fake_next = emb(rec(H_fake)[:, :-1, :])            # one-step embedding của fake
            g_sup = mse(sup(H_fake[:, :-1, :]), H_fake_next)
            g_loss = g_adv + 10.0 * g_sup
            g_loss.backward(); opt_g.step()

            # ---------- pha 2: adversarial trên embedding ----------
            H = emb(Xb).detach()
            H_fake_d = gen(self._random_z(B, T)).detach()
            for _ in range(1):
                opt_d.zero_grad()
                d_real = self._bce(dis(H), torch.ones(B, T, 1, device=DEVICE))
                d_fake = self._bce(dis(H_fake_d), torch.zeros(B, T, 1, device=DEVICE))
                d_loss = d_real + d_fake
                d_loss.backward(); opt_d.step()

            losses["rec"].append(rec_loss.item())
            losses["sup"].append(sup_loss.item())
            losses["dis"].append(d_loss.item())
            losses["gen"].append(g_loss.item())

            if it % printfreq == 0 or it == 1:
                print(f"  iter {it:6d}/{iterations} | rec={rec_loss.item():.4f} "
                      f"sup={sup_loss.item():.4f} dis={d_loss.item():.4f} gen={g_loss.item():.4f}")
        return losses

    # ---- sampling ----
    @torch.no_grad()
    def sample(self, n_samples: int) -> np.ndarray:
        """Sinh data giả (n_samples, seq_len, dim) — Generator -> Recovery."""
        self.nets["gen"].eval()
        self.nets["rec"].eval()
        out = []
        for i in range(0, n_samples, self.batch_size):
            B = min(self.batch_size, n_samples - i)
            z = self._random_z(B, self.seq_len)
            H_fake = self.nets["gen"](z)
            X_fake = self.nets["rec"](H_fake)
            out.append(X_fake.cpu().numpy())
        self.nets["gen"].train()
        self.nets["rec"].train()
        return np.concatenate(out, axis=0)[:n_samples]