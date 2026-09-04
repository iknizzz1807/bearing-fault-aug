"""Autoencoder trên feature vector (thay vì dữ liệu thô) cho anomaly detection.

Lý do dùng trên feature: nhỏ, nhanh, chạy CPU được, ổn định với dữ liệu rung công nghiệp.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 32, d_latent: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_latent))
        self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def train_ae(X_norm: np.ndarray, epochs: int = 60, batch_size: int = 128,
             lr: float = 1e-3, seed: int = 0) -> tuple[FeatureAE, np.ndarray]:
    """Train AE chỉ trên data chuẩn. Trả về (model, reconstruction_error)."""
    torch.manual_seed(seed)
    model = FeatureAE(X_norm.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.as_tensor(X_norm, dtype=torch.float32).to(DEVICE)
    n = len(Xt)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1}/{epochs} loss={ep_loss / n:.5f}")

    with torch.no_grad():
        rec_err = ((model(Xt) - Xt) ** 2).mean(dim=1).cpu().numpy()
    return model, rec_err


def threshold_from_err(rec_err: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(rec_err, percentile))


@torch.no_grad()
def rec_error(model: nn.Module, X: np.ndarray) -> np.ndarray:
    Xt = torch.as_tensor(X, dtype=torch.float32).to(DEVICE)
    return ((model(Xt) - Xt) ** 2).mean(dim=1).cpu().numpy()


def report_anomaly(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    """y_true: 0=normal, 1=anomaly(fault). scores càng lớn càng nghi ngờ."""
    pred = (scores > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary")
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    out = {"threshold": float(thr), "precision": float(p), "recall": float(r),
           "f1": float(f1), "auc": float(auc),
           "n_pred_positive": int(pred.sum()), "n_true_positive": int((y_true == 1).sum())}
    return out