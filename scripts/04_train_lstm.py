#!/usr/bin/env python3
"""SCRIPT 4 — Huấn luyện model THỜI GIAN (LSTM / 1D-CNN) trên WINDOW THÔ.

╔══════════════════════════════════════════════════════════════════════╗
║  SCRIPT NÀY CHẠY CÁI GÌ? (tóm tắt 1 đoạn)                            ║
║  Đây là hướng "DEEP TEMPORAL" — một OPTION song song với "RF trên      ║
║  feature" (script 01/02). Khác biệt ở "ai bắt tính năng":              ║
║     - RF-on-features: con NGƯỜI thiết kế feature thủ công (16–24 cột).  ║
║       Điểm yếu: feature thủ công "nghiền nát" cấu trúc thời gian trong  ║
║       cửa sổ — mất nhịp, mất sóng, mất xung va đập của lỗi.             ║
║     - LSTM/1D-CNN: để MẠNG tự học cấu trúc thời gian trên tín hiệu thô  ║
║       (mỗi mẫu = 256 điểm rung) → giữ nguyên nhịp/sóng/xung.            ║
║  Script này kiểm tra: "deep temporal có mạnh hơn RF trên feature?"       ║
║                                                                       ║
║  ĐÂU LÀ DỮ LIỆU: windows thô, chia GIỐNG pipeline (3-zone chống        ║
║  leakage): test normal KHỎE chưa thấy + lỗi THẬT chưa thấy.            ║
║  Không dùng lỗi giả — mục tiêu đo mỗi "model temporal" thuần.           ║
║                                                                       ║
║  KẾT QUẢ THỰC (xem results/compare_lstm.json & compare_cnn.json):      ║
║    LSTM : recall ~0.426 / AUC ~0.582.                                   ║
║    CNN  : recall ~0.443 / AUC ~0.588.                                   ║
║  CẢ HAI đều KÉM RF (recall ~0.886 / AUC ~0.982). Kết luận: deep         ║
║  temporal KHÔNG thắng RF-on-features trên dữ liệu này → RF vẫn là        ║
║  trụ cột (SPEC.md §3.1). Script được giữ như một experiment/deliverable. ║
║                                                                       ║
║  ĐẦU RA: results/compare_lstm.json                                     ║
╚══════════════════════════════════════════════════════════════════════╝

CÁCH CHẠY:
    python scripts/04_train_lstm.py --dataset ims --win 256 --epochs 15 --model lstm
    python scripts/04_train_lstm.py --dataset ims --model cnn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as ds               # noqa: E402
from src import pipeline as pipe         # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows_per_run(runs: list[list[np.ndarray]], win: int, stride: int) -> list[np.ndarray]:
    """Với MỖI test-run, cắt windows thô (n_run, win) — giữ thứ tự thời gian."""
    out = []
    for run in runs:
        ws = []
        for sig in run:
            sig = np.asarray(sig, dtype=np.float32).reshape(-1)
            w = ds.make_windows(sig, win=win, stride=stride)
            if len(w):
                ws.append(w)
        if ws:
            out.append(np.concatenate(ws))
    return out


def split_windows(runs_normal: list[list[np.ndarray]],
                  runs_fault: list[list[np.ndarray]],
                  win: int, stride: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia windows thô theo đúng quy ước pipeline: TRAIN / TEST / GREY.

    Trả về (Xn_train, y_base[0/1]... ) — gọn: X_train, y_train, X_test, y_test.
    """
    # normal: 3-zone trong mỗi run
    normal_runs_w = make_windows_per_run(runs_normal, win, stride)
    Xn_train_w, Xn_test_w = [], []
    for w in normal_runs_w:
        i_tr = int(len(w) * pipe.NORMAL_TRAIN_END)
        i_te = int(len(w) * pipe.NORMAL_TEST_END)
        Xn_train_w.append(w[:i_tr])
        Xn_test_w.append(w[i_tr:i_te])
    Xn_train = np.concatenate(Xn_train_w) if Xn_train_w else np.empty((0, win))
    Xn_test = np.concatenate(Xn_test_w) if Xn_test_w else np.empty((0, win))

    # fault: shuffle thì trộn 20/80
    Xf = np.concatenate(make_windows_per_run(runs_fault, win, stride), axis=0) \
        if len(runs_fault) and make_windows_per_run(runs_fault, win, stride) \
        else np.empty((0, win))
    if len(Xf):
        rng.shuffle(Xf)
        n_tr = max(1, int(0.2 * len(Xf)))
        Xf_train_w, Xf_test_w = Xf[:n_tr], Xf[n_tr:]
    else:
        Xf_train_w, Xf_test_w = np.empty((0, win)), np.empty((0, win))

    X_train = np.concatenate([Xn_train, Xf_train_w])
    y_train = np.concatenate([np.zeros(len(Xn_train)), np.ones(len(Xf_train_w))])
    X_test = np.concatenate([Xn_test, Xf_test_w])
    y_test = np.concatenate([np.zeros(len(Xn_test)), np.ones(len(Xf_test_w))])
    return X_train, y_train, X_test, y_test


class TimeNet(nn.Module):
    """LSTM hoặc 1D-CNN nhận (batch, 1, win) → 2 logit (normal/fault)."""

    def __init__(self, model: str, win: int, num_classes: int = 2):
        super().__init__()
        if model == "lstm":
            self.enc = nn.Sequential(
                nn.Conv1d(1, 8, 5, padding=2), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(8, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            )
            # sau 2 lần pool → độ dài L = (win//4); LSTM học sequence trên đó
            self.rnn = nn.LSTM(16, 32, num_layers=1, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, num_classes))
        else:  # cnn
            self.enc = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)                      # (B, C, L)
        if hasattr(self, "rnn"):
            x = x.permute(0, 2, 1)           # (B, L, C)
            x, _ = self.rnn(x)
            x = x.mean(dim=1)                # (B, 64)
        else:
            x = x.flatten(1)                 # (B, 64)
        return self.head(x)


def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
                epochs: int, bs: int, lr: float, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    X = torch.as_tensor(X[:, None, :], dtype=torch.float32, device=DEVICE)  # (B,1,win)
    y = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
    n = len(X)
    model.train()
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            out = model(X[b])
            loss = lossf(out, y[b])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, bs: int = 512) -> np.ndarray:
    model.eval()
    X = torch.as_tensor(X[:, None, :], dtype=torch.float32, device=DEVICE)
    out = []
    for i in range(0, len(X), bs):
        p = torch.softmax(model(X[i:i + bs]), dim=1)[:, 1].cpu().numpy()
        out.append(p)
    return np.concatenate(out)


def score(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    proba = predict_proba(model, X)
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "auc": float(roc_auc_score(y, proba))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--model", choices=["lstm", "cnn"], default="lstm")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(RESULTS / "compare_lstm.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    RESULTS.mkdir(exist_ok=True)

    print(f"[1] load + chia windows thô ({args.dataset}, win={args.win}) ...")
    runs_n, runs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    X_train, y_train, X_test, y_test = split_windows(runs_n, runs_f, args.win, args.stride, rng)
    print(f"    train: normal={int((y_train==0).sum())} fault={int((y_train==1).sum())} | "
          f"test: normal={int((y_test==0).sum())} fault={int((y_test==1).sum())} | "
          f"shape {X_train.shape}")

    print(f"[2] train {args.model} ({args.epochs} epochs, device={DEVICE}) ...")
    model = TimeNet(args.model, args.win).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"    params={n_param}")
    model = train_model(model, X_train, y_train, args.epochs, args.batch, args.lr)

    print("[3] đánh giá trên test (normal khỏe chưa thấy + lỗi thật chưa thấy) ...")
    r = score(model, X_test, y_test)
    print("=" * 50)
    print(f"  MODEL TEMPORAL ({args.model.upper()}) — feature THÔ, không augmentation:")
    print(f"    precision={r['precision']:.4f}  recall={r['recall']:.4f}  "
          f"f1={r['f1']:.4f}  auc={r['auc']:.4f}")
    print("=" * 50)
    print("\n  So với RF trên feature thủ công (trước augment):")
    print("    RF-16feat : recall 0.477  auc 0.582")
    print("    RF-24feat : recall 0.479  auc 0.585")
    print("    Đây là mốc — temporal phải vượt đây mới đáng đầu tư.")

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "model": args.model, "win": args.win,
                   "epochs": args.epochs, "params": n_param, **r}, f, indent=2)


if __name__ == "__main__":
    main()
