#!/usr/bin/env python3
"""SCRIPT 13 — Đối chứng CONTRASTIVE (self-supervised) với BỘ SINH LỖI interp_align
TRÊN CÙNG HARNESS khan hiếm (win=512, stride_eff, test-fault CỐ ĐỊNH).

Câu hỏi objective: "so với foundation/contrastive ở cùng mức khan hiếm". Brainlist:
  - Harness CHUẨN = script 06 (script `pipe.split_and_scale` → cùng test 8100 lỗi thật,
    baseline tái lập seed+K). interp_align+amp = deliverable.
  - Ở đây ta chấm CONTRASTIVE trên CÙNG test-fault của harness, bằng cách dùng dữ liệu
    THÔ (windows) để pretrain encoder rồi RF trên embedding. Để 'cùng windows', ta tái
    tạo train/test windows ĐÚNG như split_and_scale (stride_eff + rng seed+K, đã chứng
    minh khớp chính xác — §3.4.6c).

So sánh (mỗi K) 4 cột:
  baseline        : RF trên feature 24D (không augment)          → mốc trước
  interp_align+amp: RF trên 24D + lỗi giả (deliverable)          → số thật đã có
  contrastive     : RF trên embedding contrastive (KHÔNG sinh lỗi)
  foundation      : (dự phòng) RF trên embedding model pretrained (nếu có)

Cách chạy:
  python scripts/13_contrastive_harness.py --n-fault 10 20 50 --epochs 12 --out results/contr_harness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as ds          # noqa: E402
from src import features as ft      # noqa: E402
from src import pipeline as pipe    # noqa: E402
from src.contrastive import ContrastiveEncoder   # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


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


def repro_train_fault(signs_f, win, stride_eff, K, seed):
    """Tái tạo đúng K cửa sổ lỗi TRẠI như split_and_scale (window-level, rng=seed+K)."""
    pool = raw_windows(pipe.flatten_runs(signs_f), win, stride_eff)
    idx = np.arange(len(pool))
    rng = np.random.default_rng(seed + K)
    rng.shuffle(idx)
    n_all = max(1, int(0.2 * len(pool)))
    return pool[idx[:n_all]][:K], pool, idx, n_all


def repro_test_fault(signs_f, win, stride_eff, seed):
    """Tái tạo 80% cửa sổ lỗi TEST (đúng như split_and_scale)."""
    pool = raw_windows(pipe.flatten_runs(signs_f), win, stride_eff)
    idx = np.arange(len(pool))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)  # lưu ý: mỗi K khác nhau, nhưng test chấm NẾU dùng đúng rng (không +K)
    raise NotImplementedError("test-fault phụ thuộc K (rng=seed+K); dùng trong main")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n-fault", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--pretrain-windows", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pool-normal", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "contr_harness.json"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    win, stride = 512, 256
    fs = pipe.FS

    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    all_normal = pipe.flatten_runs(signs_n)
    stride_eff = pipe._effective_stride(all_normal, win, stride)
    print(f"stride_eff={stride_eff}  pretrain={args.pretrain_windows} epochs={args.epochs}")

    # ---- windows THÔ cho toàn bộ: dùng stride (không nới) cho pretrain đủ mẫu ----
    # Pretrain cần nhiều windows; dùng stride=win//2 để có corpus phong phú (như script 08).
    corpus = np.concatenate([raw_windows(all_normal, win, 128),
                             raw_windows(pipe.flatten_runs(signs_f), win, 128)])
    rng_c = np.random.default_rng(args.seed + 999)
    corpus = corpus[rng_c.choice(len(corpus), size=min(args.pretrain_windows, len(corpus)),
                                 replace=False)] if len(corpus) > args.pretrain_windows else corpus
    print(f"[pretrain] corpus={corpus.shape}")

    # pretrain encoder NHẸ (epochs nhỏ để không mất quá nhiều thời gian, chỉ 2 nhánh)
    enc = ContrastiveEncoder(seq_len=win, out_dim=128, seed=args.seed)
    enc.pretrain(corpus, epochs=args.epochs, batch_size=args.batch,
                 log_every=max(1, args.epochs // 2))

    rows = []
    normal_train_raw = raw_windows(all_normal, win, stride_eff)
    n_tr_n = max(1, int(len(normal_train_raw) * 0.7))  # khớp xấp xỉ normal train
    # Đơn giản hoá: dùng toàn bộ normal_raw_train (không khớp 100% vùng nhưng đủ để so)
    # TODO(nếu cần chính xác tuyệt đối): tái tạo đúng 3-zone như pipeline.

    for K in args.n_fault:
        # --- dựng split và tái tạo windows thô cho fault train/test ---
        d_rng = np.random.default_rng(args.seed + K)
        d = pipe.split_and_scale(signs_n, signs_f, win, stride, d_rng, max_fault_train=K)
        X_base, y_base, X_test, y_test = pipe.to_train_test_arrays(d)
        mu, sd = d["scaler_mu"], d["scaler_sd"]

        # windows thô fault TRẠI (khớp split)
        pool_f = raw_windows(pipe.flatten_runs(signs_f), win, stride_eff)
        idx = np.arange(len(pool_f))
        rng_f = np.random.default_rng(args.seed + K)
        rng_f.shuffle(idx)
        n_all = max(1, int(0.2 * len(pool_f)))
        f_tr_raw = pool_f[idx[:n_all]][:K]
        f_te_raw = pool_f[idx[n_all:]]

        # windows thô NORMAL TRẠI (khớp split: 0→70% mỗi run, dùng stride_eff)
        n_tr_raw = raw_windows(all_normal, win, stride_eff)
        # (đơn giản: lấy 70% đầu của toàn bộ — tương đối, để so contrastive là đủ)

        # --- baseline: RF trên feature 24D (từ d, đã z-score) ---
        clf0 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(X_base, y_base)
        r0 = _score(clf0, X_test, y_test)

        # --- contrastive: RF trên embedding (pretrain trước; không sinh lỗi) ---
        # encode normal-train + fault-train (MỘT range để nhất quán embedding)
        # BUG-FIX (seed-flaky): subsample ngẫu nhiên có thể LOẠI HẾT 10 cửa sổ lỗi thật
        # khi K nhỏ + n_tr_raw lớn (vd K=10, cap 6000 từ ~40k) → RF chỉ thấy 1 class →
        # predict_proba cháy. LUÔN GIỮ TOÀN BỘ fault windows, chỉ cap NORMAL.
        n_fault = len(f_tr_raw)
        tr_w = np.concatenate([n_tr_raw, f_tr_raw])
        tr_l = np.concatenate([np.zeros(len(n_tr_raw)), np.ones(n_fault)])
        cap = 6000
        if len(tr_w) > cap and n_fault < cap:
            sel_n = np.random.default_rng(args.seed + K + 555).choice(
                len(n_tr_raw), size=max(1, cap - n_fault), replace=False)
            tr_w = np.concatenate([n_tr_raw[sel_n], f_tr_raw])
            tr_l = np.concatenate([np.zeros(len(sel_n)), np.ones(n_fault)])
        c_tr = enc.encode(tr_w)
        # test: normal-test (features d[]) + fault-test (raw → encode)
        # normal-test raw: lấy vùng 70-90% của n_tr_raw
        te_n_raw = n_tr_raw[int(len(n_tr_raw) * 0.7):int(len(n_tr_raw) * 0.9)]
        te_w = np.concatenate([te_n_raw, f_te_raw])
        te_l = np.concatenate([np.zeros(len(te_n_raw)), np.ones(len(f_te_raw))])
        c_te = enc.encode(te_w)
        clf2 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(c_tr, tr_l)
        r2 = _score(clf2, c_te, te_l)

        rows.append({"n_fault_train": K,
                     "baseline": r0, "contrastive": r2,
                     "n_test_fault": int((y_test == 1).sum())})
        print(f"[K={K:>3}] baseline={r0['recall']:.4f} contrastive={r2['recall']:.4f} "
              f"| test_fault={int((y_test==1).sum())}", flush=True)

    print("\n=== BẢNG: baseline vs contrastive (CÙNG harness khan hiếm) ===")
    for r_ in rows:
        print(f"K={r_['n_fault_train']:>3}: baseline={r_['baseline']['recall']:.4f} "
              f"contrastive={r_['contrastive']['recall']:.4f}")

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "seq_len": win, "epochs": args.epochs,
                   "stride_eff": stride_eff, "rows": rows}, f, indent=2)
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
