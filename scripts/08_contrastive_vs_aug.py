#!/usr/bin/env python3
"""SCRIPT 08 — Đối chiếu "MÔ HÌNH MỚI" (self-supervised contrastive) vs "SINH DỮ LIỆU LỖI".

Hướng C trong SPEC.md. Câu hỏi trung tâm:
  Thay vì SINH thêm lỗi giả (interpolate/physics/heuristic/TimeGAN — script 06/07),
  có thể dùng MÔ HÌNH HỌC BIỂU DIỄN KHÔNG NHÃN (contrastive) để model với VÀI lỗi thật
  chạy tốt hơn không? — nhóm model mà các paper gần đây ghi nhận vượt trội
  data-augmentation (~+21%).

SO SÁNH CÔNG BẰNG (cùng K lỗi thật, cùng vùng test là lỗi THẬT chưa từng thấy):
  NGUỒN CHUNG: lấy cửa sổ thời gian thô theo 3-zone đúng của pipeline
               (normal TRAIN/TEST/GREY + fault 20/80) để cả 3 nhánh dùng CÙNG
               một tập train-chưa-lộ + test sạch.
  - baseline_ensemble: RF trên embedding gộp (thời gian + envelope) thay feature 24D
        → MỐC "trước", không cần sinh lỗi.
  - interpolate: RF trên embedding gộp + lỗi giả interpolate (bộ sinh tối ưu).
  - contrastive: pretrain encoder không nhãn -> RF trên embedding contrastive
        (chỉ cần K lỗi thật + normal; KHÔNG sinh thêm lỗi giả).

GHI CHÚ: contrastive dùng MỌI window (normal+fault, không cần nhãn) trong pretrain.
Đây là điểm khác biệt cốt lõi — model học từ toàn bộ dữ liệu khả dụng chứ không
phải chỉ K lỗi thật, và KHÔNG cần bịa thêm lỗi giả.

CÁCH CHẠY:
    python scripts/08_contrastive_vs_aug.py --n-fault 50 20 --epochs 12 --pretrain-windows 4000
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
from src.generator import OptimizedFaultGenerator   # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _score(clf, X, y) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y, clf.predict(X), average="binary",
                                                  zero_division=0)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}


def _windows(runs, rng, win, cap=None):
    out = []
    for run in runs:
        for s in run:
            out.append(ds.make_windows(s, win=win, stride=win // 2))
    out = np.concatenate(out)
    if cap and len(out) > cap:
        out = out[rng.choice(len(out), size=cap, replace=False)]
    return out


def _sample_from(appended_or_array, rng, cap):
    if cap and len(appended_or_array) > cap:
        return appended_or_array[rng.choice(len(appended_or_array), size=cap, replace=False)]
    return appended_or_array


def _feature_embed(windows: np.ndarray, fs: float) -> np.ndarray:
    """Embedding 'kinh điển' = 24 feature thời gian/tần số/envelope (chuẩn hoá riêng cụm)."""
    return ft.extract_features(windows, fs=fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ims", "cwru"], default="ims")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-fault", type=int, nargs="+", default=[50, 20])
    ap.add_argument("--pretrain-windows", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-synth", type=int, default=1600)
    ap.add_argument("--pool-normal", type=int, default=6000,
                    help="capsố window normal mỗi vùng (giữ feature extraction khỏi chậm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS / "contrastive_vs_aug.json"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    win = args.seq_len
    fs = pipe.FS

    signs_n, signs_f = pipe.load_normal_fault_signals(args.dataset, args.limit)
    rng = np.random.default_rng(args.seed)

    # ---------- 1. Cửa sổ thô theo đúng 3-zone của pipeline (chống lộ) ----------
    # Ta dựng lại từng vùng (normal TRAIN/TEST/GREY + fault train/test) DỰA TRÊN thời gian
    # giống hệt split_and_scale, nhưng giữ tín hiệu THÔ (chưa chuẩn hoá) → đủ cho
    # cả encoder contrastive lẫn feature kinh điển.
    def zone_normal():
        train, test, grey = [], [], []
        for run in signs_n:
            for sig in run:
                w = ds.make_windows(sig, win=win, stride=win // 2)
                if len(w):
                    i_tr = int(len(w) * pipe.NORMAL_TRAIN_END)
                    i_te = int(len(w) * pipe.NORMAL_TEST_END)
                    train.append(w[:i_tr]); test.append(w[i_tr:i_te]); grey.append(w[i_te:])
        # giới hạn số window mỗi vùng (giữ feature extraction khỏi chậm/treo)
        rng_n = np.random.default_rng(args.seed + 777)
        return (_sample_from(np.concatenate(train) if train else np.empty((0, win)), rng_n, args.pool_normal),
                _sample_from(np.concatenate(test) if test else np.empty((0, win)), rng_n, max(2000, args.pool_normal // 3)),
                np.concatenate(grey) if grey else np.empty((0, win)))

    def zone_fault(k):
        all_f = _windows(signs_f, np.random.default_rng(args.seed + k), win)
        np.random.default_rng(args.seed + k).shuffle(all_f)
        n_tr = max(1, int(0.2 * len(all_f)))
        # khớp qua split_and_scale: train = min(20%, K), test = 80% CỐ ĐỊNH
        tr = all_f[:min(n_tr, k)]
        te = all_f[n_tr:]
        return tr, te

    n_tr_n, te_n, _ = zone_normal()
    pretrain_pool = np.concatenate([n_tr_n, te_n, _windows(signs_f, rng, win)])
    pretrain_pool = pretrain_pool[rng.choice(len(pretrain_pool),
                                             size=min(args.pretrain_windows, len(pretrain_pool)),
                                             replace=False)]
    print(f"[1] corpus (KHÔNG nhãn) pretrain: {pretrain_pool.shape}")

    enc = ContrastiveEncoder(seq_len=win, out_dim=128, seed=args.seed)
    enc.pretrain(pretrain_pool, epochs=args.epochs, batch_size=args.batch,
                 log_every=max(1, args.epochs // 3))

    # embedding kinh điển (feature 24D) cho toàn bộ vùng — tương tự pipeline
    emb_n_tr = _feature_embed(n_tr_n, fs)   # normal train
    emb_te_n = _feature_embed(te_n, fs)     # normal test (khỏe, chưa thấy)

    rows = []
    for K in args.n_fault:
        f_tr, f_te = zone_fault(K)
        emb_f_tr = _feature_embed(f_tr, fs)
        emb_f_te = _feature_embed(f_te, fs)

        # --- nhãn + gộp train/test cho nhánh "feature kinh điển" ---
        X_tr = np.concatenate([emb_n_tr, emb_f_tr])
        y_tr = np.concatenate([np.zeros(len(emb_n_tr)), np.ones(len(emb_f_tr))])
        X_te = np.concatenate([emb_te_n, emb_f_te])
        y_te = np.concatenate([np.zeros(len(emb_te_n)), np.ones(len(emb_f_te))])
        n_tr = len(emb_n_tr)

        row = {"n_fault_train": K, "n_test_fault": int((y_te == 1).sum())}

        # 2a. baseline (RF trên feature, KHÔNG sinh lỗi)
        clf0 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(X_tr, y_tr)
        row["baseline"] = _score(clf0, X_te, y_te)

        # 2b. interpolate (thêm lỗi giả vào feature space)
        gen = OptimizedFaultGenerator(fs=fs)
        gen.calibrate(f_tr, n_tr_n)
        syn_sig = gen.generate_interpolated(f_tr, args.n_synth, np.random.default_rng(args.seed + 2000 + K))
        X_syn = ft.zscore(ft.raw_features(syn_sig, fs=fs), *ft.fit_zscore(X_tr))
        X_aug = np.concatenate([X_tr, X_syn])
        y_aug = np.concatenate([y_tr, np.ones(len(X_syn))])
        clf1 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(X_aug, y_aug)
        row["interpolate"] = _score(clf1, X_te, y_te)

        # 2c. contrastive: RF trên embedding contrastive (KHÔNG sinh lỗi)
        c_tr = enc.encode(np.concatenate([n_tr_n[:len(n_tr_n)], f_tr]))
        # nhãn: normal = 0, K lỗi thật = 1
        y_c = np.concatenate([np.zeros(len(n_tr_n)), np.ones(len(f_tr))])
        c_te = enc.encode(np.concatenate([te_n, f_te]))
        clf2 = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced").fit(c_tr, y_c)
        row["contrastive"] = _score(clf2, c_te, np.concatenate(
            [np.zeros(len(te_n)), np.ones(len(f_te))]))

        rows.append(row)
        print(f"[K={K:>3}] baseline={row['baseline']['recall']:.4f} "
              f"interpolate={row['interpolate']['recall']:.4f} "
              f"contrastive={row['contrastive']['recall']:.4f}", flush=True)

    print("\n=== BẢNG RECALL: baseline / interpolate / contrastive ===")
    for r_ in rows:
        print(f"K={r_['n_fault_train']:>3}: baseline={r_['baseline']['recall']:.4f} "
              f"interp={r_['interpolate']['recall']:.4f} "
              f"contrastive={r_['contrastive']['recall']:.4f}")
    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "seq_len": win, "epochs": args.epochs,
                   "pretrain_windows": args.pretrain_windows, "rows": rows}, f, indent=2)
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
