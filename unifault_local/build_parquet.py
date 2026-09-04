"""build_parquet.py — build UniFault-format parquet from local NASA_IMS data.

Produces train/val/test .parquet per scarcity level K under <DATA_ROOT>/ims_K{K}/.
Matches UniFault's PHMDataset expectations:
  - samples column = [n, seq_len, num_channels] (channels LAST, =1 here); getitem
    does x_data[index].squeeze(-1) -> model receives [B, seq_len].
  - labels column = int64 class (0 normal, 1 fault).

Run from repo root A1_predictive_maintenance/:
  python unifault_local/build_parquet.py
"""
from __future__ import annotations

import os
import glob
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMS_ROOT = os.path.join(REPO_ROOT, "data", "NASA_IMS")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NORMAL_END = 0.70
FAULT_START = 0.90
SEQ = 1024
STRIDE = 512
CHANNELS = 1
TEST_FAULT_CAP = 8100   # mirror our fixed 8100-fault test set for a fair comparison


def ims_files(test_dir: str) -> list[str]:
    pat = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
    # signal files may live directly under test_dir (test_1/2) or in a subdir like
    # test_3/txt/. Recurse one level down to catch both layouts.
    cands = []
    for sub in sorted(glob.glob(os.path.join(test_dir, "**"), recursive=False)[:3] + [test_dir]):
        for p in glob.glob(os.path.join(sub, "*")):
            if os.path.isfile(p) and pat.match(os.path.basename(p)):
                cands.append(p)
    return sorted(set(cands))


def load_channel(path: str, chan: int) -> np.ndarray | None:
    try:
        a = np.genfromtxt(path, delimiter="\t")
    except Exception:
        return None
    if a.size == 0:
        return None
    if a.ndim == 2:
        return a[:, min(chan, a.shape[1] - 1)]
    return a.ravel()


def windows(sig: np.ndarray | None, seq: int, stride: int) -> np.ndarray:
    if sig is None or len(sig) < seq:
        return np.empty((0, seq))
    n = (len(sig) - seq) // stride + 1
    idx = np.arange(n)[:, None] * stride + np.arange(seq)
    return sig[idx]


def write_parquet(arr: np.ndarray, labels: np.ndarray, path: str) -> None:
    # UniFault's PatchEmbed does x.unfold(dim=-1, ...) then rearrange 'b m n p -> (b m) n p',
    # so the model receives [B, num_channels, seq_len]. Store each sample as
    # [num_channels, seq_len] = [1, 1024] (channel SECOND). getitem's squeeze(-1) leaves
    # the 1024 dim intact (not size 1), so [B,1,1024] reaches the model.
    samples = arr[:, None, :].astype(np.float32).tolist()   # [n, 1, seq_len]
    t = pa.table({
        "samples": pa.array(samples, type=pa.list_(pa.list_(pa.float32()))),
        "labels": pa.array(labels, type=pa.int64()),
    })
    pq.write_table(t, path)


def minmax_scale(X):
    lo, hi = X.min(), X.max()
    return (X - lo) / (hi - lo + 1e-9), lo, hi


def main() -> None:
    if not os.path.isdir(IMS_ROOT):
        raise SystemExit(f"IMS_ROOT not found: {IMS_ROOT}")

    normal_wins, fault_wins, found_tests = [], [], 0
    for test_dir in sorted(glob.glob(os.path.join(IMS_ROOT, "test_*"))):
        files = ims_files(test_dir)
        if len(files) < 3:
            continue
        found_tests += 1
        n = len(files)
        i_norm, i_fault = int(n * NORMAL_END), int(n * FAULT_START)
        f_norm, f_fault = files[:i_norm], files[i_fault:]
        print(f"  test {os.path.basename(test_dir)}: {n} files "
              f"(normal={len(f_norm)}, fault={len(f_fault)})")
        for p in f_norm:
            w = windows(load_channel(p, CHANNELS), SEQ, STRIDE)
            if len(w):
                normal_wins.append(w)
        for p in f_fault:
            w = windows(load_channel(p, CHANNELS), SEQ, STRIDE)
            if len(w):
                fault_wins.append(w)

    X_norm = np.concatenate(normal_wins) if normal_wins else np.empty((0, SEQ))
    X_fault = np.concatenate(fault_wins) if fault_wins else np.empty((0, SEQ))
    print("=== RESULT ===", flush=True)
    print("tests found:", found_tests, flush=True)
    print("normal windows:", X_norm.shape, "| fault windows:", X_fault.shape, flush=True)
    assert found_tests > 0, "no valid test_* dir"
    assert len(X_fault) > 0, "no FAULT windows built"
    assert len(X_norm) > 0, "no NORMAL windows built"

    def build_dataset(K: int) -> None:
        dirpath = os.path.join(DATA_ROOT, f"ims_K{K}")
        os.makedirs(dirpath, exist_ok=True)
        rng = np.random.default_rng(42 + K)
        n_norm, n_fault = len(X_norm), len(X_fault)

        norm_idx = rng.permutation(n_norm)
        n_train_n = int(0.7 * n_norm)
        n_train_n = min(n_train_n, 40000)
        X_train_n = X_norm[norm_idx[:n_train_n]]

        if K >= n_fault:
            raise SystemExit(f"K={K} >= fault windows {n_fault}. Reduce K or add data.")
        fault_idx = rng.permutation(n_fault)
        X_train_f = X_fault[fault_idx[:K]]
        X_test_f = X_fault[fault_idx[K:K + TEST_FAULT_CAP]]
        n_test_f = len(X_test_f)
        # cap normal test too, to keep the test set balanced-ish and fast
        n_test_n = n_test_f
        X_test_n = X_norm[norm_idx[n_train_n:n_train_n + n_test_n]]

        X_train_n, lo, hi = minmax_scale(X_train_n)
        X_train_f = (X_train_f - lo) / (hi - lo + 1e-9)
        X_test_n = (X_test_n - lo) / (hi - lo + 1e-9)
        X_test_f = (X_test_f - lo) / (hi - lo + 1e-9)

        nval_n = int(0.1 * len(X_train_n))
        nval_f = int(0.1 * len(X_train_f))
        X_val_n, X_val_f = X_train_n[:nval_n], X_train_f[:nval_f]
        X_train_n = X_train_n[nval_n:]
        X_train_f = X_train_f[nval_f:]

        tr = np.concatenate([X_train_n, X_train_f])
        tl = np.concatenate([np.zeros(len(X_train_n)), np.ones(len(X_train_f))]).astype(np.int64)
        va = np.concatenate([X_val_n, X_val_f])
        vl = np.concatenate([np.zeros(len(X_val_n)), np.ones(len(X_val_f))]).astype(np.int64)
        te = np.concatenate([X_test_n, X_test_f])
        el = np.concatenate([np.zeros(len(X_test_n)), np.ones(len(X_test_f))]).astype(np.int64)
        write_parquet(tr, tl, os.path.join(dirpath, "train.parquet"))
        write_parquet(va, vl, os.path.join(dirpath, "val.parquet"))
        write_parquet(te, el, os.path.join(dirpath, "test.parquet"))
        print(f"K={K}: train {tr.shape} (fault={len(X_train_f)}), "
              f"val {va.shape}, test {te.shape} (fault={len(X_test_f)})")

    for K in (10, 20, 50):
        build_dataset(K)
    print("datasets built under", DATA_ROOT)


if __name__ == "__main__":
    main()
