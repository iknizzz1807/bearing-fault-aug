"""Loader cho các dataset ổ bi public (CWRU, NASA IMS, FEMTO)."""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_windows(sig: np.ndarray, win: int = 512, stride: int = 256) -> np.ndarray:
    """Cắt tín hiệu 1D thành ma trận (n_windows, win)."""
    if len(sig) < win:
        return np.empty((0, win))
    n = (len(sig) - win) // stride + 1
    idx = np.arange(n)[:, None] * stride + np.arange(win)
    return sig[idx]


def _key_ends_with_DE(m: dict) -> np.ndarray | None:
    for k, v in m.items():
        if not k.startswith("__") and k.endswith("DE_time"):
            return np.asarray(v).ravel()
    return None


def load_cwru(folder: str | Path = DATA_DIR / "CWRU",
              fault_subdirs: tuple[str, ...] = ("12k Drive End Bearing Fault Data",),
              normal_subdirs: tuple[str, ...] = ("Normal Baseline Data",),
              sr: int = 12000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CWRU về (sigs, labels, meta). label=0 normal, 1 fault.

    CWRU file .mat chứa nhiều biến; chỉ lấy kênh rung drive-end (DE_time).
    """
    folder = Path(folder)
    sigs, labels = [], []

    # load_cwru trả về cả normal lẫn fault; nếu fault subdir chưa tồn tại thì chỉ normal
    for sub in normal_subdirs:
        for f in sorted(glob.glob(str(folder / sub / "*.mat"))):
            try:
                m = loadmat(f)
            except Exception:
                continue  # file hỏng (vd normal baseline CWRU bị serve thiếu)
            sig = _key_ends_with_DE(m)
            if sig is not None and len(sig) > 0:
                sigs.append(sig)
                labels.append(0)
    for sub in fault_subdirs:
        for f in sorted(glob.glob(str(folder / sub / "*.mat"))):
            try:
                m = loadmat(f)
            except Exception:
                continue
            sig = _key_ends_with_DE(m)
            if sig is not None and len(sig) > 0:
                sigs.append(sig)
                labels.append(1)

    if not sigs:
        raise FileNotFoundError(
            f"CWRU chưa có dữ liệu đọc được tại {folder}. "
            "Lưu ý: 4 file Normal Baseline (97–100.mat) trên trang chính thức bị hỏng "
            "(serve thiếu bytes) — chỉ các file fault (105, 118, 130…) đọc được. "
            "Khuyến nghị: dùng --dataset ims làm nguồn chính (đã có đủ normal + fault).")
    return sigs, np.asarray(labels), None


def ims_files(test_dir: Path) -> list[str]:
    """Trả về danh sách file IMS của 1 test (bằng glob, không load nội dung — nhanh)."""
    ts_pattern = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
    return sorted(p for p in test_dir.rglob("*") if p.is_file() and ts_pattern.match(p.name))


def load_ims_paths(files: list[str] | list[Path]) -> list[np.ndarray]:
    """Load nội dung danh sách file IMS (dùng khi đã biết trước cần đọc file nào)."""
    out = []
    for f in files:
        arr = np.genfromtxt(f, delimiter="\t")
        if arr.size:
            out.append(arr.ravel())
    return out


def load_ims(folder: str | Path = DATA_DIR / "NASA_IMS",
             limit_per_test: int | None = None) -> dict[str, list[np.ndarray]]:
    """Load NASA IMS bearing dataset.

    Cấu trúc thật sau khi giải nén:
      NASA_IMS/
        test_1/ 2003.10.22.12.06.24 ...   (2156 file, 8 cột, 20480 dòng)
        test_2/ 2004.02.12.*              (984 file, 8 cột)
        test_3/txt/2004.03.04.*           (6324 file, 4 cột)
    File đặt tên theo timestamp, KHÔNG có đuôi .txt, mỗi dòng là 1 mốc thời gian
    với các kênh cảm biến. Trả về dict {tên_test: [signal thô từng file, ...]}.
    Chọn toàn bộ cột của mỗi file như một tín hiệu đa biến.

    `limit_per_test`: giới hạn số file tải cho mỗi test (0/None = tất cả).
    Tải hết dữ liệu chiếm ~8GB RAM; với baseline nên giới hạn (vd 300-500 file).
    """
    folder = Path(folder)
    per_test: dict[str, list[np.ndarray]] = {}
    ts_pattern = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
    # mỗi test nằm trong 1 thư mục con, mỗi file là 1 mốc thời gian (tên dạng timestamp)
    for test_dir in sorted(folder.iterdir()):
        if not test_dir.is_dir():
            continue
        files = []
        for p in sorted(test_dir.rglob("*")):
            if p.is_file() and ts_pattern.match(p.name):
                files.append(p)
        if not files:
            continue
        # `limit_per_test`: giới hạn số file MỖI ĐẦU (normal) + CUỐI (fault). Vì tín
        # hiệu hỏng nằm ở CUỐI chuỗi, chỉ load đầu hoặc cuối là đủ, không cần full.
        n = len(files)
        head = files[:limit_per_test] if limit_per_test else files[:]
        tail = files[-limit_per_test:] if limit_per_test else files[:]
        # gộp đầu+cuối nhưng bỏ trùng (nếu test quá ngắn)
        chosen = head + [f for f in tail if f not in head]
        seen, sigs = set(), []
        for f in chosen:
            arr = np.genfromtxt(f, delimiter="\t")
            if arr.size == 0:
                continue
            sigs.append(arr.ravel())
            seen.add(f.name)
        per_test[test_dir.name] = sigs
    if not per_test:
        raise FileNotFoundError(
            f"NASA IMS chưa có dữ liệu tại {folder}. "
            "Cấu trúc mong đợi: NASA_IMS/test_{1,2,3}/… (xem README về format .rar).")
    return per_test


def load_femto(folder: str | Path = DATA_DIR / "FEMTO") -> list[np.ndarray]:
    """Load FEMTO/PRONOSTIA. Folder con 'train'/'test'/... — lấy mọi file csv.

    FEMTO chia theo sub-dataset: Bearing1_*/Bearing2_*; file 'acc.csv' (accel) hoặc
    'acc_??.csv'. Header có tên cột.
    """
    folder = Path(folder)
    files = sorted(glob.glob(str(folder / "**" / "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"FEMTO chưa có dữ liệu tại {folder}. Chạy scripts/download_data.sh trước.")
    sigs = []
    for f in files:
        df = pd.read_csv(f)
        sigs.append(df.to_numpy(dtype=float).ravel())
    return sigs


def detect_dataset(data_dir: str | Path = DATA_DIR) -> str:
    """Tự đoán dataset hiện có: 'ims' | 'cwru' | 'femto' | 'none'."""
    d = Path(data_dir)
    if any((d / "NASA_IMS").glob("test_*/*")):
        return "ims"
    if (d / "CWRU").exists():
        return "cwru"
    if any(Path(d, "FEMTO").glob("**/*.csv")):
        return "femto"
    return "none"