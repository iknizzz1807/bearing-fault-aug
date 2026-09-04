"""pipeline.py — chuẩn bị dữ liệu dùng chung cho CẢ 3 script (01/02/03).

Tại sao phải có file này?
    Ba script đều phải làm cùng một loạt việc: tải tín hiệu → chia normal/fault
    theo thời gian → cắt windows → tính feature → chia train/test → chuẩn hoá.
    Trước đây các bước này bị copy-paste rải rác, mỗi nơi một kiểu → khó đọc,
    dễ sửa lệch chỗ này mà quên chỗ kia. Gom hết vào đây, script chỉ còn là:
    "ghi tham số → gọi pipeline → train model → in kết quả".

Tư duy chia dữ liệu (QUAN TRỌNG — phải đọc để tránh nhầm lẫn):
    Bài toán A1 là "thiếu dữ liệu lỗi". Thực tế nhà máy: máy chạy tốt 99% thời
    gian (dữ liệu NORMAL rất nhiều), chỉ thi thoảng mới hỏng (dữ liệu FAULT rất
    hiếm). Vì vậy mọi thí nghiệm phải mô phỏng đúng tỉ lệ này.

    Với dataset IMS (run-to-failure), mỗi test có nhiều file theo thời gian:
        [0% ---------- 70% ---- 90% ---- 100%]  của chuỗi thời gian
        [  NORMAL (khỏe)  | vùng mơ hồ | FAULT(suy giảm/hỏng) ]
    - 0 → 70%  : NORMAL (máy còn khỏe)
    - 70 → 90% : BỎ QUA (lúc máy bắt đầu xuống cấp, nhãn không rõ ràng)
    - 90 → 100%: FAULT (dữ liệu lỗi THẬT — chính là thứ hiếm này)

    Vì sao phải CHIA DỌC THEO THỜI GIAN của toàn chuỗi (không chia theo file tải)?
        IMS là "chạy đến khi hỏng" (run-to-failure): dấu hiệu lỗi CHỈ xuất hiện
        ở CUỐI mỗi test (file ~2000), các file đầu vẫn là normal khỏe. Nhãn
        normal/fault vì thế phải được suy ra từ VỊ TRÍ theo thời gian trong chuỗi
        ĐẦY ĐỦ, không thể suy từ chính các file đã tải vào. Đây là kịch bản điển
        hình của predictive-maintenance thực tế (xem SPEC.md §4).

    Vì sao bỏ qua vùng 70–90%? Nếu để lẫn vào sẽ khiến model học "báo động sớm
    nhưng chưa hỏng thật" → gây nhãn nhiễu. Đây là quyết định kỹ thuật để nhãn
    sạch, và nên được giải thích trong slide.

    QUAN TRỌNG — TEST NORMAL phải lấy từ vùng KHỎE, không phải normal-sát-hỏng:
    Trong phạm vi NORMAL (0–70%) của MỖI test-run, lại chia tiếp 3 vùng:
        [0% — 70% | 70% — 90% | 90% — 100%]  của riêng vùng NORMAL
        [ TRAIN    |  TEST     |  GREY(xoá) ]
    - TRAIN: normal thật khỏe (model học "thế nào là bình thường").
    - TEST : normal chưa thấy nhưng VẪN KHỎE — đây là test đánh giá.
    - GREY : normal đã suy giảm, sát vùng hỏng — LOẠI khỏi cả train lẫn test.

    Vì sao phải LOẠI GREY khỏi CẢ TRAIN LẪN TEST (chống rò rỉ nhãn)?
        GREY là normal đã xuống cấp, sát ranh giới hỏng — nhãn của nó MƠ HỒ.
        - Cho GREY vào TRAIN: model học "normal xấu" như một dạng khiếm khuyết
          → sau này bắt nhầm normal-suýt-hỏng là lỗi (báo động sớm giả).
        - Cho GREY vào TEST: test trở nên "bất công" — model phải phân biệt
          normal-sát-hỏng với lỗi thật, hai thứ gần như trùng nhau → recall/AUC
          thấp oan cho model.
        Cách sạch nhất là loại hẳn GREY khỏi cả hai phía, chỉ giữ lại vùng normal
        RÕ RÀNG khỏe và vùng lỗi RÕ RÀNG hỏng → so sánh trước/sau augmentation
        trên cùng một cột mốc công bằng.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as ds          # noqa: E402
from src import features as ft      # noqa: E402

FS = 12000        # tần số lấy mẫu mặc định (Hz) của cảm biến rung công nghiệp
NORMAL_END = 0.70   # 0→70% chuỗi thời gian = NORMAL
FAULT_START = 0.90  # 90→100% chuỗi thời gian = FAULT
                     # (vùng 70–90% bỏ qua, xem giải thích ở docstring)
# Trong vùng NORMAL của mỗi test-run: 3 vùng con (tính trên phần NORMAL)
NORMAL_TRAIN_END = 0.70   # [0 → 70%]  normal khỏe → huấn luyện
NORMAL_TEST_END = 0.90    # [70% → 90%] normal khỏe chưa thấy → TEST
                          # [90% → 100%] normal suy giảm sát hỏng → GREY (xoá)


def flatten_runs(runs: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Làm phẳng danh sách các test-run (list[list[tín hiệu]]) thành 1 list."""
    return [sig for run in runs for sig in run]


def load_normal_fault_signals(dataset: str,
                              limit_per_test: int | None = None
                              ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Tải tín hiệu, trả về (runs_normal, runs_fault).

    Mỗi phần tử của `runs_*` là MỘT test-run (chuỗi tín hiệu theo thời gian):
      - IMS: mỗi test (test_1/2/3) là 1 run độc lập.
      - CWRU: không có khái niệm run-to-failure → xem cả folder normal/fault
              như 1 run duy nhất (chia train/test vẫn theo tỉ lệ, không theo thời gian).
    Giữ nguyên cấu trúc run giúp `split_and_scale` chia TEST normal khỏe đúng cách.

    CÁCH TÁCH normal/fault với IMS (design rationale — xem SPEC.md §4):
        IMS là run-to-failure: lỗi CHỈ nằm ở cuối chuỗi. Để vùng "fault" không
        rơi vào dữ liệu normal, ta chia vùng DỰA TRÊN TỔNG SỐ FILE THẬT của test
        (glob nhanh, đếm thật), rồi tính ranh giới normal/fault bằng tỉ lệ
        NORMAL_END/FAULT_START trên con số đó — KHÔNG dựa trên `limit_per_test`.
        Bằng cách này ranh giới vùng luôn đúng dù `limit_per_test` có đổi.

        Các OPTIONS đã cân nhắc và lý do chọn:
          - Load TOÀN BỘ chuỗi rồi cắt vùng: chính xác nhất nhưng bị LOẠI vì IMS
            1 file = ~163k sample, load đủ cả test rất nặng → OOM / chạy quá chậm,
            trong khi chỉ vùng đầu (normal) + cuối (fault) là cần thiết.
          - Cắt theo MỘT điểm thời gian cố định (vd file 0 và file -1): bị LOẠI
            vì NGỮ NGHĨA SAI — hỏng ổ bi là quá trình suy giảm DẦN theo thời gian,
            không phải hiện tượng xảy ra tại một điểm thời gian cố định.
          - Lựa chọn HIỆN TẠI: đếm file thật → chỉ load ĐẦU (normal) + CUỐI
            (fault) của chuỗi. Vừa giữ đúng tỉ lệ "fault hiếm", vừa không phải
            load full → máy không OOM và chạy nhanh.
    """
    runs_normal: list[list[np.ndarray]] = []
    runs_fault: list[list[np.ndarray]] = []

    if dataset == "ims":
        # VÌ SAO PHẢI CHIA THEO SỐ FILE THẬT (xem thêm docstring): IMS là
        # run-to-failure, lỗi chỉ ở CUỐI chuỗi. Ta đếm file thật (glob nhanh),
        # rồi lấy ranh giới normal/fault theo tỉ lệ NORMAL_END/FAULT_START trên
        # con số đó. Nếu chia theo `limit_per_test` (số file tải vào) thì vùng
        # "fault" có thể rơi vào normal chưa hỏng → nhãn sai, model không tách nổi.
        for test_dir in sorted(ds.DATA_DIR.joinpath("NASA_IMS").iterdir()):
            if not test_dir.is_dir():
                continue
            files = list(ds.ims_files(test_dir))                    # đếm thật (nhanh)
            if len(files) < 3:
                continue
            n = len(files)
            i_normal = int(n * NORMAL_END)
            i_fault = int(n * FAULT_START)
            # Số file mỗi vùng muốn load: giữ bản chất "fault hiếm", đồng thời
            # không load full chuỗi (đã cân nhắc các OPTION — xem docstring).
            n_norm = limit_per_test if limit_per_test else len(files[:i_normal])
            n_fault = max(3, limit_per_test // 4) if limit_per_test else len(files[i_fault:])
            # Chỉ load ĐẦU (normal) + CUỐI (hỏng) — vùng GIỮA (70–90%, mơ hồ) vốn
            # bị loại bỏ theo anti-leakage nên không cần tải về → máy không OOM.
            normal_part = ds.load_ims_paths(files[:n_norm])
            fault_part = ds.load_ims_paths(files[i_fault:][-n_fault:])
            runs_normal.append(normal_part)
            runs_fault.append(fault_part)

    elif dataset == "cwru":
        signals, labels, _ = ds.load_cwru()
        runs_normal.append([s for s, l in zip(signals, labels) if l == 0])
        runs_fault.append([s for s, l in zip(signals, labels) if l == 1])

    else:
        raise SystemExit("Chỉ hỗ trợ dataset 'ims' hoặc 'cwru' ở bước này.")

    return runs_normal, runs_fault


def windows_to_features(signals: list[np.ndarray], win: int, stride: int) -> np.ndarray:
    """Cắt mỗi tín hiệu thành windows (cửa sổ), rồi tính FEATURE cho từng window.

    Kết quả: ma trận (n_windows, n_feature) với RAW features (chưa chuẩn hoá —
    bước chuẩn hoá được làm ở cuối trên toàn bộ tập train, xem `split_and_scale`).
    """
    out = []
    for sig in signals:
        windows = ds.make_windows(sig, win=win, stride=stride)
        if len(windows):
            out.append(ft.raw_features(windows, fs=FS))
    if not out:
        raise SystemExit("Không có windows nào được tạo — kiểm tra dữ liệu/--win/--stride.")
    return np.concatenate(out)


def _effective_stride(signals: list[np.ndarray], win: int, stride: int,
                      target_windows: int = 60_000) -> int:
    """Tự nới stride để tổng số window không quá `target_windows` (giữ máy không rung.

    Nếu dữ liệu dài (IMS 1 file = 163k sample), stride cố định 256 sẽ tạo ra vài
    trăm nghìn window → huấn luyện rất chậm. Nới stride lên thì số window giảm
    mà vẫn bao phủ hết tín hiệu.
    """
    # Ước lượng: 1 tín hiệu dài trung bình tạo ra bao nhiêu window với stride 1?
    avg_len = sum(max(1, len(s)) for s in signals[:20]) / max(1, len(signals[:20]))
    if avg_len <= win:
        return stride
    # window_count ~ (avg_len - win) / stride  →  muốn = target_windows/n_signals
    return max(stride, int((avg_len - win) / (target_windows / max(1, len(signals)))))


def _features_per_run(runs: list[list[np.ndarray]], win: int, stride: int,
                      require: bool = True) -> list[np.ndarray]:
    """Vector-hoá features CHO TỪNG test-run (giữ nguyên thứ tự thời gian).

    Trả về list các ma trận (n_windows_run, n_feature), mỗi phần tử là 1 run.
    """
    out = []
    for run in runs:
        feats = windows_to_features(run, win, stride)
        if len(feats):
            out.append(feats)
    if require and not out:
        raise SystemExit("Không có windows nào — kiểm tra dữ liệu/--win/--stride.")
    return out


def split_and_scale(runs_normal: list[list[np.ndarray]],
                    runs_fault: list[list[np.ndarray]],
                    win: int, stride: int, rng: np.random.Generator,
                    max_fault_train: int | None = None
                    ) -> dict:
    """Trả về một dict chứa MỌI thứ các script cần, đã chia train/test và chuẩn hoá.

    Quy ước chia (đúng tinh thần "fault hiếm" + "test normal KHỎE"):
      - NORMAL: trong MỖI test-run, chia theo thời gian thành 3 vùng:
            [0..NORMAL_TRAIN_END)                → TRAIN (normal khỏe)
            [NORMAL_TRAIN_END..NORMAL_TEST_END)  → TEST (normal khỏe chưa thấy)
            [NORMAL_TEST_END..100%)              → GREY, LOẠI hẳn (normal suy giảm
                                                  sát vùng hỏng — không dùng train/test)
         Các vùng từ các run được ghép lại tương ứng.
      - FAULT: trộn ngẫu nhiên → 20% làm TRAIN (phần "hiếm hoi" có nhãn), 80% TEST.
           Test dùng lỗi THẬT chưa thấy khi train → chống data leakage.

    Tham số `max_fault_train` (dùng cho thí nghiệm KHAN HIẾM — SPEC.md §5-A):
        Khi set, giới hạn SỐ window lỗi thật đưa vào train. Điểm mấu chốt để so
        sánh CÔNG BẰNG giữa các mức khan hiếm: TEST luôn là 80% lỗi thật CỐ ĐỊNH
        (không đổi dù `max_fault_train` có đổi), còn THƯỢNG NGUỒN lỗi train bị
        "bóp" xuống `max_fault_train` window; phần vượt chỉ đơn giản bị bỏ (không
        tràn vào test). Nhờ vậy cùng một test có thể đo "model đói lỗi" so với
        "model no lỗi" ở nhiều mức, và augmentation có cứu được hay không.

    DESIGN RATIONALE (định hướng chống rò rỉ — xem SPEC.md §6 "test là vùng đất thiêng"):
      - Mọi lợi thế (augmentation, scaling, chọn feature, chọn threshold) CHỈ được
        làm trong TRAIN. Test luôn là normal KHỎE chưa thấy + lỗi THẬT chưa thấy,
        không có bất kỳ manh mối nào lộ ra từ test → so sánh trước/sau augmentation
        là CÔNG BẰNG (model "sau" không được nhận thông tin từ test).
      - GREY bị loại khỏi CẢ HAI phía (không cho train, không cho test) vì là
        normal đã suy giảm, nhãn mơ hồ. Cho vào train → học "normal xấu" thành lỗi;
        cho vào test → test "bất công" so với model. Bỏ hẳn để cột mốc sạch.
      - Vùng 70–90% giữa NORMAL và FAULT (vùng "mơ hồ" của chuỗi thời gian) cũng
        bị loại theo cùng lý do — chỉ giữ normal khỏe rõ và lỗi rõ ràng.

    Chuẩn hoá z-score: fit mean/std TRÊN TRAIN rồi áp cho train + test + synthetic.
    (fit trên train trước khi chia test = không lộ thông tin test vào scaler.)
    """
    all_normal = flatten_runs(runs_normal)
    stride_eff = _effective_stride(all_normal, win, stride)

    # ---- features theo từng run (giữ thứ tự theo thời gian) ----
    Xn_run_feats = _features_per_run(runs_normal, win, stride_eff)
    Xf = windows_to_features(flatten_runs(runs_fault), win, stride_eff)

    # ---- normal: chia TRAIN / TEST / GREY trong từng run rồi ghép lại ----
    Xn_train_z, Xn_test_z, Xn_grey_z = [], [], []
    for Xn_run in Xn_run_feats:
        i_train = int(len(Xn_run) * NORMAL_TRAIN_END)
        i_test = int(len(Xn_run) * NORMAL_TEST_END)
        Xn_train_z.append(Xn_run[:i_train])
        Xn_test_z.append(Xn_run[i_train:i_test])
        Xn_grey_z.append(Xn_run[i_test:])
    Xn_train_raw = np.concatenate(Xn_train_z) if Xn_train_z else np.empty((0, Xf.shape[1]))
    Xn_test_raw = np.concatenate(Xn_test_z) if Xn_test_z else np.empty((0, Xf.shape[1]))
    Xn_grey_raw = np.concatenate(Xn_grey_z) if Xn_grey_z else np.empty((0, Xf.shape[1]))

    # ---- fault: trộn trước khi chia (bỏ cấu trúc run — chỉ lấy cụm lỗi) ----
    rng.shuffle(Xf)
    n_f_test = int((1 - 0.2) * len(Xf))          # 80% lỗi thật làm TEST — CỐ ĐỊNH
    Xf_test_raw = Xf[-n_f_test:]
    n_f_train = max(1, int(0.2 * len(Xf)))       # 20% lỗi thật dành cho train
    if max_fault_train is not None:               # === thí nghiệm khan hiếm (§5-A) ===
        n_f_train = min(n_f_train, int(max_fault_train))
    Xf_train_raw = Xf[:n_f_train]                 # phần vượt (nếu có) chỉ bị bỏ đi

    # ---- chuẩn hoá (fit trên train) ----
    mu, sd = ft.fit_zscore(np.concatenate([Xn_train_raw, Xf_train_raw]))
    Xn_train = ft.zscore(Xn_train_raw, mu, sd)
    Xn_test = ft.zscore(Xn_test_raw, mu, sd)
    Xn_grey = ft.zscore(Xn_grey_raw, mu, sd)
    Xf_train = ft.zscore(Xf_train_raw, mu, sd)
    Xf_test = ft.zscore(Xf_test_raw, mu, sd)

    return {
        "Xn_train": Xn_train,
        "Xf_train": Xf_train,
        "Xn_test": Xn_test,
        "Xf_test": Xf_test,
        "Xn_grey": Xn_grey,       # normal suy giảm (đã loại) — lưu để thống kê/visual
        "scaler_mu": mu,          # dùng lại để chuẩn hoá synthetic (script 02)
        "scaler_sd": sd,
        "stride_eff": stride_eff,
        "n_normal_train": len(Xn_train),
        "n_fault_train": len(Xf_train),
        "n_normal_test": len(Xn_test),
        "n_fault_test": len(Xf_test),
        "n_normal_grey": len(Xn_grey),
    }


def to_train_test_arrays(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ghép các mảnh đã chuẩn hoá thành (X_base, y_base, X_test, y_test) quen thuộc.

    - X_base  = train "gốc": normal + ít fault thật (model TRƯỚC augmentation).
    - X_test  = normal-chưa-thấy + fault THẬT (dùng đánh giá cho cả trước & sau).
    """
    X_base = np.concatenate([d["Xn_train"], d["Xf_train"]])
    y_base = np.concatenate([np.zeros(len(d["Xn_train"])), np.ones(len(d["Xf_train"]))])
    X_test = np.concatenate([d["Xn_test"], d["Xf_test"]])
    y_test = np.concatenate([np.zeros(len(d["Xn_test"])), np.ones(len(d["Xf_test"]))])
    return X_base, y_base, X_test, y_test