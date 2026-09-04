#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

echo "==> NASA IMS Bearings (run-to-failure, RUL)"
if [ -d "$DATA_DIR/NASA_IMS/test_1" ] && [ -d "$DATA_DIR/NASA_IMS/test_2" ] && [ -d "$DATA_DIR/NASA_IMS/test_3" ]; then
  echo "    đã có — skip"
else
  mkdir -p "$DATA_DIR/NASA_IMS"
  curl -L --fail -o "$DATA_DIR/NASA_IMS/_ims.zip" \
    "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
  unzip -o -q "$DATA_DIR/NASA_IMS/_ims.zip" -d "$DATA_DIR/NASA_IMS/"
  rm -f "$DATA_DIR/NASA_IMS/_ims.zip"
  echo "    IMS.7z bên trong — giải bằng 7z:"
  find "$DATA_DIR/NASA_IMS" -name "*.7z" -exec 7z x {} -o"$DATA_DIR/NASA_IMS/" -y >/dev/null \;
  echo "    giải rar (1st/2nd/3rd_test.rar):"
  for r in "$DATA_DIR"/NASA_IMS/*_test.rar; do
    [ -f "$r" ] && 7z x "$r" -o"$DATA_DIR/NASA_IMS/" -y >/dev/null
  done
  echo "    (lưu ý: nếu test bị trùng/thiếu, tự kiểm tra thư mục NASA_IMS theo README)"
fi

echo "==> FEMTO/PRONOSTIA Bearings (accelerated life test, RUL)"
if [ -d "$DATA_DIR/FEMTO/Learning_set" ]; then
  echo "    đã có — skip"
else
  mkdir -p "$DATA_DIR/FEMTO"
  curl -L --fail -o "$DATA_DIR/FEMTO/_training.zip" \
    "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
  unzip -o -q "$DATA_DIR/FEMTO/_training.zip" -d "$DATA_DIR/FEMTO/"
  rm -f "$DATA_DIR/FEMTO/_training.zip"
  echo "    giải các zip con (Training/Test/Validation_set.zip):"
  for z in "$DATA_DIR"/FEMTO/*.zip; do
    unzip -o -q "$z" -d "$DATA_DIR/FEMTO/" || true
  done
fi

echo "==> CWRU fault files (Normal Baseline trên site chính thức bị hỏng — skip normal)"
CWRU="$DATA_DIR/CWRU/12k Drive End Bearing Fault Data"
if [ -d "$CWRU" ] && ls "$CWRU"/*.mat >/dev/null 2>&1; then
  echo "    đã có ($(ls "$CWRU" | wc -l) file) — skip"
else
  mkdir -p "$CWRU"
  BASE="https://engineering.case.edu/sites/default/files"
  for n in 105 118 130 106 119 131 169 185 197 170 186 198 209 222 234 210 223 235; do
    curl -sL --fail -o "$CWRU/$n.mat" "$BASE/$n.mat" &
  done
  wait
  echo "    tải xong $(ls "$CWRU" | wc -l) file CWRU fault."
  echo "    Dùng 'normal' từ NASA IMS đầu chuỗi (scripts đã xử lý sẵn)."
fi

echo "==> Done."
find "$DATA_DIR" -maxdepth 1 -type d | sort