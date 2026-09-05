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

echo "==> CWRU bearing data (fault + normal baseline)"
CWRU_BASE="$DATA_DIR/CWRU"
CWRU_FAULT="$CWRU_BASE/12k Drive End Bearing Fault Data"
CWRU_NORM="$CWRU_BASE/Normal Baseline Data"
BASE="https://engineering.case.edu/sites/default/files"
FAULT_IDS="105 118 130 106 119 131 169 185 197 170 186 198 209 222 234 210 223 235"
NORMAL_IDS="97 98 99 100"

if [ -d "$CWRU_FAULT" ] && ls "$CWRU_FAULT"/*.mat >/dev/null 2>&1; then
  echo "    fault đã có ($(ls "$CWRU_FAULT" | wc -l) file) — skip"
else
  mkdir -p "$CWRU_FAULT"
  for n in $FAULT_IDS; do
    curl -sL --fail -o "$CWRU_FAULT/$n.mat" "$BASE/$n.mat" &
  done
  wait
  echo "    tải xong $(ls "$CWRU_FAULT" | wc -l) file CWRU fault."
fi

if [ -d "$CWRU_NORM" ] && ls "$CWRU_NORM"/*.mat >/dev/null 2>&1; then
  echo "    normal baseline đã có ($(ls "$CWRU_NORM" | wc -l) file) — skip"
else
  mkdir -p "$CWRU_NORM"
  for n in $NORMAL_IDS; do
    curl -sL --fail -o "$CWRU_NORM/$n.mat" "$BASE/$n.mat" &
  done
  wait
  echo "    tải xong $(ls "$CWRU_NORM" | wc -l) file CWRU normal baseline."
  echo "    (nếu vài file đọc lỗi 'thiếu bytes', tải lại thủ công theo README)"
fi

echo "==> Done."
find "$DATA_DIR" -maxdepth 1 -type d | sort