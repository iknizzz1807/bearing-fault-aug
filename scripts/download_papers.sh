#!/usr/bin/env bash
# Tải PDF các paper tham khảo về thư mục papers/ (link đã verify)
set -euo pipefail

DIR="$(dirname "$0")/../papers"
mkdir -p "$DIR"

echo "==> 1. FaultDiffusion (arXiv 2511.15174)"
curl -sL --fail -o "$DIR/FaultDiffusion_2025.pdf" \
  "https://arxiv.org/pdf/2511.15174"

echo "==> 2. TimeGAN (NeurIPS 2019)"
curl -sL --fail -o "$DIR/TimeGAN_NeurIPS2019.pdf" \
  "https://papers.nips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf"

echo "==> 3. Diffusion-TS (ICLR 2024) — openreview chặn tải script, mở tay:"
echo "    https://openreview.net/pdf?id=4h1apFjO99  (đọc online hoặc tải bằng trình duyệt)"
echo "    (code + tutorial trong repo: git clone https://github.com/Y-debug-sys/Diffusion-TS)"

echo "==> Done."
ls -la "$DIR"