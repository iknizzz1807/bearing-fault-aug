#!/usr/bin/env python3
"""Render REPORT.md -> REPORT.pdf bằng markdown + weasyprint (hỗ trợ tiếng Việt)."""
import sys, pathlib, re, subprocess
import markdown

base = pathlib.Path(__file__).resolve().parent
md_path = base / "REPORT.md"
html_path = base / "REPORT.html"
pdf_path = base / "REPORT.pdf"

text = md_path.read_text(encoding="utf-8")

# mở rộng bảng + tham chiếu chéo, gạch đầu dòng
html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "codehilite"])

CSS = """
@page { size: A4; margin: 20mm 16mm; @bottom-center { content: "Trang " counter(page) " / " counter(pages); font-size: 9pt; color: #666; } }
body { font-family: "Noto Sans", "DejaVu Sans", sans-serif; font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 19pt; color: #0b3d6b; border-bottom: 2px solid #0b3d6b; padding-bottom: 6px; }
h2 { font-size: 14pt; color: #0b3d6b; margin-top: 22px; border-bottom: 1px solid #cbd5e1; padding-bottom: 3px; }
h3 { font-size: 12pt; color: #14507f; margin-top: 16px; }
h4 { font-size: 11pt; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.3pt; }
th { background: #0b3d6b; color: white; padding: 6px 7px; text-align: left; border: 1px solid #0b3d6b; }
td { padding: 5px 7px; border: 1px solid #d3dbe4; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fb; }
code { background: #eef2f6; padding: 1px 4px; border-radius: 3px; font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt; }
pre { background: #f6f8fa; border: 1px solid #e0e6ec; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 8.8pt; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #0b3d6b; background: #eef3f9; margin: 12px 0; padding: 8px 14px; color: #23374d; }
ul, ol { margin: 6px 0; }
strong { color: #0b3d6b; }
hr { border: none; border-top: 1px solid #d3dbe4; margin: 20px 0; }
p { margin: 7px 0; }
"""

full = f"<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
html_path.write_text(full, encoding="utf-8")
print("HTML written:", html_path)

# HTML -> PDF. Weasyprint module/CLI hiện lỗi trên Python 3.14 (tinycss2.color5),
# nên dùng chromium headless (đã có trên máy) làm renderer chính.
# @page margins được chromium tôn trọng; bỏ header/footer mặc định để giữ style tùy chỉnh.
weasy = subprocess.run(
    ["chromium", "--headless", "--disable-gpu", "--no-sandbox",
     f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
     f"file://{html_path}"],
    capture_output=True, text=True,
)
if weasy.returncode != 0:
    print("chromium error:", weasy.stderr, file=sys.stderr)
    sys.exit(1)
print("PDF written:", pdf_path)
