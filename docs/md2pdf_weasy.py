#!/usr/bin/env python3
"""Chuyển markdown -> PDF bằng chromium headless (TOC tự sinh, syntax highlight)."""
import os
import re
import subprocess
import sys
from html.parser import HTMLParser

import markdown
from pygments.formatters import HtmlFormatter

CSS = """
@page {
  size: A4;
  margin: 1.8cm 1.6cm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; }
}
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Noto Sans", "DejaVu Sans", sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1a1a1a; }
h1 { font-size: 19pt; border-bottom: 2.5px solid #0969da; padding-bottom: 6px; margin-top: 0; color:#0b3d6e; }
h2 { font-size: 14pt; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; margin-top: 1.3em; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 1.15em; page-break-after: avoid; }
h4 { font-size: 11pt; margin-top: 1em; page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
a { color: #0969da; text-decoration: none; word-wrap: break-word; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 1.2em 0; }
blockquote { border-left: 3px solid #0969da; margin-left: 0; padding: 3px 14px; color: #444; background: #f0f6fc; }
blockquote p { margin: 0.4em 0; }
pre { font-family: "Noto Sans Mono", "Liberation Mono", monospace; font-size: 8.4pt; line-height: 1.5;
  background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px; padding: 9px 12px; margin: 0.7em 0;
  white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; }
code { font-family: "Noto Sans Mono", monospace; }
p code, li code, td code, h1 code, h2 code, h3 code, h4 code, blockquote code {
  background: #efefef; padding: 1px 4px; border-radius: 3px; font-size: 0.88em; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 9.3pt; }
th, td { border: 1px solid #c6cdd5; padding: 5px 9px; text-align: left; vertical-align: top; }
th { background: #eef1f5; }
tr { page-break-inside: avoid; }
img { max-width: 100%; }
.toc { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px 18px; margin: 1em 0 1.6em; }
.toc h2 { font-size: 13pt; margin-top: 0; border: none; }
.toc ul { list-style: none; padding-left: 1.2em; margin: 0.2em 0; }
.toc li { font-size: 9.6pt; margin: 0.15em 0; }
.toc a { color: #0b3d6e; }
.codehilite pre { background: transparent; border: none; padding: 0; margin: 0; }
"""


class HeadingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headings = []
        self._cur = None

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3", "h4"):
            d = dict(attrs)
            self._cur = {"level": int(tag[1]), "text": [], "id": d.get("id", "")}

    def handle_endtag(self, tag):
        if self._cur and tag == "h%d" % self._cur["level"]:
            self.headings.append((self._cur["level"], "".join(self._cur["text"]).strip(), self._cur["id"]))
            self._cur = None

    def handle_data(self, data):
        if self._cur:
            self._cur["text"].append(data)


def build_toc(headings):
    parts = ["<nav class='toc'><h2>Mục lục</h2><ul>"]
    prev = 2
    for level, text, hid in headings:
        if level < 2 or not hid or level > 3:
            continue
        text = re.sub(r'<[^>]+>', '', text)
        while level > prev:
            parts.append("<ul>"); prev += 1
        while level < prev:
            parts.append("</ul>"); prev -= 1
        if level == 3:
            parts.append(f"<li style='font-size:8.8pt;color:#555'><a href='#{hid}'>{text}</a></li>")
        else:
            parts.append(f"<li><a href='#{hid}'>{text}</a></li>")
    while prev > 2:
        parts.append("</ul>"); prev -= 1
    parts.append("</ul></nav>")
    return "".join(parts)


def render_md(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    md = markdown.Markdown(extensions=["fenced_code", "codehilite", "tables", "sane_lists", "toc", "attr_list"],
                           extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite", "linenums": False},
                                              "toc": {"toc_depth": "1-3"}})
    body = md.convert(text)
    body = re.sub(r'<details>', '<div class="details">', body)
    body = re.sub(r'</details>', '</div>', body)
    body = re.sub(r'<summary>', '<div class="details-summary">', body)
    body = re.sub(r'</summary>', '</div>', body)
    parser = HeadingParser(); parser.feed(body)
    toc = build_toc(parser.headings)
    h1 = re.search(r"<h1.*?</h1>", body, re.S)
    if h1:
        body = body[: h1.end()] + toc + body[h1.end():]
    pyg_css = HtmlFormatter().get_style_defs(".codehilite")
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}\n{pyg_css}</style></head><body>{body}</body></html>"


css = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/giaotrinh_A1.md"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/giaotrinh_A1.pdf"
html = render_md(css)
tmp = "/tmp/opencode/giaotrinh_tmp.html"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(html)
subprocess.run(["chromium", "--headless=new", "--no-sandbox", "--disable-gpu",
                "--disable-dev-shm-usage", "--no-pdf-header-footer",
                f"--print-to-pdf={out}", "file://" + tmp], check=True, capture_output=True)
print(f"OK  {out}  ({os.path.getsize(out)//1024} KB)")