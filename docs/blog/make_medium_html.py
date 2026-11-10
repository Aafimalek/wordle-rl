"""Turn medium-article.md into medium-article.html: rich text to copy from a browser and paste into Medium's editor.

Medium has no tables and no Markdown, so the results table becomes a list; images point to the public repo so Medium
imports them on paste. Title and subtitle are kept out of the body (they go in Medium's own title fields).

    python docs/blog/make_medium_html.py
"""
import re
from pathlib import Path

import markdown

HERE = Path(__file__).parent
md = (HERE / "medium-article.md").read_text(encoding="utf-8")
lines = md.splitlines()
title = lines[0].removeprefix("# ").strip()
subtitle = lines[2].removeprefix("### ").strip()
body = "\n".join(lines[3:])


def table_to_list(m):
    rows = [r.strip().strip("|").split("|") for r in m.group(0).strip().splitlines()]
    head, data = [c.strip() for c in rows[0]], rows[2:]
    out = []
    for r in data:
        cells = [c.strip() for c in r]
        parts = [f"{h.lower()}: {c}" for h, c in zip(head[1:], cells[1:], strict=True) if c not in ("–", "")]
        out.append(f"- **{cells[0]}**: " + "; ".join(parts))
    return "\n".join(out) + "\n"


body = re.sub(r"(?m)^\|.*\|\n\|[-| ]+\|\n(?:\|.*\|\n?)+", table_to_list, body)
html_body = markdown.markdown(body, extensions=["fenced_code"])
# images: <p><img alt="..." src="..."></p> -> figure with caption (Medium keeps the caption on paste)
html_body = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" ?/?></p>',
                   r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>', html_body)
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{max-width:720px;margin:40px auto;padding:0 16px;font:18px/1.6 Georgia,serif;color:#111}}
img{{max-width:100%}} figcaption{{font-size:14px;color:#666;text-align:center}} pre{{background:#f4f4f4;padding:12px;overflow-x:auto}}
.meta{{font-family:system-ui,sans-serif;font-size:14px;background:#fff8e1;border:1px solid #f0d78c;padding:12px;margin-bottom:28px}}</style>
</head><body>
<div class="meta">MEDIUM TITLE: <b id="medium-title">{title}</b><br>MEDIUM SUBTITLE: <b id="medium-subtitle">{subtitle}</b><br>
Copy everything BELOW the line into the story body.</div>
<hr>
<article id="medium-body">
{html_body}
</article>
</body></html>
"""
(HERE / "medium-article.html").write_text(page, encoding="utf-8", newline="\n")
print("title:", title)
print("subtitle:", subtitle)
print("html bytes:", len(page), "| images:", html_body.count("<img"), "| tables left:", html_body.count("<table"))
