"""Figures for the blog post (sans-serif, web sizes), drawn from the same audited summaries as the paper.

    python docs/blog/make_figures.py      # writes docs/blog/img/*.png
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
import build  # noqa: E402  (paper/build.py: summaries, palette, exact rounding)

OUT = ROOT / "docs" / "blog" / "img"
OUT.mkdir(parents=True, exist_ok=True)
S = build.load_summaries()
M = build.M
BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID = (build.BLUE, build.ORANGE, build.AQUA, build.YELLOW, build.INK,
                                              build.INK2, build.GRID)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 11,
                     "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False})


def win(split, run, suffix=""):
    s = S.get(f"{split}:{M}{run}-best{suffix}")
    return None if s is None else float(build.fields(s)["win"])


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1. progress by round: model alone and with word-level decoding (select set)
rounds = [("Teacher\ngames (v5)", ["v5-s1"]), ("Stronger\nteacher (v6)", ["v6-s1"]), ("Spelling\ndrill (v7)", ["v7-s1"]),
          ("Deep drill\n(v8, 3 seeds)", ["v8-s1", "v8-s2", "v8-s3"]), ("One more\nround (v9)", ["v9-s1"]),
          ("Recovery\nround (v10)", ["v10-s1"])]
alone, word = [], []
for _, runs in rounds:
    a = [v for r in runs if (v := win("select", r)) is not None] or [62.0]  # v5 s1: training-time select eval
    w = [v for r in runs if (v := win("select", r, "-constrained-wordle-word")) is not None]
    alone.append(sum(a) / len(a))
    word.append(sum(w) / len(w))
fig, ax = plt.subplots(figsize=(10, 4.6))
style(ax)
x = range(len(rounds))
ax.plot(x, word, color=ORANGE, linewidth=2.5, marker="o", markersize=7, label="with word-level decoding")
ax.plot(x, alone, color=BLUE, linewidth=2.5, marker="o", markersize=7, label="model alone, strict rule")
for i, (a, w) in enumerate(zip(alone, word, strict=True)):
    ax.annotate(f"{a:.1f}%", (i, a), textcoords="offset points", xytext=(0, -20), ha="center", color=BLUE, fontsize=10,
                fontweight="bold")
    ax.annotate(f"{w:.1f}%", (i, w), textcoords="offset points", xytext=(0, 10), ha="center", color=ORANGE, fontsize=10)
ax.set_xticks(list(x), [r[0] for r in rounds])
ax.set_ylim(50, 104)
ax.set_ylabel("games won on the 534-word selection set (%)")
ax.legend(loc="lower right")
ax.set_title("Each round fixed one diagnosed failure", loc="left", fontsize=14, fontweight="bold")
save(fig, "progress.png")

# 2. lost games by cause, before and after the recovery round
lb = json.loads((ROOT / "results" / "analysis" / "games.json").read_text(encoding="utf-8"))["loss_breakdown"]
order = [("v7 s1 (drill)", "v7-s1"), ("v8 s3 (deep drill)", "v8-s3"), ("v10 s1 (recovery)", "v10-s1")]
parts = [("one rejected word, written again and again", ORANGE, lambda d: d["lost_on_one_repeated"]),
         ("other non-words", YELLOW, lambda d: d["lost_with_invalid"] - d["lost_on_one_repeated"]),
         ("only real words (a trap)", BLUE, lambda d: d["lost"] - d["lost_with_invalid"])]
fig, ax = plt.subplots(figsize=(10, 3.6))
style(ax)
ax.yaxis.grid(False)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ys = list(range(len(order)))[::-1]
left = [0] * len(order)
for label, color, f in parts:
    vals = [f(lb[r]) for _, r in order]
    ax.barh(ys, vals, left=left, height=0.6, color=color, label=label, edgecolor="white", linewidth=1)
    left = [a + b for a, b in zip(left, vals, strict=True)]
for y, total in zip(ys, left, strict=True):
    ax.text(total + 1.5, y, f"{total} lost", va="center", fontsize=10, color=INK2)
ax.set_yticks(ys, [o[0] for o in order])
ax.set_xlabel("lost games out of 534 (model alone, strict rule)")
ax.set_xlim(0, max(left) * 1.18)
ax.legend(loc="lower right", fontsize=9.5)
ax.set_title("Why games were lost", loc="left", fontsize=14, fontweight="bold")
save(fig, "losses.png")

# 3. the rule matters: the same model under three rules
models = [("v8 s3 (before recovery)", "v8-s3"), ("v10 s1 (final model)", "v10-s1")]
rules = [("strict: non-words get no feedback", "", BLUE), ("retry-3: non-words are retried", "-retry3", YELLOW),
         ("leaderboard rule: non-words get colours", "-open", AQUA)]
fig, ax = plt.subplots(figsize=(10, 4.2))
style(ax)
w = 0.26
for j, (label, suf, color) in enumerate(rules):
    for i, (_, run) in enumerate(models):
        v = win("select", run, suf)
        ax.bar(i + (j - 1) * w, v, width=w - 0.03, color=color, label=label if i == 0 else None)
        ax.text(i + (j - 1) * w, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_xticks(range(len(models)), [m[0] for m in models])
ax.set_ylim(60, 104)
ax.set_ylabel("games won, model alone (%)")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=9.5, ncol=3)
ax.set_title("Same model, same words, different rule", loc="left", fontsize=14, fontweight="bold")
save(fig, "rules.png")

# 4. final test (from the README image) and the before/after board
for name in ("results.png", "board.png"):
    shutil.copy(ROOT / "docs" / "img" / name, OUT / name)
    print("copied", name)

# 5. method overview, cropped from the compiled paper (page 2)
import fitz  # noqa: E402

pdf = ROOT / "paper" / "main.pdf"
if pdf.exists():
    page = fitz.open(pdf)[1]
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 1 or "lines" in b]
    cap = next(r for r in page.search_for("Method overview"))
    top = page.search_for("Teacher games")[0].y0 - 14
    clip = fitz.Rect(page.rect.x0 + 40, top, page.rect.x1 - 40, cap.y0 - 4)
    page.get_pixmap(dpi=260, clip=clip).save(OUT / "overview.png")
    print("wrote overview.png")
