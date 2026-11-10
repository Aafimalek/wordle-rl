"""Five Letters demo: a 270M-parameter language model plays the five-letter word game.

Live tab: the final model plays a word you choose, one turn at a time, on a ZeroGPU GPU, with the same code, prompt
format and decoding as the paper (model alone, or word-level decoding).
Final-test tab: the exact games scored in the paper's final test (WordleBench, Jeffers), replayed from the records.
"""
# ruff: noqa: I001  (spaces must be imported before torch on ZeroGPU)
from __future__ import annotations

import spaces  # must come before torch

import html
import json
import random
import time
from pathlib import Path

import gradio as gr
import torch  # noqa: F401  (imported after spaces on purpose)
from huggingface_hub import snapshot_download

from fiveletters.game import Game
from fiveletters.policies.hf import HFPolicy
from fiveletters.prompts import load_format, parse_guess, parse_state
from fiveletters.words import load

MODEL_ID = "aafimalek2032/wordle-rl-gemma3-270m"
MODEL_DIR = snapshot_download(MODEL_ID)
FMT = load_format(MODEL_DIR)
VALID = load().guess_set
RECORDS = json.loads((Path(__file__).parent / "data" / "final_test_games.json").read_text(encoding="utf-8"))
TEST_WORDS = sorted({g["answer"] for g in RECORDS["wordlebench:alone"]})

# full-precision weights on the GPU, loaded once at startup; both decoding modes share them
ALONE = HFPolicy(MODEL_DIR, FMT, batch_size=1, device="cuda")
WORD = HFPolicy(MODEL_DIR, FMT, batch_size=1, device="cuda", model=ALONE.model, tokenizer=ALONE.tok,
                constrain="wordle", constrain_mode="word", beams=16)
MODES = {"Model alone": ALONE, "Word-level decoding": WORD}

# ---------------------------------------------------------------------------------------------------------------
# board rendering


def _tiles(guess: str, feedback, valid: bool) -> str:
    cells = []
    letters = (guess or "").upper()[:5].ljust(5)
    for i, ch in enumerate(letters):
        cls = {2: "g", 1: "y", 0: "x"}[feedback[i]] if valid else "rej"
        cells.append(f'<div class="fl-tile fl-{cls}" style="--i:{i}">{html.escape(ch.strip())}</div>')
    return "".join(cells)


def board_html(turns: list[dict], answer: str | None = None, status: str = "", animate_last: bool = True) -> str:
    rows = []
    for k, t in enumerate(turns):
        new = " fl-new" if animate_last and k == len(turns) - 1 else ""
        if t["valid"]:
            note = f'<span class="fl-code">{"".join("XYG"[f] for f in t["feedback"])}</span>'
        else:
            note = '<span class="fl-badge">rejected · not in the word list</span>'
        state = html.escape(t.get("state") or "")
        state_html = f'<div class="fl-state" title="{state}">{state}</div>' if state else ""
        rows.append(f'<div class="fl-row{new}"><div class="fl-n">{k + 1}</div>'
                    f'<div class="fl-tiles">{_tiles(t["guess"], t.get("feedback"), t["valid"])}</div>'
                    f'<div class="fl-meta">{note}{state_html}</div></div>')
    for k in range(len(turns), 6):
        rows.append(f'<div class="fl-row"><div class="fl-n">{k + 1}</div><div class="fl-tiles">'
                    + '<div class="fl-tile fl-empty"></div>' * 5 + '</div><div class="fl-meta"></div></div>')
    head = (f'<div class="fl-answer">hidden word <b>{html.escape(answer.upper())}</b> '
            f'<span>· the model never sees it</span></div>') if answer else '<div class="fl-answer">&nbsp;</div>'
    return (f'<div class="fl-wrap">{head}<div class="fl-board">{"".join(rows)}</div>'
            f'<div class="fl-status">{status}</div>{LEGEND}</div>')


def turns_from_game(g: Game) -> list[dict]:
    return [{"guess": t.guess or "?????", "valid": t.valid, "feedback": list(t.feedback) if t.feedback else None,
             "state": parse_state(t.raw) or ""} for t in g.turns]


def outcome(won: bool, n: int, answer: str) -> str:
    if won:
        return f'<span class="fl-ok">Solved in {n} {"turn" if n == 1 else "turns"}</span>'
    return f'<span class="fl-bad">Not solved in 6 turns</span> · the word was <b>{answer.upper()}</b>'


# ---------------------------------------------------------------------------------------------------------------
# live play (GPU)


@spaces.GPU(duration=60)
def play(word: str, mode: str = "Model alone"):
    """Let the final model play the five-letter word game against a hidden word, one turn at a time.

    Args:
        word: the hidden five-letter word (any of the 12,972 dictionary words). The model never sees it.
        mode: "Model alone" (greedy decoding) or "Word-level decoding" (guesses limited to dictionary words the
            model ranks itself).

    Yields:
        The board as HTML after each turn.
    """
    word = (word or "").strip().lower()
    if len(word) != 5 or not word.isalpha() or word not in VALID:
        yield board_html([], status='<span class="fl-bad">Choose a real five-letter word from the 12,972-word '
                                    'dictionary, or press “Random test word”.</span>', animate_last=False)
        return
    policy = MODES.get(mode, ALONE)
    g = Game(word)
    yield board_html([], word, status='<span class="fl-think">thinking…</span>', animate_last=False)
    start = time.perf_counter()
    while not g.done:
        raw = policy.act([g])[0]
        g.step(raw, parse_guess(raw), VALID)
        if g.done:
            status = f'{outcome(g.won, len(g.turns), word)} <span class="fl-time">· {time.perf_counter() - start:.1f} s</span>'
        else:
            status = '<span class="fl-think">thinking…</span>'
        yield board_html(turns_from_game(g), word, status=status)


def random_word() -> str:
    """A random word from the 100 WordleBench test words (no training game used them as answers)."""
    return random.choice(TEST_WORDS)


# ---------------------------------------------------------------------------------------------------------------
# recorded final-test games (no GPU)

SETS = {"WordleBench (100 words)": "wordlebench", "Jeffers (50 words)": "jeffers"}
SETTINGS = {"Model alone, strict rule": "alone", "Word-level decoding": "word"}


def _games(set_label: str, setting_label: str) -> list[dict]:
    return RECORDS[f"{SETS[set_label]}:{SETTINGS[setting_label]}"]


def record_choices(set_label: str, setting_label: str) -> list[str]:
    return [f'{g["answer"].upper()}  ·  {"won in " + str(len(g["turns"])) if g["won"] else "lost"}'
            for g in _games(set_label, setting_label)]


def summary_html(set_label: str, setting_label: str) -> str:
    games = _games(set_label, setting_label)
    won = [g for g in games if g["won"]]
    mean = sum(len(g["turns"]) for g in won) / max(len(won), 1)
    return (f'<div class="fl-summary"><div><b>{100 * len(won) / len(games):.1f}%</b>won</div>'
            f'<div><b>{len(won)} / {len(games)}</b>games</div><div><b>{mean:.2f}</b>guesses per win</div></div>')


def show_record(set_label: str, setting_label: str, choice: str | None) -> str:
    """Replay one recorded final-test game of the paper.

    Args:
        set_label: "WordleBench (100 words)" or "Jeffers (50 words)".
        setting_label: "Model alone, strict rule" or "Word-level decoding".
        choice: a dropdown entry such as "BATON  ·  won in 4".
    """
    if not choice:
        return board_html([], animate_last=False)
    word = choice.split()[0].lower()
    g = next(x for x in _games(set_label, setting_label) if x["answer"] == word)
    return board_html(g["turns"], word, status=outcome(g["won"], len(g["turns"]), word), animate_last=False)


def change_set(set_label: str, setting_label: str):
    choices = record_choices(set_label, setting_label)
    first = next((c for c in choices if c.endswith("lost")), choices[0])
    return (gr.update(choices=choices, value=first), summary_html(set_label, setting_label),
            show_record(set_label, setting_label, first))


# ---------------------------------------------------------------------------------------------------------------
# page

CSS = """
.gradio-container {max-width: 1100px !important; margin: 0 auto !important;}
.dark .gradio-container {color: var(--body-text-color);}
#fl-hero {padding: 26px 8px 6px; text-align: center;}
#fl-hero .kicker {font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; opacity: .65; font-weight: 700;}
#fl-hero h1 {font-size: 2.6rem; font-weight: 800; letter-spacing: -0.025em; margin: 8px 0 8px; line-height: 1.1;}
#fl-hero h1 span {color: #1baf7a;}
#fl-hero p {font-size: 1.03rem; opacity: .8; margin: 0 auto 14px; max-width: 700px;}
.fl-stats {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 8px auto 4px; max-width: 900px;}
.fl-stat {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 12px 10px; background: var(--block-background-fill);}
.fl-stat b {display: block; font-size: 1.55rem; font-weight: 800;}
.fl-stat span {font-size: .8rem; opacity: .72;}
.fl-links {margin-top: 12px; font-size: .95rem;}
.fl-links a {margin: 0 9px; text-decoration: none; font-weight: 600;}
.fl-wrap {padding: 4px 2px;}
.fl-answer {font-size: .93rem; margin-bottom: 12px; opacity: .9;}
.fl-answer b {letter-spacing: .1em;} .fl-answer span {opacity: .6;}
.fl-board {display: flex; flex-direction: column; gap: 8px;}
.fl-row {display: grid; grid-template-columns: 18px auto minmax(0, 1fr); gap: 14px; align-items: center; min-height: 54px;}
.fl-n {font-size: .78rem; opacity: .5; text-align: right;}
.fl-tiles {display: grid; grid-template-columns: repeat(5, 52px); gap: 6px;}
.fl-tile {width: 52px; height: 52px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
          font-size: 1.55rem; font-weight: 800; text-transform: uppercase;}
.fl-g {background: #1baf7a; color: #fff;} .fl-y {background: #eda100; color: #1b1b1b;} .fl-x {background: #8a8984; color: #fff;}
.fl-rej {border: 2px dashed #eb6834; color: #eb6834;} .fl-empty {border: 2px solid var(--border-color-primary);}
.fl-new .fl-tile {animation: fl-flip .5s ease both; animation-delay: calc(var(--i) * 85ms);}
@keyframes fl-flip {0% {transform: rotateX(90deg); opacity: .15;} 100% {transform: none; opacity: 1;}}
.fl-meta {display: flex; flex-direction: column; gap: 2px; min-width: 0;}
.fl-code {font-family: ui-monospace, Menlo, monospace; font-size: .84rem; letter-spacing: .14em; opacity: .7;}
.fl-badge {font-size: .76rem; font-weight: 700; color: #eb6834;}
.fl-state {font-family: ui-monospace, Menlo, monospace; font-size: .72rem; opacity: .62; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.fl-status {margin-top: 16px; font-size: 1.03rem; min-height: 1.5em;}
.fl-ok {color: #1baf7a; font-weight: 800;} .fl-bad {color: #eb6834; font-weight: 800;}
.fl-think {opacity: .7; font-style: italic;} .fl-time {opacity: .6; font-size: .9rem;}
.fl-summary {display: flex; gap: 22px; flex-wrap: wrap; padding: 4px 2px 8px; font-size: .85rem; opacity: .9;}
.fl-summary b {display: block; font-size: 1.3rem;}
.fl-legend {display: flex; gap: 12px; flex-wrap: wrap; font-size: .78rem; opacity: .8; margin-top: 14px;}
.fl-legend i {display: inline-block; width: 13px; height: 13px; border-radius: 4px; vertical-align: -2px; margin-right: 5px;}
@media (max-width: 760px) {.fl-stats {grid-template-columns: repeat(2, minmax(0, 1fr));}}
@media (max-width: 480px) {
  .fl-tiles {grid-template-columns: repeat(5, 42px); gap: 5px;}
  .fl-tile {width: 42px; height: 42px; font-size: 1.25rem;}
  .fl-row {grid-template-columns: 14px auto;} .fl-meta {grid-column: 2;}
  #fl-hero h1 {font-size: 2rem;}
}
@media (prefers-reduced-motion: reduce) {.fl-new .fl-tile {animation: none;}}
"""

LEGEND = """<div class="fl-legend"><span><i style="background:#1baf7a"></i>right place</span>
<span><i style="background:#eda100"></i>elsewhere in the word</span><span><i style="background:#8a8984"></i>not in the word</span>
<span><i style="border:2px dashed #eb6834"></i>rejected, uses the turn</span></div>"""

HERO = """
<div id="fl-hero">
  <div class="kicker">Research demo · the paper's own model and code</div>
  <h1>Five Letters, <span>270M</span> Parameters</h1>
  <p>A tiny language model, Gemma&nbsp;3&nbsp;270M, trained on free GPUs to play the five-letter word game through text.
  It reads the feedback, writes short notes on what it knows, and guesses one word per turn.</p>
  <div class="fl-stats">
    <div class="fl-stat"><b>97.0%</b><span>WordleBench, word-level decoding</span></div>
    <div class="fl-stat"><b>94.0%</b><span>WordleBench, model alone, leaderboard rule</span></div>
    <div class="fl-stat"><b>98.6%</b><span>500 TextArena games, word-level decoding</span></div>
    <div class="fl-stat"><b>$0</b><span>total compute cost</span></div>
  </div>
  <div class="fl-links">
    <a href="https://github.com/Aafimalek/wordle-rl" target="_blank">Code &amp; paper</a> ·
    <a href="https://huggingface.co/aafimalek2032/wordle-rl-gemma3-270m" target="_blank">Model</a> ·
    <a href="https://huggingface.co/collections/aafimalek2032/five-letters-wordle-rl-6af332879e666b4001ede7ed" target="_blank">Data &amp; every test game</a> ·
    <a href="https://doi.org/10.5281/zenodo.24114970" target="_blank">DOI</a>
  </div>
</div>
"""

ABOUT = """
### How it was trained
1. **Teacher games.** The model imitates a solver that only plays words consistent with every clue.
2. **Spelling drill.** It then followed the clues but could not recall real words that fit them, so it practised writing real words for thousands of boards.
3. **Recovery round.** When the game rejected a non-word, it wrote the same word again. One round of training on boards with its own rejected words fixed most of these repeats.

### Final test, scored once
| Test set | Alone, strict rule | Alone, leaderboard rule | Word-level decoding | GPT-5 (cited) |
|---|---|---|---|---|
| WordleBench (100) | 87.0% | 94.0% | **97.0%** · 3.68 guesses | 99% · 3.59 |
| Jeffers (50) | 74.0% | 88.0% | **96.0%** · 3.75 guesses | 100% · 3.58 |
| TextArena (500) | 85.8% | – | **98.6%** · 3.59 guesses | – |

- **Strict rule** (used in this demo): a guess outside the 12,972-word dictionary gets no feedback and uses the turn.
- **Leaderboard rule:** any five letters get feedback, as on the public leaderboards, whose code checks no dictionary.
- **Word-level decoding:** the output is limited to dictionary words, ranked by the model itself. The code never looks at the clues.
- GPT-5's scores are cited from the leaderboards (other harnesses and prompts, same word sets).

### Notes
- The live tab runs the final model in full precision with greedy decoding, so the same word always gives the same game, as in the paper.
- *Random test word* picks one of the 100 WordleBench words, which no training game used as its answer.
- Live games use a short slice of free ZeroGPU time from your Hugging Face quota; the recorded games use none.
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="stone"), css=CSS,
               title="Five Letters, 270M Parameters") as demo:
    gr.HTML(HERO)
    with gr.Tabs():
        with gr.Tab("Watch it play"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=260):
                    word = gr.Textbox(label="Hidden word", placeholder="e.g. baton", max_lines=1,
                                      info="The model only sees the feedback, never the word.")
                    mode = gr.Radio(list(MODES), value="Model alone", label="Decoding",
                                    info="Word-level decoding limits each guess to dictionary words the model ranks itself.")
                    with gr.Row():
                        rand = gr.Button("Random test word", variant="secondary")
                        go = gr.Button("Play", variant="primary")
                    gr.Examples([["baton"], ["touch"], ["water"], ["acorn"]], inputs=[word], label="Try these")
                with gr.Column(scale=3, min_width=340):
                    live = gr.HTML(board_html([], animate_last=False))
            rand.click(random_word, None, word, api_name="random_word").then(play, [word, mode], live)
            go.click(play, [word, mode], live, api_name="play")
            word.submit(play, [word, mode], live)
        with gr.Tab("Final-test games"):
            gr.Markdown("The exact games scored in the paper's final test, replayed from the records. "
                        "Every game here passed an independent replay audit.")
            first_set, first_setting = "WordleBench (100 words)", "Model alone, strict rule"
            first_choices = record_choices(first_set, first_setting)
            first_pick = next(c for c in first_choices if c.endswith("lost"))
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=260):
                    set_dd = gr.Radio(list(SETS), value=first_set, label="Test set")
                    setting_dd = gr.Radio(list(SETTINGS), value=first_setting, label="Setting")
                    summary = gr.HTML(summary_html(first_set, first_setting))
                    game_dd = gr.Dropdown(first_choices, value=first_pick, label="Game (hidden word · result)")
                with gr.Column(scale=3, min_width=340):
                    recorded = gr.HTML(show_record(first_set, first_setting, first_pick))
            set_dd.change(change_set, [set_dd, setting_dd], [game_dd, summary, recorded])
            setting_dd.change(change_set, [set_dd, setting_dd], [game_dd, summary, recorded])
            game_dd.change(show_record, [set_dd, setting_dd, game_dd], recorded, api_name="recorded_game")
        with gr.Tab("About"):
            gr.Markdown(ABOUT)

demo.queue(default_concurrency_limit=4).launch()
