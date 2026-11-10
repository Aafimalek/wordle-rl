"""Prompt rendering and output parsing. One stateless prompt per turn.

Each turn's prompt is a single user message holding the instructions and the full
history, so SFT samples, RL rollouts and evaluation all use the same format.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .feedback import to_str
from .game import ERR_FORMAT, ERR_HARD_MODE, ERR_NOT_A_WORD, ERR_REPEAT, Game
from .state import state_line


@dataclass(frozen=True)
class PromptFormat:
    spaced: bool = True  # "G H O S T" (one token per letter) vs "GHOST"
    state_line: bool = True  # write <state>...</state> before <guess>
    # round v6: the model writes its guess as one lowercase word ("<guess>atone</guess>"). Real words are 1-2
    # pretrained tokens, invented ones split into several, so the pretrained vocabulary favours real words.
    # The board history in the prompt keeps the `spaced` letters, so letter-level reasoning is unchanged.
    guess_word: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_FORMAT = PromptFormat()

_GUESS_RE = re.compile(r"<guess>(.*?)</guess>", re.IGNORECASE | re.DOTALL)
_STATE_RE = re.compile(r"<state>(.*?)</state>", re.IGNORECASE | re.DOTALL)
_ERROR_TEXT = {
    ERR_FORMAT: "INVALID (no <guess> found)",
    ERR_NOT_A_WORD: "INVALID (not in word list)",
    ERR_HARD_MODE: "INVALID (hard mode: reuse revealed letters)",
    ERR_REPEAT: "INVALID (already guessed)",
}


def show_word(word: str, fmt: PromptFormat) -> str:
    return " ".join(word.upper()) if fmt.spaced else word.upper()


def show_feedback(fb, fmt: PromptFormat) -> str:
    s = to_str(fb)
    return " ".join(s) if fmt.spaced else s


def show_guess(word: str, fmt: PromptFormat) -> str:
    """How the model writes its own guess inside <guess>...</guess>."""
    return word.lower() if fmt.guess_word else show_word(word, fmt)


def instructions(fmt: PromptFormat, hard_mode: bool, max_turns: int) -> str:
    example = show_guess("crane", fmt)
    lines = [
        f"Guess the hidden 5-letter English word in at most {max_turns} turns.",
        "Feedback per letter: G = right letter, right spot. Y = in the word, wrong spot. X = not in the word.",
    ]
    if hard_mode:
        lines.append("Hard mode: every guess must reuse all green and yellow letters revealed so far.")
    if fmt.state_line:
        lines.append(f"Reply with <state>what you know</state> then <guess>{example}</guess>.")
    else:
        lines.append(f"Reply with <guess>{example}</guess>.")
    return "\n".join(lines)


def render_prompt(game: Game, fmt: PromptFormat = DEFAULT_FORMAT) -> str:
    lines = [instructions(fmt, game.hard_mode, game.max_turns)]
    for i, t in enumerate(game.turns, 1):
        if t.valid:
            lines.append(f"Turn {i}: {show_word(t.guess, fmt)} -> {show_feedback(t.feedback, fmt)}")
        elif t.guess is not None:
            lines.append(f"Turn {i}: {show_word(t.guess, fmt)} -> {_ERROR_TEXT[t.error]}")
        else:
            lines.append(f"Turn {i}: (none) -> {_ERROR_TEXT[t.error]}")
    lines.append(f"Turns left: {game.turns_left}")
    return "\n".join(lines)


def render_completion(game: Game, guess: str, fmt: PromptFormat = DEFAULT_FORMAT) -> str:
    """The ideal model output for this turn: what SFT trains on and what solvers emit."""
    out = f"<guess>{show_guess(guess, fmt)}</guess>"
    if fmt.state_line:
        out = f"<state>{state_line(game.turns)}</state>\n" + out
    return out


def parse_guess(text: str) -> str | None:
    """The last <guess> tag's letters, lowercased, or None. Spaces inside the tag are ignored."""
    found = _GUESS_RE.findall(text or "")
    if not found:
        return None
    letters = re.sub(r"\s+", "", found[-1])
    if len(letters) != 5 or not letters.isascii() or not letters.isalpha():
        return None
    return letters.lower()


def load_format(model_dir) -> PromptFormat | None:
    """The prompt format a model was trained with, saved next to its weights (None for older models)."""
    import json
    from pathlib import Path

    p = Path(str(model_dir)) / "fiveletters_format.json"
    return PromptFormat(**json.loads(p.read_text())) if p.exists() else None


def parse_state(text: str) -> str | None:
    found = _STATE_RE.findall(text or "")
    return " ".join(found[-1].split()) if found else None


def as_messages(prompt: str) -> list[dict]:
    # A single user turn: Gemma's template has no system role, and this keeps every model on equal terms.
    return [{"role": "user", "content": prompt}]
