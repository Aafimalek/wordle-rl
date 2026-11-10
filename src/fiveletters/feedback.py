"""The Wordle feedback rule.

Greens are assigned first. Then, left to right, a guessed letter is yellow only
while the answer still has an unmatched copy of that letter. Everything else is grey.
"""

from __future__ import annotations

GREY, YELLOW, GREEN = 0, 1, 2
N_PATTERNS = 3**5
ALL_GREEN_CODE = N_PATTERNS - 1
ALL_GREEN = (GREEN,) * 5
_SYMBOLS = "XYG"


def feedback(guess: str, answer: str) -> tuple[int, ...]:
    if len(guess) != 5 or len(answer) != 5:
        raise ValueError(f"words must have 5 letters: {guess!r}, {answer!r}")
    result = [GREY] * 5
    unmatched: dict[str, int] = {}
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = GREEN
        else:
            unmatched[answer[i]] = unmatched.get(answer[i], 0) + 1
    for i in range(5):
        if result[i] == GREY and unmatched.get(guess[i], 0) > 0:
            result[i] = YELLOW
            unmatched[guess[i]] -= 1
    return tuple(result)


def encode(fb: tuple[int, ...]) -> int:
    """Base-3 code, first letter most significant. All green is 242."""
    code = 0
    for v in fb:
        code = code * 3 + v
    return code


def decode(code: int) -> tuple[int, ...]:
    out = []
    for _ in range(5):
        out.append(code % 3)
        code //= 3
    return tuple(reversed(out))


def to_str(fb: tuple[int, ...]) -> str:
    return "".join(_SYMBOLS[v] for v in fb)


def from_str(s: str) -> tuple[int, ...]:
    s = s.replace(" ", "").upper()
    if len(s) != 5 or any(c not in _SYMBOLS for c in s):
        raise ValueError(f"bad feedback string {s!r}")
    return tuple(_SYMBOLS.index(c) for c in s)
