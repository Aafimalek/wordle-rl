"""The canonical state summary the model writes before each guess.

Example after SALET -> YXXXG and MOIST -> XYXGG:

    fixed: _ _ _ S T | has: O(not 2) | out: A E I L M

* fixed: letters known to be in each position (greens).
* has:   letters known to be present more times than the fixed positions explain,
         with the 1-based positions where they were seen yellow.
* out:   letters that were grey and never green or yellow anywhere.
"""

from __future__ import annotations

from .feedback import GREEN, YELLOW
from .game import Turn


def state_line(turns: list[Turn]) -> str:
    fixed = ["_"] * 5
    min_count: dict[str, int] = {}
    yellow_at: dict[str, set[int]] = {}
    seen_present: set[str] = set()
    seen_grey: set[str] = set()
    for t in turns:
        if not t.valid:
            continue
        per_turn: dict[str, int] = {}
        for i, (ch, v) in enumerate(zip(t.guess, t.feedback, strict=True)):
            if v == GREEN:
                fixed[i] = ch
            if v in (GREEN, YELLOW):
                per_turn[ch] = per_turn.get(ch, 0) + 1
                seen_present.add(ch)
            else:
                seen_grey.add(ch)
            if v == YELLOW:
                yellow_at.setdefault(ch, set()).add(i + 1)
        for ch, n in per_turn.items():
            min_count[ch] = max(min_count.get(ch, 0), n)

    has = []
    for ch in sorted(min_count):
        if min_count[ch] > fixed.count(ch):
            pos = sorted(yellow_at.get(ch, ()))
            has.append(f"{ch.upper()}(not {','.join(map(str, pos))})" if pos else ch.upper())
    out = sorted(seen_grey - seen_present)
    return (
        f"fixed: {' '.join(c.upper() for c in fixed)}"
        f" | has: {' '.join(has) if has else '-'}"
        f" | out: {' '.join(c.upper() for c in out) if out else '-'}"
    )
