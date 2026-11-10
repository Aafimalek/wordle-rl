"""One Wordle game under the strict evaluation rules.

Strict means every model output uses up a turn. An output with no parsable guess,
a word outside the 12,972 valid guesses, or (in hard mode) a guess that ignores
revealed clues is recorded as an invalid turn with no letter feedback.

Retry rule (`Game(retries=K)`, a second protocol that is always reported under its own label): like the real game,
a word outside the list is rejected WITHOUT using a turn, and the model sees the rejection and tries again. At most K
rejections in a row are free; the next one uses the turn, so a game always ends. Everything else stays strict.

Leaderboard rule (`Game(open_vocab=True)`, a third protocol, always reported under its own label): the public LLM
leaderboards we cite (WordleBench, Jeffers) have no dictionary. Any five letters a-z are scored and get colours,
and the guess uses a turn. This rule exists only to compare with their numbers under their own rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .feedback import ALL_GREEN, GREEN, YELLOW, feedback

MAX_TURNS = 6

ERR_FORMAT = "format"
ERR_NOT_A_WORD = "not_in_word_list"
ERR_HARD_MODE = "hard_mode"
ERR_REPEAT = "repeated_guess"  # only used by public environments that forbid repeats (TextArena)


@dataclass
class Turn:
    raw: str
    guess: str | None
    valid: bool
    feedback: tuple[int, ...] | None = None
    error: str | None = None
    by_model: bool = True  # False for curriculum prefix moves supplied by the teacher
    free: bool = False  # retry rule only: a rejected non-word that did not use a turn
    nonword: bool = False  # leaderboard rule only: five letters outside the word list, scored anyway

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feedback"] = list(self.feedback) if self.feedback else None
        if not self.free:
            del d["free"]  # strict records keep their original shape
        if not self.nonword:
            del d["nonword"]
        return d


def fits_clues(guess: str, turns: list[Turn]) -> bool:
    """True if `guess` could still be the answer given every valid turn so far (it reproduces each feedback)."""
    return all(feedback(t.guess, guess) == tuple(t.feedback) for t in turns if t.valid)


def count_clue_contradictions(turns: list[Turn], model_only: bool = True) -> tuple[int, int]:
    """(guesses that contradict clues already known, guesses checked). The first valid guess is never counted."""
    bad = checked = 0
    for i, t in enumerate(turns):
        if not t.valid or (model_only and not t.by_model) or not any(p.valid for p in turns[:i]):
            continue
        checked += 1
        bad += not fits_clues(t.guess, turns[:i])
    return bad, checked


def hard_mode_violation(guess: str, turns: list[Turn]) -> str | None:
    """NYT hard mode: greens stay in place, and every revealed letter is reused."""
    for t in turns:
        if not t.valid:
            continue
        need: dict[str, int] = {}
        for i, (ch, v) in enumerate(zip(t.guess, t.feedback, strict=True)):
            if v == GREEN and guess[i] != ch:
                return f"letter {i + 1} must be {ch.upper()}"
            if v in (GREEN, YELLOW):
                need[ch] = need.get(ch, 0) + 1
        for ch, n in need.items():
            if guess.count(ch) < n:
                return f"guess must contain {ch.upper()}"
    return None


@dataclass
class Game:
    answer: str
    max_turns: int = MAX_TURNS
    hard_mode: bool = False
    turns: list[Turn] = field(default_factory=list)
    retries: int = 0  # retry rule: rejected non-words in a row that do not use a turn (0 = strict)
    open_vocab: bool = False  # leaderboard rule: any five letters a-z are scored, as on WordleBench and Jeffers

    @property
    def won(self) -> bool:
        return any(t.feedback == ALL_GREEN for t in self.turns)

    @property
    def turns_used(self) -> int:
        return sum(not t.free for t in self.turns)

    @property
    def done(self) -> bool:
        return self.won or self.turns_used >= self.max_turns

    @property
    def turns_left(self) -> int:
        return self.max_turns - self.turns_used

    @property
    def valid_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.valid]

    @property
    def n_invalid(self) -> int:
        return sum(not t.valid and not t.free for t in self.turns)

    def _free_streak(self) -> int:
        n = 0
        for t in reversed(self.turns):
            if not t.free:
                break
            n += 1
        return n

    def step(self, raw: str, guess: str | None, valid_words, by_model: bool = True) -> Turn:
        if self.done:
            raise RuntimeError("game is already over")
        if guess is None:
            turn = Turn(raw, None, False, error=ERR_FORMAT, by_model=by_model)
        else:
            guess = guess.lower()
            five = len(guess) == 5 and guess.isascii() and guess.isalpha()
            if guess not in valid_words and self.open_vocab and five:
                turn = Turn(raw, guess, True, feedback(guess, self.answer), by_model=by_model, nonword=True)
            elif guess not in valid_words:
                turn = Turn(raw, guess, False, error=ERR_NOT_A_WORD, by_model=by_model,
                            free=self._free_streak() < self.retries)
            elif self.hard_mode and hard_mode_violation(guess, self.turns):
                turn = Turn(raw, guess, False, error=ERR_HARD_MODE, by_model=by_model)
            else:
                turn = Turn(raw, guess, True, feedback(guess, self.answer), by_model=by_model)
        self.turns.append(turn)
        return turn

    def record(self) -> dict:
        """A JSON-ready summary of a finished game, as stored in results files."""
        rec = {
            "answer": self.answer,
            "won": self.won,
            "n_turns": self.turns_used,
            "n_invalid": self.n_invalid,
            "hard_mode": self.hard_mode,
            "turns": [t.to_dict() for t in self.turns],
        }
        if self.retries:
            rec["retries"] = self.retries
            rec["n_rejected"] = sum(t.free for t in self.turns)
        if self.open_vocab:
            rec["open_vocab"] = True
            rec["n_nonword"] = sum(t.nonword for t in self.turns)
        return rec
