# Evaluation protocol `eval-v1`

This file is frozen. Every number in the paper is produced under these rules. If a rule must change, bump `PROTOCOL_VERSION` in `src/fiveletters/__init__.py`, write the reason below, and rerun every baseline. The harness refuses to mix results from different versions in one folder.

## Word lists

| File | Words | Source |
|---|---|---|
| `resources/answers.txt` | 2,315 | the original Wordle answer list |
| `resources/extra_guesses.txt` | 10,657 | the original extra allowed guesses |

Valid guesses are the union, 12,972 words. `resources/manifest.json` stores a SHA-256 of each list and split. `fiveletters check` verifies them, and every load verifies them again.

## Splits

Built once by `words.make_splits()` with seed `20261108` and committed as text files.

| Split | Words | Use |
|---|---|---|
| `train` | 1,752 | the only answers used for traces and RL |
| `val` | 100 | model selection during training |
| `heldout` | 463 | the main test set. Never trained on, never used to pick checkpoints. |
| `full` | 2,315 | all answers, comparable with solver literature (optimal mean 3.4212) |
| `rare` | 500 | valid guesses that are never answers. Tests generalization to uncommon words. |

`rare` is a random sample of non-answer words, not a frequency-ranked list. The paper must say so.

## Rules

* Normal mode, 6 turns. Hard mode (`--hard`) is reported separately.
* One stateless prompt per turn: instructions, the full history, and turns left (`prompts.render_prompt`).
* Strict: every output uses a turn. No `<guess>` tag, a word outside the 12,972, or a hard-mode violation is an invalid turn with no letter feedback. The prompt shows it as `INVALID (...)`.
* The guess is the letters inside the last `<guess>...</guess>`, with spaces removed.
* Local models decode greedily, one attempt per word. API models use temperature 0.

## Second rule set: `retry-K` (always labelled, never the headline)

The real game rejects a word that is not in its list and does not use a turn. `fiveletters eval --retries K` scores a model under that rule:

* A five-letter guess outside the 12,972 words is rejected. The prompt shows it as `INVALID (not in word list)`, the turn counter does not move, and the model answers again.
* At most K rejections in a row are free. The next one uses the turn, so every game ends. We use K = 3.
* Everything else is eval-v1: 6 turns, greedy decoding, no dictionary help, the same prompt.
* Results go to a folder ending in `-retryK`. `meta.json` and `summary.json` carry `"rule": "retry-K"`. Extra metrics: rejected words per game, games with a rejected word.
* The strict rule above stays the headline. A `retry-K` number is reported next to it, under its own label, and is never compared with a strict number as if they were the same.
* `scripts/audit_eval.py` checks the rule independently: a free turn is always a rejected non-word, at most K come in a row, and the 6-turn limit counts every other turn.

## Third rule set: `open-vocab` (the leaderboards' own rule; always labelled, never the headline)

The two public leaderboards we cite do not check guesses against a word list (see `results/cited/leaderboards_2026-11-09.md` for the code we read). `fiveletters eval --open-vocab` scores a model the same way:

* Any guess of exactly five letters a-z is scored and gets colours, whether or not it is a real word. It uses a turn.
* A guess that is not five letters a-z, or an output with no guess, is still an invalid turn with no feedback. This is stricter than both leaderboards (Jeffers replaces such a guess with a starter word and scores that; WordleBench scores any non-empty guess).
* Colours are the standard Wordle colours, with correct handling of repeated letters. WordleBench's own scorer marks a letter yellow whenever it is anywhere in the answer; we do not copy that.
* Everything else is eval-v1: 6 turns, greedy decoding, no dictionary help, the same prompt.
* Results go to a folder ending in `-open`. `meta.json` and `summary.json` carry `"rule": "open-vocab"`. Extra metrics: non-word guess rate (non-word guesses / all turns), games with a non-word guess.
* `--open-vocab` and `--retries` cannot be combined.
* Use: only next to cited leaderboard numbers, so that both sides are scored under the leaderboards' rule. The strict rule stays the headline. An `open-vocab` number is never compared with a strict number as if they were the same.
* `scripts/audit_eval.py` checks the rule independently: an accepted guess outside the list is allowed only under this rule, must be five lowercase letters and must be marked `nonword`; its colours are recomputed like any other guess.

## Metrics (`eval/metrics.py`)

* win rate, with a 95% bootstrap interval (2,000 resamples, seed 0)
* mean guesses over wins
* mean guesses with a loss counted as 7, with a 95% interval: the headline efficiency number
* invalid turn rate: invalid turns / all turns
* histogram of 1-6 guesses and fails

## Reporting

* Numbers we run ourselves come from `fiveletters eval` and `fiveletters report`.
* Numbers copied from leaderboards (WordleBench, SnorkelWordle, Wordle Arena) are labeled as cited, with the date, because their word sets and prompts differ.
* Frontier-model comparisons are zero-shot. Our model is a specialist. The paper says both.

## Change log

* `eval-v1` (2026-11-08): initial version.
* 2026-11-10: added the separately labelled `retry-K` rule set. The eval-v1 rules did not change, so no baseline was rerun. Strict records keep their exact shape (the new fields appear only when `--retries` is set).
* 2026-11-10: added the separately labelled `open-vocab` rule set (the leaderboards' own rule). The eval-v1 rules did not change, so no baseline was rerun. Strict records keep their exact shape (the new fields appear only when `--open-vocab` is set).
