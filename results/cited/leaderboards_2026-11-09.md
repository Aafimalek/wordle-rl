# Cited leaderboard numbers (fetched 2026-11-09)

These numbers were published by others. We did not run them. In the paper, label them "cited", give the date, and compare only on the same word set.

## WordleBench (wordlebench.com, repo github.com/abronte/wordlebench)

- Word set: 100 past Wordle words, `words.txt` in the repo. Frozen here as split `wordlebench` (checksum in `resources/manifest.json`).
- Method (README): models run through OpenRouter with default settings and "reasoning set to high". Multi-turn, G/Y/B feedback, `<guess>` tag. Max guesses and invalid-word handling are not stated in the README.
- Rules read from the code on 2026-11-10 (`main.py`, repo commit 79436c8 of 2026-09-03):
  - 6 guesses.
  - **No dictionary.** The text inside `<guess>` is scored as it is. There is no word list and no length check.
  - An empty or missing `<guess>` ends the game and marks it as an error.
  - The scorer (`evaluate_guess`) marks a letter yellow whenever it is anywhere in the answer. It does not count repeated letters as real Wordle does.
  - The system prompt asks for analysis and a guess, and shows one worked example.
- Overlap with our splits: 73 words in our train split, 20 in heldout, 3 in val, and 4 not Wordle answers (beaut, grift, laser, spate). Round v4 is trained **benchmark-clean**: no game whose answer is one of these words.
- No date on the page.

| Rank | Model | Success rate | Avg guesses |
|---|---|---|---|
| 1 | gpt-5 | 99% | 3.59 |
| 2 | claude-sonnet-4.6 | 98% | 3.70 |
| 3 | gemini-3.8-flash | 98% | 3.45 |
| 4 | gpt-5.4 | 98% | 3.57 |
| 5 | claude-fable-5 | 97% | 3.51 |
| 6 | gemini-3.7-flash | 97% | 3.53 |
| 7 | gpt-5.5 | 97% | 3.57 |
| 8 | gpt-5.6-sol | 97% | 3.53 |
| 9 | qwen3.8-max | 97% | 3.55 |
| 10 | claude-opus-5 | 96% | 3.60 |
| 20 | gpt-5-mini | 94% | 3.62 |
| 38 | gemma-4-31b-it | 87% | 3.56 |
| 44 | gpt-5-nano | 85% | 3.69 |
| 64 | gpt-oss-120b | 72% | 3.74 |
| 74 | nemotron-3-nano-30b-a3b | 58% | 3.66 |
| 76 | gpt-oss-20b | 55% | 3.85 |
| 85 | gemma-4-26b-a4b-it | 49% | 3.41 |
| 96 | glm-4.7-flash | 7% | 4.00 |
| 97 | trinity-large-thinking | 3% | 2.67 |

The page shows 97 models in total.

## Jeffers Wordle-AI-Benchmark (github.com/georgejeffers/Wordle-AI-Benchmark)

- Word set: 50 curated words, `data/benchmark-words.json`. Frozen here as split `jeffers`.
- Rules (README): 6 guesses max, 5-letter words only. Win rate is the share solved within 6.
- Rules read from the code on 2026-11-10 (`lib/wordle-engine.ts`, repo commit a79acdb of 2026-02-06):
  - 6 guesses.
  - **No dictionary.** A guess is accepted when it matches `^[a-z]{5}$`. `isValidWord` is imported but never called.
  - A repeated guess, or a guess that is not five letters, is replaced by a starter word that was not used yet (crane, slate, adieu, ...). The replacement is scored and the game goes on.
- Results file metadata: `"runDate": "2026-02-06T16:41:39.119Z"`, `"version": "1.0-partial"`, 40 models.
- Overlap: 43 words in our train split, 6 in heldout, and 2 not Wordle answers (words, xenon). v4 is benchmark-clean.
- Read verbatim: GPT-5 100% (50/50), 3.58 avg guesses · OpenAI o3-mini (Reasoning) 100%, 3.74 · OpenAI o1 (Reasoning) 98% (49/50), 3.98.

## SnorkelWordle (snorkel.ai/leaderboard/snorkel-wordle) — archived

- 100 low-frequency words chosen with the Google Books Ngram corpus. **Word list not published**, so we cannot run on it. Cite only.
- gpt-5 94% · Grok 4 93% · o3 92.9% · o4 mini 91.9% · Gemini 3 Pro 91% · gpt-5-mini 91% · o3-mini 90% · … · gpt-oss-20b 65.9% · Llama 3.3 70B 10.2%.

## Not usable

- Wordle Arena (drchangliu.github.io/WordleArena): daily words and per-model tables load by script, and no list is published.

## What this means for the comparison (added 2026-11-10)

- Both leaderboards score any five letters and give colours. Our headline rule (eval-v1, strict) gives a non-word no colours and takes the turn. So the cited numbers were made under an easier rule than our headline numbers.
- For a like-for-like comparison we also score our model under the `open-vocab` rule (`docs/PROTOCOL.md`), which follows the leaderboards on the dictionary and stays stricter than both on malformed output.
- Still different, and to be said in the paper: their prompts, their sampling settings, WordleBench's simplified yellow rule, and the fact that their models are general models playing zero-shot while ours is a specialist.
