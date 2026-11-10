# Roadmap: from 95% to frontier-level, and from research code to a usable player

> **About this file.** This is the project's planning record, kept as it was written. Each section is dated, and each decision rule was written before the result it decides on. Early sections show plans and intermediate numbers that later rounds replaced. The final, audited numbers are in [RESULTS.md](RESULTS.md) and in the paper.

Written 2026-11-09, while `fl-v5-final` (final test + GRPO v5) and `fl-v5-seeds` run on Kaggle.
Free resources only (Kaggle T4 x2, laptop CPU, HF Hub free tier).

## Where we are (measured, audited)

| | Our 270M (SFT v5, word-level decoding) | Model alone (no constraint) | GPT-5 (cited) |
|---|---|---|---|
| WordleBench, 100 | 95% (CI 90–99), 3.95 guesses/win | 60% | 99%, 3.59 |
| Jeffers, 50 | 86% (CI 76–94), 3.84 guesses/win | 54% | 100%, 3.58 |
| TextArena clean, 109 | 92.7% | – | – |
| heldout, 463 (dev) | 94.2%, 3.90 guesses/win | 61.8% | – |

## Targets

- **Frontier:** WordleBench ≥ 97%, Jeffers ≥ 96%, ≤ 3.7 guesses per win, with the same decoding.
- **Model alone:** ≥ 90% heldout *without* the constraint, so the result does not depend on a wrapper.
- **Usable:** anyone can play today's Wordle with it in a browser or terminal, on a CPU, in under 2 s per guess.

## Why we lose today (from the 27 heldout losses)

1. **Obscure words picked over common ones** (AXONE, APERT, CRAME, WIDDY). The 12,972-word guess list is mostly rare words, and nothing tells the model that answers are common words.
2. **Trap patterns** (_ATCH: BATCH, HATCH, LATCH, MATCH → WATCH lost). The consistent teacher never "spends" a guess to split many candidates, so the model cannot either.
3. **Too many guesses** (3.90 vs 3.59): same cause. No information-gathering probe moves.
4. **The model alone invents words** (62% alone vs 94% with word-level decoding). It knows the pattern but not the spelling of an unseen real word.

## Plan

Each step has a measured go/no-go on the **select** split (534 never-trained words). Test sets are used only at the end of a round, once per final model, and every use is logged (see "Test-set hygiene").

### Phase 1: polish (laptop, no GPU, about 1 day)

| # | Step | Done when |
|---|---|---|
| 1.1 | `fiveletters play`: interactive terminal play. It suggests a guess, you type the colours (e.g. `gybbg`), and it suggests the next one. Plus `--answer WORD` for self-play. | a test plays a full game through the CLI |
| 1.2 | `fiveletters.solve(board) -> guess`: a small Python API that wraps model + word-level decoding | a doc example runs |
| 1.3 | Hard mode: support and measure it (heldout, select) | numbers in PAPER_NOTES |
| 1.4 | CPU speed: cache the prompt, batch the beam, try int8 / ONNX | ≤ 2 s per guess on the laptop, same results |
| 1.5 | Release: public HF model repo with an honest model card (both numbers, the constraint explained, Gemma terms), and a free HF Space (Gradio, CPU) | a link anyone can open |
| 1.6 | NYT reality check: score on post-2022 NYT answers missing from the classic list (rare-style words) | number in PAPER_NOTES, caveat in the model card |

### Phase 2: close the gap (Kaggle, about 12–16 GPU hours, 2 sessions)

| # | Step | Why it should help | Cost | Go/no-go (select) |
|---|---|---|---|---|
| 2.1 | **Distil the decoding into the model (expert iteration).** Play many games with v5 + word-level decoding on *training* answers only (train + extra, never dev/test words), keep the wins, and fine-tune on them. | The model learns to type the real word it would have picked, which closes "model alone" toward 94% and removes the wrapper dependence. | ~1 h generation + ~2 h SFT | unconstrained select ≥ 85%, word-level not lower |
| 2.2 | **Frequency-aware teacher (SFT v6).** Among the words that fit, the teacher prefers common words, using a public word-frequency list (`wordfreq`), never the answer list. | Fixes loss type 1 inside the model. | laptop traces + ~2.5 h SFT | fewer obscure-word losses; word-level select ≥ 96% |
| 2.3 | **Probe moves for traps.** The teacher stays consistent, except that it plays the best splitting word when ≥ 3 candidates remain and a wrong guess could lose. | Fixes loss types 2–3 with a rule a small model can learn (a narrow case, unlike v3's probes everywhere). | laptop traces + ~2.5 h SFT | mean guesses ≤ 3.75, wins not lower |
| 2.4 | **GRPO from the best of 2.1–2.3.** Reward = win + speed bonus − invalid penalty − clue-contradiction penalty, plus 50% extra answers. | RL optimises exactly wins and guesses; it is where probe timing is learned best. | ~3–4 h | mean guesses ↓, wins not lower |
| 2.5 | **3 seeds** of the final recipe, mean ± sd | Reviewers' standard | ~5 h (2 GPUs in parallel) | – |
| 2.6 | **Final test, once:** WordleBench, Jeffers, TextArena, rare; word-level and unconstrained | the paper's main table | ~0.5 h | – |

Order: 2.1 and 2.2 can run in parallel (one per GPU). 2.3 reuses the 2.2 traces code. 2.4 starts from whichever wins on select.

What we will **not** do: restrict guesses to clue-fitting words or to the answer list in code, or tune anything on WordleBench or Jeffers. Either would make the number meaningless.

### Phase 3: paper (after 2.6)

- Main claim, as the ablation supports it: *word-level dictionary-constrained decoding turns a 270M SFT model into a ~95% Wordle player; clue-consistent teaching makes it faster and interpretable; distillation/RL then makes the model strong on its own.* (Adjust to whatever Phase 2 actually measures.)
- Report both settings everywhere, the seeds, the CIs, the test-set access log, the threats to validity, and the $0 compute.

## Test-set hygiene

- Test-set uses so far: WordleBench and Jeffers were scored for v3c (greedy constraint + unconstrained) and for v5-s1 (declared final test, word-level + unconstrained). Every future use is appended here with date, model and setting.
- 2026-11-10: `sft-gemma270m-v10-s1` (declared final), once, Kaggle `fl-final`: WordleBench, Jeffers and rare under strict, retry-3, open-vocab and word-level; TextArena-adapted seeds 0:500 alone and word-level (`kaggle/plans/final.json`). Operator note: the plan was pushed twice by mistake (kernel versions 1 and 2 of `fl-final`); version 2 was cancelled at once, and version 1 is the only test run. Greedy decoding makes both identical in any case, and no result was seen before the cancel.
- If Phase 2 picks its final model on select only and scores it once, re-using WordleBench and Jeffers is acceptable and is stated. A fresh check: NYT answers from after a cutoff date (1.6) serve as an extra, untouched test.

## Risks

- Expert iteration can overfit to its own mistakes. Keep only wins, cap games per answer, and check the select split.
- The frequency list may favour some words unfairly. Use a general-English list, never Wordle answer statistics.
- Kaggle quota: about 30 h/week per account. Phase 2 fits in one week if each step passes its gate.

## Automatic decision rules (written 2026-11-09, before the GRPO v5 result)

These rules are applied as written, and every decision is recorded in PAPER_NOTES. Baseline = SFT v5 s1: select (word-level) 94.9% wins, 3.91 guesses per win; heldout unconstrained 61.8%. All decisions use **select** and **heldout** only, never a test set.

**After GRPO v5 s1 finishes** (from its evals in `fl-v5-final`):

| Case | Condition | Action |
|---|---|---|
| A. GRPO is enough | select word-level ≥ 97.0% **and** guesses per win ≤ 3.70 **and** heldout unconstrained ≥ 70% | Final recipe = SFT v5 + GRPO v5. Run `grpo-seeds` once `fl-v5-seeds` is done. Then `final-test-2` once on the best model by select (fill FINAL_RUN), then write the **full paper** (see "Paper"). |
| B. GRPO helps but falls short | select word-level ≥ 94.9% and (guesses per win ≤ 3.85 or heldout unconstrained ≥ 66%) | Keep GRPO in the recipe. Add only the Step 2 fixes for what still fails (largest loss type first, from the select losses): many invented words alone → 2.1 distillation; obscure-word losses → 2.2 frequency teacher; trap or too-many-guess losses → 2.3 probe teacher. Then GRPO on top, seeds, final test. |
| C. GRPO does not help | otherwise | Drop GRPO from the recipe. Do Step 2 in order 2.1 → 2.2 → 2.3 from SFT v5, each kept only if it passes its gate, then GRPO, seeds, final test. |

**Model-alone goal (cases B and C; not A, by the author's choice):** the model must also be strong *without* the decoding helper, ideally at frontier level (≥ 97% on select, unconstrained). Step 2.1 (distil the word-level choices into the weights) and the GRPO invalid-word penalty are the levers. Repeat 2.1 (generate wins with the current best model, fine-tune, check select unconstrained) while each round gains ≥ 2 points and quota allows. Report honestly how far the model alone gets; do not claim frontier level for it unless the unconstrained numbers show it.

**Case C execution details (decided 2026-11-09 17:30 UTC, before any Step 2 result):**
- 2.1 distillation round 1 runs first, from SFT v5 s1 (measures what distillation buys).
- To save quota, 2.2 (common words) and 2.3 (probe on traps) are built as **one** new teacher and trained as **one** SFT v6. Their separate effects are measured for free at the teacher level (solver win rate and guesses with and without each rule, laptop CPU), not with extra GPU runs.
- Distillation is then repeated on whichever SFT is better on select (v5 or v6), while each round gains ≥ 2 points unconstrained on select.
- GRPO is tried once more on top of the best distilled model only if ≥ 5 GPU hours remain under the 27 h guard; it is kept only if select improves.
- Then 2 more seeds of the final recipe, one final test, and the full paper. The paper is written whatever the final number is; 97% is the target, not a condition for writing it.
- **No recipe goes to Kaggle before a local end-to-end smoke test with the real model** (tiny limits, laptop CPU). This caught a missing `trace_config.json` in distillation before any GPU time was spent.

**Guards:** at most 2 Kaggle sessions at once. Stop and review the plan if a step would push the week's GPU use past 27 h, if 2 relaunches in a row fail, or if a result looks suspicious (the audit fails, or a jump of > 10 points with no explanation). Every Step 2 change ships with tests and a PR to development before a session uses it. Each final model is scored once on the test sets, and the use is logged under "Test-set hygiene".

## Paper (written when the final results are in)

A full paper, not an outline:
- **Format:** LaTeX, arXiv preprint style, compiled to PDF on the laptop with Tectonic (free). Source in `paper/`; figures generated by a script from `results/` so every number and plot is reproducible.
- **Writing guidance:** follow the official NeurIPS paper checklist and the ML Reproducibility Checklist (read from their sites). No third-party "skills" are installed from the internet; they are not vetted.
- **Sections:** abstract; introduction (why a 270M model, why Wordle, contributions); related work (Wordle solvers, LLM Wordle benchmarks, constrained decoding, SFT/GRPO for games); environment and protocol (rules, feedback rule, splits, benchmark-clean training, test-set hygiene); method (teachers, prompt with state line, SFT, GRPO, word-level decoding, distillation); experiments (main table vs cited frontier, both settings, seeds mean ± sd, ablations: teacher vs decoding, token vs word decoding, GRPO, distillation); analysis (loss types, clue-contradiction rate, guess histograms, example games); limitations and threats to validity; reproducibility and $0 compute; ethics/licence (Gemma terms); appendix (hyperparameters, all runs, retraction of v4, storage incident).
- **Figures:** pipeline diagram; Wordle board renderings of real won/lost games (drawn from `results.jsonl`, not screenshots of other sites); learning curves (select win rate vs step); bar chart of the ablation; guess-count histograms vs GPT-5; win rate with CIs per test set.
- Every number cites its `results/` file. Nothing is posted before the author has reviewed it.

## GPU ledger (account bestman2, week of 2026-11-09; quota page showed 3h03m used at ~13:30 UTC)

| Session | Hours | Note |
|---|---|---|
| fl-v5-s1 | 2.25 | SFT v5 s1 + gate (failed) |
| fl-ablate-word | 0.34 | v3c vs v5 word-level |
| fl-v5-final | ~3.8 | final test v5-s1 + GRPO v5 s1 (Kaggle still saving output) |
| fl-v5-seeds | 2.52 | SFT v5 s2, s3 |
| fl-v5-final (actual) | ~3.8 | ended 17:2x UTC after saving output |
| fl-v5-distill-r1 | 1.44 | 2.1 round 1 from SFT v5 s1 (alone +3.0, word-level −2.2: gate failed) |
| fl-v6 | 2.62 | v6w + v6 SFT and evals (157 minutes) |
| fl-v7 | 2.91 | v7 + v7s fine-tunes from SFT v5 s1 and evals (174.8 minutes). Week total after it: 15.88 h (guard 27 h) |
| fl-v8 | 3.52 | v8 s1 + s2 fine-tunes from SFT v7 s1 and evals (210.9 minutes). Week total after it: 19.40 h (guard 27 h) |
| fl-v9 | 3.55 | v8 s3 (GPU 0) + v9 s1 from v8 s2 (GPU 1) and evals (212.8 minutes). Week total after it: 22.95 h (guard 27 h) |
| fl-v10 | 1.31 | recovery round from v8 s3 (2 seeds) and evals (78.5 minutes). Week total after it: 24.26 h (guard 27 h) |
| fl-final | 0.89 | single final test of v10 s1 (53.6 minutes, after about an hour in Kaggle's queue). Week total: 25.15 h (guard 27 h) |
| fl-v10data | 0.36 | regenerated the v10 recovery samples for the public data release (21.8 minutes; no training). All seven statistics match the original run for both seeds. Week total: 25.51 h (guard 27 h) |

Test-set uses log: 2026-11-09 v5-s1 (declared final) on WordleBench, Jeffers (word-level and unconstrained), TextArena adapted (word-level), rare (word-level).

## Recipe for the model ALONE at ~97% (next phase, not started; needs about 20 GPU hours = next week's quota)

Why the model alone fails: it reasons correctly (clue contradictions 7–9%) but must *spell* the answer letter by letter in the spaced format (`A T O N E`). A 270M model knows English words as subword tokens, not as letter sequences, so for a word it never produced in training it invents a near-miss (ABONE). The helper fixes this by letting it choose among real words; the goal is to put that knowledge into the weights.

| # | Ingredient | Why | Cost |
|---|---|---|---|
| R1 | **Iterated distillation** (2.1, 2–3 rounds) | Teaches the model to type the real word the helper would pick | ~2 h per round |
| R2 | **Whole-word guess format**: keep the spaced `<state>` line, but write the guess as one word (`<guess>ATONE</guess>`) | Real words are 1–2 pretrained tokens, non-words fragment into many, so the pretrained vocabulary pushes toward real words | ~3 h (new traces + SFT), measured on select |
| R3 | **Dictionary spelling task** mixed into SFT: "these clues → a real word that fits", sampled from the public 12,972-word guess list weighted by general word frequency | Trains exactly the missing skill on the whole lexicon. **Fairness choice for the author:** the guess list contains test answers (it is public and any English speaker or LLM knows these words). Option (i): use the full list, disclosed. Option (ii): drop WordleBench/Jeffers words from the spelling targets, stricter but handicapped. Report both if budget allows | ~3 h |
| R4 | **GRPO with an invalid-word penalty** on the distilled model | RL had no signal on the memorised model; once non-words are rare, the remaining errors become learnable | ~3.5 h |
| R5 | **Self-consistency at inference** (sample k guesses, take the majority; no dictionary) | Extra compute without a helper; often fixes one-off misspellings | eval only |
| R6 | 3 seeds + one final test | Reviewers' standard | ~5 h |

Honest odds (estimates, not measurements): R1 alone 70–85%; R1+R2+R3 about 85–93%; 97% alone needs everything to work and is uncertain. Every step keeps its gate on the select split.

## Round v7 (decided 2026-11-10 from the v6 diagnosis in PAPER_NOTES)

Evidence: games without an invalid turn are won 99% of the time; the model types almost only words it typed in training (97.6% of valid guesses); win rate rises from 12.5% to 77% with how often the answer was a training target.

| | Change | Why |
|---|---|---|
| Add | **Balanced spaced-letter drill**: each of the 5,557 common words (Zipf >= 2.0) is a target about 30 times, on boards it fits (about 170k samples). Fine-tune the best spaced SFT on drill + 10–30% game replay, 1 epoch, low learning rate | Direct fix for invented words; the v6w drill had a median of 5 targets per word |
| Add | **Retry on a rejected word**, as a second, labelled protocol (real Wordle rejects a non-word without using a turn). eval-v1 strict stays the headline. Check the WordleBench rule before comparing | Measures the model alone under the real game rule; eval only, no training |
| Drop | Whole-word guess format as the main path (kept as a reported negative result) | 44% vs 59–61% alone |
| Drop | GRPO from the critical path | No gain in v5; revisit only after invalid turns fall below about 5% |
| Keep | Word-level decoding results, distillation as a later polish, seeds and the single final test | |

Gate for v7 on select: model alone >= 75% (from 61–62%) and word-level >= 93.8%. Strict ablation: same drill without the 148 benchmark words as targets.

Update 2026-11-10 (build): the strict ablation `v7s` is stricter than planned: no val, heldout, select, rare or leaderboard word is a target. Replay share is 50% of the v5 traces (about 43% of all samples). Inside the session, the word-level evals run only if the model alone beats the base model's 62% on select (in-queue gate), to save quota.

After v7 (pre-registered):

- alone >= 75% and word-level >= 93.8% on select: the recipe is kept. Next: a second round with more samples per word if the curve was still rising, then 2 more seeds and the single final test.
- alone between 62% and 75%: the drill works but is too weak. Next: more samples per word and deeper boards, one more round.
- alone <= 62% (in-queue gate fails): the drill does not transfer in this form. Stop and re-diagnose from the saved games before spending more quota.

Result 2026-11-10: v7 s1 is 74.9% alone and 97.2% word-level on select (v7s 56.2%). That is the 62-75% band, so round v8 follows.

Round v8 (pre-registered 2026-11-10, before any v8 result):

- Start from SFT v7 s1. Data: balanced drill with 40 samples per common word (v7: 30), deeper boards (1-5 guesses, mostly 3-5) and up to two rejected non-words per board (v7: one), plus the same 50% teacher replay. Same hyperparameters as v7. Reason: v7's invalid turns sit at turns 4-6 (28 / 52 / 60%), where the v7 drill had few or no boards.
- Two seeds in one session (s1, s2: same data, different order), so the decision uses a mean and a seed spread.
- In-queue gate before the word-level evals: alone >= 70% on select.

After v8 (pre-registered), using the mean of the two seeds on select, strict rule:

- v8 alone >= v7 + 2 points (>= 76.9%) and word-level >= 96%: v8 is the final recipe. Next: seed 3 of v8, then declare the final model (the seed with the best select score) and run the single final test.
- v8 within 2 points of v7 (72.9-76.9%) and word-level >= 96%: the drill has saturated. v8 is still the final recipe, because it already has two seeds. Next: seed 3 and the final test, as above.
- v8 alone < 72.9% or word-level < 96%: deeper boards hurt. The final recipe is v7. Next: v7 seeds 2 and 3, then the final test of the best v7 seed.
- In every case the final test runs once, on WordleBench, Jeffers, rare and TextArena-adapted, under strict, retry-3 and open-vocab, and is logged. No further training is chosen with test numbers.

Result 2026-11-10: v8 seed mean on select is 82.8% alone (s1 82.2%, s2 83.3%) and 98.3% word-level. First case: v8 is the final recipe; seed 3 follows, then the final test.

Round v9 (the author's choice on 2026-11-10, after the v8 result and before any v8 s3 or v9 result):

- One Kaggle session, two GPUs. GPU 0: v8 seed 3 as pre-registered (base SFT v7 s1, v8 data). GPU 1, otherwise idle: v9 s1 = the v8 recipe once more, starting from v8 s2 (the best v8 seed on select), with fresh boards (builder seed 9; v8 used 8) and the same replay shares and hyperparameters. Reason: drill rounds gained 13 and then 8 points, and 91 of v8 s1's 95 lost select games still contain a misspelled word.
- GRPO is not run. It was tried twice (v3, v5) without a select gain; this is disclosed in the paper.
- In-queue gates before word-level: v8 s3 alone >= 70%, v9 s1 alone >= 80% on select.

Final model (pre-registered, select split, strict rule, model alone):

- v9 s1 >= the v8 three-seed mean + 2 points and v9 word-level >= 96%: the final model is v9 s1. The paper reports v8 as the recipe with three seeds (mean and sd) and v9 as one further round (one seed, labelled as such).
- Otherwise: the final model is the v8 seed with the best select score alone (ties broken by word-level).
- It is declared in PAPER_NOTES before any test set is scored. Then the single final test (WordleBench, Jeffers, rare, TextArena-adapted; strict, retry-3, open-vocab, and word-level), logged in the test-set uses log.
- No further training this week: about 23 h of GPU are used after this session, the final test needs about 1 h, and the guard is 27 h.

Result 2026-11-10: v8 three-seed mean on select 83.1% alone (82.2 / 83.3 / 83.9); v9 s1 84.8% < 85.15%. Base for v10: v8 s3 (83.9%, word-level 98.7%).

Round v10, recovery (the author's choice on 2026-11-10, after the stuck-on-a-rejected-word diagnosis in PAPER_NOTES; written before any v8 s3, v9 or v10 result). It replaces the line above: one short extra session, and the final test waits for it.

- Base: the model that the "Final model" rule above picks among v8 s1-s3 and v9 s1. It is recorded in PAPER_NOTES as the base, not yet as the final model.
- Data, built inside the session by `fiveletters recovery-data`: the base plays the 4,000 most common training-side hidden words that are not train or extra answers (never a dev or test answer), alone, strict rule. GPU 0 plays the even-ranked words, GPU 1 the odd-ranked ones. Each first rejection of a non-word that leaves a turn gives up to 4 samples: the board as played, including the rejected word, and a real word that fits the clues (a common word, or the hidden word). 30% of the boards show the word rejected twice. Replay: 6% of the v8 data (drill and teacher games).
- Training: from the base, 1 epoch, lr 2e-5, warmup 30, select eval every 100 steps, best checkpoint on select. v10 s1 = GPU 0, v10 s2 = GPU 1 (different words, different order).
- In-queue gate before the word-level evals: alone on select >= the base.
- Final model: the v10 seed with the better select score alone, if it is at least 2 points above the base alone and its word-level select score is at most 1 point below the base's. Otherwise the base. Declared in PAPER_NOTES before any test set is scored; then the single final test as written above.
- Guards as before. In particular a select score alone above 90% stops the plan for an audit before the final test.
- GPU: about 1.5 h for this session and about 1 h for the final test; expected week total about 25.5 h (guard 27 h).

Result 2026-11-10: v10 s1 88.0% alone on select (base 83.9%), word-level 99.1% (base 98.7%). Final model: `sft-gemma270m-v10-s1`, declared in PAPER_NOTES before any test use.
- Amendment before launch (2026-11-10, still before any v8 s3, v9 or v10 result). A laptop check of the data volume with v8 s2 on 100 of the 4,000 words gave 36 events and 54 samples: most boards are so narrow that only the hidden word fits. 4,000 words would give about 1,100 samples per GPU, about 5% of the training mix. Changed, for volume only: play all 8,546 training-side hidden words (4,273 per GPU), replay 4% of the v8 data instead of 6%, and train 2 epochs instead of 1. Everything else, and the final-model rule, is unchanged. No select score was looked at for this change.
