# Paper notes

> **About this file.** This is the project's results log, kept as it was written: dated entries, newest at the end, nothing deleted. Early sections show goals and numbers that later rounds replaced. The final, audited numbers are in [RESULTS.md](RESULTS.md) and in the paper; the final test is the last section of this file.

A running record of decisions, findings and numbers for the paper. Every number has a source file or command. Add to it as results arrive, and never delete: mark superseded entries instead.

## Claim (the goal at the start of the project; see the final test at the end for what was reached)

"Among language models that play Wordle through text, our ~270M model has the best win rate and mean guess count on [named benchmarks], above models up to N× larger, trained only on free notebook GPUs."

Levels: (1) best sub-1B LM, and above the public Wordle RL checkpoints; (2) above frontier LMs; (3) within 5% of optimal mean guesses (≤ 3.59) with ≥ 99.5% wins.

## Benchmark (eval-v1, frozen; docs/PROTOCOL.md)

- Word lists: 2,315 answers + 10,657 extra guesses = 12,972 valid guesses, checksummed (`resources/manifest.json`).
- Splits (seed 20261108): train 1,752 · val 100 · heldout 463 · full 2,315 · rare 500.
- Strict rules: every output uses a turn; invalid = no `<guess>`, not a word, or a hard-mode violation.
- Metrics: win rate with a 95% bootstrap CI, mean guesses with a loss counted as 7, invalid-turn rate.

## Teacher

- Lookahead beam search, full Wordle vocabulary, opener SALET: **3.4216** mean guesses, 100% wins on all 2,315 answers. The proven optimum is 3.4212 (Bertsimas & Paskov). Source: `results/eval-v1/full/solver-LookaheadSolver`.
- Greedy entropy: 3.4721 (full). Random word consistent with the clues: 98.5% wins, 4.115 (full). On heldout: lookahead 3.417, entropy 3.518, random-consistent 98.7% / 3.996.
- **Finding for the motivation section:** guessing any word that fits the clues already wins about 99%. So the ~94% that frontier LMs reach on leaderboards (WordleBench) means they fail to *track the clues*. That failure is what our state line targets.
- **Recovery states use a weaker teacher (must be stated).** Traces include off-script states on purpose: 0–2 random opening words, 10% noise moves, and 10% of games with one invalid turn. These teach recovery. Without them, one mistake would put the model in states it never saw (the covariate-shift problem behind DAgger). Deep search on these mostly new states took more than a day on a laptop, so they use greedy entropy (≈ 3.47–3.48 mean). Normal states use the lookahead teacher.

## Public environments (measured)

- TextArena `Wordle-v0` (textarena 0.7.4) and verifiers `wordle` (verifiers 0.4.1) share one engine. verifiers makes task i with `random.seed(i)` then `ta.make` and `reset`, so seed i is the same secret in both.
- **Feedback rule matches TextArena exactly:** 10,000 pairs (5,000 with repeated letters), 0 mismatches. `results/eval-v1/crosscheck_textarena.json`.
- **Small and overlapping word set:** TextArena secrets are 156 five-letter nouns from NLTK's basic English list. All 156 are Wordle answers, but 122 (78%) are in our train split, 32 in heldout and 2 in val. → We report a **clean** subset (secrets not in our train answers).
- **Dictionary gap:** TextArena accepts only 8,019 of the 12,972 Wordle guesses, and it rejects SALET. One invalid move per game is forgiven and the second loses. In TextArena, the full-vocabulary entropy solver lost 15 of 40 games on invalid moves.
- **Response:** the "shared" vocabulary (8,019 words plus the 16 answers TextArena does not list = 8,035). Best opener inside it: CRATE. Lookahead + shared + CRATE = **3.4281**, 0 losses (TRACE 3.4285, SLATE 3.4311, greedy opener RAISE 3.4687). The restriction costs 0.007 guesses versus SALET. Traces v2 use it.
- Modes: `native-verifiers` (verifiers' system prompt, latest feedback per turn), `native-textarena` (full observation), `adapted` (our prompt, engine-judged). Each public checkpoint is scored in its native mode. Ours is scored in adapted mode.

## Training data

- Traces v1 (superseded): full vocabulary, SALET, 138,635 samples.
- **Traces v2 (used):** shared vocabulary, CRATE, prefix probabilities (0.6, 0.25, 0.15), noise 0.1, alternative opener 0.3, invalid-turn 0.1, 30 games per train word, **138,701 samples**. Private HF dataset `aafimalek2032/five-letters-traces-v2`.
- Gemma 3 completion lengths: median 44 tokens, max 82 (generation limit 96, so no truncation).

## Finding: SFT on training answers memorizes them (central to the paper)

- After SFT v2, Gemma 3 270M wins **20/20** games whose answer is a training answer, but only **13/20** on unseen val answers (48% on 100 val games, about 54% on heldout while running).
- Every invalid turn on unseen answers was an invented non-word (WRAND, WDARF, ANNRY, LADLY). The model tracks the clues but cannot produce a real word that fits them unless it saw that word as an answer.
- The unseen answers do appear in the traces (438/463 heldout, 98/100 val), but only as probe guesses, never as the final answer.
- GRPO rollouts on training answers already win 89–98% (sampled), so RL there has little signal (`no_signal_groups` 0.25–0.63).
- **Fix (traces v3):** add games whose answers are 1,610 common non-answer words. These are in the shared vocabulary, outside every test split, and have wordfreq zipf ≥ 2.5 (the 10th percentile of Wordle answers is 2.52). Frozen in `resources/extra_answers.txt` with a checksum. The teacher for these games is greedy entropy over all 12,972 candidates. GRPO can mix them in too (`rollout.extra_answer_frac`).
- Ablation for the paper: SFT v2 (training answers only) vs SFT v3 (plus extra answers), scored on heldout and rare.
- **Extra answers help:** heldout 55.7% → 64.6% (v2 → v3), CIs barely overlap. Remaining weakness: invented non-words (≈23% of turns), which TextArena punishes hardest (the second invalid move loses the game). GRPO v3 penalizes invalid turns and plays half its games on extra answers.
- **GRPO v3 vs SFT v3:** invalid turns down (heldout 23.3% → 19.4%), rare words up (35.6% → 41.2%), heldout not better (64.6% → 61.8%, CIs overlap). Validation picked step 550 at 66%, so selection on val is slightly optimistic. Heldout is the honest number. RL mainly improved robustness on unfamiliar answers, not the core heldout score. **Largest RL gain: TextArena clean subset 46.8% → 64.2% (+17 points)**, where the second invalid move loses the game. Fewer invalid moves is worth most there.
- **Constrained decoding:** removes every invalid word. Held-out: SFT v3 64.6% → **79.5%**, GRPO v3 61.8% → 74.1%. With the constraint, SFT v3 beats GRPO v3 on held-out by 5 points (CIs touch). GRPO's gains came mostly from fewer invalid words, which the constraint already removes, and it may have narrowed which real words the model picks. Check after v4 whether GRPO helps on top of the constraint (for example GRPO trained with constrained rollouts).
- ~~Negative result (v4)~~ **RETRACTED (2026-11-09):** the SFT v4 heldout evals (55.1% / 72.4%) did not score the final model. From 23:29 UTC every Hub upload failed (see the storage incident below), so `sft-gemma270m-v4/best` on the Hub was a stale checkpoint from about step 2,500 (val 53%), not the final best (val 74%). Those numbers measure a mid-training model and must not be used. GRPO v4 also started from that stale checkpoint and is invalid. Model selection on a 100-game val set is still noisy, so use heldout for all comparisons.
- Confirmed from three angles: heldout 55.7% vs training-answer games ~100%; TextArena clean subset 49.5% vs 87% overall (78% of its secrets are training answers); GRPO v2 on training answers flat for 600 steps.

## Leaderboard comparison (frontier, level 2)

- Usable leaderboards (published word sets): **WordleBench** (100 words; best: gpt-5 99% / 3.59, gemini-3.8-flash 98% / 3.45) and **Jeffers** (50 words; GPT-5 100% / 3.58). Snapshot with sources: `results/cited/leaderboards_2026-11-09.md`. SnorkelWordle's word list is not published, so it is cited only.
- **Contamination check:** 73 of the 100 WordleBench words and 43 of the 50 Jeffers words are in our train split. Earlier rounds (v2, v3) are therefore **not** valid on these sets.
- **Round v4 is benchmark-clean:** SFT drops every game whose answer is a leaderboard word (`data.exclude_benchmark_answers`), and GRPO never uses them as rollout answers. On WordleBench and Jeffers, v4 has never trained on a test word.
- Honest framing: frontier models have probably seen past Wordle answers on the internet, but not trained to map clues to them. Our numbers are produced by our harness on their word sets. The rules (6 guesses, strict) are stated, and their invalid-word handling is not published.

## Diagnosis of the remaining losses (SFT v3 + constrained decoding, heldout)

- In **all 95 lost games**, at most 2 words still fit the clues before the 6th guess (median 1): the answer was essentially determined.
- **86% of guesses in lost games contradict clues already known** (e.g. answer AGILE: after ANILE → GXGGG the model guesses APIAN, then AVIAN). In won games, 51% of guesses after turn 1 also contradict known clues.
- Cause: the teacher's optimal moves are often *probe* words that deliberately break the clues (best for information), which a 270M student cannot reproduce reliably. The student learns "guess something" rather than "guess a word that fits what I know".
- Implication: the main lever is **constraint-following inside the model**. A random word that fits the clues already wins 98.7% (heldout). Source: `results.jsonl` of `hf-sft-gemma270m-v3-best-constrained-wordle`, analysed with `EntropySolver.candidates`.

## What each headline number means (write it exactly like this)

- "79.5%" = SFT v3 won 368 of 463 heldout games (answers never used for training or checkpoint selection), standard rules (6 guesses, invalid outputs use a turn), greedy decoding, **with dictionary-constrained decoding**. 95% CI 75.8–83.2%. The same model **without** the constraint wins 64.6% (CI 60.0–68.9). Always report both, side by side, and label the constraint as an inference aid (real Wordle also rejects non-words).
- Efficiency: mean guesses on wins about 4.2, versus 3.42 for the optimal policy. The model wins but is not efficient.

## Threats to validity and what is still missing (must be in the paper)

1. **Single seed.** Every result is one training run. Before the final table: 3 seeds of the final recipe, reporting mean ± sd.
2. **Heldout reuse.** Several variants (v2, v3, v4, GRPO, constrained) were scored on heldout, and some decisions used those scores, which risks adaptive overfitting. Mitigation: treat **WordleBench (100) and Jeffers (50)** as the **final test, evaluated once** on the final model (clean for v3c and later benchmark-clean models). Report heldout as a development set.
3. **Small val set (100 games).** It misled once (v4: val 74% vs a stale heldout number). Use a larger val set next round.
4. **Frontier comparison is cited, not re-run.** Leaderboard numbers come from others' harnesses (OpenRouter, "reasoning high", unknown invalid-word handling). State this, and compare only on the same word sets.
5. **Retracted:** SFT v4 / GRPO v4 results (stale Hub checkpoint). Do not use them.

## Storage incident (2026-11-09) and fix

- The private runs repo reached **99.2 GB used storage**, the free private-storage cap. Each checkpoint upload with optimizer state was about 3 GB, and deleted history still counts until the Hub garbage-collects it (asynchronous). From 23:29 UTC all large uploads failed with no message. This lost the v3-clean model (its evals had nothing to load) and left stale SFT v4 weights on the Hub.
- Fix: (1) evaluations resolve `hub:` models from the session's own disk first; (2) finished best/final models are also exported to the Kaggle output folder (`/kaggle/working/models`), which survives normal session ends and can be attached to later sessions (`job.py push --from-kernel`); (3) Hub uploads omit optimizer state by default (3× smaller), retry, and log the real HTTP error; (4) resume checkpoints of finished runs were deleted.

## Round v5 design (2026-11-09, ready to launch)

Diagnosis from v3 (above): unconstrained losses are mostly invented non-words, and with constrained decoding 86% of guesses in lost games contradict clues already known. v5 targets clue-following, and fixes the selection problems in threats 1–3 without removing more training answers.

- **Consistent teacher.** Traces `aafimalek2032/five-letters-traces-v5` (143,055 samples, train answers, 30 games/word) and `-v5-extra` (118,061 samples, 1,610 extra non-test answers, 20 games/word). The lookahead solver may only guess words that fit every clue (`teacher = "consistent"`), while the game stays in normal mode. Config: shared vocab, CRATE opener, noise 0.1, invalid_prob 0.1, spaced letters and state line (`data/traces/v5/trace_config.json`).
- **Selection set `select` (534 words).** val + heldout minus the 148 leaderboard words. None of them is ever a training answer, so selection costs no training data (CI about ±4 points instead of ±9 on the old 100-game val). Leaderboard sets stay the untouched final test. Heldout is now partly used for selection, so heldout numbers from v5 on are development numbers only.
  - Superseded design: `val300` (300 train words removed from training). Dropped before launch, because v3c showed that removing 148 training answers already cost about 6 points unconstrained (64.6% → 58.5%, one seed each).
- **Benchmark-clean.** WordleBench and Jeffers answers are excluded from training answers (`exclude_benchmark_answers`).
- **GRPO v5.** Init from the SFT v5 best of the same seed, 50% extra answers, the consistent curriculum teacher, and `consistency_penalty = 0.2` per model guess that contradicts known clues. New metric `clue_contradiction_rate` in training logs and eval summaries.
- **3 seeds** of SFT and GRPO; report mean ± sd. Heldout is a development set only.
- **Final test, once:** WordleBench (100), Jeffers (50), TextArena adapted, rare; unconstrained and constrained, on the model chosen by the select split. Gate: GRPO and extra seeds start only if SFT v5 + constraint beats 79.5% on heldout. Plan: `kaggle/plans/v5.json`.
- v3c session (commit ee40e80): SFT 114 min, all 9 commands exit 0; heldout eval takes 2–3 min, so the 534-game selection eval is cheap. Model kept: Kaggle output → laptop `kaggle_runs/gemma-v3c/models/` and Hub `sft-gemma270m-v3c/best`.
- Storage: the HF runs repo was cleaned (now about 1 GB). Finished runs now delete their `latest` resume slot after `best` and `final` upload successfully.

## Round v5 result and diagnosis (2026-11-09)

Session `bestman2/fl-v5-s1` (commit d78d106): SFT 2 h 10 min, then the gate failed and both queues stopped (exit code 76), as designed. About 2.3 GPU hours used.

- **What v5 fixed:** clue-following. Without the constraint, the share of guesses that contradict known clues fell from 49.0% (v3c) to **7.5%**. Wins rose from 58.5% to 61.8%, and mean guesses on wins fell from 3.99 to 3.78. The CIs overlap, so the win gain is not significant on one seed.
- **What is left:** invented words. 175 of the 177 unconstrained losses contain non-words. **75% of the non-words fit every clue** (406/539; e.g. answer ATONE: ABONE, ASONE, AMONE; answer AUDIT: ABBIT, AFFIT). The model knows the pattern but cannot recall a real word that fits it. It also repeats an identical rejected non-word 34 times.
- **Why the constraint now helps less (+8.4 vs +18.6 for v3c):** the constraint is *token-greedy*. It commits to the most likely allowed first letters and is then forced into whatever dictionary word continues them (ADLED → ADDED, AWAGE → AWAYS). Those words often break clues (32.6% contradictions under the constraint, vs 7.5% without). The v3 model guessed memorized probe words, which survive greedy constraining better.
- Next (free, laptop CPU first): **word-level constrained decoding**, where the model scores every whole dictionary word and the best one is taken. The model still chooses: the code only enforces "a real, not yet played word", never "fits the clues". This tests whether v5's clue-following turns into wins once non-words are impossible.

- **Word-level decoding result (heldout, laptop):** 94.2%. The 27 losses are of two kinds. (1) Classic Wordle traps with many fitting words: WATCH after BATCH, HATCH, LATCH, MATCH; CRAZE after CRANE, CRAVE. (2) Obscure dictionary words chosen over common ones (AXONE, AGONE, APERT, CRAME, WIDDY), because the guess list has 12,972 words and the model does not know which ones are common. Mean guesses on wins (3.90) is above GPT-5's 3.59: the model rarely plays information-gathering probes.
- Open question for the paper (ablation running on Kaggle): does v3c also gain this much from word-level decoding, i.e. is the teacher or the decoding the main driver?

## Decision after GRPO v5 (2026-11-09 16:35 UTC, pre-registered rules in ROADMAP)

- Inputs: GRPO's best checkpoint is step 0 (= SFT v5 s1). Select word-level 94.9% (needed ≥ 97.0% for A), 3.91 guesses per win (needed ≤ 3.70 for A, ≤ 3.85 for B), heldout unconstrained 61.8% for the same weights (needed ≥ 70% for A, ≥ 66% for B). GRPO did not improve the selection score at any of its 6 checkpoints.
- **Decision: case C, "GRPO does not help".** GRPO leaves the recipe. Step 2 runs in order from SFT v5: 2.1 distillation (model-alone goal, repeated while it gains ≥ 2 points on select unconstrained), then 2.2 frequency-aware teacher, then 2.3 probe teacher, each kept only if it passes its gate.
- Likely reason (hypothesis, not tested): with greedy rollouts on mostly memorised training answers, 25% of groups had no reward signal and the learning rate decayed to 1.5e-6; RL could not move a model whose main failure (invented words) is about vocabulary, not strategy.

## Ablation: teacher vs decoding (2026-11-09, Kaggle `bestman2/fl-ablate-word`, all audited)

| | v3c (optimal teacher) | v5 (consistent teacher) |
|---|---|---|
| heldout wins, word-level decoding | 94.2% | 94.2% |
| select wins, word-level decoding | 94.9% | 94.9% |
| mean guesses on wins (heldout) | 4.08 | **3.90** |
| clue-contradiction rate (heldout) | 44.0% | **9.4%** |
| heldout wins, token-greedy constraint | 77.1% | 70.2% |
| heldout wins, unconstrained | 58.5% | 61.8% |

- **Word-level decoding is the main driver of the win rate** (+17 to +24 points for both models). The consistent teacher does **not** raise wins under it; it makes the model play like its teacher: about 0.2 fewer guesses per win and far fewer clue-breaking guesses.
- So the paper's claim must be: "word-level dictionary-constrained decoding turns a 270M SFT model into a ~94% Wordle player; teaching clue-consistent play makes it faster and more interpretable." Not "the consistent teacher raised the win rate."
- Reproducibility: the laptop CPU and the Kaggle T4 gave identical results for the same model and setting.
- The partial laptop WordleBench run (32 of 100 games) was stopped for speed and moved, **unread**, to `kaggle_runs/discarded/`. The declared final test runs once on the T4 instead (`fl-v5-final`).

## Final test: declared before running (2026-11-09)

- **Model:** `sft-gemma270m-v5-s1/best` (SHA in `kaggle_runs/v5-s1/models/...`, also Hub `aafimalek2032/five-letters-runs`). It is the only model trained with the final recipe; GRPO did not run.
- **Settings:** (a) unconstrained greedy, and (b) word-level constrained decoding, beam 16, `wordle` guess list (12,972), no repeats. Both reported, side by side. The decoding mode was chosen on heldout (development), never on a test set.
- **Test sets, each scored once:** WordleBench (100), Jeffers (50); also TextArena Wordle-v0 adapted (seeds 0–499) and rare (500). All leaderboard answers were excluded from training.
- The v3c word-level ablation (running on Kaggle) is for the analysis only and will **not** change this choice.
- Hardware: laptop CPU, fp32 (same weights and code as Kaggle; heldout re-check on a T4 running).

## Distillation round 1 decision (2026-11-09 19:05 UTC)

- Model alone (select, unconstrained): 62.0% → 65.0% (+3.0, passes the ≥ 2-point rule). Heldout alone: 61.8% → 65.7%.
- With the helper (select, word-level): 94.9% → 92.7%, **below the pre-set floor of 93.8%, so the gate FAILS** and `v5-d1` is not a candidate model. Heldout word-level: 94.2% → 92.9%.
- Reading: training only on its own word-level choices makes the model better alone but narrows what it rates as plausible, which the helper relied on (clue contradictions under the helper rose 9.0% → 12.5%). Standard remedy, to be used in the next distillation round: mix the original SFT data back in (replay), so the model keeps its earlier calibration while learning to spell.
- Session `fl-v5-distill-r1`: 86.6 min, all commands exit 0.

## Round v6 (2026-11-09 18:00 UTC, launched as `bestman2/fl-v6`)

Teacher-level ablation (laptop CPU, training answers only: 1,638 train-split answers minus leaderboard words with the answer pool, and 400 extra answers with the full 12,972-word pool; shared vocab, opener CRATE). Source: `results/teacher/v6-teacher-ablation.txt`.

| Teacher | train: wins, guesses/win | extra: wins, guesses/win |
|---|---|---|
| consistent, greedy (v5-like) | 99.69%, 3.540 | 82.25%, 4.252 |
| + common-word tie-break | 99.69%, 3.543 | 93.25%, 4.038 |
| + probe on trap sets | 100.00%, 3.492 | 95.50%, 4.291 |
| **common + probe (v6 teacher)** | **99.94%, 3.502** | **98.75%, 4.149** |

Data (all audited: 0 dev/test answers, every target parses): `five-letters-traces-v6` 141,144 and `-v6-extra` 108,391 samples (spaced format); `-v6w` 141,144 and `-v6w-extra` 148,391 (whole-word guesses; extra includes the 40,000-sample spelling drill, hidden-word pool 4,746 common training-side words, zipf ≥ 2.0). Generation took 3.5 + 5.2 minutes on the laptop (v5's lookahead teacher took about 1–2 hours).

**Result (2026-11-10, session `bestman2/fl-v6`, commit fff63d4, 157 minutes, all 8 evals pass the independent audit).** The new teacher gives no gain for the model alone (61.2% vs 62.0% on select) and the same win rate with word-level decoding (94.9%), with more guesses per win (4.08 vs 3.91). The whole-word format is clearly worse (44.9% alone, 54.5% word-level): invalid turns fall to 5%, but half of its guesses contradict the clues. Decision: keep the v5 teacher and the spaced format; round v7 starts from `sft-gemma270m-v5-s1`. Rows are in the results log.

Session `fl-v6`: GPU 0 trains v6w (teacher + whole-word guesses + drill); GPU 1 trains v6 (teacher only, v5 format) as fallback and ablation. Both are scored on select and heldout, alone and with word-level decoding.

Fairness of the spelling drill (user decision 2026-11-09: use the full public word list and disclose it): boards are built only around training-side hidden words, so no board is built around a dev or test word; the target words may be any of the 12,972 public dictionary words, test answers included, as general English knowledge.

## Engineering for free GPUs (reproducibility section)

- T4: no native bf16 (newer PyTorch reports it, but it is emulated and slow). Gemma 3 is known to overflow in fp16. Weights stay fp32. Training checks fp16 against fp32 on real batches at startup and uses fp16 only if they agree (passed for Gemma 3 270M and SmolLM2). Automatic fp32 fallback. All reported evaluations use fp32 weights unless the meta file says otherwise.
- Measured SFT throughput on one T4 in fp16, 1 epoch over v2: Gemma 3 270M about 1.1 h (≈ 1,670 completion tokens/s), SmolLM2-360M about 1.6 h, SmolLM2-135M about 0.75 h.
- Failures met and fixed, with the free-tier lesson:
  1. Kaggle allows 2 GPU sessions at once.
  2. Kaggle's 20 GB output disk filled with checkpoints. Large files now go to the 967 GB scratch disk, saves make room first, and a full disk skips the save instead of crashing.
  3. RNG state loaded onto the GPU broke resume. It now always loads on the CPU.
  4. Out-of-memory errors: generation halves its batch, and training halves the micro-batch with the effective batch unchanged.
  5. NVIDIA's free API returned only 429/504 for hours.
- Provenance: every run appends commit, library versions and GPUs to `provenance.jsonl`. Kaggle sessions save `env/pip-freeze.txt` and `env/nvidia-smi.txt`. Reference environment of the final scoring session: `results/environment/kaggle-pip-freeze.txt` and `kaggle-nvidia-smi.txt`.
- Cost: $0.

## Results log

| Date | Model | Split / env | Result | Source |
|---|---|---|---|---|
| 2026-11-08 | Gemma 3 270M SFT v2, step 1,000 (mid-training) | val, 100 games | 39% wins, mean 5.79, invalid turns 33% | Kaggle log, `sft-gemma270m-v2/best` |
| 2026-11-08 | SmolLM2-360M SFT v2, step 500 (mid-training) | val | 16% wins, invalid turns 42% | Kaggle log |
| 2026-11-08 | **Gemma 3 270M SFT v2, final (best = step 2,000 of 2,167)** | val, 100 games | **48% wins, mean 5.47, invalid turns 26.9%** | Kaggle log fl-gemma-v2 v3; Hub `sft-gemma270m-v2/best` |
| 2026-11-08 | Gemma 3 270M SFT v2 best | heldout (running total at 320/463) | ≈54% wins (final number pending fetch) | Kaggle live log |
| 2026-11-08 | Gemma 3 270M SFT v2 best | val 20 / train 20 games (local CPU) | 13/20 vs 20/20 wins; 19 invalid turns, all non-words | local diagnosis |
| 2026-11-08 | SmolLM2-360M SFT v2, final (best = step 1,500) | val, 100 | 42% wins, mean 5.73, invalid turns 31.8% | Kaggle log fl-gemma-v2 |
| 2026-11-08 | SmolLM2-135M SFT v2, final (best = step 2,167) | val, 100 | 28% wins, mean 6.13, invalid turns 41.6% | Kaggle log fl-gemma-v2 |
| 2026-11-08 | Gemma 3 270M SFT v3 (v2 + extra answers), mid-training curve | val, 100 | step 500: 22% · 1,000: 31% · 1,500: 39% · 2,000: 51% · **2,500: 54%** (invalid 41.9% → 27.5%) | synced `gemma-v3/runs/sft-gemma270m-v3/metrics.jsonl` |
| 2026-11-08 | **Gemma 3 270M SFT v2 (best)** | **heldout, 463** | **55.7% wins (CI 51.4–60.0), mean 5.24, invalid turns 25.3%** | `results/eval-v1/heldout/hf-sft-gemma270m-v2-best` |
| 2026-11-08 | SmolLM2-360M SFT v2 (best) | heldout, 463 | 33.7% (CI 29.6–38.0), mean 5.97, invalid 30.9% | `results/eval-v1/heldout/hf-sft-smollm2-360m-v2-best` |
| 2026-11-08 | SmolLM2-135M SFT v2 (best) | heldout, 463 | 23.3% (CI 19.4–27.2), mean 6.24, invalid 42.7% | `results/eval-v1/heldout/hf-sft-smollm2-135m-v2-best` |
| 2026-11-08 | Gemma 3 270M SFT v2 (best) | TextArena Wordle-v0, adapted, seeds 0–499 | all: 87.0% (CI 83.8–90.0), mean 3.89, 58 games lost on invalid moves · **clean (109 games, 32 secrets): 49.5% (CI 39.5–58.7)** | `results/eval-v1/public/Wordle-v0-adapted/hf-sft-gemma270m-v2-best` |
| 2026-11-08 | GRPO v2 (600 steps from SFT v2, training answers only) | val, 100, every 50 steps | flat 46–51% (start 48%, end 49%): no gain from RL on memorized answers | `results/runs/grpo-gemma270m-v2/metrics.jsonl` |
| 2026-11-08 | **Gemma 3 270M SFT v3 (best = step 3,873; val 56%)** | **heldout, 463** | **64.6% wins (CI 60.0–68.9), mean 5.05, invalid turns 23.3%**: +8.9 points over SFT v2 | `results/eval-v1/heldout/hf-sft-gemma270m-v3-best` |
| 2026-11-08 | Gemma 3 270M SFT v3 (best) | rare, 500 | 35.6% (CI 31.4–39.8), mean 6.16, invalid 25.8% | `results/eval-v1/rare/hf-sft-gemma270m-v3-best` |
| 2026-11-08 | Gemma 3 270M SFT v3 (best) | TextArena adapted, seeds 0–499 | all 85.4% (CI 82.4–88.4), 59 lost on invalid moves · clean (109) 46.8% (CI 37.6–56.0), 44 lost on invalid moves | `results/eval-v1/public/Wordle-v0-adapted/hf-sft-gemma270m-v3-best` |
| 2026-11-08 | Zero-shot Gemma 3 270M | TextArena native-verifiers, seeds 0–499 | 0.0%: all 500 games lost on invalid moves (format) | `results/eval-v1/public/Wordle-v0-native-verifiers/hf-gemma-3-270m-it` |
| 2026-11-08 | GRPO v3 (from SFT v3, 50% extra answers), mid-training curve | val, 100, every 50 steps | step 0: 56% · 50: 57% · 100: 54% · 150: 61% · **200: 62%** (invalid 26.6% → 23.6%); GRPO v2 stayed flat | synced `gemma-v3/runs/grpo-gemma270m-v3/metrics.jsonl` |
| 2026-11-09 | **Gemma 3 270M GRPO v3 (best = step 550; val 66%)** | **heldout, 463** | **61.8% (CI 57.5–66.1)**, mean on wins 4.04, mean (loss=7) 5.17, **invalid turns 19.4%** | `results/eval-v1/heldout/hf-grpo-gemma270m-v3-best` |
| 2026-11-09 | Gemma 3 270M GRPO v3 (best) | rare, 500 | **41.2% (CI 37.0–45.4)**, +5.6 over SFT v3, invalid 20.4% | `results/eval-v1/rare/hf-grpo-gemma270m-v3-best` |
| 2026-11-09 | Gemma 3 270M GRPO v3 (best) | **TextArena adapted, seeds 0–499** | all 88.8% (CI 86.0–91.6), mean 3.92, 46 lost on invalid moves · **clean (109): 64.2% (CI 55.1–72.5)**, 29 lost on invalid moves (SFT v3: 46.8%, 44) | `results/eval-v1/public/Wordle-v0-adapted/hf-grpo-gemma270m-v3-best` |
| 2026-11-09 | Gemma 3 270M GRPO v3 (best) | full, 2,315 (includes the 1,752 training answers) | 84.2% (CI 82.6–85.7), mean 4.24, invalid 9.7% | `results/eval-v1/full/hf-grpo-gemma270m-v3-best` |
| 2026-11-09 | Gemma 3 270M GRPO v3 (best) + **dictionary-constrained decoding** (wordle vocab) | heldout, 463 | **74.1% (CI 70.2–78.2)**, mean on wins 4.17, mean (loss=7) 4.90, invalid 0%: +12.3 over unconstrained (61.8%) | `results/eval-v1/heldout/hf-grpo-gemma270m-v3-best-constrained-wordle` |
| 2026-11-09 | **Gemma 3 270M SFT v3 (best) + constrained decoding** | **heldout, 463** | **79.5% (CI 75.8–83.2)**, mean (loss=7) 4.70, invalid 0%: **best heldout so far** | `results/eval-v1/heldout/hf-sft-gemma270m-v3-best-constrained-wordle` |
| 2026-11-09 | GRPO v3 (best) + constrained decoding | rare, 500 | 58.6% (CI 54.4–63.0), invalid 0% (unconstrained 41.2%) | `results/eval-v1/rare/hf-grpo-gemma270m-v3-best-constrained-wordle` |
| 2026-11-09 | GRPO v3 (best) + constrained decoding | full, 2,315 (incl. training answers) | 89.5% (CI 88.2–90.6), mean 4.13 | `results/eval-v1/full/hf-grpo-gemma270m-v3-best-constrained-wordle` |
| 2026-11-09 | GRPO v3 (best) + constrained decoding (shared vocab) | TextArena adapted, seeds 0–499 | all 92.0% (CI 89.6–94.2), mean 3.90, 0 lost on invalid moves · **clean (109): 71.6% (CI 63.3–79.8)** | `results/eval-v1/public/Wordle-v0-adapted/hf-grpo-gemma270m-v3-best-constrained-shared` |
| 2026-11-09 | Gemma 3 270M SFT v4 (2,484 broad extra answers, 2 epochs, benchmark-clean) | val, 100 | 74% at steps 8,000–9,238 (v3 best: 56%) | synced `gemma-v4/runs/sft-gemma270m-v4/metrics.jsonl` |
| 2026-11-09 | ~~Gemma 3 270M SFT v4 (best)~~ RETRACTED: stale step-2,500 Hub checkpoint | heldout, 463 | **55.1% (CI 50.5–59.6)**, invalid 25.9%: **worse than SFT v3 (64.6%, CI 60.0–68.9)** | `results/eval-v1/heldout/hf-sft-gemma270m-v4-best` |
| 2026-11-09 | ~~SFT v4 + constrained~~ RETRACTED: stale checkpoint | heldout, 463 | 72.4% (CI 68.0–76.2): below SFT v3 + constraint (79.5%) | `results/eval-v1/heldout/hf-sft-gemma270m-v4-best-constrained-wordle` |
| 2026-11-09 | Gemma 3 270M SFT v3c (v3 recipe, **benchmark-clean**: 148 WordleBench/Jeffers answers removed from training; best = step 3,000, val 63%) + constrained decoding | heldout, 463 | 77.1% (CI 73.2–80.8), mean on wins 4.15, mean (loss=7) 4.81, invalid 0% | `results/eval-v1/heldout/hf-sft-gemma270m-v3c-best-constrained-wordle` |
| 2026-11-09 | SFT v3c (benchmark-clean), unconstrained | heldout, 463 | 58.5% (CI 54.2–63.1), mean on wins 3.99, mean (loss=7) 5.24, invalid 24.4%: below SFT v3 (64.6%), likely because 148 fewer training answers (seed noise not ruled out) | `results/eval-v1/heldout/hf-sft-gemma270m-v3c-best` |
| 2026-11-09 | **SFT v3c + constrained decoding** | **WordleBench, 100 (clean)** | **73.0% (CI 64–81)**, mean on wins 4.14, mean (loss=7) 4.91. Cited GPT-5: 99% / 3.59 | `results/eval-v1/wordlebench/hf-sft-gemma270m-v3c-best-constrained-wordle` |
| 2026-11-09 | **SFT v3c + constrained decoding** | **Jeffers, 50 (clean)** | **64.0% (CI 52–76)**, mean on wins 4.09, mean (loss=7) 5.14. Cited GPT-5: 100% / 3.58 | `results/eval-v1/jeffers/hf-sft-gemma270m-v3c-best-constrained-wordle` |
| 2026-11-09 | SFT v3c (benchmark-clean), unconstrained | WordleBench, 100 (clean) | 56.0% (CI 46–66), mean on wins 3.95, mean (loss=7) 5.29, invalid 24.3%. Cited GPT-5: 99% / 3.59 | `results/eval-v1/wordlebench/hf-sft-gemma270m-v3c-best` |
| 2026-11-09 | SFT v3c (benchmark-clean), unconstrained | Jeffers, 50 (clean) | 56.0% (CI 42–70), mean on wins 4.00, mean (loss=7) 5.32, invalid 24.2%. Cited GPT-5: 100% / 3.58 | `results/eval-v1/jeffers/hf-sft-gemma270m-v3c-best` |
| 2026-11-09 | SFT v3c + constrained decoding | rare, 500 | 54.0% (CI 49.4–58.4), mean (loss=7) 5.82 | `results/eval-v1/rare/hf-sft-gemma270m-v3c-best-constrained-wordle` |
| 2026-11-09 | SFT v3c + constrained decoding (shared vocab) | TextArena adapted, seeds 0–499 | all 91.4% (CI 89.0–93.8), mean on wins 3.66, 0 lost on invalid moves · clean (109, 32 secrets): 72.5% (CI 64.2–80.7) | `results/eval-v1/public/Wordle-v0-adapted/hf-sft-gemma270m-v3c-best-constrained-shared` |
| 2026-11-09 | SFT v3c: **clue-contradiction rate** (share of model guesses after the first valid one that contradict known clues) | heldout / WordleBench / Jeffers | unconstrained 49.0% / 51.3% / 54.8% · constrained 55.9% / 58.5% / 64.6% (rare constrained 59.8%). Baseline for v5, which targets this. Computed on the laptop from the fetched `results.jsonl` with `metrics.summarize` (the session ran an older commit without this metric) | `kaggle_runs/gemma-v3c/runs/eval/*/results.jsonl` (local) |
| 2026-11-09 | Gemma 3 270M SFT v5 seed 1 (consistent teacher, benchmark-clean), selection curve | select, 534 | step 500: 18.5% · 1,000: 30.9% · 1,500: 40.4% · 2,000: 46.1% · 2,500: 55.6% · 3,000: 57.3% · 3,500: 60.5% · **3,927 (final = best): 62.0%** (invalid 50.6% → 25.2%) | `results/runs/sft-gemma270m-v5-s1/metrics.jsonl` |
| 2026-11-09 | **SFT v5 s1, unconstrained** | heldout, 463 | **61.8% (CI 57.5–66.1)**, mean on wins **3.78**, mean (loss=7) 5.01, invalid 25.1%, **clue-contradiction rate 7.5%** (v3c: 58.5%, 3.99, 24.4%, 49.0%) | `results/eval-v1/heldout/hf-sft-gemma270m-v5-s1-best` |
| 2026-11-09 | SFT v5 s1 + constrained decoding | heldout, 463 | 70.2% (CI 66.1–74.3), mean (loss=7) 4.83, clue-contradiction 32.6%: **below v3c + constraint (77.1%)**. **Gate FAILED** (needed > 79.5%): GRPO s1 and SFT s2 did not run | `results/eval-v1/heldout/hf-sft-gemma270m-v5-s1-best-constrained-wordle` |
| 2026-11-09 | **SFT v5 s1 + word-level constrained decoding** (beam 16 over the 12,972-word guess list; no repeats; clues never used by the code) | **heldout, 463** | **94.2% (CI 92.0–96.1)**, mean on wins 3.90, mean (loss=7) **4.08**, invalid 0%, clue-contradiction 9.4%. Laptop CPU, fp32 (Kaggle T4 re-check running). Same model with token-greedy constraint: 70.2%; unconstrained: 61.8% | `results/eval-v1/heldout/hf-sft-gemma270m-v5-s1-best-constrained-wordle-word-laptop` |
| 2026-11-09 | SFT v5 s1 + word-level decoding, **Kaggle T4 re-check** | heldout, 463 | 94.2% (CI 92.0–96.1), mean on wins 3.90, mean (loss=7) 4.08: **identical to the laptop CPU run** (same games won, same means) | `results/eval-v1/heldout/hf-sft-gemma270m-v5-s1-best-constrained-wordle-word` |
| 2026-11-09 | SFT v5 s1 + word-level decoding | select, 534 | 94.9% (CI 93.1–96.8), mean on wins 3.91, mean (loss=7) 4.06, clue-contradiction 9.0% | `results/eval-v1/select/hf-sft-gemma270m-v5-s1-best-constrained-wordle-word` |
| 2026-11-09 | **Ablation: SFT v3c (old teacher) + word-level decoding** | heldout, 463 | **94.2% (CI 92.0–96.3)**, mean on wins **4.08**, mean (loss=7) 4.25, clue-contradiction **44.0%** | `results/eval-v1/heldout/hf-sft-gemma270m-v3c-best-constrained-wordle-word` |
| 2026-11-09 | Ablation: SFT v3c + word-level decoding | select, 534 | 94.9% (CI 92.9–96.6), mean on wins 4.09, mean (loss=7) 4.24, clue-contradiction 43.2% | `results/eval-v1/select/hf-sft-gemma270m-v3c-best-constrained-wordle-word` |
| 2026-11-09 | **FINAL TEST: SFT v5 s1 + word-level decoding** | **WordleBench, 100 (clean)** | **95.0% (CI 90–99)**, mean on wins 3.95, mean (loss=7) 4.10. Cited GPT-5: 99% / 3.59 | `results/eval-v1/wordlebench/hf-sft-gemma270m-v5-s1-best-constrained-wordle-word` |
| 2026-11-09 | **FINAL TEST: SFT v5 s1 + word-level decoding** | **Jeffers, 50 (clean)** | **86.0% (CI 76–94)**, mean on wins 3.84, mean (loss=7) 4.28. Cited GPT-5: 100% / 3.58 | `results/eval-v1/jeffers/hf-sft-gemma270m-v5-s1-best-constrained-wordle-word` |
| 2026-11-09 | FINAL TEST: SFT v5 s1, unconstrained (model alone) | WordleBench, 100 | 60.0% (CI 50–70), mean on wins 3.97, invalid 25.7% | `results/eval-v1/wordlebench/hf-sft-gemma270m-v5-s1-best` |
| 2026-11-09 | FINAL TEST: SFT v5 s1, unconstrained (model alone) | Jeffers, 50 | 54.0% (CI 40–68), mean on wins 3.59, invalid 30.6% | `results/eval-v1/jeffers/hf-sft-gemma270m-v5-s1-best` |
| 2026-11-09 | FINAL TEST: SFT v5 s1 + word-level decoding (shared vocab) | TextArena Wordle-v0 adapted, seeds 0–499 | all 98.4% (CI 97.2–99.4), mean on wins 3.64, 0 lost on invalid moves · **clean (109 games, 32 secrets): 92.7% (CI 87.2–97.3)** | `results/eval-v1/public/Wordle-v0-adapted/hf-sft-gemma270m-v5-s1-best-constrained-shared-word` |
| 2026-11-09 | FINAL TEST: SFT v5 s1 + word-level decoding | rare, 500 (answers outside the classic list) | 75.8% (CI 72.0–79.6), mean on wins 4.74, mean (loss=7) 5.29, clue-contradiction 13.1% (v3c + token-greedy: 54.0%) | `results/eval-v1/rare/hf-sft-gemma270m-v5-s1-best-constrained-wordle-word` |
| 2026-11-09 | GRPO v5 s1 (600 steps from SFT v5 s1, consistent curriculum, consistency penalty 0.2, 50% extra answers), selection curve | select, 534, unconstrained | step 0: 62.0% · 100: 60.1% · 200: 60.5% · 300: 60.5% · 400: 60.3% · 500: 60.1% · 600: 60.3%. **No checkpoint beat step 0, so `grpo-gemma270m-v5-s1/best` is the unchanged SFT model** (its select word-level eval is identical to SFT's: 94.9%, 3.91). Training rollouts: reward 0.834 → 0.804, invalid 13.4% → 10.4% | `results/runs/grpo-gemma270m-v5-s1/metrics.jsonl` |
| 2026-11-09 | GRPO v5 s1 best (= step 0 = SFT v5 s1 weights) | heldout, 463 | unconstrained 61.8%, word-level 94.2%: identical to SFT v5 s1, as expected, confirming the decision inputs | `results/eval-v1/heldout/hf-grpo-gemma270m-v5-s1-best*` |
| 2026-11-09 | SFT v5 seeds 2 and 3, selection curves (unconstrained) | select, 534 | s2: 19.5% (500) → 59.6% (3,927, final = best) · s3: 18.9% (500) → 58.4% (3,927) · s1 for reference: 62.0% | `results/runs/sft-gemma270m-v5-s{2,3}/metrics.jsonl` |
| 2026-11-09 | **SFT v5, 3 seeds (1, 2, 3), mean ± sd** | select 534 / heldout 463 | word-level select **93.8 ± 1.5%** (94.9, 92.1, 94.2), 3.91 guesses/win · word-level heldout **93.6 ± 1.0%** (94.2, 92.4, 94.2), 3.90 · unconstrained heldout **60.0 ± 1.6%** (61.8, 59.6, 58.8), 3.76. All audited | `results/eval-v1/{select,heldout}/hf-sft-gemma270m-v5-s{1,2,3}-best*` |
| 2026-11-09 | **Distillation round 1** (`sft-gemma270m-v5-d1`: SFT v5 s1 fine-tuned 1 epoch, lr 2e-5, on its own won games with word-level decoding over 3,248 training answers × 2 openers), selection curve | select, 534, unconstrained | step 100: 58.2% · 200: 61.2% · 300: 65.0% · **322 (final = best): 65.2%** (training-time eval); invalid turns 25.5% → 21.1% | `results/runs/sft-gemma270m-v5-d1/metrics.jsonl` |
| 2026-11-09 | Distillation round 1, best | select, 534, **unconstrained** | **65.0% (CI 61.1–68.9)**, mean on wins 3.82, mean (loss=7) 4.94, invalid 21.2%, clue-contradiction 9.4% (standalone fp32 eval; the training-time fp16 eval said 65.2%, one game apart). Baseline SFT v5 s1: 62.0% (training-time). **+3.0 points: passes the ≥ 2-point model-alone gate**; word-level check pending | `results/eval-v1/select/hf-sft-gemma270m-v5-d1-best` |
| 2026-11-09 | Distillation round 1, best | select, 534, **word-level** | 92.7% (CI 90.5–94.8), mean on wins 3.97, clue-contradiction 12.5% (SFT v5 s1: 94.9%, 3.91, 9.0%) | `results/eval-v1/select/hf-sft-gemma270m-v5-d1-best-constrained-wordle-word` |
| 2026-11-09 | Distillation round 1, best | heldout, 463 | **alone 65.7% (CI 61.3–70.2)** vs SFT v5 s1 61.8%; word-level 92.9% (CI 90.5–95.3) vs 94.2% | `results/eval-v1/heldout/hf-sft-gemma270m-v5-d1-best*` |
| 2026-11-10 | **SFT v6 s1** (common-word + probe teacher, spaced format; best = step 3,000), **unconstrained** | select, 534 | 61.2% (CI 57.1–65.4), mean on wins 3.97, mean (loss=7) 5.14, invalid 23.6%, clue-contradiction 20.9%. SFT v5 s1: 62.0%. **No gain alone from the new teacher** | `results/eval-v1/select/hf-sft-gemma270m-v6-s1-best` |
| 2026-11-10 | SFT v6 s1, **word-level** | select, 534 | 94.9% (CI 92.9–96.6), mean on wins 4.08, mean (loss=7) 4.22 (SFT v5 s1: 94.9%, 3.91). Same wins, more guesses (the probe moves) | `results/eval-v1/select/hf-sft-gemma270m-v6-s1-best-constrained-wordle-word` |
| 2026-11-10 | SFT v6 s1 | heldout, 463 (development) | alone 61.8% (CI 57.2–66.1), invalid 23.9%; word-level **95.7% (CI 93.7–97.4)**, mean on wins 4.06 (SFT v5 s1: 61.8% / 94.2%) | `results/eval-v1/heldout/hf-sft-gemma270m-v6-s1-best*` |
| 2026-11-10 | **SFT v6w s1** (same teacher, whole-word guesses + 40k drill; best = final step 4,382), unconstrained | select, 534 | **44.9% (CI 40.6–49.1)**, invalid 5.4%, clue-contradiction 52.3%. Fewer invented words, but the model stops tracking the clues. **Negative result: whole-word format dropped** | `results/eval-v1/select/hf-sft-gemma270m-v6w-s1-best` |
| 2026-11-10 | SFT v6w s1, word-level | select, 534 | 54.5% (CI 50.2–58.8), mean on wins 4.24, clue-contradiction 51.8% | `results/eval-v1/select/hf-sft-gemma270m-v6w-s1-best-constrained-wordle-word` |
| 2026-11-10 | SFT v6w s1 | heldout, 463 (development) | alone 44.7% (CI 40.2–49.0), invalid 5.0%; word-level 55.3% (CI 50.8–59.6) | `results/eval-v1/heldout/hf-sft-gemma270m-v6w-s1-best*` |
| 2026-11-10 | SFT v5 s1 (base of v7), alone, **leaderboard rule `open-vocab`** (any five letters get colours, as on WordleBench and Jeffers; laptop CPU fp32) | select, 534 | **77.7% (CI 74.2-80.9)**, mean on wins 4.05, non-word guesses 21.6% of turns (256 games have one), clue-contradiction 4.5%. The same model is 62.0% under strict eval-v1 and 68.4% under retry-3. Audit OK | `results/eval-v1/select/hf-sft-gemma270m-v5-s1-best-open` |
| 2026-11-10 | SFT v5 s1 (base of v7), alone, **retry-3 rule** | select, 534 | 68.4% (CI 64.4-72.1), 2.33 rejected words per game, invalid turns 10.4% | `results/eval-v1/select/hf-sft-gemma270m-v5-s1-best-retry3` |
| 2026-11-10 | **SFT v7 s1** (v5 s1 + balanced spelling drill, 30 per common word, + half the v5 teacher games; best = step 4,500 of 4,567), alone | select, 534 | **74.9% (CI 71.2-78.7)** (62.0% before), mean on wins 3.71, invalid 15.2% (23.5% before), clue-contradiction 7.1% | `results/eval-v1/select/hf-sft-gemma270m-v7-s1-best` |
| 2026-11-10 | SFT v7 s1, word-level | select, 534 | **97.2% (CI 95.7-98.5)** (94.9% before), mean on wins 3.83 | `results/eval-v1/select/hf-sft-gemma270m-v7-s1-best-constrained-wordle-word` |
| 2026-11-10 | SFT v7 s1, alone, **retry-3 rule** | select, 534 | 82.0% (CI 78.7-85.2) | `results/eval-v1/select/hf-sft-gemma270m-v7-s1-best-retry3` |
| 2026-11-10 | SFT v7 s1 | heldout, 463 (development) | alone 76.7% (CI 72.6-80.6), invalid 14.4%; word-level **97.6% (CI 96.3-98.9)**, mean on wins 3.81 | `results/eval-v1/heldout/hf-sft-gemma270m-v7-s1-best*` |
| 2026-11-10 | SFT v7s s1 (strict ablation: no dev or test answer is ever a drill target; best = final step 4,187), alone | select, 534 | 56.2% (CI 52.1-60.3), **below the base model**, invalid 24.0%; retry-3 63.9%. Word-level not run (in-queue gate at 62%) | `results/eval-v1/select/hf-sft-gemma270m-v7s-s1-best*` |
| 2026-11-10 | SFT v7s s1, alone | heldout, 463 | 56.6% (CI 52.3-60.9) | `results/eval-v1/heldout/hf-sft-gemma270m-v7s-s1-best` |
| 2026-11-10 | SFT v7 s1, alone, **leaderboard rule `open-vocab`** (laptop CPU fp32) | select, 534 | **87.8% (CI 84.8-90.6)**, mean on wins 3.88, non-word guesses 10.7% of turns (143 games have one). Strict 74.9%, retry-3 82.0%. Audit OK | `results/eval-v1/select/hf-sft-gemma270m-v7-s1-best-open` |
| 2026-11-10 | SFT v7s s1, alone, leaderboard rule `open-vocab` (laptop CPU fp32) | select, 534 | 74.0% (CI 70.2-77.5), mean on wins 4.10, non-word guesses 18.7% of turns. Strict 56.2%. Audit OK | `results/eval-v1/select/hf-sft-gemma270m-v7s-s1-best-open` |
| 2026-11-10 | **SFT v8 s1** (v7 s1 + deep drill, 40 per common word, up to 2 rejected words per board, + half the v5 teacher games; best = step 5,000 of 5,436), alone | select, 534 | **82.2% (CI 78.8-85.2)** (v7: 74.9%), mean on wins 3.62, invalid 12.5%, clue-contradiction 2.3% | `results/eval-v1/select/hf-sft-gemma270m-v8-s1-best` |
| 2026-11-10 | **SFT v8 s2** (same data, other order; best = final step 5,436), alone | select, 534 | **83.3% (CI 80.0-86.5)**, mean on wins 3.64, invalid 11.5%, clue-contradiction 3.1% | `results/eval-v1/select/hf-sft-gemma270m-v8-s2-best` |
| 2026-11-10 | SFT v8 s1 / s2, word-level | select, 534 | **98.1% (CI 97.0-99.3) / 98.5% (CI 97.4-99.4)**, mean on wins 3.76 / 3.74 | `results/eval-v1/select/hf-sft-gemma270m-v8-s{1,2}-best-constrained-wordle-word` |
| 2026-11-10 | SFT v8 s1 / s2, alone, retry-3 rule | select, 534 | 84.3% (CI 81.1-87.1) / 84.8% (CI 81.8-87.8) | `results/eval-v1/select/hf-sft-gemma270m-v8-s{1,2}-best-retry3` |
| 2026-11-10 | SFT v8 s1 / s2 | heldout, 463 (development) | alone 82.3% (CI 78.6-86.0) / 84.2% (CI 81.0-87.5); word-level **98.5% (CI 97.4-99.6) / 98.9% (CI 97.8-99.8)** | `results/eval-v1/heldout/hf-sft-gemma270m-v8-s{1,2}-best*` |
| 2026-11-10 | **SFT v8 s3** (same data and base as s1/s2, seed 3; best = final step 5,436), alone | select, 534 | **83.9% (CI 80.7-86.9)**, mean on wins 3.63, invalid 11.3%; retry-3 85.0%; open-vocab 94.4% (CI 92.3-96.3); word-level **98.7% (CI 97.6-99.6)** | `results/eval-v1/select/hf-sft-gemma270m-v8-s3-best*` |
| 2026-11-10 | SFT v8 s3 | heldout, 463 | alone 84.7% (CI 81.2-87.9); word-level 99.1% (CI 98.3-99.8) | `results/eval-v1/heldout/hf-sft-gemma270m-v8-s3-best*` |
| 2026-11-10 | **SFT v9 s1** (v8 recipe once more from v8 s2, fresh boards; best = step 4,500), alone | select, 534 | **84.8% (CI 81.8-87.8)**, mean on wins 3.64, invalid 10.3%; retry-3 86.3%; open-vocab 94.2% (CI 92.1-96.1); word-level 98.1% (CI 96.8-99.3) | `results/eval-v1/select/hf-sft-gemma270m-v9-s1-best*` |
| 2026-11-10 | SFT v9 s1 | heldout, 463 | alone 85.5% (CI 82.3-88.8); word-level 98.7% (CI 97.6-99.6) | `results/eval-v1/heldout/hf-sft-gemma270m-v9-s1-best*` |
| 2026-11-10 | **SFT v10 s1** (recovery round from v8 s3: 3,077 boards from its own rejected words + 4% v8 replay, 22,046 samples, 2 epochs; best = step 300 of 688), alone | select, 534 | **88.0% (CI 85.4-90.8)**, mean on wins 3.80, invalid 10.3%; retry-3 89.7%; open-vocab 95.5% (CI 93.6-97.2); word-level **99.1% (CI 98.1-99.8)** | `results/eval-v1/select/hf-sft-gemma270m-v10-s1-best*` |
| 2026-11-10 | SFT v10 s1 | heldout, 463 | alone 88.3% (CI 85.3-91.1); word-level 98.9% (CI 97.8-99.8) | `results/eval-v1/heldout/hf-sft-gemma270m-v10-s1-best*` |
| 2026-11-10 | **SFT v10 s2** (same recipe, other half of the hidden words, seed 2: 2,879 boards, 21,848 samples; best = step 200 of 682), alone | select, 534 | **87.3% (CI 84.5-90.1)**, mean on wins 3.77, invalid 10.2%; retry-3 90.1%; open-vocab 92.9% (CI 90.6-94.9); word-level 98.1% (CI 96.8-99.2) | `results/eval-v1/select/hf-sft-gemma270m-v10-s2-best*` |
| 2026-11-10 | SFT v10 s2 | heldout, 463 | alone 87.3% (CI 84.0-90.3); word-level 98.3% (CI 97.0-99.4) | `results/eval-v1/heldout/hf-sft-gemma270m-v10-s2-best*` |
| 2026-11-10 | **SFT v8 s1**, alone, **leaderboard rule `open-vocab`** (laptop CPU fp32) | select, 534 | **94.4% (CI 92.3-96.1)**, mean on wins 3.81, non-word guesses 6.2% of turns (91 games have one), clue-contradiction 2.6%. Strict 82.2%. Audit OK | `results/eval-v1/select/hf-sft-gemma270m-v8-s1-best-open` |
| 2026-11-10 | **SFT v8 s2**, alone, **leaderboard rule `open-vocab`** (laptop CPU fp32) | select, 534 | **94.0% (CI 91.9-95.9)**, mean on wins 3.80, non-word guesses 6.3% of turns (86 games have one), clue-contradiction 3.3%. Strict 83.3%. Audit OK | `results/eval-v1/select/hf-sft-gemma270m-v8-s2-best-open` |
| 2026-11-08 | Zero-shot Gemma 3 270M | heldout, 463 | 0.0% wins, invalid turns 100% | `results/eval-v1/heldout/hf-gemma-3-270m-it` |
| 2026-11-08 | Zero-shot SmolLM2-135M-Instruct | heldout, 463 | 0.0% wins, invalid turns 100% | `results/eval-v1/heldout/hf-SmolLM2-135M-Instruct` |
| 2026-11-08 | Zero-shot SmolLM2-360M-Instruct | heldout, 463 | 0.22% wins (CI 0–0.65%), invalid turns 83.3% | `results/eval-v1/heldout/hf-SmolLM2-360M-Instruct` |
| 2026-11-08 | Zero-shot Qwen3-0.6B, no thinking | heldout, 463 | 0.0% wins, invalid turns 100% | `results/eval-v1/heldout/hf-Qwen3-0.6B-nothinking` |

Note: under our strict protocol, untrained small models almost never produce a valid `<guess>` tag, so zero-shot scores measure format following as much as skill. The native-format runs (verifiers prompts) are the fairer zero-shot comparison and must be reported next to these.

(Zero-shot and public-checkpoint baselines: pending a faster rerun, see below.)

## Open questions to answer with data

- Why are mid-training invalid turns high (33%)? Check the error mix (`not_in_word_list` vs `format`) in the final eval records.
- Does GRPO reduce invalid turns and mean guesses beyond SFT?
- Frontier comparison: open models run by us (Qwen3-4B, Gemma 3 4B) plus cited leaderboard numbers. Live frontier APIs are deferred.

## Diagnosis of the model-alone gap (2026-11-10, from `sft-gemma270m-v6-s1` on select, unconstrained, 534 games)

Computed on the laptop from the synced `results.jsonl` and the v6 + v6-extra traces (249,535 training targets).

* **Games with no invalid turn: 269 of 272 won (99%).** With 1 invalid turn 65%, with 2 17%, with 3 or more 1%. 181 of the 207 losses have at least 2 invalid turns. The play is already strong; the losses are invented words.
* All 598 invalid turns are well-formed five-letter non-words. 78% of them fit every clue, and 38% are one letter away from the answer. The model knows the pattern but not the word.
* **The model types almost only words it typed in training:** 1,894 of its 1,941 valid guesses (97.6%) were training targets. Training targets cover 6,871 of the 12,972 valid words.
* **Win rate by how often the answer was a training target:** never 12.5% (2/16) · 1–9 times 43.6% (78/179) · 10–99 times 72.4% (220/304) · 100 or more 77.1% (27/35).
* The v6w spelling drill is too small and too skewed to fix this: 40,000 samples over 5,478 targets, median 5 per word, the top 100 words take 26%, and 481 of the 534 select answers are a drill target fewer than 10 times. It was also only in the whole-word run.
* Whole-word guesses (v6w) cut invalid turns to 6% but reach only 44.4% wins at step 3,500 (v6 spaced: 59.0%): the model writes real words that do not fit the clues. Spaced letters stay the main format.
* v6 alone (61.2% select, invalid 23.6%) equals v5 alone (62.0%): the better teacher does not help the model alone, because spelling is the limit.

Consequence (round v7): a **balanced spaced-letter spelling drill**, in which every common word (5,557 words at Zipf ≥ 2.0 from `wordfreq`) is a target about 30 times on boards it fits. Boards are still built only around training-side hidden words; targets may be any public dictionary word (disclosed; a strict variant without benchmark words is the ablation).

## Round v7 build (2026-11-10, launched as `bestman2/fl-v7`)

**What is trained.** Two fine-tunes of `sft-gemma270m-v5-s1/best` (the best spaced model: select 62.0% alone, 94.9% word-level), 1 epoch, lr 5e-5, warmup 100, best checkpoint picked on select. Plan: `kaggle/plans/v7.json`.

| Run | Data (private HF dataset) | Samples | Drill | Replay |
|---|---|---|---|---|
| `sft-gemma270m-v7-s1` | `five-letters-traces-v7` | 292,343 | 166,710 = 5,557 common words x 30, none skipped | 66,783 + 58,850 (half of v5 and v5-extra teacher games) |
| `sft-gemma270m-v7s-s1` (strict ablation) | `five-letters-traces-v7s` | 268,013 | 142,380 = 4,746 words x 30 | the same |

**The balanced drill** (`drill.generate_balanced_drill`, `fiveletters build-v7`). One sample = a board and one real word that fits it, written in spaced letters. A board is an opener (CRATE 60% of the time) plus 0–3 common words that fit the clues so far. Board depth 1/2/3/4 guesses: 30/35/25/10%. 15% of boards contain one rejected near-miss non-word, to teach recovery. Every target word appears exactly 30 times.

**Fairness, measured on the built data** (scratch audit, 2026-11-10):

| Check | v7 | v7s |
|---|---|---|
| Drill boards built around a dev or test word | 0 | 0 |
| Rows from a game whose answer is a leaderboard word | 0 | 0 |
| Replay games on a val, heldout, select or leaderboard answer | 0 | 0 |
| Select words that are drill targets | 527 of 534 | 0 |
| Leaderboard words (WordleBench, Jeffers) that are drill targets | 147 of 148 | 0 |

- v7 uses the disclosed choice (user decision 2026-11-09): any common public word can be a spelling target, test answers included. The model still never plays, or sees a board built around, a dev or test word.
- v7s is stricter than the ROADMAP asked for: no val, heldout, select, rare or leaderboard word is ever a target. It measures how far the drill carries over to words it never showed, and it stays fully clean for the leaderboard test.
- The paper must report both and say which one each number comes from.

**Retry rule.** Second, labelled rule set `retry-3` (PROTOCOL.md). Scored for the base model and both v7 runs on select. It is shown as a second column next to strict. (Update, same day: the code of both leaderboards was read and neither has a dictionary at all; see the finding at the end of this file and the third rule `open-vocab`.)

**Checks before launch.** 76 tests pass, lint clean. Local end-to-end run with the real base model: build -> 4 training steps -> eval alone, `--retries 3`, word-level -> independent audit (only the expected "8 games instead of 534" notice) -> gate command.

## Finding: the cited leaderboards have no dictionary (2026-11-10)

**What we read.** The source code of both leaderboards, not only their READMEs. Details and commits: `results/cited/leaderboards_2026-11-09.md`.

| Leaderboard | Dictionary check | A five-letter non-word | A malformed or repeated guess |
|---|---|---|---|
| WordleBench (`main.py`, commit 79436c8) | none | scored, uses a turn | any non-empty text is scored; an empty guess ends the game as an error |
| Jeffers (`lib/wordle-engine.ts`, commit a79acdb) | none (`isValidWord` is imported, never called) | scored, uses a turn | replaced by an unused starter word, which is scored |
| Ours, strict eval-v1 (headline) | 12,972 words | no colours, turn lost | no colours, turn lost |
| Ours, `open-vocab` (third labelled rule) | none | scored, uses a turn | no colours, turn lost |

**Why it matters.** The frontier numbers we compare with (GPT-5 99% on WordleBench, 100% on Jeffers) were made under a rule where a misspelled word still returns colours. Under our strict rule the same misspelling returns nothing and costs a turn. Our model's main weakness alone is exactly this: about 24% of its turns are non-words, 78% of those non-words fit every clue, and 38% are one letter away from the answer (diagnosis section above). So the strict rule and the leaderboards' rule can give our model very different scores.

**What we do about it.**

- The headline stays strict eval-v1. Nothing about it changes.
- New labelled rule `open-vocab` (`fiveletters eval --open-vocab`, PROTOCOL.md). It is the part both leaderboards share: any five letters a-z are scored. On malformed output it stays strict, so it is not easier than either leaderboard there.
- During development it is scored on select only. The leaderboard sets are scored once, for the declared final model, under strict, `retry-3` and `open-vocab`, and each use is logged.
- The paper table shows cited numbers next to our `open-vocab` number (same rule) and our strict number (harder rule), each under its label.
- Not hidden: a win under `open-vocab` can include turns where the model wrote a non-word. The non-word guess rate is reported next to every `open-vocab` win rate.

## Round v7 result (2026-11-10, Kaggle `bestman2/fl-v7`, 174.8 minutes, all 9 evals audited OK)

| Model | Select alone (strict) | Select retry-3 | Select word-level | Heldout alone | Heldout word-level |
|---|---|---|---|---|---|
| v5 s1 (base) | 62.0% | 68.4% | 94.9% | 61.8% | 94.2% |
| **v7 s1** (drill targets: all common words) | **74.9%** | 82.0% | **97.2%** | **76.7%** | **97.6%** |
| v7s s1 (drill targets: no dev or test word) | 56.2% | 63.9% | not run | 56.6% | not run |

**What it shows.**

- The drill works when the words the model must spell are among the drill targets. v7 gains 13 points alone and 2.3 points with word-level decoding on select, and the same on heldout. Word-level is now 97.2% / 97.6%, the best number so far.
- The strict ablation loses 6 points. When no dev or test word is ever a target, the model spells those words worse than before. Two readings, both to be stated in the paper: (a) the spelling skill is mostly word-specific, not a general skill; (b) v7s learns that about 800 words (exactly the likely answers) are never the right output, which pushes them down. The data cannot separate the two yet.
- So the v7 gain depends on the disclosed choice: test answers can appear as spelling targets on boards built around other words. The model never plays, or sees a board built around, a dev or test word. Every v7-based number must carry this label, and the v7s numbers are reported next to it.
- Where v7 still loses (select, alone): 129 of its 134 lost games contain an invalid turn. Invalid rate by turn: 0 / 0.9 / 8.6 / 28.2 / 51.7 / 59.5% for turns 1-6. The drill only had boards of 1-4 guesses (targets at turns 2-5), with at most one rejected word, and turn 6 was never drilled. 80% of the non-words still fit the clues and 48% are one letter from the answer.
- The best checkpoint is step 4,500: 75.47% in the training-time eval, 74.91% in the separate eval of the same weights (3 games out of 534; batch size and fp16 padding differ). The pre-registered rule uses the separate eval.

**Decision (pre-registered "After v7" rule).** Select alone is 74.9%, inside the 62-75% band: one more round with more samples per word and deeper boards, starting from v7 s1. v7s is 19 points behind, so v7 is the base. Deeper boards target exactly the measured failure (turns 4-6, and boards with more than one rejected word).

## Round v8 build (2026-11-10, launched as `bestman2/fl-v8`)

**Why.** Pre-registered "After v7" rule, band 62-75%: more samples per word and deeper boards. v7's remaining invalid turns sit at turns 4-6, where its drill had few boards.

**What is trained.** Two seeds (s1, s2) of a fine-tune of `sft-gemma270m-v7-s1/best`. Same data, different data order. Same hyperparameters as v7: 1 epoch, lr 5e-5, warmup 100, best checkpoint picked on select. Plan: `kaggle/plans/v8.json`. Data: private HF dataset `five-letters-traces-v8`, 347,913 samples.

| | v7 drill | v8 drill |
|---|---|---|
| Samples per common word (5,557 words) | 30 | 40 |
| Guesses on the board before the target | 1-4 | 1-5 (weights 10/20/30/25/15%) |
| Rejected non-words per board | at most 1, on 15% of boards | at most 2, each with probability 30%, always leaving a turn for the target (measured: 0 / 1 / 2 on 71 / 22 / 7% of boards) |
| Target at turn 2 / 3 / 4 / 5 / 6 | 26 / 38 / 26 / 9 / 1% | 8 / 22 / 33 / 23 / 13% |
| Drill samples | 166,710 | 222,280 |
| Replay (half of the v5 and v5-extra teacher games) | 125,633 | 125,633 |

- Boards come out shallower than the weights ask for: a board stops early when no training-side word fits it any more. Measured depths of 1/2/3/4/5 guesses: 12/28/37/18/4%.
- The `max_invalid=1` path draws the same random numbers as before, so the v7 data can be rebuilt exactly. A test covers the new boards: the target always has a turn, at most two rejected words, and no board is built around a dev or test word.

**Fairness, measured on the built data.** Drill boards built around a dev or test word: 0. Rows from a game whose answer is a leaderboard word: 0. Replay games on a val, heldout, select or leaderboard answer: 0. Select words that are drill targets: 527 of 534. Leaderboard words that are drill targets: 147 of 148. This is the same disclosed choice as v7.

**Checks before launch.** 79 tests pass, lint clean. Local end-to-end run from the real v7 s1 weights: build -> 4 training steps -> eval alone, `--retries 3`, word-level -> independent audit -> gate command.

## Round v8 result (2026-11-10, Kaggle `bestman2/fl-v8`, 210.9 minutes, all 10 evals audited OK)

| Model | Select alone (strict) | Select retry-3 | Select word-level | Heldout alone | Heldout word-level |
|---|---|---|---|---|---|
| v5 s1 | 62.0% | 68.4% | 94.9% | 61.8% | 94.2% |
| v7 s1 (base of v8) | 74.9% | 82.0% | 97.2% | 76.7% | 97.6% |
| **v8 s1** | **82.2%** | 84.3% | **98.1%** | 82.3% | 98.5% |
| **v8 s2** | **83.3%** | 84.8% | **98.5%** | 84.2% | 98.9% |
| v8 mean of 2 seeds | **82.8%** (spread 1.1) | 84.6% | **98.3%** | 83.3% | 98.7% |

**What it shows.**

- Deeper boards work: +7.9 points alone on select (seed mean) over v7, and +1.1 with word-level decoding. Heldout moves the same way (+6.6 alone), so the gain is not a select artefact. The two seeds are 1.1 points apart.
- Word-level decoding is now 98.3% on select and 98.7% on heldout (seed means).
- Clue contradictions fell from 7.1% (v7) to 2.3-3.1%. The model now reasons from the clues almost every turn.
- Where v8 still loses (select, alone, strict): 91 of the 95 lost games of s1 (85 of 89 for s2) contain an invalid turn. Invalid rate by turn for s1: 0 / 0.2 / 5.0 / 23.2 / 54.0 / 75.0% (turns 1-6). 91% of the non-words fit the clues and 47% are one letter from the answer. The model picks the right shape of word and misspells it, mostly in the last three turns.
- The remaining gap between alone (82.8%) and word-level (98.3%) is almost all spelling. Under retry-3 the model gains only 1.8 points over strict.
- s2's best checkpoint is its last step (5,436) and s1's is step 5,000, the second-to-last eval: neither run shows a clear plateau.
- Boards in the v8 data came out shallower than the weights asked for (5-guess boards: 4% instead of 15%), so turn 6 is still under-drilled. That is where invalid turns peak.

**Decision (pre-registered "After v8" rule).** Seed mean on select is 82.8% alone (>= 76.9%) and 98.3% word-level (>= 96%): v8 is the final recipe. Next: seed 3 of v8, then declare the final model (best select seed) and run the single final test.

**Guard checks.** No select score alone above 90%; seeds 1.1 points apart (< 6); heldout tracks select. GPU this week: 19.40 h of 27.

## Diagnosis: stuck on a rejected word (2026-11-10, select, model alone, from the saved games)

Computed on the laptop from `kaggle_runs/{v7,v8}/runs/eval/select/*/results.jsonl` (scratch `v8_repeat.py`; the same counts are in `results/analysis/games.json`).

| | v7 s1 | v8 s1 | v8 s2 |
|---|---|---|---|
| Invalid turns (all are five-letter non-words) | 347 | 270 | 247 |
| ... first time the word is written in the game | 217 | 97 | 92 |
| ... exact repeat of a word already rejected in the game | 130 | 173 | 155 |
| Turn right after a first-time non-word: a valid word / the same non-word again | 13 / 124 | 1 / 88 | 3 / 78 |
| Lost games | 134 | 95 | 89 |
| Lost games whose invalid turns are one non-word, repeated | 52 | 83 | 73 |

- v8 writes a non-word in few games (s1: 91 of 534, 17%), but then it does not recover: the next turn is the same word again in 88 of 89 cases, and it loses every one of those 91 games. Example (answer BATON): CRATE, TALON, then SATON four times. The prompt does show `S A T O N -> INVALID (not in word list)`; the model's state line does not change, and it writes the same word.
- Wins by number of invalid turns in the game, v8 s1: none 439 of 443; one 0 of 2; two 0 of 14; three or more 0 of 75.
- The same happens under retry-3 (v8 s1: 223 first-time non-words, 438 repeats), which is why retry-3 adds only about 2 points.
- Under the leaderboard rule a non-word gets colours, the board changes and the model moves on: v8 s1 scores 94.4% there against 82.2% strict.
- So the remaining model-alone gap under the strict rule is one misspelling followed by no recovery. The drills did contain rejected words (v8: up to two per board), but those were random near-misses of some target, never the word the model itself would write next. The model never trained on a board where its own first choice had just been rejected. This is the covariate-shift problem again (Ross et al., DAgger), now for rejections.
- v7 recovered slightly more often than v8 (13 against 1 valid turn after a first non-word). We do not know why.

**Consequence: round v10 (recovery).** The base model plays training-side hidden words alone; each rejection of its own non-word becomes training boards with a real word that fits as the target (`fiveletters recovery-data`, `src/fiveletters/recovery.py`, test `test_recovery_samples_from_the_models_own_rejected_words`). Pre-registered in ROADMAP. The author chose on 2026-11-10 to hold the final test until this round is scored on select.

**Pre-launch check of `recovery-data` (laptop CPU, v8 s2, 100 of the 4,000 most common training-side hidden words, every 40th).** 50 of 100 games won; 43 games contain a non-word; 44 first-time non-words and 57 exact repeats (the stuck pattern also shows on training-side words); 36 events leave a turn; 54 samples. 28 of the 36 events have a single target, the hidden word, because nothing else fits the board. Targets sit at turns 4 / 5 / 6 in 5 / 24 / 25 samples. Consequence, recorded in ROADMAP as an amendment before launch: all 8,546 hidden words, replay 4%, 2 epochs.

## Round v9 result (2026-11-10, Kaggle `bestman2/fl-v9`, 212.8 minutes, all 12 evals audited OK)

| Model | Select alone | retry-3 | open-vocab | word-level | Heldout alone | Heldout word-level |
|---|---|---|---|---|---|---|
| v8 s1 | 82.2% | 84.3% | 94.4% | 98.1% | 82.3% | 98.5% |
| v8 s2 | 83.3% | 84.8% | 94.0% | 98.5% | 84.2% | 98.9% |
| **v8 s3** | **83.9%** | 85.0% | 94.4% | **98.7%** | 84.7% | 99.1% |
| v8 mean of 3 (sd) | **83.1% (0.9)** | 84.7% | 94.3% | 98.4% | 83.7% | 98.8% |
| v9 s1 (one more round from v8 s2) | 84.8% | 86.3% | 94.2% | 98.1% | 85.5% | 98.7% |

- The v8 recipe is stable across three seeds: 82.2 / 83.3 / 83.9% alone on select (sd 0.9), 98.1-98.7% word-level.
- v9 is 1.7 points above the v8 mean alone, but the difference to each v8 seed is within noise on the same 534 games (exact McNemar, scratch `v9_pair.py`): against v8 s3 only-v9 wins 29, only-v8 wins 24, p = 0.58; against s2 30 / 22, p = 0.33; against s1 36 / 22, p = 0.09. Its word-level score is below v8 s3's (98.1 against 98.7%). A second round of the same drill therefore gives little.
- Stuck pattern unchanged: after a first-time non-word the next turn is the same word in 76 of 84 cases (v8 s3) and 69 of 70 (v9 s1); 71 of 86 and 67 of 81 losses are one repeated non-word.
- Guards: no select score alone above 90%; v8 seeds 1.7 points apart; heldout tracks select.

**Decision (pre-registered "Final model" rule, applied to pick the v10 base).** v9 s1 needed at least the v8 three-seed mean + 2 = 85.15%; it has 84.83%. So the base is the v8 seed with the best select score alone: **`sft-gemma270m-v8-s3`** (83.9%, word-level 98.7%). This is the base of round v10, not yet the final model.

## Round v10 result (2026-11-10, Kaggle `bestman2/fl-v10`, 78.5 minutes, all 12 evals audited OK)

| Model | Select alone | retry-3 | open-vocab | word-level | Heldout alone | Heldout word-level |
|---|---|---|---|---|---|---|
| v8 s3 (base) | 83.9% | 85.0% | 94.4% | 98.7% | 84.7% | 99.1% |
| **v10 s1** | **88.0%** | 89.7% | 95.5% | **99.1%** | **88.3%** | 98.9% |
| v10 s2 | 87.3% | 90.1% | 92.9% | 98.1% | 87.3% | 98.3% |

- Data: the base played 4,273 training-side hidden words per GPU and won 1,340 (s1) and 1,403 (s2); these words are rarer than the answer lists. 2,261 and 2,162 rejection events gave 3,077 and 2,879 recovery boards.
- Both seeds gain about 4 points alone on select and on heldout. Heldout was not used for checkpoint choice, so the gain is not a selection effect. Paired on the same 534 select games (exact McNemar, `paper/analysis.py`): v10 s1 against v8 s3, only-v10 wins 47, only-v8 wins 25, p = 0.013; v10 s2 against v8 s3 36 / 18, p = 0.020; v10 s1 against v10 s2 28 / 32, p = 0.70.
- The stuck pattern is mostly gone. After a first-time non-word the next turn is a real word in 51 cases and the same non-word in 32 (v8 s3: 8 and 76). Losses that are one repeated non-word fall from 71 of 86 to 15 of 64. Games with exactly one invalid turn are now won 39 of 42 times (v8 s3: 3 of 8).
- What is left: the model writes a first non-word slightly more often (101 games with a non-word against 85; 150 first-time non-words against 91), and 41 games still have three or more invalid turns, all lost. 206 of 223 non-words fit the clues; 97 are one letter from a real word.
- Guards: no strict select score alone above 90% (88.0 and 87.3%). v10 s2's retry-3 score is 90.1%; retry-3 is a more lenient rule, like open-vocab (94% since v8), and the guard was written for the strict score alone, so it is not tripped. All 12 evals pass the replay audit. Seeds 0.7 points apart; heldout tracks select.

**Decision (pre-registered v10 rule).** The better v10 seed alone on select is s1 (88.01%). It needed at least the base + 2 = 85.9% alone and at least the base's word-level − 1 = 97.7%; it has 88.0% and 99.1%. **Final model: `sft-gemma270m-v10-s1`.** Declared here, before any test set is scored for it. The final test runs once from `kaggle/plans/final.json`.

## Final test result (2026-11-10, Kaggle `bestman2/fl-final` version 1, 53.6 minutes, commit 8daddd3)

Declared final model `sft-gemma270m-v10-s1` (see "Round v10 result"), scored once. All 12 eval folders pass `scripts/audit_eval.py`; the two TextArena runs are scored by TextArena's own engine (adapted mode) and are not replayed by our audit script. Win rate in % (95% CI), guesses per win in brackets.

| Set | Strict, alone | retry-3, alone | open-vocab, alone (leaderboard rule) | Word-level decoding | Cited best (leaderboard rule, no dictionary) |
|---|---|---|---|---|---|
| WordleBench (100) | 87.0 (80.0-93.0) [3.79] | 87.0 (80.0-93.0) [3.71] | 94.0 (89.0-98.0) [3.88] | **97.0 (93.0-100.0) [3.68]** | GPT-5 99 [3.59] |
| Jeffers (50) | 74.0 (62.0-86.0) [3.86] | 80.0 (68.0-90.0) [3.85] | 88.0 (78.0-96.0) [3.98] | **96.0 (90.0-100.0) [3.75]** | GPT-5 100 [3.58] |
| rare (500) | 42.2 (38.2-46.6) [4.58] | 46.0 (42.0-50.6) [4.47] | 51.4 (47.2-55.8) [4.67] | 76.6 (73.0-80.4) [4.60] | - |
| TextArena adapted, all 500 | 85.8 (82.8-88.6) [3.63] | - | - | 98.6 (97.6-99.4) [3.59] | - |
| TextArena adapted, clean 109 | 84.4 (78.0-90.8) [3.54] | - | - | 97.2 (93.6-100.0) [3.63] | - |

- Against the earlier declared final model (SFT v5 s1, same sets): word-level 95.0 -> 97.0 on WordleBench and 86.0 -> 96.0 on Jeffers; alone 60.0 -> 87.0 and 54.0 -> 74.0.
- Under the leaderboards' own rule (open-vocab) the model alone wins 94% of WordleBench games; the WordleBench leaderboard lists many frontier models at 94-98% under that rule (different harness, prompts and sampling).
- Jeffers alone (74%) is below select (88.0%); with 50 games the CI is 62-86%. Word-level decoding brings it to 96%.
- Rare answers (outside the classic answer list) stay hard: 42.2% alone, 76.6% word-level, 4.6 guesses per win. The model's word knowledge is concentrated on common words.
- No further training or selection uses these numbers.

## v10 recovery samples regenerated for the data release (2026-11-10, Kaggle `bestman2/fl-v10data`, 21.8 minutes)

The original v10 samples were generated inside `fl-v10` and not kept. `kaggle/plans/v10data.json` reruns the same two `recovery-data` commands (same model, words, parts, seeds, GPU type) with the output kept. All seven printed statistics match the original run for both seeds (games 4,273 / 4,273; won 1,340 / 1,403; games with a non-word 2,509 / 2,385; first-time non-words 2,727 / 2,567; repeated non-words 3,621 / 3,445; events 2,261 / 2,162; samples 3,077 / 2,879). Published as `aafimalek2032/five-letters-traces-v10`.

Appendix of the paper ("More games"): every WordleBench game the final model lost alone (13: ten contain non-words, nine with a rejected word written again; three are trap sets of real words, BONUS, LASER, WATER), every game lost with word-level decoding (the same three), and every game won alone after a rejected word (7).
