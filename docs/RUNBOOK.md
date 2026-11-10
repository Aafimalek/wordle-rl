# Runbook: the plan, step by step

Each phase lists the exact commands and the check that says it is done. Commands are run from the repository root. `python -m fiveletters` and `fiveletters` are the same thing.

## P0: set up (week 1)

1. Accounts: GitHub, Hugging Face, Kaggle (verify your phone to unlock GPUs), Google (for Colab and AI Studio). Optionally apply to Google TPU Research Cloud.
2. Hugging Face: create a **write** token. Accept the license at `google/gemma-3-270m-it`.
3. GitHub: only if your copy of the repository is private, create a fine-grained token with read-only *Contents* access to it.
4. Kaggle: add the secret `HF_TOKEN` (and `GITHUB_TOKEN` for a private copy). Colab: add them under *Secrets*.
5. Locally:
   ```
   pip install -e ".[train,dev]"
   python -m fiveletters check
   python -m pytest
   ```
6. Optional warm-up: run the official TRL + OpenEnv Wordle GRPO tutorial once on Kaggle to see a reference training loop.

**Done when** the tests pass locally and `fiveletters info` runs inside the Kaggle launcher notebook.

## P1: engine and teacher (done in this branch)

The game engine, repeated-letter rule, hard mode and teacher solvers are implemented and tested. Measured on all 2,315 answers:

| Solver | Mean guesses (loss = 7) | Wins |
|---|---|---|
| lookahead (teacher) | 3.4216 | 100% |
| entropy (greedy) | 3.4721 | 100% |
| known optimum | 3.4212 | 100% |

Still open: cross-check our feedback against TextArena's Wordle on random pairs (`pip install textarena`, then compare on 10,000 pairs). This is a must before the paper says "same rules as TextArena".

## P2: baselines (weeks 2-3)

```
# solver baselines (CPU)
fiveletters eval --policy solver:lookahead --split heldout
fiveletters eval --policy solver:entropy   --split heldout
fiveletters eval --policy solver:random    --split heldout
fiveletters eval --policy solver:entropy   --split rare

# zero-shot small models (Colab T4, or CPU if patient)
fiveletters eval --policy hf --model google/gemma-3-270m-it --split heldout
fiveletters eval --policy hf --model HuggingFaceTB/SmolLM2-360M-Instruct --split heldout
fiveletters eval --policy hf --model Qwen/Qwen3-0.6B --split heldout

# public Wordle RL checkpoints (the tier 1 bar). Check each model card for its prompt format.
fiveletters eval --policy hf --model PrimeIntellect/Qwen3-1.7B-Wordle-RL --split heldout

# frontier models through free APIs (laptop; resumes if the daily cap hits)
set GEMINI_API_KEY=...        (PowerShell: $env:GEMINI_API_KEY="...")
fiveletters eval --policy api --provider gemini --model gemini-2.5-flash --split heldout --rpm 8
fiveletters eval --policy api --provider groq --model llama-3.3-70b-versatile --split heldout --rpm 25

fiveletters report             # markdown table of everything so far
```

Free-tier model names change. Check each provider's list of free models before running.

**Done when** `fiveletters report` has every baseline, then tag the protocol: `git tag eval-v1`.

## P2b: public environments

```
pip install textarena nltk
fiveletters crosscheck-textarena --out results/eval-v1/crosscheck_textarena.json   # must report 0 mismatches
fiveletters public-eval --mode adapted --policy solver:lookahead --vocab shared --seeds 0:500
fiveletters public-eval --mode native-verifiers --policy hf --model PrimeIntellect/Qwen3-1.7B-Wordle-RL --max-new-tokens 1024
fiveletters public-eval --mode native-verifiers --policy api --provider nvidia --model minimaxai/minimax-m3.1-flash
```

Report the `all` and `clean` blocks of each summary. On Windows, TextArena needs `PYTHONUTF8=1`.

## P3: SFT (weeks 3-5)

1. Traces, on the laptop or a Kaggle CPU session:
   ```
   fiveletters traces --out data/traces/v2 --vocab shared --opener crate --invalid-prob 0.1        --push-to YOUR_HF_NAME/five-letters-traces-v2 --private
   ```
   About 170k samples. The first run spends a few minutes caching the teacher's search tree.
2. SFT on Kaggle. In `notebooks/kaggle_launcher.ipynb` set:
   ```
   COMMAND = "sft --config configs/sft_gemma270m.toml --set checkpoint.hub_repo=YOUR_HF_NAME/five-letters-runs --set data.hub_dataset=YOUR_HF_NAME/five-letters-traces"
   ```
   Then *Save Version → Save & Run All*. If it ends with "paused", run it again. It resumes.
3. Evaluate the best checkpoint:
   ```
   fiveletters eval --policy hf --model hub:YOUR_HF_NAME/five-letters-runs:sft-gemma270m-v1/best --split heldout
   ```

**Done when** the SFT model beats every sub-1B baseline on `heldout`, with an invalid-turn rate under 1%.

## P4: GRPO (weeks 5-8)

```
COMMAND = "grpo --config configs/grpo_gemma270m.toml --set checkpoint.hub_repo=YOUR_HF_NAME/five-letters-runs --set model.init=hub:YOUR_HF_NAME/five-letters-runs:sft-gemma270m-v1/best"
```

Watch `no_signal_groups` in the logs. If it stays near 1.0, every game in a group gets the same reward and nothing is learned. Raise the temperature, use more curriculum prefixes, or raise `speed_bonus`.

**Done when** GRPO beats SFT on `heldout` by more than the confidence interval.

## P5: ablations (weeks 8-10)

Each ablation is a separate run name plus `--set` overrides. Traces for format ablations need their own folder.

| Question | How |
|---|---|
| Letters vs whole words | `fiveletters traces --unspaced --out data/traces/unspaced`, then SFT with `data.traces_dir=data/traces/unspaced`, `run.name=sft-unspaced` |
| State line or not | `traces --no-state --out data/traces/nostate`, then SFT |
| SFT vs GRPO vs both | GRPO with `model.init=google/gemma-3-270m-it` (no SFT), `run.name=grpo-from-base` |
| Reward shape | `--set reward.binary=true`, or `--set reward.state_bonus=0.2` |
| Model size | `--set model.name=HuggingFaceTB/SmolLM2-135M-Instruct` and `...360M-Instruct` |
| Data size | `--set data.max_samples=10000` (also 50000, 200000) |
| Hard mode | `traces --hard`, GRPO with `--set format.hard_mode=true`, eval with `--hard` |
| Seeds | `--set run.seed=2` and `3` for the main runs |

## P6-P7: paper and release

`fiveletters report` gives the main table. Release the model, the traces dataset and this repo, then write the paper. See the plan page for venues and arXiv rules.
