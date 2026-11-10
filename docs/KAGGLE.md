# Running GPU jobs on Kaggle from the laptop

`kaggle/job.py` starts, watches and collects Kaggle GPU sessions (free T4 x2) without the website. Each session runs one command queue per GPU, both GPUs in parallel. All commands in a session share one time budget (`--hours`, default 8). A command that hits it saves, uploads its checkpoint and exits with code 75. Push the same job again and every command resumes by itself.

## One-time setup

| File on the laptop | Holds | Used for |
|---|---|---|
| `~/.kaggle/access_token` | Kaggle API token | pushing and reading jobs |
| `~/.five-letters-secrets/hf_token` | Hugging Face token | Gemma download, checkpoints and datasets on the Hub |
| `~/.five-letters-secrets/github_token` | GitHub fine-grained token, read-only Contents (only for a private copy of the repository) | cloning the repository inside the session |

```
python kaggle/job.py secrets   # uploads hf_token and github_token to the PRIVATE dataset five-letters-secrets
python kaggle/job.py cache     # uploads solver caches so sessions skip minutes of CPU work
```

Rerun `secrets` after rotating a token.

## Jobs

```
python kaggle/job.py push NAME --branch development --hours 8 \
    --gpu0 "sft --config configs/sft_gemma270m.toml --set checkpoint.hub_repo=aafimalek2032/five-letters-runs --set data.hub_dataset=aafimalek2032/five-letters-traces" \
    --gpu1 "eval --policy hf --model google/gemma-3-270m-it --split heldout"
python kaggle/job.py status NAME
python kaggle/job.py fetch NAME    # logs/, job_summary.json, eval results, metrics (no checkpoints)
```

Fetched files land in `kaggle_runs/NAME/`, which git ignores. Checkpoints live in a private Hugging Face model repo (`checkpoint.hub_repo`); the final model is public at `aafimalek2032/wordle-rl-gemma3-270m`.

## Switching to a new Kaggle account (fresh GPU quota)

The owner is read from the token (`kaggle config view`), so no code changes are needed.

1. Put the new account's token in `~/.kaggle/access_token` (keep the file private). Check it: `kaggle config view` shows the new username.
2. `python kaggle/job.py secrets`: re-creates the private `five-letters-secrets` dataset under the new account (HF and GitHub tokens stay the same).
3. `python kaggle/job.py cache`: re-uploads the solver caches.
4. The new account needs a verified phone number (Kaggle settings) to use GPUs. Gemma itself downloads with the HF token, so no Kaggle licence step is needed.
5. Launch from a plan file: `python kaggle/job.py plan kaggle/plans/v5.json session1`.

Kernels attached with `from_kernel` must belong to the same account. Models are also on the HF runs repo, so a later session on another account still finds them.

## Plan files

`kaggle/plans/*.json` hold reviewable sessions: `{"<session>": {"name", "hours", "pip", "gpu0": [...], "gpu1": [...], "from_kernel": [...]}}`. `python kaggle/job.py plan FILE SESSION` pushes one. `kaggle/plans/v5.json`:

| Session | GPU 0 | GPU 1 |
|---|---|---|
| `session1` | SFT v5 seed 1, heldout dev evals, **gate**, GRPO v5 seed 1, dev evals | waits for the gate, then SFT v5 seed 2, dev evals |
| `session2-grpo-seeds` | GRPO v5 seed 2, dev evals | SFT v5 seed 3, GRPO v5 seed 3, dev evals |
| `final-test` | WordleBench and Jeffers, unconstrained and constrained | TextArena adapted (both), rare constrained |

The gate (`fiveletters gate`) passes only if SFT v5 seed 1 + constrained decoding beats SFT v3's 79.5% on the full heldout set. If it fails, both queues stop (exit code 76) and the session ends early, so no quota goes to GRPO or extra seeds on a recipe that did not improve. Session 2 is launched by hand after reviewing session 1: GRPO seeds run only if GRPO s1 beats SFT s1 on the select split.

Before `final-test`, replace `FINAL_RUN` with the run chosen by the **select** split (never by a benchmark), and run it once.

## Tips that save quota

* Pair a long job on GPU 0 with evals or an ablation on GPU 1. Quota is counted per session hour, not per GPU.
* CPU work (traces, solver and API baselines) runs on the laptop and costs no quota.
* Kaggle allows only a few GPU sessions at once. Prefer one session with two queues.
