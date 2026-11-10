# Running on free resources

The project costs $0. This file lists what each free platform gives, where each job runs, and every failure the code is built to survive.

## Where each job runs

| Job | Needs a GPU? | Run it on | Notes |
|---|---|---|---|
| Tests, word-list checks | no | laptop | `pytest` (about 2 minutes) |
| Solver baselines | no | laptop | about 10 minutes the first time, while the teacher's search is cached |
| Teacher traces | no | laptop, or a **Kaggle CPU** session (no GPU quota used) | resumable, multi-process |
| API baselines (Gemini, Groq, ...) | no | laptop | resumable across days as free daily caps reset |
| Zero-shot small-model baselines | yes, briefly | Colab T4 | a few minutes each |
| SFT | yes | Kaggle T4 | about 3-5 h for the default config (measure and update) |
| GRPO | yes | Kaggle T4 | about 6 h for 600 steps (measure and update) |
| Final evals | yes | Kaggle or Colab | minutes |

Never spend GPU quota on CPU jobs. On Kaggle, set the accelerator to *None* for them.

## Platform limits (checked 2026-11-08, they change often)

| Platform | GPU | Limit | Our default |
|---|---|---|---|
| Kaggle | T4 x2, 16 GB each | about 30 GPU h/week, sessions about 9-12 h | time budget 8 h |
| Colab free | T4, 15 GB | unpublished, disconnects when idle or busy | time budget 11 h, keep Hub sync on |
| Hugging Face Hub | none | free model and dataset storage | one private runs repo, history squashed |
| GitHub Actions | none | free for public repos | CPU test suite on every push |

T4s have no bfloat16, and Gemma 3 overflows in float16, so on T4 everything runs in float32. At 270M parameters this fits: weights, gradients and AdamW state are about 4.3 GB.

## Failure cases and what the code does

| What happens | What the code does | Where |
|---|---|---|
| Session reaches its time limit | Stops `reserve_minutes` before the budget, saves, uploads to the Hub, exits with code 75 ("paused") | `runtime.TimeBudget`, trainers |
| Session is killed without warning (Colab disconnect, kernel crash) | Local checkpoint every 20 min and Hub upload every 30 min, so at most about 30 min of work is lost | `runtime.Checkpointer` |
| Killed while writing a checkpoint | Writes go to `step-N.tmp` and are renamed only when complete. `LATEST` is replaced atomically. The previous checkpoint stays intact. Leftover `.tmp` folders are deleted. | `Checkpointer.save` |
| New session on a fresh machine | Resumes from the local checkpoint, otherwise downloads `runs-repo/<run>/latest` | `Checkpointer.find_resume_point` |
| Platform sends SIGTERM before shutdown | Saves and exits at the end of the current step. A second signal exits at once. | `runtime.StopFlag` |
| Hub upload fails (network, rate limit) | Logged. Training continues, and the next sync retries. | `HubSync.upload` |
| Free Hub storage fills up with old uploads | Each upload replaces the slot's files, and the repo history is squashed | `HubSync` |
| Resumed with different hyperparameters by mistake | Refused, unless `--allow-config-change` is passed. The change is then logged in `config_changes.jsonl`. Session settings (budget, checkpointing) may change freely. | `check_or_write_config` |
| A finished run is started again | Prints "already completed" and exits without retraining | `COMPLETED` marker |
| Resumed run sees different batches | It cannot: SFT batch order and GRPO answers/prefixes are pure functions of (seed, step). RNG states are restored too. | `sft._step_plan`, `grpo.plan_step` |
| GPU count changes between sessions | CUDA RNG restore is skipped with a note, and everything else resumes | `set_rng_state` |
| Eval killed halfway | Every finished game is flushed to `results.jsonl`. A rerun skips them, and a half-written last line is cut off. | `eval/harness.py` |
| Eval rerun with different settings into the same folder | Refused, so results never mix | `meta.json` check |
| Free API daily cap or long rate limiting | Requests-per-minute limiter, backoff honoring `Retry-After`, then a clean stop with partial results saved. Rerun tomorrow to continue. | `policies/api.py` |
| Bad or missing API key | Fails at once with the variable name to set, without burning retries | `ChatAPIPolicy` |
| Trace generation killed | Shards are written to `.tmp` and renamed when complete. A rerun skips finished shards and gives identical data. | `traces.py` |
| Corrupt cache file (pattern table, solver search) | Detected and rebuilt | `patterns.py`, `solvers.py` |
| Word list edited or line endings changed by git | Checksums ignore line endings. Any real change fails `fiveletters check` and every load. | `words.py`, `.gitattributes` |
| Kaggle output folder fills up (20 GB) | Only the 2 newest checkpoints are kept (about 3.3 GB each in float32 with optimizer) | `checkpoint.keep` |
| Weekly GPU quota runs out | Nothing breaks. Do CPU work (traces, analysis, writing) until it resets, or continue on Colab: the run resumes from the Hub. | this document |

## What you must set up once

* A GitHub fine-grained token with read-only *Contents* access to this private repo, saved as the `GITHUB_TOKEN` secret. The launchers send it as a header, so it is never stored in the clone or printed.
* A Hugging Face account and a **write** token, saved as the `HF_TOKEN` secret on Kaggle and Colab. Never commit it.
* Accept the Gemma 3 license on the model page with the same account.
* `checkpoint.hub_repo = "YOUR_HF_NAME/five-letters-runs"`. Without it, a dead session can only resume from its own disk, which Kaggle and Colab erase.
