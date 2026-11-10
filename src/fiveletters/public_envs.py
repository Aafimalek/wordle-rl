"""Evaluation in the public Wordle environments: TextArena `Wordle-v0` and Prime Intellect verifiers `wordle`.

Both run the same TextArena engine. verifiers' `wordle` taskset makes task i with
`random.seed(i); ta.make("Wordle-v0"); reset()`, so seed i is the same secret word in both.
What differs is how the model is prompted:

* ``native-textarena``: one user message holding TextArena's full observation (its rules
  prompt plus the whole history), as TextArena's own LLM agents receive it.
* ``native-verifiers``: verifiers' protocol. Its system prompt and the first observation,
  then only the latest feedback as each new user turn, in one growing chat.
* ``adapted``: our prompt format, rebuilt each turn from the engine's state. Our parsed
  guess is sent to the engine as ``[word]``. Validity, feedback and the outcome all come
  from the engine. This measures playing skill under the public rules and words for a
  model trained on our format.

Rules that differ from our protocol (and must be reported):
* secret words are 5-letter *nouns* from NLTK's basic English list, not the Wordle answers;
* guesses are checked against a general English dictionary, not the Wordle guess list;
* a repeated guess is invalid; one invalid move per game is forgiven and does not use a
  turn, and the second one loses the game.

Runs are resumable and refuse mixed settings, exactly like ``eval/harness.py``.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np

from . import PROTOCOL_VERSION
from .eval.harness import EvalIncomplete, _read_results
from .feedback import feedback as our_feedback
from .feedback import from_str
from .game import ERR_FORMAT, ERR_NOT_A_WORD, ERR_REPEAT, Game, Turn
from .policies.base import Policy, QuotaExhausted
from .prompts import parse_guess
from .runtime import record_provenance
from .words import get_split, load

MODES = ("native-textarena", "native-verifiers", "adapted")

# verifiers v0.4.1, verifiers/v1/tasksets/textarena/taskset.py. Imported from verifiers when it
# is installed (the meta file records which), so the protocol cannot silently drift.
_VERIFIERS_SYSTEM_PROMPT = (
    "You are a competitive game player. Read the game instructions carefully and always "
    "use the exact move format the game requires. Think step-by-step first, then give your "
    "move as the only square-bracketed token in your reply — the game reads the first "
    "bracketed token, so don't put other words in brackets."
)


def _verifiers_protocol():
    try:
        from verifiers.v1.tasksets.textarena.taskset import SYSTEM_PROMPT, _latest_feedback

        return SYSTEM_PROMPT, _latest_feedback, "verifiers"
    except Exception:  # noqa: BLE001 - verifiers is optional
        def _latest(observation: str) -> str:
            latest = observation.split("[GAME]")[-1].strip()
            return latest.split("Feedback:")[-1].strip() if "Feedback:" in latest else latest

        return _VERIFIERS_SYSTEM_PROMPT, _latest, "copy-of-verifiers-0.4.1"


def _ta():
    try:
        import nltk
        import textarena as ta
    except ImportError as e:
        raise SystemExit("public-env evaluation needs: pip install textarena nltk") from e
    nltk.download("words", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    return ta


def _unwrap(env):
    while not hasattr(env, "_evaluate_guess") and hasattr(env, "env"):
        env = env.env
    return env


def make_game(seed: int, env_id: str = "Wordle-v0"):
    """The engine for task `seed`, made exactly the way verifiers makes it."""
    ta = _ta()
    random.seed(seed)
    env = ta.make(env_id=env_id)
    env.reset(num_players=1)
    return env


def _history(env) -> list:
    return list(_unwrap(env).state.game_state["guess_history"])


class _Episode:
    def __init__(self, seed: int, env_id: str, mode: str, system_prompt: str):
        self.seed, self.mode = seed, mode
        self.env = make_game(seed, env_id)
        self.core = _unwrap(self.env)
        self.secret = self.core.state.game_state["secret_word"]
        self.replies: list[str] = []
        _, obs = self.env.get_observation()
        self.obs = str(obs)
        self.chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": self.obs}]
        self.mirror = Game(self.secret)  # our-format view of the game, for adapted mode
        self.done = False

    def conversation(self) -> list[dict]:
        if self.mode == "native-textarena":
            return [{"role": "user", "content": self.obs}]
        return self.chat

    def step(self, reply: str, latest_feedback) -> None:
        self.replies.append(reply)
        before = len(_history(self.env))
        if self.mode == "adapted":
            guess = parse_guess(reply)
            action = f"[{guess}]" if guess else reply.replace("[", "(").replace("]", ")")
        else:
            guess, action = None, reply
        self.env.step(action)
        hist = _history(self.env)
        if self.mode == "adapted":
            if len(hist) > before:
                word, fb = hist[-1]
                self.mirror.turns.append(Turn(reply, word, True, from_str("".join(fb))))
            else:
                prev = [w for w, _ in hist]
                err = ERR_FORMAT if guess is None else ERR_REPEAT if guess in prev else ERR_NOT_A_WORD
                self.mirror.turns.append(Turn(reply, guess, False, error=err))
                self.mirror.max_turns += 1  # TextArena does not spend a turn on a forgiven invalid move
        if self.core.state.done:
            self.done = True
            return
        _, obs = self.env.get_observation()
        self.obs = str(obs)
        self.chat += [{"role": "assistant", "content": reply}, {"role": "user", "content": latest_feedback(self.obs)}]

    def record(self, wl, train: set[str]) -> dict:
        hist = _history(self.env)
        won = bool(hist) and all(f == "G" for f in hist[-1][1])
        st = self.core.state
        return {
            "seed": self.seed, "answer": self.secret, "won": won, "n_turns": len(hist),
            "n_invalid": int(getattr(st, "error_count", 0)) + int(bool(st.game_info[0].get("invalid_move"))),
            "lost_by_invalid": bool(st.game_info[0].get("invalid_move")),
            "textarena_reward": float((st.rewards or {}).get(0, 0.0)),
            "reason": str(st.game_info[0].get("reason", "")),
            "secret_in_wordle_lists": self.secret in wl.guess_set,
            "secret_in_train_answers": self.secret in train,
            "guesses": [[w, "".join(f)] for w, f in hist],
            "replies": [r[-4000:] for r in self.replies],
        }


def _policy_turn(policy: Policy, episodes: list[_Episode], mode: str) -> list[str]:
    if mode == "adapted":
        return policy.act([e.mirror for e in episodes])
    chat = getattr(policy, "chat", None)
    if chat is None:
        raise SystemExit(f"{policy.name} cannot play native mode (no chat). Use --mode adapted.")
    return chat([e.conversation() for e in episodes])


def summarize_public(records: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    def block(rs):
        if not rs:
            return {"games": 0}
        won = np.array([r["won"] for r in rs], float)
        turns = np.array([r["n_turns"] for r in rs], float)
        cost = np.where(won > 0, turns, 7)
        idx = np.random.default_rng(seed).integers(0, len(rs), size=(n_boot, len(rs)))
        lo, hi = np.percentile(won[idx].mean(1), [2.5, 97.5])
        return {
            "games": len(rs), "unique_answers": len({r["answer"] for r in rs}),
            "win_rate": round(float(won.mean()), 4), "win_rate_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "mean_guesses_wins": round(float(turns[won > 0].mean()), 4) if won.any() else None,
            "mean_guesses_fail7": round(float(cost.mean()), 4),
            "mean_textarena_reward": round(float(np.mean([r["textarena_reward"] for r in rs])), 4),
            "games_lost_by_invalid_moves": int(sum(r["lost_by_invalid"] for r in rs)),
        }

    return {
        "all": block(records),
        # answers our model could never have trained on, and that are valid Wordle words
        "clean": block([r for r in records if r["secret_in_wordle_lists"] and not r["secret_in_train_answers"]]),
        "secret_outside_wordle_lists": block([r for r in records if not r["secret_in_wordle_lists"]]),
    }


def run_public_eval(policy: Policy, out_dir: str | Path, *, env_id: str = "Wordle-v0", mode: str = "adapted",
                    seeds: range = range(500), batch_size: int = 32, time_budget_s: float | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    import importlib.metadata as md

    system_prompt, latest_feedback, protocol_source = _verifiers_protocol()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "protocol": PROTOCOL_VERSION, "public_env": env_id, "mode": mode,
        "seeds": [seeds.start, seeds.stop], "textarena_version": md.version("textarena"),
        "verifiers_protocol_source": protocol_source if mode == "native-verifiers" else None,
        "policy": policy.describe(),
    }
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise RuntimeError(f"{out} holds results from different settings. Use a new --out folder.")
    meta_path.write_text(json.dumps(meta, indent=2))
    record_provenance(out, job="public-eval")

    wl, train = load(), set(get_split("train"))
    results_path = out / "results.jsonl"
    records = _read_results(results_path)
    done = {r["seed"] for r in records}
    todo = [s for s in seeds if s not in done]
    started = time.monotonic()
    print(f"[public] {env_id} {mode}: {len(done)}/{len(seeds)} done, {len(todo)} to go")
    try:
        with open(results_path, "a", encoding="utf-8", newline="\n") as f:
            for s in range(0, len(todo), batch_size):
                if time_budget_s and time.monotonic() - started > time_budget_s:
                    raise EvalIncomplete(len(records), len(seeds), "time budget reached")
                eps = [_Episode(seed, env_id, mode, system_prompt) for seed in todo[s : s + batch_size]]
                active = eps
                turn = 0
                while active:
                    turn += 1
                    t0 = time.monotonic()
                    try:
                        replies = _policy_turn(policy, active, mode)
                        print(f"[public] turn {turn}: {len(active)} game(s) in {time.monotonic() - t0:.0f}s", flush=True)
                    except QuotaExhausted as e:
                        raise EvalIncomplete(len(records), len(seeds), str(e)) from e
                    for e, reply in zip(active, replies, strict=True):
                        e.step(reply, latest_feedback)
                    active = [e for e in active if not e.done]
                for e in eps:
                    rec = e.record(wl, train)
                    f.write(json.dumps(rec) + "\n")
                    records.append(rec)
                f.flush()
                os.fsync(f.fileno())
                print(f"[public] {len(records)}/{len(seeds)} games, win rate so far "
                      f"{sum(r['won'] for r in records) / len(records):.3f}")
    except EvalIncomplete:
        (out / "summary.partial.json").write_text(json.dumps({**meta, **summarize_public(records)}, indent=2))
        raise
    finally:
        policy.close()
    summary = {**meta, **summarize_public(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def crosscheck_feedback(n_pairs: int = 10000, seed: int = 0) -> dict:
    """Compare our feedback rule with TextArena's `_evaluate_guess` on random and repeated-letter pairs."""
    core = _unwrap(make_game(0))
    wl = load()
    rng = random.Random(seed)
    secrets = sorted(set(wl.answers) | {w for w in getattr(core, "word_list", []) if len(w) == 5 and w.isalpha()})
    repeats = [w for w in wl.guesses if len(set(w)) < 5]
    pairs = [(rng.choice(wl.guesses), rng.choice(secrets)) for _ in range(n_pairs // 2)]
    pairs += [(rng.choice(repeats), rng.choice([w for w in secrets if len(set(w)) < 5] or secrets))
              for _ in range(n_pairs - len(pairs))]
    mismatches = []
    for guess, secret in pairs:
        core.state.game_state["secret_word"] = secret
        theirs = "".join(core._evaluate_guess(guess.lower()))
        ours = "".join("XYG"[v] for v in our_feedback(guess, secret.lower()))
        if theirs != ours:
            mismatches.append({"guess": guess, "secret": secret, "ours": ours, "textarena": theirs})
    import importlib.metadata as md

    return {"pairs": len(pairs), "repeated_letter_pairs": n_pairs - n_pairs // 2, "mismatches": len(mismatches),
            "examples": mismatches[:20], "textarena_version": md.version("textarena"),
            "textarena_secret_words": len(getattr(core, "word_list", []))}
