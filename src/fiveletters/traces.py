"""Teacher traces for SFT: (prompt, completion) pairs from the lookahead solver.

Only training answers are used. Val and held-out answers never appear as targets.

Each game mixes three kinds of moves so the model learns to recover, not just to
replay one tree:

* prefix moves: 0-2 random valid words at the start (part of the history, not targets);
* noise moves: with probability ``noise``, a random word that fits the clues (history only);
* teacher moves: the lookahead solver's choice while the game is on its own path, and the
  greedy entropy solver's choice after a prefix or noise move. Only these become targets.

Output is sharded JSONL. Each shard is written to a temp file and renamed when
complete, so a killed run leaves no half shards, and a rerun skips finished shards.
Everything is seeded per (seed, word, game) with a stable hash, so the dataset is
identical on any machine, any worker count, and after any number of restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .game import Game
from .prompts import PromptFormat, render_completion, render_prompt
from .solvers import EntropySolver, LookaheadSolver
from .words import extra_answer_words, get_split, guess_vocab, load, sha256_words


@dataclass(frozen=True)
class TraceConfig:
    games_per_word: int = 30
    words_per_shard: int = 50
    seed: int = 1
    prefix_probs: tuple[float, ...] = (0.6, 0.25, 0.15)  # P(0, 1, 2 random opening words)
    noise: float = 0.1
    alt_opener_prob: float = 0.3  # use one of the top-20 openers instead of the teacher's own
    spaced: bool = True
    state_line: bool = True
    hard_mode: bool = False
    # 'shared': the teacher (and random prefixes) only use words that both Wordle and TextArena accept,
    # so one model is valid in our benchmark and in the public environments.
    vocab: str = "wordle"
    opener: str | None = None  # teacher opener; None = the solver default (SALET for 'wordle')
    invalid_prob: float = 0.0  # chance per game of one invalid turn in the history, to teach recovery
    # 'all': candidates include non-answer words. Needed when answers come from extra_answer_words(), so the
    # model learns to find any real word that fits the clues instead of recalling the training answers.
    answer_pool: str = "answers"
    # 'optimal': the lookahead teacher, whose best moves are often probe words that break known clues.
    # 'consistent': every teacher guess fits all clues so far (hard-mode style search while the game itself
    # stays in normal mode). A 270M student can learn "guess a word that fits what I know"; it could not
    # reproduce the probes (diagnosis 2026-11-09: 86% of guesses in its lost games contradicted known clues).
    teacher: str = "optimal"  # or 'common-probe' (round v6, see solvers.CommonProbeSolver)


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


_WORKER: dict = {}


def _teachers(cfg: TraceConfig):
    if cfg.teacher not in ("optimal", "consistent", "common-probe"):
        raise ValueError(f"unknown teacher {cfg.teacher!r}")
    if cfg.teacher == "common-probe":  # round v6: fits the clues, prefers common words, probes on traps
        from .solvers import CommonProbeSolver

        key = ("cp", cfg.vocab, cfg.opener, cfg.answer_pool)
        if key not in _WORKER:
            main = CommonProbeSolver(pool=cfg.answer_pool, vocab=cfg.vocab, opener=cfg.opener or "crate")
            _WORKER[key] = (main, main, main.ranked_openers(20))
        return _WORKER[key]
    # the solver's hard_mode restricts its guesses to words that fit every clue; the game rules stay cfg.hard_mode
    fit = cfg.hard_mode or cfg.teacher == "consistent"
    key = ("t", cfg.hard_mode, cfg.vocab, cfg.opener, cfg.answer_pool, cfg.teacher)
    if key not in _WORKER and cfg.answer_pool == "all":
        # A deep search over 12,972 candidates is far too slow; greedy entropy plays every move here.
        main = EntropySolver(pool="all", hard_mode=fit, vocab=cfg.vocab, opener=cfg.opener)
        _WORKER[key] = (main, main, main.ranked_openers(20))
    if key not in _WORKER:
        kw = {"opener": cfg.opener} if cfg.opener else {}
        main = LookaheadSolver(hard_mode=fit, vocab=cfg.vocab, **kw)
        # Off-tree states (after random prefixes or noise) are almost all new, so a deep search
        # there costs minutes per game. Greedy entropy (3.47 mean vs 3.42) is ample for teaching recovery.
        fast = EntropySolver(hard_mode=fit, vocab=cfg.vocab)
        openers = [w for w in main.ranked_openers(20)]
        _WORKER[key] = (main, fast, openers)
    return _WORKER[key]


def play_teacher_game(answer: str, game_idx: int, cfg: TraceConfig) -> list[dict]:
    wl = load()
    main, fast, openers = _teachers(cfg)
    vocab = guess_vocab(cfg.vocab)
    fmt = PromptFormat(spaced=cfg.spaced, state_line=cfg.state_line)
    rng = _rng(cfg.seed, answer, game_idx)
    game = Game(answer, hard_mode=cfg.hard_mode)
    samples: list[dict] = []
    k = rng.choices(range(len(cfg.prefix_probs)), weights=cfg.prefix_probs)[0]
    on_tree = True
    for _ in range(k):
        if game.done:
            break
        if cfg.hard_mode:  # a random word that is still legal in hard mode
            cands = main.candidates(game.turns)
            word = main.pool[int(cands[rng.randrange(len(cands))])]
        else:
            word = rng.choice(vocab)
        game.step(render_completion(game, word, fmt), word, wl.guess_set, by_model=False)
        on_tree = False
    if not game.turns and rng.random() < cfg.alt_opener_prob:
        word = rng.choice(openers)
        game.step(render_completion(game, word, fmt), word, wl.guess_set, by_model=False)
        on_tree = False
    invalid_at = rng.randrange(1, 5) if rng.random() < cfg.invalid_prob else -1
    while not game.done:
        if len(game.turns) == invalid_at and game.turns_left > 1:
            # one invalid turn: a non-word or a missing tag. The teacher's next move shows recovery.
            if rng.random() < 0.5:
                junk = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
                while junk in wl.guess_set:
                    junk = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
                game.step(f"<guess>{junk.upper()}</guess>", junk, wl.guess_set, by_model=False)
            else:
                game.step("I think the word is close.", None, wl.guess_set, by_model=False)
            on_tree = False
            continue
        if game.turns and rng.random() < cfg.noise and game.turns_left > 1:
            cands = main.candidates(game.turns)
            word = main.pool[int(cands[rng.randrange(len(cands))])]
            game.step(render_completion(game, word, fmt), word, wl.guess_set, by_model=False)
            on_tree = False
            continue
        teacher = main if on_tree else fast
        word = teacher.choose(game.turns)
        samples.append({
            "prompt": render_prompt(game, fmt),
            "completion": render_completion(game, word, fmt),
            "answer": answer,
            "game": game_idx,
            "turn": len(game.turns) + 1,
        })
        game.step(samples[-1]["completion"], word, wl.guess_set)
    for s in samples:
        s["game_won"] = game.won
    return samples


def warm_teacher(cfg: TraceConfig) -> None:
    """Play the teacher on every answer once so its search tree is cached on disk.
    About 7 minutes on a laptop CPU the first time, instant afterwards."""
    wl = load()
    main = _teachers(cfg)[0]
    hard_mode = cfg.hard_mode
    if not isinstance(main, LookaheadSolver) or main._memo_loaded:
        return
    print("[traces] building the teacher's search cache (first run only, a few minutes)")
    for a in wl.answers:
        g = Game(a, hard_mode=hard_mode)
        while not g.done:
            g.step("", main.choose(g.turns), wl.guess_set)
    main.save_memo()


def _write_shard(args) -> tuple[str, int]:
    path, words, cfg_dict = args
    cfg = TraceConfig(**{**cfg_dict, "prefix_probs": tuple(cfg_dict["prefix_probs"])})
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for w in words:
            for gi in range(cfg.games_per_word):
                for s in play_teacher_game(w, gi, cfg):
                    f.write(json.dumps(s) + "\n")
                    n += 1
    os.replace(tmp, path)
    getattr(_teachers(cfg)[0], "save_memo", lambda: None)()
    return path, n


def generate_traces(out_dir: str | Path, cfg: TraceConfig, workers: int = 1, split: str = "train") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if split in ("extra", "extra-broad"):
        if cfg.answer_pool != "all":
            raise ValueError("extra answers need answer_pool='all' (the answer must be a candidate)")
        words = extra_answer_words(broad=split == "extra-broad")
    else:
        words = get_split(split)
    if split not in ("train", "extra", "extra-broad"):
        print(f"[traces] warning: building traces from split {split!r}; never train on val or heldout answers")
    config = {"split": split, "split_sha256": sha256_words(words), **asdict(cfg)}
    cfg_path = out / "trace_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != json.loads(json.dumps(config)):
        raise RuntimeError(f"{out} was generated with a different config; use a new folder")
    cfg_path.write_text(json.dumps(config, indent=2))

    order = sorted(words)
    _rng(cfg.seed, "shard-order").shuffle(order)
    shards = [order[i : i + cfg.words_per_shard] for i in range(0, len(order), cfg.words_per_shard)]
    jobs = []
    for i, ws in enumerate(shards):
        path = out / f"shard-{i:05d}.jsonl"
        if path.exists():
            continue
        jobs.append((str(path), ws, asdict(cfg)))
    for stale in out.glob("*.tmp"):
        stale.unlink()  # leftovers from a killed run
    print(f"[traces] {len(shards) - len(jobs)}/{len(shards)} shards already done; writing {len(jobs)} with {workers} worker(s)")
    if jobs:
        warm_teacher(cfg)  # build the shared search cache once, before workers start
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for path, n in ex.map(_write_shard, jobs):
                    print(f"[traces] {Path(path).name}: {n} samples")
        else:
            for job in jobs:
                path, n = _write_shard(job)
                print(f"[traces] {Path(path).name}: {n} samples")
    total = 0
    for p in sorted(out.glob("shard-*.jsonl")):
        with open(p, encoding="utf-8") as f:
            total += sum(1 for _ in f)
    manifest = {"shards": len(shards), "samples": total, "config": config}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def iter_samples(trace_dir: str | Path):
    for p in sorted(Path(trace_dir).glob("shard-*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
