"""Metrics for a set of finished games, with 95% bootstrap intervals."""

from __future__ import annotations

import numpy as np

FAIL_COST = 7  # a lost game counts as 7 guesses, the usual convention


def summarize(records: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    n = len(records)
    if n == 0:
        return {"games": 0}
    won = np.array([r["won"] for r in records], dtype=float)
    turns = np.array([r["n_turns"] for r in records], dtype=float)
    cost = np.where(won > 0, turns, FAIL_COST)
    invalid = np.array([r["n_invalid"] for r in records], dtype=float)
    hist = {str(k): int(((turns == k) & (won > 0)).sum()) for k in range(1, 7)}
    bad = checked = 0
    for r in records:
        if not r.get("turns"):
            continue
        from ..game import Turn, count_clue_contradictions

        ts = [Turn(t["raw"], t["guess"], t["valid"], tuple(t["feedback"]) if t["feedback"] else None,
                   t.get("error"), t.get("by_model", True)) for t in r["turns"]]
        b, c = count_clue_contradictions(ts)
        bad, checked = bad + b, checked + c
    hist["fail"] = int((won == 0).sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_win = won[idx].mean(1)
    boot_cost = cost[idx].mean(1)

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    return {
        "games": n,
        "win_rate": round(float(won.mean()), 4),
        "win_rate_ci95": ci(boot_win),
        "mean_guesses_wins": round(float(turns[won > 0].mean()), 4) if won.any() else None,
        "mean_guesses_fail7": round(float(cost.mean()), 4),
        "mean_guesses_fail7_ci95": ci(boot_cost),
        "invalid_turn_rate": round(float(invalid.sum() / turns.sum()), 4),
        "clue_contradiction_rate": round(bad / checked, 4) if checked else None,
        "games_with_invalid": int((invalid > 0).sum()),
        **({"rejected_words_per_game": round(float(np.mean([r["n_rejected"] for r in records])), 4),
            "games_with_rejected": int(sum(r["n_rejected"] > 0 for r in records))}
           if records and "n_rejected" in records[0] else {}),
        **({"nonword_guess_rate": round(float(sum(r["n_nonword"] for r in records) / turns.sum()), 4),
            "games_with_nonword": int(sum(r["n_nonword"] > 0 for r in records))}
           if records and "n_nonword" in records[0] else {}),
        "histogram": hist,
    }
