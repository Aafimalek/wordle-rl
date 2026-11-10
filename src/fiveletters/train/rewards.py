"""Game rewards and group-relative advantages for GRPO."""

from __future__ import annotations

from ..game import Game, count_clue_contradictions


def game_reward(game: Game, cfg: dict, state_hits: int = 0, state_total: int = 0) -> float:
    """Win reward, minus a penalty per invalid model turn, plus optional bonuses.

    binary=True gives 1 for a win and 0 otherwise (TRL reported cleaner learning with it).
    Otherwise a win earns `win` + `speed_bonus` scaled by the turns left unused.
    """
    if cfg.get("binary"):
        return 1.0 if game.won else 0.0
    r = 0.0
    if game.won:
        unused = game.max_turns - len(game.turns)
        r += cfg["win"] + cfg["speed_bonus"] * unused / (game.max_turns - 1)
    invalid_by_model = sum(1 for t in game.turns if t.by_model and not t.valid)
    r -= cfg["invalid_penalty"] * invalid_by_model
    if cfg.get("consistency_penalty"):
        bad, _ = count_clue_contradictions(game.turns)
        r -= cfg["consistency_penalty"] * bad
    if cfg.get("state_bonus") and state_total:
        r += cfg["state_bonus"] * state_hits / state_total
    return r


def group_advantages(rewards: list[float], group_size: int, eps: float = 1e-6) -> tuple[list[float], float]:
    """(reward - group mean) / group std. Groups with identical rewards get 0 (no signal).
    Returns the advantages and the fraction of groups that had no signal."""
    if len(rewards) % group_size:
        raise ValueError("rewards must be a whole number of groups")
    adv: list[float] = []
    flat = 0
    for s in range(0, len(rewards), group_size):
        g = rewards[s : s + group_size]
        mean = sum(g) / len(g)
        var = sum((x - mean) ** 2 for x in g) / len(g)
        std = var**0.5
        if std < eps:
            flat += 1
            adv.extend([0.0] * len(g))
        else:
            adv.extend([(x - mean) / (std + eps) for x in g])
    return adv, flat / (len(rewards) // group_size)
