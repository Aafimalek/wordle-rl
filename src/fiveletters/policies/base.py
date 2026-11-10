from __future__ import annotations

from abc import ABC, abstractmethod

from ..game import Game
from ..prompts import DEFAULT_FORMAT, PromptFormat, render_completion


class QuotaExhausted(RuntimeError):
    """A free API stopped answering (daily cap or sustained rate limiting). Progress is kept; rerun later."""


class Policy(ABC):
    name: str = "policy"

    @abstractmethod
    def act(self, games: list[Game]) -> list[str]:
        """Return one raw text output per game. The harness parses and validates it."""

    def describe(self) -> dict:
        return {"name": self.name}

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class SolverPolicy(Policy):
    """Wraps a solver so it emits text in the same format a model must, then goes through the same parser."""

    def __init__(self, solver, fmt: PromptFormat = DEFAULT_FORMAT):
        self.solver = solver
        self.fmt = fmt
        self.name = f"solver-{solver.describe()['solver']}"

    def act(self, games: list[Game]) -> list[str]:
        return [render_completion(g, self.solver.choose(g.turns), self.fmt) for g in games]

    def describe(self) -> dict:
        return {"name": self.name, **self.solver.describe(), "format": self.fmt.to_dict()}

    def close(self) -> None:
        save = getattr(self.solver, "save_memo", None)
        if save:
            save()
