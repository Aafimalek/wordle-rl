"""Public-environment adapters. Skipped unless textarena is installed (pip install textarena nltk)."""

import pytest

pytest.importorskip("textarena")

from fiveletters.policies.base import Policy, SolverPolicy  # noqa: E402
from fiveletters.public_envs import _VERIFIERS_SYSTEM_PROMPT, crosscheck_feedback, make_game, run_public_eval  # noqa: E402
from fiveletters.solvers import EntropySolver  # noqa: E402


def test_feedback_matches_textarena():
    r = crosscheck_feedback(2000, seed=1)
    assert r["mismatches"] == 0, r["examples"]


def test_seed_gives_same_secret_as_verifiers():
    a = make_game(7).state.game_state["secret_word"]
    b = make_game(7).state.game_state["secret_word"]
    assert a == b


def test_verifiers_prompt_copy_is_exact():
    verifiers = pytest.importorskip("verifiers.v1.tasksets.textarena.taskset")
    assert verifiers.SYSTEM_PROMPT == _VERIFIERS_SYSTEM_PROMPT


class _Echo(Policy):
    name = "echo"

    def chat(self, convs):
        return ["[crane]" if len(c) < 3 else "[moist]" for c in convs]

    def act(self, games):
        raise NotImplementedError


@pytest.mark.parametrize("mode", ["native-verifiers", "native-textarena"])
def test_native_modes_run_and_resume(tmp_path, mode):
    first = run_public_eval(_Echo(), tmp_path, mode=mode, seeds=range(3), batch_size=2)
    again = run_public_eval(_Echo(), tmp_path, mode=mode, seeds=range(3), batch_size=2)  # resumes: nothing to do
    assert first["all"]["games"] == again["all"]["games"] == 3
    with pytest.raises(RuntimeError, match="different settings"):
        run_public_eval(_Echo(), tmp_path, mode=mode, seeds=range(4), batch_size=2)


def test_adapted_mode_uses_engine_feedback(tmp_path):
    s = run_public_eval(SolverPolicy(EntropySolver(vocab="shared")), tmp_path, mode="adapted", seeds=range(10), batch_size=5)
    assert s["all"]["games"] == 10 and s["all"]["games_lost_by_invalid_moves"] == 0 and s["all"]["win_rate"] >= 0.9
