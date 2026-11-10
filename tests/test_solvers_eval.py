import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fiveletters import words
from fiveletters.eval.harness import EvalIncomplete, run_eval
from fiveletters.eval.metrics import summarize
from fiveletters.game import Game
from fiveletters.policies.base import Policy, QuotaExhausted, SolverPolicy
from fiveletters.solvers import EntropySolver, LookaheadSolver, RandomConsistentSolver


def _play(solver, answers, hard=False):
    vw = words.load().guess_set
    games = []
    for a in answers:
        g = Game(a, hard_mode=hard)
        while not g.done:
            g.step("", solver.choose(g.turns), vw)
        games.append(g)
    return games


def test_entropy_solver_wins_sample():
    games = _play(EntropySolver(), words.get_split("val"))
    assert all(g.won for g in games)
    assert summarize([g.record() for g in games])["mean_guesses_fail7"] < 3.7


def test_lookahead_solver_on_sample_and_memo_persists():
    s = LookaheadSolver()
    games = _play(s, words.get_split("val")[:8])
    assert all(g.won for g in games)
    s.save_memo()
    again = LookaheadSolver()
    assert again._memo_loaded == len(s._memo) > 0


def test_hard_mode_solver_is_legal():
    games = _play(EntropySolver(hard_mode=True), words.get_split("val")[:30], hard=True)
    assert all(t.valid for g in games for t in g.turns)


def test_random_consistent_never_contradicts_clues():
    s = RandomConsistentSolver(seed=3)
    for g in _play(s, words.get_split("val")[:30]):
        assert all(t.valid for t in g.turns)


@pytest.mark.slow
def test_lookahead_full_answer_set_near_optimal():
    games = _play(LookaheadSolver(), words.get_split("full"))
    m = summarize([g.record() for g in games])
    assert m["win_rate"] == 1.0 and m["mean_guesses_fail7"] <= 3.43


class _Flaky(Policy):
    """Plays like the solver but runs out of quota after a number of calls."""
    name = "flaky"

    def __init__(self, fail_after):
        self.inner = SolverPolicy(EntropySolver())
        self.calls, self.fail_after = 0, fail_after

    def act(self, games):
        self.calls += 1
        if self.calls > self.fail_after:
            raise QuotaExhausted("daily cap")
        return self.inner.act(games)


def test_harness_resumes_after_quota_and_crash(tmp_path):
    ws = words.get_split("val")[:20]
    with pytest.raises(EvalIncomplete):
        run_eval(_Flaky(fail_after=8), ws, tmp_path, word_set="val", batch_size=5)
    done = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert 0 < len(done) < 20 and (tmp_path / "summary.partial.json").exists()
    with open(tmp_path / "results.jsonl", "a") as f:
        f.write('{"answer": "trunc')  # simulate a crash mid-write
    s = run_eval(_Flaky(fail_after=10**9), ws, tmp_path, word_set="val", batch_size=5)
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert s["games"] == 20 and len(lines) == 20 and len({json.loads(x)["answer"] for x in lines}) == 20


def test_harness_refuses_mixed_settings(tmp_path):
    ws = words.get_split("val")[:5]
    run_eval(SolverPolicy(EntropySolver()), ws, tmp_path, word_set="val")
    with pytest.raises(RuntimeError, match="different settings"):
        run_eval(SolverPolicy(EntropySolver()), ws, tmp_path, word_set="val", hard_mode=True)


class _Handler(BaseHTTPRequestHandler):
    plan: list = []

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        code = self.plan.pop(0) if self.plan else 200
        self.send_response(code)
        if code == 429:
            self.send_header("Retry-After", "0")
        self.end_headers()
        if code == 200:
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "<guess>CRANE</guess>"}}]}).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def api_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_api_policy_retries_then_gives_up(api_server, monkeypatch):
    from fiveletters.policies.api import ChatAPIPolicy

    monkeypatch.setenv("TEST_KEY", "k")
    _Handler.plan = [429, 503]
    p = ChatAPIPolicy("custom", "m", base_url=api_server, api_key_env="TEST_KEY", requests_per_minute=0)
    assert p.act([Game("ghost")]) == ["<guess>CRANE</guess>"]
    _Handler.plan = [429] * 50
    p = ChatAPIPolicy("custom", "m", base_url=api_server, api_key_env="TEST_KEY", requests_per_minute=0,
                      max_retries=3, give_up_after_s=30)
    with pytest.raises(QuotaExhausted):
        p.act([Game("ghost")])
