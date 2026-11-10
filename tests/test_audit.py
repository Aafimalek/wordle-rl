"""The independent audit recomputes Wordle colours on its own and catches tampered results."""

import importlib.util
import json
from pathlib import Path

from fiveletters.feedback import feedback

_spec = importlib.util.spec_from_file_location("audit_eval", Path(__file__).resolve().parents[1] / "scripts" / "audit_eval.py")
audit_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_eval)


def test_independent_colours_agree_with_the_engine_on_repeated_letters():
    for g, a in [("speed", "abide"), ("eerie", "elder"), ("llama", "label"), ("crate", "crate"), ("aaaaa", "abbey")]:
        assert audit_eval.wordle_colours(g, a) == tuple(feedback(g, a)), (g, a)


def _write(folder, records, win_rate):
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"win_rate": win_rate, "word_set": "custom"}))
    (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))


def _turn(g, a, valid=True):
    return {"guess": g, "valid": valid, "feedback": list(audit_eval.wordle_colours(g, a)) if valid else None}


def test_audit_passes_honest_and_flags_tampered(tmp_path):
    words = {"crate", "ghost", "slate"}
    ok = [{"answer": "ghost", "won": True, "turns": [_turn("crate", "ghost"), _turn("ghost", "ghost")]}]
    _write(tmp_path / "ok", ok, 1.0)
    assert audit_eval.audit(tmp_path / "ok", words) == []
    lie = [{"answer": "ghost", "won": True, "turns": [_turn("crate", "ghost"), _turn("slate", "ghost")]}]
    _write(tmp_path / "lie", lie, 1.0)
    assert any("replay says False" in p for p in audit_eval.audit(tmp_path / "lie", words))
    nonword = [{"answer": "ghost", "won": True, "turns": [_turn("gxost", "ghost"), _turn("ghost", "ghost")]}]
    _write(tmp_path / "nonword", nonword, 1.0)
    assert any("not in the official guess list" in p for p in audit_eval.audit(tmp_path / "nonword", words))
