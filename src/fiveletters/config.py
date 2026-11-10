"""TOML configs with defaults and command-line overrides (``--set train.lr=1e-4``)."""

from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        where = f"{path}.{k}" if path else k
        if k not in out:
            raise KeyError(f"unknown config key {where!r}")
        if isinstance(out[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"{where} must be a table")
            out[k] = _merge(out[k], v, where)
        else:
            out[k] = v
    return out


def _parse_value(text: str):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(defaults: dict, path: str | None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(defaults)
    if path:
        with open(Path(path), "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"unknown config section {key!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = _parse_value(raw.strip())
    return cfg
