"""Tiny YAML config system.

* `base: other.yaml` inside a config inherits from another file (relative to the config).
* `--set a.b.c=value` on the command line overrides any key (value parsed as YAML).
* Configs are plain dicts with attribute access: cfg.train.lr
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class Config(dict):
    """dict with attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def to_dict(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: to_dict(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_dict(v) for v in x]
    return x


def _merge(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> Dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base:
        return _merge(_load_raw((path.parent / base).resolve()), raw)
    return raw


def apply_override(cfg: Dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"Override must look like key.sub=value, got: {assignment!r}")
    key, value = assignment.split("=", 1)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Config:
    raw = _load_raw(Path(path).resolve())
    for o in overrides or []:
        apply_override(raw, o)
    return _wrap(raw)


def save_config(cfg: Dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


def parse_args(description: str = "") -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set train.epochs=2 data.root=/content/x")
    return p.parse_args()
