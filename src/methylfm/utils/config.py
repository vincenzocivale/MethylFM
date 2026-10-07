from __future__ import annotations

import os
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config, resolving a top-level `includes: {key: relative_path}`
    map by recursively loading each referenced file (relative to `path`'s
    directory) under `key`, then deep-merging any directly-specified `key`
    section in `path` on top as an override."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    includes = data.pop("includes", None)
    if includes:
        base_dir = os.path.dirname(os.path.abspath(path))
        for key, rel_path in includes.items():
            sub_config = load_config(os.path.join(base_dir, rel_path))
            data[key] = _deep_merge(sub_config, data.get(key, {}))

    return data
