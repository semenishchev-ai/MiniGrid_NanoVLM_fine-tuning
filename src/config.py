import yaml
from pathlib import Path

def load_config(path="configs/base.yaml"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def merge_configs(*paths):
    merged = {}
    for p in paths:
        merged = _deep_merge(merged, load_config(p))
    return merged