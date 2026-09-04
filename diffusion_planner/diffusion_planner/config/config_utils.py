import json
from dataclasses import fields
from pathlib import Path
from typing import Any


def save_config(cfg: Any, out_root: str | Path, filename: str) -> None:
    if not filename.endswith(".json"):
        filename += ".json"
    out_root = Path(out_root)
    config_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.repr}
    with open(out_root / filename, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
