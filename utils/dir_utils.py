# utils/dir_utils.py

from config.config import Config
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json

from pathlib import Path

from enum import Enum


def _json_default(o):
    if isinstance(o, Enum):
        return o.name
    return str(o)


def make_run_dir(config: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{config.model_name}"
    run_dir = Path(config.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "save").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    cfg = asdict(config) if is_dataclass(config) else dict(config.__dict__)
    cfg["dataset_order"] = [(ds.name, bs) for ds, bs in config.dataset_order]

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=_json_default)

    return run_dir
