# utils/dir_utils.py

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from config.config import Config


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


def make_run_dir_gpt(config: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpt_tag = str(config.gpt_model)
    gpt_tag = gpt_tag.replace("/", "_").replace(":", "_").replace(" ", "_")

    run_name = f"{ts}_GPT_{gpt_tag}"
    run_dir = Path(config.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "save").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    cfg = asdict(config) if is_dataclass(config) else dict(config.__dict__)
    cfg["dataset_order"] = [(ds.name, bs) for ds, bs in config.dataset_order]
    cfg["_run_mode"] = "GPT_INFER"
    cfg["_created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=_json_default)
    return run_dir
