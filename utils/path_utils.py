# utils/path_utils.py

from pathlib import Path
from config.config import Config


def get_memory_root(config: Config):
    ali = "ali_on" if bool(config.actual_label_intervention) else "ali_off"
    root = (
        Path(config.memory_dir)
        / str(config.api_model.name)
        / str(config.chain_mode.value)
        / ali
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_test_root(config: Config):
    ali = "ali_on" if bool(config.actual_label_intervention) else "ali_off"
    model_root = (
        Path(config.run_dir)
        / "test_memory"
        / str(config.api_model.name)
        / str(config.chain_mode.value)
        / ali
    )
    model_root.mkdir(parents=True, exist_ok=True)
    return model_root


def get_online_root(config: Config):
    model_root = (
        Path(config.run_dir)
        / "online_test"
        / str(config.api_model.name)
        / str(config.chain_mode.value)
    )
    model_root.mkdir(parents=True, exist_ok=True)
    return model_root


def make_unique_dir(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir

    i = 1
    while True:
        cand = Path(str(base_dir) + f"_{i}")
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        i += 1
