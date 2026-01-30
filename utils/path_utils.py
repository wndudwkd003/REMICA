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
