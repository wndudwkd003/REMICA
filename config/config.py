from dataclasses import dataclass, field


@dataclass
class Config:
    datasets_dir: str = "datasets/selects"

    do_mode: str = "REM_Stage_1"  # REM_Stage_1 | REM_Stage_2 |
