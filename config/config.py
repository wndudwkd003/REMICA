from dataclasses import dataclass, field


@dataclass
class Config:
    datasets_dir: str = "datasets_processed"

    dataset_order: list[str] = field(
        default_factory=lambda: [
            "DiaSafety",
            "GabHate",
            "HateXplain",
            "HSOL",
            "RealToxicityPrompts",
        ]
    )

    do_mode: str = "REM_Stage_1"  # REM_Stage_1 | REM_Stage_2 |

    model_name: str = "ModernBERT-base"
    model_id: str = "answerdotai/ModernBERT-base"

    run_dir: str = "runs"
    train_mode: str = "train_test"  # "train" or "test"
    load_run_dir: str | None = None  # test에서 특정 run 폴더를 지정하고 싶으면 사용

    seed: int = 42
    max_len: int = 2048
    batch_size: int = 32
    num_epochs: int = 100
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_workers: int = 0

    meta_to_text: bool = False
    hidden_dim: int = 512

    early_stopping_patience: int = 15
    early_stopping_delta: float = 0.001
