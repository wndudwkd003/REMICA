from dataclasses import dataclass, field
from enum import Enum


class DatasetEnum(Enum):
    DiaSafety = "DiaSafety"
    GabHate = "gab_hate"
    HSOL = "hate-speech-and-offensive-language-master"
    HateXplain = "hatexplain"
    RealToxicityPrompts = "real-toxicity-prompts"
    OffenseEval = "offenseval"
    HSD = "hate-speech-dataset-master"
    ToxiGen = "toxigen"
    ToxiSpanSE = "ToxiSpanSE"
    ToxiCR = "toxicr"
    HSDCD = "hsdcd"
    ISHate = "ISHate"


class ModelEnum(Enum):
    ModernBERT = "answerdotai/ModernBERT-base"
    RoBERTa = "FacebookAI/roberta-base"
    DeBERTaV3 = "microsoft/deberta-v3-base"
    Longformer = "allenai/longformer-base-4096"
    BigBird = "google/bigbird-roberta-base"


@dataclass
class Config:
    datasets_dir: str = "datasets_processed"

    dataset_order: list[tuple[DatasetEnum, int]] = field(
        default_factory=lambda: [
            (DatasetEnum.DiaSafety, 32),
            (DatasetEnum.GabHate, 32),
            (DatasetEnum.HateXplain, 32),
            (DatasetEnum.HSOL, 32),
            (DatasetEnum.RealToxicityPrompts, 32),
            (DatasetEnum.OffenseEval, 32),
            (DatasetEnum.HSD, 32),
            (DatasetEnum.ToxiGen, 32),
            (DatasetEnum.ToxiSpanSE, 4),
            (DatasetEnum.ToxiCR, 16),
            (DatasetEnum.HSDCD, 16),
            (DatasetEnum.ISHate, 16),
        ]
    )

    do_mode: str = "REM_Stage_1"  # REM_Stage_1 | REM_Stage_2 |

    model_name: str = ModelEnum.ModernBERT.name
    model_id: str = ModelEnum.ModernBERT.value

    run_dir: str = "runs"
    train_mode: str = "train_test"  # "train" or "test"
    load_run_dir: str | None = None  # test에서 특정 run 폴더를 지정하고 싶으면 사용

    seed: int = 42
    max_len: int = 2048
    num_epochs: int = 200
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_workers: int = 0

    meta_to_text: bool = False
    hidden_dim: int = 512

    early_stopping_patience: int = 30
    early_stopping_delta: float = 0.001
